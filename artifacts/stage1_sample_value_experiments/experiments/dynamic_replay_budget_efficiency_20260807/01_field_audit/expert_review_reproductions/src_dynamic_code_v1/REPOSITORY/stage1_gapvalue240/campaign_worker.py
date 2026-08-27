"""Execution primitives for one preregistered dynamic replay physical job."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
import uuid

import pandas as pd

from .errors import ValidationError
from .aiops import exit_code_for_exception
from .campaign_dynamic_training import (
    DynamicTrainingSpec,
    clone_branch_workspace,
    run_dynamic_training_segment,
)
from .campaign_checkpoint_predictions import (
    CampaignCheckpointPredictionSpec,
    run_key_checkpoint_predictions,
)
from .campaign_controller import load_campaign_release
from .campaign_assignment import CampaignAssignment, load_campaign_assignment
from .campaign_lease import claim_job_lease
from .campaign_layout import CAMPAIGN_ID
from .campaign_process_telemetry import ProcessTelemetrySpec
from .campaign_run_queue import build_replay_identity_manifest
from .hardlink_staging import (
    prepare_base_cache,
    staged_identity_replay_session,
    storage_preflight,
)
from .locks import RunLock
from .machine import MachineConfig, load_machine_config
from .machine_assets import validate_machine_asset_report
from .monitor import ResourceMonitor
from .runtime_contract import load_runtime_contract
from .util import atomic_write_json, sha256_file


class CampaignWorkerError(ValidationError):
    """Raised when a physical campaign job cannot be executed as registered."""


@dataclass(frozen=True)
class CampaignJob:
    queue_dir: Path
    queue_order: int
    job_id: str
    cycle_id: str
    release_state: str
    machine_id: str
    seed_id: str
    training_seed: int
    logical_run_id: str
    logical_arm_id: str
    schedule_id: str
    job_kind: str
    segment_index: int
    segment_start_epoch: int
    segment_end_epoch: int
    normal_replay_slots: int
    defect_guard_slots: int
    total_replay_slots: int
    expected_steps_batch128: int
    selection_pool_id: str
    selection_pool_digest: str
    active_selection_digest: str
    active_selection_rows: int
    selection_template: Path
    monitor_manifest: Path
    canonical_lock: Path
    canonical_lock_file_sha256: str
    dependency_job_id: str
    dependency_output_relpath: str
    resume_from_epoch: int
    branch_checkpoint_required: bool
    machine_output_relpath: str
    retained_checkpoint_epochs: tuple[int, ...]


@dataclass(frozen=True)
class SegmentExecutionDecision:
    action: str
    actual_start_epoch: int
    resume_checkpoint: Path | None
    branch_parent_output: Path | None = None


@dataclass(frozen=True)
class CampaignJobResult:
    job_id: str
    action: str
    output_dir: Path
    completed_epoch: int
    state_path: Path
    result_path: Path


_ZERO_EPOCH_TRANSACTION = ".zero_epoch_restart_transaction.json"
_ZERO_EPOCH_READY = "zero_epoch_restart_ready.json"
_ZERO_EPOCH_RUNTIME_PATHS = (
    "trainer",
    "training_state",
    "process_telemetry",
    "key_checkpoint_predictions",
    "dynamic_training_audit.json",
    "resolved_training_args.json",
)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0", ""}:
        return False
    raise CampaignWorkerError(f"invalid boolean in campaign queue: {value!r}")


def _safe_queue_path(queue: Path, relative_text: str, name: str) -> Path:
    relative = Path(str(relative_text).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise CampaignWorkerError(f"unsafe {name} path in queue: {relative}")
    path = (queue / relative).resolve()
    try:
        path.relative_to(queue)
    except ValueError as exc:
        raise CampaignWorkerError(f"{name} path escapes queue: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _retained_epochs(value: Any) -> tuple[int, ...]:
    text = str(value).strip()
    if not text:
        return ()
    try:
        epochs = tuple(int(part) for part in text.split(";") if part)
    except ValueError as exc:
        raise CampaignWorkerError(f"invalid checkpoint epoch list: {value!r}") from exc
    if any(epoch <= 0 for epoch in epochs) or len(set(epochs)) != len(epochs):
        raise CampaignWorkerError(f"invalid checkpoint epoch list: {value!r}")
    return epochs


def load_campaign_job(
    queue_dir: str | Path,
    job_id: str,
    *,
    expected_machine_id: str | None = None,
) -> CampaignJob:
    """Load one job only after verifying the immutable queue and input hashes."""

    queue = Path(queue_dir).resolve()
    registry = queue / "JOB_EXECUTION_REGISTRY.csv"
    validation_path = queue / "RUN_QUEUE_VALIDATION.json"
    if not registry.is_file() or not validation_path.is_file():
        raise FileNotFoundError(f"campaign queue is incomplete: {queue}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise CampaignWorkerError("campaign run queue validation status is not PASS")
    if validation.get("job_registry_sha256") != sha256_file(registry):
        raise CampaignWorkerError("campaign job registry checksum mismatch")
    frame = pd.read_csv(registry, keep_default_na=False)
    if len(frame) != int(validation.get("job_count", -1)):
        raise CampaignWorkerError("campaign job registry row count mismatch")
    match = frame.loc[frame.job_id.astype(str) == str(job_id)]
    if len(match) != 1:
        raise CampaignWorkerError(f"campaign job is not unique: {job_id}")
    row = match.iloc[0]
    machine_id = str(row.machine_id)
    if expected_machine_id is not None and machine_id != expected_machine_id:
        raise CampaignWorkerError(f"job {job_id} is assigned to {machine_id}, not {expected_machine_id}")
    template = _safe_queue_path(queue, str(row.active_selection_template_relpath), "selection template")
    if sha256_file(template) != str(row.active_selection_template_sha256).upper():
        raise CampaignWorkerError(f"selection template checksum mismatch for {job_id}")
    monitor = _safe_queue_path(queue, str(row.monitor_manifest_relpath), "monitor manifest")
    if sha256_file(monitor) != str(row.monitor_manifest_sha256).upper():
        raise CampaignWorkerError(f"monitor manifest checksum mismatch for {job_id}")
    if str(validation.get("monitor_manifest_sha256", "")).upper() != sha256_file(monitor):
        raise CampaignWorkerError("queue-level monitor manifest checksum mismatch")
    canonical_lock = _safe_queue_path(
        queue,
        str(row.canonical_lock_relpath),
        "canonical training lock",
    )
    canonical_lock_sha = str(row.canonical_lock_file_sha256).upper()
    if sha256_file(canonical_lock) != canonical_lock_sha:
        raise CampaignWorkerError(f"canonical lock checksum mismatch for {job_id}")
    if str(validation.get("canonical_lock_file_sha256", "")).upper() != canonical_lock_sha:
        raise CampaignWorkerError("queue-level canonical lock checksum mismatch")
    normal = int(row.normal_replay_slots)
    defect = int(row.defect_guard_slots)
    total = int(row.total_replay_slots)
    if normal + defect != total or total != int(row.active_selection_rows):
        raise CampaignWorkerError(f"active replay count contract mismatch for {job_id}")
    start = int(row.segment_start_epoch)
    end = int(row.segment_end_epoch)
    resume_from = int(row.resume_from_epoch)
    if not (1 <= start <= end <= 200) or resume_from != start - 1:
        raise CampaignWorkerError(f"segment boundary contract mismatch for {job_id}")
    return CampaignJob(
        queue_dir=queue,
        queue_order=int(row.queue_order),
        job_id=str(row.job_id),
        cycle_id=str(row.cycle_id),
        release_state=str(row.release_state),
        machine_id=machine_id,
        seed_id=str(row.seed_id),
        training_seed=int(row.training_seed),
        logical_run_id=str(row.logical_run_id),
        logical_arm_id=str(row.logical_arm_id),
        schedule_id=str(row.schedule_id),
        job_kind=str(row.job_kind),
        segment_index=int(row.segment_index),
        segment_start_epoch=start,
        segment_end_epoch=end,
        normal_replay_slots=normal,
        defect_guard_slots=defect,
        total_replay_slots=total,
        expected_steps_batch128=int(row.expected_steps_batch128),
        selection_pool_id=str(row.selection_pool_id),
        selection_pool_digest=str(row.selection_pool_digest).upper(),
        active_selection_digest=str(row.active_selection_digest).upper(),
        active_selection_rows=int(row.active_selection_rows),
        selection_template=template,
        monitor_manifest=monitor,
        canonical_lock=canonical_lock,
        canonical_lock_file_sha256=canonical_lock_sha,
        dependency_job_id=str(row.dependency_job_id),
        dependency_output_relpath=str(row.dependency_output_relpath),
        resume_from_epoch=resume_from,
        branch_checkpoint_required=_bool(row.branch_checkpoint_required),
        machine_output_relpath=str(row.machine_output_relpath),
        retained_checkpoint_epochs=_retained_epochs(row.checkpoint_epochs_to_retain),
    )


def _completed_epoch(output: Path) -> int:
    audit_path = output / "dynamic_training_audit.json"
    if not audit_path.is_file():
        raise CampaignWorkerError(f"run output exists without dynamic audit: {output}")
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        return int(audit["completed_epochs"])
    except Exception as exc:
        raise CampaignWorkerError(f"dynamic audit is unreadable: {audit_path}") from exc


def _has_completed_checkpoint(output: Path) -> bool:
    state = output / "training_state"
    if not state.is_dir():
        return False
    return (state / "last.pt").is_file() or any(state.glob("checkpoint_epoch_*.pt"))


def _safe_attempt_id(value: str) -> str:
    attempt_id = str(value).strip()
    if (
        not attempt_id
        or Path(attempt_id).name != attempt_id
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in attempt_id)
    ):
        raise CampaignWorkerError(f"unsafe failed-attempt identifier: {value!r}")
    return attempt_id


def _archive_manifest_entries(archive: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in archive.rglob("*") if candidate.is_file()):
        if path.name == "ATTEMPT_ARCHIVE_MANIFEST.json":
            continue
        entries.append(
            {
                "relative_path": path.relative_to(archive).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries


def archive_zero_epoch_attempt(
    output_dir: str | Path,
    *,
    attempt_id: str | None = None,
) -> Path:
    """Preserve a failed pre-epoch workspace before restarting from the base checkpoint."""

    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise CampaignWorkerError(f"zero-epoch recovery output is unavailable: {output}")
    transaction_path = output / _ZERO_EPOCH_TRANSACTION
    ready_path = output / _ZERO_EPOCH_READY
    audit_path = output / "dynamic_training_audit.json"
    transaction: dict[str, Any] | None = None
    if transaction_path.is_file():
        try:
            transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CampaignWorkerError(
                f"zero-epoch recovery transaction is unreadable: {transaction_path}"
            ) from exc
        transaction_id = _safe_attempt_id(str(transaction.get("attempt_id", "")))
        if attempt_id is not None and _safe_attempt_id(attempt_id) != transaction_id:
            raise CampaignWorkerError("zero-epoch recovery attempt conflicts with active transaction")
        attempt_id = transaction_id
    elif not audit_path.is_file() and ready_path.is_file():
        try:
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
            archive = (output / str(ready["archive_relpath"])).resolve()
            archive.relative_to(output)
        except Exception as exc:
            raise CampaignWorkerError(f"zero-epoch recovery marker is unreadable: {ready_path}") from exc
        manifest = archive / "ATTEMPT_ARCHIVE_MANIFEST.json"
        if not manifest.is_file() or sha256_file(manifest) != str(ready.get("manifest_sha256", "")).upper():
            raise CampaignWorkerError("zero-epoch recovery archive no longer matches its ready marker")
        return archive
    else:
        if audit_path.is_file() and _completed_epoch(output) != 0:
            raise CampaignWorkerError("zero-epoch recovery refused after a completed epoch")
        if _has_completed_checkpoint(output):
            raise CampaignWorkerError("zero-epoch recovery refused because a completed checkpoint exists")
        attempt_id = _safe_attempt_id(
            attempt_id or f"attempt_{int(time.time())}_{uuid.uuid4().hex[:12]}"
        )
        transaction = {
            "schema_version": "stage1.zero_epoch_restart_transaction.v1",
            "attempt_id": attempt_id,
            "started_at_unix": time.time(),
        }
        atomic_write_json(transaction_path, transaction, overwrite=False)

    assert attempt_id is not None
    archive = output / "failed_attempts" / attempt_id
    archive.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for relative_text in _ZERO_EPOCH_RUNTIME_PATHS:
        source = output / relative_text
        destination = archive / relative_text
        if source.exists():
            if destination.exists():
                raise CampaignWorkerError(
                    f"zero-epoch recovery has both live and archived copies: {relative_text}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            moved.append(relative_text)
        elif destination.exists():
            moved.append(relative_text)

    manifest_path = archive / "ATTEMPT_ARCHIVE_MANIFEST.json"
    if not manifest_path.is_file():
        entries = _archive_manifest_entries(archive)
        atomic_write_json(
            manifest_path,
            {
                "schema_version": "stage1.zero_epoch_failed_attempt_archive.v1",
                "status": "COMPLETE",
                "attempt_id": attempt_id,
                "reason": "NO_COMPLETED_EPOCH_AND_NO_RESUMABLE_CHECKPOINT",
                "archived_at_unix": time.time(),
                "moved_top_level_paths": sorted(set(moved)),
                "file_count": len(entries),
                "total_size_bytes": sum(int(row["size_bytes"]) for row in entries),
                "files": entries,
            },
            overwrite=False,
        )
    atomic_write_json(
        ready_path,
        {
            "schema_version": "stage1.zero_epoch_restart_ready.v1",
            "status": "READY_FROM_BASE_CHECKPOINT",
            "attempt_id": attempt_id,
            "archive_relpath": archive.relative_to(output).as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
        },
        overwrite=True,
    )
    transaction_path.unlink(missing_ok=True)
    return archive


def resolve_segment_execution(
    job: CampaignJob,
    output_dir: str | Path,
    *,
    dependency_output: str | Path | None = None,
) -> SegmentExecutionDecision:
    """Resolve new, branch, partial-resume, or already-complete execution."""

    output = Path(output_dir).resolve()
    dependency = Path(dependency_output).resolve() if dependency_output is not None else None
    if not output.exists():
        if job.segment_start_epoch == 1:
            return SegmentExecutionDecision("NEW", 1, None)
        if not job.branch_checkpoint_required:
            raise CampaignWorkerError(
                f"segment output is missing at expected boundary {job.resume_from_epoch}: {output}"
            )
        if dependency is None or not dependency.is_dir():
            raise CampaignWorkerError(f"branch dependency output is unavailable: {dependency}")
        completed = _completed_epoch(dependency)
        if completed != job.resume_from_epoch:
            raise CampaignWorkerError(
                f"branch dependency completed epoch {completed} != {job.resume_from_epoch}"
            )
        checkpoint = dependency / f"training_state/checkpoint_epoch_{job.resume_from_epoch:04d}.pt"
        if not checkpoint.is_file():
            raise CampaignWorkerError(f"branch dependency checkpoint is missing: {checkpoint}")
        return SegmentExecutionDecision(
            "BRANCH",
            job.segment_start_epoch,
            None,
            branch_parent_output=dependency,
        )

    if not output.is_dir():
        raise CampaignWorkerError(f"run output path is not a directory: {output}")
    audit_path = output / "dynamic_training_audit.json"
    if job.segment_start_epoch == 1 and (
        (output / _ZERO_EPOCH_TRANSACTION).is_file()
        or (output / _ZERO_EPOCH_READY).is_file()
        or not audit_path.is_file()
    ):
        if _has_completed_checkpoint(output):
            raise CampaignWorkerError(
                "run output has a completed checkpoint but no readable dynamic audit"
            )
        return SegmentExecutionDecision("RESTART_ZERO_EPOCH", 1, None)
    completed = _completed_epoch(output)
    if job.segment_start_epoch == 1 and completed == 0:
        if _has_completed_checkpoint(output):
            raise CampaignWorkerError(
                "zero-epoch audit conflicts with an unexpected completed checkpoint"
            )
        return SegmentExecutionDecision("RESTART_ZERO_EPOCH", 1, None)
    if completed >= job.segment_end_epoch:
        return SegmentExecutionDecision("SKIP_COMPLETE", job.segment_end_epoch + 1, None)
    if completed < job.resume_from_epoch:
        raise CampaignWorkerError(
            f"run completed epoch {completed}; expected boundary {job.resume_from_epoch} before {job.job_id}"
        )
    if completed >= job.segment_start_epoch:
        actual_start = completed + 1
    elif completed == job.resume_from_epoch:
        actual_start = job.segment_start_epoch
    else:  # pragma: no cover - guarded by completed < resume_from above
        raise AssertionError("unreachable segment state")
    checkpoint = output / "training_state/last.pt"
    if not checkpoint.is_file():
        raise CampaignWorkerError(f"resumable last checkpoint is missing: {checkpoint}")
    return SegmentExecutionDecision("RESUME", actual_start, checkpoint)


def _machine_output(machine: MachineConfig, relative_text: str) -> Path:
    relative = Path(str(relative_text).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise CampaignWorkerError(f"unsafe machine output path: {relative}")
    root = machine.path_value("output_root")
    output = (root / relative).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise CampaignWorkerError(f"machine output escapes configured root: {output}") from exc
    return output


def _machine_campaign_root(machine: MachineConfig) -> Path:
    return (machine.path_value("output_root") / CAMPAIGN_ID).resolve()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _code_provenance(repo: Path) -> dict[str, Any]:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=repo, text=True
        ).strip()
        tracked_status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CampaignWorkerError(f"unable to capture git code provenance: {exc}") from exc
    return {
        "git_head": head,
        "git_branch": branch,
        "tracked_worktree_clean": not bool(tracked_status),
        "tracked_status": tracked_status,
    }


def _validate_live_assets(machine: MachineConfig) -> dict[str, Any]:
    repo = machine.path_value("repo_root")
    runtime = load_runtime_contract(repo / "configs/stage1_gapvalue240/RUNTIME_CONTRACT_v1_2.yaml")
    report_path = machine.path_value("machine_asset_report")
    validated = validate_machine_asset_report(
        runtime,
        report_path,
        expected_machine_id=str(machine.data["machine_id"]),
        minimum_image_verification="existence",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks: dict[str, Any] = {}
    for role in ("train_defect", "train_normal", "val_model_defect", "val_model_normal"):
        record = report["manifests"][role]
        key = str(record["machine_config_key"])
        path = machine.path_value(key)
        actual = sha256_file(path)
        if Path(str(record["path"])).resolve() != path or actual != str(record["sha256"]).upper():
            raise CampaignWorkerError(f"live machine manifest differs from asset snapshot: {role}")
        checks[role] = {"path": str(path), "sha256": actual, "rows": int(record["rows"])}
    checkpoint = machine.path_value("base_checkpoint")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != str(runtime.data["checkpoint"]["sha256"]).upper():
        raise CampaignWorkerError("base checkpoint differs from frozen runtime contract")
    return {
        **validated,
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "live_manifests": checks,
    }


def _job_state_path(machine: MachineConfig, job_id: str) -> Path:
    return _machine_campaign_root(machine) / "09_aiops/job_states" / f"{job_id}.json"


def _write_job_state(
    machine: MachineConfig,
    job: CampaignJob,
    state: str,
    *,
    assignment: CampaignAssignment,
    action: str | None = None,
    error: BaseException | None = None,
    completed_epoch: int | None = None,
    lease_token: str | None = None,
) -> Path:
    path = _job_state_path(machine, job.job_id)
    previous: dict[str, Any] = {}
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    attempts = int(previous.get("attempt_count", 0)) + (1 if state == "RUNNING" else 0)
    payload = {
        "schema_version": "stage1.dynamic_campaign_job_state.v1",
        "campaign_id": CAMPAIGN_ID,
        "job_id": job.job_id,
        "machine_id": str(machine.data["machine_id"]),
        "planned_machine_slot": job.machine_id,
        "assignment_id": assignment.assignment_id,
        "assignment_sha256": assignment.sha256,
        "lease_token": lease_token,
        "logical_run_id": job.logical_run_id,
        "logical_arm_id": job.logical_arm_id,
        "state": state,
        "action": action,
        "completed_epoch": completed_epoch,
        "attempt_count": attempts,
        "pid": os.getpid(),
        "updated_at_unix": time.time(),
        "retryable": bool(error is not None and exit_code_for_exception(error) == 20),
        "error": None if error is None else f"{type(error).__name__}: {error}",
    }
    atomic_write_json(path, payload, overwrite=True)
    return path


def _dependency_output(job: CampaignJob) -> Path | None:
    if not job.dependency_job_id:
        return None
    registry = pd.read_csv(job.queue_dir / "JOB_EXECUTION_REGISTRY.csv", keep_default_na=False)
    match = registry.loc[registry.job_id.astype(str) == job.dependency_job_id]
    if len(match) != 1:
        raise CampaignWorkerError(f"dependency job is not unique: {job.dependency_job_id}")
    row = match.iloc[0]
    return Path(str(row.machine_output_relpath).replace("\\", "/"))


def _ensure_job_inputs(
    job: CampaignJob,
    output: Path,
    machine: MachineConfig,
    assignment: CampaignAssignment,
) -> tuple[Path, Path]:
    inputs = output / "job_inputs" / job.job_id
    inputs.mkdir(parents=True, exist_ok=True)
    template_copy = inputs / "selection_template.csv"
    monitor_copy = inputs / "causal_monitor_samples.csv"
    lock_copy = inputs / "canonical_training_lock.json"
    for source, destination in (
        (job.selection_template, template_copy),
        (job.monitor_manifest, monitor_copy),
        (job.canonical_lock, lock_copy),
    ):
        if destination.is_file():
            if sha256_file(destination) != sha256_file(source):
                raise CampaignWorkerError(f"frozen job input changed: {destination}")
        else:
            _atomic_copy(source, destination)
    replay_identity = inputs / "replay_identity_manifest.csv"
    if replay_identity.is_file():
        candidate = inputs / f".replay_identity_candidate_{uuid.uuid4().hex}.csv"
        try:
            built = build_replay_identity_manifest(
                template_copy,
                machine.path_value("normal_train_manifest"),
                machine.path_value("train_manifest"),
                run_slot=job.job_id,
                output_path=candidate,
            )
            if built.sha256 != sha256_file(replay_identity):
                raise CampaignWorkerError(f"existing replay identity differs from frozen template: {replay_identity}")
        finally:
            candidate.unlink(missing_ok=True)
    else:
        build_replay_identity_manifest(
            template_copy,
            machine.path_value("normal_train_manifest"),
            machine.path_value("train_manifest"),
            run_slot=job.job_id,
            output_path=replay_identity,
        )
    atomic_write_json(
        inputs / "assignment_authorization.json",
        {
            "schema_version": "stage1.dynamic_campaign_assignment_authorization.v1",
            "assignment_id": assignment.assignment_id,
            "assignment_sha256": assignment.sha256,
            "assignment_manifest": str(assignment.manifest_path),
            "release_id": assignment.release_id,
            "release_sha256": assignment.release_sha256,
            "job_id": job.job_id,
            "planned_machine_slot": job.machine_id,
            "assigned_machine_id": str(machine.data["machine_id"]),
        },
        overwrite=True,
    )
    atomic_write_json(
        inputs / "job_input_identity.json",
        {
            "schema_version": "stage1.dynamic_campaign_job_input.v1",
            "job": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in job.__dict__.items()
            },
            "selection_template_sha256": sha256_file(template_copy),
            "monitor_manifest_sha256": sha256_file(monitor_copy),
            "canonical_lock_file_sha256": sha256_file(lock_copy),
            "replay_identity_sha256": sha256_file(replay_identity),
            "base_train_normal_manifest_sha256": sha256_file(machine.path_value("normal_train_manifest")),
            "base_train_defect_manifest_sha256": sha256_file(machine.path_value("train_manifest")),
            "assignment_id": assignment.assignment_id,
            "assignment_sha256": assignment.sha256,
            "assigned_machine_id": str(machine.data["machine_id"]),
        },
        overwrite=True,
    )
    return replay_identity, monitor_copy


def _validate_job_segment(output: Path, job: CampaignJob) -> int:
    completed = _completed_epoch(output)
    if completed < job.segment_end_epoch:
        raise CampaignWorkerError(
            f"job returned at epoch {completed}, before {job.segment_end_epoch}"
        )
    results = output / "trainer/results.csv"
    if not results.is_file() or len(pd.read_csv(results)) < job.segment_end_epoch:
        raise CampaignWorkerError(f"training results do not cover segment end: {job.job_id}")
    checkpoint = output / f"training_state/checkpoint_epoch_{job.segment_end_epoch:04d}.pt"
    if not checkpoint.is_file():
        raise CampaignWorkerError(f"segment boundary checkpoint is missing: {checkpoint}")
    for epoch in range(job.segment_start_epoch, job.segment_end_epoch + 1):
        parquet = output / f"process_telemetry/epoch_{epoch:04d}_process_telemetry.parquet"
        sidecar = parquet.with_suffix(".json")
        if not parquet.is_file() or not sidecar.is_file():
            raise CampaignWorkerError(f"process telemetry is missing for epoch {epoch}: {job.job_id}")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        expected = {
            "status": "COMPLETE",
            "run_id": job.logical_run_id,
            "arm_id": job.logical_arm_id,
            "epoch": epoch,
            "observed_epoch_samples": 120_000 + job.total_replay_slots,
            "observed_replay_samples": job.total_replay_slots,
        }
        mismatch = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
        if mismatch or metadata.get("parquet_sha256") != sha256_file(parquet):
            raise CampaignWorkerError(f"process telemetry validation failed at epoch {epoch}: {mismatch}")
    return completed


def run_campaign_job(
    machine_config: str | Path,
    campaign_root: str | Path,
    job_id: str,
    *,
    release_path: str | Path,
    assignment_path: str | Path,
    expected_release_id: str,
    expected_canonical_lock_sha256: str,
    allow_dirty_code: bool = False,
) -> CampaignJobResult:
    """Execute exactly one dependency-resolved physical segment with durable state."""

    machine = load_machine_config(machine_config)
    if bool(machine.data.get("dry_run", False)):
        raise CampaignWorkerError("formal dynamic campaign worker refuses dry_run machine configs")
    if int(machine.data.get("num_workers", -1)) != 4:
        raise CampaignWorkerError(
            "formal dynamic campaign requires the canonical DataLoader workers=4"
        )
    campaign = Path(campaign_root).resolve()
    release = load_campaign_release(
        campaign / "04_run_queue",
        release_path,
        expected_campaign_id=CAMPAIGN_ID,
    )
    if release.release_id != str(expected_release_id):
        raise CampaignWorkerError(
            f"release identity {release.release_id} != command {expected_release_id}"
        )
    if str(job_id) not in set(release.job_ids):
        raise CampaignWorkerError(
            f"job {job_id} is not authorized by release {release.release_id}"
        )
    assignment = load_campaign_assignment(
        campaign / "04_run_queue",
        release_path,
        assignment_path,
        expected_campaign_id=CAMPAIGN_ID,
        expected_machine_id=str(machine.data["machine_id"]),
        expected_job_id=str(job_id),
        repo_root=machine.path_value("repo_root"),
    )
    job = load_campaign_job(
        campaign / "04_run_queue",
        job_id,
    )
    if job.canonical_lock_file_sha256 != str(expected_canonical_lock_sha256).upper():
        raise CampaignWorkerError(
            "canonical lock identity differs from the standalone command"
        )
    output = _machine_output(machine, job.machine_output_relpath)
    dependency_relative = _dependency_output(job)
    dependency_output = _machine_output(machine, dependency_relative) if dependency_relative else None
    state_path = _job_state_path(machine, job.job_id)
    locks = _machine_campaign_root(machine) / "09_aiops/locks"
    coordination_root = machine.path_value("coordination_root")
    lease = claim_job_lease(
        coordination_root,
        campaign_id=CAMPAIGN_ID,
        release_id=release.release_id,
        assignment_id=assignment.assignment_id,
        assignment_sha256=assignment.sha256,
        job_id=job.job_id,
        machine_id=str(machine.data["machine_id"]),
        ttl_seconds=float(machine.data.get("job_lease_ttl_seconds", 180.0)),
        heartbeat_seconds=float(machine.data.get("job_lease_heartbeat_seconds", 30.0)),
    )
    with lease, RunLock(
        locks / f"job_{job.job_id}.lock",
        {
            "job_id": job.job_id,
            "machine_id": str(machine.data["machine_id"]),
            "assignment_id": assignment.assignment_id,
            "lease_token": lease.token,
        },
        reclaim_dead_local=True,
    ), RunLock(
        locks / f"gpu_{machine.data['gpu_id']}.lock",
        {
            "job_id": job.job_id,
            "machine_id": str(machine.data["machine_id"]),
            "assignment_id": assignment.assignment_id,
            "lease_token": lease.token,
        },
        reclaim_dead_local=True,
    ):
        decision: SegmentExecutionDecision | None = None
        recovery_archive: Path | None = None
        monitor: ResourceMonitor | None = None
        try:
            code = _code_provenance(machine.path_value("repo_root"))
            if not allow_dirty_code and not code["tracked_worktree_clean"]:
                raise CampaignWorkerError("formal campaign requires a clean tracked worktree")
            assets = _validate_live_assets(machine)
            decision = resolve_segment_execution(
                job,
                output,
                dependency_output=dependency_output,
            )
            _write_job_state(
                machine,
                job,
                "RUNNING",
                assignment=assignment,
                action=decision.action,
                lease_token=lease.token,
            )
            if decision.action == "RESTART_ZERO_EPOCH":
                recovery_archive = archive_zero_epoch_attempt(output)
            if decision.action == "BRANCH":
                assert decision.branch_parent_output is not None
                branch = clone_branch_workspace(
                    decision.branch_parent_output,
                    output,
                    branch_run_id=job.logical_run_id,
                    branch_arm_id=job.logical_arm_id,
                    schedule_id=job.schedule_id,
                    selection_digest=job.selection_pool_digest,
                    branch_epoch=job.resume_from_epoch,
                )
                decision = SegmentExecutionDecision(
                    "BRANCH",
                    job.segment_start_epoch,
                    branch.resume_checkpoint,
                    decision.branch_parent_output,
                )
            replay_identity, monitor_manifest = _ensure_job_inputs(
                job,
                output,
                machine,
                assignment,
            )
            input_dir = replay_identity.parent
            atomic_write_json(input_dir / "machine_assets_validation.json", assets, overwrite=True)
            atomic_write_json(input_dir / "code_provenance.json", code, overwrite=True)
            log_path = output / "resource_logs" / f"{job.job_id}_{int(time.time())}.csv"
            monitor = ResourceMonitor(
                log_path,
                machine.data["gpu_id"],
                str(machine.data.get("nvidia_smi_path") or "nvidia-smi"),
                process_pid=os.getpid(),
                disk_path=output,
            )
            monitor.set_phase("STORAGE_PREFLIGHT")
            monitor.start()

            if decision.action != "SKIP_COMPLETE":
                manifests = {
                    "train_defect": machine.path_value("train_manifest"),
                    "train_normal": machine.path_value("normal_train_manifest"),
                    "val_defect": machine.path_value("val_model_defect_manifest"),
                    "val_normal": machine.path_value("val_model_normal_manifest"),
                }
                gib = 1024**3
                preflight = storage_preflight(
                    dataset_root=machine.path_value("dataset_root"),
                    staging_root=machine.path_value("staging_root"),
                    output_root=output,
                    hardlink_probe_manifest=manifests["train_defect"],
                    expected_staging_files=144_000 + job.total_replay_slots + 10,
                    maximum_staging_files=int(machine.data.get("maximum_staging_files", 151_000)),
                    minimum_staging_free_bytes=int(float(machine.data.get("minimum_staging_free_gib", 2)) * gib),
                    minimum_output_free_bytes=int(float(machine.data.get("minimum_output_free_gib", 20)) * gib),
                )
                atomic_write_json(input_dir / "storage_preflight.json", preflight, overwrite=True)
                monitor.set_phase("BASE_CACHE_PREPARE")
                cache = prepare_base_cache(
                    machine.path_value("dataset_root"),
                    machine.path_value("staging_root"),
                    manifests,
                )
                segment_id = (
                    f"{job.job_id}_E{decision.actual_start_epoch:03d}_{job.segment_end_epoch:03d}_"
                    f"A{int(time.time())}"
                )
                telemetry = ProcessTelemetrySpec(
                    run_id=job.logical_run_id,
                    arm_id=job.logical_arm_id,
                    segment_id=segment_id,
                    output_dir=output / "process_telemetry",
                    base_normal_manifest=manifests["train_normal"],
                    base_defect_manifest=manifests["train_defect"],
                    replay_identity_manifest=replay_identity,
                    monitor_manifest=monitor_manifest,
                    expected_epoch_samples=120_000 + job.total_replay_slots,
                    expected_replay_samples=job.total_replay_slots,
                )
                spec = DynamicTrainingSpec(
                    run_id=job.logical_run_id,
                    arm_id=job.logical_arm_id,
                    schedule_id=job.schedule_id,
                    selection_digest=job.selection_pool_digest,
                    active_selection_digest=job.active_selection_digest,
                    dataset_dir=cache.dataset_dir,
                    checkpoint=machine.path_value("base_checkpoint"),
                    output_dir=output,
                    yolo_root=machine.path_value("repo_root") / "YOLOv11",
                    total_epochs=200,
                    segment_start_epoch=decision.actual_start_epoch,
                    segment_end_epoch=job.segment_end_epoch,
                    batch=128,
                    imgsz=224,
                    seed=job.training_seed,
                    device=str(machine.data["gpu_id"]),
                    workers=int(machine.data["num_workers"]),
                    expected_steps_per_epoch=job.expected_steps_batch128,
                    retained_checkpoint_epochs=(120, 140, 150, 160, 180, 200),
                    execution_mode="FORMAL",
                    resume_checkpoint=decision.resume_checkpoint,
                    segment_id=segment_id,
                    process_telemetry=telemetry,
                    canonical_lock_path=job.canonical_lock,
                    canonical_lock_file_sha256=job.canonical_lock_file_sha256,
                    runtime_health_check=lease.raise_if_lost,
                )
                with staged_identity_replay_session(
                    cache,
                    replay_identity,
                    run_slot=job.job_id,
                    expected_replay_rows=job.total_replay_slots,
                ):
                    monitor.set_phase("TRAIN_COMPUTE_AND_EPOCH_EVAL")
                    run_dynamic_training_segment(spec)
            monitor.set_phase("SEGMENT_ARTIFACT_VALIDATION")
            lease.raise_if_lost()
            completed = _validate_job_segment(output, job)
            prediction_result = None
            if job.retained_checkpoint_epochs:
                monitor.set_phase("KEY_CHECKPOINT_PREDICTION")
                prediction_result = run_key_checkpoint_predictions(
                    CampaignCheckpointPredictionSpec(
                        run_id=job.logical_run_id,
                        arm_id=job.logical_arm_id,
                        job_id=job.job_id,
                        checkpoint_epochs=job.retained_checkpoint_epochs,
                        training_state_dir=output / "training_state",
                        output_dir=output / "key_checkpoint_predictions",
                        dataset_root=machine.path_value("dataset_root"),
                        normal_train_manifest=machine.path_value("normal_train_manifest"),
                        defect_train_manifest=machine.path_value("train_manifest"),
                        monitor_manifest=monitor_manifest,
                        val_op_normal_manifest=machine.path_value("val_op_normal_manifest"),
                        val_op_defect_manifest=machine.path_value("val_op_defect_manifest"),
                        yolo_root=machine.path_value("repo_root") / "YOLOv11",
                        python_executable=str(machine.data.get("python_executable") or sys.executable),
                        gpu_id=str(machine.data["gpu_id"]),
                        batch=int(machine.data.get("prediction_batch_size", 256)),
                        workers=int(machine.data.get("prediction_workers", machine.data["num_workers"])),
                        imgsz=224,
                        accepted_defect_names=(
                            "defect",
                            "Defect",
                            "def",
                            "Def",
                            "abnormal",
                            "Abnormal",
                            "1",
                            "target_defect",
                        ),
                    )
                )
            result_path = output / "job_results" / f"{job.job_id}.json"
            monitor.set_phase("RESULT_PUBLICATION")
            lease.check_now()
            atomic_write_json(
                result_path,
                {
                    "schema_version": "stage1.dynamic_campaign_job_result.v1",
                    "status": "COMPLETE",
                    "job_id": job.job_id,
                    "logical_run_id": job.logical_run_id,
                    "logical_arm_id": job.logical_arm_id,
                    "action": decision.action,
                    "completed_epoch": completed,
                    "segment_end_epoch": job.segment_end_epoch,
                    "active_selection_digest": job.active_selection_digest,
                    "selection_pool_digest": job.selection_pool_digest,
                    "canonical_lock_file_sha256": job.canonical_lock_file_sha256,
                    "assignment_id": assignment.assignment_id,
                    "assignment_sha256": assignment.sha256,
                    "assigned_machine_id": str(machine.data["machine_id"]),
                    "planned_machine_slot": job.machine_id,
                    "lease_token": lease.token,
                    "coordination_root": str(coordination_root),
                    "zero_epoch_recovery_archive": None
                    if recovery_archive is None
                    else str(recovery_archive),
                    "zero_epoch_recovery_manifest_sha256": None
                    if recovery_archive is None
                    else sha256_file(recovery_archive / "ATTEMPT_ARCHIVE_MANIFEST.json"),
                    "checkpoint_sha256": sha256_file(
                        output / f"training_state/checkpoint_epoch_{job.segment_end_epoch:04d}.pt"
                    ),
                    "checkpoint_predictions": None
                    if prediction_result is None
                    else {
                        "status": prediction_result.status,
                        "epoch_count": prediction_result.epoch_count,
                        "published_split_count": prediction_result.published_split_count,
                        "skipped_split_count": prediction_result.skipped_split_count,
                        "output_dir": str(prediction_result.output_dir),
                    },
                    "code_provenance": code,
                },
                overwrite=True,
            )
            _write_job_state(
                machine,
                job,
                "COMPLETE",
                assignment=assignment,
                action=decision.action,
                completed_epoch=completed,
                lease_token=lease.token,
            )
            return CampaignJobResult(job.job_id, decision.action, output, completed, state_path, result_path)
        except Exception as exc:
            _write_job_state(
                machine,
                job,
                "FAILED",
                assignment=assignment,
                action=decision.action if decision else None,
                error=exc,
                completed_epoch=None,
                lease_token=lease.token,
            )
            raise
        finally:
            if monitor is not None:
                monitor.stop()


__all__ = [
    "CampaignJob",
    "CampaignWorkerError",
    "CampaignJobResult",
    "SegmentExecutionDecision",
    "archive_zero_epoch_attempt",
    "load_campaign_job",
    "resolve_segment_execution",
    "run_campaign_job",
]
