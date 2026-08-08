"""Dependency-aware release gates and scheduling for the dynamic replay campaign."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid
from typing import Any, Mapping

import pandas as pd

from .errors import ValidationError
from .campaign_engineering_gate import (
    ENGINEERING_GATE_SCHEMA_V2,
    ValidationIdentity,
    validate_engineering_gate_v2,
)
from .subprocesses import terminate_process_tree
from .util import atomic_write_json, sha256_file


class CampaignControllerError(ValidationError):
    """Raised when a release or controller state violates the frozen queue."""


@dataclass(frozen=True)
class CampaignReleaseFiles:
    pilot_release: Path
    confirmatory_hold: Path
    future_cycle_hold: Path


@dataclass(frozen=True)
class CampaignRelease:
    release_id: str
    campaign_id: str
    scope: str
    seed_ids: tuple[str, ...]
    job_ids: tuple[str, ...]
    path: Path
    sha256: str


@dataclass(frozen=True)
class ControllerIterationPlan:
    next_job_id: str | None
    overall_state: str
    job_states: dict[str, str]
    counts: dict[str, int]


def _process_group_options() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def run_worker_process(
    command: list[str],
    *,
    cwd: str | Path,
    log_path: str | Path,
    heartbeat,
    poll_seconds: float = 30.0,
) -> dict[str, Any]:
    """Run one isolated worker while the controller remains lightweight and alive."""

    if poll_seconds <= 0:
        raise CampaignControllerError("worker poll interval must be positive")
    workdir = Path(cwd).resolve()
    log = Path(log_path).resolve()
    log.parent.mkdir(parents=True, exist_ok=True)
    result_path = log.with_suffix(log.suffix + ".result.json")
    started = time.time()
    process: subprocess.Popen | None = None
    termination = {"terminated_pids": [], "killed_pids": [], "errors": []}
    try:
        with log.open("wb") as stream:
            process = subprocess.Popen(
                list(map(str, command)),
                cwd=workdir,
                stdout=stream,
                stderr=subprocess.STDOUT,
                **_process_group_options(),
            )
            while True:
                heartbeat(process.pid)
                try:
                    returncode = process.wait(timeout=poll_seconds)
                    break
                except subprocess.TimeoutExpired:
                    continue
    except BaseException:
        if process is not None and process.poll() is None:
            termination = terminate_process_tree(process.pid)
        raise
    ended = time.time()
    result = {
        "schema_version": "stage1.dynamic_campaign_worker_process.v1",
        "status": "PASS" if returncode == 0 else "FAILED",
        "command": list(map(str, command)),
        "cwd": str(workdir),
        "pid": process.pid,
        "returncode": int(returncode),
        "started_at_unix": started,
        "ended_at_unix": ended,
        "duration_seconds": ended - started,
        "log": str(log),
        "termination": termination,
    }
    atomic_write_json(result_path, result, overwrite=True)
    return result


def _load_frozen_registry(queue_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    registry_path = queue_dir / "JOB_EXECUTION_REGISTRY.csv"
    validation_path = queue_dir / "RUN_QUEUE_VALIDATION.json"
    if not registry_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError(f"campaign queue is incomplete: {queue_dir}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("schema_version") != "stage1.dynamic_campaign_run_queue.v2":
        raise CampaignControllerError("new campaign releases require run queue v2")
    if validation.get("status") != "PASS":
        raise CampaignControllerError("campaign queue validation status is not PASS")
    actual_sha = sha256_file(registry_path)
    if str(validation.get("job_registry_sha256", "")).upper() != actual_sha:
        raise CampaignControllerError("campaign job registry checksum mismatch")
    frame = pd.read_csv(registry_path, keep_default_na=False)
    required = {
        "queue_order",
        "job_id",
        "machine_id",
        "seed_id",
        "dependency_job_id",
        "cycle_id",
        "release_state",
    }
    missing = required - set(frame.columns)
    if missing:
        raise CampaignControllerError(f"campaign registry missing columns: {sorted(missing)}")
    if len(frame) != int(validation.get("job_count", -1)):
        raise CampaignControllerError("campaign job registry row count mismatch")
    if frame.job_id.astype(str).duplicated().any():
        raise CampaignControllerError("campaign job registry contains duplicate job_id")
    return frame.sort_values("queue_order", kind="stable"), validation


def _release_payload(
    *,
    campaign_id: str,
    release_id: str,
    scope: str,
    release_status: str,
    queue_sha256: str,
    canonical_lock_file_sha256: str,
    rows: pd.DataFrame,
    engineering_gate_report_sha256: str | None,
    engineering_gate_schema_version: str | None,
    engineering_gate_source_tree_sha256: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": "stage1.dynamic_campaign_release.v2",
        "campaign_id": campaign_id,
        "release_id": release_id,
        "scope": scope,
        "release_status": release_status,
        "queue_registry_sha256": queue_sha256,
        "canonical_lock_file_sha256": canonical_lock_file_sha256,
        "seed_ids": rows.seed_id.astype(str).drop_duplicates().tolist(),
        "cycle_ids": rows.cycle_id.astype(str).drop_duplicates().tolist(),
        "job_ids": rows.job_id.astype(str).tolist(),
        "job_count": len(rows),
        "dependency_policy": "CLOSED_WITHIN_RELEASE",
        "engineering_gate_report_sha256": engineering_gate_report_sha256,
        "engineering_gate_schema_version": engineering_gate_schema_version,
        "engineering_gate_source_tree_sha256": engineering_gate_source_tree_sha256,
    }


def _validate_engineering_gate(
    path: str | Path | None,
    *,
    validation: Mapping[str, Any],
) -> tuple[Path, str, dict[str, Any]]:
    if path is None:
        raise CampaignControllerError(
            "engineering gate report is required before any Cycle-1 release"
        )
    report_path = Path(path).resolve()
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CampaignControllerError("engineering gate report is unreadable") from exc
    if report.get("schema_version") != ENGINEERING_GATE_SCHEMA_V2:
        raise CampaignControllerError(
            "new campaign releases require engineering gate v2; v1 is historical only"
        )
    identity_payload = report.get("identity", {})
    try:
        expected_identity = ValidationIdentity(
            source_tree_sha256=str(identity_payload.get("source_tree_sha256", "")),
            queue_registry_sha256=str(validation["job_registry_sha256"]),
            canonical_lock_file_sha256=str(
                validation.get("canonical_lock_file_sha256", "")
            ),
        )
        validated = validate_engineering_gate_v2(
            report_path,
            expected_identity=expected_identity,
            allowed_root=report_path.parent,
        )
    except (ValidationError, KeyError) as exc:
        raise CampaignControllerError(f"engineering gate report is not complete: {exc}") from exc
    return report_path, sha256_file(report_path), validated

def build_campaign_release_manifests(
    queue_dir: str | Path,
    output_dir: str | Path,
    *,
    campaign_id: str,
    pilot_seed_ids: tuple[str, ...] = ("S001", "S002"),
    engineering_gate_report: str | Path | None = None,
) -> CampaignReleaseFiles:
    """Freeze a runnable pilot and a non-runnable confirmatory hold manifest."""

    queue = Path(queue_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"release output is not empty: {output}")
    registry, validation = _load_frozen_registry(queue)
    gate_path, gate_sha, gate_report = _validate_engineering_gate(
        engineering_gate_report,
        validation=validation,
    )
    cycle_one = registry.loc[registry.cycle_id.astype(str).eq("CYCLE_1")].copy()
    future = registry.loc[~registry.cycle_id.astype(str).eq("CYCLE_1")].copy()
    if cycle_one.empty or not cycle_one.release_state.astype(str).eq("ENGINEERING_GATE").all():
        raise CampaignControllerError("Cycle-1 queue is missing or has invalid release states")
    if future.empty:
        raise CampaignControllerError("future-cycle scientific hold is empty")
    all_seeds = set(cycle_one.seed_id.astype(str))
    requested = set(map(str, pilot_seed_ids))
    if not requested or not requested <= all_seeds:
        raise CampaignControllerError(
            f"pilot seeds are not a non-empty subset of the queue: {sorted(requested - all_seeds)}"
        )
    pilot = cycle_one.loc[cycle_one.seed_id.astype(str).isin(requested)].copy()
    confirmatory = cycle_one.loc[~cycle_one.seed_id.astype(str).isin(requested)].copy()
    staging = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmpdir"
    try:
        staging.mkdir(parents=True)
        queue_sha = str(validation["job_registry_sha256"]).upper()
        canonical_lock_sha = str(validation["canonical_lock_file_sha256"]).upper()
        gate_identity = ValidationIdentity.from_mapping(gate_report["identity"])
        pilot_token = "_".join(pilot.seed_id.astype(str).drop_duplicates().tolist())
        pilot_path = staging / f"PILOT_{pilot_token}_RELEASED.json"
        hold_path = staging / "CONFIRMATORY_REMAINING_HOLD.json"
        future_hold_path = staging / "FUTURE_CYCLES_SCIENTIFIC_HOLD.json"
        shutil.copy2(gate_path, staging / "ENGINEERING_GATE_REPORT.json")
        atomic_write_json(
            pilot_path,
            _release_payload(
                campaign_id=campaign_id,
                release_id=f"PILOT_{pilot_token}",
                scope="PILOT",
                release_status="RELEASED",
                queue_sha256=queue_sha,
                canonical_lock_file_sha256=canonical_lock_sha,
                rows=pilot,
                engineering_gate_report_sha256=gate_sha,
                engineering_gate_schema_version=ENGINEERING_GATE_SCHEMA_V2,
                engineering_gate_source_tree_sha256=gate_identity.source_tree_sha256,
            ),
        )
        atomic_write_json(
            hold_path,
            _release_payload(
                campaign_id=campaign_id,
                release_id="CONFIRMATORY_REMAINING",
                scope="CONFIRMATORY",
                release_status="HOLD",
                queue_sha256=queue_sha,
                canonical_lock_file_sha256=canonical_lock_sha,
                rows=confirmatory,
                engineering_gate_report_sha256=gate_sha,
                engineering_gate_schema_version=ENGINEERING_GATE_SCHEMA_V2,
                engineering_gate_source_tree_sha256=gate_identity.source_tree_sha256,
            ),
        )
        atomic_write_json(
            future_hold_path,
            _release_payload(
                campaign_id=campaign_id,
                release_id="FUTURE_CYCLES",
                scope="FUTURE_SCIENTIFIC_CYCLES",
                release_status="HOLD",
                queue_sha256=queue_sha,
                canonical_lock_file_sha256=canonical_lock_sha,
                rows=future,
                engineering_gate_report_sha256=gate_sha,
                engineering_gate_schema_version=ENGINEERING_GATE_SCHEMA_V2,
                engineering_gate_source_tree_sha256=gate_identity.source_tree_sha256,
            ),
        )
        if output.exists():
            output.rmdir()
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return CampaignReleaseFiles(
        output / pilot_path.name,
        output / hold_path.name,
        output / future_hold_path.name,
    )


def load_campaign_release(
    queue_dir: str | Path,
    release_path: str | Path,
    *,
    expected_campaign_id: str,
) -> CampaignRelease:
    queue = Path(queue_dir).resolve()
    path = Path(release_path).resolve()
    registry, validation = _load_frozen_registry(queue)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "stage1.dynamic_campaign_release.v2":
        raise CampaignControllerError("unsupported campaign release schema; v2 required")
    if payload.get("campaign_id") != expected_campaign_id:
        raise CampaignControllerError("campaign release is bound to a different campaign")
    if payload.get("release_status") != "RELEASED":
        raise CampaignControllerError(f"campaign release is not RELEASED: {payload.get('release_status')}")
    if str(payload.get("queue_registry_sha256", "")).upper() != str(
        validation["job_registry_sha256"]
    ).upper():
        raise CampaignControllerError("campaign release queue checksum mismatch")
    expected_lock_sha = str(validation.get("canonical_lock_file_sha256", "")).upper()
    if str(payload.get("canonical_lock_file_sha256", "")).upper() != expected_lock_sha:
        raise CampaignControllerError("campaign release canonical lock checksum mismatch")
    if payload.get("engineering_gate_schema_version") != ENGINEERING_GATE_SCHEMA_V2:
        raise CampaignControllerError("campaign release is not bound to engineering gate v2")
    gate_copy = path.parent / "ENGINEERING_GATE_REPORT.json"
    if not gate_copy.is_file():
        raise CampaignControllerError("campaign release has no immutable engineering gate copy")
    if str(payload.get("engineering_gate_report_sha256", "")).upper() != sha256_file(gate_copy):
        raise CampaignControllerError("campaign release engineering gate checksum mismatch")
    gate_payload = json.loads(gate_copy.read_text(encoding="utf-8"))
    if gate_payload.get("schema_version") != ENGINEERING_GATE_SCHEMA_V2 or gate_payload.get("status") != "PASS":
        raise CampaignControllerError("campaign release engineering gate copy is invalid")
    gate_identity = ValidationIdentity.from_mapping(gate_payload.get("identity", {}))
    if gate_identity.queue_registry_sha256 != str(validation["job_registry_sha256"]).upper():
        raise CampaignControllerError("campaign release engineering gate queue identity mismatch")
    if gate_identity.canonical_lock_file_sha256 != expected_lock_sha:
        raise CampaignControllerError("campaign release engineering gate canonical identity mismatch")
    if str(payload.get("engineering_gate_source_tree_sha256", "")).upper() != gate_identity.source_tree_sha256:
        raise CampaignControllerError("campaign release engineering gate source identity mismatch")
    job_ids = tuple(map(str, payload.get("job_ids", [])))
    if not job_ids or len(job_ids) != len(set(job_ids)):
        raise CampaignControllerError("campaign release has empty or duplicate job_ids")
    indexed = registry.set_index(registry.job_id.astype(str), drop=False)
    unknown = sorted(set(job_ids) - set(indexed.index))
    if unknown:
        raise CampaignControllerError(f"campaign release contains unknown jobs: {unknown}")
    released = indexed.loc[list(job_ids)].copy()
    expected_order = released.sort_values("queue_order", kind="stable").job_id.astype(str).tolist()
    if list(job_ids) != expected_order or len(job_ids) != int(payload.get("job_count", -1)):
        raise CampaignControllerError("campaign release job order/count mismatch")
    release_set = set(job_ids)
    missing_dependencies = sorted(
        {
            str(row.dependency_job_id)
            for row in released.itertuples(index=False)
            if str(row.dependency_job_id) and str(row.dependency_job_id) not in release_set
        }
    )
    if missing_dependencies:
        raise CampaignControllerError(
            f"campaign release violates dependency closure: {missing_dependencies}"
        )
    actual_seed_ids = tuple(released.seed_id.astype(str).drop_duplicates().tolist())
    if tuple(map(str, payload.get("seed_ids", []))) != actual_seed_ids:
        raise CampaignControllerError("campaign release seed identity mismatch")
    actual_cycle_ids = tuple(released.cycle_id.astype(str).drop_duplicates().tolist())
    if tuple(map(str, payload.get("cycle_ids", []))) != actual_cycle_ids:
        raise CampaignControllerError("campaign release cycle identity mismatch")
    if actual_cycle_ids != ("CYCLE_1",) or not released.release_state.astype(str).eq(
        "ENGINEERING_GATE"
    ).all():
        raise CampaignControllerError("released jobs escape the Cycle-1 engineering gate")
    return CampaignRelease(
        release_id=str(payload["release_id"]),
        campaign_id=str(payload["campaign_id"]),
        scope=str(payload["scope"]),
        seed_ids=actual_seed_ids,
        job_ids=job_ids,
        path=path,
        sha256=sha256_file(path),
    )


def plan_controller_iteration(
    jobs: pd.DataFrame,
    observed_states: Mapping[str, Mapping[str, Any]],
    *,
    now_unix: float | None = None,
    max_attempts: int = 3,
    retry_delay_seconds: float = 300.0,
) -> ControllerIterationPlan:
    """Resolve one deterministic scheduling decision without performing I/O."""

    if max_attempts <= 0 or retry_delay_seconds < 0:
        raise CampaignControllerError("invalid retry policy")
    required = {"queue_order", "job_id", "dependency_job_id"}
    missing = required - set(jobs.columns)
    if missing:
        raise CampaignControllerError(f"controller jobs missing columns: {sorted(missing)}")
    frame = jobs.sort_values("queue_order", kind="stable").copy()
    if frame.job_id.astype(str).duplicated().any():
        raise CampaignControllerError("controller jobs contain duplicate job_id")
    job_ids = set(frame.job_id.astype(str))
    now = time.time() if now_unix is None else float(now_unix)
    states: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        job_id = str(row.job_id)
        payload = dict(observed_states.get(job_id, {}))
        state = str(payload.get("state", "PENDING"))
        if state == "COMPLETE":
            states[job_id] = "COMPLETE"
        elif state == "RUNNING":
            states[job_id] = "RUNNING_EXTERNAL"
        elif state == "FAILED":
            attempts = int(payload.get("attempt_count", 0))
            retryable = bool(payload.get("retryable", False))
            elapsed = now - float(payload.get("updated_at_unix", 0.0))
            if retryable and attempts < max_attempts:
                states[job_id] = "RETRY_READY" if elapsed >= retry_delay_seconds else "RETRY_WAIT"
            else:
                states[job_id] = "FAILED_TERMINAL"
        else:
            states[job_id] = "PENDING"

    unresolved = True
    while unresolved:
        unresolved = False
        for row in frame.itertuples(index=False):
            job_id = str(row.job_id)
            if states[job_id] not in {"PENDING", "RETRY_READY", "RETRY_WAIT"}:
                continue
            dependency = str(row.dependency_job_id)
            if not dependency:
                if states[job_id] == "PENDING":
                    states[job_id] = "READY"
                continue
            if dependency not in job_ids:
                raise CampaignControllerError(f"controller dependency is outside release: {dependency}")
            dependency_state = states[dependency]
            if dependency_state == "COMPLETE":
                if states[job_id] == "PENDING":
                    states[job_id] = "READY"
            elif dependency_state in {"FAILED_TERMINAL", "BLOCKED_DEPENDENCY"}:
                states[job_id] = "BLOCKED_DEPENDENCY"
                unresolved = True
            elif states[job_id] == "PENDING":
                states[job_id] = "WAITING_DEPENDENCY"

    runnable = frame.loc[
        frame.job_id.astype(str).map(states).isin({"READY", "RETRY_READY"})
    ]
    next_job_id = None if runnable.empty else str(runnable.iloc[0].job_id)
    counts = pd.Series(list(states.values()), dtype="string").value_counts().sort_index().to_dict()
    if next_job_id is not None:
        overall = "RUNNABLE"
    elif counts.get("RETRY_WAIT", 0):
        overall = "WAITING_RETRY"
    elif counts.get("RUNNING_EXTERNAL", 0):
        overall = "RUNNING_EXTERNAL"
    elif counts.get("FAILED_TERMINAL", 0) or counts.get("BLOCKED_DEPENDENCY", 0):
        overall = "PARTIAL_FAILURE"
    elif counts.get("COMPLETE", 0) == len(frame):
        overall = "COMPLETE"
    else:
        overall = "DEADLOCK"
    return ControllerIterationPlan(next_job_id, overall, states, {str(k): int(v) for k, v in counts.items()})


__all__ = [
    "CampaignControllerError",
    "CampaignReleaseFiles",
    "CampaignRelease",
    "ControllerIterationPlan",
    "build_campaign_release_manifests",
    "load_campaign_release",
    "plan_controller_iteration",
    "run_worker_process",
]
