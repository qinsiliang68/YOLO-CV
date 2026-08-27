"""Versioned machine assignments for immutable dynamic-campaign jobs.

Scientific job identity lives in the frozen queue. Machine placement lives in a
separate assignment artifact so a whole seed block can move to a reserve node
without editing code or rebuilding the scientific manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping
import uuid

import pandas as pd
import yaml

from .campaign_controller import CampaignRelease, load_campaign_release
from .errors import ValidationError
from .util import atomic_write_bytes, atomic_write_json, sha256_file


ASSIGNMENT_SCHEMA = "stage1.dynamic_campaign_assignment.v2"


class CampaignAssignmentError(ValidationError):
    """Raised when a machine assignment is incomplete, stale, or tampered."""


@dataclass(frozen=True)
class CampaignAssignmentFiles:
    manifest_path: Path
    job_assignments_path: Path
    block_assignments_path: Path
    standalone_commands_path: Path


@dataclass(frozen=True)
class CampaignAssignment:
    assignment_id: str
    campaign_id: str
    release_id: str
    release_sha256: str
    rows: pd.DataFrame
    manifest_path: Path
    sha256: str

    def assigned_machine(self, job_id: str) -> str:
        match = self.rows.loc[self.rows.job_id.astype(str).eq(str(job_id))]
        if len(match) != 1:
            raise CampaignAssignmentError(f"assignment has no unique job: {job_id}")
        return str(match.iloc[0].assigned_machine_id)


_ASSIGNMENT_COLUMNS = {
    "assignment_order",
    "assignment_id",
    "campaign_id",
    "release_id",
    "release_sha256",
    "queue_registry_sha256",
    "job_id",
    "cycle_id",
    "seed_id",
    "block_id",
    "planned_machine_slot",
    "assigned_machine_id",
    "machine_config_path",
    "machine_config_sha256",
    "dependency_job_id",
}


def _read_queue(queue: Path) -> tuple[pd.DataFrame, dict[str, Any], str]:
    registry = queue / "JOB_EXECUTION_REGISTRY.csv"
    validation_path = queue / "RUN_QUEUE_VALIDATION.json"
    if not registry.is_file() or not validation_path.is_file():
        raise FileNotFoundError(f"campaign queue is incomplete: {queue}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    registry_sha = sha256_file(registry)
    if validation.get("schema_version") != "stage1.dynamic_campaign_run_queue.v2":
        raise CampaignAssignmentError("new assignments require run queue v2")
    if validation.get("status") != "PASS":
        raise CampaignAssignmentError("campaign queue validation status is not PASS")
    if str(validation.get("job_registry_sha256", "")).upper() != registry_sha:
        raise CampaignAssignmentError("campaign queue registry checksum mismatch")
    frame = pd.read_csv(registry, keep_default_na=False)
    required = {
        "queue_order",
        "job_id",
        "cycle_id",
        "release_state",
        "machine_id",
        "seed_id",
        "logical_run_id",
        "logical_arm_id",
        "dependency_job_id",
    }
    missing = required - set(frame.columns)
    if missing:
        raise CampaignAssignmentError(f"campaign queue missing columns: {sorted(missing)}")
    if len(frame) != int(validation.get("job_count", -1)):
        raise CampaignAssignmentError("campaign queue row count mismatch")
    if frame.job_id.astype(str).duplicated().any():
        raise CampaignAssignmentError("campaign queue contains duplicate job_id")
    return frame.sort_values("queue_order", kind="stable"), validation, registry_sha


def _machine_configs(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    result: dict[str, Path] = {}
    for path in sorted((*root.glob("*.yaml"), *root.glob("*.yml"))):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CampaignAssignmentError(f"machine config is unreadable: {path}") from exc
        machine_id = str(payload.get("machine_id", "")).strip() if isinstance(payload, dict) else ""
        if not machine_id:
            raise CampaignAssignmentError(f"machine config has no machine_id: {path}")
        if machine_id in result:
            raise CampaignAssignmentError(f"duplicate machine config identity: {machine_id}")
        result[machine_id] = path.resolve()
    if not result:
        raise CampaignAssignmentError(f"machine config directory is empty: {root}")
    return result


def _safe_relative(root: Path, relative_text: str, label: str) -> Path:
    relative = Path(str(relative_text).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise CampaignAssignmentError(f"unsafe {label} path: {relative_text}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CampaignAssignmentError(f"{label} path escapes assignment: {path}") from exc
    return path


def _portable_path(path: Path, repo_root: Path | None) -> str:
    if repo_root is not None:
        try:
            return path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            pass
    return path.resolve().as_posix()


_PLAIN_TOKEN = re.compile(r"^[A-Za-z0-9_./:\\-]+$")


def _powershell_token(value: str) -> str:
    return value if _PLAIN_TOKEN.fullmatch(value) else "'" + value.replace("'", "''") + "'"


def _command(
    *,
    repo_root: Path | None,
    machine_config: Path,
    campaign_root: Path,
    release_path: Path,
    assignment_manifest: Path,
    job_id: str,
    release_id: str,
    canonical_lock_sha256: str,
) -> str:
    worker = (
        (repo_root / "scripts/stage1_gapvalue240/dynamic_campaign_train_worker.py")
        if repo_root is not None
        else Path("scripts/stage1_gapvalue240/dynamic_campaign_train_worker.py")
    )
    tokens = [
        "uv",
        "run",
        "python",
        _portable_path(worker, repo_root),
        "--machine-config",
        _portable_path(machine_config, repo_root),
        "--campaign-root",
        _portable_path(campaign_root, repo_root),
        "--job-id",
        job_id,
        "--release",
        _portable_path(release_path, repo_root),
        "--assignment",
        _portable_path(assignment_manifest, repo_root),
        "--expected-release-id",
        release_id,
        "--expected-canonical-lock-sha256",
        canonical_lock_sha256,
    ]
    return " ".join(_powershell_token(str(token)) for token in tokens)


def _validate_assignment_rows(
    rows: pd.DataFrame,
    released: pd.DataFrame,
    *,
    assignment_id: str,
    campaign_id: str,
    release: CampaignRelease,
    queue_sha: str,
    config_paths: Mapping[str, Path],
    repo_root: Path | None,
) -> None:
    missing = _ASSIGNMENT_COLUMNS - set(rows.columns)
    if missing:
        raise CampaignAssignmentError(f"assignment rows missing columns: {sorted(missing)}")
    expected_jobs = released.job_id.astype(str).tolist()
    observed_jobs = rows.sort_values("assignment_order", kind="stable").job_id.astype(str).tolist()
    if observed_jobs != expected_jobs or rows.job_id.astype(str).duplicated().any():
        raise CampaignAssignmentError("assignment job order/count differs from release")
    constants = {
        "assignment_id": assignment_id,
        "campaign_id": campaign_id,
        "release_id": release.release_id,
        "release_sha256": release.sha256,
        "queue_registry_sha256": queue_sha,
    }
    for column, expected in constants.items():
        if set(rows[column].astype(str)) != {str(expected)}:
            raise CampaignAssignmentError(f"assignment {column} identity mismatch")
    per_seed = rows.groupby(["cycle_id", "seed_id"], sort=False).assigned_machine_id.nunique()
    if not per_seed.eq(1).all():
        raise CampaignAssignmentError("a seed block is split across machines")
    machine_by_job = rows.set_index(rows.job_id.astype(str)).assigned_machine_id.astype(str).to_dict()
    for row in rows.itertuples(index=False):
        dependency = str(row.dependency_job_id)
        if dependency and machine_by_job.get(dependency) != str(row.assigned_machine_id):
            raise CampaignAssignmentError(
                f"dependency is split across machines: {dependency} -> {row.job_id}"
            )
        machine_id = str(row.assigned_machine_id)
        path = config_paths.get(machine_id)
        if path is None:
            raise CampaignAssignmentError(f"assignment references unknown machine: {machine_id}")
        if str(row.machine_config_path) != _portable_path(path, repo_root):
            raise CampaignAssignmentError(f"assignment machine config path mismatch: {machine_id}")
        if str(row.machine_config_sha256).upper() != sha256_file(path):
            raise CampaignAssignmentError(f"assignment machine config checksum mismatch: {machine_id}")


def build_campaign_assignment(
    queue_dir: str | Path,
    release_path: str | Path,
    output_dir: str | Path,
    *,
    campaign_id: str,
    assignment_id: str,
    machine_configs_dir: str | Path,
    slot_mapping: Mapping[str, str],
    seed_overrides: Mapping[str, str] | None = None,
    supersedes_assignment: str | Path | None = None,
    reassignment_reason: str | None = None,
    repo_root: str | Path | None = None,
) -> CampaignAssignmentFiles:
    """Build an immutable assignment version and one standalone command per job."""

    queue = Path(queue_dir).resolve()
    release_path_resolved = Path(release_path).resolve()
    output = Path(output_dir).resolve()
    config_root = Path(machine_configs_dir).resolve()
    repo = Path(repo_root).resolve() if repo_root is not None else None
    if not str(assignment_id).strip():
        raise CampaignAssignmentError("assignment_id must not be empty")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"assignment output is not empty: {output}")
    registry, validation, queue_sha = _read_queue(queue)
    canonical_lock_sha = str(validation.get("canonical_lock_file_sha256", "")).upper()
    if len(canonical_lock_sha) != 64:
        raise CampaignAssignmentError("campaign queue has no canonical lock SHA-256")
    release = load_campaign_release(queue, release_path_resolved, expected_campaign_id=campaign_id)
    released = registry.loc[registry.job_id.astype(str).isin(release.job_ids)].copy()
    released = released.sort_values("queue_order", kind="stable")
    configs = _machine_configs(config_root)
    overrides = {str(key): str(value) for key, value in (seed_overrides or {}).items()}
    known_seeds = set(released.seed_id.astype(str))
    unknown_seeds = sorted(set(overrides) - known_seeds)
    if unknown_seeds:
        raise CampaignAssignmentError(f"seed override contains unknown seed: {unknown_seeds}")
    unknown_machines = sorted(
        ({str(value) for value in slot_mapping.values()} | set(overrides.values())) - set(configs)
    )
    if unknown_machines:
        raise CampaignAssignmentError(f"assignment references unknown machine: {unknown_machines}")

    previous: CampaignAssignment | None = None
    if supersedes_assignment is not None:
        if not str(reassignment_reason or "").strip():
            raise CampaignAssignmentError("a superseding assignment requires a reassignment reason")
        previous = load_campaign_assignment(
            queue,
            release_path_resolved,
            supersedes_assignment,
            expected_campaign_id=campaign_id,
            repo_root=repo,
        )
        if previous.assignment_id == assignment_id:
            raise CampaignAssignmentError("superseding assignment must use a new assignment_id")
    elif overrides and not str(reassignment_reason or "").strip():
        raise CampaignAssignmentError("seed overrides require a reassignment reason")

    previous_machine: dict[str, str] = {}
    if previous is not None:
        previous_machine = previous.rows.set_index(previous.rows.job_id.astype(str)).assigned_machine_id.astype(str).to_dict()
    machine_by_seed: dict[str, str] = {}
    planning_slots_by_seed: dict[str, tuple[str, ...]] = {}
    for seed_id, seed_rows in released.groupby("seed_id", sort=False):
        seed = str(seed_id)
        ordered_slots = tuple(dict.fromkeys(seed_rows.machine_id.astype(str).tolist()))
        if not ordered_slots:
            raise CampaignAssignmentError(f"seed block has no planning slot: {seed}")
        planning_slots_by_seed[seed] = ordered_slots
        if seed in overrides:
            machine = overrides[seed]
        elif previous is not None:
            machines = {previous_machine[str(job_id)] for job_id in seed_rows.job_id.astype(str)}
            if len(machines) != 1:
                raise CampaignAssignmentError(f"previous assignment split seed block: {seed}")
            machine = machines.pop()
        else:
            # Planning slots are advisory placement hints from the frozen queue.
            # Assignment v2 is the sole placement layer and must co-locate the
            # entire cycle/seed block.  Use the first queue-ordered slot as the
            # deterministic anchor and preserve every original slot per job.
            anchor_slot = ordered_slots[0]
            if anchor_slot not in slot_mapping:
                raise CampaignAssignmentError(
                    f"planning slot has no machine mapping: {anchor_slot}"
                )
            machine = str(slot_mapping[anchor_slot])
        if machine not in configs:
            raise CampaignAssignmentError(f"assignment references unknown machine: {machine}")
        machine_by_seed[seed] = machine

    assignment_manifest = output / "ASSIGNMENT_MANIFEST.json"
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(released.itertuples(index=False), start=1):
        machine_id = machine_by_seed[str(row.seed_id)]
        rows.append(
            {
                "assignment_order": index,
                "assignment_id": assignment_id,
                "campaign_id": campaign_id,
                "release_id": release.release_id,
                "release_sha256": release.sha256,
                "queue_registry_sha256": queue_sha,
                "job_id": str(row.job_id),
                "cycle_id": str(row.cycle_id),
                "seed_id": str(row.seed_id),
                "block_id": f"{row.cycle_id}:{row.seed_id}",
                "planned_machine_slot": str(row.machine_id),
                "assigned_machine_id": machine_id,
                "machine_config_path": _portable_path(configs[machine_id], repo),
                "machine_config_sha256": sha256_file(configs[machine_id]),
                "dependency_job_id": str(row.dependency_job_id),
            }
        )
    assignments = pd.DataFrame(rows)
    _validate_assignment_rows(
        assignments,
        released,
        assignment_id=assignment_id,
        campaign_id=campaign_id,
        release=release,
        queue_sha=queue_sha,
        config_paths=configs,
        repo_root=repo,
    )

    staging = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmpdir"
    try:
        staging.mkdir(parents=True)
        assignments_path = staging / "JOB_ASSIGNMENTS.csv"
        atomic_write_bytes(
            assignments_path,
            assignments.to_csv(index=False, lineterminator="\n").encode("utf-8"),
        )
        block_rows = []
        for (cycle_id, seed_id), frame in assignments.groupby(["cycle_id", "seed_id"], sort=False):
            block_rows.append(
                {
                    "assignment_id": assignment_id,
                    "block_id": str(frame.block_id.iloc[0]),
                    "cycle_id": str(cycle_id),
                    "seed_id": str(seed_id),
                    "assigned_machine_id": str(frame.assigned_machine_id.iloc[0]),
                    "machine_config_path": str(frame.machine_config_path.iloc[0]),
                    "machine_config_sha256": str(frame.machine_config_sha256.iloc[0]),
                    "job_count": len(frame),
                    "first_queue_order": int(frame.assignment_order.min()),
                    "last_queue_order": int(frame.assignment_order.max()),
                }
            )
        blocks_path = staging / "BLOCK_ASSIGNMENTS.csv"
        atomic_write_bytes(
            blocks_path,
            pd.DataFrame(block_rows).to_csv(index=False, lineterminator="\n").encode("utf-8"),
        )

        command_rows = []
        final_manifest_path = output / assignment_manifest.name
        for row in assignments.itertuples(index=False):
            command = _command(
                repo_root=repo,
                machine_config=configs[str(row.assigned_machine_id)],
                campaign_root=queue.parent,
                release_path=release.path,
                assignment_manifest=final_manifest_path,
                job_id=str(row.job_id),
                release_id=release.release_id,
                canonical_lock_sha256=canonical_lock_sha,
            )
            command_rows.append(
                {
                    "assignment_order": int(row.assignment_order),
                    "assignment_id": assignment_id,
                    "block_id": str(row.block_id),
                    "job_id": str(row.job_id),
                    "assigned_machine_id": str(row.assigned_machine_id),
                    "machine_config_path": str(row.machine_config_path),
                    "release_id": release.release_id,
                    "canonical_lock_file_sha256": canonical_lock_sha,
                    "command": command,
                    "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest().upper(),
                }
            )
        commands_path = staging / "STANDALONE_JOB_COMMANDS.csv"
        atomic_write_bytes(
            commands_path,
            pd.DataFrame(command_rows).to_csv(index=False, lineterminator="\n").encode("utf-8"),
        )
        machine_dir = staging / "machines"
        for machine_id, frame in assignments.groupby("assigned_machine_id", sort=True):
            atomic_write_bytes(
                machine_dir / f"{machine_id}.csv",
                frame.to_csv(index=False, lineterminator="\n").encode("utf-8"),
            )
        atomic_write_json(
            staging / assignment_manifest.name,
            {
                "schema_version": ASSIGNMENT_SCHEMA,
                "status": "COMPLETE",
                "campaign_id": campaign_id,
                "assignment_id": assignment_id,
                "release_id": release.release_id,
                "release_sha256": release.sha256,
                "queue_registry_sha256": queue_sha,
                "job_count": len(assignments),
                "block_count": len(block_rows),
                "machine_count": int(assignments.assigned_machine_id.nunique()),
                "planning_slots_by_seed": {
                    seed: list(slots) for seed, slots in planning_slots_by_seed.items()
                },
                "multi_slot_seed_blocks_collapsed": sorted(
                    seed for seed, slots in planning_slots_by_seed.items() if len(slots) > 1
                ),
                "seed_block_anchor_policy": "FIRST_QUEUE_ORDERED_PLANNING_SLOT",
                "job_assignments_relpath": assignments_path.relative_to(staging).as_posix(),
                "job_assignments_sha256": sha256_file(assignments_path),
                "block_assignments_relpath": blocks_path.relative_to(staging).as_posix(),
                "block_assignments_sha256": sha256_file(blocks_path),
                "standalone_commands_relpath": commands_path.relative_to(staging).as_posix(),
                "standalone_commands_sha256": sha256_file(commands_path),
                "supersedes_assignment_id": previous.assignment_id if previous else None,
                "supersedes_assignment_sha256": (
                    sha256_file(previous.manifest_path) if previous else None
                ),
                "reassignment_reason": reassignment_reason,
                "assignment_unit": "CYCLE_AND_SEED_BLOCK",
                "worker_entrypoint": "scripts/stage1_gapvalue240/dynamic_campaign_train_worker.py",
                "controller_optional": True,
                "standalone_execution_unit": "PHYSICAL_JOB",
                "standalone_command_count": len(command_rows),
                "single_job_per_process": True,
                "implicit_next_job_forbidden": True,
                "dynamic_reassignment_mode": "NEW_IMMUTABLE_ASSIGNMENT_ONLY",
                "training_code_edits_required": False,
                "coordination_required": True,
                "assignment_activation_entrypoint": (
                    "scripts/stage1_gapvalue240/activate_dynamic_campaign_assignment.py"
                ),
                "lease_unit": "PHYSICAL_JOB",
                "lease_fencing": "ACTIVE_ASSIGNMENT_SHA256_AND_RANDOM_TOKEN",
                "machine_config_path_mode": (
                    "REPO_RELATIVE_WHEN_PORTABLE" if repo is not None else "ABSOLUTE"
                ),
            },
        )
        if output.exists():
            output.rmdir()
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return CampaignAssignmentFiles(
        output / "ASSIGNMENT_MANIFEST.json",
        output / "JOB_ASSIGNMENTS.csv",
        output / "BLOCK_ASSIGNMENTS.csv",
        output / "STANDALONE_JOB_COMMANDS.csv",
    )


def load_campaign_assignment(
    queue_dir: str | Path,
    release_path: str | Path,
    assignment_manifest: str | Path,
    *,
    expected_campaign_id: str,
    expected_machine_id: str | None = None,
    expected_job_id: str | None = None,
    repo_root: str | Path | None = None,
) -> CampaignAssignment:
    """Validate an assignment version before a worker reaches the GPU."""

    queue = Path(queue_dir).resolve()
    manifest_path = Path(assignment_manifest).resolve()
    repo = Path(repo_root).resolve() if repo_root is not None else None
    registry, _validation, queue_sha = _read_queue(queue)
    release = load_campaign_release(queue, release_path, expected_campaign_id=expected_campaign_id)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CampaignAssignmentError("assignment manifest is unreadable") from exc
    expected = {
        "schema_version": ASSIGNMENT_SCHEMA,
        "status": "COMPLETE",
        "campaign_id": expected_campaign_id,
        "release_id": release.release_id,
        "release_sha256": release.sha256,
        "queue_registry_sha256": queue_sha,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if str(manifest.get(key, "")) != str(value)
    }
    if mismatches:
        raise CampaignAssignmentError(f"assignment manifest identity mismatch: {mismatches}")
    root = manifest_path.parent.resolve()
    assignments_path = _safe_relative(root, manifest.get("job_assignments_relpath", ""), "assignments")
    commands_path = _safe_relative(root, manifest.get("standalone_commands_relpath", ""), "commands")
    for path, key in (
        (assignments_path, "job_assignments_sha256"),
        (commands_path, "standalone_commands_sha256"),
    ):
        if not path.is_file() or sha256_file(path) != str(manifest.get(key, "")).upper():
            raise CampaignAssignmentError(f"assignment artifact checksum mismatch: {path}")
    rows = pd.read_csv(assignments_path, keep_default_na=False)
    commands = pd.read_csv(commands_path, keep_default_na=False)
    if len(rows) != int(manifest.get("job_count", -1)) or len(commands) != len(rows):
        raise CampaignAssignmentError("assignment artifact row count mismatch")
    if commands.job_id.astype(str).tolist() != rows.job_id.astype(str).tolist():
        raise CampaignAssignmentError("standalone command identities differ from assignment")
    if commands.job_id.astype(str).duplicated().any():
        raise CampaignAssignmentError("standalone commands contain duplicate job_id")
    released = registry.loc[registry.job_id.astype(str).isin(release.job_ids)].sort_values(
        "queue_order", kind="stable"
    )
    config_paths: dict[str, Path] = {}
    for row in rows.itertuples(index=False):
        machine_id = str(row.assigned_machine_id)
        configured_path = Path(str(row.machine_config_path).replace("\\", "/"))
        if configured_path.is_absolute():
            path = configured_path.resolve()
        else:
            if repo is None or ".." in configured_path.parts:
                raise CampaignAssignmentError(
                    "relative machine config path requires a trusted repo_root"
                )
            path = (repo / configured_path).resolve()
            try:
                path.relative_to(repo)
            except ValueError as exc:
                raise CampaignAssignmentError(
                    f"machine config path escapes repo root: {configured_path}"
                ) from exc
        if not path.is_file():
            raise CampaignAssignmentError(f"assigned machine config is missing: {path}")
        if sha256_file(path) != str(row.machine_config_sha256).upper():
            raise CampaignAssignmentError(f"assigned machine config checksum mismatch: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or str(payload.get("machine_id", "")) != machine_id:
            raise CampaignAssignmentError(f"assigned machine config identity mismatch: {path}")
        known = config_paths.get(machine_id)
        if known is not None and known != path:
            raise CampaignAssignmentError(f"machine has multiple config paths: {machine_id}")
        config_paths[machine_id] = path
    assignment_id = str(manifest.get("assignment_id", ""))
    _validate_assignment_rows(
        rows,
        released,
        assignment_id=assignment_id,
        campaign_id=expected_campaign_id,
        release=release,
        queue_sha=queue_sha,
        config_paths=config_paths,
        repo_root=repo,
    )
    if expected_job_id is not None:
        match = rows.loc[rows.job_id.astype(str).eq(str(expected_job_id))]
        if len(match) != 1:
            raise CampaignAssignmentError(f"assignment does not contain job: {expected_job_id}")
        assigned = str(match.iloc[0].assigned_machine_id)
        if expected_machine_id is not None and assigned != str(expected_machine_id):
            raise CampaignAssignmentError(
                f"job {expected_job_id} is assigned to {assigned}, not {expected_machine_id}"
            )
    elif expected_machine_id is not None:
        if not rows.assigned_machine_id.astype(str).eq(str(expected_machine_id)).any():
            raise CampaignAssignmentError(f"assignment contains no jobs for {expected_machine_id}")
    return CampaignAssignment(
        assignment_id=assignment_id,
        campaign_id=expected_campaign_id,
        release_id=release.release_id,
        release_sha256=release.sha256,
        rows=rows,
        manifest_path=manifest_path,
        sha256=sha256_file(manifest_path),
    )


__all__ = [
    "ASSIGNMENT_SCHEMA",
    "CampaignAssignment",
    "CampaignAssignmentError",
    "CampaignAssignmentFiles",
    "build_campaign_assignment",
    "load_campaign_assignment",
]
