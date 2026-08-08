"""Compile the staged preregistration into an immutable physical run queue."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import uuid
from typing import Any

import pandas as pd

from .util import atomic_write_bytes, atomic_write_json, sha256_file, stable_hash


class RunQueueError(RuntimeError):
    """Raised when preregistered jobs cannot be converted into an executable queue."""


@dataclass(frozen=True)
class CampaignRunQueueResult:
    queue_dir: Path
    job_registry: Path
    monitor_manifest: Path
    validation_path: Path


@dataclass(frozen=True)
class ReplayIdentityResult:
    path: Path
    row_count: int
    sha256: str


_SELECTION_COLUMNS = {
    "selection_rank",
    "role_rank",
    "sample_id",
    "y_true",
    "replay_role",
}
_GRAPH_COLUMNS = {
    "job_id",
    "cycle_id",
    "release_state",
    "seed_id",
    "training_seed",
    "logical_arm_id",
    "schedule_id",
    "selection_id",
    "segment_index",
    "segment_start_epoch",
    "segment_end_epoch",
    "normal_replay_slots",
    "defect_guard_slots",
    "total_replay_slots",
    "expected_steps_batch128",
    "dependency_job_id",
    "resume_from_epoch",
    "machine_id",
    "checkpoint_epochs_to_retain",
    "branch_checkpoint_required",
    "canonical_lock_file_sha256",
}
_MATRIX_COLUMNS = {
    "logical_run_id",
    "cycle_id",
    "release_state",
    "seed_id",
    "training_seed",
    "arm_id",
    "selection_id",
    "selection_digest",
    "schedule_id",
    "canonical_lock_file_sha256",
}
_BINDING_COLUMNS = {
    "selection_id",
    "selection_file",
    "selection_digest",
}
_MONITOR_SOURCE_COLUMNS = {
    "sample_id",
    "y_true",
    "oof_fold",
    "dynamic_bucket",
    "mean_p_defect",
    "correct_rate",
    "std_p_defect",
}
_SAFE_RUN_SLOT = re.compile(r"^[A-Za-z0-9_-]+$")


def _read(path: Path, required: set[str], name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, keep_default_na=False)
    missing = required - set(frame.columns)
    if missing:
        raise RunQueueError(f"{name} missing columns: {sorted(missing)}")
    return frame


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0", ""}:
        return False
    raise RunQueueError(f"invalid boolean value: {value!r}")


def _selection_digest(frame: pd.DataFrame) -> str:
    fields = ["selection_rank", "role_rank", "sample_id", "y_true", "replay_role"]
    return stable_hash(frame[fields].to_dict("records"))


def _load_selection(path: Path, selection_id: str) -> pd.DataFrame:
    frame = _read(path, _SELECTION_COLUMNS, f"selection {selection_id}")
    if "selection_id" in frame and len(frame) and set(frame.selection_id.astype(str)) != {
        selection_id
    }:
        raise RunQueueError(f"selection_id mismatch in {path}")
    if frame.sample_id.astype(str).duplicated().any():
        raise RunQueueError(f"selection contains duplicate sample_id: {selection_id}")
    if len(frame):
        labels = pd.to_numeric(frame.y_true, errors="raise").astype(int)
        expected_roles = labels.map({0: "normal_replay", 1: "defect_guard"})
        if expected_roles.isna().any() or not expected_roles.equals(frame.replay_role.astype(str)):
            raise RunQueueError(f"selection label/replay_role contract mismatch: {selection_id}")
        ranks = pd.to_numeric(frame.selection_rank, errors="raise").astype(int)
        if ranks.tolist() != list(range(1, len(frame) + 1)):
            raise RunQueueError(f"selection_rank is not contiguous: {selection_id}")
    return frame


def _binding_maps(
    prereg: Path,
) -> tuple[dict[str, Path], dict[str, str], dict[str, pd.DataFrame]]:
    bindings = _read(prereg / "SELECTION_BINDINGS.csv", _BINDING_COLUMNS, "selection bindings")
    if bindings.selection_id.astype(str).duplicated().any():
        raise RunQueueError("selection bindings contain duplicate selection_id")
    paths: dict[str, Path] = {}
    digests: dict[str, str] = {}
    frames: dict[str, pd.DataFrame] = {}
    for row in bindings.itertuples(index=False):
        selection_id = str(row.selection_id)
        relative = Path(str(row.selection_file).replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise RunQueueError(f"unsafe selection_file: {relative}")
        path = (prereg / relative).resolve()
        try:
            path.relative_to(prereg)
        except ValueError as exc:
            raise RunQueueError(f"selection_file escapes preregistration: {path}") from exc
        frame = _load_selection(path, selection_id)
        expected = str(row.selection_digest).upper()
        actual = _selection_digest(frame)
        if actual != expected:
            raise RunQueueError(
                f"selection digest mismatch for {selection_id}: {actual} != {expected}"
            )
        paths[selection_id] = path
        digests[selection_id] = expected
        frames[selection_id] = frame
    return paths, digests, frames


def _canonical_lock_binding(prereg: Path) -> tuple[Path, str, dict[str, Any]]:
    binding_path = prereg / "CANONICAL_TRAINING_LOCK_BINDING.json"
    if not binding_path.is_file():
        raise FileNotFoundError(binding_path)
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    lock_path = Path(str(binding.get("canonical_lock_path", ""))).resolve()
    expected = str(binding.get("canonical_lock_file_sha256", "")).upper()
    if not lock_path.is_file() or len(expected) != 64:
        raise RunQueueError("canonical lock binding is incomplete")
    actual = sha256_file(lock_path)
    if actual != expected:
        raise RunQueueError(f"canonical lock file mismatch: {actual} != {expected}")
    return lock_path, expected, binding


def _stratified_positions(length: int, count: int) -> list[int]:
    if count <= 0 or length <= 0:
        return []
    count = min(count, length)
    return [(index * length) // count for index in range(count)]


def _monitor_rows(
    monitor_source: str | Path,
    treatment: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_path = Path(monitor_source).resolve()
    source = _read(source_path, _MONITOR_SOURCE_COLUMNS, "OOF monitor source")
    if source.sample_id.astype(str).eq("").any() or source.sample_id.astype(str).duplicated().any():
        raise RunQueueError("OOF monitor source sample_id values must be non-empty and unique")
    labels = pd.to_numeric(source.y_true, errors="raise").astype(int)
    if set(labels) != {0, 1}:
        raise RunQueueError("OOF monitor source must contain both normal and defect samples")
    source = source.assign(y_true=labels)
    source_index = source.set_index(source.sample_id.astype(str), drop=False)
    missing_treatment = sorted(set(treatment.sample_id.astype(str)) - set(source_index.index))
    if missing_treatment:
        raise RunQueueError(
            f"OOF monitor source is missing treatment samples: {missing_treatment[:3]}"
        )
    normal_population = int((labels == 0).sum())
    defect_population = int((labels == 1).sum())
    normal_core_count = max(1, round(normal_population * 0.005))
    defect_core_count = max(1, round(defect_population * 0.005))
    defect_five_percent = max(defect_core_count + 1, round(defect_population * 0.05))

    treatment = treatment.sort_values("selection_rank", kind="stable")
    normal_core = treatment.head(normal_core_count)
    normal_remainder_population = treatment.iloc[normal_core_count:]
    normal_remainder = normal_remainder_population.iloc[
        _stratified_positions(len(normal_remainder_population), normal_core_count)
    ]

    defects = source.loc[source.y_true == 1].copy()
    defects["_mean"] = pd.to_numeric(defects.mean_p_defect, errors="raise")
    defects = defects.sort_values(["_mean", "sample_id"], kind="stable")
    defect_core = defects.head(defect_core_count)
    defect_buffer_population = defects.iloc[defect_core_count:defect_five_percent]
    defect_buffer = defect_buffer_population.iloc[
        _stratified_positions(len(defect_buffer_population), defect_core_count)
    ]

    panels = (
        (
            normal_core.sample_id.astype(str).tolist(),
            "NORMAL_REPLAY_CORE_0P5_CLASS_PERCENT",
            len(normal_core),
            "top nested Treatment ranks covering 0.5 percent of the normal class",
        ),
        (
            normal_remainder.sample_id.astype(str).tolist(),
            "NORMAL_REPLAY_REMAINDER_STRATIFIED",
            len(normal_remainder_population),
            "equal-rank strata over the remaining frozen Treatment pool",
        ),
        (
            defect_core.sample_id.astype(str).tolist(),
            "WEAK_DEFECT_CORE_0P5_CLASS_PERCENT",
            len(defect_core),
            "lowest OOF mean_p_defect covering 0.5 percent of the defect class",
        ),
        (
            defect_buffer.sample_id.astype(str).tolist(),
            "WEAK_DEFECT_BUFFER_0P5_TO_5_CLASS_PERCENT_STRATIFIED",
            len(defect_buffer_population),
            "equal-rank strata from the 0.5 to 5 percent weak-defect tail",
        ),
    )
    rows: list[dict[str, Any]] = []
    for sample_ids, group, group_population_size, rule in panels:
        for group_rank, sample_id in enumerate(sample_ids, start=1):
            row = source_index.loc[sample_id]
            rows.append(
                {
                    "monitor_group": group,
                    "monitor_group_rank": group_rank,
                    "sample_id": sample_id,
                    "y_true": int(row.y_true),
                    "oof_fold": str(row.oof_fold),
                    "dynamic_bucket": str(row.dynamic_bucket),
                    "oof_mean_p_defect": float(row.mean_p_defect),
                    "oof_correct_rate": float(row.correct_rate),
                    "oof_std_p_defect": float(row.std_p_defect),
                    "group_population_size": group_population_size,
                    "sampling_rule": rule,
                }
            )
    monitor = pd.DataFrame(rows)
    if monitor.sample_id.astype(str).duplicated().any():
        raise RunQueueError("causal monitor groups overlap")
    monitor.insert(0, "monitor_rank", range(1, len(monitor) + 1))
    definition = {
        "schema_version": "stage1.dynamic_causal_monitor.v2",
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "normal_population": normal_population,
        "defect_population": defect_population,
        "monitor_sample_count": len(monitor),
        "group_counts": {
            str(group): int(count)
            for group, count in monitor.groupby("monitor_group").size().items()
        },
        "test_data_used": False,
    }
    return monitor, definition


def _template(
    source: pd.DataFrame,
    *,
    selection_id: str,
    normal_slots: int,
    defect_slots: int,
) -> pd.DataFrame:
    normal = source.loc[source.replay_role.astype(str) == "normal_replay"].sort_values(
        ["role_rank", "selection_rank"], kind="stable"
    )
    defect = source.loc[source.replay_role.astype(str) == "defect_guard"].sort_values(
        ["role_rank", "selection_rank"], kind="stable"
    )
    if normal_slots > len(normal) or defect_slots > len(defect):
        raise RunQueueError(
            f"selection {selection_id} cannot satisfy N={normal_slots}, D={defect_slots}"
        )
    selected = pd.concat([normal.head(normal_slots), defect.head(defect_slots)], ignore_index=True)
    selected = selected.copy()
    selected["selection_rank"] = range(1, len(selected) + 1)
    columns = ["selection_rank", "role_rank", "sample_id", "y_true", "replay_role"]
    if "selection_id" in selected:
        selected["selection_id"] = selection_id
        columns.insert(0, "selection_id")
    return selected[columns]


def _topological_order(graph: pd.DataFrame) -> list[str]:
    job_ids = graph.job_id.astype(str).tolist()
    if len(set(job_ids)) != len(job_ids):
        raise RunQueueError("physical job graph contains duplicate job_id")
    dependencies = {
        str(row.job_id): str(row.dependency_job_id) for row in graph.itertuples(index=False)
    }
    unknown = sorted({value for value in dependencies.values() if value and value not in dependencies})
    if unknown:
        raise RunQueueError(f"physical job graph has unknown dependencies: {unknown}")
    ordered: list[str] = []
    remaining = set(job_ids)
    while remaining:
        ready = sorted(job for job in remaining if not dependencies[job] or dependencies[job] in ordered)
        if not ready:
            raise RunQueueError("physical job graph contains a dependency cycle")
        ordered.extend(ready)
        remaining.difference_update(ready)
    return ordered


def _queue_status(release_state: str, dependency: str) -> str:
    if release_state == "ENGINEERING_GATE":
        return "BLOCKED_ENGINEERING_GATE" if dependency else "HELD_ENGINEERING_GATE"
    if release_state == "HELD":
        return "HELD_SCIENTIFIC_GATE"
    raise RunQueueError(f"unknown preregistered release state: {release_state}")


def _readme() -> str:
    return """# Dynamic replay physical run queue

This queue contains all frozen Cycle 1/2 physical jobs, compact selection templates,
the fixed OOF-only causal monitor panel, and a byte-identical canonical training lock.

No job is READY at build time. Cycle 1 is held at the engineering gate; Cycle 2 is held
at the registered Cycle 1 scientific decision. Cycle 3/4 do not have physical jobs yet.
Absolute replay rows are derived from percentage schedules and are implementation fields.
"""


def build_campaign_run_queue(
    preregistration_dir: str | Path,
    output_dir: str | Path,
    *,
    monitor_source: str | Path,
) -> CampaignRunQueueResult:
    """Compile the preregistration into compact, gated, machine-local queues."""

    prereg = Path(preregistration_dir).resolve()
    output = Path(output_dir).resolve()
    if not prereg.is_dir():
        raise FileNotFoundError(prereg)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"run queue output is not empty: {output}")
    graph = _read(prereg / "PHYSICAL_JOB_GRAPH.csv", _GRAPH_COLUMNS, "physical job graph")
    matrix = _read(prereg / "EXPERIMENT_MATRIX.csv", _MATRIX_COLUMNS, "experiment matrix")
    _, pool_digests, selections = _binding_maps(prereg)
    lock_path, lock_sha, lock_binding = _canonical_lock_binding(prereg)
    for name, frame in (("physical graph", graph), ("experiment matrix", matrix)):
        observed = set(frame.canonical_lock_file_sha256.astype(str).str.upper())
        if observed != {lock_sha}:
            raise RunQueueError(f"{name} canonical lock drift: {sorted(observed)}")
    order = _topological_order(graph)
    order_index = {job_id: index for index, job_id in enumerate(order)}
    graph = graph.assign(_order=graph.job_id.astype(str).map(order_index)).sort_values(
        "_order", kind="stable"
    )
    machine_by_job = {
        str(row.job_id): str(row.machine_id) for row in graph.itertuples(index=False)
    }
    treatment_ids = [
        value
        for value, frame in selections.items()
        if len(frame) and set(pd.to_numeric(frame.y_true, errors="raise").astype(int)) == {0}
    ]
    if len(treatment_ids) != 1:
        raise RunQueueError("queue requires one unique frozen normal Treatment pool")
    treatment_id = treatment_ids[0]
    monitor_frame, monitor_definition = _monitor_rows(
        monitor_source,
        selections[treatment_id],
    )

    staging = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmpdir"
    try:
        templates_dir = staging / "templates"
        templates_dir.mkdir(parents=True)
        lock_copy = staging / "contracts/CANONICAL_TRAINING_LOCK_v1.json"
        atomic_write_bytes(lock_copy, lock_path.read_bytes())
        if sha256_file(lock_copy) != lock_sha:
            raise RunQueueError("copied canonical lock checksum mismatch")
        monitor = staging / "monitor/CAUSAL_MONITOR_SAMPLES.csv"
        atomic_write_bytes(
            monitor,
            monitor_frame.to_csv(index=False, lineterminator="\n").encode("utf-8"),
        )
        monitor_definition["monitor_manifest_sha256"] = sha256_file(monitor)
        atomic_write_json(staging / "monitor/MONITOR_DEFINITION.json", monitor_definition)
        monitor_sha = sha256_file(monitor)

        template_registry: dict[tuple[str, int, int], dict[str, Any]] = {}
        rows: list[dict[str, Any]] = []
        output_by_job: dict[str, str] = {}
        matrix_index = {
            (str(row.seed_id), str(row.arm_id)): row for row in matrix.itertuples(index=False)
        }
        if len(matrix_index) != len(matrix):
            raise RunQueueError("experiment matrix has duplicate seed/arm rows")

        for graph_row in graph.itertuples(index=False):
            job_id = str(graph_row.job_id)
            arm = str(graph_row.logical_arm_id)
            seed_id = str(graph_row.seed_id)
            matrix_row = matrix_index.get((seed_id, arm))
            shared_prefix = str(getattr(graph_row, "job_kind", "")) == "SHARED_TREATMENT_PREFIX"
            if shared_prefix:
                logical_run_id = f"{graph_row.cycle_id}_{seed_id}_{arm}"
                selection_id = str(graph_row.selection_id)
                schedule_id = str(graph_row.schedule_id)
                training_seed = int(graph_row.training_seed)
            else:
                if matrix_row is None:
                    raise RunQueueError(f"job has no logical matrix row: {job_id}")
                logical_run_id = str(matrix_row.logical_run_id)
                selection_id = str(matrix_row.selection_id)
                schedule_id = str(matrix_row.schedule_id)
                training_seed = int(matrix_row.training_seed)
                if training_seed != int(graph_row.training_seed):
                    raise RunQueueError(f"training seed mismatch for job: {job_id}")
                if str(matrix_row.selection_digest).upper() != pool_digests[selection_id]:
                    raise RunQueueError(f"selection pool digest mismatch for job: {job_id}")
                if selection_id != str(graph_row.selection_id):
                    raise RunQueueError(f"graph/matrix selection mismatch for job: {job_id}")

            normal_slots = int(graph_row.normal_replay_slots)
            defect_slots = int(graph_row.defect_guard_slots)
            if normal_slots + defect_slots != int(graph_row.total_replay_slots):
                raise RunQueueError(f"replay slot arithmetic mismatch for job: {job_id}")
            if selection_id not in selections:
                raise RunQueueError(f"job references unknown selection: {selection_id}")
            key = (selection_id, normal_slots, defect_slots)
            if key not in template_registry:
                template_id = f"TPL_{selection_id}_N{normal_slots:05d}_D{defect_slots:05d}"
                frame = _template(
                    selections[selection_id],
                    selection_id=selection_id,
                    normal_slots=normal_slots,
                    defect_slots=defect_slots,
                )
                path = templates_dir / f"{template_id}.csv"
                atomic_write_bytes(
                    path,
                    frame.to_csv(index=False, lineterminator="\n").encode("utf-8"),
                )
                template_registry[key] = {
                    "template_id": template_id,
                    "path": path,
                    "row_count": len(frame),
                    "active_selection_digest": _selection_digest(frame),
                    "file_sha256": sha256_file(path),
                }
            template = template_registry[key]
            dependency = str(graph_row.dependency_job_id)
            if dependency and machine_by_job[dependency] != str(graph_row.machine_id):
                raise RunQueueError(
                    "dependency crosses machines and cannot use local exact checkpoints: "
                    f"{dependency} -> {job_id}"
                )
            run_output = f"../05_training_runs/{logical_run_id}"
            machine_output = (
                f"dynamic_replay_budget_efficiency_20260807/05_training_runs/{logical_run_id}"
            )
            output_by_job[job_id] = run_output
            dependency_output = output_by_job.get(dependency, "") if dependency else ""
            if dependency and not dependency_output:
                raise RunQueueError(f"dependency was not ordered before child: {job_id}")
            release_state = str(graph_row.release_state)
            rows.append(
                {
                    "queue_order": order_index[job_id] + 1,
                    "job_id": job_id,
                    "cycle_id": str(graph_row.cycle_id),
                    "release_state": release_state,
                    "queue_status": _queue_status(release_state, dependency),
                    "machine_id": str(graph_row.machine_id),
                    "seed_id": seed_id,
                    "training_seed": training_seed,
                    "logical_run_id": logical_run_id,
                    "logical_arm_id": arm,
                    "schedule_id": schedule_id,
                    "job_kind": str(getattr(graph_row, "job_kind", "ARM_SEGMENT")),
                    "segment_index": int(graph_row.segment_index),
                    "segment_start_epoch": int(graph_row.segment_start_epoch),
                    "segment_end_epoch": int(graph_row.segment_end_epoch),
                    "normal_replay_slots": normal_slots,
                    "defect_guard_slots": defect_slots,
                    "total_replay_slots": int(graph_row.total_replay_slots),
                    "expected_steps_batch128": int(graph_row.expected_steps_batch128),
                    "selection_pool_id": selection_id,
                    "selection_pool_digest": pool_digests[selection_id],
                    "active_selection_template_id": template["template_id"],
                    "active_selection_template_relpath": template["path"].relative_to(
                        staging
                    ).as_posix(),
                    "active_selection_template_sha256": template["file_sha256"],
                    "active_selection_digest": template["active_selection_digest"],
                    "active_selection_rows": template["row_count"],
                    "monitor_manifest_relpath": monitor.relative_to(staging).as_posix(),
                    "monitor_manifest_sha256": monitor_sha,
                    "canonical_lock_relpath": lock_copy.relative_to(staging).as_posix(),
                    "canonical_lock_file_sha256": lock_sha,
                    "dependency_job_id": dependency,
                    "dependency_output_relpath": dependency_output,
                    "resume_from_epoch": int(graph_row.resume_from_epoch),
                    "branch_checkpoint_required": _bool(graph_row.branch_checkpoint_required),
                    "run_output_relpath": run_output,
                    "machine_output_relpath": machine_output,
                    "checkpoint_epochs_to_retain": str(graph_row.checkpoint_epochs_to_retain),
                    "planned_epoch_equivalents": float(
                        getattr(graph_row, "planned_epoch_equivalents", 0.0)
                    ),
                }
            )

        registry = pd.DataFrame(rows)
        registry_path = staging / "JOB_EXECUTION_REGISTRY.csv"
        atomic_write_bytes(
            registry_path,
            registry.to_csv(index=False, lineterminator="\n").encode("utf-8"),
        )
        machine_dir = staging / "machines"
        for machine_id, frame in registry.groupby("machine_id", sort=True):
            atomic_write_bytes(
                machine_dir / f"{machine_id}.csv",
                frame.sort_values("queue_order", kind="stable").to_csv(
                    index=False, lineterminator="\n"
                ).encode("utf-8"),
            )
        template_rows = [
            {
                "selection_pool_id": key[0],
                "normal_replay_slots": key[1],
                "defect_guard_slots": key[2],
                "template_id": value["template_id"],
                "template_relpath": value["path"].relative_to(staging).as_posix(),
                "row_count": value["row_count"],
                "active_selection_digest": value["active_selection_digest"],
                "file_sha256": value["file_sha256"],
            }
            for key, value in sorted(template_registry.items())
        ]
        atomic_write_bytes(
            staging / "TEMPLATE_REGISTRY.csv",
            pd.DataFrame(template_rows).to_csv(index=False, lineterminator="\n").encode(
                "utf-8"
            ),
        )
        validation = {
            "schema_version": "stage1.dynamic_campaign_run_queue.v2",
            "status": "PASS",
            "job_count": len(registry),
            "cycle_job_counts": {
                str(cycle): int(count)
                for cycle, count in registry.groupby("cycle_id").size().items()
            },
            "template_count": len(template_registry),
            "machine_count": int(registry.machine_id.nunique()),
            "dependency_edge_count": int(registry.dependency_job_id.astype(str).ne("").sum()),
            "ready_job_count": int(registry.queue_status.eq("READY").sum()),
            "engineering_gate_job_count": int(
                registry.release_state.eq("ENGINEERING_GATE").sum()
            ),
            "scientific_hold_job_count": int(registry.release_state.eq("HELD").sum()),
            "monitor_sample_count": len(monitor_frame),
            "monitor_manifest_sha256": monitor_sha,
            "monitor_source_sha256": monitor_definition["source_sha256"],
            "canonical_lock_file_sha256": lock_sha,
            "canonical_lock_payload_sha256": lock_binding[
                "canonical_lock_payload_sha256"
            ],
            "canonical_lock_relpath": lock_copy.relative_to(staging).as_posix(),
            "job_registry_sha256": sha256_file(registry_path),
            "preregistration_inputs": {
                name: sha256_file(prereg / name)
                for name in (
                    "PHYSICAL_JOB_GRAPH.csv",
                    "EXPERIMENT_MATRIX.csv",
                    "SELECTION_BINDINGS.csv",
                    "CANONICAL_TRAINING_LOCK_BINDING.json",
                )
            },
        }
        atomic_write_json(staging / "RUN_QUEUE_VALIDATION.json", validation)
        atomic_write_bytes(staging / "README.md", _readme().encode("utf-8"))
        if output.exists():
            output.rmdir()
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return CampaignRunQueueResult(
        queue_dir=output,
        job_registry=output / "JOB_EXECUTION_REGISTRY.csv",
        monitor_manifest=output / "monitor/CAUSAL_MONITOR_SAMPLES.csv",
        validation_path=output / "RUN_QUEUE_VALIDATION.json",
    )


def build_replay_identity_manifest(
    selection_template: str | Path,
    base_normal_manifest: str | Path,
    base_defect_manifest: str | Path,
    *,
    run_slot: str,
    output_path: str | Path,
) -> ReplayIdentityResult:
    """Resolve one compact template to exact runtime hardlink names and sources."""

    if not _SAFE_RUN_SLOT.fullmatch(run_slot):
        raise RunQueueError(f"unsafe run_slot for staged filenames: {run_slot!r}")
    template = _read(Path(selection_template).resolve(), _SELECTION_COLUMNS, "selection template")
    normal = _read(
        Path(base_normal_manifest).resolve(),
        {"canonical_image_relpath", "Filename"},
        "base normal manifest",
    )
    defect = _read(
        Path(base_defect_manifest).resolve(),
        {"canonical_image_relpath", "Filename"},
        "base defect manifest",
    )
    sources: dict[tuple[int, str], tuple[str, str]] = {}
    for frame, y_true in ((normal, 0), (defect, 1)):
        for row in frame.itertuples(index=False):
            key = (y_true, str(row.canonical_image_relpath))
            if key in sources:
                raise RunQueueError(f"duplicate base sample identity: {key}")
            sources[key] = (str(row.canonical_image_relpath), str(row.Filename))
    ranks = pd.to_numeric(template.selection_rank, errors="raise").astype(int)
    if ranks.tolist() != list(range(1, len(template) + 1)):
        raise RunQueueError("selection template ranks must be contiguous")
    rows: list[dict[str, Any]] = []
    for row in template.itertuples(index=False):
        y_true = int(row.y_true)
        sample_id = str(row.sample_id)
        role = str(row.replay_role)
        expected_role = "normal_replay" if y_true == 0 else "defect_guard"
        if role != expected_role:
            raise RunQueueError(f"selection template role mismatch for {sample_id}")
        source = sources.get((y_true, sample_id))
        if source is None:
            raise RunQueueError(f"selection template sample absent from base manifests: {sample_id}")
        source_relpath, filename = source
        rank = int(row.selection_rank)
        rows.append(
            {
                "selection_rank": rank,
                "role_rank": int(row.role_rank),
                "sample_id": sample_id,
                "y_true": y_true,
                "replay_role": role,
                "source_canonical_image_relpath": source_relpath,
                "source_filename": filename,
                "staged_filename": (
                    f"replay__{run_slot}__{rank:05d}__{Path(filename).name}"
                ),
            }
        )
    output = Path(output_path).resolve()
    atomic_write_bytes(
        output,
        pd.DataFrame(rows).to_csv(index=False, lineterminator="\n").encode("utf-8"),
    )
    return ReplayIdentityResult(output, len(rows), sha256_file(output))


__all__ = [
    "CampaignRunQueueResult",
    "ReplayIdentityResult",
    "RunQueueError",
    "build_campaign_run_queue",
    "build_replay_identity_manifest",
]
