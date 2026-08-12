"""Read-only machine, GPU, runtime, resume, and snapshot audit for 240 runs.

The canonical inventory chooses the attempt and the source-file ledger chooses
the files.  No directory discovery or "latest attempt" heuristic is allowed in
this module, which prevents historical and failed attempts from entering the
resource analysis.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


class ResourceAuditError(RuntimeError):
    """Raised when canonical resource evidence is missing or ambiguous."""


@dataclass(frozen=True)
class ResourceReliabilityResult:
    runs: pd.DataFrame
    machines: pd.DataFrame
    triads: pd.DataFrame
    summary: dict[str, Any]


_INVENTORY_COLUMNS = {
    "run_slot",
    "triad_id",
    "phase",
    "condition_id",
    "method",
    "budget",
    "arm",
    "training_seed",
    "package",
    "machine_id",
    "attempt_id",
    "resume_count",
    "input_snapshot_id",
}

_LEDGER_COLUMNS = {
    "run_slot",
    "attempt_id",
    "attempt_dir",
    "relative_path",
    "canonical_attempt",
}

_SINGLE_REQUIRED_PATHS = (
    "00_identity/environment_controller.json",
    "00_identity/environment_training.json",
    "00_identity/run_identity.json",
    "02_logs/epoch_training_metrics.csv",
    "02_logs/training_execution_audit.json",
    "07_validation/preflight_report.json",
    "07_validation/storage_preflight.json",
    "08_status/status.json",
)


def _truthy(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact decoder text is unstable
        raise ResourceAuditError(f"cannot read JSON evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResourceAuditError(f"JSON evidence is not an object: {path}")
    return value


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ResourceAuditError(f"{label} missing columns: {missing}")


def _segment_clock_seconds(times: pd.Series) -> tuple[float, int]:
    values = pd.to_numeric(times, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return float("nan"), 0
    boundaries = np.r_[0, np.flatnonzero(np.diff(values) < 0) + 1, values.size]
    total = 0.0
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        segment = values[start:stop]
        if segment.size:
            total += float(np.nanmax(segment))
    return total, len(boundaries) - 1


def _safe_quantile(values: pd.Series, quantile: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.quantile(quantile)) if not numeric.empty else float("nan")


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else float("nan")


def _safe_max(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.max()) if not numeric.empty else float("nan")


def _mode_or_nan(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.mode().iloc[0])


def _relative_rows(ledger: pd.DataFrame) -> dict[str, list[pd.Series]]:
    result: dict[str, list[pd.Series]] = {}
    for _, row in ledger.iterrows():
        relative = str(row["relative_path"]).replace("\\", "/")
        result.setdefault(relative, []).append(row)
    return result


def _one_path(mapping: dict[str, list[pd.Series]], relative: str) -> Path:
    rows = mapping.get(relative, [])
    if len(rows) != 1:
        raise ResourceAuditError(
            f"required evidence {relative!r} has {len(rows)} canonical rows"
        )
    path = Path(str(rows[0]["attempt_dir"])) / relative
    if not path.is_file():
        raise ResourceAuditError(f"ledger evidence does not exist: {path}")
    return path


def _glob_rows(
    mapping: dict[str, list[pd.Series]],
    *,
    prefix: str,
    suffix: str,
) -> list[Path]:
    paths = []
    for relative, rows in mapping.items():
        if relative.startswith(prefix) and relative.endswith(suffix):
            if len(rows) != 1:
                raise ResourceAuditError(
                    f"duplicate canonical ledger path {relative}: {len(rows)}"
                )
            path = Path(str(rows[0]["attempt_dir"])) / relative
            if not path.is_file():
                raise ResourceAuditError(f"ledger evidence does not exist: {path}")
            paths.append(path)
    return sorted(paths)


def _summarize_gpu(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ResourceAuditError("canonical run has no gpu_usage log")
    frames = []
    monitor_seconds = 0.0
    required = [
        "timestamp_unix",
        "gpu_util_pct",
        "memory_used_mb",
        "memory_total_mb",
        "temperature_c",
        "power_w",
        "status",
    ]
    for path in paths:
        frame = pd.read_csv(path, usecols=required)
        if frame.empty:
            raise ResourceAuditError(f"empty GPU log: {path}")
        timestamps = pd.to_numeric(frame["timestamp_unix"], errors="coerce").dropna()
        if timestamps.empty:
            raise ResourceAuditError(f"GPU log has no timestamps: {path}")
        monitor_seconds += max(0.0, float(timestamps.max() - timestamps.min()))
        frames.append(frame)
    gpu = pd.concat(frames, ignore_index=True)
    status = gpu["status"].astype(str).str.upper()
    status_counts = {
        str(key): int(value)
        for key, value in status.value_counts(dropna=False).sort_index().items()
    }
    util = pd.to_numeric(gpu["gpu_util_pct"], errors="coerce")
    active = util[util > 0]
    return {
        "gpu_log_count": len(paths),
        "gpu_sample_count": len(gpu),
        "gpu_monitor_seconds": monitor_seconds,
        "gpu_util_mean_pct": _safe_mean(util),
        "gpu_util_active_mean_pct": _safe_mean(active),
        "gpu_util_p50_pct": _safe_quantile(util, 0.5),
        "gpu_util_p95_pct": _safe_quantile(util, 0.95),
        "gpu_zero_util_fraction": float((util.fillna(0) <= 0).mean()),
        "gpu_memory_mean_mb": _safe_mean(gpu["memory_used_mb"]),
        "gpu_memory_peak_mb": _safe_max(gpu["memory_used_mb"]),
        "gpu_memory_total_mb": _mode_or_nan(gpu["memory_total_mb"]),
        "gpu_temperature_mean_c": _safe_mean(gpu["temperature_c"]),
        "gpu_temperature_max_c": _safe_max(gpu["temperature_c"]),
        "gpu_power_mean_w": _safe_mean(gpu["power_w"]),
        "gpu_power_max_w": _safe_max(gpu["power_w"]),
        "gpu_non_ok_samples": int((status != "OK").sum()),
        "gpu_status_counts_json": json.dumps(
            status_counts, ensure_ascii=False, sort_keys=True
        ),
    }


def _summarize_train_results(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ResourceAuditError("canonical run has no completed train result JSON")
    records = [_load_json(path) for path in paths]
    durations = [float(item.get("duration_seconds", 0.0) or 0.0) for item in records]
    return {
        "train_result_count": len(records),
        "train_result_duration_seconds": float(sum(durations)),
        "train_result_all_pass": all(
            item.get("status") == "PASS"
            and int(item.get("returncode", -1)) == 0
            and not bool(item.get("timed_out", False))
            for item in records
        ),
    }


def _audit_one_run(
    inventory_row: dict[str, Any],
    ledger_rows: pd.DataFrame,
) -> dict[str, Any]:
    run_slot = str(inventory_row["run_slot"])
    if not ledger_rows["canonical_attempt"].map(_truthy).all():
        raise ResourceAuditError(f"non-canonical ledger row selected for {run_slot}")
    expected_attempt = str(inventory_row["attempt_id"])
    actual_attempts = set(ledger_rows["attempt_id"].astype(str))
    if actual_attempts != {expected_attempt}:
        raise ResourceAuditError(
            f"attempt mismatch for {run_slot}: expected {expected_attempt}, got {actual_attempts}"
        )

    mapping = _relative_rows(ledger_rows)
    paths = {relative: _one_path(mapping, relative) for relative in _SINGLE_REQUIRED_PATHS}
    controller = _load_json(paths["00_identity/environment_controller.json"])
    environment = _load_json(paths["00_identity/environment_training.json"])
    identity = _load_json(paths["00_identity/run_identity.json"])
    execution = _load_json(paths["02_logs/training_execution_audit.json"])
    preflight = _load_json(paths["07_validation/preflight_report.json"])
    storage = _load_json(paths["07_validation/storage_preflight.json"])
    status = _load_json(paths["08_status/status.json"])
    epochs = pd.read_csv(
        paths["02_logs/epoch_training_metrics.csv"], usecols=["epoch", "time"]
    )
    epoch_clock_seconds, epoch_clock_segments = _segment_clock_seconds(epochs["time"])

    gpu_paths = _glob_rows(
        mapping, prefix="02_logs/gpu_usage_", suffix=".csv"
    )
    train_result_paths = _glob_rows(
        mapping, prefix="02_logs/train_", suffix=".log.result.json"
    )
    gpu = _summarize_gpu(gpu_paths)
    train_results = _summarize_train_results(train_result_paths)

    env_actual = environment.get("actual", {})
    env_checks = environment.get("checks", {})
    checks_ok = bool(env_checks) and all(
        bool(value.get("ok")) for value in env_checks.values() if isinstance(value, dict)
    )
    inventory_resume = int(inventory_row["resume_count"])
    identity_resume = int(identity.get("resume_count", -1))
    execution_resume = int(execution.get("resume_count", -1))
    segment_count = len(identity.get("resume_segments", []))
    resume_segments = identity.get("resume_segments", [])
    segment_statuses = [str(item.get("status", "UNKNOWN")) for item in resume_segments]
    resume_interrupted_segments = sum(
        status_value == "INTERRUPTED" for status_value in segment_statuses
    )
    resume_completed_segments = sum(
        status_value == "COMPLETED" for status_value in segment_statuses
    )
    resume_missing_end_timestamp_segments = sum(
        item.get("ended_at") is None for item in resume_segments
    )
    resume_resumed_segments = sum(bool(item.get("resumed")) for item in resume_segments)
    identity_attempt = str(identity.get("attempt_id", ""))
    expected_identity_attempt = expected_attempt.removeprefix("attempt_")
    identity_consistent = all(
        (
            str(identity.get("run_slot")) == run_slot,
            identity_attempt in {expected_attempt, expected_identity_attempt},
            str(identity.get("machine_id")) == str(inventory_row["machine_id"]),
            str(identity.get("input_snapshot_id"))
            == str(inventory_row["input_snapshot_id"]),
            inventory_resume == identity_resume == execution_resume,
            segment_count == inventory_resume + 1,
        )
    )
    completed_epochs = int(execution.get("completed_epochs", -1))
    expected_epochs = int(execution.get("expected_epochs", -1))
    optimizer_steps = int(execution.get("optimizer_steps_total", -1))
    epoch_samples = int(
        preflight.get("replay_manifest_summary", {}).get("epoch_samples", 0)
    )
    monitor_seconds = float(gpu["gpu_monitor_seconds"])
    artifact_mismatches = 0
    if "artifact_manifest_size_match" in ledger_rows:
        matches = ledger_rows["artifact_manifest_size_match"]
        observed = matches[matches.notna()]
        artifact_mismatches = int((~observed.map(_truthy)).sum())

    validated_complete = all(
        (
            status.get("state") == "VALIDATED",
            status.get("phase") == "complete",
            int(status.get("last_epoch", -1)) == 200,
            completed_epochs == expected_epochs == 200,
            len(epochs) == 200,
            bool(execution.get("loss_finite")),
            environment.get("status") == "PASS",
            checks_ok,
            storage.get("status") == "PASS",
            preflight.get("status") == "PASS",
            not preflight.get("issues"),
            bool(train_results["train_result_all_pass"]),
            artifact_mismatches == 0,
        )
    )

    result = dict(inventory_row)
    result.update(
        {
            "controller_platform": controller.get("platform"),
            "controller_python": controller.get("python"),
            "training_python": env_actual.get("python"),
            "pytorch": env_actual.get("pytorch"),
            "cuda_build": env_actual.get("cuda_build"),
            "ultralytics": env_actual.get("ultralytics"),
            "numpy": env_actual.get("numpy"),
            "pandas": env_actual.get("pandas"),
            "polars": env_actual.get("polars"),
            "scikit_learn": env_actual.get("scikit_learn"),
            "environment_checks_ok": checks_ok,
            "resume_count_identity": identity_resume,
            "resume_count_execution": execution_resume,
            "resume_mode": execution.get("resume_mode"),
            "resume_segment_count": segment_count,
            "resume_interrupted_segments": resume_interrupted_segments,
            "resume_completed_segments": resume_completed_segments,
            "resume_missing_end_timestamp_segments": (
                resume_missing_end_timestamp_segments
            ),
            "resume_resumed_segments": resume_resumed_segments,
            "resume_segment_statuses": ";".join(segment_statuses),
            "resumed": inventory_resume > 0,
            "completed_epochs": completed_epochs,
            "expected_epochs": expected_epochs,
            "expected_steps_per_epoch": execution.get("expected_steps_per_epoch"),
            "optimizer_steps_total": optimizer_steps,
            "effective_batch_size": execution.get("effective_batch_size"),
            "epoch_samples": epoch_samples,
            "epoch_clock_seconds": epoch_clock_seconds,
            "epoch_clock_segment_count": epoch_clock_segments,
            "training_samples_per_second": (
                epoch_samples * completed_epochs / monitor_seconds
                if monitor_seconds > 0
                else float("nan")
            ),
            "optimizer_steps_per_second": (
                optimizer_steps / monitor_seconds
                if monitor_seconds > 0
                else float("nan")
            ),
            "dataset_volume": storage.get("dataset_volume"),
            "staging_volume": storage.get("staging_volume"),
            "expected_staging_files": storage.get("expected_staging_files"),
            "maximum_staging_files": storage.get("maximum_staging_files"),
            "output_free_gib": float(storage.get("output_free_bytes", 0)) / 2**30,
            "staging_free_gib": float(storage.get("staging_free_bytes", 0)) / 2**30,
            "output_free_margin_gib": (
                float(storage.get("output_free_bytes", 0))
                - float(storage.get("minimum_output_free_bytes", 0))
            )
            / 2**30,
            "staging_free_margin_gib": (
                float(storage.get("staging_free_bytes", 0))
                - float(storage.get("minimum_staging_free_bytes", 0))
            )
            / 2**30,
            "artifact_size_mismatch_count": artifact_mismatches,
            "manual_recovery_file_count": int(
                ledger_rows["relative_path"]
                .astype(str)
                .str.contains("manual_recovery|pre_repair|sac_repair", regex=True)
                .sum()
            ),
            "identity_consistent": identity_consistent,
            "validated_complete": validated_complete,
            **gpu,
            **train_results,
        }
    )
    return result


def _summarize_machines(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for machine, group in runs.groupby("machine_id", sort=True):
        arm_counts = group["arm"].value_counts()
        rows.append(
            {
                "machine_id": machine,
                "run_count": len(group),
                "triad_count": group["triad_id"].nunique(),
                "t_runs": int(arm_counts.get("T", 0)),
                "r1_runs": int(arm_counts.get("R1", 0)),
                "r2_runs": int(arm_counts.get("R2", 0)),
                "resumed_runs": int(group["resumed"].sum()),
                "resume_events": int(group["resume_count"].sum()),
                "snapshot_count": group["input_snapshot_id"].nunique(),
                "snapshots": ";".join(sorted(group["input_snapshot_id"].astype(str).unique())),
                "gpu_total_hours": group["gpu_monitor_seconds"].sum() / 3600.0,
                "gpu_median_hours_per_run": group["gpu_monitor_seconds"].median()
                / 3600.0,
                "throughput_median_samples_per_second": group[
                    "training_samples_per_second"
                ].median(),
                "throughput_p10_samples_per_second": group[
                    "training_samples_per_second"
                ].quantile(0.1),
                "throughput_p90_samples_per_second": group[
                    "training_samples_per_second"
                ].quantile(0.9),
                "gpu_util_mean_pct": group["gpu_util_mean_pct"].mean(),
                "gpu_memory_total_mb": _mode_or_nan(group["gpu_memory_total_mb"]),
                "gpu_temperature_max_c": group["gpu_temperature_max_c"].max(),
                "gpu_non_ok_samples": int(group["gpu_non_ok_samples"].sum()),
                "all_identity_consistent": bool(group["identity_consistent"].all()),
                "all_validated_complete": bool(group["validated_complete"].all()),
            }
        )
    return pd.DataFrame(rows).sort_values("machine_id").reset_index(drop=True)


def _summarize_triads(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for triad_id, group in runs.groupby("triad_id", sort=True):
        if len(group) != 3 or set(group["arm"]) != {"T", "R1", "R2"}:
            raise ResourceAuditError(
                f"triad {triad_id} is not one complete T/R1/R2 unit"
            )
        by_arm = group.set_index("arm")
        machines = {arm: str(by_arm.loc[arm, "machine_id"]) for arm in ("T", "R1", "R2")}
        snapshots = {
            arm: str(by_arm.loc[arm, "input_snapshot_id"])
            for arm in ("T", "R1", "R2")
        }
        throughput = pd.to_numeric(
            group["training_samples_per_second"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "triad_id": triad_id,
                "phase": by_arm.loc["T", "phase"],
                "condition_id": by_arm.loc["T", "condition_id"],
                "method": by_arm.loc["T", "method"],
                "budget": by_arm.loc["T", "budget"],
                "training_seed": by_arm.loc["T", "training_seed"],
                "t_machine_id": machines["T"],
                "r1_machine_id": machines["R1"],
                "r2_machine_id": machines["R2"],
                "all_arms_same_machine": len(set(machines.values())) == 1,
                "t_r1_same_machine": machines["T"] == machines["R1"],
                "t_r2_same_machine": machines["T"] == machines["R2"],
                "all_arms_same_snapshot": len(set(snapshots.values())) == 1,
                "resumed_arm_count": int(group["resumed"].sum()),
                "resume_event_count": int(group["resume_count"].sum()),
                "gpu_total_hours": group["gpu_monitor_seconds"].sum() / 3600.0,
                "throughput_min_samples_per_second": (
                    float(throughput.min()) if not throughput.empty else float("nan")
                ),
                "throughput_max_samples_per_second": (
                    float(throughput.max()) if not throughput.empty else float("nan")
                ),
                "throughput_max_min_ratio": (
                    float(throughput.max() / throughput.min())
                    if not throughput.empty and throughput.min() > 0
                    else float("nan")
                ),
                "all_validated_complete": bool(group["validated_complete"].all()),
                "all_identity_consistent": bool(group["identity_consistent"].all()),
            }
        )
    return pd.DataFrame(rows).sort_values("triad_id").reset_index(drop=True)


def analyze_canonical_resources(
    inventory_path: str | Path,
    source_ledger_path: str | Path,
    *,
    expected_runs: int = 240,
    expected_triads: int = 80,
    max_workers: int = 8,
) -> ResourceReliabilityResult:
    """Analyze only canonical attempts and their registered source-ledger files."""

    inventory = pd.read_csv(inventory_path)
    ledger = pd.read_csv(source_ledger_path)
    _require_columns(inventory, _INVENTORY_COLUMNS, "canonical inventory")
    _require_columns(ledger, _LEDGER_COLUMNS, "source ledger")
    if len(inventory) != expected_runs or inventory["run_slot"].nunique() != expected_runs:
        raise ResourceAuditError(
            f"expected {expected_runs} unique canonical runs, got {len(inventory)} rows "
            f"and {inventory['run_slot'].nunique()} slots"
        )
    if inventory["triad_id"].nunique() != expected_triads:
        raise ResourceAuditError(
            f"expected {expected_triads} triads, got {inventory['triad_id'].nunique()}"
        )
    inventory_slots = set(inventory["run_slot"].astype(str))
    ledger_slots = set(ledger["run_slot"].astype(str))
    if ledger_slots != inventory_slots:
        raise ResourceAuditError(
            "source ledger slots do not exactly match canonical inventory: "
            f"missing={sorted(inventory_slots-ledger_slots)}, "
            f"extra={sorted(ledger_slots-inventory_slots)}"
        )

    grouped_ledger = {
        str(slot): group.copy()
        for slot, group in ledger.groupby("run_slot", sort=False)
    }
    items = [row._asdict() for row in inventory.itertuples(index=False)]

    def worker(row: dict[str, Any]) -> dict[str, Any]:
        return _audit_one_run(row, grouped_ledger[str(row["run_slot"])])

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        records = list(pool.map(worker, items))
    runs = pd.DataFrame(records).sort_values("run_slot").reset_index(drop=True)
    machines = _summarize_machines(runs)
    triads = _summarize_triads(runs)
    summary = {
        "canonical_runs": len(runs),
        "triads": len(triads),
        "machines": runs["machine_id"].nunique(),
        "resumed_runs": int(runs["resumed"].sum()),
        "resume_events": int(runs["resume_count"].sum()),
        "input_snapshots": runs["input_snapshot_id"].nunique(),
        "cross_machine_triads": int((~triads["all_arms_same_machine"]).sum()),
        "cross_snapshot_triads": int((~triads["all_arms_same_snapshot"]).sum()),
        "gpu_log_count": int(runs["gpu_log_count"].sum()),
        "gpu_sample_count": int(runs["gpu_sample_count"].sum()),
        "gpu_non_ok_samples": int(runs["gpu_non_ok_samples"].sum()),
        "gpu_total_hours": float(runs["gpu_monitor_seconds"].sum() / 3600.0),
        "train_result_total_hours": float(
            runs["train_result_duration_seconds"].sum() / 3600.0
        ),
        "identity_inconsistent_runs": int((~runs["identity_consistent"]).sum()),
        "not_validated_complete_runs": int((~runs["validated_complete"]).sum()),
    }
    return ResourceReliabilityResult(
        runs=runs,
        machines=machines,
        triads=triads,
        summary=summary,
    )


__all__ = [
    "ResourceAuditError",
    "ResourceReliabilityResult",
    "analyze_canonical_resources",
]
