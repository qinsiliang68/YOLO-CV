"""Fail-closed validation of 1..200 low-cost dynamic-training telemetry."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

import pandas as pd

from .errors import ValidationError
from .util import atomic_write_json, sha256_file


ALL_EPOCH_SCHEMA = "stage1.all_epoch_telemetry_validation.v1"
SCHEDULE_COLUMNS = {
    "epoch",
    "segment_id",
    "base_sample_exposures",
    "normal_replay_exposures",
    "defect_guard_exposures",
    "total_sample_exposures",
    "expected_optimizer_steps",
}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"unreadable {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{label} is not a JSON object: {path}")
    return value


def _validate_segments(segments: list[dict[str, Any]], expected_epochs: int) -> list[str]:
    issues: list[str] = []
    ordered = sorted(segments, key=lambda row: int(row.get("start_epoch", -1)))
    expected_start = 1
    seen: set[str] = set()
    for segment in ordered:
        identity = str(segment.get("segment_id", ""))
        start = int(segment.get("start_epoch", -1))
        end = int(segment.get("end_epoch", segment.get("planned_end_epoch", -1)) or -1)
        if not identity or identity in seen:
            issues.append(f"invalid or duplicate segment_id: {identity!r}")
        seen.add(identity)
        if start != expected_start or end < start:
            issues.append(
                f"resume segment boundary is not contiguous: {identity} {start}-{end}, expected start {expected_start}"
            )
        expected_start = end + 1
        if segment.get("status") != "COMPLETED":
            issues.append(f"segment is not COMPLETED: {identity}")
    if expected_start != expected_epochs + 1:
        issues.append(f"segment coverage ends at {expected_start - 1}, expected {expected_epochs}")
    return issues


def _role_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in summary.get("roles", []):
        if row.get("record_type") != "ROLE":
            continue
        role = str(row.get("exposure_role", ""))
        if not role or role in result:
            raise ValidationError(f"role loss summary has invalid duplicate role: {role!r}")
        result[role] = row
    return result


def validate_all_epoch_telemetry(
    audit_path: str | Path,
    schedule_path: str | Path,
    process_telemetry_dir: str | Path,
    *,
    output_path: str | Path,
    expected_epochs: int = 200,
    expected_arm_id: str | None = None,
) -> dict[str, Any]:
    """Validate exact epoch coverage, schedule dose, role losses and atomic sidecars."""

    if expected_epochs <= 0:
        raise ValidationError("expected_epochs must be positive")
    audit_file = Path(audit_path).resolve()
    schedule_file = Path(schedule_path).resolve()
    telemetry_root = Path(process_telemetry_dir).resolve()
    audit = _read_json(audit_file, "dynamic training audit")
    schedule = pd.read_csv(schedule_file, keep_default_na=False)
    missing = SCHEDULE_COLUMNS - set(schedule.columns)
    if missing:
        raise ValidationError(f"schedule missing columns: {sorted(missing)}")
    issues: list[str] = []
    epochs = pd.to_numeric(schedule.epoch, errors="raise").astype(int).tolist()
    if epochs != list(range(1, expected_epochs + 1)):
        issues.append("schedule epoch grid must be exactly 1..expected_epochs with no duplicates")
        report = {
            "schema_version": ALL_EPOCH_SCHEMA,
            "status": "FAIL",
            "created_at_unix": time.time(),
            "issues": issues,
            "expected_epochs": expected_epochs,
            "validated_epoch_count": 0,
            "identity": {
                "audit_sha256": sha256_file(audit_file),
                "schedule_sha256": sha256_file(schedule_file),
            },
        }
        atomic_write_json(output_path, report, overwrite=True)
        raise ValidationError(f"all-epoch telemetry validation failed; see {output_path}")
    if expected_arm_id is not None and str(audit.get("arm_id")) != str(expected_arm_id):
        issues.append("audit arm identity mismatch")
    if int(audit.get("total_epochs", -1)) != expected_epochs:
        issues.append("audit total_epochs mismatch")
    if int(audit.get("completed_epochs", -1)) != expected_epochs:
        issues.append("audit is not complete")
    records = audit.get("epoch_records", [])
    observed_epochs = [int(row.get("epoch", -1)) for row in records]
    if observed_epochs != list(range(1, expected_epochs + 1)):
        issues.append("audit epoch records must be unique and cover exactly 1..expected_epochs")
    issues.extend(_validate_segments(list(audit.get("segments", [])), expected_epochs))
    observed_steps = list(audit.get("observed_steps_by_epoch", []))
    expected_steps = list(audit.get("expected_steps_by_epoch", []))
    if len(observed_steps) != expected_epochs or len(expected_steps) != expected_epochs:
        issues.append("audit step vectors do not cover all epochs")

    sidecar_bytes = 0
    parquet_bytes = 0
    role_summary_bytes = 0
    cumulative_expected_replay = 0
    cumulative_observed_replay = 0
    role_loss_epochs = 0
    per_epoch: list[dict[str, Any]] = []
    no_replay = bool((pd.to_numeric(schedule.normal_replay_exposures) == 0).all()) and bool(
        (pd.to_numeric(schedule.defect_guard_exposures) == 0).all()
    )
    record_by_epoch = {int(row.get("epoch", -1)): row for row in records}
    schedule_by_epoch = schedule.set_index(pd.to_numeric(schedule.epoch).astype(int))
    for epoch in range(1, expected_epochs + 1):
        row = schedule_by_epoch.loc[epoch]
        base = int(row.base_sample_exposures)
        normal = int(row.normal_replay_exposures)
        guard = int(row.defect_guard_exposures)
        total = int(row.total_sample_exposures)
        steps = int(row.expected_optimizer_steps)
        if min(base, normal, guard, total, steps) < 0 or total != base + normal + guard:
            issues.append(f"schedule exposure arithmetic invalid at epoch {epoch}")
        cumulative_expected_replay += normal + guard
        audit_row = record_by_epoch.get(epoch, {})
        if int(audit_row.get("optimizer_steps_epoch", -1)) != steps:
            issues.append(f"optimizer steps differ from schedule at epoch {epoch}")
        if epoch <= len(observed_steps) and int(observed_steps[epoch - 1]) != steps:
            issues.append(f"observed step vector differs from schedule at epoch {epoch}")
        if epoch <= len(expected_steps) and int(expected_steps[epoch - 1]) != steps:
            issues.append(f"expected step vector differs from schedule at epoch {epoch}")

        parquet = telemetry_root / f"epoch_{epoch:04d}_process_telemetry.parquet"
        sidecar = parquet.with_suffix(".json")
        if not parquet.is_file() or not sidecar.is_file():
            issues.append(f"process telemetry epoch {epoch} is incomplete")
            continue
        metadata = _read_json(sidecar, f"process telemetry sidecar epoch {epoch}")
        expected_sidecar = {
            "schema_version": "stage1.process_telemetry.v1",
            "status": "COMPLETE",
            "epoch": epoch,
            "observed_epoch_samples": total,
            "observed_replay_samples": normal + guard,
        }
        for key, value in expected_sidecar.items():
            if metadata.get(key) != value:
                issues.append(f"sidecar {key} mismatch at epoch {epoch}")
        if str(metadata.get("parquet_sha256", "")).upper() != sha256_file(parquet):
            issues.append(f"process telemetry parquet checksum mismatch at epoch {epoch}")
        cumulative_observed_replay += int(metadata.get("observed_replay_samples", -1))
        sidecar_bytes += sidecar.stat().st_size
        parquet_bytes += parquet.stat().st_size

        role_rel = str(metadata.get("role_summary_relpath", ""))
        role_path = telemetry_root / role_rel if role_rel else Path()
        if not role_rel or not role_path.is_file():
            issues.append(f"role loss summary missing at epoch {epoch}")
        else:
            if str(metadata.get("role_summary_sha256", "")).upper() != sha256_file(role_path):
                issues.append(f"role loss summary checksum mismatch at epoch {epoch}")
            summary = _read_json(role_path, f"role loss summary epoch {epoch}")
            if summary.get("schema_version") != "stage1.process_role_loss_summary.v1" or summary.get("status") != "COMPLETE":
                issues.append(f"role loss summary schema/status mismatch at epoch {epoch}")
            roles = _role_map(summary)
            expected_counts = {
                "normal_replay": normal,
                "defect_guard": guard,
            }
            for role, expected_count in expected_counts.items():
                role_row = roles.get(role)
                if expected_count == 0 and role_row is not None and int(role_row.get("total_exposure_count", 0)) != 0:
                    issues.append(f"zero-exposure role contains observations: {role} epoch {epoch}")
                if expected_count > 0:
                    if role_row is None or int(role_row.get("total_exposure_count", -1)) != expected_count:
                        issues.append(f"role exposure count mismatch: {role} epoch {epoch}")
                    elif role_row.get("loss_mean") is None or not math.isfinite(float(role_row["loss_mean"])):
                        issues.append(f"role loss missing/non-finite: {role} epoch {epoch}")
            role_summary_bytes += role_path.stat().st_size
            role_loss_epochs += 1
        per_epoch.append(
            {
                "epoch": epoch,
                "base_sample_exposures": base,
                "normal_replay_exposures": normal,
                "defect_guard_exposures": guard,
                "optimizer_steps": steps,
                "sidecar_sha256": sha256_file(sidecar),
            }
        )
    if no_replay and cumulative_observed_replay != 0:
        issues.append("no-replay arm has non-zero replay exposure")
    if cumulative_expected_replay != cumulative_observed_replay:
        issues.append("cumulative replay dose differs from discrete schedule")
    if telemetry_root.exists() and any(telemetry_root.glob("*.tmp")):
        issues.append("temporary telemetry artifacts remain after publication")
    report = {
        "schema_version": ALL_EPOCH_SCHEMA,
        "status": "PASS" if not issues else "FAIL",
        "created_at_unix": time.time(),
        "issues": issues,
        "run_id": audit.get("run_id"),
        "arm_id": audit.get("arm_id"),
        "expected_epochs": expected_epochs,
        "validated_epoch_count": len(per_epoch),
        "no_replay_arm": no_replay,
        "cumulative_expected_replay_exposures": cumulative_expected_replay,
        "cumulative_observed_replay_exposures": cumulative_observed_replay,
        "role_loss_epoch_count": role_loss_epochs,
        "storage": {
            "parquet_bytes": parquet_bytes,
            "sidecar_bytes": sidecar_bytes,
            "role_summary_bytes": role_summary_bytes,
            "total_bytes": parquet_bytes + sidecar_bytes + role_summary_bytes,
            "mean_bytes_per_epoch": (
                (parquet_bytes + sidecar_bytes + role_summary_bytes) / expected_epochs
            ),
        },
        "identity": {
            "audit_sha256": sha256_file(audit_file),
            "schedule_sha256": sha256_file(schedule_file),
        },
        "per_epoch": per_epoch,
    }
    atomic_write_json(output_path, report, overwrite=True)
    if issues:
        raise ValidationError(f"all-epoch telemetry validation failed; see {output_path}")
    return report
