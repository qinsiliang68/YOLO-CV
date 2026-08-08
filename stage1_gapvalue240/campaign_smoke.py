"""Fail-closed validation for the local real-data campaign smoke run."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from .campaign_process_telemetry import (
    ProcessTelemetrySpec,
    validate_process_telemetry_epoch,
)
from .errors import ValidationError
from .monitor import RESOURCE_COLUMNS
from .util import atomic_write_bytes, atomic_write_json, sha256_file


class CampaignSmokeError(ValidationError):
    """Raised when smoke evidence is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class LocalSmokeValidationSpec:
    run_id: str
    arm_id: str
    output_dir: Path
    telemetry: ProcessTelemetrySpec
    expected_epochs: int
    expected_steps_per_epoch: int
    canonical_lock_file_sha256: str
    declared_smoke_overrides: tuple[str, ...]
    resource_log: Path
    telemetry_segment_ids_by_epoch: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir).resolve())
        object.__setattr__(self, "resource_log", Path(self.resource_log).resolve())
        object.__setattr__(
            self,
            "declared_smoke_overrides",
            tuple(sorted(map(str, self.declared_smoke_overrides))),
        )
        if not self.run_id or not self.arm_id:
            raise CampaignSmokeError("smoke identity must not be empty")
        if self.expected_epochs <= 0 or self.expected_steps_per_epoch <= 0:
            raise CampaignSmokeError("smoke expected epoch/step counts must be positive")
        if len(str(self.canonical_lock_file_sha256)) != 64:
            raise CampaignSmokeError("smoke canonical lock SHA must have 64 characters")
        if self.telemetry.output_dir != self.output_dir / "process_telemetry":
            raise CampaignSmokeError("smoke telemetry must belong to the validated output")
        segment_ids = self.telemetry_segment_ids_by_epoch
        if segment_ids is None:
            segment_ids = (self.telemetry.segment_id,) * self.expected_epochs
        segment_ids = tuple(map(str, segment_ids))
        if len(segment_ids) != self.expected_epochs or any(not value for value in segment_ids):
            raise CampaignSmokeError(
                "smoke telemetry segment identity must be registered for every epoch"
            )
        object.__setattr__(self, "telemetry_segment_ids_by_epoch", segment_ids)


@dataclass(frozen=True)
class LocalSmokeValidationResult:
    report_path: Path
    details_path: Path
    artifact_manifest_path: Path
    artifact_count: int
    total_bytes: int


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CampaignSmokeError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CampaignSmokeError(f"unreadable {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CampaignSmokeError(f"{label} is not a JSON object: {path}")
    return value


def _temporary_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (path.suffix.lower() == ".tmp" or ".tmp." in path.name or path.name.endswith(".tmpdir"))
    )


def _validate_audit(spec: LocalSmokeValidationSpec) -> dict[str, Any]:
    audit = _load_json(spec.output_dir / "dynamic_training_audit.json", "dynamic audit")
    expected = {
        "run_id": spec.run_id,
        "arm_id": spec.arm_id,
        "execution_mode": "SMOKE",
        "canonical_lock_file_sha256": str(spec.canonical_lock_file_sha256).upper(),
        "completed_epochs": spec.expected_epochs,
        "loss_finite": True,
    }
    mismatch = {key: (audit.get(key), value) for key, value in expected.items() if audit.get(key) != value}
    if mismatch:
        raise CampaignSmokeError(f"smoke dynamic audit mismatch: {mismatch}")
    if tuple(sorted(map(str, audit.get("smoke_canonical_overrides", [])))) != spec.declared_smoke_overrides:
        raise CampaignSmokeError("smoke canonical override declaration drifted in dynamic audit")
    expected_steps = list(audit.get("expected_steps_by_epoch", []))[: spec.expected_epochs]
    observed_steps = list(audit.get("observed_steps_by_epoch", []))[: spec.expected_epochs]
    required_steps = [spec.expected_steps_per_epoch] * spec.expected_epochs
    if expected_steps != required_steps or observed_steps != required_steps:
        raise CampaignSmokeError(
            f"smoke step coverage mismatch: expected={expected_steps}, observed={observed_steps}"
        )
    records = list(audit.get("epoch_records", []))
    record_epochs = [int(row.get("epoch", -1)) for row in records]
    if record_epochs != list(range(1, spec.expected_epochs + 1)):
        raise CampaignSmokeError(f"smoke epoch records are incomplete or duplicated: {record_epochs}")
    segments = list(audit.get("segments", []))
    if not segments or segments[-1].get("status") != "COMPLETED":
        raise CampaignSmokeError("smoke final segment is not COMPLETED")
    return audit


def _validate_resolved_args(spec: LocalSmokeValidationSpec) -> dict[str, Any]:
    resolved = _load_json(spec.output_dir / "resolved_training_args.json", "resolved args")
    if resolved.get("execution_mode") != "SMOKE":
        raise CampaignSmokeError("resolved args do not identify SMOKE execution")
    if str(resolved.get("canonical_lock_file_sha256", "")).upper() != str(
        spec.canonical_lock_file_sha256
    ).upper():
        raise CampaignSmokeError("resolved args canonical lock SHA mismatch")
    validation = resolved.get("canonical_lock_validation")
    if not isinstance(validation, dict) or validation.get("status") != "PASS_WITH_DECLARED_SMOKE_OVERRIDES":
        raise CampaignSmokeError("resolved args lack canonical smoke validation")
    observed = tuple(sorted(map(str, validation.get("declared_smoke_overrides", []))))
    if observed != spec.declared_smoke_overrides:
        raise CampaignSmokeError("resolved args smoke override declaration mismatch")
    return resolved


def _validate_checkpoint(spec: LocalSmokeValidationSpec) -> dict[str, Any]:
    state = spec.output_dir / "training_state"
    checkpoint = state / f"checkpoint_epoch_{spec.expected_epochs:04d}.pt"
    sidecar = checkpoint.with_suffix(".json")
    stable_last = state / "last.pt"
    metadata = _load_json(sidecar, "checkpoint sidecar")
    if not checkpoint.is_file() or not stable_last.is_file():
        raise CampaignSmokeError("smoke resumable checkpoint is missing")
    checkpoint_sha = sha256_file(checkpoint)
    expected = {
        "epoch": spec.expected_epochs,
        "sha256": checkpoint_sha,
        "resumable_expected": True,
    }
    mismatch = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatch:
        raise CampaignSmokeError(f"smoke checkpoint sidecar mismatch: {mismatch}")
    if sha256_file(stable_last) != checkpoint_sha:
        raise CampaignSmokeError("stable last checkpoint differs from final smoke checkpoint")
    return {"epoch": spec.expected_epochs, "sha256": checkpoint_sha}


def _validate_telemetry(spec: LocalSmokeValidationSpec) -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    identity_sha: str | None = None
    for epoch in range(1, spec.expected_epochs + 1):
        assert spec.telemetry_segment_ids_by_epoch is not None
        segment_id = spec.telemetry_segment_ids_by_epoch[epoch - 1]
        epoch_spec = replace(spec.telemetry, segment_id=segment_id)
        try:
            metadata = validate_process_telemetry_epoch(epoch_spec, epoch)
        except Exception as exc:
            raise CampaignSmokeError(f"process telemetry validation failed at epoch {epoch}: {exc}") from exc
        parquet = spec.telemetry.output_dir / f"epoch_{epoch:04d}_process_telemetry.parquet"
        try:
            frame = pl.read_parquet(parquet)
        except Exception as exc:
            raise CampaignSmokeError(f"process telemetry parquet unreadable at epoch {epoch}") from exc
        required = {
            "record_type",
            "segment_id",
            "epoch",
            "sample_id",
            "exposure_role",
            "total_exposure_count",
            "base_exposure_count",
            "replay_normal_exposure_count",
            "guard_defect_exposure_count",
        }
        missing = required - set(frame.columns)
        if missing:
            raise CampaignSmokeError(f"process telemetry epoch {epoch} missing columns: {sorted(missing)}")
        observed_segments = set(map(str, frame.get_column("segment_id").unique().to_list()))
        if observed_segments != {segment_id}:
            raise CampaignSmokeError(
                f"process telemetry epoch {epoch} segment identity mismatch: {observed_segments}"
            )
        epoch_rows = frame.filter(pl.col("record_type") == "EPOCH")
        if epoch_rows.height != 1:
            raise CampaignSmokeError(f"process telemetry epoch {epoch} has {epoch_rows.height} EPOCH rows")
        total = int(epoch_rows.item(0, "total_exposure_count"))
        if total != spec.telemetry.expected_epoch_samples:
            raise CampaignSmokeError(f"process telemetry epoch {epoch} exposure count mismatch")
        sample_ids = frame.filter(pl.col("record_type") == "SAMPLE").get_column("sample_id").to_list()
        if len(sample_ids) != len(set(map(str, sample_ids))):
            raise CampaignSmokeError(f"process telemetry epoch {epoch} has duplicate sample records")
        observed_identity = str(metadata.get("identity_index_sha256", ""))
        if identity_sha is None:
            identity_sha = observed_identity
        elif observed_identity != identity_sha:
            raise CampaignSmokeError("process telemetry identity index changed across epochs")
        records.append(
            {
                "epoch": epoch,
                "segment_id": segment_id,
                "row_count": int(metadata["row_count"]),
                "sample_record_count": int(metadata["sample_record_count"]),
                "parquet_sha256": str(metadata["parquet_sha256"]),
                "identity_index_sha256": observed_identity,
            }
        )
    return records, str(identity_sha or "")


def _validate_resource_log(spec: LocalSmokeValidationSpec) -> dict[str, Any]:
    if not spec.resource_log.is_file():
        raise CampaignSmokeError(f"smoke resource log is missing: {spec.resource_log}")
    frame = pd.read_csv(spec.resource_log, keep_default_na=False)
    missing = set(RESOURCE_COLUMNS) - set(frame.columns)
    if missing or frame.empty:
        raise CampaignSmokeError(f"smoke resource log is incomplete: missing={sorted(missing)}")
    numeric_positive = ("process_rss_bytes", "system_ram_available_bytes", "disk_free_bytes")
    for column in numeric_positive:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() == 0 or float(values.max()) <= 0:
            raise CampaignSmokeError(f"smoke resource log has no positive {column}")
    return {
        "row_count": len(frame),
        "runtime_phases": sorted(set(frame.runtime_phase.astype(str))),
        "gpu_unavailable_rows": int(frame.status.astype(str).str.startswith("GPU_UNAVAILABLE:").sum()),
        "sha256": sha256_file(spec.resource_log),
    }


def _artifact_rows(root: Path, excluded: set[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.resolve() in excluded:
            continue
        if path.suffix.lower() == ".tmp" or ".tmp." in path.name:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def validate_local_smoke_run(spec: LocalSmokeValidationSpec) -> LocalSmokeValidationResult:
    """Validate real smoke artifacts and publish a complete hashed evidence set."""

    if not spec.output_dir.is_dir():
        raise FileNotFoundError(spec.output_dir)
    temporary = _temporary_files(spec.output_dir)
    if temporary:
        raise CampaignSmokeError(f"smoke output contains temporary artifact: {temporary[0]}")
    results_path = spec.output_dir / "trainer/results.csv"
    if not results_path.is_file() or len(pd.read_csv(results_path)) != spec.expected_epochs:
        raise CampaignSmokeError("smoke trainer results do not cover every expected epoch")
    audit = _validate_audit(spec)
    resolved = _validate_resolved_args(spec)
    checkpoint = _validate_checkpoint(spec)
    telemetry, identity_sha = _validate_telemetry(spec)
    resources = _validate_resource_log(spec)

    validation_dir = spec.output_dir / "smoke_validation"
    if validation_dir.exists() and any(validation_dir.iterdir()):
        raise FileExistsError(f"smoke validation output is not empty: {validation_dir}")
    validation_dir.mkdir(parents=True, exist_ok=True)
    details_path = validation_dir / "SMOKE_VALIDATION_DETAILS.json"
    manifest_path = validation_dir / "ARTIFACT_MANIFEST.csv"
    report_path = validation_dir / "SMOKE_VALIDATION.json"
    atomic_write_json(
        details_path,
        {
            "schema_version": "stage1.local_real_data_smoke_details.v1",
            "run_id": spec.run_id,
            "arm_id": spec.arm_id,
            "validated_epochs": list(range(1, spec.expected_epochs + 1)),
            "expected_steps_per_epoch": spec.expected_steps_per_epoch,
            "canonical_lock_file_sha256": str(spec.canonical_lock_file_sha256).upper(),
            "declared_smoke_overrides": list(spec.declared_smoke_overrides),
            "telemetry_identity_index_sha256": identity_sha,
            "telemetry_epochs": telemetry,
            "checkpoint": checkpoint,
            "resource_log": resources,
            "audit_schema_version": audit.get("schema_version"),
            "resolved_canonical_validation": resolved.get("canonical_lock_validation"),
        },
    )
    excluded = {manifest_path.resolve(), report_path.resolve()}
    rows = _artifact_rows(spec.output_dir, excluded)
    atomic_write_bytes(
        manifest_path,
        pd.DataFrame(rows, columns=["relative_path", "size_bytes", "sha256"])
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8"),
    )
    total_bytes = int(sum(int(row["size_bytes"]) for row in rows))
    atomic_write_json(
        report_path,
        {
            "schema_version": "stage1.local_real_data_smoke_validation.v1",
            "status": "PASS",
            "run_id": spec.run_id,
            "arm_id": spec.arm_id,
            "validated_epochs": list(range(1, spec.expected_epochs + 1)),
            "local_real_data_smoke": "PASS",
            "all_epoch_telemetry": "PASS",
            "canonical_lock_binding": "PASS",
            "checkpoint_resume_contract": "PASS",
            "resource_log_contract": "PASS",
            "no_partial_artifacts": "PASS",
            "canonical_lock_file_sha256": str(spec.canonical_lock_file_sha256).upper(),
            "artifact_manifest_sha256": sha256_file(manifest_path),
            "artifact_count": len(rows),
            "total_bytes": total_bytes,
        },
    )
    return LocalSmokeValidationResult(
        report_path=report_path,
        details_path=details_path,
        artifact_manifest_path=manifest_path,
        artifact_count=len(rows),
        total_bytes=total_bytes,
    )


__all__ = [
    "CampaignSmokeError",
    "LocalSmokeValidationResult",
    "LocalSmokeValidationSpec",
    "validate_local_smoke_run",
]
