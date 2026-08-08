"""Construction and fail-closed checks for local real-data failure drills."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .campaign_canonical_lock import load_canonical_training_lock
from .campaign_dynamic_training import DynamicTrainingSpec, SmokeFailureInjection
from .campaign_process_telemetry import ProcessTelemetrySpec
from .campaign_smoke_dataset import LocalSmokeDataset
from .errors import ValidationError
from .util import sha256_file


class FailureSmokeError(ValidationError):
    """Raised when a failure drill leaves ambiguous or falsely complete state."""


@dataclass(frozen=True)
class FailureSmokeSegment:
    training: DynamicTrainingSpec
    telemetry: ProcessTelemetrySpec
    smoke_canonical_overrides: tuple[str, ...]


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FailureSmokeError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FailureSmokeError(f"unreadable {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise FailureSmokeError(f"{label} is not a JSON object: {path}")
    return payload


def build_failure_smoke_segment(
    *,
    repo_root: str | Path,
    subset: LocalSmokeDataset,
    output_dir: str | Path,
    run_id: str,
    total_epochs: int,
    segment_start_epoch: int,
    segment_end_epoch: int,
    batch: int,
    workers: int,
    device: str,
    seed: int,
    segment_id: str,
    resume_checkpoint: str | Path | None = None,
    failure_injection: SmokeFailureInjection | None = None,
) -> FailureSmokeSegment:
    """Bind one failure-test segment to the same canonical training lock as formal jobs."""

    repo = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    validation = _load_json(subset.validation_path, "local smoke dataset validation")
    if validation.get("status") != "PASS":
        raise FailureSmokeError("local smoke dataset validation is not PASS")
    for path in (
        subset.dataset_dir,
        subset.base_normal_manifest,
        subset.base_defect_manifest,
        subset.replay_identity_manifest,
        subset.monitor_manifest,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    canonical_lock = repo / "configs/stage1_gapvalue240/CANONICAL_TRAINING_LOCK_v1.json"
    lock = load_canonical_training_lock(canonical_lock)
    dimensions = {
        "epochs": int(total_epochs),
        "batch": int(batch),
        "imgsz": 224,
        "workers": int(workers),
    }
    overrides = tuple(
        key for key, value in dimensions.items() if value != lock.immutable_args[key]
    )
    telemetry = ProcessTelemetrySpec(
        run_id=run_id,
        arm_id="FAILURE_SMOKE",
        segment_id=segment_id,
        output_dir=output / "process_telemetry",
        base_normal_manifest=subset.base_normal_manifest,
        base_defect_manifest=subset.base_defect_manifest,
        replay_identity_manifest=subset.replay_identity_manifest,
        monitor_manifest=subset.monitor_manifest,
        expected_epoch_samples=subset.expected_epoch_samples,
        expected_replay_samples=subset.expected_replay_samples,
    )
    expected_steps = (subset.expected_epoch_samples + batch - 1) // batch
    resume = Path(resume_checkpoint).resolve() if resume_checkpoint is not None else None
    training = DynamicTrainingSpec(
        run_id=run_id,
        arm_id="FAILURE_SMOKE",
        schedule_id="LOCAL_FAILURE_INJECTION",
        selection_digest="F" * 64,
        active_selection_digest="E" * 64,
        dataset_dir=subset.dataset_dir,
        checkpoint=repo / "yolo11l-cls.pt",
        output_dir=output,
        yolo_root=repo / "YOLOv11",
        total_epochs=total_epochs,
        segment_start_epoch=segment_start_epoch,
        segment_end_epoch=segment_end_epoch,
        batch=batch,
        imgsz=224,
        seed=seed,
        device=str(device),
        workers=workers,
        expected_steps_per_epoch=expected_steps,
        retained_checkpoint_epochs=tuple(sorted({segment_end_epoch, total_epochs})),
        execution_mode="SMOKE",
        resume_checkpoint=resume,
        segment_id=segment_id,
        process_telemetry=telemetry,
        canonical_lock_path=canonical_lock,
        canonical_lock_file_sha256=sha256_file(canonical_lock),
        smoke_canonical_overrides=overrides,
        smoke_failure_injection=failure_injection,
    )
    return FailureSmokeSegment(training, telemetry, overrides)


def validate_interrupted_boundary(
    output_dir: str | Path,
    *,
    expected_completed_epoch: int,
    expected_next_epoch: int,
) -> dict[str, Any]:
    """Prove a killed process exposed only complete prior epochs and one resumable checkpoint."""

    output = Path(output_dir).resolve()
    audit = _load_json(output / "dynamic_training_audit.json", "interrupted dynamic audit")
    if int(audit.get("completed_epochs", -1)) != expected_completed_epoch:
        raise FailureSmokeError("interrupted audit completed epoch mismatch")
    record_epochs = [int(row.get("epoch", -1)) for row in audit.get("epoch_records", [])]
    if record_epochs != list(range(1, expected_completed_epoch + 1)):
        raise FailureSmokeError(f"interrupted audit epoch records are invalid: {record_epochs}")
    results = output / "trainer/results.csv"
    if not results.is_file() or len(pd.read_csv(results)) != expected_completed_epoch:
        raise FailureSmokeError("interrupted trainer results do not match the durable boundary")

    state = output / "training_state"
    stable_last = state / "last.pt"
    checkpoint = state / f"checkpoint_epoch_{expected_completed_epoch:04d}.pt"
    sidecar = checkpoint.with_suffix(".json")
    metadata = _load_json(sidecar, "interrupted checkpoint sidecar")
    if not checkpoint.is_file() or not stable_last.is_file():
        raise FailureSmokeError("interrupted run has no resumable boundary checkpoint")
    checkpoint_sha = sha256_file(checkpoint)
    if (
        int(metadata.get("epoch", -1)) != expected_completed_epoch
        or str(metadata.get("sha256", "")).upper() != checkpoint_sha
        or sha256_file(stable_last) != checkpoint_sha
    ):
        raise FailureSmokeError("interrupted boundary checkpoint identity mismatch")

    telemetry = output / "process_telemetry"
    for epoch in range(1, expected_completed_epoch + 1):
        parquet = telemetry / f"epoch_{epoch:04d}_process_telemetry.parquet"
        telemetry_meta = _load_json(parquet.with_suffix(".json"), "completed telemetry sidecar")
        if (
            not parquet.is_file()
            or telemetry_meta.get("status") != "COMPLETE"
            or int(telemetry_meta.get("epoch", -1)) != epoch
            or str(telemetry_meta.get("parquet_sha256", "")).upper() != sha256_file(parquet)
        ):
            raise FailureSmokeError(f"completed telemetry epoch {epoch} is invalid")
    next_prefix = f"epoch_{expected_next_epoch:04d}_process_telemetry"
    next_artifacts = sorted(telemetry.glob(f"*{next_prefix}*"))
    if next_artifacts:
        raise FailureSmokeError(f"next epoch telemetry was falsely published: {next_artifacts}")
    return {
        "schema_version": "stage1.interrupted_boundary_validation.v1",
        "status": "PASS",
        "completed_epoch": expected_completed_epoch,
        "next_epoch_absent": expected_next_epoch,
        "resume_checkpoint_sha256": sha256_file(stable_last),
    }


def validate_telemetry_write_interruption(
    output_dir: str | Path,
    *,
    failed_epoch: int,
) -> dict[str, Any]:
    """Verify an interrupted parquet/sidecar transaction published no completed epoch."""

    output = Path(output_dir).resolve()
    audit = _load_json(output / "dynamic_training_audit.json", "write-failure dynamic audit")
    segments = list(audit.get("segments", []))
    if not segments or segments[-1].get("status") != "FAILED":
        raise FailureSmokeError("telemetry write interruption did not fail the training segment")
    telemetry = output / "process_telemetry"
    prefix = f"epoch_{failed_epoch:04d}_process_telemetry"
    published = sorted(telemetry.glob(f"*{prefix}*")) if telemetry.is_dir() else []
    if published:
        raise FailureSmokeError(f"telemetry write interruption left published telemetry: {published}")
    temporary = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and (path.suffix.lower() == ".tmp" or ".tmp." in path.name)
    )
    if temporary:
        raise FailureSmokeError(f"telemetry write interruption left temporary files: {temporary}")
    return {
        "schema_version": "stage1.telemetry_write_interruption_validation.v1",
        "status": "PASS",
        "failed_epoch": failed_epoch,
        "audit_completed_epochs": int(audit.get("completed_epochs", -1)),
        "published_epoch_artifacts": 0,
    }


__all__ = [
    "FailureSmokeError",
    "FailureSmokeSegment",
    "build_failure_smoke_segment",
    "validate_interrupted_boundary",
    "validate_telemetry_write_interruption",
]
