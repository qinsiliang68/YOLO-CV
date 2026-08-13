from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .columnar import partition_identity, write_zstd_parquet
from .errors import ErrorCode, SctsrError
from .ledger_schema import EXPOSURE_SCHEMA

REQUIRED_FIELDS = frozenset(EXPOSURE_SCHEMA.names)


def _quantiles(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {"multiplicity_q0": 0.0, "multiplicity_q25": 0.0, "multiplicity_q50": 0.0, "multiplicity_q75": 0.0, "multiplicity_q100": 0.0}
    result = np.quantile(np.asarray(values, dtype=np.float64), [0.0, 0.25, 0.5, 0.75, 1.0], method="linear")
    return {name: float(value) for name, value in zip(("multiplicity_q0", "multiplicity_q25", "multiplicity_q50", "multiplicity_q75", "multiplicity_q100"), result, strict=True)}


def build_exposure_row(
    *,
    run_id: str,
    parent_id: str,
    arm_id: str,
    training_seed: int,
    epoch: int,
    base_denominator: int,
    replay_rate_numerator: int,
    replay_rate_denominator: int,
    replay_sample_ids: Sequence[str],
    optimizer_steps: int,
    expected_optimizer_steps: int,
    base_order_digest: str,
    base_augmentation_digest: str,
    schedule_digest: str,
    identity_pool_digest: str,
    occurrence_partition_sha: str,
    step_partition_sha: str,
    telemetry_partition_sha: str,
    checkpoint_sha: str,
    cumulative_occurrences: int,
    ema_updates_delta: int,
    scheduler_epoch_transitions_delta: int,
    write_seconds: float,
    dataloader_wait_seconds: float,
    training_seconds: float,
    evaluation_seconds: float,
    disk_bytes_written: int,
    transaction_generation: int,
    actual_base_denominator: int | None = None,
    status: str = "PASS",
) -> dict[str, Any]:
    expected_replay, remainder = divmod(base_denominator * replay_rate_numerator, replay_rate_denominator)
    if remainder:
        raise SctsrError(ErrorCode.RATE_NOT_INTEGRAL, "Exposure rate is not integral")
    actual_replay = len(replay_sample_ids)
    if actual_replay != expected_replay:
        raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "Actual replay occurrences differ from rate plan", observed=actual_replay, expected=expected_replay)
    actual_base = base_denominator if actual_base_denominator is None else int(actual_base_denominator)
    if actual_base != base_denominator:
        raise SctsrError(ErrorCode.DENOMINATOR_IDENTITY_MISMATCH, "Actual base denominator differs from its frozen plan", observed=actual_base, expected=base_denominator)
    if optimizer_steps != expected_optimizer_steps:
        raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "Optimizer step count differs from frozen base process")
    if ema_updates_delta not in {0, expected_optimizer_steps}:
        raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "EMA update count is inconsistent with the frozen base process")
    if scheduler_epoch_transitions_delta != 1:
        raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "Scheduler must advance exactly once per completed epoch")
    multiplicity = Counter(replay_sample_ids)
    values = sorted(multiplicity.values())
    row = {
        "run_id": run_id,
        "parent_id": parent_id,
        "arm_id": arm_id,
        "training_seed": training_seed,
        "epoch": epoch,
        "denominator_role": "CANONICAL_BASE_OPTIMIZER_VISIBLE_EXPOSURE",
        "base_denominator_planned": base_denominator,
        "base_denominator_actual": actual_base,
        "rate_numerator": replay_rate_numerator,
        "rate_denominator": replay_rate_denominator,
        "replay_numerator_planned": expected_replay,
        "replay_numerator_actual": actual_replay,
        "unique_replay_ids": len(multiplicity),
        "repeat_occurrences": actual_replay - len(multiplicity),
        "cumulative_occurrences": cumulative_occurrences,
        "multiplicity_min": min(values) if values else 0,
        "multiplicity_max": max(values) if values else 0,
        "multiplicity_mean": (sum(values) / len(values)) if values else 0.0,
        **_quantiles(values),
        "base_optimizer_steps_planned": expected_optimizer_steps,
        "base_optimizer_steps_actual": optimizer_steps,
        "ema_updates_delta": ema_updates_delta,
        "scheduler_epoch_transitions_delta": scheduler_epoch_transitions_delta,
        "base_order_digest": base_order_digest,
        "base_augmentation_digest": base_augmentation_digest,
        "replay_schedule_digest": schedule_digest,
        "identity_pool_digest": identity_pool_digest,
        "occurrence_partition_sha256": occurrence_partition_sha,
        "step_partition_sha256": step_partition_sha,
        "telemetry_partition_sha256": telemetry_partition_sha,
        "checkpoint_sha256": checkpoint_sha,
        "write_seconds": write_seconds,
        "dataloader_wait_seconds": dataloader_wait_seconds,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "disk_bytes_written": disk_bytes_written,
        "transaction_generation": transaction_generation,
        "validation_status": status,
    }
    validate_exposure_rows([row])
    return row


def validate_exposure_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Exposure partition may not be empty")
    for index, row in enumerate(rows):
        observed = set(row)
        if observed != REQUIRED_FIELDS:
            raise SctsrError(
                ErrorCode.SCHEMA_VALIDATION_FAILED,
                "Epoch exposure row fields do not exactly match the taskbook schema",
                failing_field=f"row[{index}]",
                observed={"missing": sorted(REQUIRED_FIELDS - observed), "extra": sorted(observed - REQUIRED_FIELDS)},
            )
        if row["denominator_role"] != "CANONICAL_BASE_OPTIMIZER_VISIBLE_EXPOSURE":
            raise SctsrError(ErrorCode.DENOMINATOR_IDENTITY_MISMATCH, "Exposure denominator role is not canonical")
        if int(row["base_denominator_planned"]) != int(row["base_denominator_actual"]):
            raise SctsrError(ErrorCode.DENOMINATOR_IDENTITY_MISMATCH, "Exposure planned/actual base denominator mismatch")
        if int(row["replay_numerator_planned"]) != int(row["replay_numerator_actual"]):
            raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "Exposure planned/actual replay numerator mismatch")
        if int(row["base_optimizer_steps_planned"]) != int(row["base_optimizer_steps_actual"]):
            raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "Exposure planned/actual base step mismatch")
        if int(row["repeat_occurrences"]) != int(row["replay_numerator_actual"]) - int(row["unique_replay_ids"]):
            raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "Exposure unique/repeat conservation failed")
        for field in ("occurrence_partition_sha256", "step_partition_sha256", "telemetry_partition_sha256", "checkpoint_sha256"):
            if len(str(row[field])) != 64:
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Exposure partition binding is not SHA-256", failing_field=field)
        if int(row["transaction_generation"]) < 1 or row["validation_status"] != "PASS":
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Exposure row is not a validated transaction generation")


def write_exposure_partition(rows: Sequence[Mapping[str, Any]], path: str | Path):
    validate_exposure_rows(rows)
    run_id, epoch = partition_identity(path, required=True)
    if any(str(row["run_id"]) != run_id or int(row["epoch"]) != epoch for row in rows):
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Exposure row identity differs from its run/epoch partition")
    return write_zstd_parquet(
        rows,
        path,
        schema_version="stage1.sctsr.epoch_exposure_ledger.v1",
        schema=EXPOSURE_SCHEMA,
        require_run_epoch_partition=True,
    )
