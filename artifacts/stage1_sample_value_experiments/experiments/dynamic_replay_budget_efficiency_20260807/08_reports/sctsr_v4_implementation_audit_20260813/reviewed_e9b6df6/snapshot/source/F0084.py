from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .columnar import partition_identity, write_zstd_parquet
from .errors import ErrorCode, SctsrError
from .ledger_schema import STEP_SCHEMA

REQUIRED_FIELDS = frozenset(STEP_SCHEMA.names)


def _normalize_optimizer_groups(groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for index, group in enumerate(groups):
        betas = group.get("betas")
        beta1 = group.get("beta1")
        beta2 = group.get("beta2")
        if betas is not None:
            beta1, beta2 = betas
        normalized.append(
            {
                "group_index": int(group.get("group_index", index)),
                "lr": float(group.get("lr", 0.0)),
                "initial_lr": None if group.get("initial_lr") is None else float(group["initial_lr"]),
                "momentum": None if group.get("momentum") is None else float(group["momentum"]),
                "beta1": None if beta1 is None else float(beta1),
                "beta2": None if beta2 is None else float(beta2),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "dampening": None if group.get("dampening") is None else float(group["dampening"]),
                "nesterov": None if group.get("nesterov") is None else bool(group["nesterov"]),
            }
        )
    return normalized


def validate_step_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Optimizer-step partition may not be empty")
    for index, row in enumerate(rows):
        observed = set(row)
        if observed != REQUIRED_FIELDS:
            raise SctsrError(
                ErrorCode.SCHEMA_VALIDATION_FAILED,
                "Optimizer-step row fields do not exactly match the taskbook schema",
                failing_field=f"row[{index}]",
                observed={"missing": sorted(REQUIRED_FIELDS - observed), "extra": sorted(observed - REQUIRED_FIELDS)},
            )
        for field in STEP_SCHEMA:
            value = row[field.name]
            if isinstance(value, str) and value.strip() in {"", "unknown", "UNKNOWN", "TODO", "TBD", "同上"}:
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Optimizer-step row contains an invalid placeholder", failing_field=field.name)
        if int(row["optimizer_step_count_delta"]) != 1:
            raise SctsrError(ErrorCode.REPLAY_ADDED_OPTIMIZER_STEP, "Each base step must advance optimizer count exactly once")
        if int(row["global_step_after"]) - int(row["global_step_before"]) != 1:
            raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "Global step delta must be one")
        if bool(row["overflow_or_step_skipped"]):
            raise SctsrError(ErrorCode.OPTIMIZER_STEP_SKIPPED, "Skipped AMP step invalidates a canonical paired epoch")
        if int(row["ema_updates_after"]) - int(row["ema_updates_before"]) not in {0, 1}:
            raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "EMA update delta must be zero or one")
        base_size = int(row["base_batch_size"])
        replay_size = int(row["replay_microbatch_size"])
        if not 1 <= base_size <= 128 or replay_size < 0 or replay_size > base_size // 4:
            raise SctsrError(ErrorCode.REPLAY_MICROBATCH_CAP_EXCEEDED, "Replay microbatch exceeds the actual base-batch 25% cap")
        for field in (
            "base_loss", "replay_loss", "combined_loss_for_reporting", "parameter_grad_norm_before_clip",
            "parameter_grad_norm_after_clip", "warmup_progress", "dataloader_wait_seconds", "base_forward_seconds",
            "replay_forward_seconds", "backward_seconds", "optimizer_seconds",
        ):
            if not math.isfinite(float(row[field])) or float(row[field]) < 0:
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Step numeric value is invalid", failing_field=field)
        if abs(float(row["combined_loss_for_reporting"]) - (float(row["base_loss"]) + float(row["replay_loss"]))) > 1e-8:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Combined reporting loss does not equal base plus replay")
        if not row["learning_rates"] or len(row["learning_rates"]) != len(row["optimizer_hyperparameters"]):
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Optimizer param-group evidence is incomplete")
        replay = replay_size > 0
        if len(str(row["rng_digest_before_base"])) != 64:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Base-step RNG digest is missing")
        for before, after, reason, name in (
            (row["bn_digest_before_replay"], row["bn_digest_after_replay_restore"], row["bn_reason"], "BN"),
            (row["rng_digest_before_replay"], row["rng_digest_after_replay_restore"], row["rng_reason"], "RNG"),
        ):
            if replay and (before is None or after is None or before != after or reason != "PRESENT"):
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, f"{name} replay isolation evidence is invalid")
            if not replay and (before is not None or after is not None or reason != "NO_REPLAY_IN_STEP"):
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, f"{name} no-replay reason is invalid")
        if replay and (row["replay_augmentation_digest"] is None or row["replay_augmentation_reason"] != "PRESENT"):
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Replay augmentation evidence is missing")
        if not replay and (row["replay_augmentation_digest"] is not None or row["replay_augmentation_reason"] != "NO_REPLAY_IN_STEP"):
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "No-replay augmentation reason is invalid")
        if replay and (row["replay_rng_fork_digest"] is None or len(str(row["replay_rng_fork_digest"])) != 64 or row["replay_rng_fork_reason"] != "PRESENT"):
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Replay RNG fork evidence is missing")
        if not replay and (row["replay_rng_fork_digest"] is not None or row["replay_rng_fork_reason"] != "NO_REPLAY_IN_STEP"):
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "No-replay RNG fork reason is invalid")
        if row["amp_scale_before"] is None or row["amp_scale_after"] is None:
            if row["amp_scale_before"] is not None or row["amp_scale_after"] is not None or row["amp_reason"] not in {"AMP_DISABLED", "CPU_GRADSCALER_SYNTHETIC"}:
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "AMP nullable fields have inconsistent reasons")
        elif row["amp_reason"] != "PRESENT":
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Present AMP scales require PRESENT reason")


def write_step_partition(rows: Sequence[Mapping[str, Any]], path: str | Path):
    validate_step_rows(rows)
    run_id, epoch = partition_identity(path, required=True)
    if any(str(row["run_id"]) != run_id or int(row["epoch"]) != epoch for row in rows):
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Step row identity differs from its run/epoch partition")
    normalized = []
    for row in rows:
        value = dict(row)
        value["base_loss_items"] = {str(key): float(item) for key, item in dict(value["base_loss_items"]).items()}
        value["learning_rates"] = [float(item) for item in value["learning_rates"]]
        value["optimizer_hyperparameters"] = _normalize_optimizer_groups(value["optimizer_hyperparameters"])
        normalized.append(value)
    return write_zstd_parquet(
        normalized,
        path,
        schema_version="stage1.sctsr.optimizer_step_ledger.v1",
        schema=STEP_SCHEMA,
        require_run_epoch_partition=True,
    )
