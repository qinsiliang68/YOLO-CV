from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .columnar import partition_identity, write_zstd_parquet
from .errors import ErrorCode, SctsrError
from .ledger_schema import OCCURRENCE_SCHEMA

REQUIRED_FIELDS = frozenset(OCCURRENCE_SCHEMA.names)
OOF_GROUP_SEMANTIC = "FILENAME_BUCKET_SURROGATE_NOT_TRUE_VIDEO_ID"
_INVALID_TEXT = {"", "unknown", "UNKNOWN", "待补", "同上", "TODO", "TBD"}
_NULL_OOF_REASONS = {"REGISTERED_NOT_AVAILABLE", "REGISTERED_NOT_AVAILABLE_SYNTHETIC"}
_NULL_RHO_REASONS = {"REGISTERED_NOT_REPORTED", "REGISTERED_NOT_AVAILABLE"}


def _reject_invalid_text(row: Mapping[str, Any], index: int) -> None:
    for field in OCCURRENCE_SCHEMA:
        value = row[field.name]
        if isinstance(value, str) and value.strip() in _INVALID_TEXT:
            raise SctsrError(
                ErrorCode.SCHEMA_VALIDATION_FAILED,
                "Occurrence ledger contains an empty or unregistered placeholder",
                failing_field=f"row[{index}].{field.name}",
                observed=value,
            )


def _validate_nullable(value: Any, reason: str, *, present_reason: str, null_reasons: set[str], field: str) -> None:
    if value is None and reason not in null_reasons:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Nullable occurrence value lacks a registered null reason", failing_field=field, observed=reason)
    if value is not None and reason != present_reason:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Present occurrence value must use PRESENT reason", failing_field=field, observed=reason)


def validate_occurrence_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Occurrence partition may not be empty")
    for index, row in enumerate(rows):
        observed = set(row)
        if observed != REQUIRED_FIELDS:
            raise SctsrError(
                ErrorCode.SCHEMA_VALIDATION_FAILED,
                "Occurrence ledger row fields do not exactly match the taskbook schema",
                failing_field=f"row[{index}]",
                observed={"missing": sorted(REQUIRED_FIELDS - observed), "extra": sorted(observed - REQUIRED_FIELDS)},
            )
        _reject_invalid_text(row, index)
        role = row["occurrence_role"]
        if role not in {"BASE", "REPLAY"}:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Invalid occurrence role")
        epoch = int(row["epoch"])
        if not 1 <= epoch <= 200:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Occurrence epoch out of range")
        if not 0 <= int(row["base_batch_index"]) <= 937:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Occurrence base batch index out of range")
        if int(row["y_true"]) not in {0, 1} or int(row["predicted_label_argmax"]) not in {0, 1}:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Occurrence labels must be binary")
        if row["oof_group_semantic"] != OOF_GROUP_SEMANTIC:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "OOF group semantic is incorrect")
        if not 0 <= int(row["oof_fold"]) <= 9:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "OOF fold is outside 0..9")
        if not 0 <= int(row["augmentation_seed"]) < 2**64:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Augmentation seed is outside uint64")
        if len(str(row["augmentation_trace_digest"])) != 64:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Augmentation trace digest must be SHA-256")
        for field in ("logit_normal", "logit_defect", "p_defect_raw", "ce_unreduced", "margin_defect_minus_normal"):
            if not math.isfinite(float(row[field])):
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Occurrence numeric value is not finite", failing_field=field)
        if not 0.0 <= float(row["p_defect_raw"]) <= 1.0 or float(row["ce_unreduced"]) < 0.0:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Occurrence probability or CE is outside its valid range")
        _validate_nullable(
            row["oof_reference_probability"], str(row["oof_reference_reason"]),
            present_reason="PRESENT", null_reasons=_NULL_OOF_REASONS, field="oof_reference_probability",
        )
        rho = row["rho_candidate_signal"]
        rho_reason = str(row["rho_reason"])
        if rho is None and rho_reason not in _NULL_RHO_REASONS:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Null RHO requires a registered reason")
        if rho is not None and rho_reason != "CANDIDATE_SIGNAL_NOT_UTILITY":
            raise SctsrError(ErrorCode.CANDIDATE_SIGNAL_AS_UTILITY_FORBIDDEN, "RHO candidate signal must be marked non-utility")

        before = int(row["replay_count_before"])
        after = int(row["replay_count_after"])
        cumulative_before = int(row["cumulative_replay_count_before"])
        cumulative_after = int(row["cumulative_replay_count_after"])
        last_epoch = row["last_replay_epoch"]
        since = row["epochs_since_last_replay"]
        if role == "BASE":
            expected = {
                "replay_role": "NOT_APPLICABLE_BASE",
                "identity_pool_id": "NOT_APPLICABLE_BASE",
                "identity_group": "NOT_APPLICABLE_BASE",
                "last_replay_epoch_reason": "NOT_APPLICABLE_BASE",
                "epochs_since_last_replay_reason": "NOT_APPLICABLE_BASE",
                "planned_replay_epoch_reason": "NOT_APPLICABLE_BASE",
                "planned_step_slot_reason": "NOT_APPLICABLE_BASE",
                "schedule_family": "BASE_CANONICAL",
                "fallback_state": "NOT_APPLICABLE_BASE",
            }
            for field, value in expected.items():
                if row[field] != value:
                    raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "BASE occurrence uses invalid replay sentinel", failing_field=field, observed=row[field], expected=value)
            if before != 0 or after != 0 or last_epoch is not None or since is not None:
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "BASE occurrence must not claim replay history")
            if row["planned_replay_epoch"] is not None or row["planned_step_slot"] is not None or int(row["pool_multiplicity_target"]) != 0:
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "BASE occurrence must not claim a replay plan")
            if cumulative_before != cumulative_after:
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "BASE occurrence may not advance replay cumulative count")
        else:
            if after != before + 1 or cumulative_after != cumulative_before + 1:
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "REPLAY occurrence counts must advance exactly once")
            if row["planned_replay_epoch"] != epoch or int(row["planned_step_slot"]) != int(row["base_batch_index"]):
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "REPLAY occurrence differs from its materialized epoch/step slot")
            if row["planned_replay_epoch_reason"] != "PRESENT" or row["planned_step_slot_reason"] != "PRESENT":
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "REPLAY plan fields require PRESENT reasons")
            if int(row["pool_multiplicity_target"]) <= 0:
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "REPLAY occurrence lacks a positive pool multiplicity target")
            if last_epoch is None:
                if row["last_replay_epoch_reason"] != "NEVER_REPLAYED" or since is not None or row["epochs_since_last_replay_reason"] != "NEVER_REPLAYED":
                    raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "First replay occurrence uses inconsistent history reasons")
            else:
                expected_since = epoch - int(last_epoch)
                if row["last_replay_epoch_reason"] != "PRESENT" or row["epochs_since_last_replay_reason"] != "PRESENT" or int(since) != expected_since or expected_since < 0:
                    raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Replay exposure history is inconsistent")


def write_occurrence_partition(rows: Sequence[Mapping[str, Any]], path: str | Path):
    validate_occurrence_rows(rows)
    run_id, epoch = partition_identity(path, required=True)
    if any(str(row["run_id"]) != run_id or int(row["epoch"]) != epoch for row in rows):
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Occurrence row identity differs from its run/epoch partition")
    return write_zstd_parquet(
        rows,
        path,
        schema_version="stage1.sctsr.occurrence_ledger.v1",
        schema=OCCURRENCE_SCHEMA,
        require_run_epoch_partition=True,
    )
