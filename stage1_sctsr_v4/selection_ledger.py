from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .columnar import partition_identity, write_zstd_parquet
from .errors import ErrorCode, SctsrError
from .ledger_schema import SELECTION_SCHEMA
from .terminal_field_guard import FORBIDDEN_FIELDS

REQUIRED_FIELDS = frozenset(SELECTION_SCHEMA.names)


def validate_selection_rows(rows: Sequence[Mapping[str, Any]], *, r2: bool = False) -> None:
    if not rows:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Selection ledger must preserve a non-empty candidate universe")
    selected_ids: set[str] = set()
    candidate_ids: set[str] = set()
    for index, row in enumerate(rows):
        if r2:
            forbidden = set(row) & set(FORBIDDEN_FIELDS)
            if forbidden:
                raise SctsrError(ErrorCode.TERMINAL_FIELD_ACCESS_FORBIDDEN, "R2 selection ledger contains terminal fields", observed=sorted(forbidden))
        observed = set(row)
        if observed != REQUIRED_FIELDS:
            raise SctsrError(
                ErrorCode.SCHEMA_VALIDATION_FAILED,
                "Selection ledger row fields do not exactly match the taskbook schema",
                failing_field=f"row[{index}]",
                observed={"missing": sorted(REQUIRED_FIELDS - observed), "extra": sorted(observed - REQUIRED_FIELDS)},
            )
        sample_id = str(row["candidate_sample_id"])
        if sample_id in candidate_ids:
            raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "Selection candidate universe contains a duplicate identity", observed=sample_id)
        candidate_ids.add(sample_id)
        if any(isinstance(value, str) and value.strip() in {"", "unknown", "UNKNOWN", "TODO", "TBD", "同上"} for value in row.values()):
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Selection ledger contains an invalid placeholder")
        if bool(row["selected"]):
            selected_ids.add(sample_id)
            if not bool(row["eligibility"]):
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Ineligible selection candidate was selected")
            if row["selected_pool"] == "NOT_SELECTED":
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Selected candidate has NOT_SELECTED pool")
        elif row["selected_pool"] != "NOT_SELECTED":
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Unselected candidate claims a selected pool")
        if int(row["stratum_quota_required"]) < 0 or int(row["stratum_quota_available"]) < 0:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Selection quota counts may not be negative")
        if len(str(row["selection_counter_hash"])) != 64 or len(str(row["source_row_asset_sha256"])) != 64:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Selection ledger digest field is not SHA-256")
        if r2:
            if row["terminal_field_status"] != "TERMINAL_FIELDS_NOT_LOADED" or row["duplicate_overlap_status"] != "ZERO_OVERLAP":
                raise SctsrError(ErrorCode.TERMINAL_FIELD_ACCESS_FORBIDDEN, "R2 selection evidence does not prove terminal isolation and zero overlap")
            if len(str(row["terminal_field_guard_digest"])) != 64:
                raise SctsrError(ErrorCode.TERMINAL_FIELD_ACCESS_FORBIDDEN, "R2 terminal-field guard digest is missing")


def write_selection_partition(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    r2: bool = False,
    policy: str | None = None,
):
    r2 = r2 or policy == "R2_MATCHED_RANDOM"
    validate_selection_rows(rows, r2=r2)
    run_id, epoch = partition_identity(path, required=True)
    if epoch != 0:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Selection construction evidence must use epoch=0000 partition", observed=epoch)
    return write_zstd_parquet(
        rows,
        path,
        schema_version="stage1.sctsr.selection_ledger.v1",
        schema=SELECTION_SCHEMA,
        require_run_epoch_partition=True,
    )
