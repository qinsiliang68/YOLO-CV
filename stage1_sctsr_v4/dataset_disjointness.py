from __future__ import annotations

import csv
import io
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import ErrorCode, SctsrError
from .serialization import atomic_write_json, atomic_write_text, sha256_file, stable_digest


CONTENT_EXCLUSION_SCHEMA_VERSION = "stage1.sctsr.dataset_content_exclusions.v1"
CONTENT_EXCLUSION_FIELDS = (
    "sample_id",
    "split_role",
    "y_true",
    "image_sha256",
    "retained_sample_id",
    "retained_split_role",
    "reason",
)

_ROLE_PRIORITY = {"base": 0, "val_model": 1, "val_cal": 2, "val_op": 3}


def scientific_split_role(value: str) -> str:
    role = str(value).strip().lower()
    if role in {"train", "normal_train"}:
        return "base"
    for candidate in ("val_model", "val_cal", "val_op"):
        if role in {candidate, f"normal_{candidate}"}:
            return candidate
    raise SctsrError(
        ErrorCode.DATASET_CONTENT_MISMATCH,
        "Dataset content row has an unknown SCTSR split role",
        observed=value,
    )


def _semantic_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    required = {"sample_id", "split_role", "y_true", "image_sha256"}
    if not required.issubset(raw):
        raise SctsrError(
            ErrorCode.SCHEMA_VALIDATION_FAILED,
            "Dataset disjointness input lacks required content fields",
            observed=sorted(set(raw)),
            expected=sorted(required),
        )
    sample_id = str(raw["sample_id"]).replace("\\", "/").strip()
    split_role = str(raw["split_role"]).strip()
    label = int(raw["y_true"])
    image_sha = str(raw["image_sha256"]).upper()
    if not sample_id or label not in {0, 1} or len(image_sha) != 64:
        raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Dataset disjointness row is malformed", observed=sample_id)
    return {
        "sample_id": sample_id,
        "split_role": split_role,
        "scientific_role": scientific_split_role(split_role),
        "y_true": label,
        "image_sha256": image_sha,
    }


def derive_content_exclusion_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Derive the only allowed effective-evaluation exclusion set.

    Historical base occurrences are immutable. Equal image bytes inside base
    are reported but retained. Evaluation occurrences colliding with base are
    excluded. Evaluation-only collisions retain exactly one row using the
    frozen priority val_model > val_cal > val_op, then canonical sample ID.
    """

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for raw in rows:
        row = _semantic_row(raw)
        if row["sample_id"] in seen_ids:
            raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "Disjointness input repeats a sample ID", observed=row["sample_id"])
        seen_ids.add(row["sample_id"])
        groups[row["image_sha256"]].append(row)

    exclusions: list[dict[str, Any]] = []
    duplicate_groups = 0
    cross_role_groups = 0
    base_internal_groups = 0
    for image_sha, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        duplicate_groups += 1
        roles = {row["scientific_role"] for row in members}
        if len(roles) > 1:
            cross_role_groups += 1
        base_members = sorted(
            (row for row in members if row["scientific_role"] == "base"),
            key=lambda row: row["sample_id"],
        )
        if base_members:
            retained = base_members[0]
            if len(base_members) > 1:
                base_internal_groups += 1
            excluded = [row for row in members if row["scientific_role"] != "base"]
        else:
            ordered = sorted(
                members,
                key=lambda row: (_ROLE_PRIORITY[row["scientific_role"]], row["sample_id"]),
            )
            retained = ordered[0]
            excluded = ordered[1:]
        for row in excluded:
            reason = (
                "CONTENT_SHA_DUPLICATE_WITHIN_EVALUATION_ROLE"
                if row["scientific_role"] == retained["scientific_role"]
                else "CONTENT_SHA_COLLIDES_WITH_HIGHER_PRIORITY_ROLE"
            )
            exclusions.append(
                {
                    "sample_id": row["sample_id"],
                    "split_role": row["split_role"],
                    "y_true": row["y_true"],
                    "image_sha256": image_sha,
                    "retained_sample_id": retained["sample_id"],
                    "retained_split_role": retained["split_role"],
                    "reason": reason,
                }
            )
    exclusions.sort(key=lambda row: row["sample_id"])
    audit = {
        "schema_version": CONTENT_EXCLUSION_SCHEMA_VERSION,
        "policy": "HISTORICAL_BASE_THEN_VAL_MODEL_THEN_VAL_CAL_THEN_VAL_OP_V1",
        "raw_rows": len(rows),
        "raw_unique_image_sha256": len(groups),
        "duplicate_content_groups": duplicate_groups,
        "cross_scientific_role_groups": cross_role_groups,
        "base_internal_duplicate_groups": base_internal_groups,
        "excluded_evaluation_occurrences": len(exclusions),
        "exclusion_digest": stable_digest(exclusions),
    }
    return exclusions, audit


def write_content_exclusion_assets(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: str | Path,
    receipt_path: str | Path,
    source_ledger_sha256: str,
) -> dict[str, Any]:
    """Write one immutable LF-normalized exclusion CSV and provenance receipt."""

    output = Path(output_path)
    receipt = Path(receipt_path)
    if output.exists() or receipt.exists():
        raise SctsrError(
            ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE,
            "Content exclusion assets are immutable; choose fresh output paths",
            observed={"output_exists": output.exists(), "receipt_exists": receipt.exists()},
        )
    ledger_sha = str(source_ledger_sha256).upper()
    if len(ledger_sha) != 64 or any(character not in "0123456789ABCDEF" for character in ledger_sha):
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Source dataset ledger SHA-256 is invalid")
    exclusions, audit = derive_content_exclusion_rows(rows)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CONTENT_EXCLUSION_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(exclusions)
    atomic_write_text(output, buffer.getvalue())
    core = {
        "schema_version": CONTENT_EXCLUSION_SCHEMA_VERSION,
        "status": "BUILT_NOT_FORMAL_TRAINING",
        "source_ledger_sha256": ledger_sha,
        "relative_or_absolute_path": output.as_posix(),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "row_count": len(exclusions),
        "line_ending": "LF",
        "policy": audit["policy"],
        "audit": audit,
        "formal_training_started": False,
        "blind_holdout_opened": False,
        "test_accessed": False,
    }
    report = {**core, "receipt_digest": stable_digest(core)}
    atomic_write_json(receipt, report)
    return report


def load_registered_content_exclusions(registry: Any, repository_root: str | Path) -> dict[str, dict[str, Any]]:
    asset_id = getattr(registry, "content_exclusion_asset_id", None)
    if not asset_id:
        return {}
    records = [record for record in registry.assets if record.asset_id == asset_id]
    if len(records) != 1:
        raise SctsrError(
            ErrorCode.DATASET_SPLIT_CONTENT_LEAKAGE,
            "Content exclusion registry must name exactly one asset",
            observed=asset_id,
        )
    record = records[0]
    root = Path(repository_root).resolve()
    path = (root / record.relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Content exclusion asset escapes repository", artifact_path=str(path)) from exc
    if not path.is_file() or path.stat().st_size != record.bytes or sha256_file(path) != record.sha256:
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Content exclusion asset bytes changed", artifact_path=str(path))
    output: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CONTENT_EXCLUSION_FIELDS:
            raise SctsrError(
                ErrorCode.SCHEMA_VALIDATION_FAILED,
                "Content exclusion CSV fields are not canonical",
                observed=reader.fieldnames,
                expected=CONTENT_EXCLUSION_FIELDS,
            )
        for index, raw in enumerate(reader, start=2):
            sample_id = str(raw["sample_id"]).replace("\\", "/")
            row = {
                **{field: str(raw[field]) for field in CONTENT_EXCLUSION_FIELDS},
                "sample_id": sample_id,
                "y_true": int(raw["y_true"]),
                "image_sha256": str(raw["image_sha256"]).upper(),
            }
            if sample_id in output or scientific_split_role(row["split_role"]) == "base":
                raise SctsrError(
                    ErrorCode.DATASET_SPLIT_CONTENT_LEAKAGE,
                    "Content exclusion rows must be unique evaluation identities",
                    failing_field=f"row[{index}]",
                    observed=sample_id,
                )
            if row["reason"] not in {
                "CONTENT_SHA_COLLIDES_WITH_HIGHER_PRIORITY_ROLE",
                "CONTENT_SHA_DUPLICATE_WITHIN_EVALUATION_ROLE",
            }:
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Content exclusion reason is not registered", observed=row["reason"])
            output[sample_id] = row
    if record.row_count is None or len(output) != record.row_count:
        raise SctsrError(
            ErrorCode.ASSET_VALIDATION_FAILED,
            "Content exclusion row count differs from registry",
            observed=len(output),
            expected=record.row_count,
        )
    return output
