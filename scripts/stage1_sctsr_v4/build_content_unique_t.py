from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.asset_registry import load_asset_registry
from stage1_sctsr_v4.dataset_content_ledger import load_registered_dataset_content_map
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.serialization import atomic_write_json, atomic_write_text, sha256_file, stable_digest
from stage1_sctsr_v4.t_content_repair import repair_t_content_duplicates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a content-unique SCTSR v4 T stress manifest without changing historical T")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--asset-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def _asset(registry, asset_id: str):
    records = [record for record in registry.assets if record.asset_id == asset_id]
    if len(records) != 1:
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Required T repair asset is not unique", observed=asset_id)
    return records[0]


def _role_identity_digest(rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: (str(item["replay_role"]), str(item["sample_id"]))):
        digest.update(str(row["replay_role"]).encode("utf-8"))
        digest.update(b"\t")
        digest.update(str(row["sample_id"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    output = args.output.resolve()
    receipt_path = args.receipt.resolve()
    if output.exists() or receipt_path.exists():
        raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Derived T outputs are immutable")
    registry = load_asset_registry(args.asset_registry)
    t_asset = _asset(registry, "t_stress_manifest")
    value_asset = _asset(registry, "sample_value_table")
    oof_asset = _asset(registry, "oof_assignments")
    t_path = (root / t_asset.relative_path).resolve()
    with t_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        t_rows = list(reader)
    oof = (
        pl.scan_csv(root / oof_asset.relative_path)
        .select(pl.col("canonical_image_relpath").alias("sample_id"), "oof_group_id")
        .collect()
    )
    candidates = (
        pl.scan_csv(root / value_asset.relative_path)
        .select(
            "sample_id",
            "y_true",
            "oof_fold",
            "dynamic_bucket",
            "mean_p_defect",
            "correct_rate",
            "std_p_defect",
            "gap_critical_score",
        )
        .collect()
        .join(oof, on="sample_id", how="inner", validate="1:1")
    )
    oof_groups = dict(oof.select("sample_id", "oof_group_id").iter_rows())
    content = load_registered_dataset_content_map(registry=registry, repository_root=root)
    repaired, audit = repair_t_content_duplicates(
        t_rows,
        candidate_rows=list(candidates.iter_rows(named=True)),
        oof_groups=oof_groups,
        content_by_id=content,
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(repaired)
    try:
        derived_path = output.relative_to(root).as_posix()
    except ValueError as exc:
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Derived T output must remain inside repository root") from exc
    atomic_write_text(output, buffer.getvalue())
    core = {
        "schema_version": "stage1.sctsr.t_content_repair.v1",
        "status": "BUILT_NOT_FORMAL_TRAINING",
        "historical_t_path": t_asset.relative_path,
        "historical_t_sha256": t_asset.sha256,
        "derived_t_path": derived_path,
        "derived_t_bytes": output.stat().st_size,
        "derived_t_sha256": sha256_file(output),
        "row_count": len(repaired),
        "identity_digest_algorithm": "SHA256_SORTED_REPLAY_ROLE_TAB_SAMPLE_ID_LF",
        "identity_digest": _role_identity_digest(repaired),
        "content_unique_count": len({content[str(row["sample_id"])]["image_sha256"] for row in repaired}),
        "repair_audit": audit,
        "formal_training_started": False,
        "blind_holdout_opened": False,
        "test_accessed": False,
    }
    report = {**core, "receipt_digest": stable_digest(core)}
    atomic_write_json(receipt_path, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
