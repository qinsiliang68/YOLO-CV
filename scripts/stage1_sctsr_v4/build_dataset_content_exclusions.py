from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.asset_registry import load_asset_registry
from stage1_sctsr_v4.columnar import read_columnar
from stage1_sctsr_v4.dataset_content_ledger import DATASET_CONTENT_ASSET_ID
from stage1_sctsr_v4.dataset_disjointness import write_content_exclusion_assets
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.serialization import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the deterministic SCTSR evaluation content-exclusion overlay")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--asset-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = args.repository_root.resolve()
    registry = load_asset_registry(args.asset_registry)
    records = [record for record in registry.assets if record.asset_id == DATASET_CONTENT_ASSET_ID]
    if len(records) != 1:
        raise SctsrError(
            ErrorCode.DATASET_CONTENT_LEDGER_REQUIRED,
            "Asset registry must contain exactly one dataset content ledger",
        )
    record = records[0]
    ledger = (repository / record.relative_path).resolve()
    try:
        ledger.relative_to(repository)
    except ValueError as exc:
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Dataset content ledger escapes repository") from exc
    if not ledger.is_file() or ledger.stat().st_size != record.bytes or sha256_file(ledger) != record.sha256:
        raise SctsrError(
            ErrorCode.DATASET_CONTENT_MISMATCH,
            "Dataset content ledger bytes differ from the registered source",
            artifact_path=str(ledger),
        )
    report = write_content_exclusion_assets(
        read_columnar(ledger),
        output_path=args.output,
        receipt_path=args.receipt,
        source_ledger_sha256=record.sha256,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
