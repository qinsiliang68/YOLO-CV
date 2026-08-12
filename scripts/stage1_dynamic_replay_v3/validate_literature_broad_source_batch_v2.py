#!/usr/bin/env python3
"""Validate one broad-screen source batch and publish a fail-closed report."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.literature_broad_source_validation_v2 import (  # noqa: E402
    validate_broad_source_batch,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--acquisitions", type=Path, nargs="+", required=True)
    parser.add_argument("--failures", type=Path, nargs="+", required=True)
    parser.add_argument("--supersessions", type=Path, nargs="+")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = validate_broad_source_batch(
        corpus_root=args.corpus_root,
        queue_path=args.queue,
        acquisition_ledger=args.acquisitions,
        failure_ledger=args.failures,
        supersession_ledger=args.supersessions,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = args.output_csv.with_suffix(args.output_csv.suffix + ".tmp")
    if result.rows:
        with temporary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result.rows[0]))
            writer.writeheader()
            writer.writerows(result.rows)
    else:
        temporary_csv.write_text("paper_id\n", encoding="utf-8-sig")
    os.replace(temporary_csv, args.output_csv)
    payload = {
        "schema_version": "1.0",
        "status": result.status,
        "expected_count": result.expected_count,
        "verified_count": result.verified_count,
        "failed_count": result.failed_count,
        "missing_ids": list(result.missing_ids),
        "reading_credit_granted": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_json, args.output_json)
    print(
        f"status={result.status} expected={result.expected_count} "
        f"verified={result.verified_count} failed={result.failed_count}"
    )
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
