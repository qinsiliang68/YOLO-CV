#!/usr/bin/env python3
"""Validate a hash-bound primary-source PDF inventory and publish its evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.literature_source_inventory_v2 import (  # noqa: E402
    validate_source_inventory,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--acquisition-ledger", type=Path, action="append", required=True)
    parser.add_argument("--expected-ledger", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output-inventory", type=Path, required=True)
    parser.add_argument("--output-validation", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = validate_source_inventory(
        corpus_root=args.corpus_root,
        acquisition_ledgers=args.acquisition_ledger,
        expected_ledger=args.expected_ledger,
        expected_count=args.expected_count,
    )
    args.output_inventory.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(result.rows[0])
    temporary_csv = args.output_inventory.with_suffix(args.output_inventory.suffix + ".tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.rows)
    temporary_csv.replace(args.output_inventory)

    validation = {
        "schema_version": "1.0",
        "status": result.status,
        "expected_count": result.expected_count,
        "verified_count": result.verified_count,
        "inventory_path": args.output_inventory.as_posix(),
        "formal_deep_reading_credit_granted": False,
        "note": (
            "Source-byte validation is not reading evidence. Papers receive DEEP credit only after "
            "their v2 notes and page anchors pass the corpus validator."
        ),
    }
    args.output_validation.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = args.output_validation.with_suffix(args.output_validation.suffix + ".tmp")
    temporary_json.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_json.replace(args.output_validation)
    print(f"status={result.status} verified={result.verified_count}/{result.expected_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
