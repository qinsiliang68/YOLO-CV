#!/usr/bin/env python3
"""Validate the Stage1 core-method anchor contract without granting reading credit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.literature_anchor_contract_v2 import (  # noqa: E402
    build_anchor_source_expected_rows,
    validate_anchor_contract,
)


def _write_source_expected(path: Path, rows: tuple[dict[str, str], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("paper_id", "title"))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--broad-membership",
        type=Path,
        required=True,
        help="Frozen BROAD CSV used to bind anchors already admitted in v2.",
    )
    parser.add_argument("--expected-count", type=int, default=40)
    parser.add_argument(
        "--output-source-expected",
        type=Path,
        help="Atomically write identities whose primary PDFs must be acquired separately.",
    )
    args = parser.parse_args()
    result = validate_anchor_contract(
        args.contract,
        expected_count=args.expected_count,
        broad_membership_path=args.broad_membership,
    )
    source_expected_rows = build_anchor_source_expected_rows(
        args.contract,
        broad_membership_path=args.broad_membership,
    )
    if args.output_source_expected is not None:
        _write_source_expected(args.output_source_expected, source_expected_rows)
    print(
        json.dumps(
            {
                "status": result.status,
                "anchor_count": result.anchor_count,
                "status_counts": result.status_counts,
                "contract_sha256": result.contract_sha256,
                "formal_broad_increment": result.formal_broad_increment,
                "formal_screened_increment": result.formal_screened_increment,
                "formal_deep_increment": result.formal_deep_increment,
                "supplemental_source_expected_count": len(source_expected_rows),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
