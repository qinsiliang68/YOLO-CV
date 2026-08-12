#!/usr/bin/env python3
"""Validate one manual-screening batch and publish a non-credit receipt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.literature_manual_screening_v2 import (  # noqa: E402
    merge_and_validate_manual_screening,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--decision-dir", type=Path, required=True)
    parser.add_argument(
        "--decision-file",
        type=Path,
        action="append",
        help="Exact decision CSV to validate; repeat for multiple files. Defaults to all CSVs.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _atomic_write_rows(path: Path, rows: tuple[dict[str, str], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = _parser().parse_args()
    result = merge_and_validate_manual_screening(
        queue_path=args.queue,
        decision_dir=args.decision_dir,
        decision_paths=args.decision_file,
    )
    _atomic_write_rows(args.output_csv, result.rows)

    decision_files = []
    selected_paths = (
        args.decision_file
        if args.decision_file is not None
        else sorted(args.decision_dir.glob("*.csv"), key=lambda item: item.name.casefold())
    )
    for path in sorted(selected_paths, key=lambda item: item.name.casefold()):
        decision_files.append(
            {
                "file_name": path.name,
                "row_count": _csv_row_count(path),
                "sha256": _sha256(path),
            }
        )
    eligible_ids = [row["queue_id"] for row in result.rows if row["decision"] == "ELIGIBLE_BROAD"]
    excluded_ids = [row["queue_id"] for row in result.rows if row["decision"] == "EXCLUDE"]
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "status": result.status,
        "evidence_stage": "MANUAL_SCREENED_CANDIDATE_BATCH",
        "reviewed_count": result.reviewed_count,
        "eligible_candidate_count": result.eligible_count,
        "excluded_count": result.excluded_count,
        "eligible_queue_ids": eligible_ids,
        "excluded_queue_ids": excluded_ids,
        "reading_credit_granted": False,
        "formal_broad_corpus_count_increment": 0,
        "credit_note": (
            "Validated manual screening creates eligible candidates only; formal BROAD_500 "
            "credit requires promotion into the exact deduplicated corpus."
        ),
        "queue_file_name": args.queue.name,
        "queue_sha256": _sha256(args.queue),
        "decision_files": decision_files,
        "merged_output_file_name": args.output_csv.name,
        "merged_output_sha256": _sha256(args.output_csv),
    }
    _atomic_write_json(args.output_json, payload)
    print(
        f"status={result.status} reviewed={result.reviewed_count} "
        f"eligible_candidates={result.eligible_count} excluded={result.excluded_count} "
        "formal_broad_increment=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
