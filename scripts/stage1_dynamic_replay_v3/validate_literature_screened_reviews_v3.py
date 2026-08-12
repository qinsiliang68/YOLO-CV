#!/usr/bin/env python3
"""Validate hash- and page-bound SCREENED full-text review records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.literature_screened_review_v3 import (  # noqa: E402
    validate_screened_review_records,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--screening-queue", type=Path, required=True)
    parser.add_argument("--extraction-ledger", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _read_records(directory: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    # The review directory may also contain migration receipts and manifests.
    # Only the contractually named one-paper records are scientific review inputs.
    for path in sorted(
        directory.glob("P[0-9][0-9][0-9][0-9].json"),
        key=lambda item: item.name.casefold(),
    ):
        value = json.loads(path.read_text(encoding="utf-8"))
        batch = value if isinstance(value, list) else [value]
        if not all(isinstance(item, dict) for item in batch):
            raise ValueError(f"review JSON must contain object(s): {path}")
        records.extend(batch)
        manifest.append(
            {
                "path": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "record_count": len(batch),
            }
        )
    if not manifest:
        raise ValueError(f"no review JSON files found in {directory}")
    return records, manifest


def main() -> int:
    args = _parser().parse_args()
    queue_rows = _read_csv(args.screening_queue)
    extraction_rows = _read_csv(args.extraction_ledger)
    records, review_manifest = _read_records(args.review_dir)
    result = validate_screened_review_records(
        corpus_root=args.corpus_root,
        queue_rows=queue_rows,
        extraction_rows=extraction_rows,
        records=records,
        require_queue_coverage=not args.allow_partial,
    )
    payload = {
        "schema_version": "3.0",
        "status": result.status,
        "reviewed_count": result.reviewed_count,
        "eligible_count": result.eligible_count,
        "excluded_count": result.excluded_count,
        "reviewed_paper_ids": [record["paper_id"] for record in result.records],
        "screening_queue_sha256": _sha256(args.screening_queue),
        "extraction_ledger_sha256": _sha256(args.extraction_ledger),
        "review_files": review_manifest,
        "full_queue_coverage_required": not args.allow_partial,
        "formal_screened_increment": 0,
        "reading_credit_note": (
            "This receipt validates full-text screening evidence, but formal SCREENED credit "
            "requires promotion into the exact nested 500/300/100 corpus and full corpus audit."
        ),
        "formal_training_started": False,
        "engineering_gate_generated": False,
        "blind_holdout_opened": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_json)
    print(
        f"status=PASS reviewed={result.reviewed_count} eligible={result.eligible_count} "
        f"excluded={result.excluded_count} formal_screened_increment=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
