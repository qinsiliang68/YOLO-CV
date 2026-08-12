#!/usr/bin/env python3
"""Extract hash-bound full text from a validated literature source batch."""

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

from stage1_dynamic_replay_v3.literature_text_extraction_v2 import (  # noqa: E402
    extract_literature_text_batch,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-ledger", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--minimum-characters", type=int, default=500)
    return parser


def _write_atomic(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = _parser().parse_args()
    result = extract_literature_text_batch(
        corpus_root=args.corpus_root,
        validation_path=args.validation,
        output_dir=args.output_dir,
        minimum_characters=args.minimum_characters,
    )
    if result.rows:
        args.output_ledger.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_ledger.with_suffix(args.output_ledger.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result.rows[0]))
            writer.writeheader()
            writer.writerows(result.rows)
        os.replace(temporary, args.output_ledger)
    payload = {
        "schema_version": "1.0",
        "status": result.status,
        "extracted_count": result.extracted_count,
        "reading_credit_granted": False,
    }
    _write_atomic(args.output_json, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"status={result.status} extracted={result.extracted_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
