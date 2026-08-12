#!/usr/bin/env python3
"""Validate the Stage1 500/300/100 literature evidence corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.literature_evidence_v2 import (  # noqa: E402
    TierCounts,
    audit_completion,
    audit_corpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--expected-broad", type=int, default=500)
    parser.add_argument("--expected-screened", type=int, default=300)
    parser.add_argument("--expected-deep", type=int, default=100)
    parser.add_argument(
        "--skip-pdf-page-inspection",
        action="store_true",
        help="Development-only: verify PDF bytes/hash/signature but do not call pdfinfo.",
    )
    parser.add_argument(
        "--corpus-only",
        action="store_true",
        help="Development-only: skip discovery, source-acquisition, random-audit, and second-pass gates.",
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = TierCounts(
        broad=args.expected_broad,
        screened=args.expected_screened,
        deep=args.expected_deep,
    )
    if args.corpus_only:
        report = audit_corpus(
            args.corpus_root,
            expected=expected,
            inspect_pdf_pages=not args.skip_pdf_page_inspection,
        )
    else:
        report = audit_completion(
            args.corpus_root,
            expected=expected,
            inspect_pdf_pages=not args.skip_pdf_page_inspection,
        )
    output = args.report or args.corpus_root / "validation" / "CORPUS_VALIDATION.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = report.as_dict()
    payload["corpus_root"] = str(args.corpus_root.resolve())
    payload["pdf_page_inspection"] = not args.skip_pdf_page_inspection
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
