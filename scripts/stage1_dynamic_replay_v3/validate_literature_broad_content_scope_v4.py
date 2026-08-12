#!/usr/bin/env python3
"""Publish a fail-closed BROAD content-scope and canonical-version audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.literature_broad_content_scope_v4 import (  # noqa: E402
    audit_broad_content_scope,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=500)
    parser.add_argument(
        "--report",
        type=Path,
        help=(
            "Output JSON; defaults to "
            "<corpus-root>/validation/BROAD_CONTENT_SCOPE_VALIDATION_v4.json"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = audit_broad_content_scope(
        args.corpus_root,
        expected_count=args.expected_count,
    )
    output = args.report or (
        args.corpus_root / "validation" / "BROAD_CONTENT_SCOPE_VALIDATION_v4.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    payload = report.as_dict()
    counts = payload["counts"]
    print(
        f"status={report.status} promotion_allowed={str(report.promotion_allowed).lower()} "
        f"observed={report.observed_count} fail_findings={counts['fail_findings']} "
        f"review_required_findings={counts['review_required_findings']} report={output}"
    )
    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
