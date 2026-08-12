#!/usr/bin/env python3
"""Build the verified-PDF source ledger for SCREENED text extraction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.literature_screened_text_ledger_v2 import (  # noqa: E402
    write_screened_text_ledger,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-queue", type=Path, required=True)
    parser.add_argument("--broad-staging-relative", required=True)
    parser.add_argument("--output-ledger", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    count = write_screened_text_ledger(
        args.screening_queue,
        broad_staging_relative=args.broad_staging_relative,
        output_ledger=args.output_ledger,
        output_receipt=args.output_receipt,
    )
    print(f"status=PASS sources={count} output={args.output_ledger.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
