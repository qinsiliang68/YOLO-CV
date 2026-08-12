#!/usr/bin/env python3
"""Build page-located SCREENED review packets from extracted full text."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.literature_screening_packet_v2 import (  # noqa: E402
    build_screening_packets,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--screening-queue", type=Path, required=True)
    parser.add_argument("--extraction-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replace-existing", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = build_screening_packets(
        corpus_root=args.corpus_root,
        screening_queue=args.screening_queue,
        extraction_ledger=args.extraction_ledger,
        output_root=args.output_root,
        replace_existing=args.replace_existing,
    )
    print(f"status=PASS output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
