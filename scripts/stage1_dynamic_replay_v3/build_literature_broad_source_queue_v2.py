#!/usr/bin/env python3
"""Build a deterministic broad-screen primary-source acquisition queue."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.literature_broad_source_queue_v2 import (  # noqa: E402
    build_broad_source_requests,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-subdir", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    with args.input_queue.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    requests = build_broad_source_requests(rows, source_subdir=args.source_subdir)
    if not requests:
        raise ValueError("input queue produced no source requests")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(requests[0]))
        writer.writeheader()
        writer.writerows(requests)
    os.replace(temporary, args.output)
    full_text = sum(row["full_text_claimed"] == "true" for row in requests)
    print(f"requests={len(requests)} full_text_targets={full_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
