#!/usr/bin/env python3
"""Promote validated supplemental anchors into one standard manual batch."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.literature_anchor_promotion_v2 import (  # noqa: E402
    promote_anchor_batch,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output-discovery-root", type=Path)
    parser.add_argument("--batch-number", type=int, default=24)
    args = parser.parse_args()
    result = promote_anchor_batch(
        args.corpus_root,
        output_discovery_root=args.output_discovery_root,
        batch_number=args.batch_number,
    )
    print(
        f"status={result.status} promoted={result.promoted_count} "
        f"formal_broad_increment={result.formal_broad_increment} "
        f"receipt={result.receipt_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
