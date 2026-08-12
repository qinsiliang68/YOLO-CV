#!/usr/bin/env python3
"""Publish the lossless Stage1 literature candidate universe v2."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.literature_candidate_universe_v2 import (  # noqa: E402
    publish_candidate_universe,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openalex", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--targeted", type=Path, required=True)
    parser.add_argument("--query-plan", type=Path, required=True)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--manual-queue", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = publish_candidate_universe(
        openalex_path=args.openalex,
        legacy_path=args.legacy,
        targeted_path=args.targeted,
        query_plan_path=args.query_plan,
        universe_path=args.universe,
        manual_queue_path=args.manual_queue,
        manifest_path=args.manifest,
    )
    print(
        f"candidate_versions={manifest['candidate_version_count']} "
        f"manual_review_groups={manifest['manual_review_group_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
