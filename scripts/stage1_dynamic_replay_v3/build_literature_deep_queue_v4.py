#!/usr/bin/env python3
"""Freeze the provisional 100-paper DEEP full-text review queue."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.literature_deep_queue_v4 import (  # noqa: E402
    build_deep_review_queue,
)
from stage1_dynamic_replay_v3.literature_tier_freeze_v2 import (  # noqa: E402
    TierSelectionPolicy,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--canonical-registry", type=Path, required=True)
    parser.add_argument("--screened-primary", type=Path, required=True)
    parser.add_argument("--core-anchors", type=Path, required=True)
    parser.add_argument("--screened-review-dir", type=Path, required=True)
    parser.add_argument("--legacy-fulltext-ledger", type=Path, required=True)
    parser.add_argument("--legacy-note-dir", type=Path)
    parser.add_argument(
        "--output-relative",
        type=Path,
        default=Path("staging/deep_review_queue_v4"),
    )
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--minimum-per-rq", type=int, default=10)
    parser.add_argument("--maximum-per-rq", type=int, default=20)
    parser.add_argument("--maximum-transfer", type=int, default=35)
    parser.add_argument("--minimum-counterevidence-per-rq", type=int, default=1)
    parser.add_argument(
        "--frozen-seed",
        default="stage1-literature-deep-review-v4-20260810",
    )
    parser.add_argument("--replace-existing", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    policy = TierSelectionPolicy(
        total=args.total,
        minimum_per_rq=args.minimum_per_rq,
        maximum_per_rq=args.maximum_per_rq,
        maximum_transfer=args.maximum_transfer,
        minimum_counterevidence_per_rq=args.minimum_counterevidence_per_rq,
        mandatory_canonical_work_ids=(),
        tier_label="DEEP",
        frozen_seed=args.frozen_seed,
    )
    result = build_deep_review_queue(
        args.corpus_root,
        canonical_registry_path=args.canonical_registry,
        screened_primary_path=args.screened_primary,
        core_anchors_path=args.core_anchors,
        screened_review_dir=args.screened_review_dir,
        legacy_fulltext_ledger_path=args.legacy_fulltext_ledger,
        legacy_note_dir=args.legacy_note_dir,
        output_relative=args.output_relative,
        policy=policy,
        replace_existing=args.replace_existing,
    )
    print(
        f"status=PASS selected={result.selected_count} reserves={result.reserve_count} "
        f"mandatory={result.mandatory_count} ready={result.ready_union_count} "
        f"output={result.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
