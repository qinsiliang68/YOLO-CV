#!/usr/bin/env python3
"""Build the validated BROAD-500 staging corpus without publishing formal credit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.literature_broad_staging_v2 import (  # noqa: E402
    build_broad_staging,
)
from stage1_dynamic_replay_v3.literature_anchor_contract_v2 import (  # noqa: E402
    load_validated_anchor_work_ids,
)
from stage1_dynamic_replay_v3.literature_tier_freeze_v2 import (  # noqa: E402
    TierSelectionPolicy,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--batch-start", type=int, default=1)
    parser.add_argument("--batch-end", type=int, default=23)
    parser.add_argument(
        "--batch-number",
        type=int,
        action="append",
        default=[],
        help="Additional non-contiguous reviewed batch; repeat as needed.",
    )
    parser.add_argument("--anchor-contract", type=Path, required=True)
    parser.add_argument("--broad-membership", type=Path, required=True)
    parser.add_argument(
        "--merge-ledger",
        type=Path,
        default=Path("discovery/CANONICAL_MERGES_v2.csv"),
        help="Corpus-relative, immutable explicit version-merge ledger.",
    )
    parser.add_argument("--expected-anchor-count", type=int, default=40)
    parser.add_argument(
        "--output-relative",
        type=Path,
        default=Path("staging/broad_freeze_v2"),
    )
    parser.add_argument("--total", type=int, default=500)
    parser.add_argument("--minimum-per-rq", type=int, default=40)
    parser.add_argument("--maximum-per-rq", type=int, default=100)
    parser.add_argument("--maximum-transfer", type=int, default=100)
    parser.add_argument(
        "--frozen-seed",
        default="stage1-literature-tier-freeze-v2-20260810",
    )
    parser.add_argument("--replace-existing", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.batch_start < 1 or args.batch_end < args.batch_start:
        raise SystemExit("invalid batch range")
    if any(number < 1 for number in args.batch_number):
        raise SystemExit("additional batch numbers must be positive")
    batch_numbers = tuple(
        sorted(set(range(args.batch_start, args.batch_end + 1)) | set(args.batch_number))
    )
    anchor_work_ids = load_validated_anchor_work_ids(
        args.anchor_contract,
        expected_count=args.expected_anchor_count,
        broad_membership_path=args.broad_membership,
    )
    policy = TierSelectionPolicy(
        total=args.total,
        minimum_per_rq=args.minimum_per_rq,
        maximum_per_rq=args.maximum_per_rq,
        maximum_transfer=args.maximum_transfer,
        mandatory_canonical_work_ids=anchor_work_ids,
        frozen_seed=args.frozen_seed,
    )
    result = build_broad_staging(
        args.corpus_root,
        batch_numbers=batch_numbers,
        policy=policy,
        merge_ledger_path=args.merge_ledger,
        policy_source_paths=(args.anchor_contract, args.broad_membership),
        output_relative=args.output_relative,
        replace_existing=args.replace_existing,
    )
    print(
        f"status={result.status} selected={result.selected_count} "
        f"reserves={result.reserve_count} formal_broad_increment={result.formal_broad_increment} "
        f"anchors={len(anchor_work_ids)} output={result.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
