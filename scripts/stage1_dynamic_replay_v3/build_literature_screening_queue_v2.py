#!/usr/bin/env python3
"""Build a provisional SCREENED reading queue from validated BROAD staging."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.literature_screening_queue_v2 import (  # noqa: E402
    build_screening_queue,
)
from stage1_dynamic_replay_v3.literature_anchor_contract_v2 import (  # noqa: E402
    load_validated_anchor_work_ids,
)
from stage1_dynamic_replay_v3.literature_tier_freeze_v2 import (  # noqa: E402
    TierSelectionPolicy,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broad-staging-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--anchor-contract", type=Path, required=True)
    parser.add_argument("--broad-membership", type=Path, required=True)
    parser.add_argument("--expected-anchor-count", type=int, default=40)
    parser.add_argument(
        "--method-source-overrides",
        type=Path,
        help="Hash-validated PDF overrides keyed by canonical_work_id.",
    )
    parser.add_argument("--total", type=int, default=300)
    parser.add_argument("--minimum-per-rq", type=int, default=25)
    parser.add_argument("--maximum-per-rq", type=int, default=60)
    parser.add_argument("--maximum-transfer", type=int, default=60)
    parser.add_argument("--minimum-counterevidence-per-rq", type=int, default=2)
    parser.add_argument("--reserve-read-count", type=int, default=60)
    parser.add_argument("--required-source-format", choices=("PDF", "HTML"))
    parser.add_argument(
        "--frozen-seed",
        default="stage1-literature-screened-queue-v2-20260810",
    )
    parser.add_argument("--replace-existing", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
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
        minimum_counterevidence_per_rq=args.minimum_counterevidence_per_rq,
        mandatory_canonical_work_ids=anchor_work_ids,
        tier_label="SCREENED",
        frozen_seed=args.frozen_seed,
    )
    result = build_screening_queue(
        args.broad_staging_root,
        output_root=args.output_root,
        policy=policy,
        reserve_read_count=args.reserve_read_count,
        required_source_format=args.required_source_format,
        policy_source_paths=(args.anchor_contract, args.broad_membership),
        method_source_override_path=args.method_source_overrides,
        replace_existing=args.replace_existing,
    )
    print(
        f"status={result.status} primary={result.primary_count} "
        f"reserves={result.reserve_count} reading_queue={result.reading_queue_count} "
        f"formal_screened_increment={result.formal_screened_increment} "
        f"anchors={len(anchor_work_ids)} output={result.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
