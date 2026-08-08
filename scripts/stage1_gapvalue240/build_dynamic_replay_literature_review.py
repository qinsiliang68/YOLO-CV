"""Build the registered primary-source literature evidence snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_gapvalue240.campaign_literature_core import build_deep_read_records
from stage1_gapvalue240.campaign_literature_pipeline import publish_campaign_literature


CAMPAIGN_ID = "dynamic_replay_budget_efficiency_20260807"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "stage1_sample_value_experiments"
    / "experiments"
    / CAMPAIGN_ID
    / "02_literature"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish the traceable Stage1 dynamic-replay literature review."
    )
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-count", type=int, default=155)
    parser.add_argument("--method-target", type=int, default=55)
    parser.add_argument("--raw-results", type=int, default=1900)
    parser.add_argument("--deduplicated-results", type=int, default=1403)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = publish_campaign_literature(
        args.candidate_source,
        args.output_dir,
        campaign_id=CAMPAIGN_ID,
        core_records=build_deep_read_records(),
        target_count=args.target_count,
        method_target=args.method_target,
        min_screened=150,
        min_method=50,
        min_deep=20,
        discovery_counts={
            "raw_results": args.raw_results,
            "deduplicated": args.deduplicated_results,
        },
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
