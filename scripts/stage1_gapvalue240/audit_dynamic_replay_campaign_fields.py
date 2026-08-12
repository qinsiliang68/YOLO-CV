from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_gapvalue240.campaign_field_audit import publish_campaign_field_audit


# Historical evidence namespace only.  The old campaign runtime/layout module was
# retired after the completed 240-run study; this audit does not activate it.
CAMPAIGN_ID = "dynamic_replay_budget_efficiency_20260807"


OLD_REPORT_ROOT = REPO_ROOT / (
    "artifacts/stage1_sample_value_experiments/experiments/"
    "oof_dynamics_gap_value_20260708/06_reports/"
    "gapvalue240_goal_analysis_20260806_v1"
)
OOF_EXPERIMENT_ROOT = REPO_ROOT / (
    "artifacts/stage1_sample_value_experiments/experiments/"
    "oof_dynamics_gap_value_20260708"
)
CAMPAIGN_ROOT = REPO_ROOT / (
    "artifacts/stage1_sample_value_experiments/experiments/"
    f"{CAMPAIGN_ID}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish field sufficiency and storage audit for dynamic replay."
    )
    parser.add_argument(
        "--refined-ledger",
        type=Path,
        default=OLD_REPORT_ROOT / "audit/DATA_USAGE_LEDGER_REFINED.csv",
    )
    parser.add_argument(
        "--source-file-ledger",
        type=Path,
        default=OLD_REPORT_ROOT / "audit/source_file_ledger.csv",
    )
    parser.add_argument(
        "--oof-experiment-root", type=Path, default=OOF_EXPERIMENT_ROOT
    )
    parser.add_argument(
        "--output-dir", type=Path, default=CAMPAIGN_ROOT / "01_field_audit"
    )
    parser.add_argument(
        "--seed-counts", type=int, nargs="+", default=[14, 22, 30]
    )
    parser.add_argument("--arm-count", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = publish_campaign_field_audit(
        refined_ledger_path=args.refined_ledger,
        source_file_ledger_path=args.source_file_ledger,
        oof_experiment_root=args.oof_experiment_root,
        output_dir=args.output_dir,
        campaign_id=CAMPAIGN_ID,
        seed_counts=tuple(args.seed_counts),
        arm_count=args.arm_count,
    )
    for key, value in result.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
