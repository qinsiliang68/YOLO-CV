from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

from stage1_gapvalue240.campaign_preregistration import build_campaign_preregistration
from stage1_gapvalue240.campaign_layout import CAMPAIGN_ID, active_preregistration_dir

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Freeze the Stage1 dynamic replay campaign preregistration.")
    parser.add_argument(
        "--campaign-root",
        default=str(
            repo
            / "artifacts/stage1_sample_value_experiments/experiments"
            / CAMPAIGN_ID
        ),
    )
    parser.add_argument(
        "--treatment-ranking-source",
        default=str(
            repo
            / "artifacts/stage1_sample_value_experiments/contracts/gapvalue240_v1_1"
            / "precomputed_direct_assets/rankings/GapCritical-Strict.csv"
        ),
    )
    parser.add_argument(
        "--canonical-lock",
        default=str(repo / "configs/stage1_gapvalue240/CANONICAL_TRAINING_LOCK_v1.json"),
    )
    parser.add_argument("--machine-count", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    campaign = Path(args.campaign_root).resolve()
    result = build_campaign_preregistration(
        active_preregistration_dir(campaign),
        treatment_ranking_source=Path(args.treatment_ranking_source).resolve(),
        canonical_lock_path=Path(args.canonical_lock).resolve(),
        machine_count=args.machine_count,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
