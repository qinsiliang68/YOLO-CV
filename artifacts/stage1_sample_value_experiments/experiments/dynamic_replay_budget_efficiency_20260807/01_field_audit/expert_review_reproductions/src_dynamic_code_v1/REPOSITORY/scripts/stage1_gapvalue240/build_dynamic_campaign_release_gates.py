from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

from stage1_gapvalue240.campaign_controller import build_campaign_release_manifests
from stage1_gapvalue240.campaign_layout import CAMPAIGN_ID


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_campaign = (
        _BootstrapPath(__file__).resolve().parents[2]
        / "artifacts/stage1_sample_value_experiments/experiments"
        / CAMPAIGN_ID
    )
    parser = argparse.ArgumentParser(description="Freeze pilot and confirmatory campaign release gates.")
    parser.add_argument("--campaign-root", default=str(default_campaign))
    parser.add_argument("--pilot-seeds", nargs="+", default=["S001", "S002"])
    parser.add_argument(
        "--engineering-gate-report",
        required=True,
        help="PASS report from local real-data smoke, failure injection, and ten-machine canary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    campaign = Path(args.campaign_root).resolve()
    result = build_campaign_release_manifests(
        campaign / "04_run_queue",
        campaign / "04_run_queue/releases",
        campaign_id=CAMPAIGN_ID,
        pilot_seed_ids=tuple(args.pilot_seeds),
        engineering_gate_report=Path(args.engineering_gate_report).resolve(),
    )
    print(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "pilot_release": str(result.pilot_release),
                "confirmatory_hold": str(result.confirmatory_hold),
                "future_cycle_hold": str(result.future_cycle_hold),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
