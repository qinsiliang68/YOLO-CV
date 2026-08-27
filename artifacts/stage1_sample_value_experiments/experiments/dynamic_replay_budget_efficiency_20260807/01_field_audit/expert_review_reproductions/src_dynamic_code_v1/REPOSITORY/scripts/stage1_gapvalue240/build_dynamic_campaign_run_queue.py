from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

from stage1_gapvalue240.campaign_layout import CAMPAIGN_ID
from stage1_gapvalue240.campaign_run_queue import build_campaign_run_queue


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo = _BootstrapPath(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Freeze the dynamic replay physical run queue.")
    parser.add_argument(
        "--campaign-root",
        default=str(
            repo
            / "artifacts/stage1_sample_value_experiments/experiments"
            / CAMPAIGN_ID
        ),
    )
    parser.add_argument(
        "--monitor-source",
        default=str(
            repo
            / "artifacts/stage1_sample_value_experiments/experiments"
            / "oof_dynamics_gap_value_20260708/02_sample_value_tables/sample_value_table.csv"
        ),
        help="OOF-only 120k sample-value table used to freeze process probes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    campaign = Path(args.campaign_root).resolve()
    result = build_campaign_run_queue(
        campaign / "03_preregistration",
        campaign / "04_run_queue",
        monitor_source=Path(args.monitor_source).resolve(),
    )
    print(
        json.dumps(
            {
                "queue_dir": str(result.queue_dir),
                "job_registry": str(result.job_registry),
                "monitor_manifest": str(result.monitor_manifest),
                "validation": str(result.validation_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
