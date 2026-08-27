from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

from stage1_gapvalue240.campaign_gradient_queue import build_gradient_candidate_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen Treatment plus OOF-tail gradient diagnostic panel."
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    campaign = args.campaign_root.resolve()
    treatment = (
        campaign
        / "03_preregistration/frozen_selections/TREATMENT_GAPCRITICAL_NESTED.csv"
    )
    monitor = campaign / "04_run_queue/monitor/CAUSAL_MONITOR_SAMPLES.csv"
    queue_validation = json.loads(
        (campaign / "04_run_queue/RUN_QUEUE_VALIDATION.json").read_text(encoding="utf-8")
    )
    result = build_gradient_candidate_manifest(
        treatment,
        monitor,
        campaign / "04_run_queue/gradient",
        canonical_lock_file_sha256=queue_validation["canonical_lock_file_sha256"],
        expected_target_counts={"normal": 300, "defect": 300},
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "candidate_manifest": str(result.candidate_manifest),
                "union_sample_count": result.union_sample_count,
                "normal_target_count": result.normal_target_count,
                "defect_target_count": result.defect_target_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
