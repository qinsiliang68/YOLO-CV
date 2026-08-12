"""Publish current-v3 P0 review reproductions without running expert code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.review_reproduction import build_v3_p0_reproduction


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "stage1_sample_value_experiments"
    / "experiments"
    / "dynamic_replay_budget_efficiency_20260807"
    / "01_field_audit"
    / "expert_review_reproductions"
    / "local_v3_p0_20260809"
    / "V3_P0_REPRODUCTION.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the expert P0 findings against current v3.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    payload = build_v3_p0_reproduction(REPO_ROOT, parse_args().output)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
