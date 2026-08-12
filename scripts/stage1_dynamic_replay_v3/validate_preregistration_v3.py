#!/usr/bin/env python3
"""Validate the Stage1 Q/R/A/D preregistration without running experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.preregistration_v3 import (  # noqa: E402
    validate_preregistration_v3,
    write_validation_report,
)


DEFAULT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "stage1_sample_value_experiments"
    / "experiments"
    / "dynamic_replay_budget_efficiency_20260807"
    / "03_preregistration_v3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_preregistration_v3(args.preregistration_root)
    output = args.output or args.preregistration_root / "PREREGISTRATION_VALIDATION.json"
    write_validation_report(report, output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
