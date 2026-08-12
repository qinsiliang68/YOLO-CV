#!/usr/bin/env python3
"""Validate Stage1 expert/current-v3 crosswalks without executing evidence commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.tripartite_crosswalk import (  # noqa: E402
    validate_tripartite_crosswalks,
    write_validation_report,
)


EXPERIMENT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "stage1_sample_value_experiments"
    / "experiments"
    / "dynamic_replay_budget_efficiency_20260807"
)
REPRODUCTION_ROOT = EXPERIMENT_ROOT / "01_field_audit" / "expert_review_reproductions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--budgeted-matrix",
        type=Path,
        default=REPRODUCTION_ROOT / "expert_vs_v3_tripartite_v2.csv",
    )
    parser.add_argument(
        "--dynamic-matrix",
        type=Path,
        default=REPRODUCTION_ROOT / "dynamic_review_vs_v3_tripartite_v2.csv",
    )
    parser.add_argument(
        "--expert-inventory",
        type=Path,
        default=(
            EXPERIMENT_ROOT
            / "01_field_audit"
            / "expert_delivery_audit_v3"
            / "expert_v1_inventory.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRODUCTION_ROOT / "TRIPARTITE_CROSSWALK_VALIDATION_v2.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_tripartite_crosswalks(
        repo_root=args.repo_root,
        budgeted_matrix=args.budgeted_matrix,
        dynamic_matrix=args.dynamic_matrix,
        expert_inventory=args.expert_inventory,
    )
    write_validation_report(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
