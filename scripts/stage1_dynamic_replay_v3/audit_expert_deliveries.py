"""Build the read-only expert delivery inventory for the registered Stage1 audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.expert_delivery_audit import audit_expert_deliveries


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "stage1_sample_value_experiments"
    / "experiments"
    / "dynamic_replay_budget_efficiency_20260807"
    / "01_field_audit"
    / "expert_delivery_audit_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory, safely extract, and hash expert delivery artifacts without executing them."
    )
    parser.add_argument("--downloads-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_expert_deliveries(args.downloads_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] in {"PASS", "PASS_WITH_AUXILIARY_GAPS", "INCOMPLETE_SOURCE_MISSING"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
