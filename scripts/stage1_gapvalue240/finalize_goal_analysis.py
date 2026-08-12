from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_gapvalue240.goal_finalizer import finalize_goal_analysis  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and atomically publish the final Stage1 GapValue 240-run report."
    )
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    args = parser.parse_args()
    final = finalize_goal_analysis(args.report_root, inventory_path=args.inventory)
    print(
        json.dumps(
            {"status": "COMPLETE", "final_report_root": str(final)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
