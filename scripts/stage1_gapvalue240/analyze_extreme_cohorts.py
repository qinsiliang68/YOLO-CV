from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the read-only S/A/B/M/H extreme-cohort analysis for the "
            "frozen Stage1 GapValue 240-run experiment."
        )
    )
    parser.add_argument("--v2-report-dir", type=Path, required=True)
    parser.add_argument("--expert-package-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from stage1_gapvalue240.extreme_pipeline import run_extreme_cohort_analysis

    result = run_extreme_cohort_analysis(
        v2_report_dir=args.v2_report_dir.resolve(),
        expert_package_root=args.expert_package_root.resolve(),
        selection_root=args.selection_root.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
