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
            "Reconcile expert relative-control cohorts with zero-replay, "
            "same-FN raw-score performance frontiers."
        )
    )
    parser.add_argument("--expert-zip", type=Path, required=True)
    parser.add_argument("--v5-report-dir", type=Path, required=True)
    parser.add_argument("--v3-report-dir", type=Path, required=True)
    parser.add_argument("--full-analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from stage1_gapvalue240.reconciliation_analysis import (
        run_reconciliation_analysis,
    )

    result = run_reconciliation_analysis(
        expert_zip=args.expert_zip,
        v5_report_dir=args.v5_report_dir,
        v3_report_dir=args.v3_report_dir,
        full_analysis_dir=args.full_analysis_dir,
        output_dir=args.output_dir,
        progress=lambda message: print(message, flush=True),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
