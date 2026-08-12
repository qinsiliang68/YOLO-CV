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
            "Compare Stage1 240/40/120-run results with the zero-replay yolo11l "
            "baseline using tie-safe, same-FN performance frontiers."
        )
    )
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--run40-root", type=Path, required=True)
    parser.add_argument("--run120-root", type=Path, required=True)
    parser.add_argument("--v3-report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from stage1_gapvalue240.performance_pipeline import (
        run_performance_frontier_analysis,
    )

    result = run_performance_frontier_analysis(
        extracted_root=args.extracted_root,
        inventory_path=args.inventory,
        matrix_path=args.matrix,
        baseline_root=args.baseline_root,
        run40_root=args.run40_root,
        run120_root=args.run120_root,
        v3_report_dir=args.v3_report_dir,
        output_dir=args.output_dir,
        progress=lambda message: print(message, flush=True),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
