from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the read-only deep analysis for the frozen Stage1 "
            "GapValue 240-run experiment."
        )
    )
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--completeness-audit", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path)
    parser.add_argument("--value-table", type=Path)
    parser.add_argument(
        "--skip-prediction-recompute",
        action="store_true",
        help=(
            "Skip the formal val_op metric recomputation. Intended only for "
            "fast report iteration; formal analysis recomputes by default."
        ),
    )
    return parser


def _resolved(path: Path | None) -> Path | None:
    return path.resolve() if path is not None else None


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    # Kept local so --help and parser tests do not initialize the analysis stack.
    from stage1_gapvalue240.deep_pipeline import run_deep_analysis

    result = run_deep_analysis(
        extracted_root=args.extracted_root.resolve(),
        inventory_path=args.inventory.resolve(),
        completeness_audit_path=args.completeness_audit.resolve(),
        matrix_path=args.matrix.resolve(),
        output_dir=args.output_dir.resolve(),
        selection_root=_resolved(args.selection_root),
        value_table=_resolved(args.value_table),
        recompute_predictions=not args.skip_prediction_recompute,
    )
    if isinstance(result, Mapping):
        summary = dict(result)
    else:
        summary = {"status": "PASS", "output": str(Path(result).resolve())}
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
