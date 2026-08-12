#!/usr/bin/env python3
"""Build the SCREENED method-source acquisition and discovery plan."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.literature_screened_source_plan_v2 import (  # noqa: E402
    write_screened_source_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-queue", type=Path, required=True)
    parser.add_argument(
        "--discovery-glob",
        help="Glob resolving manual review input CSVs, for example discovery/**/review_input_*.csv",
    )
    parser.add_argument(
        "--discovery-path",
        type=Path,
        action="append",
        default=[],
        help="Exact discovery CSV; repeat for non-contiguous immutable batches.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source-subdir",
        default="sources/screened_method_v2",
    )
    parser.add_argument("--replace-existing", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    discovery_paths = list(args.discovery_path)
    if args.discovery_glob:
        discovery_paths.extend(Path().glob(args.discovery_glob))
    discovery_paths = sorted(
        {path.resolve() for path in discovery_paths},
        key=lambda path: path.as_posix().casefold(),
    )
    if not discovery_paths:
        raise SystemExit("at least one --discovery-path or matching --discovery-glob is required")
    output = write_screened_source_plan(
        args.screening_queue,
        discovery_paths=discovery_paths,
        output_root=args.output_root,
        source_subdir=args.source_subdir,
        replace_existing=args.replace_existing,
    )
    print(f"status=PASS output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
