"""Build the recoverable all-file/all-field foundation for the active Goal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=32)
    return parser


def main() -> int:
    from stage1_gapvalue240.goal_pipeline import publish_evidence_foundation

    args = build_parser().parse_args()
    state = publish_evidence_foundation(
        extracted_root=args.extracted_root,
        inventory_path=args.inventory,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
