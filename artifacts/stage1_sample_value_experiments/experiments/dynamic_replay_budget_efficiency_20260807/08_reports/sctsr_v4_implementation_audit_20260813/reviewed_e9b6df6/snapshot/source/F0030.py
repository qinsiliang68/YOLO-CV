from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.cli_support import add_output_argument, run_cli
from stage1_sctsr_v4.manual_line_review import validate_manual_line_review


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SCTSR SA-280 through SA-289 source-line review anchors")
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    add_output_argument(parser)
    arguments = parser.parse_args()
    return run_cli(
        "validate_manual_line_review",
        arguments.output,
        lambda: validate_manual_line_review(arguments.review, repository_root=arguments.repository_root),
    )


if __name__ == "__main__":
    raise SystemExit(main())
