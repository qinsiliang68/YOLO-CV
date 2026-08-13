from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.asset_registry import load_asset_registry, validate_asset_registry
from stage1_sctsr_v4.cli_support import add_output_argument, run_cli


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only full SHA/content validation of every frozen SCTSR asset")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    add_output_argument(parser)
    arguments = parser.parse_args()
    return run_cli(
        "validate_assets",
        arguments.output,
        lambda: validate_asset_registry(load_asset_registry(arguments.registry), arguments.repository_root, verify_large_files=True),
    )


if __name__ == "__main__":
    raise SystemExit(main())
