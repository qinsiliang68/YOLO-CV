from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.asset_registry import (
    build_split_identity_bundle,
    load_asset_registry,
)
from stage1_sctsr_v4.cli_support import add_output_argument, run_cli
from stage1_sctsr_v4.errors import ErrorCode, SctsrError


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build one immutable identity bundle from registered validation split components",
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--split-role", choices=("val_model", "val_cal", "val_op"), required=True)
    parser.add_argument("--bundle-output", type=Path, required=True)
    add_output_argument(parser)
    arguments = parser.parse_args()

    def action():
        if arguments.bundle_output.resolve() == arguments.output.resolve():
            raise SctsrError(
                ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE,
                "Split identity bundle and CLI receipt require distinct paths",
            )
        return build_split_identity_bundle(
            load_asset_registry(arguments.registry),
            repository_root=arguments.repository_root,
            registry_path=arguments.registry,
            split_role=arguments.split_role,
            output_path=arguments.bundle_output,
        )

    return run_cli("build_split_identity_bundle", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
