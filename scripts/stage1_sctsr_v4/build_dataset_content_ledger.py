from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.asset_registry import load_asset_registry
from stage1_sctsr_v4.cli_support import add_output_argument, run_cli
from stage1_sctsr_v4.dataset_content_ledger import build_dataset_content_ledger
from stage1_sctsr_v4.errors import ErrorCode, SctsrError


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the immutable SHA-256/decode ledger for every registered non-test image",
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--ledger-output", type=Path, required=True)
    add_output_argument(parser)
    arguments = parser.parse_args()

    def action():
        if arguments.ledger_output.resolve() == arguments.output.resolve():
            raise SctsrError(
                ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE,
                "Dataset content ledger and CLI receipt require distinct immutable paths",
            )
        return build_dataset_content_ledger(
            registry=load_asset_registry(arguments.registry),
            repository_root=arguments.repository_root,
            dataset_root=arguments.dataset_root,
            output_path=arguments.ledger_output,
        )

    return run_cli("build_dataset_content_ledger", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
