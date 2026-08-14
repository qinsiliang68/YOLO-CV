from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.asset_registry import load_asset_registry
from stage1_sctsr_v4.cli_support import add_output_argument, run_cli
from stage1_sctsr_v4.dataset_content_ledger import (
    registered_dataset_manifest_asset_ids,
    validate_registered_dataset_content,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify every registered non-test physical image against the immutable content ledger",
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--ledger-only", action="store_true", help="Validate ledger/manifests without rehashing physical images")
    add_output_argument(parser)
    arguments = parser.parse_args()

    def action():
        registry = load_asset_registry(arguments.registry)
        return validate_registered_dataset_content(
            registry=registry,
            repository_root=arguments.repository_root,
            dataset_root=arguments.dataset_root,
            required_manifest_asset_ids=registered_dataset_manifest_asset_ids(registry),
            verify_physical_files=not arguments.ledger_only,
        )

    return run_cli("validate_dataset_content", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
