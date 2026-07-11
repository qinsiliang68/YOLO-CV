from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

from stage1_gapvalue240.errors import GapValueError
from stage1_gapvalue240.machine import load_machine_config
from stage1_gapvalue240.machine_assets import (
    build_machine_asset_report,
    validate_machine_asset_report,
)
from stage1_gapvalue240.runtime_contract import (
    load_runtime_contract,
    validate_runtime_links,
    validation_status_for_mode,
    verify_all_selections_against_index,
    verify_release_identity,
    verify_selection_against_index,
)


DEFAULT_CONTRACT = "configs/stage1_gapvalue240/RUNTIME_CONTRACT_v1_2.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Stage1 GapValue v1.2 immutable runtime inputs."
    )
    parser.add_argument("--runtime-contract", default=DEFAULT_CONTRACT)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("links", "release", "all-selections"):
        command = sub.add_parser(name)
        command.add_argument("--repo-root", default=".")
    selection = sub.add_parser("selection")
    selection.add_argument("--repo-root", default=".")
    selection.add_argument("--run-slot", required=True)
    selection.add_argument("--selection-path")
    build = sub.add_parser("build-machine-assets")
    build.add_argument("--machine-config", required=True)
    build.add_argument("--output", required=True)
    build.add_argument(
        "--image-verification", choices=("none", "existence", "sha256"), default="existence"
    )
    build.add_argument("--overwrite", action="store_true")
    validate = sub.add_parser("validate-machine-assets")
    validate.add_argument("--report", required=True)
    validate.add_argument("--machine-id")
    status = sub.add_parser("status-for-mode")
    status.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        contract = load_runtime_contract(args.runtime_contract)
        if args.command == "links":
            result = validate_runtime_links(contract, args.repo_root)
        elif args.command == "release":
            result = verify_release_identity(contract, args.repo_root)
        elif args.command == "all-selections":
            result = verify_all_selections_against_index(contract, args.repo_root)
        elif args.command == "selection":
            result = verify_selection_against_index(
                contract, args.repo_root, args.run_slot, args.selection_path
            )
        elif args.command == "build-machine-assets":
            machine = load_machine_config(args.machine_config)
            result = build_machine_asset_report(
                contract,
                machine,
                args.output,
                image_verification=args.image_verification,
                overwrite=args.overwrite,
            )
        elif args.command == "validate-machine-assets":
            result = validate_machine_asset_report(
                contract, args.report, expected_machine_id=args.machine_id
            )
        elif args.command == "status-for-mode":
            result = {
                "status": "PASS",
                "validation_status": validation_status_for_mode(args.dry_run, contract),
                "dry_run": bool(args.dry_run),
            }
        else:  # argparse requires a known subcommand
            raise AssertionError(args.command)
    except (GapValueError, FileNotFoundError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2), file=sys.stderr)
        return 30
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
