from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

from stage1_gapvalue240.campaign_contract_validation import (
    build_source_tree_manifest,
    validate_assignment_reassignment,
    validate_source_tree_immutability,
    validate_standalone_entry,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Validate Stage1 dynamic-campaign contracts")
    sub = root.add_subparsers(dest="action", required=True)

    source = sub.add_parser("source-manifest")
    source.add_argument("--source-root", required=True)
    source.add_argument("--output", required=True)

    immutable = sub.add_parser("source-immutability")
    immutable.add_argument("--source-root", required=True)
    immutable.add_argument("--baseline-manifest", required=True)
    immutable.add_argument("--output", required=True)

    standalone = sub.add_parser("standalone")
    standalone.add_argument("--queue-dir", required=True)
    standalone.add_argument("--release", required=True)
    standalone.add_argument("--assignment", required=True)
    standalone.add_argument("--repo-root", required=True)
    standalone.add_argument("--controller-offline-smoke", required=True)
    standalone.add_argument("--output", required=True)

    reassignment = sub.add_parser("reassignment")
    reassignment.add_argument("--queue-dir", required=True)
    reassignment.add_argument("--release", required=True)
    reassignment.add_argument("--old-assignment", required=True)
    reassignment.add_argument("--new-assignment", required=True)
    reassignment.add_argument("--source-root", required=True)
    reassignment.add_argument("--source-manifest-before", required=True)
    reassignment.add_argument("--output", required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.action == "source-manifest":
        build_source_tree_manifest(args.source_root, args.output)
    elif args.action == "source-immutability":
        validate_source_tree_immutability(args.source_root, args.baseline_manifest, args.output)
    elif args.action == "standalone":
        validate_standalone_entry(
            args.queue_dir,
            args.release,
            args.assignment,
            repo_root=args.repo_root,
            controller_offline_smoke_report=args.controller_offline_smoke,
            output_path=args.output,
        )
    else:
        validate_assignment_reassignment(
            args.queue_dir,
            args.release,
            args.old_assignment,
            args.new_assignment,
            source_root=args.source_root,
            source_manifest_before=args.source_manifest_before,
            output_path=args.output,
        )
    print(Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
