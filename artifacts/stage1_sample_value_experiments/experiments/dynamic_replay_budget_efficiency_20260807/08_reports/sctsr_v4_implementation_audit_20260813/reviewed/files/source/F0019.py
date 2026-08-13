from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.baseline_reference import SOURCE_TREE_INCLUDE_PATHS, source_external_references
from stage1_sctsr_v4.cli_support import add_output_argument, run_cli
from stage1_sctsr_v4.serialization import atomic_write_json
from stage1_sctsr_v4.source_identity import build_source_tree_manifest, validate_source_tree_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the exact registered SCTSR source-tree manifest")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    add_output_argument(parser)
    arguments = parser.parse_args()

    def action():
        manifest = build_source_tree_manifest(
            arguments.repository_root,
            SOURCE_TREE_INCLUDE_PATHS,
            external_references=source_external_references(),
        )
        atomic_write_json(arguments.manifest_output, manifest)
        validation = validate_source_tree_manifest(
            arguments.manifest_output,
            arguments.repository_root,
            require_clean=False,
        )
        return {
            **validation,
            "git_head": manifest["git_head"],
            "git_dirty": manifest["git_dirty"],
            "runtime_environment_digest": manifest["runtime_environment_digest"],
        }

    return run_cli("build_source_tree_manifest", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
