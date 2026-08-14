from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.cli_support import add_output_argument, run_cli
from stage1_sctsr_v4.run_intent import build_runbook_manifest, validate_runbook_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an immutable SHA-bound SCTSR operator runbook manifest")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--document", type=Path, action="append", required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    add_output_argument(parser)
    arguments = parser.parse_args()

    def action():
        build_runbook_manifest(
            repository_root=arguments.repository_root,
            document_paths=arguments.document,
            output_path=arguments.manifest_output,
        )
        return validate_runbook_manifest(arguments.manifest_output, repository_root=arguments.repository_root)

    return run_cli("build_runbook_manifest", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
