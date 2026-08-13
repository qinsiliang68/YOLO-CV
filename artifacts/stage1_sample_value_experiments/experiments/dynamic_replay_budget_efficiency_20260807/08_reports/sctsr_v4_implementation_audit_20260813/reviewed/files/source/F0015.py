from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.cli_support import add_output_argument, run_cli
from stage1_sctsr_v4.repository_state_audit import audit_repository_state
from stage1_sctsr_v4.serialization import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit SCTSR source scope, protected history and prohibited side effects")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--implementation-start-commit", required=True)
    parser.add_argument("--implementation-source-commit", required=True)
    parser.add_argument("--state-output", type=Path, required=True)
    parser.add_argument("--changed-file-ledger-output", type=Path, required=True)
    add_output_argument(parser)
    arguments = parser.parse_args()

    def action():
        report = audit_repository_state(
            arguments.repository_root,
            baseline_commit=arguments.baseline_commit,
            implementation_start_commit=arguments.implementation_start_commit,
            implementation_source_commit=arguments.implementation_source_commit,
        )
        atomic_write_json(arguments.state_output, report)
        atomic_write_json(arguments.changed_file_ledger_output, report["changed_file_ledger"])
        return {
            "status": report["status"],
            "audit_digest": report["audit_digest"],
            "changed_file_count": report["changed_file_ledger"]["file_count"],
            "legacy_file_count": report["legacy_evidence"]["file_count"],
            "side_effects": report["side_effects"],
        }

    return run_cli("audit_repository_state", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
