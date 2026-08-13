from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.cli_support import add_output_argument, run_cli
from stage1_sctsr_v4.implementation_self_audit import (
    build_implementation_self_audit_from_plan,
    validate_implementation_self_audit,
)
from stage1_sctsr_v4.serialization import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the exact 206-row SCTSR Appendix-D self-audit")
    parser.add_argument("--taskbook", type=Path, required=True)
    parser.add_argument("--implementation-source-commit", required=True)
    parser.add_argument("--generated-by", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-state", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    add_output_argument(parser)
    arguments = parser.parse_args()

    def action():
        audit = build_implementation_self_audit_from_plan(
            taskbook_path=arguments.taskbook,
            implementation_source_commit=arguments.implementation_source_commit,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            generated_by=arguments.generated_by,
            plan=arguments.plan,
            repository_state=arguments.repository_state,
        )
        atomic_write_json(arguments.audit_output, audit)
        return validate_implementation_self_audit(
            arguments.audit_output,
            taskbook_path=arguments.taskbook,
            evidence_root=arguments.evidence_root,
        )

    return run_cli("build_implementation_self_audit", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
