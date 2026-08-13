from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.cli_support import add_output_argument, run_cli
from stage1_sctsr_v4.implementation_self_audit import validate_implementation_self_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate all taskbook Appendix-D SCTSR implementation checks")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--taskbook", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    add_output_argument(parser)
    arguments = parser.parse_args()
    return run_cli(
        "validate_implementation_self_audit",
        arguments.output,
        lambda: validate_implementation_self_audit(
            arguments.audit,
            taskbook_path=arguments.taskbook,
            evidence_root=arguments.evidence_root,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
