from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

from stage1_gapvalue240.campaign_engineering_gate import (
    REQUIRED_EVIDENCE_SCHEMAS,
    ValidationIdentity,
    bind_validation_evidence,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Bind one lower-level validation report to campaign identities")
    p.add_argument("--report", required=True)
    p.add_argument("--evidence-type", choices=sorted(REQUIRED_EVIDENCE_SCHEMAS), required=True)
    p.add_argument("--source-tree-sha256", required=True)
    p.add_argument("--queue-sha256", required=True)
    p.add_argument("--canonical-lock-sha256", required=True)
    p.add_argument("--allowed-root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    bind_validation_evidence(
        args.report,
        evidence_type=args.evidence_type,
        expected_schema=REQUIRED_EVIDENCE_SCHEMAS[args.evidence_type],
        identity=ValidationIdentity(
            args.source_tree_sha256, args.queue_sha256, args.canonical_lock_sha256
        ),
        output_path=args.output,
        allowed_root=args.allowed_root,
    )
    print(Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
