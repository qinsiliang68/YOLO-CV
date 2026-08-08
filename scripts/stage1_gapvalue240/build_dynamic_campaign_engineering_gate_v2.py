from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

from stage1_gapvalue240.campaign_engineering_gate import (
    REQUIRED_EVIDENCE_SCHEMAS,
    ValidationIdentity,
    build_engineering_gate_v2,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build fail-closed Stage1 engineering gate v2")
    p.add_argument("--evidence", action="append", required=True, help="TYPE=PATH")
    p.add_argument("--source-tree-sha256", required=True)
    p.add_argument("--queue-sha256", required=True)
    p.add_argument("--canonical-lock-sha256", required=True)
    p.add_argument("--allowed-root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    evidence = {}
    for item in args.evidence:
        if "=" not in item:
            p.error("--evidence must be TYPE=PATH")
        key, value = item.split("=", 1)
        if key in evidence:
            p.error(f"duplicate evidence type: {key}")
        evidence[key] = value
    if set(evidence) != set(REQUIRED_EVIDENCE_SCHEMAS):
        p.error("--evidence must contain every required evidence type exactly once")
    build_engineering_gate_v2(
        evidence,
        expected_identity=ValidationIdentity(
            args.source_tree_sha256, args.queue_sha256, args.canonical_lock_sha256
        ),
        allowed_root=args.allowed_root,
        output_path=args.output,
    )
    print(Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
