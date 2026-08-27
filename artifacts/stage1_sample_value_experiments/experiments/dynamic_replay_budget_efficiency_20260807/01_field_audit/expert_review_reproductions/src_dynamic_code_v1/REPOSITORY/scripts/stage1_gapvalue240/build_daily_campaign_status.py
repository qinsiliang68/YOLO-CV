from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse

from stage1_gapvalue240.campaign_aiops_reporting import build_daily_campaign_status


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build read-only daily Stage1 campaign status")
    p.add_argument("--status-events", required=True)
    p.add_argument("--release", required=True)
    p.add_argument("--assignment", required=True)
    p.add_argument("--canonical-lock", required=True)
    p.add_argument("--preregistered-gate")
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-markdown", required=True)
    args = p.parse_args(argv)
    build_daily_campaign_status(
        args.status_events,
        release_path=args.release,
        assignment_manifest_path=args.assignment,
        canonical_lock_path=args.canonical_lock,
        preregistered_gate_path=args.preregistered_gate,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
