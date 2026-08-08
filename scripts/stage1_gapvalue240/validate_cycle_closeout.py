from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse

from stage1_gapvalue240.campaign_aiops_reporting import validate_cycle_closeout


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate read-only cycle closeout")
    p.add_argument("--status-events", required=True)
    p.add_argument("--expected-job", action="append", required=True)
    p.add_argument("--release", required=True)
    p.add_argument("--assignment", required=True)
    p.add_argument("--canonical-lock", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    validate_cycle_closeout(
        args.status_events,
        expected_jobs=args.expected_job,
        release_path=args.release,
        assignment_manifest_path=args.assignment,
        canonical_lock_path=args.canonical_lock,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
