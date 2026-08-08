from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

from stage1_gapvalue240.campaign_canary import run_coordination_root_canary


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run one node of the shared coordination-root canary")
    p.add_argument("--coordination-root", required=True)
    p.add_argument("--machine-id", required=True)
    p.add_argument("--campaign-id", required=True)
    p.add_argument("--generation", required=True)
    p.add_argument("--expected-machine-ids", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--visibility-timeout-seconds", type=float, default=60.0)
    args = p.parse_args(argv)
    report = run_coordination_root_canary(
        args.coordination_root,
        machine_id=args.machine_id,
        campaign_id=args.campaign_id,
        generation=args.generation,
        expected_machine_ids=_csv(args.expected_machine_ids),
        output_dir=args.output_dir,
        visibility_timeout_seconds=args.visibility_timeout_seconds,
    )
    print(Path(args.output_dir).resolve() / f"{report['machine_id']}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
