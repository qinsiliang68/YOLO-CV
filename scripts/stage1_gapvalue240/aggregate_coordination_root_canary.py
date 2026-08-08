from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

from stage1_gapvalue240.campaign_canary import aggregate_coordination_root_canary


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate shared-root canary node reports")
    p.add_argument("--reports-dir", required=True)
    p.add_argument("--expected-machine-ids", required=True)
    p.add_argument("--campaign-id", required=True)
    p.add_argument("--generation", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    aggregate_coordination_root_canary(
        args.reports_dir,
        expected_machine_ids=_csv(args.expected_machine_ids),
        campaign_id=args.campaign_id,
        generation=args.generation,
        output_path=args.output,
    )
    print(Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
