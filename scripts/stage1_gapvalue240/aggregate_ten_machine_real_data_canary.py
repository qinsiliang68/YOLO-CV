from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

from stage1_gapvalue240.campaign_canary import aggregate_ten_machine_real_data_canary


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate exactly ten real-data canary reports")
    p.add_argument("--reports-dir", required=True)
    p.add_argument("--expected-machine-ids", required=True)
    p.add_argument("--expected-commands", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    aggregate_ten_machine_real_data_canary(
        args.reports_dir,
        expected_machine_ids=_csv(args.expected_machine_ids),
        expected_commands_path=args.expected_commands,
        output_path=args.output,
    )
    print(Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
