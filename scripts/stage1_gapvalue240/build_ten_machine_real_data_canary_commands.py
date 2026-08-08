from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

from stage1_gapvalue240.campaign_canary import build_ten_machine_real_data_canary_commands


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate exactly one real-data canary job per machine")
    p.add_argument("--standalone-commands", required=True)
    p.add_argument("--expected-machine-ids", required=True)
    p.add_argument("--canary-job", action="append", default=[], help="MACHINE=JOB")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args(argv)
    mapping = {}
    for item in args.canary_job:
        if "=" not in item:
            p.error("--canary-job must be MACHINE=JOB")
        machine, job = item.split("=", 1)
        mapping[machine] = job
    paths = build_ten_machine_real_data_canary_commands(
        args.standalone_commands,
        output_dir=args.output_dir,
        expected_machine_ids=_csv(args.expected_machine_ids),
        canary_job_ids=mapping or None,
    )
    for path in paths.values():
        print(Path(path).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
