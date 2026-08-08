from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

from stage1_gapvalue240.campaign_lease_validation import (
    run_fencing_validation,
    run_lease_concurrency_validation,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run real local lease/fencing validations")
    p.add_argument("--coordination-root", required=True)
    p.add_argument("--concurrency-output", required=True)
    p.add_argument("--fencing-output", required=True)
    p.add_argument("--thread-rounds", type=int, default=100)
    p.add_argument("--thread-contenders", type=int, default=8)
    p.add_argument("--process-contenders", type=int, default=32)
    args = p.parse_args(argv)
    run_lease_concurrency_validation(
        args.coordination_root,
        output_path=args.concurrency_output,
        thread_rounds=args.thread_rounds,
        thread_claimants=args.thread_contenders,
        process_claimants=args.process_contenders,
    )
    run_fencing_validation(args.coordination_root, output_path=args.fencing_output)
    print(Path(args.concurrency_output).resolve())
    print(Path(args.fencing_output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
