from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan initial/best/last parameter drift for all canonical GapValue runs."
    )
    parser.add_argument("--extracted-root", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--epoch-curves", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-runs", type=int, default=240)
    parser.add_argument("--expected-epochs", type=int, default=200)
    return parser


def main(argv: list[str] | None = None) -> int:
    from stage1_gapvalue240.checkpoint_drift import scan_inventory_checkpoints

    args = _parser().parse_args(argv)

    def progress(done: int, total: int, run_slot: str) -> None:
        if done == 1 or done == total or done % 5 == 0:
            print(f"checkpoint_scan {done}/{total} {run_slot}", flush=True)

    report = scan_inventory_checkpoints(
        extracted_root=Path(args.extracted_root),
        inventory_path=Path(args.inventory),
        initial_checkpoint=Path(args.initial_checkpoint),
        epoch_curves_path=Path(args.epoch_curves),
        output_dir=Path(args.output_dir),
        expected_runs=args.expected_runs,
        expected_epochs=args.expected_epochs,
        progress=progress,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
