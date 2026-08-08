from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

from stage1_gapvalue240.campaign_all_epoch_validation import validate_all_epoch_telemetry


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate exact 1..200 campaign telemetry and replay dose")
    p.add_argument("--audit", required=True)
    p.add_argument("--schedule", required=True)
    p.add_argument("--process-telemetry-dir", required=True)
    p.add_argument("--expected-epochs", type=int, default=200)
    p.add_argument("--expected-arm-id")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    validate_all_epoch_telemetry(
        args.audit,
        args.schedule,
        args.process_telemetry_dir,
        output_path=args.output,
        expected_epochs=args.expected_epochs,
        expected_arm_id=args.expected_arm_id,
    )
    print(Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
