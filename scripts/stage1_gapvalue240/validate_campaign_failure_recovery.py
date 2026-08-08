from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

from stage1_gapvalue240.campaign_recovery_validation import REQUIRED_SCENARIOS, validate_failure_recovery_evidence


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate Stage1 failure/recovery evidence")
    p.add_argument("--scenario", action="append", required=True, help="NAME=PATH")
    p.add_argument("--allow-cross-machine-resume", action="store_true")
    p.add_argument("--full-state-package-validation")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    reports = {}
    for item in args.scenario:
        if "=" not in item:
            p.error("--scenario must be NAME=PATH")
        name, path = item.split("=", 1)
        reports[name] = path
    if set(reports) != set(REQUIRED_SCENARIOS):
        p.error("--scenario must contain every required scenario exactly once")
    validate_failure_recovery_evidence(
        reports,
        output_path=args.output,
        allow_cross_machine_resume=args.allow_cross_machine_resume,
        full_state_package_validation=args.full_state_package_validation,
    )
    print(Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
