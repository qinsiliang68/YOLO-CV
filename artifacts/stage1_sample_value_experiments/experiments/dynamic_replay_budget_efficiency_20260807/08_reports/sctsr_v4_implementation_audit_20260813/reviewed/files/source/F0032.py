from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.arm_spec import ArmId
from stage1_sctsr_v4.cli_support import add_output_argument, run_cli
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.schedule import schedule_from_dict, validate_common_prefix, validate_schedule, validate_u_f_parity
from stage1_sctsr_v4.serialization import load_json, stable_digest
from stage1_sctsr_v4.synthetic_canary import _build_all_schedules
from stage1_sctsr_v4.synthetic_fixture import build_synthetic_fixture


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the exact cross-arm SCTSR schedule matrix")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--schedule", type=Path, action="append", default=[])
    parser.add_argument("--training-seed", type=int, default=20260812)
    add_output_argument(parser)
    arguments = parser.parse_args()

    def action():
        if arguments.synthetic:
            if arguments.schedule:
                raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Synthetic schedule validation may not mix external schedule files")
            plans = _build_all_schedules(build_synthetic_fixture(training_seed=arguments.training_seed))
        else:
            if len(arguments.schedule) != len(ArmId):
                raise SctsrError(
                    ErrorCode.CONFIGURATION_MISMATCH,
                    "Formal schedule validation requires exactly one file for every phase-1 arm",
                    observed=len(arguments.schedule),
                    expected=len(ArmId),
                )
            plans = {}
            for path in arguments.schedule:
                plan = schedule_from_dict(load_json(path))
                if plan.arm_id in plans:
                    raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Formal schedule set contains a duplicate arm", observed=plan.arm_id.value)
                plans[plan.arm_id] = plan
        required = set(ArmId)
        if set(plans) != required:
            raise SctsrError(
                ErrorCode.CONFIGURATION_MISMATCH,
                "Schedule set differs from the exact eight-arm phase-1 matrix",
                observed=sorted(item.value for item in plans),
                expected=sorted(item.value for item in required),
            )
        for plan in plans.values():
            validate_schedule(plan)
        validate_u_f_parity(plans[ArmId.T_U], plans[ArmId.T_F])
        validate_u_f_parity(plans[ArmId.R2_U], plans[ArmId.R2_F])
        validate_common_prefix(plans[ArmId.T_U], plans[ArmId.T_TO_R2_AT_160])
        validate_common_prefix(plans[ArmId.T_U], plans[ArmId.T_TO_NR_AT_160])
        digests = {arm.value: plans[arm].plan_digest for arm in sorted(plans, key=lambda item: item.value)}
        return {
            "status": "PASS",
            "validated_arms": sorted(digests),
            "schedule_digests": digests,
            "cross_arm_parity": "PASS",
            "digest": stable_digest(digests),
            "formal_training_started": False,
        }

    return run_cli("validate_schedule", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
