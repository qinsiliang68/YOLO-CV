from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.arm_spec import ArmId
from stage1_sctsr_v4.asset_registry import load_asset_registry
from stage1_sctsr_v4.cli_support import add_output_argument, run_cli
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.formal_cli import _load_identity_pool_artifact
from stage1_sctsr_v4.formal_pool_inputs import load_formal_pool_inputs
from stage1_sctsr_v4.schedule import build_schedule, schedule_to_dict
from stage1_sctsr_v4.serialization import atomic_write_json, stable_digest
from stage1_sctsr_v4.synthetic_canary import _build_all_schedules
from stage1_sctsr_v4.synthetic_fixture import build_synthetic_fixture


_PRIMARY_ROLE = {
    ArmId.R1_U: "R1_GLOBAL_RANDOM",
    ArmId.R2_U: "R2_MATCHED_RANDOM",
    ArmId.T_U: "T_STRESS",
    ArmId.R2_F: "R2_MATCHED_RANDOM",
    ArmId.T_F: "T_STRESS",
    ArmId.T_TO_R2_AT_160: "T_STRESS",
    ArmId.T_TO_NR_AT_160: "T_STRESS",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one materialized E1-E200 SCTSR schedule from immutable pool artifacts")
    parser.add_argument("--arm", choices=[item.value for item in ArmId], required=True)
    parser.add_argument("--schedule-output", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--asset-registry", type=Path)
    parser.add_argument("--primary-pool", type=Path)
    parser.add_argument("--fallback-pool", type=Path)
    parser.add_argument("--training-seed", type=int, default=20260812)
    add_output_argument(parser)
    arguments = parser.parse_args()

    def action():
        if arguments.schedule_output.exists():
            raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Schedule output already exists", artifact_path=str(arguments.schedule_output))
        arm = ArmId(arguments.arm)
        if arguments.synthetic:
            if any((arguments.repository_root, arguments.asset_registry, arguments.primary_pool, arguments.fallback_pool)):
                raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Synthetic schedule construction may not mix formal assets")
            plan = _build_all_schedules(build_synthetic_fixture(training_seed=arguments.training_seed))[arm]
            asset_binding = "SYNTHETIC_NOT_SCIENTIFIC_RESULT"
        else:
            if arguments.repository_root is None or arguments.asset_registry is None:
                raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Formal schedule construction requires repository root and asset registry")
            root = arguments.repository_root.resolve()
            registry = load_asset_registry(arguments.asset_registry)
            inputs = load_formal_pool_inputs(registry, root)
            denominator = registry.base_denominator
            primary_groups = None
            primary_digest = "NONE"
            fallback_groups = None
            fallback_digest = None
            if arm is ArmId.NR:
                if arguments.primary_pool is not None or arguments.fallback_pool is not None:
                    raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "NR may not bind an identity pool")
            else:
                if arguments.primary_pool is None:
                    raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Replay schedule requires a canonical primary pool manifest")
                primary, primary_groups, _ = _load_identity_pool_artifact(
                    arguments.primary_pool,
                    expected_base_denominator=denominator,
                    expected_base_manifest_sha256=inputs.base_manifest_sha256,
                )
                expected_role = _PRIMARY_ROLE[arm]
                if primary.spec.pool_role != expected_role:
                    raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Primary pool role differs from arm", observed=primary.spec.pool_role, expected=expected_role)
                primary_digest = primary.spec.identity_digest
                if arm is ArmId.T_TO_R2_AT_160:
                    if arguments.fallback_pool is None:
                        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Fallback arm requires a canonical R2 pool manifest")
                    fallback, fallback_groups, _ = _load_identity_pool_artifact(
                        arguments.fallback_pool,
                        expected_base_denominator=denominator,
                        expected_base_manifest_sha256=inputs.base_manifest_sha256,
                    )
                    if fallback.spec.pool_role != "R2_MATCHED_RANDOM":
                        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Fallback pool is not R2_MATCHED_RANDOM")
                    if {record.sample_id for record in primary.records} & {record.sample_id for record in fallback.records}:
                        raise SctsrError(ErrorCode.R2_OVERLAPS_T, "Fallback R2 overlaps T")
                    fallback_digest = fallback.spec.identity_digest
                elif arguments.fallback_pool is not None:
                    raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Non-fallback arm may not bind a fallback pool")
            plan = build_schedule(
                arm,
                primary_groups=primary_groups,
                primary_digest=primary_digest,
                fallback_groups=fallback_groups,
                fallback_digest=fallback_digest,
                base_denominator=denominator,
            )
            asset_binding = stable_digest({"asset_registry_digest": registry.digest, "base_manifest_sha256": inputs.base_manifest_sha256})
        payload = schedule_to_dict(plan)
        atomic_write_json(arguments.schedule_output, payload)
        return {
            "arm_id": arguments.arm,
            "schedule_path": arguments.schedule_output.resolve().as_posix(),
            "plan_digest": plan.plan_digest,
            "total_occurrences": plan.total_occurrences,
            "base_denominator": plan.base_denominator,
            "asset_binding": asset_binding,
            "formal_training_started": False,
        }

    return run_cli("build_schedule", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
