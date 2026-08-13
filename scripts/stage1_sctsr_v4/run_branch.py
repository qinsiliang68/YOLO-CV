from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.arm_spec import ArmId
from stage1_sctsr_v4.cli_support import add_execution_arguments, add_output_argument, require_receipt_outside_artifact_root, run_cli
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.formal_cli import (
    build_prepared_trainer,
    load_formal_identity,
    load_lineage,
    prepare_formal_authorization,
    validate_identity_pool_artifacts,
    validate_parent_artifact_index,
)
from stage1_sctsr_v4.formal_training import run_prepared_branch
from stage1_sctsr_v4.schedule import schedule_from_dict
from stage1_sctsr_v4.serialization import load_json
from stage1_sctsr_v4.synthetic_execution import run_synthetic_branch


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an E121-E200 SCTSR child from a byte-bound E120 parent")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--arm-id", choices=[item.value for item in ArmId], required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--epoch", type=int, default=121)
    parser.add_argument("--identity-manifest", type=Path)
    parser.add_argument("--trainer-overrides", type=Path)
    parser.add_argument("--formal-identity", type=Path)
    parser.add_argument("--lineage", type=Path)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--identity-pool", type=Path, action="append", default=[])
    parser.add_argument("--parent-artifact-index", type=Path)
    add_execution_arguments(parser)
    add_output_argument(parser)
    arguments = parser.parse_args()

    def action():
        require_receipt_outside_artifact_root(arguments.output, arguments.output_root)
        if arguments.execution_mode == "synthetic":
            return run_synthetic_branch(
                arguments.output_root,
                repository_root=arguments.repository_root,
                parent_checkpoint=arguments.parent_checkpoint,
                arm_id=ArmId(arguments.arm_id),
                training_seed=arguments.training_seed,
                epoch=arguments.epoch,
            )
        required = {
            "identity_manifest": arguments.identity_manifest,
            "trainer_overrides": arguments.trainer_overrides,
            "formal_identity": arguments.formal_identity,
            "lineage": arguments.lineage,
            "schedule": arguments.schedule,
            "parent_artifact_index": arguments.parent_artifact_index,
            "release_authorization": arguments.release_authorization,
            "release_trust_policy": arguments.release_trust_policy,
            "source_tree_manifest": arguments.source_tree_manifest,
            "contract": arguments.contract,
            "arms": arguments.arms,
            "asset_registry": arguments.asset_registry,
            "runtime_config": arguments.runtime_config,
            "seed_registry": arguments.seed_registry,
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise SctsrError(ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED, "Formal mode lacks mandatory trust or identity inputs", observed=missing)
        identity = load_formal_identity(arguments.formal_identity)
        lineage = load_lineage(arguments.lineage)
        schedule = schedule_from_dict(load_json(arguments.schedule))
        if schedule.arm_id.value != arguments.arm_id or identity.training_seed != arguments.training_seed:
            raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Formal identity or schedule differs from CLI arm/seed")
        authorization = prepare_formal_authorization(
            repository_root=arguments.repository_root,
            identity=identity,
            release_authorization=arguments.release_authorization,
            release_trust_policy=arguments.release_trust_policy,
            source_tree_manifest=arguments.source_tree_manifest,
            contract_path=arguments.contract,
            arms_path=arguments.arms,
            asset_registry_path=arguments.asset_registry,
            runtime_config_path=arguments.runtime_config,
            seed_registry_path=arguments.seed_registry,
        )
        pool_binding = validate_identity_pool_artifacts(
            arguments.identity_pool,
            schedule=schedule,
            expected_base_denominator=120_000,
            expected_base_manifest_sha256=identity.base_manifest_sha256,
        )
        parent_binding = validate_parent_artifact_index(
            parent_checkpoint=arguments.parent_checkpoint,
            parent_artifact_index=arguments.parent_artifact_index,
        )
        trainer, binding, trainer_binding = build_prepared_trainer(
            repository_root=arguments.repository_root,
            identity_manifest=arguments.identity_manifest,
            trainer_overrides_path=arguments.trainer_overrides,
            identity=identity,
            output_root=arguments.output_root,
            asset_registry_path=arguments.asset_registry,
            schedule=schedule,
            identity_pool_manifests=arguments.identity_pool,
        )
        result = run_prepared_branch(
            trainer=trainer,
            identity=identity,
            parent_checkpoint=arguments.parent_checkpoint,
            lineage=lineage,
            schedule=schedule,
            replay_batch_provider=trainer.replay_batch_provider,
            output_root=arguments.output_root,
            release_authorization=arguments.release_authorization,
            release_trust_policy=arguments.release_trust_policy,
            release_expected_bindings=authorization["expected_bindings"],
            identity_pool_binding=pool_binding,
            parent_artifact_index_binding=parent_binding,
            prepared_trainer_binding=trainer_binding,
            execution_mode="formal",
        )
        return {
            **result,
            "upstream_binding_digest": binding.binding_digest,
            "prepared_trainer_binding": trainer_binding,
            "formal_authorization": authorization,
            "identity_pool_binding": pool_binding,
            "parent_artifact_index_binding": parent_binding,
        }

    return run_cli("run_branch", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
