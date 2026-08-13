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
from stage1_sctsr_v4.prediction_runtime import publish_formal_endpoint
from stage1_sctsr_v4.recovery import prepare_formal_resume_context
from stage1_sctsr_v4.schedule import schedule_from_dict
from stage1_sctsr_v4.serialization import atomic_write_json, load_json, sha256_file, stable_digest
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-setup-root", type=Path)
    add_execution_arguments(parser)
    add_output_argument(parser)
    arguments = parser.parse_args()

    def action():
        require_receipt_outside_artifact_root(arguments.output, arguments.output_root)
        if arguments.execution_mode == "synthetic":
            if arguments.resume or arguments.resume_setup_root is not None:
                raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Formal resume flags are forbidden in synthetic mode")
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
        if arguments.epoch != 121:
            raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Formal branch epoch is derived from fresh/resume state and CLI --epoch must remain 121")
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
        runtime_policy = load_json(arguments.runtime_config)
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
        resume_context = None
        trainer_setup_root = arguments.output_root
        if arguments.resume:
            if arguments.resume_setup_root is None:
                raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "--resume requires --resume-setup-root")
            parent_sha = sha256_file(arguments.parent_checkpoint)
            resume_context = prepare_formal_resume_context(
                run_root=arguments.output_root,
                expected_run_id=lineage.logical_run_id,
                expected_arm_id=schedule.arm_id.value,
                expected_training_seed=identity.training_seed,
                expected_source_tree_digest=identity.source_tree_digest,
                expected_contract_digest=identity.effective_contract_digest,
                expected_asset_registry_digest=identity.asset_registry_digest,
                expected_previous_checkpoint_sha256=parent_sha,
                expected_previous_generation_digest=stable_digest(
                    {
                        "role": "BRANCH_START",
                        "parent_checkpoint_sha256": parent_sha,
                        "lineage_digest": lineage.lineage_digest,
                    }
                ),
                epoch_start=121,
                epoch_end=200,
                minimum_free_bytes=int(runtime_policy["minimum_resume_free_bytes"]),
            )
            trainer_setup_root = arguments.resume_setup_root.resolve()
            allowed_setup_root = (arguments.output_root.resolve() / "10_resume_setup").resolve()
            try:
                trainer_setup_root.relative_to(allowed_setup_root)
            except ValueError as exc:
                raise SctsrError(
                    ErrorCode.RESUME_GENERATION_MISMATCH,
                    "Resume trainer setup root must be contained under <run>/10_resume_setup",
                    artifact_path=str(trainer_setup_root),
                ) from exc
            expected_leaf = f"epoch_{resume_context.resume_epoch:04d}.generation_1"
            if trainer_setup_root.name != expected_leaf:
                raise SctsrError(
                    ErrorCode.RESUME_GENERATION_MISMATCH,
                    "Resume trainer setup generation name is noncanonical",
                    observed=trainer_setup_root.name,
                    expected=expected_leaf,
                )
        elif arguments.resume_setup_root is not None:
            raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "--resume-setup-root is forbidden without --resume")
        trainer, binding, trainer_binding = build_prepared_trainer(
            repository_root=arguments.repository_root,
            identity_manifest=arguments.identity_manifest,
            trainer_overrides_path=arguments.trainer_overrides,
            identity=identity,
            output_root=trainer_setup_root,
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
            resume_context=resume_context,
            execution_mode="formal",
        )
        endpoint_variant = str(runtime_policy["formal_endpoint_model_variant"])
        if endpoint_variant == "EMA":
            endpoint_model = getattr(getattr(trainer, "ema", None), "ema", None)
        elif endpoint_variant == "MODEL":
            endpoint_model = getattr(trainer, "model", None)
        else:
            raise SctsrError(
                ErrorCode.PREDICTION_IDENTITY_MISMATCH,
                "Runtime policy selects an unsupported formal endpoint model variant",
                observed=endpoint_variant,
            )
        if endpoint_model is None:
            raise SctsrError(
                ErrorCode.PARENT_CHECKPOINT_INCOMPLETE,
                "Prepared trainer does not expose the frozen endpoint model variant",
                observed=endpoint_variant,
            )
        transform = getattr(getattr(getattr(trainer, "test_loader", None), "dataset", None), "torch_transforms", None)
        if not callable(transform):
            raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Prepared val_model loader has no frozen evaluation transform")
        endpoint = publish_formal_endpoint(
            model=endpoint_model,
            transform=transform,
            run_root=arguments.output_root,
            repository_root=arguments.repository_root,
            asset_registry_path=arguments.asset_registry,
            checkpoint_path=Path(result["fixed_formal_endpoint"]["path"]),
            run_id=lineage.logical_run_id,
            arm_id=schedule.arm_id.value,
            model_variant=endpoint_variant,
            batch_size=int(runtime_policy["formal_endpoint_batch_size"]),
        )
        branch_receipt_path = arguments.output_root / "BRANCH_RECEIPT.json"
        branch_receipt = load_json(branch_receipt_path)
        atomic_write_json(branch_receipt_path, {**branch_receipt, "formal_endpoint_evidence": endpoint})
        from stage1_sctsr_v4.run_validation import build_artifact_index

        atomic_write_json(arguments.output_root / "ARTIFACT_INDEX.json", build_artifact_index(arguments.output_root))
        return {
            **result,
            "formal_endpoint_evidence": endpoint,
            "upstream_binding_digest": binding.binding_digest,
            "prepared_trainer_binding": trainer_binding,
            "formal_authorization": authorization,
            "identity_pool_binding": pool_binding,
            "parent_artifact_index_binding": parent_binding,
            "resume_context": None if resume_context is None else resume_context.as_dict(),
        }

    return run_cli("run_branch", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
