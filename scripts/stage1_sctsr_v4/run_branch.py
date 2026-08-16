from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.arm_spec import ArmId
from stage1_sctsr_v4.cli_support import add_execution_arguments, add_output_argument, require_receipt_outside_artifact_root, run_cli
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.formal_execution import (
    build_execution_job_bindings,
    claim_formal_execution,
    execute_claimed_phase,
    execute_fenced_finalization,
)
from stage1_sctsr_v4.formal_completion import publish_formal_completion
from stage1_sctsr_v4.formal_cli import (
    build_prepared_trainer,
    load_formal_identity,
    load_lineage,
    prepare_formal_authorization,
    validate_identity_pool_artifacts,
    validate_parent_artifact_index,
)
from stage1_sctsr_v4.formal_training import (
    publish_formal_run_manifest_and_indexes,
    run_prepared_branch,
    validate_formal_run_runtime_identity,
)
from stage1_sctsr_v4.prediction_runtime import publish_formal_endpoint
from stage1_sctsr_v4.recovery import inspect_formal_resume_context, prepare_formal_resume_context
from stage1_sctsr_v4.run_intent import prepare_formal_run_intent_binding
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
            "execution_token": arguments.execution_token,
            "execution_claim_root": arguments.execution_claim_root,
            "source_tree_manifest": arguments.source_tree_manifest,
            "contract": arguments.contract,
            "arms": arguments.arms,
            "asset_registry": arguments.asset_registry,
            "runtime_config": arguments.runtime_config,
            "seed_registry": arguments.seed_registry,
            "run_intent_acknowledgement": arguments.run_intent_acknowledgement,
            "runbook_manifest": arguments.runbook_manifest,
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
        parent_sha = sha256_file(arguments.parent_checkpoint)
        resume_context = None
        resume_preview = None
        resume_kwargs = None
        trainer_setup_root = arguments.output_root
        if arguments.resume:
            if arguments.resume_setup_root is None:
                raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "--resume requires --resume-setup-root")
            resume_kwargs = {
                "run_root": arguments.output_root,
                "expected_run_id": lineage.logical_run_id,
                "expected_arm_id": schedule.arm_id.value,
                "expected_training_seed": identity.training_seed,
                "expected_source_tree_digest": identity.source_tree_digest,
                "expected_contract_digest": identity.effective_contract_digest,
                "expected_asset_registry_digest": identity.asset_registry_digest,
                "expected_previous_checkpoint_sha256": parent_sha,
                "expected_previous_generation_digest": stable_digest(
                    {
                        "role": "BRANCH_START",
                        "parent_checkpoint_sha256": parent_sha,
                        "lineage_digest": lineage.lineage_digest,
                    }
                ),
                "epoch_start": 121,
                "epoch_end": 200,
                "minimum_free_bytes": int(runtime_policy["minimum_resume_free_bytes"]),
                "allow_terminal_epoch_for_finalization": True,
            }
            resume_preview = inspect_formal_resume_context(**resume_kwargs)
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
            expected_leaf = f"epoch_{resume_preview.resume_epoch:04d}.generation_1"
            if trainer_setup_root.name != expected_leaf:
                raise SctsrError(
                    ErrorCode.RESUME_GENERATION_MISMATCH,
                    "Resume trainer setup generation name is noncanonical",
                    observed=trainer_setup_root.name,
                    expected=expected_leaf,
                )
        elif arguments.resume_setup_root is not None:
            raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "--resume-setup-root is forbidden without --resume")
        execution_job = build_execution_job_bindings(
            action="RESUME" if arguments.resume else "START",
            run_role="BRANCH",
            logical_run_id=lineage.logical_run_id,
            arm_id=schedule.arm_id.value,
            training_seed=identity.training_seed,
            output_root=arguments.output_root,
            parent_checkpoint_sha256=parent_sha,
            resume_checkpoint_sha256="0" * 64 if resume_preview is None else resume_preview.checkpoint_sha256,
            lineage_digest=lineage.lineage_digest,
            schedule_digest=schedule.plan_digest,
            resume_from_receipt_digest="0" * 64 if resume_preview is None else resume_preview.receipt_chain_digest,
        )
        run_intent_binding = prepare_formal_run_intent_binding(
            acknowledgement_path=arguments.run_intent_acknowledgement,
            runbook_manifest_path=arguments.runbook_manifest,
            repository_root=arguments.repository_root,
            output_root=arguments.output_root,
            action="RESUME" if arguments.resume else "START",
            run_role="BRANCH",
            logical_run_id=lineage.logical_run_id,
            arm_id=schedule.arm_id.value,
            training_seed=identity.training_seed,
            formal_identity_path=arguments.formal_identity,
            trainer_overrides_path=arguments.trainer_overrides,
            identity_manifest_path=arguments.identity_manifest,
            source_tree_digest=identity.source_tree_digest,
            contract_digest=identity.effective_contract_digest,
            asset_registry_digest=identity.asset_registry_digest,
            runtime_config_digest=identity.runtime_config_digest,
            asset_registry_path=arguments.asset_registry,
            parent_checkpoint_sha256=parent_sha,
            schedule_digest=schedule.plan_digest,
            identity_pool_binding_digest=pool_binding["binding_digest"],
            release_manifest_path=arguments.release_authorization,
            execution_token_path=arguments.execution_token,
            claim_registry_root=arguments.execution_claim_root,
            resume_checkpoint_sha256="0" * 64 if resume_preview is None else resume_preview.checkpoint_sha256,
            resume_receipt_digest="0" * 64 if resume_preview is None else resume_preview.receipt_chain_digest,
        )
        execution_claim = claim_formal_execution(
            arguments.execution_token,
            claim_registry_root=arguments.execution_claim_root,
            release=arguments.release_authorization,
            release_trust_policy=arguments.release_trust_policy,
            expected_release_bindings=authorization["expected_bindings"],
            release_manifest_sha256=authorization["release_manifest_sha256"],
            expected_job_bindings=execution_job,
        )

        def execute_claimed_runner_phase():
            claimed_resume_context = resume_context
            if resume_kwargs is not None:
                claimed_resume_context = prepare_formal_resume_context(**resume_kwargs)
                if (
                    claimed_resume_context.checkpoint_sha256 != resume_preview.checkpoint_sha256
                    or claimed_resume_context.receipt_chain_digest != resume_preview.receipt_chain_digest
                    or claimed_resume_context.resume_epoch != resume_preview.resume_epoch
                ):
                    raise SctsrError(
                        ErrorCode.RESUME_GENERATION_MISMATCH,
                        "Resume state changed between read-only inspection and fenced preparation",
                    )
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
                formal_input_binding=authorization["formal_input_binding"],
                execution_claim_binding=execution_claim,
                run_intent_binding=run_intent_binding,
                resume_context=claimed_resume_context,
                execution_mode="formal",
            )
            finalization_context = result.pop("_finalization_context")

            def finalize_branch():
                final_indexes = publish_formal_run_manifest_and_indexes(
                    root=arguments.output_root.resolve(),
                    identity=identity,
                    run_role="BRANCH",
                    run_id=lineage.logical_run_id,
                    arm_id=schedule.arm_id.value,
                    release_authorization=arguments.release_authorization,
                    release_expected_bindings=authorization["expected_bindings"],
                    execution_claim_binding=execution_claim,
                    execution_claim_snapshot=finalization_context["execution_claim_snapshot"],
                    run_intent_snapshot=finalization_context["run_intent_snapshot"],
                    prepared_trainer_binding=trainer_binding,
                )
                branch_receipt_path = arguments.output_root / "BRANCH_RECEIPT.json"
                branch_receipt = load_json(branch_receipt_path)
                atomic_write_json(branch_receipt_path, {**branch_receipt, "final_indexes": final_indexes})
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
                from stage1_sctsr_v4.dataset_adapter import revalidate_materialized_dataset_binding

                for materialized_role in ("train_materialized_content_binding", "val_model_materialized_content_binding"):
                    revalidate_materialized_dataset_binding(trainer_binding["dataset_binding"][materialized_role])
                endpoint = publish_formal_endpoint(
                    model=endpoint_model,
                    transform=transform,
                    run_root=arguments.output_root,
                    repository_root=arguments.repository_root,
                    dataset_root=trainer_binding["dataset_content_binding"]["dataset_root"],
                    asset_registry_path=arguments.asset_registry,
                    checkpoint_path=Path(result["fixed_formal_endpoint"]["path"]),
                    run_id=lineage.logical_run_id,
                    arm_id=schedule.arm_id.value,
                    model_variant=endpoint_variant,
                    batch_size=int(runtime_policy["formal_endpoint_batch_size"]),
                )
                branch_receipt = load_json(branch_receipt_path)
                atomic_write_json(
                    branch_receipt_path,
                    {
                        **branch_receipt,
                        "status": "FORMAL_BRANCH_ENDPOINT_COMPLETE_PENDING_COMMIT",
                        "formal_endpoint_evidence": endpoint,
                    },
                )
                validate_formal_run_runtime_identity(arguments.output_root)
                from stage1_sctsr_v4.run_validation import build_artifact_index

                atomic_write_json(arguments.output_root / "ARTIFACT_INDEX.json", build_artifact_index(arguments.output_root))
                completion = publish_formal_completion(
                    arguments.output_root,
                    run_role="BRANCH",
                    run_id=lineage.logical_run_id,
                    arm_id=schedule.arm_id.value,
                    training_seed=identity.training_seed,
                    terminal_epoch=200,
                    fixed_checkpoint_sha256=result["fixed_formal_endpoint"]["sha256"],
                )
                return {
                    **result,
                    "status": completion["status"],
                    "formal_endpoint_evidence": endpoint,
                    "formal_completion": completion,
                    "upstream_binding_digest": binding.binding_digest,
                    "prepared_trainer_binding": trainer_binding,
                    "formal_authorization": authorization,
                    "execution_claim": execution_claim,
                    "run_intent_binding": run_intent_binding,
                    "identity_pool_binding": pool_binding,
                    "parent_artifact_index_binding": parent_binding,
                    "resume_context": None if claimed_resume_context is None else claimed_resume_context.as_dict(),
                }

            return execute_fenced_finalization(
                execution_claim,
                expected_job_bindings=execution_job,
                operation=finalize_branch,
            )

        return execute_claimed_phase(
            execution_claim,
            expected_job_bindings=execution_job,
            operation=execute_claimed_runner_phase,
        )

    return run_cli("run_branch", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
