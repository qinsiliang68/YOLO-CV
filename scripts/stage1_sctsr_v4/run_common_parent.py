from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.cli_support import add_execution_arguments, add_output_argument, require_receipt_outside_artifact_root, run_cli
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.formal_execution import build_execution_job_bindings, claim_formal_execution
from stage1_sctsr_v4.formal_cli import build_prepared_trainer, load_formal_identity, prepare_formal_authorization
from stage1_sctsr_v4.formal_training import run_prepared_common_parent
from stage1_sctsr_v4.recovery import prepare_formal_resume_context
from stage1_sctsr_v4.serialization import load_json, stable_digest
from stage1_sctsr_v4.synthetic_execution import run_synthetic_common_parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the E1-E120 SCTSR common parent")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--identity-manifest", type=Path)
    parser.add_argument("--trainer-overrides", type=Path)
    parser.add_argument("--formal-identity", type=Path)
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
            return run_synthetic_common_parent(
                arguments.output_root,
                repository_root=arguments.repository_root,
                training_seed=arguments.training_seed,
            )
        required = {
            "identity_manifest": arguments.identity_manifest,
            "trainer_overrides": arguments.trainer_overrides,
            "formal_identity": arguments.formal_identity,
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
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise SctsrError(ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED, "Formal mode lacks mandatory trust or identity inputs", observed=missing)
        identity = load_formal_identity(arguments.formal_identity)
        if identity.training_seed != arguments.training_seed:
            raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Formal identity training seed differs from CLI seed")
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
        resume_context = None
        trainer_setup_root = arguments.output_root
        if arguments.resume:
            if arguments.resume_setup_root is None:
                raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "--resume requires --resume-setup-root")
            runtime_policy = load_json(arguments.runtime_config)
            resume_context = prepare_formal_resume_context(
                run_root=arguments.output_root,
                expected_run_id=f"PARENT_{identity.training_seed}",
                expected_arm_id="COMMON_PARENT_NR",
                expected_training_seed=identity.training_seed,
                expected_source_tree_digest=identity.source_tree_digest,
                expected_contract_digest=identity.effective_contract_digest,
                expected_asset_registry_digest=identity.asset_registry_digest,
                expected_previous_checkpoint_sha256=identity.initial_checkpoint_sha256,
                expected_previous_generation_digest=stable_digest(
                    {"role": "COMMON_PARENT_START", "initial_checkpoint_sha256": identity.initial_checkpoint_sha256}
                ),
                epoch_start=1,
                epoch_end=120,
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
        execution_job = build_execution_job_bindings(
            action="RESUME" if arguments.resume else "START",
            run_role="COMMON_PARENT",
            logical_run_id=f"PARENT_{identity.training_seed}",
            arm_id="COMMON_PARENT_NR",
            training_seed=identity.training_seed,
            output_root=arguments.output_root,
            parent_checkpoint_sha256=identity.initial_checkpoint_sha256,
            resume_checkpoint_sha256="0" * 64 if resume_context is None else resume_context.checkpoint_sha256,
            lineage_digest=stable_digest({"role": "NOT_APPLICABLE_COMMON_PARENT"}),
            schedule_digest=stable_digest({"role": "COMMON_PARENT_NR", "epochs": [1, 120]}),
            resume_from_receipt_digest="0" * 64 if resume_context is None else resume_context.receipt_chain_digest,
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
        trainer, binding, trainer_binding = build_prepared_trainer(
            repository_root=arguments.repository_root,
            identity_manifest=arguments.identity_manifest,
            trainer_overrides_path=arguments.trainer_overrides,
            identity=identity,
            output_root=trainer_setup_root,
            asset_registry_path=arguments.asset_registry,
        )
        result = run_prepared_common_parent(
            trainer=trainer,
            identity=identity,
            output_root=arguments.output_root,
            release_authorization=arguments.release_authorization,
            release_trust_policy=arguments.release_trust_policy,
            release_expected_bindings=authorization["expected_bindings"],
            prepared_trainer_binding=trainer_binding,
            formal_input_binding=authorization["formal_input_binding"],
            execution_claim_binding=execution_claim,
            resume_context=resume_context,
            execution_mode="formal",
        )
        return {
            **result,
            "upstream_binding_digest": binding.binding_digest,
            "prepared_trainer_binding": trainer_binding,
            "formal_authorization": authorization,
            "execution_claim": execution_claim,
            "resume_context": None if resume_context is None else resume_context.as_dict(),
        }

    return run_cli("run_common_parent", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
