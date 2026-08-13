from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.cli_support import add_execution_arguments, add_output_argument, require_receipt_outside_artifact_root, run_cli
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.formal_cli import build_prepared_trainer, load_formal_identity, prepare_formal_authorization
from stage1_sctsr_v4.formal_training import run_prepared_common_parent
from stage1_sctsr_v4.synthetic_execution import run_synthetic_common_parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the E1-E120 SCTSR common parent")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--identity-manifest", type=Path)
    parser.add_argument("--trainer-overrides", type=Path)
    parser.add_argument("--formal-identity", type=Path)
    add_execution_arguments(parser)
    add_output_argument(parser)
    arguments = parser.parse_args()

    def action():
        require_receipt_outside_artifact_root(arguments.output, arguments.output_root)
        if arguments.execution_mode == "synthetic":
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
        trainer, binding, trainer_binding = build_prepared_trainer(
            repository_root=arguments.repository_root,
            identity_manifest=arguments.identity_manifest,
            trainer_overrides_path=arguments.trainer_overrides,
            identity=identity,
            output_root=arguments.output_root,
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
            execution_mode="formal",
        )
        return {
            **result,
            "upstream_binding_digest": binding.binding_digest,
            "prepared_trainer_binding": trainer_binding,
            "formal_authorization": authorization,
        }

    return run_cli("run_common_parent", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
