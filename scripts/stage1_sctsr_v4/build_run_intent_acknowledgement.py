from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_sctsr_v4.cli_support import add_output_argument, run_cli
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.formal_cli import load_formal_identity
from stage1_sctsr_v4.run_intent import build_run_intent_acknowledgement, derive_formal_run_intent_context
from stage1_sctsr_v4.serialization import atomic_write_json, sha256_file, stable_digest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive and explicitly acknowledge one exact SCTSR START/RESUME job; this does not claim or train"
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--action", choices=("START", "RESUME"), required=True)
    parser.add_argument("--run-role", choices=("COMMON_PARENT", "BRANCH"), required=True)
    parser.add_argument("--logical-run-id", required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--formal-identity", type=Path, required=True)
    parser.add_argument("--trainer-overrides", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--asset-registry", type=Path, required=True)
    parser.add_argument("--parent-checkpoint-sha256", required=True)
    parser.add_argument("--schedule-digest", required=True)
    parser.add_argument("--identity-pool-binding-digest", required=True)
    parser.add_argument("--release-authorization", type=Path, required=True)
    parser.add_argument("--execution-token", type=Path, required=True)
    parser.add_argument("--execution-claim-root", type=Path, required=True)
    parser.add_argument("--resume-checkpoint-sha256", required=True)
    parser.add_argument("--resume-receipt-digest", required=True)
    parser.add_argument("--runbook-manifest", type=Path, required=True)
    parser.add_argument("--acknowledgement-id", required=True)
    parser.add_argument("--operator-agent-id", required=True)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--acknowledge-all-required-statements", action="store_true")
    parser.add_argument("--acknowledgement-output", type=Path, required=True)
    add_output_argument(parser)
    arguments = parser.parse_args()

    def action():
        if not arguments.acknowledge_all_required_statements:
            raise SctsrError(
                ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED,
                "The operator must explicitly pass --acknowledge-all-required-statements after reading the frozen runbook",
            )
        if arguments.acknowledgement_output.resolve() == arguments.output.resolve():
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Acknowledgement and CLI receipt require different paths")
        if arguments.acknowledgement_output.exists():
            raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Run-intent acknowledgement is immutable; choose a new output path")
        identity = load_formal_identity(arguments.formal_identity)
        if identity.training_seed != arguments.training_seed:
            raise SctsrError(ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED, "Acknowledged seed differs from the formal identity")
        context = derive_formal_run_intent_context(
            runbook_manifest_path=arguments.runbook_manifest,
            repository_root=arguments.repository_root,
            output_root=arguments.output_root,
            action=arguments.action,
            run_role=arguments.run_role,
            logical_run_id=arguments.logical_run_id,
            arm_id=arguments.arm_id,
            training_seed=arguments.training_seed,
            formal_identity_path=arguments.formal_identity,
            trainer_overrides_path=arguments.trainer_overrides,
            identity_manifest_path=arguments.identity_manifest,
            source_tree_digest=identity.source_tree_digest,
            contract_digest=identity.effective_contract_digest,
            asset_registry_digest=identity.asset_registry_digest,
            runtime_config_digest=identity.runtime_config_digest,
            asset_registry_path=arguments.asset_registry,
            parent_checkpoint_sha256=arguments.parent_checkpoint_sha256,
            schedule_digest=arguments.schedule_digest,
            identity_pool_binding_digest=arguments.identity_pool_binding_digest,
            release_manifest_path=arguments.release_authorization,
            execution_token_path=arguments.execution_token,
            claim_registry_root=arguments.execution_claim_root,
            resume_checkpoint_sha256=arguments.resume_checkpoint_sha256,
            resume_receipt_digest=arguments.resume_receipt_digest,
        )
        payload = build_run_intent_acknowledgement(
            context=context,
            acknowledgement_id=arguments.acknowledgement_id,
            operator_agent_id=arguments.operator_agent_id,
            machine_id=arguments.machine_id,
            created_at_utc=datetime.now(timezone.utc),
        )
        atomic_write_json(arguments.acknowledgement_output, payload)
        return {
            "status": "PASS_ACKNOWLEDGED_NOT_EXECUTED",
            "acknowledgement_path": arguments.acknowledgement_output.resolve().as_posix(),
            "acknowledgement_bytes": arguments.acknowledgement_output.stat().st_size,
            "acknowledgement_sha256": sha256_file(arguments.acknowledgement_output),
            "acknowledgement_digest": payload["acknowledgement_digest"],
            "context_digest": stable_digest(context),
            "formal_training_started": False,
        }

    return run_cli("build_run_intent_acknowledgement", arguments.output, action)


if __name__ == "__main__":
    raise SystemExit(main())
