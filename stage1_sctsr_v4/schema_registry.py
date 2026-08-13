from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .errors import ErrorCode, SctsrError
from .serialization import stable_digest


REQUIRED_SCHEMAS = {
    "arms": "stage1.sctsr.arms.v1",
    "artifact_index": "stage1.sctsr.artifact_index.v1",
    "asset_registry": "stage1.sctsr.asset_registry.v1",
    "checkpoint": "stage1.sctsr.checkpoint.v1",
    "cli_receipt": "stage1.sctsr.cli_receipt.v1",
    "closeout": "stage1.sctsr.closeout.v2",
    "combined_artifact_index": "stage1.sctsr.combined_artifact_index.v1",
    "completion_audit": "stage1.sctsr.completion_audit.v1",
    "contract": "stage1.sctsr.contract.v1",
    "contrast_family": "stage1.sctsr.contrast_family.v1",
    "disabled_phase2": "stage1.sctsr.disabled_phase2.v1",
    "epoch_artifact_index": "stage1.sctsr.epoch_artifact_index.v1",
    "epoch_evidence_summary": "stage1.sctsr.epoch_evidence_summary.v1",
    "epoch_exposure_ledger": "stage1.sctsr.epoch_exposure_ledger.v1",
    "epoch_generation": "stage1.sctsr.epoch_generation.v2",
    "epoch_receipt": "stage1.sctsr.epoch_receipt.v1",
    "epoch_transaction_identity": "stage1.sctsr.epoch_transaction_identity.v1",
    "error": "stage1.sctsr.error.v1",
    "formal_branch_receipt": "stage1.sctsr.formal_branch_receipt.v1",
    "formal_identity": "stage1.sctsr.formal_identity.v1",
    "formal_parent_receipt": "stage1.sctsr.formal_parent_receipt.v1",
    "formal_release": "stage1.sctsr.formal_release.v1",
    "formal_run_manifest": "stage1.sctsr.formal_run_manifest.v1",
    "frontier": "stage1.sctsr.frontier.v1",
    "frontier_summary": "stage1.sctsr.frontier_summary.v1",
    "identity_group_membership": "stage1.sctsr.identity_group_membership.v1",
    "identity_pool_manifest": "stage1.sctsr.identity_pool_manifest.v1",
    "identity_pool_quota_audit": "stage1.sctsr.identity_pool_quota_audit.v1",
    "logical_artifact_index": "stage1.sctsr.logical_artifact_index.v1",
    "occurrence_ledger": "stage1.sctsr.occurrence_ledger.v1",
    "optimizer_step_ledger": "stage1.sctsr.optimizer_step_ledger.v1",
    "parent_artifact_binding": "stage1.sctsr.parent_artifact_binding.v1",
    "prediction_artifact": "stage1.sctsr.prediction_artifact.v2",
    "prediction_artifact_summary": "stage1.sctsr.prediction_artifact_summary.v2",
    "prepared_trainer_binding": "stage1.sctsr.prepared_trainer_binding.v1",
    "prepared_trainer_setup_failure": "stage1.sctsr.prepared_trainer_setup_failure.v1",
    "quarantine_receipt": "stage1.sctsr.quarantine_receipt.v1",
    "release_trust": "stage1.sctsr.release_trust.v1",
    "resource_telemetry": "stage1.sctsr.resource_telemetry.v1",
    "resume_prepared_trainer_receipt": "stage1.sctsr.resume_prepared_trainer_receipt.v1",
    "rolling_recovery_pointer": "stage1.sctsr.rolling_recovery_pointer.v2",
    "runtime_policy": "stage1.sctsr.runtime_policy.v1",
    "schedule": "stage1.sctsr.schedule.v1",
    "schema_registry": "stage1.sctsr.schema_registry.v1",
    "seed_registry": "stage1.sctsr.seed_registry.v1",
    "selection_ledger": "stage1.sctsr.selection_ledger.v1",
    "source_tree_manifest": "stage1.sctsr.source_tree_manifest.v1",
    "split_identity_bundle": "stage1.sctsr.split_identity_bundle.v1",
    "synthetic_artifact_index": "stage1.sctsr.synthetic_artifact_index.v1",
    "synthetic_asset_registry": "stage1.sctsr.synthetic_asset_registry.v1",
    "synthetic_branch_receipt": "stage1.sctsr.synthetic_branch_receipt.v1",
    "synthetic_canary_receipt": "stage1.sctsr.synthetic_canary_receipt.v1",
    "synthetic_checkpoint_resume_receipt": "stage1.sctsr.synthetic_checkpoint_resume_receipt.v1",
    "synthetic_failure_injection_summary": "stage1.sctsr.synthetic_failure_injection_summary.v1",
    "synthetic_mechanism_audit": "stage1.sctsr.synthetic_mechanism_audit.v1",
    "synthetic_parent_receipt": "stage1.sctsr.synthetic_parent_receipt.v1",
    "synthetic_run_manifest": "stage1.sctsr.synthetic_run_manifest.v1",
    "training_identity_manifest_binding": "stage1.sctsr.training_identity_manifest_binding.v1",
    "training_identity_manifest_summary": "stage1.sctsr.training_identity_manifest_summary.v1",
    "upstream_files_manifest": "stage1.sctsr.upstream_files_manifest.v2",
}


@dataclass(frozen=True, slots=True)
class SchemaRegistry:
    schema_version: str
    schemas: Mapping[str, str]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SchemaRegistry":
        return cls(str(value.get("schema_version", "")), dict(value.get("schemas", {})))

    @property
    def digest(self) -> str:
        return stable_digest({"schema_version": self.schema_version, "schemas": dict(self.schemas)})

    def validate(self) -> None:
        if self.schema_version != "stage1.sctsr.schema_registry.v1":
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Unknown schema registry version")
        observed = dict(self.schemas)
        if observed != REQUIRED_SCHEMAS:
            raise SctsrError(
                ErrorCode.SCHEMA_VALIDATION_FAILED,
                "Schema registry must exactly match all frozen public schema identities",
                observed=observed,
                expected=REQUIRED_SCHEMAS,
            )
