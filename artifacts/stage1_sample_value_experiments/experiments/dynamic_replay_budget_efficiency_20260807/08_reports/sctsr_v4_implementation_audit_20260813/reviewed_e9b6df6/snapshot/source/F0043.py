from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .checkpointing import validate_checkpoint_payload
from .errors import ErrorCode, SctsrError
from .rate_spec import ZERO_RATE, ReplayRateSpec
from .serialization import stable_digest


@dataclass(frozen=True, slots=True)
class CommonParentSpec:
    parent_id: str
    training_seed: int
    canonical_training_lock_sha256: str
    initial_checkpoint_sha256: str
    base_manifest_sha256: str
    source_tree_digest: str
    runtime_config_digest: str
    asset_registry_digest: str
    expected_base_steps_per_epoch: int = 938
    checkpoint_schema_version: str = "stage1.sctsr.checkpoint.v1"
    epoch_start: int = 1
    epoch_end: int = 120
    arm_id: str = "COMMON_PARENT_NR"
    replay_rate: ReplayRateSpec = ZERO_RATE

    @property
    def digest(self) -> str:
        return stable_digest(self)

    def validate(self) -> None:
        if (self.epoch_start, self.epoch_end, self.arm_id) != (1, 120, "COMMON_PARENT_NR"):
            raise SctsrError(ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, "Common parent must be E1-E120 no replay")
        if self.replay_rate.numerator != 0:
            raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "Common parent replay rate must be zero")
        if self.expected_base_steps_per_epoch != 938:
            raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "Canonical parent requires 938 base steps per epoch")
        for field in (
            "canonical_training_lock_sha256",
            "initial_checkpoint_sha256",
            "base_manifest_sha256",
            "source_tree_digest",
            "runtime_config_digest",
            "asset_registry_digest",
        ):
            value = str(getattr(self, field))
            if len(value) != 64 or value.upper() != value or any(char not in "0123456789ABCDEF" for char in value):
                raise SctsrError(ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, "Common-parent identity must use canonical SHA-256", failing_field=field, observed=value)
        if not self.parent_id.strip():
            raise SctsrError(ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, "Common-parent ID is empty")


def validate_parent_checkpoint(payload: Mapping[str, Any], spec: CommonParentSpec) -> None:
    spec.validate()
    validate_checkpoint_payload(payload, expected_epoch=120)
    checks = {
        "training_seed": spec.training_seed,
        "canonical_training_lock_sha256": spec.canonical_training_lock_sha256,
        "initial_checkpoint_sha256": spec.initial_checkpoint_sha256,
        "base_manifest_sha256": spec.base_manifest_sha256,
        "source_tree_digest": spec.source_tree_digest,
    }
    checks["runtime_config_digest"] = spec.runtime_config_digest
    checks["asset_registry_digest"] = spec.asset_registry_digest
    for field, expected in checks.items():
        if payload[field] != expected:
            raise SctsrError(ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, "Parent checkpoint identity mismatch", failing_field=field, observed=payload[field], expected=expected)
    expected_global_steps = spec.epoch_end * spec.expected_base_steps_per_epoch
    if int(payload["global_step"]) != expected_global_steps:
        raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "Parent checkpoint global step is not E120 x 938", observed=payload["global_step"], expected=expected_global_steps)
    if int(payload["base_sampler_generation"]) != spec.epoch_end:
        raise SctsrError(ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, "Parent sampler generation is not E120", observed=payload["base_sampler_generation"], expected=spec.epoch_end)
    if int(payload["ema_updates"]) != expected_global_steps:
        raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "Parent EMA update count is not E120 x 938", observed=payload["ema_updates"], expected=expected_global_steps)
