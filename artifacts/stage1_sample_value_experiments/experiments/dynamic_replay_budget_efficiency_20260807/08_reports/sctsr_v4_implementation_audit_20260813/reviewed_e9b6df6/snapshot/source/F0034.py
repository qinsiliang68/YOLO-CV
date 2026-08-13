from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .errors import ErrorCode, SctsrError


class ArmId(str, Enum):
    NR = "NR"
    R1_U = "R1_U"
    R2_U = "R2_U"
    T_U = "T_U"
    R2_F = "R2_F"
    T_F = "T_F"
    T_TO_R2_AT_160 = "T_TO_R2_AT_160"
    T_TO_NR_AT_160 = "T_TO_NR_AT_160"


PHASE1_ARM_ORDER = tuple(ArmId)


@dataclass(frozen=True, slots=True)
class SctsrArmSpec:
    arm_id: ArmId
    display_name: str
    phase: str
    identity_policy: str
    schedule_policy: str
    scientific_role: str
    allowed_comparators: tuple[ArmId, ...]
    epoch_start: int = 121
    epoch_end: int = 200
    fallback_epoch: int | None = None
    fallback_identity_policy: str | None = None
    requires_common_parent: bool = True
    optimizer_step_policy: str = "BASE_STEP_LOCK"
    formal_state: str = "HELD"

    def validate(self) -> None:
        if not self.requires_common_parent:
            raise SctsrError(ErrorCode.BRANCH_LINEAGE_MISMATCH, "Every phase-1 arm requires the E120 common parent")
        if self.optimizer_step_policy != "BASE_STEP_LOCK":
            raise SctsrError(ErrorCode.REPLAY_ADDED_OPTIMIZER_STEP, "Phase-1 arms must use BASE_STEP_LOCK")
        if (self.epoch_start, self.epoch_end) != (121, 200):
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Phase-1 child range is frozen at E121-E200")
        if self.formal_state != "HELD":
            raise SctsrError(ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED, "Phase-1 arm must remain HELD before release")


def default_phase1_arms() -> tuple[SctsrArmSpec, ...]:
    specs = (
        SctsrArmSpec(ArmId.NR, "No replay", "PHASE1", "NONE", "NR", "BASE_PROCESS", ()),
        SctsrArmSpec(ArmId.R1_U, "Global random uniform", "PHASE1", "R1_GLOBAL_RANDOM", "U", "IDENTITY_NEUTRAL_RANDOM_REPLAY", (ArmId.NR,)),
        SctsrArmSpec(ArmId.R2_U, "Matched random uniform", "PHASE1", "R2_MATCHED_RANDOM", "U", "STRICT_IDENTITY_COMPARATOR", (ArmId.T_U, ArmId.NR)),
        SctsrArmSpec(ArmId.T_U, "T stress-set uniform", "PHASE1", "T_STRESS", "U", "HISTORICAL_SIGN_REVERSAL_STRESS_SET", (ArmId.R2_U,)),
        SctsrArmSpec(ArmId.R2_F, "Matched random front-loaded", "PHASE1", "R2_MATCHED_RANDOM", "F", "STRICT_TIMING_COMPARATOR", (ArmId.T_F, ArmId.R2_U)),
        SctsrArmSpec(ArmId.T_F, "T stress-set front-loaded", "PHASE1", "T_STRESS", "F", "TREATMENT_TIMING_ARM", (ArmId.R2_F, ArmId.T_U)),
        SctsrArmSpec(ArmId.T_TO_R2_AT_160, "T then R2 fallback", "PHASE1", "T_STRESS", "T_TO_R2", "STOP_TARGETED_FALLBACK_POLICY", (ArmId.T_U,), fallback_epoch=160, fallback_identity_policy="R2_MATCHED_RANDOM"),
        SctsrArmSpec(ArmId.T_TO_NR_AT_160, "T then no replay", "PHASE1", "T_STRESS", "T_TO_NR", "STOP_ALL_REPLAY_POLICY", (ArmId.T_TO_R2_AT_160,), fallback_epoch=160, fallback_identity_policy="NONE"),
    )
    for spec in specs:
        spec.validate()
    return specs


@dataclass(frozen=True, slots=True)
class ContrastSpec:
    contrast_id: str
    treatment: ArmId
    comparator: ArmId
    question: str
    prohibited_inference: str


def default_contrasts() -> tuple[ContrastSpec, ...]:
    return (
        ContrastSpec("C01", ArmId.T_U, ArmId.R2_U, "T identity effect under U", "Does not prove intrinsic T value"),
        ContrastSpec("C02", ArmId.T_F, ArmId.R2_F, "T identity effect under F", "Does not isolate timing alone"),
        ContrastSpec("C03", ArmId.T_F, ArmId.T_U, "Timing effect with fixed T identity/dose/multiplicity", "Does not generalize to arbitrary selector"),
        ContrastSpec("C04", ArmId.R2_F, ArmId.R2_U, "Timing effect for matched random identity", "Does not prove T value"),
        ContrastSpec("C05", ArmId.T_TO_R2_AT_160, ArmId.T_U, "Fallback policy after E160", "Not a pure stop effect"),
        ContrastSpec("C06", ArmId.T_TO_NR_AT_160, ArmId.T_TO_R2_AT_160, "Late replay versus no replay after common prefix", "Late dose intentionally differs"),
        ContrastSpec("C07", ArmId.R2_U, ArmId.NR, "General matched-random exposure effect", "Does not prove selection"),
        ContrastSpec("C08", ArmId.R1_U, ArmId.NR, "General global-random exposure effect", "Does not prove matching"),
    )


def validate_phase1_registry(arms: Iterable[SctsrArmSpec]) -> tuple[SctsrArmSpec, ...]:
    values = tuple(arms)
    ids = tuple(item.arm_id for item in values)
    if ids != PHASE1_ARM_ORDER:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Phase-1 arm set and order must exactly match the taskbook", observed=[x.value for x in ids], expected=[x.value for x in PHASE1_ARM_ORDER])
    if len(set(ids)) != 8:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Phase-1 registry must contain exactly eight unique arms")
    for item in values:
        item.validate()
    return values
