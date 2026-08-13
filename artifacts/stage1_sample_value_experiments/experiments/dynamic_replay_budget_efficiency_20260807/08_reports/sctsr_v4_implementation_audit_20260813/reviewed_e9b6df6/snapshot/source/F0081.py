from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .errors import ErrorCode, SctsrError
from .qrad_contract import validate_candidate_signal_semantic
from .rate_spec import RateSemantic, ReplayRateSpec


_HEX = frozenset("0123456789ABCDEF")
ALLOWED_PREDICTOR_FAMILIES = {
    "LOGISTIC_REGRESSION",
    "SHALLOW_DECISION_TREE",
    "MONOTONE_LOW_CAPACITY",
}


def _is_sha256(value: str | None) -> bool:
    return value is not None and len(value) == 64 and value == value.upper() and all(char in _HEX for char in value)


@dataclass(frozen=True, slots=True)
class ShortBranchSpec:
    anchor_epoch: int
    horizon: int
    microset_rate: ReplayRateSpec
    treatment_pool_digest: str
    matched_r2_pool_digest: str
    enabled: bool = False
    matched_quota_digest: str | None = None
    treatment_sample_ids: tuple[str, ...] = ()
    matched_r2_sample_ids: tuple[str, ...] = ()

    def validate(
        self,
        *,
        phase1_gate_passed: bool = False,
        phase1_gate_receipt_sha256: str | None = None,
        phase2_release_authorized: bool = False,
    ) -> None:
        if self.anchor_epoch not in {120, 140, 160}:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Short-branch anchor must be E120, E140, or E160")
        if self.horizon not in {5, 10} or self.anchor_epoch + self.horizon > 200:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Short-branch horizon must be 5 or 10 and end no later than E200")
        if self.microset_rate.semantic is not RateSemantic.IDENTITY_POOL_RATE or self.microset_rate.numerator <= 0:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Short-branch microset size must be a positive rational identity-pool percentage")
        if not _is_sha256(self.treatment_pool_digest) or not _is_sha256(self.matched_r2_pool_digest):
            raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Short-branch pools require canonical content digests")
        if self.treatment_pool_digest == self.matched_r2_pool_digest:
            raise SctsrError(ErrorCode.R2_OVERLAPS_T, "Treatment and matched-R2 microsets may not share an identity digest")
        if self.treatment_sample_ids or self.matched_r2_sample_ids:
            treatment = set(self.treatment_sample_ids)
            comparator = set(self.matched_r2_sample_ids)
            if not treatment or not comparator or len(treatment) != len(self.treatment_sample_ids) or len(comparator) != len(self.matched_r2_sample_ids):
                raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "Short-branch microset memberships are empty or duplicated")
            overlap = sorted(treatment & comparator)
            if overlap:
                raise SctsrError(ErrorCode.R2_OVERLAPS_T, "Treatment and matched-R2 microsets overlap", observed=overlap)
            if len(treatment) != len(comparator) or not _is_sha256(self.matched_quota_digest):
                raise SctsrError(ErrorCode.R2_QUOTA_INFEASIBLE, "Short-branch R2 lacks equal-size exact-quota evidence")
        if self.enabled:
            raise SctsrError(
                ErrorCode.SHORT_BRANCH_DISABLED,
                "Short-branch execution is disabled in this frozen v4 baseline even if caller booleans claim prerequisites",
                observed={
                    "phase1_gate_passed": phase1_gate_passed,
                    "gate_receipt_is_sha256": _is_sha256(phase1_gate_receipt_sha256),
                    "phase2_release_authorized_claim": phase2_release_authorized,
                },
                required_action="Create a new preregistration and cryptographically verified phase-2 release.",
            )


@dataclass(frozen=True, slots=True)
class UtilityOutcome:
    safety_pass: bool
    primary_nauc_delta: float
    tn_at_fn95_delta: int
    fn_at_tn68253_delta: int | None

    def validate(self) -> None:
        if not math.isfinite(float(self.primary_nauc_delta)):
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Short-branch primary delta is not finite")
        if type(self.tn_at_fn95_delta) is not int:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Short-branch TN delta is not integral")
        if self.fn_at_tn68253_delta is not None and type(self.fn_at_tn68253_delta) is not int:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Short-branch FN delta is invalid")
        if self.fn_at_tn68253_delta is None and self.safety_pass:
            raise SctsrError(ErrorCode.FRONTIER_INVALID, "An unreachable safety anchor cannot pass the safety gate")
        if self.tn_at_fn95_delta < 0 and self.fn_at_tn68253_delta is not None and self.fn_at_tn68253_delta > 0 and self.safety_pass:
            raise SctsrError(ErrorCode.FRONTIER_INVALID, "A dual-end degradation cannot pass the safety gate")


@dataclass(frozen=True, slots=True)
class PredictorFeatureRow:
    microset_id: str
    training_seed: int
    anchor_epoch: int
    horizon: int
    q_reliability_features: Mapping[str, float]
    r_reducible_features: Mapping[str, float]
    a_direction_features: Mapping[str, float] | None
    a_unavailable_reason: str | None
    d_set_coverage_features: Mapping[str, float]
    state_features: Mapping[str, float]
    cumulative_replay_exposure: int
    outcome: UtilityOutcome
    candidate_signal_semantic: str = "CANDIDATE_SIGNAL_NOT_UTILITY"

    def validate(self) -> None:
        if not self.microset_id.strip() or self.anchor_epoch not in {120, 140, 160} or self.horizon not in {5, 10}:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Predictor feature row has an invalid identity or short-branch state")
        validate_candidate_signal_semantic("predictor_features", self.candidate_signal_semantic)
        if self.a_direction_features is None and self.a_unavailable_reason != "BLOCKED_BY_VAL_TARGET":
            raise SctsrError(ErrorCode.BLOCKED_BY_VAL_TARGET, "Missing A features require the registered val_target blocker")
        if self.a_direction_features is not None:
            raise SctsrError(ErrorCode.BLOCKED_BY_VAL_TARGET, "Current baseline cannot contain computed A features")
        for name, features in (
            ("Q", self.q_reliability_features),
            ("R", self.r_reducible_features),
            ("D", self.d_set_coverage_features),
            ("state", self.state_features),
        ):
            if not features or any(not math.isfinite(float(value)) for value in features.values()):
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, f"Predictor {name} feature block is empty or non-finite")
            if any("utility" in str(key).lower() or "score" in str(key).lower() for key in features):
                raise SctsrError(ErrorCode.CANDIDATE_SIGNAL_AS_UTILITY_FORBIDDEN, f"Predictor {name} feature name implies utility or a blended score")
        if self.cumulative_replay_exposure < 0:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Cumulative replay exposure is negative")
        self.outcome.validate()


def require_predictor_training_allowed(
    *,
    phase1_gate_passed: bool,
    model_family: str,
    phase1_gate_receipt_sha256: str | None = None,
    phase2_release_authorized: bool = False,
) -> None:
    if model_family not in ALLOWED_PREDICTOR_FAMILIES:
        raise SctsrError(ErrorCode.SELECTOR_TRAINING_FORBIDDEN, "Only registered low-capacity predictor interfaces are allowed; RL is forbidden")
    raise SctsrError(
        ErrorCode.SELECTOR_TRAINING_FORBIDDEN,
        "Predictor training is not implemented or authorized in this frozen v4 baseline",
        observed={
            "phase1_gate_passed": phase1_gate_passed,
            "gate_receipt_is_sha256": _is_sha256(phase1_gate_receipt_sha256),
            "phase2_release_authorized_claim": phase2_release_authorized,
        },
        required_action="Create a new preregistration and verified phase-2 release after phase-1 evidence passes.",
    )
