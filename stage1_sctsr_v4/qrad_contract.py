from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import ErrorCode, SctsrError
from .serialization import sha256_file, stable_digest


QRAD_DEFINITIONS = {
    "Q": "RELIABILITY",
    "R": "RESIDUAL_REDUCIBLE_LEARNABILITY",
    "A": "INDEPENDENT_FN95_TARGET_DIRECTION",
    "D": "SET_CONDITIONED_COVERAGE_REDUNDANCY",
}
CANDIDATE_SIGNALS = QRAD_DEFINITIONS
ALLOWED_QRAD_USES = {"SEQUENTIAL_GATE", "STRATIFICATION", "PREREGISTERED_FACTORIAL"}
ALLOWED_CANDIDATE_SEMANTICS = {"CANDIDATE_SIGNAL_NOT_UTILITY", "DIAGNOSTIC_NOT_UTILITY"}
REQUIRED_DISJOINT_ROLES = frozenset({"train", "OOF", "val_model", "val_cal", "val_op", "test"})
_HEX = frozenset("0123456789ABCDEF")


def _is_sha256(value: str | None) -> bool:
    return value is not None and len(value) == 64 and value == value.upper() and all(char in _HEX for char in value)


def _walk(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, Mapping):
        for key, child in value.items():
            token = str(key)
            yield path + (token,), child
            yield from _walk(child, path + (token,))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk(child, path + (str(index),))


def reject_weighted_qrad(config: Mapping[str, object]) -> None:
    forbidden_exact = {
        "weights", "weighted_score", "score_weights", "q_weight", "r_weight", "a_weight", "d_weight",
        "coefficients", "linear_combination", "aggregate_score", "ranking_score",
    }
    arithmetic = re.compile(r"(?:^|[^a-z])(q|r|a|d)\s*(?:\*|×)|(?:\*|×)\s*(q|r|a|d)(?:$|[^a-z])", re.IGNORECASE)
    for path, value in _walk(config):
        key = path[-1].strip().lower().replace("-", "_")
        if key == "weighted_total_score_allowed" and value is False:
            continue
        if key in forbidden_exact or key.endswith("_weight") or "weighted" in key:
            raise SctsrError(
                ErrorCode.WEIGHTED_QRAD_FORBIDDEN,
                "Weighted Q/R/A/D total score is forbidden at every nesting level",
                failing_field=".".join(path),
            )
        if isinstance(value, str):
            normalized = value.replace(" ", "")
            if arithmetic.search(value) or any(token in normalized.lower() for token in ("wq*q", "wr*r", "wa*a", "wd*d", "q+r+a+d")):
                raise SctsrError(
                    ErrorCode.WEIGHTED_QRAD_FORBIDDEN,
                    "Arithmetic aggregation of Q/R/A/D is forbidden",
                    failing_field=".".join(path),
                    observed=value,
                )


@dataclass(frozen=True, slots=True)
class QradFactorContract:
    usage: str
    gate_order: tuple[str, ...]
    comparator_by_factor: Mapping[str, str]
    candidate_signal_semantic: str = "CANDIDATE_SIGNAL_NOT_UTILITY"

    def validate(self) -> None:
        if self.usage not in ALLOWED_QRAD_USES:
            raise SctsrError(ErrorCode.WEIGHTED_QRAD_FORBIDDEN, "Q/R/A/D may only be used as gates, strata, or preregistered factors")
        if self.gate_order != ("Q", "R", "A", "D"):
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Q/R/A/D sequential contract must preserve Q then R then A then D")
        expected = {"R", "A", "D"}
        if set(self.comparator_by_factor) != expected:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "R/A/D each require an explicit method-matched comparator identity")
        values = tuple(self.comparator_by_factor[factor] for factor in ("R", "A", "D"))
        if any(not value.strip() for value in values) or len(set(values)) != len(values):
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "One opaque R2 comparator may not absorb R/A/D semantics")
        validate_candidate_signal_semantic("Q/R/A/D", self.candidate_signal_semantic)


@dataclass(frozen=True, slots=True)
class TargetAlignmentConfig:
    enabled: bool
    val_target_manifest_sha256: str | None = None
    identity_disjoint: bool = False
    group_disjoint: bool = False
    local_fn95_formula: str | None = None
    checkpoint_lag: int | None = None
    val_target_manifest_path: str | None = None
    split_role: str = "val_target"
    identity_disjoint_from_roles: tuple[str, ...] = ()
    formula_digest: str | None = None

    def validate_future_registration(self) -> None:
        if self.split_role in {"test", "val_op"}:
            raise SctsrError(ErrorCode.TEST_ACCESS_FORBIDDEN if self.split_role == "test" else ErrorCode.VAL_OP_SELECTION_FORBIDDEN, "A target alignment source may not be test or val_op")
        if self.split_role != "val_target":
            raise SctsrError(ErrorCode.BLOCKED_BY_VAL_TARGET, "A target alignment split must be registered as val_target")
        if not _is_sha256(self.val_target_manifest_sha256) or not self.val_target_manifest_path:
            raise SctsrError(ErrorCode.BLOCKED_BY_VAL_TARGET, "val_target requires a frozen manifest path and SHA-256")
        manifest = Path(self.val_target_manifest_path)
        if not manifest.is_file() or sha256_file(manifest) != self.val_target_manifest_sha256:
            raise SctsrError(ErrorCode.BLOCKED_BY_VAL_TARGET, "val_target manifest bytes do not match the frozen SHA-256")
        if not self.identity_disjoint or set(self.identity_disjoint_from_roles) != REQUIRED_DISJOINT_ROLES:
            raise SctsrError(ErrorCode.BLOCKED_BY_VAL_TARGET, "val_target identity disjointness is incomplete")
        if not self.group_disjoint:
            raise SctsrError(ErrorCode.BLOCKED_BY_VAL_TARGET, "val_target must be group-disjoint")
        expected_formula_digest = stable_digest(
            {"local_fn95_formula": self.local_fn95_formula, "checkpoint_lag": self.checkpoint_lag}
        )
        if not self.local_fn95_formula or self.formula_digest != expected_formula_digest:
            raise SctsrError(ErrorCode.BLOCKED_BY_VAL_TARGET, "The local FN95 target formula is not frozen")
        if type(self.checkpoint_lag) is not int or self.checkpoint_lag <= 0:
            raise SctsrError(ErrorCode.BLOCKED_BY_VAL_TARGET, "Target-alignment checkpoint lag must be a frozen positive integer")

    def validate(self, *, val_target_available: bool = False) -> None:
        if not self.enabled:
            return
        # The current baseline has no val_target asset or release binding.  Do
        # not let a caller-controlled boolean turn the scaffold into an arm or
        # cause a gradient forward.  A future preregistration must replace this
        # baseline guard after registering all prerequisites.
        raise SctsrError(
            ErrorCode.BLOCKED_BY_VAL_TARGET,
            "A/gradient alignment is disabled in this SCTSR v4 baseline because no independent frozen val_target exists",
            observed={"caller_claimed_val_target_available": bool(val_target_available)},
            required_action="Register a new preregistration and signed release; do not mutate this frozen baseline.",
        )


def validate_candidate_signal_semantic(name: str, semantic: str) -> None:
    if semantic not in ALLOWED_CANDIDATE_SEMANTICS:
        raise SctsrError(
            ErrorCode.CANDIDATE_SIGNAL_AS_UTILITY_FORBIDDEN,
            f"Candidate signal {name} must use an exact registered non-utility semantic",
            observed=semantic,
            expected=sorted(ALLOWED_CANDIDATE_SEMANTICS),
        )


def validate_disabled_phase2_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    expected_fields = {
        "schema_version", "state", "enabled", "current_loss_arm_state", "predictor", "qrad",
        "short_branch", "target_alignment", "weighted_total_score_allowed",
    }
    if set(payload) != expected_fields:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Disabled phase-2 config fields do not exactly match the registry")
    if payload["schema_version"] != "stage1.sctsr.disabled_phase2.v1" or payload["state"] != "HELD_UNTIL_PHASE1_GATE":
        raise SctsrError(ErrorCode.SHORT_BRANCH_DISABLED, "Phase-2 config is not in the registered held state")
    if payload["enabled"] or payload["weighted_total_score_allowed"]:
        raise SctsrError(ErrorCode.WEIGHTED_QRAD_FORBIDDEN, "Phase-2 or weighted score was enabled")
    predictor = payload["predictor"]
    short_branch = payload["short_branch"]
    alignment = payload["target_alignment"]
    if predictor.get("enabled") or predictor.get("reinforcement_learning_allowed") or short_branch.get("enabled") or alignment.get("enabled"):
        raise SctsrError(ErrorCode.SELECTOR_TRAINING_FORBIDDEN, "A held phase-2 component was enabled")
    if alignment.get("state") != "BLOCKED_BY_VAL_TARGET" or alignment.get("val_target_available") is not False:
        raise SctsrError(ErrorCode.BLOCKED_BY_VAL_TARGET, "Current A scaffold must explicitly record val_target unavailable")
    reject_weighted_qrad(payload)
    return {
        "status": "PASS_HELD_DISABLED",
        "path": source.resolve().as_posix(),
        "sha256": sha256_file(source),
        "contract_digest": stable_digest(payload),
        "qrad_definitions": dict(QRAD_DEFINITIONS),
    }
