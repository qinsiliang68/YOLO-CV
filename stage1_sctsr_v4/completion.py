from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Sequence

from .errors import ErrorCode, SctsrError


REQUIRED_IMPLEMENTATION_CHECKS = frozenset(
    {
        "source_tree_manifest",
        "changed_file_ledger",
        "frozen_asset_validation",
        "contract_validation",
        "rate_validation",
        "t_identity_validation",
        "r1_validation",
        "r2_zero_overlap_quota_fail_closed",
        "schedule_parity",
        "parent_checkpoint_completeness",
        "branch_lineage",
        "logical_artifact_index",
        "fixed_base_step",
        "replay_gradient",
        "rng_isolation",
        "bn_isolation",
        "exposure_conservation",
        "telemetry",
        "prediction_identity",
        "frontier",
        "statistics",
        "recovery",
        "synthetic_canary",
        "v3_regression",
        "v4_test_suite",
        "known_blockers_registered",
        "prohibited_side_effect_audit",
        "self_audit",
    }
)

BLOCKABLE_READINESS_CHECKS = frozenset(
    {
        "formal_training_release",
        "formal_seed_registry",
        "val_target",
        "blind_holdout_test",
        "r2_formal_asset_feasibility",
    }
)

REQUIRED_COMPLETION_CHECKS = REQUIRED_IMPLEMENTATION_CHECKS | BLOCKABLE_READINESS_CHECKS


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and value == value.upper() and all(character in "0123456789ABCDEF" for character in value)


@dataclass(frozen=True, slots=True)
class CompletionCheckEvidence:
    command: str
    exit_code: int
    stdout_stderr_path: str
    stdout_stderr_bytes: int
    stdout_stderr_sha256: str
    observed: str
    expected: str
    reviewed_source_files: tuple[str, ...]
    reviewed_test_files: tuple[str, ...]
    remaining_risk: str
    required_action: str

    def validate(self, *, status: str) -> None:
        string_fields = (
            self.command,
            self.stdout_stderr_path,
            self.observed,
            self.expected,
            self.remaining_risk,
            self.required_action,
        )
        if any(not str(value).strip() for value in string_fields):
            raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Completion evidence contains an empty required field")
        if self.stdout_stderr_bytes <= 0 or not _is_sha256(self.stdout_stderr_sha256):
            raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Completion evidence log identity is invalid")
        if not self.reviewed_source_files or not self.reviewed_test_files:
            raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Completion evidence must name reviewed source and test files")
        if status == "PASS" and self.exit_code != 0:
            raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "A PASS completion check has a non-zero evidence command")


@dataclass(frozen=True, slots=True)
class CompletionAudit:
    implementation_status: str
    source_tree_digest: str
    checks: Mapping[str, str]
    blocker_reasons: Mapping[str, str] = field(default_factory=dict)
    check_evidence: Mapping[str, CompletionCheckEvidence] = field(default_factory=dict)
    commit_list: tuple[str, ...] = ()
    changed_file_ledger_path: str = "NOT_YET_BOUND"
    formal_training_started: bool = False
    engineering_gate_generated: bool = False
    assignments_generated: bool = False
    pilot_release_generated: bool = False
    blind_holdout_opened: bool = False
    selector_trained: bool = False
    method_effectiveness_claimed: bool = False
    val_target_available: bool = False
    legacy_engineering_gate_detected: bool = False
    legacy_pilot_release_detected: bool = False
    legacy_assignments_detected: bool = False
    active_v4_engineering_gate_generated: bool = False
    active_v4_pilot_release_generated: bool = False
    active_v4_assignments_generated: bool = False
    completion_state: str = "IMPLEMENTATION_ACCEPTANCE_PASS_NOT_TRAINING_AUTHORIZATION"

    def validate(self, *, require_evidence: bool = False) -> None:
        if self.implementation_status not in {
            "IMPLEMENTED_NOT_FORMALLY_RUN",
            "IMPLEMENTATION_ACCEPTANCE_PASS_NOT_TRAINING_AUTHORIZATION",
        }:
            raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Implementation status is not an allowed implementation-only state")
        if not _is_sha256(self.source_tree_digest):
            raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Completion source tree digest is not a canonical SHA-256")
        prohibited = {
            "formal_training_started": self.formal_training_started,
            "engineering_gate_generated": self.engineering_gate_generated,
            "assignments_generated": self.assignments_generated,
            "pilot_release_generated": self.pilot_release_generated,
            "blind_holdout_opened": self.blind_holdout_opened,
            "selector_trained": self.selector_trained,
            "method_effectiveness_claimed": self.method_effectiveness_claimed,
            "active_v4_engineering_gate_generated": self.active_v4_engineering_gate_generated,
            "active_v4_pilot_release_generated": self.active_v4_pilot_release_generated,
            "active_v4_assignments_generated": self.active_v4_assignments_generated,
        }
        enabled = [name for name, value in prohibited.items() if value]
        if enabled:
            raise SctsrError(
                ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED,
                "Completion audit detected prohibited formal side effects",
                observed=enabled,
            )
        if self.val_target_available:
            raise SctsrError(ErrorCode.BLOCKED_BY_VAL_TARGET, "This implementation baseline has no registered independent val_target")
        if self.completion_state != "IMPLEMENTATION_ACCEPTANCE_PASS_NOT_TRAINING_AUTHORIZATION":
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Completion state could be confused with training authorization")

        observed_keys = set(self.checks)
        if observed_keys != REQUIRED_COMPLETION_CHECKS:
            raise SctsrError(
                ErrorCode.CLOSEOUT_NOT_VALIDATED,
                "Completion audit does not contain the exact registered check set",
                observed={
                    "missing": sorted(REQUIRED_COMPLETION_CHECKS - observed_keys),
                    "extra": sorted(observed_keys - REQUIRED_COMPLETION_CHECKS),
                },
            )
        implementation_failures = {
            key: self.checks[key]
            for key in REQUIRED_IMPLEMENTATION_CHECKS
            if self.checks[key] != "PASS"
        }
        if implementation_failures:
            raise SctsrError(
                ErrorCode.CLOSEOUT_NOT_VALIDATED,
                "Implementation, tests, synthetic evidence, and audit checks may not be hidden as blockers",
                observed=implementation_failures,
            )
        invalid_readiness = {
            key: self.checks[key]
            for key in BLOCKABLE_READINESS_CHECKS
            if self.checks[key] not in {"PASS", "BLOCKED_WITH_REASON"}
        }
        if invalid_readiness:
            raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Readiness check has an invalid state", observed=invalid_readiness)
        if self.checks["formal_training_release"] != "BLOCKED_WITH_REASON":
            raise SctsrError(ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED, "Current implementation closeout must not contain a formal training release")
        if self.checks["formal_seed_registry"] != "BLOCKED_WITH_REASON":
            raise SctsrError(ErrorCode.FORMAL_SEEDS_BLOCKED_UNTIL_RELEASE, "Formal seed registry must remain blocked until release")
        if self.checks["val_target"] != "BLOCKED_WITH_REASON":
            raise SctsrError(ErrorCode.BLOCKED_BY_VAL_TARGET, "val_target must remain explicitly blocked in this baseline")
        if self.checks["blind_holdout_test"] != "BLOCKED_WITH_REASON":
            raise SctsrError(ErrorCode.TEST_ACCESS_FORBIDDEN, "Blind holdout/test must remain explicitly sealed")

        blocked = {key for key, status in self.checks.items() if status == "BLOCKED_WITH_REASON"}
        missing_reason = sorted(key for key in blocked if not str(self.blocker_reasons.get(key, "")).strip())
        extra_reason = sorted(set(self.blocker_reasons) - blocked)
        if missing_reason or extra_reason:
            raise SctsrError(
                ErrorCode.SCHEMA_VALIDATION_FAILED,
                "Every blocked check requires exactly one non-empty blocker reason",
                observed={"missing_reason": missing_reason, "extra_reason": extra_reason},
                expected=sorted(blocked),
            )
        if require_evidence:
            if set(self.check_evidence) != REQUIRED_COMPLETION_CHECKS:
                raise SctsrError(
                    ErrorCode.CLOSEOUT_NOT_VALIDATED,
                    "Strict completion lacks per-check command/log/review evidence",
                    observed={
                        "missing": sorted(REQUIRED_COMPLETION_CHECKS - set(self.check_evidence)),
                        "extra": sorted(set(self.check_evidence) - REQUIRED_COMPLETION_CHECKS),
                    },
                )
            for key in sorted(REQUIRED_COMPLETION_CHECKS):
                self.check_evidence[key].validate(status=self.checks[key])
            if not self.commit_list or self.changed_file_ledger_path == "NOT_YET_BOUND":
                raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Strict completion lacks commit list or changed-file ledger binding")


def completion_audit_from_mapping(value: Mapping[str, Any]) -> CompletionAudit:
    """Materialize the machine closeout schema without silently dropping keys."""

    expected = {item.name for item in fields(CompletionAudit)} | {"schema_version"}
    if set(value) != expected:
        raise SctsrError(
            ErrorCode.CLOSEOUT_NOT_VALIDATED,
            "Completion audit fields do not exactly match the registered schema",
            observed={"missing": sorted(expected - set(value)), "extra": sorted(set(value) - expected)},
        )
    if value.get("schema_version") != "stage1.sctsr.completion_audit.v1":
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Completion audit schema version is invalid")
    raw_evidence = value.get("check_evidence")
    if not isinstance(raw_evidence, Mapping):
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Completion check evidence must be a mapping")
    evidence: dict[str, CompletionCheckEvidence] = {}
    evidence_fields = {item.name for item in fields(CompletionCheckEvidence)}
    for name, raw in raw_evidence.items():
        if not isinstance(raw, Mapping) or set(raw) != evidence_fields:
            raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Completion check evidence schema is invalid", failing_field=str(name))
        payload = dict(raw)
        payload["reviewed_source_files"] = tuple(payload["reviewed_source_files"])
        payload["reviewed_test_files"] = tuple(payload["reviewed_test_files"])
        evidence[str(name)] = CompletionCheckEvidence(**payload)
    payload = {key: item for key, item in value.items() if key not in {"schema_version", "check_evidence"}}
    payload["checks"] = dict(payload["checks"])
    payload["blocker_reasons"] = dict(payload["blocker_reasons"])
    payload["check_evidence"] = evidence
    payload["commit_list"] = tuple(payload["commit_list"])
    return CompletionAudit(**payload)


def completion_check_template(*, r2_formal_asset_feasible: bool = False) -> dict[str, str]:
    checks = {key: "PASS" for key in REQUIRED_IMPLEMENTATION_CHECKS}
    checks.update(
        {
            "formal_training_release": "BLOCKED_WITH_REASON",
            "formal_seed_registry": "BLOCKED_WITH_REASON",
            "val_target": "BLOCKED_WITH_REASON",
            "blind_holdout_test": "BLOCKED_WITH_REASON",
            "r2_formal_asset_feasibility": "PASS" if r2_formal_asset_feasible else "BLOCKED_WITH_REASON",
        }
    )
    return checks


def blocker_reason_template(*, r2_formal_asset_feasible: bool = False) -> dict[str, str]:
    reasons = {
        "formal_training_release": "No signed, trusted, identity-bound formal training release exists.",
        "formal_seed_registry": "Formal discovery and confirmation seeds are created only by a future release authority.",
        "val_target": "No independent, identity- and group-disjoint val_target is registered.",
        "blind_holdout_test": "Blind holdout and test remain sealed until all methods, code, thresholds, and stop rules are frozen.",
    }
    if not r2_formal_asset_feasible:
        reasons["r2_formal_asset_feasibility"] = "The frozen real assets cannot satisfy exact zero-overlap joint-stratum R2 quotas; no matching relaxation is permitted."
    return reasons
