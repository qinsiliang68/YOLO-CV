from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .arm_spec import default_phase1_arms
from .baseline_reference import MAIN_COMMIT, TASKBOOK_BLOB_SHA, TASKBOOK_PATH
from .errors import ErrorCode, SctsrError
from .formal_release import verify_formal_release
from .r2_addendum import (
    R2_APPROVED_IDENTITY_DIGEST,
    R2_APPROVED_SELECTION_SEED,
    R2_MINIMUM_OOF_GROUP_DISPLACEMENT_POLICY,
    validate_r2_matching_policy_mapping,
)
from .rate_spec import F_RATE, IDENTITY_POOL_RATE, U_RATE, ZERO_RATE
from .serialization import load_json, sha256_file, stable_digest


@dataclass(frozen=True, slots=True)
class ContractValidationResult:
    status: str
    contract_digest: str
    checks: Mapping[str, str]


def validate_contract_mapping(contract: Mapping[str, Any], arms: list[Mapping[str, Any]] | None = None) -> ContractValidationResult:
    if contract.get("schema_version") != "stage1.sctsr.contract.v1":
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Unknown SCTSR contract schema")
    baseline = contract.get("baseline")
    expected_baseline = {
        "repository": "qinsiliang68/YOLO-CV",
        "branch": "main",
        "commit": MAIN_COMMIT,
        "taskbook_path": TASKBOOK_PATH,
        "taskbook_blob_sha": TASKBOOK_BLOB_SHA,
    }
    if baseline != expected_baseline:
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Contract baseline identity differs from the frozen taskbook baseline", failing_field="baseline", observed=baseline, expected=expected_baseline)
    fixed = contract.get("fixed_anchors", {})
    if fixed != {"common_parent_epoch": 120, "fallback_epoch": 160, "formal_endpoint_epoch": 200}:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "E120/E160/E200 anchors are immutable", observed=fixed)
    if contract.get("formal_execution_default", True):
        raise SctsrError(ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED, "Formal execution must default to false")
    if contract.get("test_sealed") is not True:
        raise SctsrError(ErrorCode.TEST_ACCESS_FORBIDDEN, "Blind/test must remain sealed")
    if contract.get("val_target_available") is not False:
        raise SctsrError(ErrorCode.BLOCKED_BY_VAL_TARGET, "Current contract must record val_target unavailable")
    if contract.get("weighted_qrad_allowed") is not False:
        raise SctsrError(ErrorCode.WEIGHTED_QRAD_FORBIDDEN, "Weighted Q/R/A/D score is forbidden")
    if str(contract.get("formal_checkpoint", "")).lower().endswith("best.pt"):
        raise SctsrError(ErrorCode.BEST_PT_FORBIDDEN, "best.pt is forbidden")
    if int(contract.get("formal_checkpoint_epoch", -1)) != 200:
        raise SctsrError(ErrorCode.CHECKPOINT_EPOCH_FORBIDDEN, "Formal checkpoint is fixed at E200")
    if contract.get("optimizer_step_policy") != "BASE_STEP_LOCK":
        raise SctsrError(ErrorCode.REPLAY_ADDED_OPTIMIZER_STEP, "Optimizer steps must remain locked to base steps")
    if contract.get("single_process_single_gpu_only") is not True:
        raise SctsrError(ErrorCode.DISTRIBUTED_MODE_NOT_SUPPORTED_IN_V4_PHASE1, "Phase 1 is single-process single-GPU only")
    if contract.get("replay_loss_denominator") != 128:
        raise SctsrError(ErrorCode.DENOMINATOR_IDENTITY_MISMATCH, "Replay CE denominator is the canonical base batch size 128")
    if contract.get("replay_microbatch_max_fraction") != {"numerator": 1, "denominator": 4}:
        raise SctsrError(ErrorCode.REPLAY_MICROBATCH_CAP_EXCEEDED, "Replay microbatch cap is fixed at one quarter of the base batch")
    r2_policy = contract.get("r2_matching_policy")
    expected_r2_fields = {
        "path",
        "sha256",
        "policy_digest",
        "policy_id",
        "selection_seed",
        "expected_identity_digest",
    }
    if not isinstance(r2_policy, Mapping) or set(r2_policy) != expected_r2_fields:
        raise SctsrError(
            ErrorCode.SCHEMA_VALIDATION_FAILED,
            "Contract does not exactly bind the owner-approved R2 addendum",
            failing_field="r2_matching_policy",
        )
    expected_r2_values = {
        "path": "configs/stage1_sctsr_v4/r2_matching_policy_v1.json",
        "policy_id": R2_MINIMUM_OOF_GROUP_DISPLACEMENT_POLICY,
        "selection_seed": R2_APPROVED_SELECTION_SEED,
        "expected_identity_digest": R2_APPROVED_IDENTITY_DIGEST,
    }
    for field, expected_value in expected_r2_values.items():
        if r2_policy.get(field) != expected_value:
            raise SctsrError(
                ErrorCode.CONFIGURATION_MISMATCH,
                "Contract R2 addendum binding differs from the approved policy",
                failing_field=f"r2_matching_policy.{field}",
                observed=r2_policy.get(field),
                expected=expected_value,
            )
    for field in ("sha256", "policy_digest"):
        token = str(r2_policy.get(field, ""))
        if len(token) != 64 or token != token.upper() or any(character not in "0123456789ABCDEF" for character in token):
            raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Contract R2 addendum binding is not canonical SHA-256", failing_field=f"r2_matching_policy.{field}")

    expected_rates = {
        "identity_pool": IDENTITY_POOL_RATE.as_dict(),
        "uniform": U_RATE.as_dict(),
        "front_loaded_active": F_RATE.as_dict(),
        "none": ZERO_RATE.as_dict(),
    }
    rates = contract.get("rates")
    if not isinstance(rates, Mapping):
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Contract rates must be a mapping")
    for name, rate in rates.items():
        if not isinstance(rate, Mapping):
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Each contract rate must be an object", failing_field=f"rates.{name}")
        forbidden = sorted(set(rate) - {"numerator", "denominator", "semantic", "denominator_role", "canonical_token"})
        if forbidden:
            raise SctsrError(ErrorCode.ABSOLUTE_BUDGET_FORBIDDEN, "Rate configuration contains non-rational or derived budget fields", failing_field=f"rates.{name}", observed=forbidden)
    if dict(rates) != expected_rates:
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Contract rates differ from the frozen rational percentage specifications", failing_field="rates", observed=rates, expected=expected_rates)

    side_effects = contract.get("formal_side_effects")
    if not isinstance(side_effects, Mapping) or not side_effects or any(value is not False for value in side_effects.values()):
        raise SctsrError(ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED, "All formal side-effect flags must remain explicitly false", failing_field="formal_side_effects", observed=side_effects)
    if arms is not None:
        expected_rows = []
        for spec in default_phase1_arms():
            row = asdict(spec)
            row["arm_id"] = spec.arm_id.value
            row["allowed_comparators"] = [item.value for item in spec.allowed_comparators]
            expected_rows.append(row)
        if arms != expected_rows:
            extra_budget_fields = sorted({key for row in arms if isinstance(row, Mapping) for key in row if key not in expected_rows[0]})
            if extra_budget_fields:
                raise SctsrError(ErrorCode.ABSOLUTE_BUDGET_FORBIDDEN, "Arm configuration contains unregistered or absolute budget fields", failing_field="arms", observed=extra_budget_fields)
            raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Configured arm registry differs from the exact taskbook registry", observed=arms, expected=expected_rows)
        if any(row.get("arm_id") == "CURRENT_LOSS_U" for row in arms):
            raise SctsrError(ErrorCode.CURRENT_LOSS_HELD, "CURRENT_LOSS_U is held and cannot enter phase 1")
    checks = {
        "fixed_anchors": "PASS",
        "baseline_identity": "PASS",
        "rational_rates": "PASS",
        "formal_default_off": "PASS",
        "test_sealed": "PASS",
        "val_target_blocked": "PASS",
        "weighted_qrad_forbidden": "PASS",
        "formal_checkpoint_e200": "PASS",
        "r2_addendum_bound": "PASS",
        "phase1_arms": "PASS" if arms is not None else "NOT_LOADED",
    }
    return ContractValidationResult("PASS", stable_digest(contract), checks)


def validate_contract_files(contract_path: str | Path, arms_path: str | Path | None = None) -> ContractValidationResult:
    contract_source = Path(contract_path).resolve()
    contract = load_json(contract_source)
    arms = load_json(arms_path).get("arms") if arms_path else None
    result = validate_contract_mapping(contract, arms)
    binding = contract["r2_matching_policy"]
    repository_root = contract_source.parents[2]
    policy_path = (repository_root / str(binding["path"])).resolve()
    try:
        policy_path.relative_to(repository_root)
    except ValueError as exc:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "R2 policy path escapes repository root") from exc
    if not policy_path.is_file() or sha256_file(policy_path) != binding["sha256"]:
        raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "R2 policy bytes differ from the contract binding", artifact_path=str(policy_path))
    policy = validate_r2_matching_policy_mapping(load_json(policy_path))
    if policy["policy_digest"] != binding["policy_digest"]:
        raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "R2 policy semantic digest differs from the contract binding")
    return result


def require_synthetic_or_authorized(
    execution_mode: str,
    release_authorization: str | Path | Mapping[str, Any] | None = None,
    *,
    trust_policy: str | Path | Mapping[str, Any] | None = None,
    expected_bindings: Mapping[str, str] | None = None,
    verification_secret: bytes | str | None = None,
    now_utc: Any = None,
) -> None:
    if execution_mode == "synthetic":
        return
    if execution_mode != "formal":
        raise SctsrError(ErrorCode.INVALID_EXECUTION_MODE, "execution_mode must be synthetic or formal")
    if release_authorization is None:
        raise SctsrError(ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED, "Formal execution requires a future signed release manifest")
    if not isinstance(release_authorization, Mapping) and not Path(release_authorization).is_file():
        raise SctsrError(ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED, "Formal release manifest does not exist", artifact_path=str(release_authorization))
    if trust_policy is None or expected_bindings is None:
        raise SctsrError(ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED, "Formal execution requires an explicit trust policy and complete prepared-run identity bindings")
    verify_formal_release(
        release_authorization,
        trust_policy=trust_policy,
        expected_bindings=expected_bindings,
        verification_secret=verification_secret,
        now_utc=now_utc,
    )
