from __future__ import annotations

import pytest

from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.qrad_contract import QradFactorContract, TargetAlignmentConfig, reject_weighted_qrad, validate_candidate_signal_semantic, validate_disabled_phase2_config
from stage1_sctsr_v4.rate_spec import IDENTITY_POOL_RATE, U_RATE
from stage1_sctsr_v4.seed_registry import SeedRegistry
from stage1_sctsr_v4.short_branch_scaffold import PredictorFeatureRow, ShortBranchSpec, UtilityOutcome, require_predictor_training_allowed
from stage1_sctsr_v4.serialization import sha256_file, stable_digest


def test_nested_weighted_qrad_is_rejected_recursively():
    with pytest.raises(SctsrError) as caught:
        reject_weighted_qrad({"selector": {"scoring": {"weights": {"Q": 0.25, "R": 0.25}}}})
    assert caught.value.code is ErrorCode.WEIGHTED_QRAD_FORBIDDEN


def test_current_baseline_cannot_enable_a_even_when_caller_claims_target_available():
    config = TargetAlignmentConfig(True, "A" * 64, True, True, "FN95_LOCAL", 1)
    with pytest.raises(SctsrError) as caught:
        config.validate(val_target_available=True)
    assert caught.value.code is ErrorCode.BLOCKED_BY_VAL_TARGET


def test_candidate_signal_semantic_must_be_an_exact_registered_nonutility_token():
    with pytest.raises(SctsrError) as caught:
        validate_candidate_signal_semantic("RHO", "UTILITY_NOT_UTILITY")
    assert caught.value.code is ErrorCode.CANDIDATE_SIGNAL_AS_UTILITY_FORBIDDEN


def test_short_branch_rejects_unbound_or_same_treatment_and_r2_pools():
    for treatment_digest, r2_digest in (("T", "R2"), ("A" * 64, "A" * 64)):
        with pytest.raises(SctsrError):
            ShortBranchSpec(120, 5, U_RATE, treatment_digest, r2_digest, False).validate()


def test_predictor_requires_more_than_a_boolean_phase1_claim():
    with pytest.raises(SctsrError) as caught:
        require_predictor_training_allowed(phase1_gate_passed=True, model_family="LOGISTIC_REGRESSION")
    assert caught.value.code is ErrorCode.SELECTOR_TRAINING_FORBIDDEN


def test_blocked_seed_registry_cannot_smuggle_formal_seed_values():
    registry = SeedRegistry(
        "stage1.sctsr.seed_registry.v1",
        "FORMAL_SEEDS_BLOCKED_UNTIL_RELEASE",
        (),
        (20260606,),
        tuple(range(8)),
        (),
    )
    with pytest.raises(SctsrError) as caught:
        registry.validate()
    assert caught.value.code is ErrorCode.FORMAL_SEEDS_BLOCKED_UNTIL_RELEASE


def test_formal_seed_state_string_alone_is_not_release_authorization():
    registry = SeedRegistry(
        "stage1.sctsr.seed_registry.v1",
        "FROZEN_BY_RELEASE_AUTHORITY",
        (),
        (),
        tuple(range(8)),
        tuple(range(20, 34)),
    )
    with pytest.raises(SctsrError) as caught:
        registry.validate(formal=True)
    assert caught.value.code is ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED


def test_disabled_phase2_config_is_sha_bound_and_every_action_is_held(repository_root):
    receipt = validate_disabled_phase2_config(repository_root / "configs" / "stage1_sctsr_v4" / "disabled_phase2_v1.json")
    assert receipt["status"] == "PASS_HELD_DISABLED"
    assert len(receipt["sha256"]) == 64


def test_qrad_factor_contract_uses_distinct_stage_comparators():
    contract = QradFactorContract(
        usage="SEQUENTIAL_GATE",
        gate_order=("Q", "R", "A", "D"),
        comparator_by_factor={"R": "R2_R", "A": "R2_A", "D": "R2_D"},
    )
    contract.validate()
    with pytest.raises(SctsrError):
        QradFactorContract(
            usage="SEQUENTIAL_GATE",
            gate_order=("Q", "R", "A", "D"),
            comparator_by_factor={"R": "R2", "A": "R2", "D": "R2"},
        ).validate()


def test_predictor_feature_row_is_vector_label_with_a_explicitly_blocked():
    row = PredictorFeatureRow(
        microset_id="M001",
        training_seed=1,
        anchor_epoch=120,
        horizon=5,
        q_reliability_features={"oof_agreement": 0.9},
        r_reducible_features={"rho": 0.2},
        a_direction_features=None,
        a_unavailable_reason="BLOCKED_BY_VAL_TARGET",
        d_set_coverage_features={"feature_coverage": 0.7},
        state_features={"learning_rate": 0.001},
        cumulative_replay_exposure=0,
        outcome=UtilityOutcome(True, 0.01, 3, -1),
    )
    row.validate()


def test_future_short_branch_requires_zero_overlap_memberships_and_quota_digest():
    spec = ShortBranchSpec(
        120,
        5,
        IDENTITY_POOL_RATE,
        "A" * 64,
        "B" * 64,
        False,
        matched_quota_digest="C" * 64,
        treatment_sample_ids=("T1", "T2"),
        matched_r2_sample_ids=("R1", "R2"),
    )
    spec.validate()


def test_caller_booleans_cannot_enable_short_branch_or_predictor_in_current_baseline():
    spec = ShortBranchSpec(120, 5, IDENTITY_POOL_RATE, "A" * 64, "B" * 64, True)
    with pytest.raises(SctsrError) as branch_error:
        spec.validate(phase1_gate_passed=True, phase1_gate_receipt_sha256="C" * 64, phase2_release_authorized=True)
    assert branch_error.value.code is ErrorCode.SHORT_BRANCH_DISABLED
    with pytest.raises(SctsrError) as predictor_error:
        require_predictor_training_allowed(
            phase1_gate_passed=True,
            model_family="LOGISTIC_REGRESSION",
            phase1_gate_receipt_sha256="C" * 64,
            phase2_release_authorized=True,
        )
    assert predictor_error.value.code is ErrorCode.SELECTOR_TRAINING_FORBIDDEN


def test_future_target_registration_interface_checks_bytes_roles_formula_and_lag(tmp_path):
    manifest = tmp_path / "val_target.json"
    manifest.write_text('{"role":"val_target"}', encoding="utf-8")
    formula = "LOCAL_FN95_GRADIENT_VECTOR_V1"
    config = TargetAlignmentConfig(
        enabled=False,
        val_target_manifest_sha256=sha256_file(manifest),
        identity_disjoint=True,
        group_disjoint=True,
        local_fn95_formula=formula,
        checkpoint_lag=1,
        val_target_manifest_path=manifest.as_posix(),
        split_role="val_target",
        identity_disjoint_from_roles=("train", "OOF", "val_model", "val_cal", "val_op", "test"),
        formula_digest=stable_digest({"local_fn95_formula": formula, "checkpoint_lag": 1}),
    )
    config.validate_future_registration()


def test_safety_vector_rejects_dual_end_degradation_as_pass():
    with pytest.raises(SctsrError) as caught:
        UtilityOutcome(True, 0.01, -1, 1).validate()
    assert caught.value.code is ErrorCode.FRONTIER_INVALID
