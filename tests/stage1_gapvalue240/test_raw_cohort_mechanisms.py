from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stage1_gapvalue240.raw_cohort_mechanisms import (
    RawCohortMechanismError,
    attach_cohort_memberships,
    build_pair_mechanism_features,
    seed_stratified_permutation,
)


def _outcomes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "triad_id": ["T1", "T2", "T3", "T4"],
            "condition_id": ["C1", "C2", "C1", "C2"],
            "training_seed": [11, 11, 22, 22],
            "dual_improvement": [True, False, False, False],
            "high_value": [True, True, False, False],
            "dual_harm": [False, False, True, False],
        }
    )


def test_cohort_membership_preserves_overlap_and_defines_mixed() -> None:
    membership = attach_cohort_memberships(_outcomes(), expected_triads=4)

    observed = {
        cohort: set(group.triad_id)
        for cohort, group in membership.groupby("cohort", sort=True)
    }
    assert observed == {
        "DUAL_HARM": {"T3"},
        "DUAL_IMPROVEMENT": {"T1"},
        "HIGH_VALUE": {"T1", "T2"},
        "MIXED": {"T2", "T4"},
    }


def test_cohort_membership_rejects_logically_overlapping_harm() -> None:
    outcomes = _outcomes()
    outcomes.loc[0, "dual_harm"] = True
    with pytest.raises(RawCohortMechanismError, match="both dual improvement and dual harm"):
        attach_cohort_memberships(outcomes, expected_triads=4)


def test_pair_feature_builder_uses_treatment_minus_each_control() -> None:
    outcomes = _outcomes().iloc[:1].copy()
    tails = []
    for score_type in ("raw", "calibrated"):
        for control_arm in ("R1", "R2"):
            for label, scope, value in (
                ("normal", "operational", -0.1),
                ("defect", "operational", 0.2),
                ("normal", "tail_gap", -0.3),
                ("defect", "tail_gap", 0.4),
            ):
                tails.append(
                    {
                        "triad_id": "T1",
                        "control_arm": control_arm,
                        "score_type": score_type,
                        "label": label,
                        "scope": scope,
                        "n": 95 if label == "defect" and scope == "operational" else 10,
                        "mean_shift": value,
                        "median_shift": value / 2,
                        "beneficial_rate": 0.75,
                    }
                )
    dominance = pd.DataFrame(
        [
            {
                "triad_id": "T1",
                "control_arm": control_arm,
                "score_type": score_type,
                "safe_positive_budget_share": 0.8,
                "safe_min_delta_TN": 10,
                "safe_mean_delta_TN": 20,
                "delta_TN_at_FN95": 30,
                "safe_frontier_dominant": True,
            }
            for score_type in ("raw", "calibrated")
            for control_arm in ("R1", "R2")
        ]
    )
    probability_rows = []
    for score_type in ("raw", "calibrated"):
        for arm, auroc, brier in (("T", 0.9, 0.1), ("R1", 0.8, 0.2), ("R2", 0.7, 0.3)):
            probability_rows.append(
                {
                    "triad_id": "T1",
                    "run_slot": f"{arm}_{score_type}",
                    "arm": arm,
                    "score_type": score_type,
                    "auroc": auroc,
                    "auprc": auroc - 0.1,
                    "brier": brier,
                    "log_loss": brier + 0.1,
                    "ece": brier + 0.2,
                }
            )

    pairs = build_pair_mechanism_features(
        outcomes,
        pd.DataFrame(tails),
        dominance,
        pd.DataFrame(probability_rows),
        expected_triads=1,
    )

    assert len(pairs) == 4
    r1_raw = pairs[(pairs.control_arm == "R1") & (pairs.score_type == "raw")].iloc[0]
    assert r1_raw["raw_tail__normal_operational__mean_shift"] == pytest.approx(-0.1)
    assert r1_raw["raw_tail__defect_operational__mean_shift"] == pytest.approx(0.2)
    assert r1_raw["probability__delta_auroc"] == pytest.approx(0.1)
    assert r1_raw["probability__delta_brier"] == pytest.approx(-0.1)
    assert r1_raw["frontier__safe_mean_delta_TN"] == pytest.approx(20)


def test_seed_stratified_permutation_is_deterministic_and_seed_blocked() -> None:
    frame = pd.DataFrame(
        {
            "training_seed": [1, 1, 2, 2, 3, 3],
            "condition_id": ["A", "B", "A", "B", "A", "B"],
            "cohort": ["GOOD", "HARM"] * 3,
            "value": [3.0, 0.0, 4.0, 1.0, 5.0, 2.0],
        }
    )

    first = seed_stratified_permutation(
        frame,
        cohort_a="GOOD",
        cohort_b="HARM",
        value_column="value",
        permutations=999,
        random_seed=7,
    )
    second = seed_stratified_permutation(
        frame,
        cohort_a="GOOD",
        cohort_b="HARM",
        value_column="value",
        permutations=999,
        random_seed=7,
    )

    assert first == second
    assert first["seed_blocked_mean_difference"] == pytest.approx(3.0)
    assert first["eligible_seed_count"] == 3
    assert 0.0 < first["permutation_p_two_sided"] <= 1.0
    assert np.isfinite(first["seed_cluster_bootstrap_ci_low"])
    assert np.isfinite(first["condition_cluster_bootstrap_ci_high"])
