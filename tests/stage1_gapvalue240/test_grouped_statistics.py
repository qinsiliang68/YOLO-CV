from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from stage1_gapvalue240.grouped_statistics import (
    assert_frozen_baseline,
    baseline_counts,
    benjamini_hochberg,
    build_descriptive_cohort_contrasts,
    build_joint_outcomes,
    build_paired_inference,
    eligible_feature_columns,
    evaluate_candidate_predictions,
    minimum_confirmations,
    one_sided_binomial_lower_bound,
    paired_sign_flip_test,
    run_candidate_validation,
    stratified_permutation_cohort_contrasts,
    validate_frozen_split,
)


def _paired_effects() -> pd.DataFrame:
    specifications = {
        # Strict two-control improvement (and high value).
        "TRIAD_001": {"R1": (500.0, -2.0), "R2": (350.0, 0.0)},
        # High-value only: both controls gain >=300 TN and worst FN is +2.
        "TRIAD_002": {"R1": (400.0, 2.0), "R2": (300.0, 1.0)},
        # Harm must be worse against both controls on both axes.
        "TRIAD_003": {"R1": (-10.0, 1.0), "R2": (-20.0, 3.0)},
        # One control wins and one loses: never call this harm or success.
        "TRIAD_004": {"R1": (100.0, -1.0), "R2": (-5.0, 2.0)},
    }
    rows: list[dict[str, object]] = []
    for triad_id, controls in specifications.items():
        for control, (delta_tn, delta_fn) in controls.items():
            rows.append(
                {
                    "triad_id": triad_id,
                    "control": control,
                    "delta_TN": delta_tn,
                    "delta_FN": delta_fn,
                    "training_seed": 1,
                    "condition_slot": "A01",
                }
            )
    return pd.DataFrame(rows)


def test_joint_outcomes_use_one_triad_unit_and_exact_two_control_logic() -> None:
    outcomes = build_joint_outcomes(_paired_effects()).set_index("triad_id")

    assert len(outcomes) == 4
    assert outcomes.loc["TRIAD_001", "G_TN"] == 350
    assert outcomes.loc["TRIAD_001", "G_FN"] == 0
    assert bool(outcomes.loc["TRIAD_001", "dual_improvement"])
    assert bool(outcomes.loc["TRIAD_002", "high_value"])
    assert not bool(outcomes.loc["TRIAD_002", "dual_improvement"])
    assert bool(outcomes.loc["TRIAD_003", "dual_harm"])
    assert not bool(outcomes.loc["TRIAD_004", "dual_harm"])
    assert outcomes.loc["TRIAD_004", "exclusive_cohort"] == "MIXED_OR_REVERSAL"


def test_baseline_counts_are_explicit_and_frozen_gate_rejects_drift() -> None:
    outcomes = build_joint_outcomes(_paired_effects())
    counts = baseline_counts(outcomes).set_index("label")

    assert counts.loc["ALL_TRIADS", "count"] == 4
    assert counts.loc["DUAL_IMPROVEMENT", "count"] == 1
    assert counts.loc["HIGH_VALUE", "count"] == 2
    assert counts.loc["DUAL_HARM", "count"] == 1
    assert counts.loc["MIXED_OR_REVERSAL_EXCLUSIVE", "count"] == 1
    assert_frozen_baseline(
        outcomes,
        expected_triads=4,
        expected_dual_improvement=1,
        expected_high_value=2,
        expected_dual_harm=1,
    )
    with pytest.raises(ValueError, match="dual_improvement"):
        assert_frozen_baseline(
            outcomes,
            expected_triads=4,
            expected_dual_improvement=2,
            expected_high_value=2,
            expected_dual_harm=1,
        )


def test_sign_flip_and_bh_fdr_are_exact_and_directional() -> None:
    greater = paired_sign_flip_test([1.0, 2.0, 3.0], alternative="greater")
    less = paired_sign_flip_test([-1.0, -2.0, -3.0], alternative="less")

    assert greater["method"] == "exact_cluster_sign_flip"
    assert greater["p_value"] == pytest.approx(1 / 8)
    assert less["p_value"] == pytest.approx(1 / 8)
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, np.nan])
    assert adjusted[:3].tolist() == pytest.approx([0.03, 0.04, 0.04])
    assert math.isnan(adjusted[3])


def test_paired_inference_never_pools_r1_r2_and_clusters_by_seed() -> None:
    rows: list[dict[str, object]] = []
    for seed, offset in ((1, 0.0), (2, 0.1), (3, -0.1)):
        for condition in ("A01", "A02"):
            for control, control_offset in (("R1", 0.0), ("R2", 0.5)):
                rows.append(
                    {
                        "triad_id": f"{condition}_{seed}",
                        "control": control,
                        "training_seed": seed,
                        "condition_slot": condition,
                        "feature": 1.0 + offset + control_offset,
                    }
                )

    inference = build_paired_inference(
        pd.DataFrame(rows),
        feature_columns=["feature"],
        bootstrap_resamples=200,
        random_state=9,
    )

    assert len(inference) == 2
    assert set(inference["control"]) == {"R1", "R2"}
    assert set(inference["n_triads"]) == {6}
    assert set(inference["bootstrap_cluster"]) == {"training_seed"}
    assert inference["q_value_bh"].notna().all()


def test_descriptive_contrasts_report_seed_and_condition_cluster_bootstraps() -> None:
    rows = []
    for seed in (1, 2, 3):
        for condition, cohort, value in (
            ("A01", "DUAL_IMPROVEMENT", 2.0),
            ("A02", "DUAL_HARM", -1.0),
        ):
            rows.append(
                {
                    "triad_id": f"{condition}_{seed}",
                    "training_seed": seed,
                    "condition_slot": condition,
                    "exclusive_cohort": cohort,
                    "feature": value + seed * 0.01,
                }
            )

    result = build_descriptive_cohort_contrasts(
        pd.DataFrame(rows),
        feature_columns=["feature"],
        bootstrap_resamples=200,
        random_state=11,
    ).iloc[0]

    assert result["mean_difference"] == pytest.approx(3.0)
    assert result["seed_bootstrap_ci_low"] > 2.9
    assert result["condition_bootstrap_ci_low"] > 2.9
    assert result["interpretation"] == "DESCRIPTIVE_NOT_PREDICTIVE"


def test_stratified_cohort_permutation_preserves_seed_composition_and_fdr() -> None:
    rows = []
    for seed in (1, 2, 3):
        for index in range(8):
            positive = index < 4
            rows.append(
                {
                    "triad_id": f"{seed}_{index}",
                    "training_seed": seed,
                    "exclusive_cohort": (
                        "DUAL_IMPROVEMENT" if positive else "DUAL_HARM"
                    ),
                    "strong": 10.0 if positive else -10.0,
                    "null": float((index * 7 + seed) % 5),
                }
            )

    result = stratified_permutation_cohort_contrasts(
        pd.DataFrame(rows),
        feature_columns=["strong", "null"],
        resamples=2000,
        random_state=5,
    ).set_index("feature")

    assert result.loc["strong", "positive_minus_negative"] == 20.0
    assert result.loc["strong", "p_value"] < 0.01
    assert result.loc["strong", "q_value_bh"] < 0.02
    assert result.loc["strong", "stratification"] == "training_seed"


def test_feature_time_registry_is_allowlist_and_cutoff_gate() -> None:
    registry = pd.DataFrame(
        [
            {
                "feature": "train_loss__at_120",
                "available_epoch": 120,
                "allowed_as_predictor": True,
            },
            {
                "feature": "train_loss__at_200",
                "available_epoch": 200,
                "allowed_as_predictor": True,
            },
            {
                "feature": "TN_at_FN95",
                "available_epoch": 0,
                "allowed_as_predictor": False,
            },
        ]
    )
    columns = [
        "R1__delta__train_loss__at_120",
        "R2__delta__train_loss__at_120",
        "R1__delta__train_loss__at_200",
        "TN_at_FN95",
        "unregistered_feature",
    ]

    assert eligible_feature_columns(columns, registry, cutoff=150) == [
        "R1__delta__train_loss__at_120",
        "R2__delta__train_loss__at_120",
    ]


def test_binomial_lower_bound_and_minimum_confirmation_contract() -> None:
    assert one_sided_binomial_lower_bound(14, 14) > 0.8
    assert one_sided_binomial_lower_bound(13, 13) < 0.8
    confirmations = minimum_confirmations(target_rate=0.8, alpha=0.05, max_failures=2)

    assert confirmations.to_dict("records") == [
        {
            "failures": 0,
            "successes": 14,
            "total": 14,
            "lower_bound": pytest.approx(0.8073638243498646),
        },
        {
            "failures": 1,
            "successes": 21,
            "total": 22,
            "lower_bound": pytest.approx(0.8018778683631184),
        },
        {
            "failures": 2,
            "successes": 28,
            "total": 30,
            "lower_bound": pytest.approx(0.804673956345074),
        },
    ]


def test_candidate_metrics_distinguish_precision_coverage_harm_and_auc() -> None:
    predictions = pd.DataFrame(
        {
            "y_true": [1, 1, 0, 0],
            "dual_harm": [0, 0, 1, 0],
            "probability": [0.9, 0.8, 0.7, 0.1],
            "selected": [True, True, True, False],
        }
    )

    metrics = evaluate_candidate_predictions(predictions)

    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["coverage"] == pytest.approx(3 / 4)
    assert metrics["harmful_rate"] == pytest.approx(1 / 3)
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["success_recall"] == 1.0
    assert metrics["specificity"] == 0.5
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["f1"] == pytest.approx(0.8)
    assert not metrics["confirmed_above_target"]
    assert metrics["auc_is_not_success_probability"]


def _candidate_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    discovery_seeds = (101, 102, 103)
    for seed_index, seed in enumerate(discovery_seeds):
        for condition_index in range(25):
            success = condition_index % 5 == 0
            harm = condition_index % 5 == 1
            rows.append(
                {
                    "triad_id": f"D_{seed}_{condition_index:02d}",
                    "phase": "A" if condition_index < 19 else "B",
                    "condition_slot": f"A{condition_index + 1:02d}",
                    "discovery_or_confirmation": "discovery",
                    "training_seed": seed,
                    "selection_digest": f"digest_D_{seed}_{condition_index:02d}",
                    "dual_improvement": success,
                    "dual_harm": harm,
                    "feature_120": float(success) + seed_index * 0.01,
                    # This perfect future feature must be excluded at cutoff 150.
                    "feature_200": float(success),
                }
            )
    for external_index, seed in enumerate((201, 202, 203, 204, 205)):
        success = external_index in (0, 2)
        rows.append(
            {
                "triad_id": f"C_{seed}",
                "phase": "C",
                "condition_slot": "A02",
                "discovery_or_confirmation": "confirmation",
                "training_seed": seed,
                "selection_digest": f"digest_C_{seed}",
                "dual_improvement": success,
                "dual_harm": not success,
                "feature_120": float(success),
                "feature_200": float(success),
            }
        )
    registry = pd.DataFrame(
        [
            {
                "feature": "feature_120",
                "available_epoch": 120,
                "allowed_as_predictor": True,
            },
            {
                "feature": "feature_200",
                "available_epoch": 200,
                "allowed_as_predictor": True,
            },
            {
                "feature": "dual_improvement",
                "available_epoch": 0,
                "allowed_as_predictor": False,
            },
        ]
    )
    return pd.DataFrame(rows), registry


def test_frozen_split_requires_75_discovery_and_five_a02_external() -> None:
    frame, _ = _candidate_frame()
    summary = validate_frozen_split(frame)

    assert summary == {
        "all_triads": 80,
        "discovery_triads": 75,
        "discovery_seeds": 3,
        "conditions_per_discovery_seed": 25,
        "external_triads": 5,
        "external_seeds": 5,
    }
    broken = frame.copy()
    broken.loc[broken["phase"] == "C", "condition_slot"] = "A03"
    with pytest.raises(ValueError, match="A02"):
        validate_frozen_split(broken)


def test_candidate_validation_is_loso_leakage_safe_and_external_is_falsification() -> None:
    frame, registry = _candidate_frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_candidate_validation(
            frame,
            registry,
            cutoff=150,
            max_features=1,
            target_precision=0.8,
            max_harmful_rate=0.2,
            include_digest_diagnostics=False,
            random_state=17,
        )
    assert not [item for item in caught if issubclass(item.category, RuntimeWarning)]
    assert not [item for item in caught if "are constant" in str(item.message)]

    assert result.feature_columns == ("feature_120",)
    assert set(result.predictions["validation_scheme"]) == {
        "DISCOVERY_LOSO_SEED",
        "PHASE_C_EXTERNAL_FALSIFICATION",
    }
    discovery = result.predictions.query(
        "validation_scheme == 'DISCOVERY_LOSO_SEED'"
    )
    external = result.predictions.query(
        "validation_scheme == 'PHASE_C_EXTERNAL_FALSIFICATION'"
    )
    assert len(discovery) == 75
    assert len(external) == 5
    assert result.summaries.set_index("validation_scheme").loc[
        "PHASE_C_EXTERNAL_FALSIFICATION", "interpretation"
    ] == "EXTERNAL_FALSIFICATION_ONLY"

    # Held-seed labels cannot influence its own scores or fold thresholds.
    changed = frame.copy()
    held = changed["training_seed"] == 101
    changed.loc[held, "dual_improvement"] = ~changed.loc[
        held, "dual_improvement"
    ].astype(bool)
    rerun = run_candidate_validation(
        changed,
        registry,
        cutoff=150,
        max_features=1,
        include_digest_diagnostics=False,
        random_state=17,
    )
    left = discovery[discovery.training_seed == 101].sort_values("triad_id")
    right = rerun.predictions.query(
        "validation_scheme == 'DISCOVERY_LOSO_SEED' and training_seed == 101"
    ).sort_values("triad_id")
    assert left["probability"].tolist() == pytest.approx(
        right["probability"].tolist()
    )
    assert left["threshold"].tolist() == pytest.approx(right["threshold"].tolist())


def test_candidate_validation_includes_digest_and_double_exclusion_diagnostics() -> None:
    frame, registry = _candidate_frame()
    result = run_candidate_validation(
        frame,
        registry,
        cutoff=150,
        max_features=1,
        include_digest_diagnostics=True,
        random_state=3,
    )

    discovery = result.predictions[result.predictions.phase != "C"]
    counts = discovery.groupby("validation_scheme").size().to_dict()
    assert counts == {
        "DISCOVERY_DOUBLE_EXCLUSION_SEED_DIGEST": 75,
        "DISCOVERY_LEAVE_SELECTION_DIGEST_OUT": 75,
        "DISCOVERY_LOSO_SEED": 75,
    }
    assert set(result.fold_details["validation_scheme"]) >= {
        "DISCOVERY_LOSO_SEED",
        "DISCOVERY_LEAVE_SELECTION_DIGEST_OUT",
        "DISCOVERY_DOUBLE_EXCLUSION_SEED_DIGEST",
    }


def test_fold_local_pipeline_keeps_features_that_are_empty_in_one_training_fold() -> None:
    frame, registry = _candidate_frame()
    frame["sparse_120"] = np.nan
    frame.loc[frame["training_seed"] == 101, "sparse_120"] = 1.0
    registry = pd.concat(
        [
            registry,
            pd.DataFrame(
                [
                    {
                        "feature": "sparse_120",
                        "available_epoch": 120,
                        "allowed_as_predictor": True,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_candidate_validation(
            frame,
            registry,
            cutoff=150,
            max_features=2,
            include_digest_diagnostics=False,
            random_state=19,
        )

    assert not [item for item in caught if "Skipping features" in str(item.message)]
    assert not [item for item in caught if "are constant" in str(item.message)]
    assert len(result.predictions) == 80
