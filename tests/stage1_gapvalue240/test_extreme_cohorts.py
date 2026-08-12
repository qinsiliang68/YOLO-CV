from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stage1_gapvalue240.deep_analysis import CanonicalInputError
from stage1_gapvalue240.extreme_cohorts import (
    build_outcome_mechanism_pairs,
    build_stratified_extreme_contrasts,
    build_training_window_features,
    classify_triad_cohorts,
    compute_operational_sample_dynamics,
    pair_selection_feature_deltas,
    summarize_extreme_feature_contrasts,
    summarize_leave_one_group_out,
    summarize_selection_operational_features,
    summarize_selection_set_outcomes,
)


def _pair(
    triad_id: str,
    *,
    r1_tn: float,
    r1_fn: float,
    r2_tn: float,
    r2_fn: float,
    seed: int = 11,
) -> list[dict[str, object]]:
    common = {
        "triad_id": triad_id,
        "phase": "A",
        "condition_slot": triad_id.replace("TRIAD_", "A"),
        "condition_id": f"condition_{triad_id}",
        "method": "test",
        "budget": 600,
        "guard_ratio": 0.0,
        "training_seed": seed,
        "discovery_or_confirmation": "discovery",
        "machine_pair": "same_machine",
        "same_machine": True,
        "any_resumed": False,
    }
    return [
        {**common, "control": "R1", "delta_TN": r1_tn, "delta_FN": r1_fn},
        {**common, "control": "R2", "delta_TN": r2_tn, "delta_FN": r2_fn},
    ]


def test_classify_triad_cohorts_builds_five_mutually_exclusive_tiers() -> None:
    rows = [
        *_pair("TRIAD_001", r1_tn=500, r1_fn=-2, r2_tn=400, r2_fn=0),
        *_pair("TRIAD_002", r1_tn=100, r1_fn=0, r2_tn=500, r2_fn=-3),
        *_pair("TRIAD_003", r1_tn=500, r1_fn=2, r2_tn=400, r2_fn=-1),
        *_pair("TRIAD_004", r1_tn=-10, r1_fn=2, r2_tn=-20, r2_fn=3),
        *_pair("TRIAD_005", r1_tn=20, r1_fn=4, r2_tn=-5, r2_fn=-1),
    ]

    result = classify_triad_cohorts(pd.DataFrame(rows)).set_index("triad_id")

    assert result["cohort_code"].to_dict() == {
        "TRIAD_001": "S",
        "TRIAD_002": "A",
        "TRIAD_003": "B",
        "TRIAD_004": "H",
        "TRIAD_005": "M",
    }
    assert result["strong_positive"].sum() == 2
    assert result["high_value"].sum() == 2
    assert result["harmful"].sum() == 1
    assert result["cohort_code"].notna().all()
    assert result["cohort_reason"].str.len().gt(0).all()


def test_classify_triad_cohorts_rejects_duplicate_control_rows() -> None:
    rows = _pair("TRIAD_001", r1_tn=1, r1_fn=0, r2_tn=1, r2_fn=0)
    rows.append(dict(rows[0]))

    with pytest.raises(CanonicalInputError, match="one R1 and one R2"):
        classify_triad_cohorts(pd.DataFrame(rows))


def test_training_window_features_preserve_endpoint_and_robust_definitions() -> None:
    rows = []
    for control in ("R1", "R2"):
        for epoch in range(1, 201):
            rows.append(
                {
                    "triad_id": "TRIAD_001",
                    "condition_slot": "A01",
                    "condition_id": "A01_test",
                    "phase": "A",
                    "training_seed": 11,
                    "control": control,
                    "machine_pair": "same_machine",
                    "any_resumed": False,
                    "epoch": epoch,
                    "delta_train_loss": 1.0 - epoch / 1000.0,
                    "delta_val_loss": epoch / 1000.0,
                    "delta_top1": epoch / 10000.0,
                }
            )
    cohorts = classify_triad_cohorts(
        pd.DataFrame(
            _pair("TRIAD_001", r1_tn=500, r1_fn=-2, r2_tn=400, r2_fn=0)
        )
    )

    result = build_training_window_features(pd.DataFrame(rows), cohorts)

    assert len(result) == 2
    row = result.loc[result["control"] == "R1"].iloc[0]
    assert row["cohort_code"] == "S"
    assert row["train_loss_extra_drop_epoch121_to_200"] == pytest.approx(0.079)
    assert row["train_loss_robust_drop_121_130_to_191_200"] == pytest.approx(0.070)
    assert row["train_loss_slope_121_200"] == pytest.approx(-0.001)
    assert row["mean_delta_train_loss_e001_040"] == pytest.approx(0.9795)
    assert row["mean_delta_val_loss_e161_200"] == pytest.approx(0.1805)


def test_operational_dynamics_uses_fn_threshold_and_skips_repaired_fold_epoch() -> None:
    probabilities = np.asarray(
        [
            [0.60, 0.10, 0.50, 0.80],
            [0.40, 0.10, 0.50, 0.80],
            [0.60, 0.10, 0.50, 0.80],
            [0.40, 0.10, 0.50, 0.80],
            [0.60, 0.10, 0.50, 0.80],
        ],
        dtype=np.float64,
    )
    samples = pd.DataFrame(
        {
            "sample_id": ["n_fold01", "n_clean", "d_weak", "d_strong"],
            "y_true": [0, 0, 1, 1],
            "oof_fold": [1, 2, 1, 2],
        }
    )

    sample_result, epochs, audit = compute_operational_sample_dynamics(
        probabilities,
        samples,
        fn_limit=0,
        repaired_fold=1,
        repaired_epoch=3,
        windows=((1, 2, "early"), (4, 5, "late")),
    )

    assert epochs["threshold"].tolist() == [0.5] * 5
    row = sample_result.set_index("sample_id").loc["n_fold01"]
    assert row["valid_epoch_count"] == 4
    assert row["operational_forgetting_count"] == 1
    assert row["operational_recovery_count"] == 1
    assert row["error_rate_early"] == pytest.approx(0.5)
    assert row["error_rate_late"] == pytest.approx(0.5)
    assert audit["affected_sample_count"] == 2
    assert audit["excluded_cell_count"] == 2
    assert audit["threshold_rule"] == "predict_defect_when_score_gte_threshold"


def test_selection_set_outcomes_exposes_same_images_with_opposite_results() -> None:
    selections = pd.DataFrame(
        {
            "triad_id": ["T1", "T2", "T3", "T4"],
            "sample_set_digest": ["same", "same", "other", "other"],
            "cohort_code": ["S", "H", "M", "M"],
            "condition_slot": ["A01", "A01", "A02", "A02"],
            "training_seed": [1, 2, 1, 2],
        }
    )

    result = summarize_selection_set_outcomes(selections).set_index(
        "sample_set_digest"
    )

    assert result.loc["same", "spans_exceptional_and_harmful"]
    assert result.loc["same", "cohort_codes"] == "H|S"
    assert result.loc["same", "triad_count"] == 2
    assert not result.loc["other", "spans_exceptional_and_harmful"]


def test_selection_operational_summary_and_pairing_keep_scopes_separate(
    tmp_path,
) -> None:
    dynamics = pd.DataFrame(
        {
            "sample_id": ["n1", "n2", "d1", "d2"],
            "y_true": [0, 0, 1, 1],
            "operational_error_rate": [0.8, 0.2, 0.6, 0.1],
            "operational_forgetting_count": [3, 0, 2, 0],
            "operational_recovery_count": [2, 0, 1, 0],
            "score_direction_changes": [4, 1, 3, 1],
            "operational_correction": [0.7, 0.0, 0.5, 0.0],
            "error_rate_early_001_040": [1.0, 0.0, 1.0, 0.0],
            "error_rate_late_161_200": [0.3, 0.0, 0.5, 0.0],
            "p_defect_linear_slope": [-0.1, 0.0, 0.1, 0.0],
            "trajectory_type": [
                "corrected",
                "stable_correct",
                "persistent_wrong",
                "stable_correct",
            ],
        }
    )
    runs = pd.DataFrame(
        {
            "run_slot": ["RUN_001", "RUN_002", "RUN_003"],
            "triad_id": ["TRIAD_001"] * 3,
            "condition_slot": ["A01"] * 3,
            "condition_id": ["A01_test"] * 3,
            "phase": ["A"] * 3,
            "method": ["test"] * 3,
            "budget": [2] * 3,
            "guard_ratio": [0.0] * 3,
            "arm": ["T", "R1", "R2"],
            "training_seed": [11] * 3,
        }
    )
    selected = {
        "RUN_001": ["n1", "n2"],
        "RUN_002": ["n2", "d2"],
        "RUN_003": ["n1", "d1"],
    }
    for run_slot, ids in selected.items():
        target = tmp_path / run_slot
        target.mkdir()
        lookup = dynamics.set_index("sample_id")
        pd.DataFrame(
            {
                "sample_id": ids,
                "y_true": [int(lookup.loc[sample_id, "y_true"]) for sample_id in ids],
            }
        ).to_csv(target / "selection_manifest.csv", index=False)
    cohorts = classify_triad_cohorts(
        pd.DataFrame(
            _pair("TRIAD_001", r1_tn=500, r1_fn=-2, r2_tn=400, r2_fn=0)
        )
    )

    summary, treatment_sets = summarize_selection_operational_features(
        runs, tmp_path, dynamics, cohorts
    )
    paired = pair_selection_feature_deltas(summary, cohorts)

    assert len(treatment_sets) == 1
    assert treatment_sets.iloc[0]["cohort_code"] == "S"
    t_normal = summary.loc[
        (summary["run_slot"] == "RUN_001") & (summary["scope"] == "normal")
    ].iloc[0]
    assert t_normal["mean_operational_error_rate"] == pytest.approx(0.5)
    assert t_normal["share_corrected"] == pytest.approx(0.5)
    assert set(paired["control"]) == {"R1", "R2"}
    assert set(paired["scope"]) >= {"all"}


def test_outcome_mechanism_pairs_keep_threshold_as_post_training_diagnostic() -> None:
    cohorts = classify_triad_cohorts(
        pd.DataFrame(
            _pair("TRIAD_001", r1_tn=500, r1_fn=-2, r2_tn=400, r2_fn=0)
        )
    )
    pairs = pd.DataFrame(
        {
            "triad_id": ["TRIAD_001", "TRIAD_001"],
            "control": ["R1", "R2"],
            "t_run_slot": ["RUN_T", "RUN_T"],
            "control_run_slot": ["RUN_R1", "RUN_R2"],
            "delta_TN": [500, 400],
            "delta_FN": [-2, 0],
        }
    )
    runs = pd.DataFrame(
        {
            "run_slot": ["RUN_T", "RUN_R1", "RUN_R2"],
            "threshold_at_FN95": [0.03, 0.02, 0.025],
            "raw_threshold_at_FN95": [0.003, 0.002, 0.0025],
        }
    )
    calibration = pd.DataFrame(
        {
            "run_slot": ["RUN_T", "RUN_R1", "RUN_R2"],
            "split": ["val_op"] * 3,
            "auroc": [0.99, 0.98, 0.985],
            "auroc_raw": [0.991, 0.981, 0.986],
        }
    )

    result = build_outcome_mechanism_pairs(pairs, runs, calibration, cohorts)

    assert result.loc[result["control"] == "R1", "delta_threshold"].iloc[0] == pytest.approx(0.01)
    assert result.loc[result["control"] == "R2", "delta_auroc"].iloc[0] == pytest.approx(0.005)
    assert set(result["evidence_role"]) == {"post_training_diagnostic_only"}


def test_extreme_contrasts_do_not_treat_images_as_replicates() -> None:
    rows = []
    for control in ("R1", "R2"):
        for index, (tier, value, seed) in enumerate(
            [("S", 1.0, 1), ("S", 2.0, 2), ("H", 5.0, 1), ("H", 7.0, 2)]
        ):
            rows.append(
                {
                    "triad_id": f"{control}_{index}",
                    "control": control,
                    "cohort_code": tier,
                    "phase": "A",
                    "budget": 600,
                    "training_seed": seed,
                    "machine_pair": "same_machine",
                    "any_resumed": False,
                    "feature": value,
                }
            )

    result = summarize_extreme_feature_contrasts(
        pd.DataFrame(rows), ["feature"], random_seed=7, bootstrap_samples=200
    )

    overall = result.loc[result["analysis_scope"] == "all"]
    assert set(overall["control"]) == {"R1", "R2"}
    assert set(overall["n_exceptional"]) == {2}
    assert set(overall["n_harmful"]) == {2}
    assert set(overall["mean_difference_S_minus_H"]) == {-4.5}
    assert set(overall["statistical_unit"]) == {"triad"}


def test_stratified_and_leave_one_seed_out_keep_controls_separate() -> None:
    rows = []
    for control in ("R1", "R2"):
        for seed in (1, 2):
            rows.extend(
                [
                    {
                        "triad_id": f"{control}_{seed}_S",
                        "control": control,
                        "cohort_code": "S",
                        "phase": "A",
                        "budget": 600,
                        "training_seed": seed,
                        "feature": float(seed),
                    },
                    {
                        "triad_id": f"{control}_{seed}_H",
                        "control": control,
                        "cohort_code": "H",
                        "phase": "A",
                        "budget": 600,
                        "training_seed": seed,
                        "feature": float(seed + 4),
                    },
                ]
            )
    frame = pd.DataFrame(rows)

    stratified = build_stratified_extreme_contrasts(frame, ["feature"])
    leaveout = summarize_leave_one_group_out(
        frame, ["feature"], group_column="training_seed"
    )

    assert len(stratified) == 4
    assert set(stratified["mean_difference_S_minus_H"]) == {-4.0}
    assert set(stratified["n_exceptional"]) == {1}
    assert set(leaveout["omitted_training_seed"].astype(int)) == {1, 2}
    assert set(leaveout["mean_difference_S_minus_H"]) == {-4.0}
