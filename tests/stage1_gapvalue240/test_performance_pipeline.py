from __future__ import annotations

import pandas as pd

from stage1_gapvalue240.performance_pipeline import (
    build_120_control_gates,
    build_240_control_gates,
    classify_outcome_cohorts,
)


def _frontier(values: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fn_budget": list(range(len(values))),
            "actual_fn": list(range(len(values))),
            "TN": values,
            "FP": [100 - value for value in values],
            "threshold": [0.5] * len(values),
        }
    )


def test_240_gate_requires_treatment_to_beat_r1_and_r2() -> None:
    matrix = pd.DataFrame(
        {
            "run_slot": ["T", "R1", "R2"],
            "triad_id": ["TRIAD_001"] * 3,
            "condition_id": ["A01"] * 3,
            "arm": ["T", "R1", "R2"],
            "training_seed": [7, 7, 7],
        }
    )
    frontiers = {
        "T": _frontier([10, 21, 31, 41]),
        "R1": _frontier([10, 20, 30, 40]),
        # T is worse than R2 at the baseline FN budget.
        "R2": _frontier([10, 22, 32, 42]),
    }

    pairs, gates = build_240_control_gates(matrix, frontiers, baseline_fn=2)

    assert len(pairs) == 2
    assert set(pairs["control"]) == {"R1", "R2"}
    assert not gates.loc[0, "paired_control_pass"]


def test_120_gate_requires_median_and_two_of_three_random_controls() -> None:
    hn = _frontier([10, 21, 31, 41])
    controls = {
        "RN1A-01": _frontier([10, 20, 30, 40]),
        "RN1B-01": _frontier([10, 19, 29, 39]),
        "RN1C-01": _frontier([10, 22, 32, 42]),
    }

    pairs, gate = build_120_control_gates(
        "HN1-01", hn, controls, baseline_fn=2
    )

    assert len(pairs) == 4  # three controls plus the median reference
    assert pairs.loc[pairs["control"] != "RN_MEDIAN", "real_gain"].sum() == 2
    assert gate["wins_individual_controls"] == 2
    assert gate["paired_control_pass"]


def test_outcome_cohort_keeps_absolute_and_method_gates_separate() -> None:
    rows = pd.DataFrame(
        {
            "run_id": ["strong", "robust", "absolute_only", "controlled", "harmful"],
            "performance_class": [
                "DUAL_GAIN",
                "DUAL_GAIN",
                "FN_SAFE_GAIN",
                "CONTROLLED_GAIN_2",
                "DOMINATED",
            ],
            "paired_control_pass": [True, True, False, True, False],
            "safe_frontier_dominant": [False, True, False, False, False],
            "paired_safe_frontier_pass": [False, True, False, False, False],
            "paired_operating_harmful": [False, False, False, False, True],
            "delta_TN_at_baseline_fn": [10, 10, 10, 10, -10],
            "all_controls_dominated": [False, False, False, False, True],
        }
    )

    result = classify_outcome_cohorts(rows).set_index("run_id")

    assert result.loc["strong", "outcome_cohort"] == "LOCAL_PARETO_DOUBLE_GATE"
    assert result.loc["robust", "outcome_cohort"] == "ROBUST_SAFE_DOUBLE_GATE"
    assert result.loc["absolute_only", "outcome_cohort"] == "ABSOLUTE_ONLY"
    assert result.loc["controlled", "outcome_cohort"] == "SECONDARY_CONTROLLED"
    assert result.loc["harmful", "outcome_cohort"] == "JOINTLY_HARMFUL"
