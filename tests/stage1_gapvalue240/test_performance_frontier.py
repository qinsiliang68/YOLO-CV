from __future__ import annotations

import pandas as pd

from stage1_gapvalue240.performance_frontier import (
    build_method_repeatability,
    classify_frontier_against_reference,
    compare_frontiers,
    frontier_from_predictions,
)


def _frontier(values: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fn_budget": list(range(len(values))),
            "actual_fn": list(range(len(values))),
            "TN": values,
            "FP": [100 - value for value in values],
            "threshold": [1.0 - index / 100 for index in range(len(values))],
        }
    )


def test_frontier_uses_whole_tie_groups_and_best_tn_at_each_fn_budget() -> None:
    predictions = pd.DataFrame(
        {
            "sample_id": ["d1", "d2", "n1", "n2"],
            "y_true": [1, 1, 0, 0],
            # d2 and n1 must move as one tie group.
            "score_raw": [0.9, 0.5, 0.5, 0.1],
        }
    )

    frontier = frontier_from_predictions(predictions, score_column="score_raw")

    assert frontier["fn_budget"].tolist() == [0, 1, 2]
    assert frontier["actual_fn"].tolist() == [0, 1, 2]
    assert frontier["TN"].tolist() == [1, 2, 2]
    assert frontier.loc[frontier["fn_budget"] == 1, "threshold"].iloc[0] == 0.9


def test_frontier_classification_distinguishes_real_and_controlled_gain() -> None:
    baseline = _frontier([10, 20, 30, 40, 50, 60, 70, 80])

    dual = _frontier([10, 20, 31, 51, 52, 62, 72, 82])
    safe = _frontier([10, 20, 30, 40, 51, 61, 71, 81])
    controlled = _frontier([10, 20, 30, 40, 50, 61, 71, 81])
    dominated = _frontier([9, 19, 29, 39, 49, 59, 69, 79])

    assert classify_frontier_against_reference(dual, baseline, baseline_fn=4)[
        "performance_class"
    ] == "DUAL_GAIN"
    assert classify_frontier_against_reference(safe, baseline, baseline_fn=4)[
        "performance_class"
    ] == "FN_SAFE_GAIN"
    assert classify_frontier_against_reference(controlled, baseline, baseline_fn=4)[
        "performance_class"
    ] == "CONTROLLED_GAIN_1"
    assert classify_frontier_against_reference(dominated, baseline, baseline_fn=4)[
        "performance_class"
    ] == "DOMINATED"


def test_compare_frontiers_never_compares_different_fn_budgets() -> None:
    baseline = _frontier([10, 20, 30])
    candidate = _frontier([11, 19, 35])

    compared = compare_frontiers(candidate, baseline)

    assert compared["fn_budget"].tolist() == [0, 1, 2]
    assert compared["delta_TN"].tolist() == [1, -1, 5]
    assert (compared["candidate_fn_budget"] == compared["reference_fn_budget"]).all()


def test_method_repeatability_requires_double_gate_and_multiple_seeds() -> None:
    rows = pd.DataFrame(
        {
            "experiment_family": ["240"] * 6 + ["40"],
            "condition_id": ["A"] * 3 + ["B"] * 3 + ["HN01"],
            "training_seed": [1, 2, 3, 1, 2, 3, 20260606],
            "absolute_baseline_pass": [True, True, False, True, True, True, True],
            "paired_control_pass": [True, True, True, False, False, False, True],
        }
    )

    result = build_method_repeatability(rows).set_index("condition_id")

    assert result.loc["A", "double_gate_passes"] == 2
    assert result.loc["A", "repeatability_class"] == "REPEATABLE_STRONG"
    assert result.loc["B", "repeatability_class"] == "UNSTABLE_OR_HARMFUL"
    assert result.loc["HN01", "repeatability_class"] == "SINGLE_SEED_PROMISING"
