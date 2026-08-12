from __future__ import annotations

import pandas as pd
import pytest

from stage1_gapvalue240.reversal_mechanisms import (
    ReversalAnalysisError,
    blocked_feature_contrasts,
    identify_same_selection_reversals,
)


def test_identify_same_selection_reversals_keeps_all_rows_in_spanning_blocks() -> None:
    rows = pd.DataFrame(
        {
            "triad_id": ["t1", "t2", "t3", "t4", "t5", "t6"],
            "sample_set_digest": ["a", "a", "a", "b", "b", "c"],
            "selected_count": [3] * 6,
            "training_seed": [1, 2, 3, 1, 2, 1],
            "condition_id": ["A", "A", "A", "B", "B", "C"],
            "exclusive_cohort": [
                "DUAL_IMPROVEMENT",
                "DUAL_HARM",
                "MIXED_OR_REVERSAL",
                "DUAL_IMPROVEMENT",
                "DUAL_HARM",
                "DUAL_IMPROVEMENT",
            ],
            "dual_improvement": [True, False, False, True, False, True],
            "dual_harm": [False, True, False, False, True, False],
        }
    )

    details, summary = identify_same_selection_reversals(
        rows,
        expected_triads=6,
    )

    assert details.triad_id.tolist() == ["t1", "t2", "t3", "t4", "t5"]
    assert summary.sample_set_digest.tolist() == ["a", "b"]
    assert summary.triad_count.tolist() == [3, 2]
    assert summary.dual_improvement_count.tolist() == [1, 1]
    assert summary.dual_harm_count.tolist() == [1, 1]


def test_blocked_permutation_finds_consistent_within_digest_difference() -> None:
    rows: list[dict[str, object]] = []
    for digest_index in range(6):
        rows.extend(
            [
                {
                    "sample_set_digest": f"d{digest_index}",
                    "exclusive_cohort": "DUAL_IMPROVEMENT",
                    "signal": 2.0 + digest_index,
                    "constant": 1.0,
                },
                {
                    "sample_set_digest": f"d{digest_index}",
                    "exclusive_cohort": "DUAL_HARM",
                    "signal": 1.0 + digest_index,
                    "constant": 1.0,
                },
            ]
        )
    frame = pd.DataFrame(rows)
    registry = pd.DataFrame(
        {
            "feature": ["signal", "constant"],
            "feature_family": ["TEST", "TEST"],
            "available_epoch": [0, 0],
            "allowed_as_predictor": [True, True],
            "analysis_role": ["SELECTION_NUMERIC_PREDICTOR"] * 2,
        }
    )

    result = blocked_feature_contrasts(
        frame,
        registry,
        bootstrap_resamples=1_000,
        random_seed=7,
    ).set_index("feature")

    assert result.loc["signal", "digest_equal_weight_mean_difference"] == pytest.approx(
        1.0
    )
    assert result.loc["signal", "exact_blocked_permutation_count"] == 64
    assert result.loc["signal", "permutation_p_two_sided"] == pytest.approx(2 / 64)
    assert result.loc["signal", "bootstrap_ci_low"] == pytest.approx(1.0)
    assert result.loc["signal", "bootstrap_ci_high"] == pytest.approx(1.0)
    assert result.loc["constant", "within_digest_constant_all"]
    assert result.loc["constant", "digest_equal_weight_mean_difference"] == 0.0


def test_reversal_identification_rejects_duplicate_triads() -> None:
    rows = pd.DataFrame(
        {
            "triad_id": ["t1", "t1"],
            "sample_set_digest": ["a", "a"],
            "selected_count": [3, 3],
            "training_seed": [1, 2],
            "condition_id": ["A", "A"],
            "exclusive_cohort": ["DUAL_IMPROVEMENT", "DUAL_HARM"],
            "dual_improvement": [True, False],
            "dual_harm": [False, True],
        }
    )
    with pytest.raises(ReversalAnalysisError, match="duplicate triad_id"):
        identify_same_selection_reversals(rows, expected_triads=2)
