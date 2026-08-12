from __future__ import annotations

import pytest

from stage1_dynamic_replay_v3.prediction_identity import PredictionIdentityError, validate_prediction_identity


def test_prediction_identity_requires_exact_ids_and_labels() -> None:
    expected = (("sample-a", 0), ("sample-b", 1))
    observed = (("sample-a", 0), ("sample-b", 1))
    report = validate_prediction_identity(expected, observed)
    assert report.status == "PASS"
    assert report.expected_digest == report.observed_digest


def test_wrong_ids_fail_even_when_counts_match() -> None:
    with pytest.raises(PredictionIdentityError, match="identity"):
        validate_prediction_identity(
            (("sample-a", 0), ("sample-b", 1)),
            (("WRONG_NORMAL", 0), ("WRONG_DEFECT", 1)),
        )
