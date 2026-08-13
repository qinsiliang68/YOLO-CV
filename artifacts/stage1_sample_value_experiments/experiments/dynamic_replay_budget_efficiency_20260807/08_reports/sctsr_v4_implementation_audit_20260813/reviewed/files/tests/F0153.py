from __future__ import annotations

import math

import pytest

from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.statistics import Endpoint, holm_adjust, load_contrast_family_spec, paired_contrast, validate_confirmation, validate_discovery


def test_seven_of_eight_negative_deltas_are_contradicted_not_supported():
    treatment = {seed: Endpoint(0.4 if seed < 7 else 0.51, 100, 1) for seed in range(8)}
    comparator = {seed: Endpoint(0.5, 100, 1) for seed in range(8)}
    result = paired_contrast("C01", treatment, comparator)
    assert validate_discovery(result) == "CONTRADICTED"


def test_unreachable_target_anchor_fails_confirmation_closed():
    treatment = {seed: Endpoint(0.6, 100, None) for seed in range(14)}
    comparator = {seed: Endpoint(0.5, 99, 2) for seed in range(14)}
    result = paired_contrast("C01", treatment, comparator)
    assert validate_confirmation(result) == "FAIL_UNREACHABLE_TARGET_TN"


@pytest.mark.parametrize("bad_p", [-0.1, 1.1, math.nan, math.inf])
def test_holm_rejects_invalid_raw_p_values(bad_p):
    with pytest.raises(SctsrError) as caught:
        holm_adjust({"C01": bad_p})
    assert caught.value.code is ErrorCode.SCHEMA_VALIDATION_FAILED


def test_holm_requires_complete_frozen_c01_to_c08_family():
    with pytest.raises(SctsrError) as caught:
        holm_adjust({"C01": 0.01}, expected_contrast_ids=tuple(f"C{i:02d}" for i in range(1, 9)))
    assert caught.value.code is ErrorCode.STATISTICS_PAIR_MISSING


def test_confirmation_allows_twelve_positive_and_two_zero_with_nonnegative_worst():
    treatment = {seed: Endpoint(0.6 if seed < 12 else 0.5, 100, 1) for seed in range(14)}
    comparator = {seed: Endpoint(0.5, 99, 2) for seed in range(14)}
    result = paired_contrast("C01", treatment, comparator, treatment_id="T_F", comparator_id="R2_F")
    assert validate_confirmation(result) == "SUPPORTED"
    assert result.positive_count == 12
    assert result.worst_delta == 0
    assert len(result.treatment_endpoints) == 14


def test_holm_golden_threshold_adjustment_and_stepdown_stop():
    result = holm_adjust({"C01": 0.01, "C02": 0.03, "C03": 0.04}, alpha=0.05)
    assert [row["holm_threshold"] for row in result] == pytest.approx([0.05 / 3, 0.025, 0.05])
    assert [row["reject"] for row in result] == [True, False, False]
    assert [row["holm_adjusted_p"] for row in result] == pytest.approx([0.03, 0.06, 0.06])


def test_registered_contrast_family_is_complete_but_not_yet_formally_frozen(repository_root):
    path = repository_root / "configs" / "stage1_sctsr_v4" / "contrast_family_v1.json"
    spec = load_contrast_family_spec(path)
    assert spec.contrasts == tuple(f"C{i:02d}" for i in range(1, 9))
    with pytest.raises(SctsrError) as caught:
        load_contrast_family_spec(path, require_release_frozen=True)
    assert caught.value.code is ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED
