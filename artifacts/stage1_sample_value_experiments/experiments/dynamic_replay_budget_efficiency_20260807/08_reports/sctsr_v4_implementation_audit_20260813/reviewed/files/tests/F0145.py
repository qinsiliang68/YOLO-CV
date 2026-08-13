from __future__ import annotations

import pytest

from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.schedule import schedule_from_dict
from stage1_sctsr_v4.serialization import stable_digest


def _raw_t_u():
    ids = {f"G{group}": [f"S{group}_{index}" for index in range(10)] for group in range(5)}
    epochs = []
    for epoch in range(1, 201):
        active = epoch >= 121
        group = f"G{(epoch - 121) % 5}" if active else None
        rate = {
            "numerator": 5 if active else 0,
            "denominator": 1000,
            "semantic": "PER_EPOCH_REPLAY_RATE",
            "denominator_role": "CANONICAL_BASE_OPTIMIZER_VISIBLE_EXPOSURE",
            "canonical_token": "1/200" if active else "0/1",
        }
        epochs.append(
            {
                "epoch": epoch,
                "arm_id": "T_U",
                "identity_policy": "T_STRESS" if active else "NONE",
                "schedule_family": "U" if active else "NR",
                "rate": rate,
                "group_ids": [group] if active else [],
                "sample_ids": ids[group] if active else [],
                "fallback_state": "NOT_APPLICABLE",
            }
        )
    raw = {
        "schema_version": "stage1.sctsr.schedule.v1",
        "arm_id": "T_U",
        "base_denominator": 2000,
        "identity_pool_digest": "A" * 64,
        "epochs": epochs,
    }
    raw["plan_digest"] = stable_digest({
        "arm_id": raw["arm_id"],
        "base_denominator": raw["base_denominator"],
        "identity_pool_digest": raw["identity_pool_digest"],
        "epochs": raw["epochs"],
    })
    return raw


def _refresh_digest(raw):
    raw["plan_digest"] = stable_digest({
        "arm_id": raw["arm_id"],
        "base_denominator": raw["base_denominator"],
        "identity_pool_digest": raw["identity_pool_digest"],
        "epochs": raw["epochs"],
    })


def test_sa_070_forged_plan_digest_is_rejected():
    raw = _raw_t_u()
    raw["plan_digest"] = "FORGED"

    with pytest.raises(SctsrError) as exc:
        schedule_from_dict(raw)

    assert exc.value.code is ErrorCode.IDENTITY_DIGEST_MISMATCH


def test_sa_071_empty_treatment_schedule_is_rejected():
    raw = _raw_t_u()
    for epoch in raw["epochs"]:
        epoch["sample_ids"] = []
    _refresh_digest(raw)

    with pytest.raises(SctsrError) as exc:
        schedule_from_dict(raw)

    assert exc.value.code is ErrorCode.SCHEDULE_EXPOSURE_MISMATCH


def test_sa_072_rate_occurrence_mismatch_is_rejected():
    raw = _raw_t_u()
    raw["epochs"][120]["rate"] = {
        "numerator": 0,
        "denominator": 1000,
        "semantic": "PER_EPOCH_REPLAY_RATE",
        "denominator_role": "CANONICAL_BASE_OPTIMIZER_VISIBLE_EXPOSURE",
        "canonical_token": "0/1",
    }
    _refresh_digest(raw)

    with pytest.raises(SctsrError) as exc:
        schedule_from_dict(raw)

    assert exc.value.code is ErrorCode.SCHEDULE_EXPOSURE_MISMATCH
