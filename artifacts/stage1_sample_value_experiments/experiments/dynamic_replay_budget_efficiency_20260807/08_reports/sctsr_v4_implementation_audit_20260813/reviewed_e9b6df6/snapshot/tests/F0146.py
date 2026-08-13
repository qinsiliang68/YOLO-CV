from __future__ import annotations

import pytest

from stage1_sctsr_v4.arm_spec import ArmId
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.schedule import validate_phase1_schedule_registry


def test_sa_069_all_eight_arms_are_required_for_cross_arm_validation(synthetic_fixture, phase1_schedules):
    plans = dict(phase1_schedules)
    del plans[ArmId.R2_F]

    with pytest.raises(SctsrError) as exc:
        validate_phase1_schedule_registry(
            plans,
            t_pool_digest=synthetic_fixture.t_pool.spec.identity_digest,
            r1_pool_digest=synthetic_fixture.r1_result.pool.spec.identity_digest,
            r2_pool_digest=synthetic_fixture.r2_result.pool.spec.identity_digest,
        )

    assert exc.value.code is ErrorCode.CONFIGURATION_MISMATCH


def test_sa_074_cross_arm_registry_binds_t_r1_r2_and_parity(synthetic_fixture, phase1_schedules):
    result = validate_phase1_schedule_registry(
        phase1_schedules,
        t_pool_digest=synthetic_fixture.t_pool.spec.identity_digest,
        r1_pool_digest=synthetic_fixture.r1_result.pool.spec.identity_digest,
        r2_pool_digest=synthetic_fixture.r2_result.pool.spec.identity_digest,
    )

    assert result["status"] == "PASS"
    assert result["base_denominator"] == synthetic_fixture.base_denominator
    assert len(result["schedule_digests"]) == 8
