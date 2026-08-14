from __future__ import annotations

import pytest

from stage1_sctsr_v4.asset_registry import load_asset_registry
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.formal_pool_inputs import (
    R2_APPROVED_IDENTITY_DIGEST,
    R2_APPROVED_SELECTION_SEED,
    T_CANONICAL_IDENTITY_DIGEST,
    build_registered_r1,
    build_registered_r2,
    load_formal_pool_inputs,
)


@pytest.fixture(scope="module")
def registered_inputs(repository_root):
    registry = load_asset_registry(repository_root / "configs/stage1_sctsr_v4/asset_registry_v1.json")
    return load_formal_pool_inputs(registry, repository_root)


def test_sa_045_registered_t_is_exact_3000_id_frozen_stress_set(registered_inputs):
    assert len(registered_inputs.t_pool.records) == 3000
    assert registered_inputs.t_pool.spec.identity_digest == T_CANONICAL_IDENTITY_DIGEST
    assert registered_inputs.t_pool.spec.selection_semantic == "HISTORICAL_SIGN_REVERSAL_STRESS_SET_NOT_VALIDATED_SELECTOR"


def test_sa_046_registered_r1_uses_all_120000_base_candidates(registered_inputs):
    result = build_registered_r1(registered_inputs, base_denominator=120000, selection_seed=20260812)
    assert result.audit.candidate_count == 120000
    assert result.audit.selected_count == 3000
    assert result.audit.candidate_universe_digest != "NOT_RECORDED"


def test_sa_053_registered_r2_uses_only_the_owner_approved_minimum_displacement_addendum(registered_inputs):
    result = build_registered_r2(
        registered_inputs,
        base_denominator=120000,
        selection_seed=R2_APPROVED_SELECTION_SEED,
    )

    assert result.pool.spec.identity_digest == R2_APPROVED_IDENTITY_DIGEST
    assert result.audit.displacement_count == 378
    assert result.audit.displacement_lower_bound == 378
    assert result.audit.relaxed_fields == ("oof_group_id",)
