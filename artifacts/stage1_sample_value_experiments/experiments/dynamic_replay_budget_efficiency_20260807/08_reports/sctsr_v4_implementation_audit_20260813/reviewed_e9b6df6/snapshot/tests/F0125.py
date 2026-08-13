from dataclasses import replace
import pytest
from stage1_sctsr_v4.errors import ErrorCode,SctsrError
from stage1_sctsr_v4.identity_pool import IdentityPool,IdentityRecord,identity_digest,partition_five_groups

def test_identity_digest_golden():
    records=(IdentityRecord('b',0,'N','E',0,'g'),IdentityRecord('a',1,'D','H',1,'h'))
    assert identity_digest(records)=='67A7FB52C1DDC3F6721B50734197936643834FCC5B3E4AF6221CE1B3B63F221D'

def test_synthetic_t_pool_valid(synthetic_fixture):
    synthetic_fixture.t_pool.validate(base_denominator=synthetic_fixture.base_denominator,base_ids=set(synthetic_fixture.base_ids));assert len(synthetic_fixture.t_pool.records)==50

def test_five_groups_equal_and_exhaustive(synthetic_fixture):
    g=partition_five_groups(synthetic_fixture.t_pool,base_denominator=synthetic_fixture.base_denominator);assert set(g)=={f'G{i}' for i in range(5)};assert {len(v) for v in g.values()}=={10};assert len({r.sample_id for v in g.values() for r in v})==50

def test_duplicate_pool_rejected(synthetic_fixture):
    pool=IdentityPool(synthetic_fixture.t_pool.spec,(synthetic_fixture.t_pool.records[0],)*50)
    with pytest.raises(SctsrError) as e:pool.validate(base_denominator=synthetic_fixture.base_denominator)
    assert e.value.code is ErrorCode.DUPLICATE_IDENTITY

def test_t_cannot_be_labeled_validated_selector(synthetic_fixture):
    bad=replace(synthetic_fixture.t_pool.spec,selection_semantic='VALIDATED_SELECTOR')
    with pytest.raises(SctsrError) as e:IdentityPool(bad,synthetic_fixture.t_pool.records).validate(base_denominator=synthetic_fixture.base_denominator)
    assert e.value.code is ErrorCode.T_POOL_ROLE_MISMATCH
