import pytest
from stage1_sctsr_v4.errors import ErrorCode,SctsrError
from stage1_sctsr_v4.rate_spec import IDENTITY_POOL_RATE,U_RATE,F_RATE,ZERO_RATE,RateSemantic,ReplayRateSpec

def test_frozen_rates_derive_counts():
    assert IDENTITY_POOL_RATE.derive_count(120000)==3000
    assert U_RATE.derive_count(120000)==600
    assert F_RATE.derive_count(120000)==1200
    assert ZERO_RATE.derive_count(120000)==0

def test_rate_reduces_and_basis_points():
    rate=ReplayRateSpec(10,2000,RateSemantic.PER_EPOCH_REPLAY_RATE)
    assert rate.canonical_token()=="1/200"
    assert rate.to_basis_points()==50

def test_float_rate_rejected():
    with pytest.raises(SctsrError) as e:ReplayRateSpec(0.5,1000,RateSemantic.PER_EPOCH_REPLAY_RATE)
    assert e.value.code is ErrorCode.FLOAT_RATE_FORBIDDEN

def test_boolean_rate_rejected():
    with pytest.raises(SctsrError):ReplayRateSpec(True,1000,RateSemantic.PER_EPOCH_REPLAY_RATE)

def test_non_integral_count_rejected():
    with pytest.raises(SctsrError) as e:U_RATE.derive_count(1999)
    assert e.value.code is ErrorCode.RATE_NOT_INTEGRAL
