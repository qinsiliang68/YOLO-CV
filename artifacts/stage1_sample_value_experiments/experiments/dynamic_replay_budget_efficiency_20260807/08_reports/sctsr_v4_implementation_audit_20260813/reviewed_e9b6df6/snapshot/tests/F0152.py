import pytest
from stage1_sctsr_v4.errors import ErrorCode,SctsrError
from stage1_sctsr_v4.statistics import Endpoint,exact_paired_sign_flip,holm_adjust,paired_contrast,validate_confirmation,validate_discovery

def test_exact_sign_flip_small_n():assert exact_paired_sign_flip([1,1])==0.25

def test_missing_pair_fails():
    with pytest.raises(SctsrError) as e:paired_contrast('C',{1:Endpoint(1,1,1)},{2:Endpoint(1,1,1)})
    assert e.value.code is ErrorCode.STATISTICS_PAIR_MISSING

def test_discovery_7_of_8_advances():
    t={i:Endpoint(0.6 if i<7 else 0.49,100,1) for i in range(8)};c={i:Endpoint(0.5,99,2) for i in range(8)};r=paired_contrast('C',t,c);assert validate_discovery(r)=='SUPPORTED_FOR_CONFIRMATION_NOT_EFFICACY'

def test_confirmation_12_of_14_but_negative_worst_contradicted():
    t={i:Endpoint(0.6 if i<12 else 0.49,100,1) for i in range(14)};c={i:Endpoint(0.5,99,2) for i in range(14)};assert validate_confirmation(paired_contrast('C',t,c))=='CONTRADICTED'

def test_holm_is_stepdown():
    out=holm_adjust({'a':0.001,'b':0.02,'c':0.2});assert out[0]['reject'] is True;assert out[-1]['reject'] is False;assert [x['holm_rank'] for x in out]==[1,2,3]
