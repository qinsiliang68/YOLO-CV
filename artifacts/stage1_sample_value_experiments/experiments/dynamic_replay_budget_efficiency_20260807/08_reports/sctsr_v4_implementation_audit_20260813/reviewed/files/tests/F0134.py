import pytest
from stage1_sctsr_v4.errors import ErrorCode,SctsrError
from stage1_sctsr_v4.qrad_contract import TargetAlignmentConfig,reject_weighted_qrad,validate_candidate_signal_semantic

def test_weighted_qrad_rejected():
    with pytest.raises(SctsrError) as e:reject_weighted_qrad({'weights':{'Q':1}})
    assert e.value.code is ErrorCode.WEIGHTED_QRAD_FORBIDDEN

def test_target_alignment_blocked_without_val_target():
    with pytest.raises(SctsrError) as e:TargetAlignmentConfig(True,'A'*64,True,True,'FN95',1).validate(val_target_available=False)
    assert e.value.code is ErrorCode.BLOCKED_BY_VAL_TARGET

def test_candidate_signal_cannot_be_called_utility():
    with pytest.raises(SctsrError):validate_candidate_signal_semantic('RHO','UTILITY')

def test_candidate_signal_nonutility_allowed():validate_candidate_signal_semantic('RHO','CANDIDATE_SIGNAL_NOT_UTILITY')
