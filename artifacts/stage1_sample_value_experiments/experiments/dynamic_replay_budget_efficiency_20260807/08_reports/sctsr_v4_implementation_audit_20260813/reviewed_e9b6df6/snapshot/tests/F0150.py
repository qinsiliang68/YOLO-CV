import pytest
from stage1_sctsr_v4.errors import ErrorCode,SctsrError
from stage1_sctsr_v4.rate_spec import IDENTITY_POOL_RATE
from stage1_sctsr_v4.short_branch_scaffold import ShortBranchSpec,require_predictor_training_allowed

def test_disabled_short_branch_schema_valid():ShortBranchSpec(120,5,IDENTITY_POOL_RATE,'A'*64,'B'*64,False).validate()

def test_enabled_before_gate_rejected():
    with pytest.raises(SctsrError) as e:ShortBranchSpec(120,5,IDENTITY_POOL_RATE,'A'*64,'B'*64,True).validate(phase1_gate_passed=False)
    assert e.value.code is ErrorCode.SHORT_BRANCH_DISABLED

def test_rl_selector_forbidden_even_after_gate():
    with pytest.raises(SctsrError):require_predictor_training_allowed(phase1_gate_passed=True,model_family='REINFORCEMENT_LEARNING')
