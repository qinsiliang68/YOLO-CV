from copy import deepcopy
import pytest
from stage1_sctsr_v4.contracts import require_synthetic_or_authorized,validate_contract_files,validate_contract_mapping
from stage1_sctsr_v4.errors import ErrorCode,SctsrError
from stage1_sctsr_v4.serialization import load_json

def test_repository_contract_passes(repository_root):
    r=validate_contract_files(repository_root/'configs/stage1_sctsr_v4/contract_v1.json',repository_root/'configs/stage1_sctsr_v4/arms_phase1_v1.json')
    assert r.status=='PASS'

def test_anchor_change_rejected(repository_root):
    c=load_json(repository_root/'configs/stage1_sctsr_v4/contract_v1.json');c['fixed_anchors']['fallback_epoch']=159
    with pytest.raises(SctsrError):validate_contract_mapping(c)

def test_best_pt_rejected(repository_root):
    c=load_json(repository_root/'configs/stage1_sctsr_v4/contract_v1.json');c['formal_checkpoint']='best.pt'
    with pytest.raises(SctsrError) as e:validate_contract_mapping(c)
    assert e.value.code is ErrorCode.BEST_PT_FORBIDDEN

def test_formal_defaults_off(repository_root):
    c=load_json(repository_root/'configs/stage1_sctsr_v4/contract_v1.json');c['formal_execution_default']=True
    with pytest.raises(SctsrError):validate_contract_mapping(c)

def test_synthetic_authorized_without_release():require_synthetic_or_authorized('synthetic')

def test_formal_rejected_without_release(tmp_path):
    with pytest.raises(SctsrError) as e:require_synthetic_or_authorized('formal',tmp_path/'missing.json')
    assert e.value.code is ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED
