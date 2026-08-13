import pytest
from stage1_sctsr_v4.branch_lineage import BranchLineage
from stage1_sctsr_v4.errors import ErrorCode,SctsrError

def lineage():return BranchLineage.create(logical_run_id='R',parent_id='P',parent_checkpoint_path='p.pt',parent_checkpoint_sha256='A'*64,training_seed=1,arm_id='T_U',child_source_tree_digest='B'*64,child_contract_digest='C'*64,created_at_utc='2026-08-12T00:00:00+00:00')

def test_lineage_recomputes_and_validates():
    x=lineage();x.validate(parent_sha='A'*64,training_seed=1,arm_id='T_U',source_digest='B'*64,contract_digest='C'*64);assert x.lineage_digest==x.compute_digest()

def test_lineage_wrong_parent_rejected():
    with pytest.raises(SctsrError) as e:lineage().validate(parent_sha='D'*64,training_seed=1,arm_id='T_U',source_digest='B'*64,contract_digest='C'*64)
    assert e.value.code is ErrorCode.BRANCH_LINEAGE_MISMATCH
