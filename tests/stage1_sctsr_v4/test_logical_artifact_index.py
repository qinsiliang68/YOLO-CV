import pytest
from stage1_sctsr_v4.errors import ErrorCode,SctsrError
from stage1_sctsr_v4.logical_artifact_index import LogicalArtifactEntry,LogicalArtifactIndex

def entry(epoch,owner):return LogicalArtifactEntry('R',epoch,owner,'P' if owner=='PARENT' else 'R',f'e{epoch}.json','A'*64,'B'*64,'C'*64,'D'*64)

def test_parent_and_child_ownership_valid():
    idx=LogicalArtifactIndex([entry(120,'PARENT'),entry(121,'CHILD')]);idx.validate();assert len(idx.digest)==64

def test_child_cannot_own_prebranch_epoch():
    with pytest.raises(SctsrError) as e:LogicalArtifactIndex([entry(100,'CHILD')]).validate()
    assert e.value.code is ErrorCode.LOGICAL_ARTIFACT_IDENTITY_MISMATCH
