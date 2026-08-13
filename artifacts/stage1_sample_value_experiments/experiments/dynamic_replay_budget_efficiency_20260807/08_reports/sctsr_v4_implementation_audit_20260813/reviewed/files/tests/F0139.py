from pathlib import Path
import pytest
from stage1_sctsr_v4.epoch_transaction import EpochTransaction
from stage1_sctsr_v4.errors import ErrorCode,SctsrError
from stage1_sctsr_v4.recovery import ResumeIdentity,find_last_complete_epoch,quarantine_inprogress,validate_recovery_pointer

def test_find_last_complete_and_pointer(tmp_path):
    root=tmp_path/'04';
    for e in (121,122):
        tx=EpochTransaction(root,'r',e,1);tx.begin();tx.write_json('x.json',{'e':e});tx.commit()
    assert find_last_complete_epoch(root)['epoch']==122;assert validate_recovery_pointer(tmp_path/'ROLLING_RECOVERY_POINTER.json')['epoch']==122

def test_quarantine_inprogress(tmp_path):
    p=tmp_path/'04'/'epoch_0123.generation_1.inprogress';p.mkdir(parents=True);(p/'x').write_text('x')
    moved=quarantine_inprogress(tmp_path/'04',tmp_path/'q',reason='kill');assert len(moved)==1 and Path(moved[0]).is_dir()

def test_resume_identity_mismatch_rejected():
    a=ResumeIdentity('r','A','T',1,'S','C','R','G');b=ResumeIdentity('r','B','T',1,'S','C','R','G')
    with pytest.raises(SctsrError) as e:a.validate(b)
    assert e.value.code is ErrorCode.RESUME_GENERATION_MISMATCH

def test_corrupt_pointer_rejected(tmp_path):
    p=tmp_path/'p.json';p.write_text('{bad')
    with pytest.raises(SctsrError):validate_recovery_pointer(p)
