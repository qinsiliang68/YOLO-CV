import pytest
from stage1_sctsr_v4.epoch_transaction import EpochTransaction
from stage1_sctsr_v4.errors import SctsrError

def test_atomic_epoch_commit(tmp_path):
    root=tmp_path/'04_ledgers';tx=EpochTransaction(root,'r',121,1);tx.begin();tx.write_json('x.json',{'a':1});m=tx.commit();assert tx.complete.is_dir();assert not tx.inprogress.exists();assert m['epoch']==121;assert (tmp_path/'ROLLING_RECOVERY_POINTER.json').is_file()

def test_exception_quarantines_partial(tmp_path):
    root=tmp_path/'04_ledgers'
    with pytest.raises(RuntimeError):
        with EpochTransaction(root,'r',121,1) as tx:
            tx.write_json('x.json',{'a':1});raise RuntimeError('kill')
    assert list((tmp_path/'09_quarantine').glob('*.quarantined.*'))

def test_duplicate_generation_rejected(tmp_path):
    tx=EpochTransaction(tmp_path/'x','r',1,1);tx.begin()
    with pytest.raises(SctsrError):EpochTransaction(tmp_path/'x','r',1,1).begin()
