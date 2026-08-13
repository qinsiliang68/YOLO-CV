import hashlib
from pathlib import Path
import pytest
from stage1_sctsr_v4.asset_registry import AssetRecord,AssetRegistry,validate_asset_registry
from stage1_sctsr_v4.errors import ErrorCode,SctsrError

def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest().upper()

def test_asset_registry_validates_exact_file(tmp_path):
    p=tmp_path/'a.bin';p.write_bytes(b'abc')
    r=AssetRegistry('stage1.sctsr.asset_registry.v1',2000,(AssetRecord('a','a.bin',sha(b'abc'),3,'TEST'),),False)
    out=validate_asset_registry(r,tmp_path);assert out['status']=='PASS'

def test_asset_sha_mismatch_fails(tmp_path):
    p=tmp_path/'a.bin';p.write_bytes(b'abc')
    r=AssetRegistry('stage1.sctsr.asset_registry.v1',2000,(AssetRecord('a','a.bin','0'*64,3,'TEST'),),False)
    with pytest.raises(SctsrError) as e:validate_asset_registry(r,tmp_path)
    assert e.value.code is ErrorCode.ASSET_VALIDATION_FAILED

def test_missing_required_asset_fails(tmp_path):
    r=AssetRegistry('stage1.sctsr.asset_registry.v1',2000,(AssetRecord('a','missing','0'*64,0,'TEST'),),False)
    with pytest.raises(SctsrError):validate_asset_registry(r,tmp_path)

def test_val_target_cannot_be_claimed(tmp_path):
    r=AssetRegistry('stage1.sctsr.asset_registry.v1',2000,(),True)
    with pytest.raises(SctsrError) as e:validate_asset_registry(r,tmp_path)
    assert e.value.code is ErrorCode.BLOCKED_BY_VAL_TARGET
