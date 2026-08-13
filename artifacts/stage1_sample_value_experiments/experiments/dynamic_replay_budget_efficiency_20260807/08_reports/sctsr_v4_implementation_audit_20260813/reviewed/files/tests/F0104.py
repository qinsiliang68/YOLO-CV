import os
import pytest
from stage1_sctsr_v4.columnar import PORTABLE_MAGIC,parquet_engine_available,read_columnar,write_zstd_parquet
from stage1_sctsr_v4.errors import ErrorCode,SctsrError

def test_synthetic_fallback_is_explicitly_not_parquet(tmp_path):
    p=tmp_path/'x.parquet';m=write_zstd_parquet([{'a':1}],p,schema_version='x',allow_synthetic_portable_fallback=True)
    if parquet_engine_available():assert m.canonical_parquet and p.read_bytes()[:4]==b'PAR1'
    else:assert not m.canonical_parquet and p.read_bytes().startswith(PORTABLE_MAGIC);assert read_columnar(p,allow_synthetic_portable_fallback=True)==[{'a':1}]

def test_formal_write_requires_real_engine(tmp_path,monkeypatch):
    if parquet_engine_available():return
    monkeypatch.delenv('SCTSR_ALLOW_SYNTHETIC_COLUMNAR_FALLBACK',raising=False)
    with pytest.raises(SctsrError) as e:write_zstd_parquet([{'a':1}],tmp_path/'x.parquet',schema_version='x')
    assert e.value.code is ErrorCode.COLUMNAR_ENGINE_UNAVAILABLE

def test_synthetic_fallback_cannot_be_read_as_canonical(tmp_path):
    p=tmp_path/'x.parquet';write_zstd_parquet([{'a':1}],p,schema_version='x',allow_synthetic_portable_fallback=True)
    if not parquet_engine_available():
        with pytest.raises(SctsrError) as e:read_columnar(p)
        assert e.value.code is ErrorCode.SYNTHETIC_RESULT_MISLABELLED
