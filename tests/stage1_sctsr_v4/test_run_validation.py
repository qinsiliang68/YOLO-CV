import pytest
from stage1_sctsr_v4.errors import ErrorCode,SctsrError
from stage1_sctsr_v4.run_validation import validate_run_tree

def test_synthetic_canary_validates(canary_root):
    r=validate_run_tree(canary_root,allow_synthetic_portable_fallback=True);assert r['status']=='PASS';assert r['artifact_count']>100

def test_canary_uses_canonical_parquet_without_portable_fallback(canary_root):
    result = validate_run_tree(canary_root, allow_synthetic_portable_fallback=False)
    assert result["status"] == "PASS"
