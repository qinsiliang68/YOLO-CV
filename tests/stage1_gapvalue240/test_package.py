from pathlib import Path
from stage1_gapvalue240.package_validation import validate_package

ROOT=Path(__file__).resolve().parents[2]

def test_package_structure(tmp_path):
    r=validate_package(ROOT,tmp_path/'validation.json')
    assert r['status']=='PASS'
