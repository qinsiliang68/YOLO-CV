import pytest
from stage1_gapvalue240.util import atomic_write_text

def test_no_overwrite(tmp_path):
    p=tmp_path/'x.txt'; atomic_write_text(p,'a')
    with pytest.raises(FileExistsError): atomic_write_text(p,'b')
    assert p.read_text()=='a'
