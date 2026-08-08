import pytest
from stage1_gapvalue240.util import atomic_write_text

def test_no_overwrite(tmp_path):
    p=tmp_path/'x.txt'; atomic_write_text(p,'a')
    with pytest.raises(FileExistsError): atomic_write_text(p,'b')
    assert p.read_text()=='a'


def test_atomic_write_uses_a_short_temp_name_for_deep_windows_paths(tmp_path):
    parent = tmp_path
    while len(str(parent)) < 225:
        parent = parent / "deep"
    parent.mkdir(parents=True)
    target = parent / "ATTEMPT_ARCHIVE_MANIFEST.json"

    assert len(str(target)) < 260
    atomic_write_text(target, "ok")

    assert target.read_text() == "ok"
    assert not list(parent.glob("*.tmp"))
