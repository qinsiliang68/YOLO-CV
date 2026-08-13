import pytest

from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.source_identity import (
    build_source_tree_manifest,
    validate_source_tree_manifest,
)

def test_source_manifest_hashes_registered_files(tmp_path):
    (tmp_path/'a').mkdir();(tmp_path/'a'/'x.py').write_text('x=1\n')
    m=build_source_tree_manifest(tmp_path,['a']);assert len(m['files'])==1;assert len(m['source_tree_digest'])==64


def test_source_manifest_rejects_unregistered_importable_file(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "registered.py").write_text("VALUE = 1\n", encoding="utf-8")
    manifest = build_source_tree_manifest(tmp_path, ["package"])
    manifest["git_dirty"] = False

    assert validate_source_tree_manifest(manifest, tmp_path)["status"] == "PASS"

    (package / "unregistered.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(SctsrError) as caught:
        validate_source_tree_manifest(manifest, tmp_path)
    assert caught.value.code is ErrorCode.SOURCE_TREE_MISMATCH


def test_source_manifest_rejects_missing_or_overlapping_include_paths(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(SctsrError) as missing:
        build_source_tree_manifest(tmp_path, ["missing"])
    assert missing.value.code is ErrorCode.SOURCE_TREE_MISMATCH

    with pytest.raises(SctsrError) as overlapping:
        build_source_tree_manifest(tmp_path, ["package", "package/module.py"])
    assert overlapping.value.code is ErrorCode.SOURCE_TREE_MISMATCH
