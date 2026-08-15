import subprocess
from pathlib import Path

import pytest

from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4 import source_identity
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

    assert validate_source_tree_manifest(manifest, tmp_path, require_clean=False)["status"] == "PASS"

    (package / "unregistered.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(SctsrError) as caught:
        validate_source_tree_manifest(manifest, tmp_path, require_clean=False)
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


def test_source_manifest_records_runtime_dependency_identity(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    manifest = build_source_tree_manifest(tmp_path, ["package"])

    environment = manifest["runtime_environment"]
    assert environment["python"]["status"] == "AVAILABLE"
    assert environment["python"]["version"]
    assert environment["torch"]["status"] in {"AVAILABLE", "UNAVAILABLE"}
    assert environment["nvidia_driver"]["status"] in {"AVAILABLE", "UNAVAILABLE"}
    assert environment["ultralytics"]["status"] in {"AVAILABLE", "UNAVAILABLE"}
    assert len(manifest["runtime_environment_digest"]) == 64


def test_source_manifest_validation_reprobes_live_runtime_and_rejects_drift(tmp_path, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    manifest = build_source_tree_manifest(tmp_path, ["package"])
    observed_calls = 0
    frozen = manifest["runtime_environment"]

    def drifted_probe(_root):
        nonlocal observed_calls
        observed_calls += 1
        changed = {**frozen, "nvidia_driver": {"status": "AVAILABLE", "versions": ["DRIFTED"]}}
        return changed

    monkeypatch.setattr(source_identity, "probe_runtime_environment", drifted_probe)
    with pytest.raises(SctsrError) as caught:
        validate_source_tree_manifest(manifest, tmp_path, require_clean=False)

    assert observed_calls == 1
    assert caught.value.code is ErrorCode.SOURCE_TREE_MISMATCH


def test_source_identity_ignores_ephemeral_interpreter_parent_directory(tmp_path, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")

    monkeypatch.setattr(source_identity.sys, "executable", str(tmp_path / ".tmp-first" / "Scripts" / "python.exe"))
    first = build_source_tree_manifest(tmp_path, ["package"])
    monkeypatch.setattr(source_identity.sys, "executable", str(tmp_path / ".tmp-second" / "Scripts" / "python.exe"))
    second = build_source_tree_manifest(tmp_path, ["package"])

    assert Path(first["runtime_environment"]["python"]["executable"]).name == "python.exe"
    assert first["runtime_environment"] == second["runtime_environment"]
    assert first["runtime_environment_digest"] == second["runtime_environment_digest"]
    assert first["source_tree_digest"] == second["source_tree_digest"]


def test_clean_source_manifest_revalidates_current_git_head_and_worktree(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "sctsr-test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "SCTSR Test"], cwd=tmp_path, check=True)
    package = tmp_path / "package"
    package.mkdir()
    source = package / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    outside = tmp_path / "tracked_outside_include.txt"
    outside.write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "add", "package/module.py", "tracked_outside_include.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "freeze"], cwd=tmp_path, check=True)

    manifest = build_source_tree_manifest(tmp_path, ["package"])
    assert manifest["git_dirty"] is False
    assert validate_source_tree_manifest(manifest, tmp_path, require_clean=True)["status"] == "PASS"

    outside.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(SctsrError) as dirty:
        validate_source_tree_manifest(manifest, tmp_path, require_clean=True)
    assert dirty.value.code is ErrorCode.SOURCE_TREE_MISMATCH

    subprocess.run(["git", "add", "tracked_outside_include.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "drift"], cwd=tmp_path, check=True)
    with pytest.raises(SctsrError) as head:
        validate_source_tree_manifest(manifest, tmp_path, require_clean=True)
    assert head.value.code is ErrorCode.SOURCE_TREE_MISMATCH
