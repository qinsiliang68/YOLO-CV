from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from stage1_dynamic_replay_v3.provenance import ProvenanceError, build_source_tree_identity, validate_live_source


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "stage1_dynamic_replay_v3").mkdir(parents=True)
    (repo / "stage1_dynamic_replay_v3" / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    return repo


def test_live_source_recomputes_commit_and_tree(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    identity = build_source_tree_identity(repo)
    report = validate_live_source(
        repo,
        expected_git_commit=identity.git_commit,
        expected_source_tree_sha256=identity.source_tree_sha256,
    )
    assert report.status == "PASS"

    with pytest.raises(ProvenanceError, match="commit"):
        validate_live_source(repo, expected_git_commit="0" * 40, expected_source_tree_sha256=identity.source_tree_sha256)


def test_untracked_importable_source_fails(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    identity = build_source_tree_identity(repo)
    (repo / "stage1_dynamic_replay_v3" / "untracked.py").write_text("BAD = True\n", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="untracked"):
        validate_live_source(
            repo,
            expected_git_commit=identity.git_commit,
            expected_source_tree_sha256=identity.source_tree_sha256,
        )
