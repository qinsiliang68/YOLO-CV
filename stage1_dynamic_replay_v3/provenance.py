"""Live source identity checks for formal Stage1 v3 workers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import subprocess
from typing import Iterable


SOURCE_PREFIXES = (
    "stage1_dynamic_replay_v3/",
    "scripts/stage1_dynamic_replay_v3/",
    "configs/stage1_dynamic_replay_v3/",
    "stage1_gapvalue240/",
    "YOLOv11/ultralytics/",
)
SOURCE_FILES = {"pyproject.toml", "uv.lock"}
IMPORTABLE_SUFFIXES = {".py", ".pyi", ".yaml", ".yml", ".json", ".toml"}


class ProvenanceError(ValueError):
    """Raised when a live checkout differs from the frozen source identity."""


@dataclass(frozen=True)
class SourceTreeIdentity:
    status: str
    git_commit: str
    source_tree_sha256: str
    file_count: int
    files: tuple[str, ...]


def _git(repo: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=repo, check=True, capture_output=True, text=True, encoding="utf-8"
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise ProvenanceError(f"git command failed: {detail.strip()}") from exc
    return result.stdout


def _is_controlled_source(relative_path: str) -> bool:
    normalized = PurePosixPath(relative_path.replace("\\", "/")).as_posix()
    return normalized in SOURCE_FILES or any(normalized.startswith(prefix) for prefix in SOURCE_PREFIXES)


def _tracked_source_files(repo: Path) -> list[str]:
    files = [line.strip() for line in _git(repo, "ls-files", "-z").split("\0") if line.strip()]
    selected = sorted(relative for relative in files if _is_controlled_source(relative))
    if not selected:
        raise ProvenanceError("source tree contains no controlled files")
    return selected


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(repo: Path, relative_paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = repo / Path(relative)
        if not path.is_file():
            raise ProvenanceError(f"tracked source file is missing: {relative}")
        digest.update(f"{relative}\0{path.stat().st_size}\0{_file_digest(path)}\n".encode("utf-8"))
    return digest.hexdigest()


def _untracked_importable_source(repo: Path) -> list[str]:
    lines = [
        line.strip()
        for line in _git(repo, "ls-files", "--others", "--exclude-standard", "-z").split("\0")
        if line.strip()
    ]
    return sorted(
        relative
        for relative in lines
        if _is_controlled_source(relative) and PurePosixPath(relative).suffix.lower() in IMPORTABLE_SUFFIXES
    )


def build_source_tree_identity(repo: str | Path) -> SourceTreeIdentity:
    root = Path(repo).resolve()
    commit = _git(root, "rev-parse", "HEAD").strip()
    if len(commit) != 40:
        raise ProvenanceError(f"git commit identity is invalid: {commit!r}")
    files = _tracked_source_files(root)
    return SourceTreeIdentity("PASS", commit, _tree_digest(root, files), len(files), tuple(files))


def validate_live_source(
    repo: str | Path, *, expected_git_commit: str, expected_source_tree_sha256: str
) -> SourceTreeIdentity:
    root = Path(repo).resolve()
    current_commit = _git(root, "rev-parse", "HEAD").strip()
    if current_commit.lower() != str(expected_git_commit).strip().lower():
        raise ProvenanceError(
            f"live git commit mismatch: expected {expected_git_commit}, observed {current_commit}"
        )
    untracked = _untracked_importable_source(root)
    if untracked:
        raise ProvenanceError(f"untracked importable source is forbidden: {untracked[:10]}")
    identity = build_source_tree_identity(root)
    if identity.source_tree_sha256.lower() != str(expected_source_tree_sha256).strip().lower():
        raise ProvenanceError(
            "live source tree sha256 mismatch: "
            f"expected {expected_source_tree_sha256}, observed {identity.source_tree_sha256}"
        )
    return identity


__all__ = [
    "ProvenanceError",
    "SourceTreeIdentity",
    "build_source_tree_identity",
    "validate_live_source",
]
