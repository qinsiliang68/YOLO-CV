from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from .errors import ErrorCode, SctsrError
from .serialization import sha256_file, stable_digest


DEFAULT_ALLOWED_PREFIXES = (
    "stage1_sctsr_v4/",
    "scripts/stage1_sctsr_v4/",
    "configs/stage1_sctsr_v4/",
    "tests/stage1_sctsr_v4/",
    "docs/stage1_sctsr_v4/",
    "integrations/ultralytics/",
    "artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/08_reports/sctsr_v4_",
)
DEFAULT_ALLOWED_FILES = (
    ".gitattributes",
    "README.md",
    "docs/README.md",
    "pyproject.toml",
    "uv.lock",
    "requirements-sctsr-v4.txt",
)
DEFAULT_PROTECTED_PREFIXES = (
    "stage1_gapvalue240/",
    "stage1_dynamic_replay_v3/",
    "YOLOv11/ultralytics/",
)
DEFAULT_LEGACY_MARKERS = (
    "/04_run_queue/",
    "/04_run_queue_v2/",
    "/05_training_runs/",
    "/releases/",
    "/assignments/",
)
SOURCE_SUFFIXES = (".py", ".pyi", ".json", ".toml", ".yaml", ".yml")


def _git(root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=not binary,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SctsrError(
            ErrorCode.SOURCE_TREE_MISMATCH,
            "Repository audit Git command failed",
            observed={"arguments": list(arguments), "exception": str(exc)},
        ) from exc
    return completed.stdout if binary else completed.stdout.strip()


def _commit_time(root: Path, commit: str) -> datetime:
    value = str(_git(root, "show", "-s", "--format=%cI", commit))
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Implementation-start commit time is invalid") from exc
    if parsed.tzinfo is None:
        raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Implementation-start commit time lacks timezone")
    return parsed.astimezone(timezone.utc)


def _changed_paths(root: Path, baseline: str, source: str) -> list[tuple[str, str]]:
    output = str(_git(root, "diff", "--name-status", "--find-renames", f"{baseline}..{source}"))
    rows: list[tuple[str, str]] = []
    for line in output.splitlines():
        parts = line.split("\t")
        if not parts:
            continue
        status = parts[0]
        path = parts[-1].replace("\\", "/")
        rows.append((status, path))
    return rows


def _is_allowed(path: str, prefixes: Sequence[str], files: Sequence[str]) -> bool:
    return path in files or any(path.startswith(prefix) for prefix in prefixes)


def _owner_lifecycle(path: str) -> tuple[str, str]:
    if path.startswith("tests/"):
        return "SCTSR_V4_TEST_OWNER", "VERSIONED_TEST_EVIDENCE"
    if path.startswith("docs/") or path in {"README.md", ".gitattributes"}:
        return "SCTSR_V4_DOCUMENTATION_OWNER", "VERSIONED_HUMAN_CONTRACT"
    if path.startswith("artifacts/"):
        return "SCTSR_V4_AUDIT_OWNER", "IMMUTABLE_IMPLEMENTATION_EVIDENCE"
    if path.startswith("configs/") or path.endswith((".toml", ".lock", ".txt")):
        return "SCTSR_V4_CONFIGURATION_OWNER", "VERSIONED_MACHINE_CONTRACT"
    return "SCTSR_V4_IMPLEMENTATION_OWNER", "VERSIONED_SOURCE"


def audit_repository_state(
    repository_root: str | Path,
    *,
    baseline_commit: str,
    implementation_start_commit: str,
    implementation_source_commit: str,
    allowed_prefixes: Sequence[str] = DEFAULT_ALLOWED_PREFIXES,
    allowed_files: Sequence[str] = DEFAULT_ALLOWED_FILES,
    protected_prefixes: Sequence[str] = DEFAULT_PROTECTED_PREFIXES,
    legacy_markers: Sequence[str] = DEFAULT_LEGACY_MARKERS,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    head = str(_git(root, "rev-parse", "HEAD"))
    if head != implementation_source_commit:
        raise SctsrError(
            ErrorCode.SOURCE_TREE_MISMATCH,
            "Repository audit must run at the exact implementation source commit",
            observed=head,
            expected=implementation_source_commit,
        )
    _git(root, "merge-base", "--is-ancestor", baseline_commit, implementation_source_commit)
    tracked_status = str(_git(root, "status", "--porcelain=v1", "--untracked-files=no"))
    if tracked_status:
        raise SctsrError(
            ErrorCode.SOURCE_TREE_MISMATCH,
            "Implementation source checkout has tracked or staged drift",
            observed=tracked_status.splitlines(),
        )

    changed = _changed_paths(root, baseline_commit, implementation_source_commit)
    forbidden_changes = [path for _status, path in changed if not _is_allowed(path, allowed_prefixes, allowed_files)]
    deleted = [path for status, path in changed if status.startswith("D")]
    protected_changes = [
        path
        for _status, path in changed
        if any(path.startswith(prefix) for prefix in protected_prefixes)
    ]
    if forbidden_changes or deleted or protected_changes:
        raise SctsrError(
            ErrorCode.SOURCE_TREE_MISMATCH,
            "Implementation diff contains forbidden, deleted, or protected files",
            observed={
                "forbidden_changes": forbidden_changes,
                "deleted_files": deleted,
                "protected_changes": protected_changes,
            },
        )

    untracked_output = str(_git(root, "ls-files", "--others", "--exclude-standard"))
    untracked = [line.replace("\\", "/") for line in untracked_output.splitlines() if line]
    untracked_source = [
        path
        for path in untracked
        if _is_allowed(path, allowed_prefixes, ())
        and path.lower().endswith(SOURCE_SUFFIXES)
        and not path.startswith("artifacts/")
    ]
    if untracked_source:
        raise SctsrError(
            ErrorCode.SOURCE_TREE_MISMATCH,
            "Untracked importable or contract file exists inside an implementation root",
            observed=untracked_source,
        )

    changed_files: list[dict[str, Any]] = []
    for status, relative in changed:
        path = root / relative
        if not path.is_file():
            raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Changed-file ledger path is missing", artifact_path=str(path))
        owner, lifecycle = _owner_lifecycle(relative)
        changed_files.append(
            {
                "relative_path": relative,
                "git_status": status,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "owner": owner,
                "source": f"git:{implementation_source_commit}:{relative}",
                "consumer": "SCTSR_V4_IMPLEMENTATION_AND_INDEPENDENT_REVIEW",
                "lifecycle": lifecycle,
                "verification": "PATH_BYTES_SHA256_AND_GIT_COMMIT",
            }
        )
    changed_files.sort(key=lambda row: row["relative_path"])
    ledger_core = {
        "schema_version": "stage1.sctsr.changed_file_ledger.v1",
        "baseline_commit": baseline_commit,
        "implementation_source_commit": implementation_source_commit,
        "file_count": len(changed_files),
        "files": changed_files,
    }
    changed_file_ledger = {**ledger_core, "ledger_digest": stable_digest(ledger_core)}

    start_time = _commit_time(root, implementation_start_commit)
    tracked_paths = str(_git(root, "ls-tree", "-r", "--name-only", implementation_source_commit)).splitlines()
    normalized_tracked_paths = [path.replace("\\", "/") for path in tracked_paths]
    legacy_paths = sorted(
        path
        for path in normalized_tracked_paths
        if any(marker in f"/{path}" for marker in legacy_markers)
    )
    legacy_files: list[dict[str, Any]] = []
    for relative in legacy_paths:
        path = root / relative
        if not path.is_file():
            raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Protected legacy file is missing", artifact_path=str(path))
        baseline_oid = str(_git(root, "rev-parse", f"{baseline_commit}:{relative}"))
        source_oid = str(_git(root, "rev-parse", f"{implementation_source_commit}:{relative}"))
        normalized_worktree_oid = str(_git(root, "hash-object", f"--path={relative}", relative))
        baseline_bytes = _git(root, "cat-file", "blob", baseline_oid, binary=True)
        source_bytes = _git(root, "cat-file", "blob", source_oid, binary=True)
        assert isinstance(baseline_bytes, bytes) and isinstance(source_bytes, bytes)
        baseline_sha = hashlib.sha256(baseline_bytes).hexdigest().upper()
        source_sha = hashlib.sha256(source_bytes).hexdigest().upper()
        worktree_sha = sha256_file(path)
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        # Git commit timestamps have one-second resolution while NTFS mtimes are
        # sub-second.  Treat the commit's whole recorded second as pre-start.
        mtime_cutoff = start_time + timedelta(seconds=1)
        if baseline_oid != source_oid or baseline_sha != source_sha or normalized_worktree_oid != source_oid or modified >= mtime_cutoff:
            raise SctsrError(
                ErrorCode.SOURCE_TREE_MISMATCH,
                "Protected legacy Git identity, normalized worktree content, or mtime changed during v4 implementation",
                observed={
                    "path": relative,
                    "source_git_blob_oid": source_oid,
                    "worktree_normalized_git_blob_oid": normalized_worktree_oid,
                    "worktree_sha256": worktree_sha,
                    "mtime_utc": modified.isoformat(),
                },
                expected={
                    "baseline_git_blob_oid": baseline_oid,
                    "baseline_blob_sha256": baseline_sha,
                    "mtime_before_utc": mtime_cutoff.isoformat(),
                },
            )
        legacy_files.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "worktree_sha256": worktree_sha,
                "worktree_normalized_git_blob_oid": normalized_worktree_oid,
                "baseline_git_blob_oid": baseline_oid,
                "source_git_blob_oid": source_oid,
                "baseline_blob_bytes": len(baseline_bytes),
                "source_blob_bytes": len(source_bytes),
                "baseline_blob_sha256": baseline_sha,
                "source_blob_sha256": source_sha,
                "worktree_representation_matches_source_blob_bytes": worktree_sha == source_sha,
                "mtime_utc": modified.isoformat(),
                "implementation_start_utc": start_time.isoformat(),
                "implementation_start_mtime_cutoff_utc": mtime_cutoff.isoformat(),
                "unchanged": True,
            }
        )
    legacy_core = {
        "schema_version": "stage1.sctsr.legacy_evidence_audit.v1",
        "file_count": len(legacy_files),
        "files": legacy_files,
    }
    legacy_evidence = {**legacy_core, "audit_digest": stable_digest(legacy_core)}

    experiment_relative = "artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807"
    experiment_root = root / experiment_relative
    ignored_output = str(
        _git(
            root,
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--",
            experiment_relative,
        )
    )
    ignored = [line.replace("\\", "/") for line in ignored_output.splitlines() if line]
    all_known_paths = sorted(set([*normalized_tracked_paths, *untracked, *ignored]))
    formal_manifests: list[str] = []
    for relative in all_known_paths:
        if not relative.startswith(f"{experiment_relative}/") or not relative.endswith("/RUN_MANIFEST.json"):
            continue
        manifest_path = root / relative
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if payload.get("schema_version") == "stage1.sctsr.formal_run_manifest.v1" or payload.get("formal_training_started") is True:
            formal_manifests.append(relative)
    active_side_effect_paths = [
        path
        for path in all_known_paths
        if "sctsr_v4" in path.lower()
        and any(token in Path(path).name.lower() for token in ("assignment", "engineering_gate", "pilot_release"))
        and not path.startswith(("stage1_sctsr_v4/", "scripts/", "tests/", "docs/", "configs/"))
    ]
    if formal_manifests or active_side_effect_paths:
        raise SctsrError(
            ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED,
            "Repository contains an active SCTSR formal run or orchestration side effect",
            observed={"formal_manifests": formal_manifests, "active_side_effect_paths": active_side_effect_paths},
        )
    old_root = experiment_root
    legacy_detected = {
        "legacy_engineering_gate_detected": (old_root / "04_run_queue_v2/JOB_EXECUTION_REGISTRY.csv").is_file(),
        "legacy_pilot_release_detected": any((old_root / "04_run_queue/releases").glob("*")) if (old_root / "04_run_queue/releases").is_dir() else False,
        "legacy_assignments_detected": any((old_root / "04_run_queue_v2/machines").glob("*.csv")) if (old_root / "04_run_queue_v2/machines").is_dir() else False,
    }
    side_effects = {
        "formal_training_started": False,
        "engineering_gate_generated": False,
        "assignments_generated": False,
        "pilot_release_generated": False,
        "blind_holdout_opened": False,
        "selector_trained": False,
        "method_effectiveness_claimed": False,
        "val_target_available": False,
        "synthetic_registered_as_scientific": False,
    }
    report_core = {
        "schema_version": "stage1.sctsr.repository_state_audit.v1",
        "status": "PASS",
        "repository_root": root.as_posix(),
        "baseline_commit": baseline_commit,
        "implementation_start_commit": implementation_start_commit,
        "implementation_source_commit": implementation_source_commit,
        "tracked_worktree_clean": True,
        "untracked_file_count": len(untracked),
        "untracked_paths_not_staged": untracked,
        "protected_prefixes": list(protected_prefixes),
        "protected_changes": [],
        "changed_file_ledger": changed_file_ledger,
        "legacy_evidence": legacy_evidence,
        "legacy_detected": legacy_detected,
        "side_effects": side_effects,
    }
    return {**report_core, "audit_digest": stable_digest(report_core)}
