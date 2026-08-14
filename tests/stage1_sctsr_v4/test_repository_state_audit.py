from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone

import pytest

from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.filesystem import windows_safe_resolved_path
from stage1_sctsr_v4.repository_state_audit import audit_repository_state


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _commit(root, message):
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", message], check=True)
    return _git(root, "rev-parse", "HEAD")


def _repository(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    _git(tmp_path, "config", "user.email", "audit@example.invalid")
    _git(tmp_path, "config", "user.name", "Audit Test")
    for path, content in {
        ".gitattributes": "* text=auto\n",
        "stage1_gapvalue240/frozen.py": "VALUE = 1\n",
        "stage1_dynamic_replay_v3/frozen.py": "VALUE = 1\n",
        "YOLOv11/ultralytics/frozen.py": "VALUE = 1\n",
        "artifacts/legacy/04_run_queue_v2/queue.json": "{}\n",
        "README.md": "base\n",
    }.items():
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    base = _commit(tmp_path, "base")
    (tmp_path / "stage1_sctsr_v4").mkdir()
    (tmp_path / "stage1_sctsr_v4" / "new.py").write_text("VALUE = 2\n", encoding="utf-8")
    source = _commit(tmp_path, "implementation")
    return base, source


def test_repository_audit_binds_allowed_changes_and_legacy_sha_mtime(tmp_path):
    base, source = _repository(tmp_path)
    old = datetime(2000, 1, 1, tzinfo=timezone.utc).timestamp()
    os.utime(tmp_path / "artifacts/legacy/04_run_queue_v2/queue.json", (old, old))
    report = audit_repository_state(
        tmp_path,
        baseline_commit=base,
        implementation_start_commit=source,
        implementation_source_commit=source,
        allowed_prefixes=("stage1_sctsr_v4/",),
        allowed_files=(),
        protected_prefixes=("stage1_gapvalue240/", "stage1_dynamic_replay_v3/", "YOLOv11/ultralytics/"),
        legacy_markers=("/04_run_queue_v2/",),
    )
    assert report["status"] == "PASS"
    assert report["changed_file_ledger"]["file_count"] == 1
    assert report["legacy_evidence"]["file_count"] == 1


def test_default_scope_allows_registered_v4_preregistration_evidence_only(tmp_path):
    base, implementation_start = _repository(tmp_path)
    _git(tmp_path, "config", "core.longpaths", "true")
    registered_relative = (
        "artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/"
        "03_preregistration_v4_sctsr/assets/DATASET_CONTENT_LEDGER_BUILD_RECEIPT_v1.json"
    )
    registered = windows_safe_resolved_path(tmp_path / registered_relative)
    registered.parent.mkdir(parents=True, exist_ok=True)
    registered.write_text('{"status":"PASS"}\n', encoding="utf-8")
    source = _commit(tmp_path, "registered v4 preregistration evidence")
    old = datetime(2000, 1, 1, tzinfo=timezone.utc).timestamp()
    os.utime(tmp_path / "artifacts/legacy/04_run_queue_v2/queue.json", (old, old))

    report = audit_repository_state(
        tmp_path,
        baseline_commit=base,
        implementation_start_commit=implementation_start,
        implementation_source_commit=source,
    )

    changed = {row["relative_path"] for row in report["changed_file_ledger"]["files"]}
    assert registered_relative in changed


def test_default_scope_rejects_unregistered_neighboring_experiment_artifact(tmp_path):
    base, implementation_start = _repository(tmp_path)
    _git(tmp_path, "config", "core.longpaths", "true")
    unregistered = windows_safe_resolved_path(
        tmp_path
        / "artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807"
        / "03_preregistration_v5_unapproved/claim.json"
    )
    unregistered.parent.mkdir(parents=True, exist_ok=True)
    unregistered.write_text('{"status":"PASS"}\n', encoding="utf-8")
    source = _commit(tmp_path, "unregistered neighboring artifact")

    with pytest.raises(SctsrError) as caught:
        audit_repository_state(
            tmp_path,
            baseline_commit=base,
            implementation_start_commit=implementation_start,
            implementation_source_commit=source,
        )

    assert caught.value.code is ErrorCode.SOURCE_TREE_MISMATCH


def test_repository_audit_uses_git_blob_identity_when_clean_worktree_line_endings_differ(tmp_path):
    base, source = _repository(tmp_path)
    relative = "artifacts/legacy/04_run_queue_v2/queue.json"
    path = tmp_path / relative
    blob_bytes = subprocess.check_output(["git", "-C", str(tmp_path), "show", f"{source}:{relative}"])
    path.write_bytes(blob_bytes)
    subprocess.run(["git", "-C", str(tmp_path), "add", "--", relative], check=True)
    assert _git(tmp_path, "status", "--porcelain=v1", "--untracked-files=no") == ""
    (tmp_path / "stage1_sctsr_v4/new.py").write_text("VALUE = 3\n", encoding="utf-8")
    source = _commit(tmp_path, "implementation after worktree representation")

    report = audit_repository_state(
        tmp_path,
        baseline_commit=base,
        implementation_start_commit=source,
        implementation_source_commit=source,
        allowed_prefixes=("stage1_sctsr_v4/",),
        allowed_files=(),
        protected_prefixes=("stage1_gapvalue240/", "stage1_dynamic_replay_v3/", "YOLOv11/ultralytics/"),
        legacy_markers=("/04_run_queue_v2/",),
    )

    evidence = report["legacy_evidence"]["files"][0]
    assert evidence["baseline_git_blob_oid"] == evidence["source_git_blob_oid"]
    assert evidence["baseline_blob_sha256"] == evidence["source_blob_sha256"]
    assert evidence["worktree_sha256"]


def test_repository_audit_does_not_treat_fresh_checkout_mtime_as_content_identity(tmp_path):
    base, source = _repository(tmp_path)
    relative = "artifacts/legacy/04_run_queue_v2/queue.json"
    path = tmp_path / relative
    future = datetime(2100, 1, 1, tzinfo=timezone.utc).timestamp()
    os.utime(path, (future, future))

    report = audit_repository_state(
        tmp_path,
        baseline_commit=base,
        implementation_start_commit=source,
        implementation_source_commit=source,
        allowed_prefixes=("stage1_sctsr_v4/",),
        allowed_files=(),
        protected_prefixes=("stage1_gapvalue240/", "stage1_dynamic_replay_v3/", "YOLOv11/ultralytics/"),
        legacy_markers=("/04_run_queue_v2/",),
    )

    evidence = report["legacy_evidence"]["files"][0]
    assert evidence["content_identity_unchanged"] is True
    assert evidence["mtime_before_implementation_start"] is False
    assert evidence["mtime_verification_status"] == "INFORMATIONAL_NOT_CONTENT_IDENTITY"


def test_repository_audit_rejects_protected_source_change(tmp_path):
    base, source = _repository(tmp_path)
    (tmp_path / "stage1_gapvalue240/frozen.py").write_text("VALUE = 9\n", encoding="utf-8")
    source = _commit(tmp_path, "forbidden")
    with pytest.raises(SctsrError) as caught:
        audit_repository_state(
            tmp_path,
            baseline_commit=base,
            implementation_start_commit=source,
            implementation_source_commit=source,
            allowed_prefixes=("stage1_sctsr_v4/",),
            allowed_files=(),
            protected_prefixes=("stage1_gapvalue240/",),
            legacy_markers=("/04_run_queue_v2/",),
        )
    assert caught.value.code is ErrorCode.SOURCE_TREE_MISMATCH


def test_repository_audit_rejects_untracked_importable_overlay_file(tmp_path):
    base, source = _repository(tmp_path)
    (tmp_path / "stage1_sctsr_v4/untracked.py").write_text("VALUE = 3\n", encoding="utf-8")
    with pytest.raises(SctsrError) as caught:
        audit_repository_state(
            tmp_path,
            baseline_commit=base,
            implementation_start_commit=source,
            implementation_source_commit=source,
            allowed_prefixes=("stage1_sctsr_v4/",),
            allowed_files=(),
            protected_prefixes=("stage1_gapvalue240/",),
            legacy_markers=("/04_run_queue_v2/",),
        )
    assert caught.value.code is ErrorCode.SOURCE_TREE_MISMATCH


def test_repository_audit_does_not_follow_broken_historical_directory_links(tmp_path):
    base, source = _repository(tmp_path)
    experiment = tmp_path / "artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807"
    experiment.mkdir(parents=True)
    target = tmp_path / "historical_external_target"
    target.mkdir()
    link = experiment / "historical_broken_link"
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    else:
        link.symlink_to(target, target_is_directory=True)
    target.rmdir()

    report = audit_repository_state(
        tmp_path,
        baseline_commit=base,
        implementation_start_commit=source,
        implementation_source_commit=source,
        allowed_prefixes=("stage1_sctsr_v4/",),
        allowed_files=(),
        protected_prefixes=("stage1_gapvalue240/", "stage1_dynamic_replay_v3/", "YOLOv11/ultralytics/"),
        legacy_markers=("/04_run_queue_v2/",),
    )
    assert report["status"] == "PASS"


@pytest.mark.skipif(os.name != "nt", reason="Win32 extended path contract")
def test_repository_audit_records_changed_file_beyond_max_path(tmp_path):
    base, _source = _repository(tmp_path)
    _git(tmp_path, "config", "core.longpaths", "true")
    relative = (
        "stage1_sctsr_v4/"
        + "registered_component_" + "a" * 70 + "/"
        + "evidence_partition_" + "b" * 70 + "/"
        + "artifact_identity_" + "c" * 50 + ".json"
    )
    destination = windows_safe_resolved_path(tmp_path / relative)
    destination.parent.mkdir(parents=True)
    destination.write_text('{"status":"PASS"}\n', encoding="utf-8")
    expected_bytes = len(destination.read_bytes())
    source = _commit(tmp_path, "implementation with long registered artifact")
    old = datetime(2000, 1, 1, tzinfo=timezone.utc).timestamp()
    os.utime(tmp_path / "artifacts/legacy/04_run_queue_v2/queue.json", (old, old))
    assert len(str((tmp_path / relative).resolve())) > 260

    report = audit_repository_state(
        tmp_path,
        baseline_commit=base,
        implementation_start_commit=source,
        implementation_source_commit=source,
        allowed_prefixes=("stage1_sctsr_v4/",),
        allowed_files=(),
        protected_prefixes=("stage1_gapvalue240/", "stage1_dynamic_replay_v3/", "YOLOv11/ultralytics/"),
        legacy_markers=("/04_run_queue_v2/",),
    )

    changed = {row["relative_path"]: row for row in report["changed_file_ledger"]["files"]}
    assert changed[relative]["bytes"] == expected_bytes
    assert len(changed[relative]["sha256"]) == 64
