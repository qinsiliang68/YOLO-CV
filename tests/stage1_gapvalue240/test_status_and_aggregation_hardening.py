from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.aggregate import collect_validated
from stage1_gapvalue240.errors import ValidationError
from stage1_gapvalue240.locks import RunLock
from stage1_gapvalue240.status import read_status, set_status
from stage1_gapvalue240.validation import write_permanent_artifact_manifest


def test_status_json_is_authoritative_and_resume_is_recorded(tmp_path):
    attempt = tmp_path / "attempt_x.inprogress"
    set_status(attempt, "PLANNED", {"run_slot": "RUN_001"})
    set_status(attempt, "STAGED", {"phase": "prepare"})
    set_status(attempt, "RUNNING", {"pid": 123})
    set_status(attempt, "RECOVERING", {"resume_count": 1, "last_epoch": 17})
    set_status(attempt, "RUNNING", {"resume_count": 1, "last_epoch": 17})

    status = read_status(attempt)
    assert status["state"] == "RUNNING"
    assert status["resume_count"] == 1
    assert status["last_epoch"] == 17
    assert (attempt / "08_status/status.json").exists()
    assert [p.name for p in (attempt / "08_status").iterdir() if p.name.isupper()] == ["RUNNING"]


def test_dry_run_has_non_scientific_terminal_state(tmp_path):
    attempt = tmp_path / "attempt_dry.inprogress"
    set_status(attempt, "PLANNED")
    set_status(attempt, "STAGED")
    set_status(attempt, "RUNNING")
    set_status(attempt, "TRAIN_COMPLETED")
    set_status(attempt, "EVALUATED")
    set_status(attempt, "DRY_RUN_VALIDATED", {"dry_run": True})
    assert read_status(attempt)["state"] == "DRY_RUN_VALIDATED"


def test_retryable_train_failure_can_be_sealed_when_resume_checkpoint_is_corrupt(tmp_path):
    attempt = tmp_path / "attempt_corrupt.inprogress"
    set_status(attempt, "PLANNED")
    set_status(attempt, "STAGED")
    set_status(attempt, "RUNNING")
    set_status(attempt, "FAILED_TRAIN_RETRYABLE")
    set_status(attempt, "FAILED_TRAIN", {"error_code": "CORRUPT_RESUME_CHECKPOINT"})
    assert read_status(attempt)["state"] == "FAILED_TRAIN"


def test_run_lock_rejects_double_start_and_releases(tmp_path):
    path = tmp_path / "RUN_001.lock"
    first = RunLock(path, {"run_slot": "RUN_001"})
    first.acquire()
    with pytest.raises(ValidationError):
        RunLock(path, {"run_slot": "RUN_001"}).acquire()
    first.release()
    second = RunLock(path, {"run_slot": "RUN_001"})
    second.acquire()
    second.release()


def test_run_lock_can_reclaim_only_a_dead_local_owner(tmp_path):
    path = tmp_path / "RUN_001.lock"
    path.write_text(json.dumps({
        "pid": 2147483647,
        "hostname": socket.gethostname(),
        "created_at_unix": 1,
    }), encoding="utf-8")
    lock = RunLock(path, {"run_slot": "RUN_001"}, reclaim_dead_local=True)
    lock.acquire()
    stale = list(tmp_path.glob("RUN_001.lock.stale.*"))
    assert len(stale) == 1
    assert json.loads(stale[0].read_text(encoding="utf-8"))["pid"] == 2147483647
    lock.release()


def test_run_lock_never_reclaims_a_live_or_remote_owner(tmp_path):
    path = tmp_path / "RUN_001.lock"
    path.write_text(json.dumps({
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "created_at_unix": 1,
    }), encoding="utf-8")
    with pytest.raises(ValidationError, match="already exists"):
        RunLock(path, reclaim_dead_local=True).acquire()

    path.write_text(json.dumps({
        "pid": 2147483647,
        "hostname": "another-host",
        "created_at_unix": 1,
    }), encoding="utf-8")
    with pytest.raises(ValidationError, match="already exists"):
        RunLock(path, reclaim_dead_local=True).acquire()


def test_run_lock_reclaims_a_reused_local_pid(tmp_path):
    path = tmp_path / "RUN_001.lock"
    path.write_text(json.dumps({
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "process_create_time": 1.0,
        "created_at_unix": 1,
    }), encoding="utf-8")
    lock = RunLock(path, reclaim_dead_local=True)
    lock.acquire()
    assert list(tmp_path.glob("RUN_001.lock.stale.*"))
    lock.release()

def _write_attempt(root: Path, slot: str, attempt_id: str, *, state: str, dry_run: bool = False,
                   resume_count: int = 0, selection_sha: str = "SEL") -> Path:
    attempt = root / "runs" / slot / f"attempt_{attempt_id}"
    for rel in ("00_identity", "05_metrics", "07_validation", "08_status"):
        (attempt / rel).mkdir(parents=True, exist_ok=True)
    identity = {
        "run_slot": slot,
        "attempt_id": attempt_id,
        "dry_run": dry_run,
        "release_ref": "stage1-gapvalue240-runtime-v1.2.0",
        "runtime_contract_sha256": "RUNTIME",
        "science_contract_sha256": "SCIENCE",
        "matrix_sha256": "MATRIX",
        "selection_sha256": selection_sha,
        "input_snapshot_id": "INPUTS",
        "resume_mode": "native_approximate",
        "resume_count": resume_count,
    }
    (attempt / "00_identity/run_identity.json").write_text(json.dumps(identity), encoding="utf-8")
    metrics = {
        "TN_at_FN95": {"actual_TN": 10, "actual_FN": 1},
        "FN_at_TN68253": {"actual_TN": 9, "actual_FN": 2},
        "gap_q68_q050": 0.2,
        "tail_gap_q90_q05": 0.1,
    }
    (attempt / "05_metrics/operational_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (attempt / "08_status/status.json").write_text(json.dumps({"state": state}), encoding="utf-8")
    (attempt / f"08_status/{state}").write_text("{}", encoding="utf-8")
    (attempt / "07_validation/postflight_report.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    write_permanent_artifact_manifest(attempt, attempt / "07_validation/artifact_manifest.csv")
    return attempt


def test_multi_root_aggregation_excludes_dry_runs_and_inprogress(tmp_path):
    matrix = tmp_path / "matrix.csv"
    pd.DataFrame([
        {"run_slot": "RUN_001", "triad_id": "TRIAD_001", "arm": "T"},
        {"run_slot": "RUN_002", "triad_id": "TRIAD_001", "arm": "R1"},
        {"run_slot": "RUN_003", "triad_id": "TRIAD_001", "arm": "R2"},
    ]).to_csv(matrix, index=False)
    root_a, root_b = tmp_path / "machine_a", tmp_path / "machine_b"
    _write_attempt(root_a, "RUN_001", "a", state="VALIDATED", resume_count=2)
    _write_attempt(root_a, "RUN_002", "dry", state="DRY_RUN_VALIDATED", dry_run=True)
    partial = _write_attempt(root_b, "RUN_003", "partial", state="VALIDATED")
    partial.rename(partial.with_name(partial.name + ".inprogress"))

    result = collect_validated(
        [root_a, root_b], matrix,
        expected_identity={
            "release_ref": "stage1-gapvalue240-runtime-v1.2.0",
            "runtime_contract_sha256": "RUNTIME",
            "science_contract_sha256": "SCIENCE",
            "matrix_sha256": "MATRIX",
            "input_snapshot_id": "INPUTS",
        },
    )
    row = result[result.run_slot == "RUN_001"].iloc[0]
    assert row.TN_at_FN95 == 10
    assert row.resume_count == 2
    assert result[result.run_slot == "RUN_002"].TN_at_FN95.isna().all()
    assert result[result.run_slot == "RUN_003"].TN_at_FN95.isna().all()


def test_multi_root_aggregation_rejects_duplicate_active_results(tmp_path):
    matrix = tmp_path / "matrix.csv"
    pd.DataFrame([{"run_slot": "RUN_001", "triad_id": "TRIAD_001", "arm": "T"}]).to_csv(matrix, index=False)
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    _write_attempt(root_a, "RUN_001", "one", state="VALIDATED")
    _write_attempt(root_b, "RUN_001", "two", state="VALIDATED")
    with pytest.raises(RuntimeError, match="Multiple validated attempts"):
        collect_validated([root_a, root_b], matrix)
