from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
import threading

import pytest

import stage1_gapvalue240.campaign_lease as lease_module
from stage1_gapvalue240.campaign_lease import (
    _ControlLock,
    activate_assignment,
    claim_job_lease,
)
from stage1_gapvalue240.campaign_lease_validation import (
    run_fencing_validation,
    run_lease_concurrency_validation,
)
from stage1_gapvalue240.errors import LockHeldError


CAMPAIGN = "CAMPAIGN_STRESS"
RELEASE = "RELEASE_STRESS"
ASSIGNMENT = "ASSIGNMENT_STRESS"
ASSIGNMENT_SHA = "A" * 64
JOB = "JOB_STRESS"


class _SharingViolation(PermissionError):
    winerror = 32


def _process_claim(root: str, start, results) -> None:
    start.wait()
    try:
        claimed = claim_job_lease(
            root,
            campaign_id=CAMPAIGN,
            release_id=RELEASE,
            assignment_id=ASSIGNMENT,
            assignment_sha256=ASSIGNMENT_SHA,
            job_id=JOB,
            machine_id=f"machine_{__import__('os').getpid()}",
            ttl_seconds=30,
            heartbeat_seconds=5,
        )
    except LockHeldError:
        results.put("LOSER")
    except BaseException as exc:  # surfaced in parent assertion
        results.put(f"ERROR:{type(exc).__name__}:{exc}")
    else:
        results.put(f"WINNER:{claimed.token}")


def _activate(root: Path) -> None:
    activate_assignment(
        root,
        campaign_id=CAMPAIGN,
        release_id=RELEASE,
        assignment_id=ASSIGNMENT,
        assignment_sha256=ASSIGNMENT_SHA,
        job_ids=(JOB,),
    )


def test_control_lock_retries_windows_sharing_violation(monkeypatch, tmp_path: Path) -> None:
    calls = {"count": 0}
    original = lease_module._exclusive_json

    def flaky(path, payload):
        calls["count"] += 1
        if calls["count"] == 1:
            raise _SharingViolation(13, "sharing violation")
        return original(path, payload)

    monkeypatch.setattr(lease_module, "_exclusive_json", flaky)
    with _ControlLock(tmp_path, timeout_seconds=0.5, retry_interval_seconds=0.001):
        assert (tmp_path / "control.lock").is_file()
    assert calls["count"] >= 2
    assert not (tmp_path / "control.lock").exists()


def test_control_lock_does_not_hide_real_permission_error(monkeypatch, tmp_path: Path) -> None:
    denied = PermissionError(13, "real ACL failure")

    def fail(_path, _payload):
        raise denied

    monkeypatch.setattr(lease_module, "_exclusive_json", fail)
    monkeypatch.setattr(lease_module, "_is_retryable_lock_contention", lambda *_: False)
    with pytest.raises(PermissionError, match="real ACL failure"):
        with _ControlLock(tmp_path, timeout_seconds=0.01):
            pass


def test_thread_race_100_rounds_has_exactly_one_winner(tmp_path: Path) -> None:
    report = run_lease_concurrency_validation(
        tmp_path / "coordination",
        tmp_path / "LEASE_CONCURRENCY_VALIDATION.json",
        thread_rounds=100,
        thread_claimants=8,
        process_claimants=0,
    )
    assert report["status"] == "PASS"
    assert report["thread_race"]["rounds"] == 100
    assert report["thread_race"]["winner_count_total"] == 100
    assert report["thread_race"]["unexpected_error_count"] == 0
    assert report["residual_control_lock_count"] == 0


def test_32_process_race_has_exactly_one_winner(tmp_path: Path) -> None:
    root = tmp_path / "coordination"
    _activate(root)
    ctx = mp.get_context("spawn")
    start = ctx.Event()
    results = ctx.Queue()
    processes = [ctx.Process(target=_process_claim, args=(str(root), start, results)) for _ in range(32)]
    for process in processes:
        process.start()
    start.set()
    observed = [results.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    winners = [value for value in observed if value.startswith("WINNER:")]
    losers = [value for value in observed if value == "LOSER"]
    errors = [value for value in observed if value.startswith("ERROR:")]
    assert len(winners) == 1
    assert len(losers) == 31
    assert errors == []


def test_two_generation_fencing_validation(tmp_path: Path) -> None:
    report = run_fencing_validation(
        tmp_path / "coordination",
        tmp_path / "LEASE_FENCING_VALIDATION.json",
    )
    assert report["status"] == "PASS"
    assert report["old_holder_heartbeat"] == "FENCED"
    assert report["old_holder_publish"] == "FENCED"
    assert json.loads((tmp_path / "LEASE_FENCING_VALIDATION.json").read_text())["status"] == "PASS"


def test_lease_validation_cli_uses_public_dimension_names(tmp_path: Path) -> None:
    from scripts.stage1_gapvalue240.validate_campaign_leases import main
    code = main([
        "--coordination-root", str(tmp_path / "coord"),
        "--concurrency-output", str(tmp_path / "concurrency.json"),
        "--fencing-output", str(tmp_path / "fencing.json"),
        "--thread-rounds", "2",
        "--thread-contenders", "4",
        "--process-contenders", "0",
    ])
    assert code == 0
    assert (tmp_path / "concurrency.json").is_file()
    assert (tmp_path / "fencing.json").is_file()
