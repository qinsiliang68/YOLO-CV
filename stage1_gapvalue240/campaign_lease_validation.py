"""Executable concurrency and fencing validators for shared campaign leases."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
import queue
import shutil
import threading
import time
from typing import Any
import uuid

from .campaign_lease import (
    LeaseLostError,
    activate_assignment,
    claim_job_lease,
)
from .errors import LockHeldError, ValidationError
from .util import atomic_write_json, environment_snapshot


CONCURRENCY_SCHEMA = "stage1.lease_concurrency_validation.v1"
FENCING_SCHEMA = "stage1.lease_fencing_validation.v1"


@dataclass(frozen=True)
class _Identity:
    campaign_id: str
    release_id: str
    assignment_id: str
    assignment_sha256: str
    job_id: str


def _identity(prefix: str) -> _Identity:
    return _Identity(
        campaign_id=f"{prefix}_CAMPAIGN",
        release_id=f"{prefix}_RELEASE",
        assignment_id=f"{prefix}_ASSIGNMENT",
        assignment_sha256="A" * 64,
        job_id=f"{prefix}_JOB",
    )


def _activate(root: Path, identity: _Identity) -> None:
    activate_assignment(
        root,
        campaign_id=identity.campaign_id,
        release_id=identity.release_id,
        assignment_id=identity.assignment_id,
        assignment_sha256=identity.assignment_sha256,
        job_ids=(identity.job_id,),
    )


def _claim(root: Path, identity: _Identity, machine_id: str):
    return claim_job_lease(
        root,
        campaign_id=identity.campaign_id,
        release_id=identity.release_id,
        assignment_id=identity.assignment_id,
        assignment_sha256=identity.assignment_sha256,
        job_id=identity.job_id,
        machine_id=machine_id,
        ttl_seconds=30.0,
        heartbeat_seconds=5.0,
    )


def _thread_round(root: Path, identity: _Identity, claimants: int) -> dict[str, Any]:
    barrier = threading.Barrier(claimants)
    winners = []
    losers: list[str] = []
    unexpected: list[str] = []
    guard = threading.Lock()

    def attempt(index: int) -> None:
        try:
            barrier.wait(timeout=10)
            lease = _claim(root, identity, f"thread_{index:03d}")
        except LockHeldError as exc:
            with guard:
                losers.append(f"{type(exc).__name__}: {exc}")
        except BaseException as exc:  # validator records every leaked exception
            with guard:
                unexpected.append(f"{type(exc).__name__}: {exc}")
        else:
            with guard:
                winners.append(lease)

    threads = [threading.Thread(target=attempt, args=(index,)) for index in range(claimants)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    alive = [thread.name for thread in threads if thread.is_alive()]
    if len(winners) == 1:
        winners[0].release(status="VALIDATION_COMPLETE")
    return {
        "winner_count": len(winners),
        "loser_count": len(losers),
        "unexpected_errors": unexpected,
        "alive_threads": alive,
    }


def _process_attempt(
    root: str,
    identity_payload: dict[str, str],
    start_event,
    release_event,
    result_queue,
    index: int,
) -> None:
    identity = _Identity(**identity_payload)
    start_event.wait()
    try:
        lease = _claim(Path(root), identity, f"process_{index:03d}")
    except LockHeldError as exc:
        result_queue.put({"status": "LOSER", "error": f"{type(exc).__name__}: {exc}"})
    except BaseException as exc:
        result_queue.put({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
    else:
        result_queue.put({"status": "WINNER", "token": lease.token})
        release_event.wait(timeout=30)
        lease.release(status="VALIDATION_COMPLETE")


def _process_race(root: Path, identity: _Identity, claimants: int) -> dict[str, Any]:
    if claimants <= 0:
        return {
            "claimants": 0,
            "winner_count": 0,
            "loser_count": 0,
            "unexpected_errors": [],
            "exitcodes": [],
            "status": "NOT_RUN",
        }
    context = mp.get_context("spawn")
    start_event = context.Event()
    release_event = context.Event()
    result_queue = context.Queue()
    payload = {
        "campaign_id": identity.campaign_id,
        "release_id": identity.release_id,
        "assignment_id": identity.assignment_id,
        "assignment_sha256": identity.assignment_sha256,
        "job_id": identity.job_id,
    }
    processes = [
        context.Process(
            target=_process_attempt,
            args=(str(root), payload, start_event, release_event, result_queue, index),
        )
        for index in range(claimants)
    ]
    for process in processes:
        process.start()
    start_event.set()
    results = []
    try:
        for _ in processes:
            results.append(result_queue.get(timeout=60))
    except queue.Empty:
        results.append({"status": "ERROR", "error": "result queue timeout"})
    finally:
        release_event.set()
    for process in processes:
        process.join(timeout=60)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
    winners = [result for result in results if result.get("status") == "WINNER"]
    losers = [result for result in results if result.get("status") == "LOSER"]
    errors = [result for result in results if result.get("status") == "ERROR"]
    exitcodes = [process.exitcode for process in processes]
    return {
        "claimants": claimants,
        "winner_count": len(winners),
        "loser_count": len(losers),
        "unexpected_errors": errors,
        "exitcodes": exitcodes,
        "status": "PASS"
        if len(winners) == 1
        and len(losers) == claimants - 1
        and not errors
        and all(code == 0 for code in exitcodes)
        else "FAIL",
    }


def run_lease_concurrency_validation(
    coordination_root: str | Path,
    output_path: str | Path,
    *,
    thread_rounds: int = 100,
    thread_claimants: int = 8,
    process_claimants: int = 32,
) -> dict[str, Any]:
    """Run bounded real thread/process races and atomically publish the report."""

    if thread_rounds < 1 or thread_claimants < 2 or process_claimants < 0:
        raise ValidationError("invalid lease concurrency validation dimensions")
    base = Path(coordination_root).resolve()
    validation_root = base / f"validation_{uuid.uuid4().hex}"
    validation_root.mkdir(parents=True)
    identity = _identity("THREAD")
    _activate(validation_root, identity)
    thread_results: list[dict[str, Any]] = []
    for _round in range(thread_rounds):
        thread_results.append(_thread_round(validation_root, identity, thread_claimants))
    process_root = base / f"process_validation_{uuid.uuid4().hex}"
    process_root.mkdir(parents=True)
    process_identity = _identity("PROCESS")
    _activate(process_root, process_identity)
    process_result = _process_race(process_root, process_identity, process_claimants)
    unexpected = [
        error
        for result in thread_results
        for error in result["unexpected_errors"]
    ]
    alive = [name for result in thread_results for name in result["alive_threads"]]
    thread_pass = all(
        result["winner_count"] == 1
        and result["loser_count"] == thread_claimants - 1
        and not result["unexpected_errors"]
        and not result["alive_threads"]
        for result in thread_results
    )
    residual_control = list(base.rglob("control.lock"))
    report = {
        "schema_version": CONCURRENCY_SCHEMA,
        "status": "PASS"
        if thread_pass
        and (process_claimants == 0 or process_result["status"] == "PASS")
        and not residual_control
        else "FAIL",
        "created_at_unix": time.time(),
        "environment": environment_snapshot(),
        "thread_race": {
            "rounds": thread_rounds,
            "claimants_per_round": thread_claimants,
            "winner_count_total": sum(result["winner_count"] for result in thread_results),
            "loser_count_total": sum(result["loser_count"] for result in thread_results),
            "unexpected_error_count": len(unexpected),
            "unexpected_errors": unexpected,
            "alive_thread_count": len(alive),
        },
        "process_race": process_result,
        "residual_control_lock_count": len(residual_control),
        "residual_control_locks": [str(path) for path in residual_control],
    }
    atomic_write_json(output_path, report, overwrite=True)
    if report["status"] != "PASS":
        raise ValidationError(f"lease concurrency validation failed: {output_path}")
    return report


def run_fencing_validation(
    coordination_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Exercise lease expiry, assignment supersession, and stale-holder fencing."""

    root = Path(coordination_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    campaign = "FENCE_CAMPAIGN"
    release = "FENCE_RELEASE"
    job = "FENCE_JOB"
    assignment_a = "FENCE_A"
    assignment_b = "FENCE_B"
    sha_a = "A" * 64
    sha_b = "B" * 64
    activate_assignment(
        root,
        campaign_id=campaign,
        release_id=release,
        assignment_id=assignment_a,
        assignment_sha256=sha_a,
        job_ids=(job,),
    )
    old = claim_job_lease(
        root,
        campaign_id=campaign,
        release_id=release,
        assignment_id=assignment_a,
        assignment_sha256=sha_a,
        job_id=job,
        machine_id="machine_old",
        ttl_seconds=0.2,
        heartbeat_seconds=0.1,
    )
    # Force the heartbeat stale without relying on wall-clock sleeps.
    heartbeat = __import__("json").loads(old.heartbeat_path.read_text(encoding="utf-8"))
    heartbeat["heartbeat_at_unix"] = time.time() - 10
    old.heartbeat_path.write_text(__import__("json").dumps(heartbeat), encoding="utf-8")
    activate_assignment(
        root,
        campaign_id=campaign,
        release_id=release,
        assignment_id=assignment_b,
        assignment_sha256=sha_b,
        job_ids=(job,),
        expected_previous_assignment_sha256=sha_a,
    )
    new = claim_job_lease(
        root,
        campaign_id=campaign,
        release_id=release,
        assignment_id=assignment_b,
        assignment_sha256=sha_b,
        job_id=job,
        machine_id="machine_new",
        ttl_seconds=30,
        heartbeat_seconds=5,
    )
    heartbeat_status = "NOT_FENCED"
    publish_status = "NOT_FENCED"
    try:
        old.heartbeat()
    except LeaseLostError:
        heartbeat_status = "FENCED"
    try:
        old.check_now()  # the same guard used immediately before result publication
    except LeaseLostError:
        publish_status = "FENCED"
    new.check_now()
    new.release(status="VALIDATION_COMPLETE")
    old.release(status="FENCED")
    report = {
        "schema_version": FENCING_SCHEMA,
        "status": "PASS"
        if heartbeat_status == "FENCED" and publish_status == "FENCED"
        else "FAIL",
        "created_at_unix": time.time(),
        "old_assignment_sha256": sha_a,
        "new_assignment_sha256": sha_b,
        "old_holder_heartbeat": heartbeat_status,
        "old_holder_publish": publish_status,
        "new_holder_claimed": True,
        "environment": environment_snapshot(),
    }
    atomic_write_json(output_path, report, overwrite=True)
    if report["status"] != "PASS":
        raise ValidationError(f"lease fencing validation failed: {output_path}")
    return report


__all__ = [
    "CONCURRENCY_SCHEMA",
    "FENCING_SCHEMA",
    "run_fencing_validation",
    "run_lease_concurrency_validation",
]
