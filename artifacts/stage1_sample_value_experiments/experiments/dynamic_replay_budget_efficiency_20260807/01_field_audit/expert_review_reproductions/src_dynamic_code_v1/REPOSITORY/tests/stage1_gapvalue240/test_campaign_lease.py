from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import pytest

from stage1_gapvalue240.campaign_lease import (
    AssignmentInactiveError,
    LeaseLostError,
    activate_assignment,
    claim_job_lease,
)
from stage1_gapvalue240.errors import LockHeldError


CAMPAIGN = "CAMPAIGN_X"
RELEASE = "RELEASE_X"


def _activate(root: Path, *, assignment: str = "A1", sha: str = "A" * 64, previous=None):
    return activate_assignment(
        root,
        campaign_id=CAMPAIGN,
        release_id=RELEASE,
        assignment_id=assignment,
        assignment_sha256=sha,
        job_ids=("JOB_1", "JOB_2"),
        expected_previous_assignment_sha256=previous,
    )


def test_job_lease_is_exclusive_heartbeated_and_releasable(tmp_path: Path) -> None:
    _activate(tmp_path)
    lease = claim_job_lease(
        tmp_path,
        campaign_id=CAMPAIGN,
        release_id=RELEASE,
        assignment_id="A1",
        assignment_sha256="A" * 64,
        job_id="JOB_1",
        machine_id="machine_01",
        ttl_seconds=2.0,
        heartbeat_seconds=0.05,
    )
    first = json.loads(lease.heartbeat_path.read_text(encoding="utf-8"))["heartbeat_at_unix"]
    lease.start_heartbeat()
    time.sleep(0.12)
    second = json.loads(lease.heartbeat_path.read_text(encoding="utf-8"))["heartbeat_at_unix"]
    assert second > first

    with pytest.raises(LockHeldError, match="active job lease"):
        claim_job_lease(
            tmp_path,
            campaign_id=CAMPAIGN,
            release_id=RELEASE,
            assignment_id="A1",
            assignment_sha256="A" * 64,
            job_id="JOB_1",
            machine_id="machine_02",
            ttl_seconds=2.0,
            heartbeat_seconds=0.05,
        )
    lease.release(status="COMPLETE")

    replacement = claim_job_lease(
        tmp_path,
        campaign_id=CAMPAIGN,
        release_id=RELEASE,
        assignment_id="A1",
        assignment_sha256="A" * 64,
        job_id="JOB_1",
        machine_id="machine_02",
        ttl_seconds=2.0,
        heartbeat_seconds=0.05,
    )
    replacement.release(status="TEST_COMPLETE")


def test_superseding_assignment_refuses_live_claim_and_fences_stale_owner(tmp_path: Path) -> None:
    _activate(tmp_path)
    lease = claim_job_lease(
        tmp_path,
        campaign_id=CAMPAIGN,
        release_id=RELEASE,
        assignment_id="A1",
        assignment_sha256="A" * 64,
        job_id="JOB_1",
        machine_id="machine_01",
        ttl_seconds=1.0,
        heartbeat_seconds=0.2,
    )
    with pytest.raises(LockHeldError, match="active claims"):
        _activate(tmp_path, assignment="A2", sha="B" * 64, previous="A" * 64)

    heartbeat = json.loads(lease.heartbeat_path.read_text(encoding="utf-8"))
    heartbeat["heartbeat_at_unix"] = time.time() - 10
    lease.heartbeat_path.write_text(json.dumps(heartbeat), encoding="utf-8")
    activation = _activate(tmp_path, assignment="A2", sha="B" * 64, previous="A" * 64)
    assert activation.assignment_id == "A2"
    with pytest.raises(LeaseLostError, match="active assignment changed"):
        lease.check_now()
    lease.release(status="FENCED")

    with pytest.raises(AssignmentInactiveError, match="not active"):
        claim_job_lease(
            tmp_path,
            campaign_id=CAMPAIGN,
            release_id=RELEASE,
            assignment_id="A1",
            assignment_sha256="A" * 64,
            job_id="JOB_2",
            machine_id="machine_01",
            ttl_seconds=1.0,
            heartbeat_seconds=0.2,
        )


def test_atomic_claim_has_one_winner_under_concurrency(tmp_path: Path) -> None:
    _activate(tmp_path)
    barrier = threading.Barrier(8)
    winners = []
    errors = []
    lock = threading.Lock()

    def attempt(index: int) -> None:
        barrier.wait()
        try:
            lease = claim_job_lease(
                tmp_path,
                campaign_id=CAMPAIGN,
                release_id=RELEASE,
                assignment_id="A1",
                assignment_sha256="A" * 64,
                job_id="JOB_1",
                machine_id=f"machine_{index:02d}",
                ttl_seconds=5.0,
                heartbeat_seconds=1.0,
            )
        except LockHeldError as exc:
            with lock:
                errors.append(exc)
        else:
            with lock:
                winners.append(lease)

    threads = [threading.Thread(target=attempt, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(winners) == 1
    assert len(errors) == 7
    winners[0].release(status="TEST_COMPLETE")
