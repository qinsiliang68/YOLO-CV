from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from stage1_gapvalue240.campaign_controller import (
    CampaignControllerError,
    build_campaign_release_manifests,
    load_campaign_release,
    plan_controller_iteration,
    run_worker_process,
)
from stage1_gapvalue240.util import atomic_write_json, sha256_file
from stage1_gapvalue240.campaign_engineering_gate import (
    REQUIRED_EVIDENCE_SCHEMAS,
    ValidationIdentity,
    bind_validation_evidence,
    build_engineering_gate_v2,
)


def _queue(tmp_path: Path) -> Path:
    queue = tmp_path / "04_run_queue"
    queue.mkdir()
    rows = [
        {
            "queue_order": 1,
            "job_id": "JOB_S001_A",
            "machine_id": "machine_01",
            "seed_id": "S001",
            "dependency_job_id": "",
            "cycle_id": "CYCLE_1",
            "release_state": "ENGINEERING_GATE",
        },
        {
            "queue_order": 2,
            "job_id": "JOB_S001_B",
            "machine_id": "machine_01",
            "seed_id": "S001",
            "dependency_job_id": "JOB_S001_A",
            "cycle_id": "CYCLE_1",
            "release_state": "ENGINEERING_GATE",
        },
        {
            "queue_order": 3,
            "job_id": "JOB_S002_C",
            "machine_id": "machine_02",
            "seed_id": "S002",
            "dependency_job_id": "",
            "cycle_id": "CYCLE_1",
            "release_state": "ENGINEERING_GATE",
        },
        {
            "queue_order": 4,
            "job_id": "JOB_S003_D",
            "machine_id": "machine_01",
            "seed_id": "S003",
            "dependency_job_id": "",
            "cycle_id": "CYCLE_1",
            "release_state": "ENGINEERING_GATE",
        },
        {
            "queue_order": 5,
            "job_id": "JOB_S001_CYCLE2",
            "machine_id": "machine_01",
            "seed_id": "S001",
            "dependency_job_id": "",
            "cycle_id": "CYCLE_2",
            "release_state": "HELD",
        },
    ]
    registry = queue / "JOB_EXECUTION_REGISTRY.csv"
    pd.DataFrame(rows).to_csv(registry, index=False)
    (queue / "RUN_QUEUE_VALIDATION.json").write_text(
        json.dumps(
            {
                "schema_version": "stage1.dynamic_campaign_run_queue.v2",
                "status": "PASS",
                "job_count": len(rows),
                "job_registry_sha256": sha256_file(registry),
                "canonical_lock_file_sha256": "A" * 64,
            }
        ),
        encoding="utf-8",
    )
    return queue


def _engineering_gate(queue: Path, path: Path, *, status: str = "PASS") -> Path:
    validation = json.loads((queue / "RUN_QUEUE_VALIDATION.json").read_text(encoding="utf-8"))
    root = path.parent / f"{path.stem}_evidence"
    raw_root = root / "raw"
    envelope_root = root / "envelopes"
    raw_root.mkdir(parents=True)
    envelope_root.mkdir(parents=True)
    identity = ValidationIdentity(
        source_tree_sha256="D" * 64,
        queue_registry_sha256=validation["job_registry_sha256"],
        canonical_lock_file_sha256=validation["canonical_lock_file_sha256"],
    )
    evidence = {}
    for key, schema in REQUIRED_EVIDENCE_SCHEMAS.items():
        raw = raw_root / f"{key}.json"
        atomic_write_json(raw, {"schema_version": schema, "status": "PASS", "check": key})
        envelope = envelope_root / f"{key}.json"
        bind_validation_evidence(
            raw, evidence_type=key, expected_schema=schema, identity=identity,
            output_path=envelope, allowed_root=root,
        )
        evidence[key] = envelope
    gate = root / path.name
    build_engineering_gate_v2(
        evidence, expected_identity=identity, allowed_root=root, output_path=gate,
    )
    if status != "PASS":
        payload = json.loads(gate.read_text(encoding="utf-8"))
        payload["status"] = status
        atomic_write_json(gate, payload, overwrite=True)
    return gate


def test_release_manifests_freeze_pilot_and_hold_confirmatory(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    gate = _engineering_gate(queue, tmp_path / "engineering_gate.json")
    releases = build_campaign_release_manifests(
        queue,
        tmp_path / "releases",
        campaign_id="CAMPAIGN_X",
        pilot_seed_ids=("S001", "S002"),
        engineering_gate_report=gate,
    )

    pilot = json.loads(releases.pilot_release.read_text(encoding="utf-8"))
    confirmatory = json.loads(releases.confirmatory_hold.read_text(encoding="utf-8"))
    future = json.loads(releases.future_cycle_hold.read_text(encoding="utf-8"))
    assert pilot["release_status"] == "RELEASED"
    assert pilot["job_ids"] == ["JOB_S001_A", "JOB_S001_B", "JOB_S002_C"]
    assert confirmatory["release_status"] == "HOLD"
    assert confirmatory["job_ids"] == ["JOB_S003_D"]
    assert future["release_status"] == "HOLD"
    assert future["job_ids"] == ["JOB_S001_CYCLE2"]
    assert pilot["cycle_ids"] == ["CYCLE_1"]
    assert pilot["engineering_gate_report_sha256"] == sha256_file(gate)
    assert pilot["schema_version"] == "stage1.dynamic_campaign_release.v2"

    loaded = load_campaign_release(
        queue,
        releases.pilot_release,
        expected_campaign_id="CAMPAIGN_X",
    )
    assert loaded.job_ids == tuple(pilot["job_ids"])
    with pytest.raises(CampaignControllerError, match="not RELEASED"):
        load_campaign_release(
            queue,
            releases.confirmatory_hold,
            expected_campaign_id="CAMPAIGN_X",
        )


def test_release_loader_rejects_historical_v1_release(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    historical = tmp_path / "historical_release_v1.json"
    historical.write_text(
        json.dumps(
            {
                "schema_version": "stage1.dynamic_campaign_release.v1",
                "campaign_id": "CAMPAIGN_X",
                "release_id": "PILOT_V1",
                "release_status": "RELEASED",
                "job_ids": ["JOB_S001_A"],
                "job_count": 1,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CampaignControllerError, match="v2 required"):
        load_campaign_release(queue, historical, expected_campaign_id="CAMPAIGN_X")


def test_release_rejects_checksum_drift_and_missing_dependency(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    gate = _engineering_gate(queue, tmp_path / "engineering_gate.json")
    releases = build_campaign_release_manifests(
        queue,
        tmp_path / "releases",
        campaign_id="CAMPAIGN_X",
        pilot_seed_ids=("S001", "S002"),
        engineering_gate_report=gate,
    )
    payload = json.loads(releases.pilot_release.read_text(encoding="utf-8"))
    payload["job_ids"] = ["JOB_S001_B"]
    payload["job_count"] = 1
    broken = releases.pilot_release.parent / "broken.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CampaignControllerError, match="dependency closure"):
        load_campaign_release(queue, broken, expected_campaign_id="CAMPAIGN_X")

    registry = queue / "JOB_EXECUTION_REGISTRY.csv"
    registry.write_text(registry.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(CampaignControllerError, match="checksum"):
        load_campaign_release(
            queue,
            releases.pilot_release,
            expected_campaign_id="CAMPAIGN_X",
        )


def test_release_requires_a_complete_engineering_gate_and_never_leaks_cycle_two(
    tmp_path: Path,
) -> None:
    queue = _queue(tmp_path)
    with pytest.raises(CampaignControllerError, match="engineering gate"):
        build_campaign_release_manifests(
            queue,
            tmp_path / "missing_gate",
            campaign_id="CAMPAIGN_X",
            pilot_seed_ids=("S001",),
        )

    bad = _engineering_gate(queue, tmp_path / "bad_gate.json", status="FAILED")
    with pytest.raises(CampaignControllerError, match="engineering gate"):
        build_campaign_release_manifests(
            queue,
            tmp_path / "bad_release",
            campaign_id="CAMPAIGN_X",
            pilot_seed_ids=("S001",),
            engineering_gate_report=bad,
        )


def test_planner_continues_independent_work_after_terminal_failure() -> None:
    jobs = pd.DataFrame(
        [
            {"queue_order": 1, "job_id": "A", "dependency_job_id": ""},
            {"queue_order": 2, "job_id": "B", "dependency_job_id": "A"},
            {"queue_order": 3, "job_id": "C", "dependency_job_id": ""},
        ]
    )
    states = {
        "A": {
            "state": "FAILED",
            "retryable": False,
            "attempt_count": 1,
            "updated_at_unix": 10.0,
        }
    }

    plan = plan_controller_iteration(
        jobs,
        states,
        now_unix=100.0,
        max_attempts=3,
        retry_delay_seconds=30.0,
    )

    assert plan.next_job_id == "C"
    assert plan.job_states["A"] == "FAILED_TERMINAL"
    assert plan.job_states["B"] == "BLOCKED_DEPENDENCY"
    assert plan.job_states["C"] == "READY"


def test_planner_retries_only_after_delay_and_stops_at_attempt_limit() -> None:
    jobs = pd.DataFrame([{"queue_order": 1, "job_id": "A", "dependency_job_id": ""}])
    retryable = {
        "A": {
            "state": "FAILED",
            "retryable": True,
            "attempt_count": 2,
            "updated_at_unix": 90.0,
        }
    }
    waiting = plan_controller_iteration(
        jobs,
        retryable,
        now_unix=100.0,
        max_attempts=3,
        retry_delay_seconds=30.0,
    )
    assert waiting.next_job_id is None
    assert waiting.job_states["A"] == "RETRY_WAIT"

    ready = plan_controller_iteration(
        jobs,
        retryable,
        now_unix=121.0,
        max_attempts=3,
        retry_delay_seconds=30.0,
    )
    assert ready.next_job_id == "A"
    assert ready.job_states["A"] == "RETRY_READY"

    retryable["A"]["attempt_count"] = 3
    exhausted = plan_controller_iteration(
        jobs,
        retryable,
        now_unix=121.0,
        max_attempts=3,
        retry_delay_seconds=30.0,
    )
    assert exhausted.next_job_id is None
    assert exhausted.job_states["A"] == "FAILED_TERMINAL"


def test_worker_process_writes_durable_log_result_and_heartbeats(tmp_path: Path) -> None:
    heartbeats: list[int] = []
    result = run_worker_process(
        [sys.executable, "-c", "import sys,time; print('worker-ok'); time.sleep(0.05); sys.exit(20)"],
        cwd=tmp_path,
        log_path=tmp_path / "worker.log",
        heartbeat=lambda pid: heartbeats.append(pid),
        poll_seconds=0.01,
    )

    assert result["returncode"] == 20
    assert result["status"] == "FAILED"
    assert "worker-ok" in (tmp_path / "worker.log").read_text(encoding="utf-8")
    assert (tmp_path / "worker.log.result.json").is_file()
    assert heartbeats and all(pid == result["pid"] for pid in heartbeats)


def test_long_lived_campaign_controller_does_not_import_training_runtime() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts/stage1_gapvalue240/run_dynamic_campaign_controller.py"
    code = (
        "import runpy,sys; "
        f"runpy.run_path({str(script)!r}, run_name='campaign_controller_import_test'); "
        "print(int('torch' in sys.modules), int('ultralytics' in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout
    assert result.stdout.strip().endswith("0 0"), result.stdout
