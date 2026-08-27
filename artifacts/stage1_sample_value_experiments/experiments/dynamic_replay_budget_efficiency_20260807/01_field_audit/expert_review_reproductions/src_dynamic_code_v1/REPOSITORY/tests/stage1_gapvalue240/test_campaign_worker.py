from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.campaign_worker import (
    CampaignJob,
    CampaignWorkerError,
    archive_zero_epoch_attempt,
    load_campaign_job,
    resolve_segment_execution,
)
from stage1_gapvalue240.util import sha256_file


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "configs/stage1_gapvalue240/CANONICAL_TRAINING_LOCK_v1.json"


def _queue(tmp_path: Path) -> Path:
    root = tmp_path / "04_run_queue"
    template = root / "templates/TPL.csv"
    monitor = root / "monitor/CAUSAL_MONITOR_SAMPLES.csv"
    template.parent.mkdir(parents=True)
    monitor.parent.mkdir(parents=True)
    lock = root / "contracts/CANONICAL_TRAINING_LOCK_v1.json"
    lock.parent.mkdir(parents=True)
    lock.write_bytes(LOCK.read_bytes())
    pd.DataFrame(
        [
            {
                "selection_rank": 1,
                "role_rank": 1,
                "sample_id": "normal/n1.jpg",
                "y_true": 0,
                "replay_role": "normal_replay",
            }
        ]
    ).to_csv(template, index=False)
    pd.DataFrame(
        [{"sample_id": "normal/n1.jpg", "monitor_group": "A02_TREATMENT_NORMAL"}]
    ).to_csv(monitor, index=False)
    row = {
        "queue_order": 1,
        "job_id": "JOB_A",
        "cycle_id": "CYCLE_1",
        "release_state": "ENGINEERING_GATE",
        "queue_status": "READY",
        "machine_id": "machine_01",
        "seed_id": "S001",
        "training_seed": 123,
        "logical_run_id": "DRBE_S001_A",
        "logical_arm_id": "T_DYNAMIC_DECAY",
        "schedule_id": "SCHEDULE_T_DYNAMIC_DECAY",
        "job_kind": "ARM_SEGMENT",
        "segment_index": 1,
        "segment_start_epoch": 1,
        "segment_end_epoch": 140,
        "normal_replay_slots": 1,
        "defect_guard_slots": 0,
        "total_replay_slots": 1,
        "expected_steps_batch128": 938,
        "selection_pool_id": "A02",
        "selection_pool_digest": "A" * 64,
        "active_selection_template_id": "TPL",
        "active_selection_template_relpath": "templates/TPL.csv",
        "active_selection_template_sha256": sha256_file(template),
        "active_selection_digest": "B" * 64,
        "active_selection_rows": 1,
        "monitor_manifest_relpath": "monitor/CAUSAL_MONITOR_SAMPLES.csv",
        "monitor_manifest_sha256": sha256_file(monitor),
        "canonical_lock_relpath": "contracts/CANONICAL_TRAINING_LOCK_v1.json",
        "canonical_lock_file_sha256": sha256_file(lock),
        "dependency_job_id": "",
        "dependency_output_relpath": "",
        "resume_from_epoch": 0,
        "branch_checkpoint_required": False,
        "run_output_relpath": "../05_training_runs/DRBE_S001_A",
        "machine_output_relpath": "dynamic_replay_budget_efficiency_20260807/05_training_runs/DRBE_S001_A",
        "checkpoint_epochs_to_retain": "120;140",
        "planned_epoch_equivalents": 140.0,
    }
    registry = root / "JOB_EXECUTION_REGISTRY.csv"
    pd.DataFrame([row]).to_csv(registry, index=False)
    (root / "RUN_QUEUE_VALIDATION.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "job_count": 1,
                "job_registry_sha256": sha256_file(registry),
                "monitor_manifest_sha256": sha256_file(monitor),
                "canonical_lock_file_sha256": sha256_file(lock),
                "canonical_lock_relpath": "contracts/CANONICAL_TRAINING_LOCK_v1.json",
            }
        ),
        encoding="utf-8",
    )
    return root


def test_load_campaign_job_verifies_machine_and_frozen_hashes(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    job = load_campaign_job(queue, "JOB_A", expected_machine_id="machine_01")

    assert job.job_id == "JOB_A"
    assert job.segment_start_epoch == 1
    assert job.selection_template.name == "TPL.csv"
    assert job.monitor_manifest.name == "CAUSAL_MONITOR_SAMPLES.csv"
    assert job.canonical_lock == queue / "contracts/CANONICAL_TRAINING_LOCK_v1.json"
    assert job.canonical_lock_file_sha256 == sha256_file(LOCK)

    with pytest.raises(CampaignWorkerError, match="assigned to machine_01"):
        load_campaign_job(queue, "JOB_A", expected_machine_id="machine_02")

    job.selection_template.write_text("changed", encoding="utf-8")
    with pytest.raises(CampaignWorkerError, match="template checksum"):
        load_campaign_job(queue, "JOB_A", expected_machine_id="machine_01")


def test_load_campaign_job_rejects_canonical_lock_drift(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    lock = queue / "contracts/CANONICAL_TRAINING_LOCK_v1.json"
    lock.write_text("{}", encoding="utf-8")
    with pytest.raises(CampaignWorkerError, match="canonical lock checksum"):
        load_campaign_job(queue, "JOB_A", expected_machine_id="machine_01")


def _job(tmp_path: Path, *, start: int = 1, end: int = 140, branch: bool = False) -> CampaignJob:
    queue = _queue(tmp_path)
    base = load_campaign_job(queue, "JOB_A", expected_machine_id="machine_01")
    return CampaignJob(
        **{
            **base.__dict__,
            "segment_start_epoch": start,
            "segment_end_epoch": end,
            "resume_from_epoch": start - 1,
            "branch_checkpoint_required": branch,
        }
    )


def _audit(output: Path, completed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_state").mkdir(exist_ok=True)
    (output / "training_state/last.pt").write_bytes(b"last")
    (output / "dynamic_training_audit.json").write_text(
        json.dumps({"completed_epochs": completed}), encoding="utf-8"
    )


def test_resolve_segment_execution_supports_new_partial_resume_and_skip(tmp_path: Path) -> None:
    job = _job(tmp_path)
    output = tmp_path / "run"
    fresh = resolve_segment_execution(job, output)
    assert fresh.action == "NEW"
    assert fresh.actual_start_epoch == 1
    assert fresh.resume_checkpoint is None

    _audit(output, 57)
    resumed = resolve_segment_execution(job, output)
    assert resumed.action == "RESUME"
    assert resumed.actual_start_epoch == 58
    assert resumed.resume_checkpoint == output / "training_state/last.pt"

    _audit(output, 140)
    skipped = resolve_segment_execution(job, output)
    assert skipped.action == "SKIP_COMPLETE"
    assert skipped.actual_start_epoch == 141


def test_resolve_segment_execution_requires_exact_branch_parent(tmp_path: Path) -> None:
    job = _job(tmp_path, start=141, end=150, branch=True)
    child = tmp_path / "child"
    parent = tmp_path / "parent"

    with pytest.raises(CampaignWorkerError, match="dependency output"):
        resolve_segment_execution(job, child, dependency_output=parent)

    _audit(parent, 140)
    checkpoint = parent / "training_state/checkpoint_epoch_0140.pt"
    checkpoint.write_bytes(b"branch")
    decision = resolve_segment_execution(job, child, dependency_output=parent)
    assert decision.action == "BRANCH"
    assert decision.actual_start_epoch == 141
    assert decision.branch_parent_output == parent


def test_resolve_segment_execution_rejects_gap_before_registered_segment(tmp_path: Path) -> None:
    job = _job(tmp_path, start=141, end=150, branch=False)
    output = tmp_path / "run"
    _audit(output, 139)
    with pytest.raises(CampaignWorkerError, match="expected boundary 140"):
        resolve_segment_execution(job, output)


def test_zero_epoch_failure_is_archived_and_restarted_from_base(tmp_path: Path) -> None:
    job = _job(tmp_path)
    output = tmp_path / "run"
    (output / "trainer/weights").mkdir(parents=True)
    (output / "training_state").mkdir()
    (output / "process_telemetry").mkdir()
    (output / "job_inputs").mkdir()
    (output / "trainer/args.yaml").write_text("epochs: 140\n", encoding="utf-8")
    (output / "process_telemetry/.epoch_0001.parquet.partial.tmp").write_bytes(b"partial")
    (output / "job_inputs/selection_template.csv").write_text("sample_id\nN1\n", encoding="utf-8")
    (output / "resolved_training_args.json").write_text("{}", encoding="utf-8")
    (output / "dynamic_training_audit.json").write_text(
        json.dumps({"completed_epochs": 0, "segments": [{"status": "FAILED"}]}),
        encoding="utf-8",
    )

    decision = resolve_segment_execution(job, output)
    assert decision.action == "RESTART_ZERO_EPOCH"
    assert decision.actual_start_epoch == 1
    assert decision.resume_checkpoint is None

    archive = archive_zero_epoch_attempt(output, attempt_id="attempt_test")

    assert archive == output / "failed_attempts/attempt_test"
    assert (archive / "dynamic_training_audit.json").is_file()
    assert (archive / "trainer/args.yaml").is_file()
    assert (archive / "process_telemetry/.epoch_0001.parquet.partial.tmp").is_file()
    assert (archive / "ATTEMPT_ARCHIVE_MANIFEST.json").is_file()
    assert not (output / "dynamic_training_audit.json").exists()
    assert not (output / "trainer").exists()
    assert not (output / "training_state").exists()
    assert not (output / "process_telemetry").exists()
    assert (output / "job_inputs/selection_template.csv").is_file()

    restarted = resolve_segment_execution(job, output)
    assert restarted.action == "RESTART_ZERO_EPOCH"
    assert restarted.actual_start_epoch == 1


def test_zero_epoch_restart_recovers_an_interrupted_archive_transaction(tmp_path: Path) -> None:
    job = _job(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / ".zero_epoch_restart_transaction.json").write_text(
        json.dumps({"attempt_id": "attempt_interrupted"}), encoding="utf-8"
    )

    decision = resolve_segment_execution(job, output)
    assert decision.action == "RESTART_ZERO_EPOCH"

    archive = archive_zero_epoch_attempt(output)
    assert archive.name == "attempt_interrupted"
    assert not (output / ".zero_epoch_restart_transaction.json").exists()


def test_zero_epoch_recovery_is_forbidden_after_a_completed_epoch(tmp_path: Path) -> None:
    job = _job(tmp_path)
    output = tmp_path / "run"
    _audit(output, 1)
    (output / "training_state/last.pt").unlink()

    with pytest.raises(CampaignWorkerError, match="resumable last checkpoint is missing"):
        resolve_segment_execution(job, output)
