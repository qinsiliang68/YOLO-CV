from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.stage1_gapvalue240.dynamic_campaign_train_worker import parse_args as parse_worker_args
from stage1_gapvalue240.campaign_assignment import (
    CampaignAssignmentError,
    build_campaign_assignment,
    load_campaign_assignment,
)
from stage1_gapvalue240.campaign_layout import CAMPAIGN_ID
from stage1_gapvalue240.campaign_lease import (
    LeaseLostError,
    activate_assignment,
    claim_job_lease,
)
from stage1_gapvalue240.util import sha256_file


def _queue_and_release(tmp_path: Path) -> tuple[Path, Path]:
    queue = tmp_path / "04_run_queue"
    queue.mkdir()
    rows = [
        {
            "queue_order": 1,
            "job_id": "S001_PREFIX",
            "cycle_id": "CYCLE_1",
            "release_state": "ENGINEERING_GATE",
            "machine_id": "M01",
            "seed_id": "S001",
            "logical_run_id": "C1_S001_PREFIX",
            "logical_arm_id": "T_PREFIX",
            "dependency_job_id": "",
        },
        {
            "queue_order": 2,
            "job_id": "S001_CHILD",
            "cycle_id": "CYCLE_1",
            "release_state": "ENGINEERING_GATE",
            "machine_id": "M01",
            "seed_id": "S001",
            "logical_run_id": "C1_S001_CHILD",
            "logical_arm_id": "T_DECAY",
            "dependency_job_id": "S001_PREFIX",
        },
        {
            "queue_order": 3,
            "job_id": "S002_PREFIX",
            "cycle_id": "CYCLE_1",
            "release_state": "ENGINEERING_GATE",
            "machine_id": "M02",
            "seed_id": "S002",
            "logical_run_id": "C1_S002_PREFIX",
            "logical_arm_id": "T_PREFIX",
            "dependency_job_id": "",
        },
        {
            "queue_order": 4,
            "job_id": "S002_CHILD",
            "cycle_id": "CYCLE_1",
            "release_state": "ENGINEERING_GATE",
            "machine_id": "M02",
            "seed_id": "S002",
            "logical_run_id": "C1_S002_CHILD",
            "logical_arm_id": "T_DECAY",
            "dependency_job_id": "S002_PREFIX",
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
                "canonical_lock_file_sha256": "C" * 64,
            }
        ),
        encoding="utf-8",
    )
    source_tree_sha256 = "D" * 64
    gate = tmp_path / "ENGINEERING_GATE_REPORT.json"
    gate.write_text(
        json.dumps(
            {
                "schema_version": "stage1.dynamic_campaign_engineering_gate.v2",
                "status": "PASS",
                "validation_complete": True,
                "identity": {
                    "source_tree_sha256": source_tree_sha256,
                    "queue_registry_sha256": sha256_file(registry),
                    "canonical_lock_file_sha256": "C" * 64,
                },
                "evidence": {},
                "evidence_digest": "E" * 64,
            }
        ),
        encoding="utf-8",
    )
    release = tmp_path / "PILOT_RELEASED.json"
    release.write_text(
        json.dumps(
            {
                "schema_version": "stage1.dynamic_campaign_release.v2",
                "campaign_id": CAMPAIGN_ID,
                "release_id": "PILOT_S001_S002",
                "scope": "PILOT",
                "release_status": "RELEASED",
                "queue_registry_sha256": sha256_file(registry),
                "canonical_lock_file_sha256": "C" * 64,
                "seed_ids": ["S001", "S002"],
                "cycle_ids": ["CYCLE_1"],
                "job_ids": [row["job_id"] for row in rows],
                "job_count": len(rows),
                "dependency_policy": "CLOSED_WITHIN_RELEASE",
                "engineering_gate_report_sha256": sha256_file(gate),
                "engineering_gate_schema_version": "stage1.dynamic_campaign_engineering_gate.v2",
                "engineering_gate_source_tree_sha256": source_tree_sha256,
            }
        ),
        encoding="utf-8",
    )
    return queue, release


def _machine_configs(tmp_path: Path) -> Path:
    root = tmp_path / "machines"
    root.mkdir()
    for machine_id in ("machine_01", "machine_02", "machine_11"):
        (root / f"{machine_id}.yaml").write_text(
            yaml.safe_dump({"machine_id": machine_id}, sort_keys=True),
            encoding="utf-8",
        )
    return root


def test_assignment_maps_planning_slots_and_emits_one_command_per_job(tmp_path: Path) -> None:
    queue, release = _queue_and_release(tmp_path)
    configs = _machine_configs(tmp_path)
    queue_sha_before = sha256_file(queue / "JOB_EXECUTION_REGISTRY.csv")
    release_sha_before = sha256_file(release)

    result = build_campaign_assignment(
        queue,
        release,
        tmp_path / "assignment_v1",
        campaign_id=CAMPAIGN_ID,
        assignment_id="ASSIGNMENT_V1",
        machine_configs_dir=configs,
        slot_mapping={"M01": "machine_01", "M02": "machine_02"},
    )

    assignments = pd.read_csv(result.job_assignments_path, keep_default_na=False)
    commands = pd.read_csv(result.standalone_commands_path, keep_default_na=False)
    assert assignments.groupby("seed_id").assigned_machine_id.nunique().eq(1).all()
    assert assignments.set_index("job_id").loc["S001_CHILD", "assigned_machine_id"] == "machine_01"
    assert len(commands) == len(assignments) == 4
    assert commands.job_id.is_unique
    assert commands.command.str.contains("dynamic_campaign_train_worker.py", regex=False).all()
    assert ~commands.command.str.contains("run_dynamic_campaign_controller.py", regex=False).any()
    assert commands.apply(lambda row: f"--job-id {row.job_id}" in row.command, axis=1).all()
    assert commands.command.str.count(r"(?:^|\s)--job-id(?:\s|$)").eq(1).all()
    for forbidden in ("--max-jobs", "--job-list", "--job-range", "--once"):
        assert ~commands.command.str.contains(forbidden, regex=False).any()
    assert commands.command.str.contains("--assignment", regex=False).all()
    assert commands.command.str.contains("--expected-release-id PILOT_S001_S002", regex=False).all()
    assert commands.command.str.contains("--expected-canonical-lock-sha256", regex=False).all()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["standalone_execution_unit"] == "PHYSICAL_JOB"
    assert manifest["standalone_command_count"] == len(assignments)
    assert manifest["single_job_per_process"] is True
    assert manifest["implicit_next_job_forbidden"] is True
    assert manifest["dynamic_reassignment_mode"] == "NEW_IMMUTABLE_ASSIGNMENT_ONLY"
    assert manifest["training_code_edits_required"] is False
    assert sha256_file(queue / "JOB_EXECUTION_REGISTRY.csv") == queue_sha_before
    assert sha256_file(release) == release_sha_before


def test_whole_seed_can_be_reassigned_without_queue_or_code_changes(tmp_path: Path) -> None:
    queue, release = _queue_and_release(tmp_path)
    configs = _machine_configs(tmp_path)
    first = build_campaign_assignment(
        queue,
        release,
        tmp_path / "assignment_v1",
        campaign_id=CAMPAIGN_ID,
        assignment_id="ASSIGNMENT_V1",
        machine_configs_dir=configs,
        slot_mapping={"M01": "machine_01", "M02": "machine_02"},
    )
    queue_sha_before = sha256_file(queue / "JOB_EXECUTION_REGISTRY.csv")

    second = build_campaign_assignment(
        queue,
        release,
        tmp_path / "assignment_v2",
        campaign_id=CAMPAIGN_ID,
        assignment_id="ASSIGNMENT_V2",
        machine_configs_dir=configs,
        slot_mapping={"M01": "machine_01", "M02": "machine_02"},
        seed_overrides={"S001": "machine_11"},
        supersedes_assignment=first.manifest_path,
        reassignment_reason="machine_01 unavailable before dispatch",
    )

    rows = pd.read_csv(second.job_assignments_path, keep_default_na=False)
    assert set(rows.loc[rows.seed_id.eq("S001"), "assigned_machine_id"]) == {"machine_11"}
    assert set(rows.loc[rows.seed_id.eq("S002"), "assigned_machine_id"]) == {"machine_02"}
    manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    assert manifest["supersedes_assignment_sha256"] == sha256_file(first.manifest_path)
    assert manifest["reassignment_reason"] == "machine_01 unavailable before dispatch"
    first_rows = pd.read_csv(first.job_assignments_path, keep_default_na=False)
    scientific_identity = [
        "campaign_id",
        "release_id",
        "release_sha256",
        "queue_registry_sha256",
        "job_id",
        "cycle_id",
        "seed_id",
        "block_id",
        "planned_machine_slot",
        "dependency_job_id",
    ]
    pd.testing.assert_frame_equal(
        first_rows[scientific_identity].reset_index(drop=True),
        rows[scientific_identity].reset_index(drop=True),
    )
    assert sha256_file(queue / "JOB_EXECUTION_REGISTRY.csv") == queue_sha_before


def test_reassignment_generation_fences_the_old_standalone_worker(tmp_path: Path) -> None:
    queue, release = _queue_and_release(tmp_path)
    configs = _machine_configs(tmp_path)
    first = build_campaign_assignment(
        queue,
        release,
        tmp_path / "assignment_v1",
        campaign_id=CAMPAIGN_ID,
        assignment_id="ASSIGNMENT_V1",
        machine_configs_dir=configs,
        slot_mapping={"M01": "machine_01", "M02": "machine_02"},
    )
    second = build_campaign_assignment(
        queue,
        release,
        tmp_path / "assignment_v2",
        campaign_id=CAMPAIGN_ID,
        assignment_id="ASSIGNMENT_V2",
        machine_configs_dir=configs,
        slot_mapping={"M01": "machine_01", "M02": "machine_02"},
        seed_overrides={"S001": "machine_11"},
        supersedes_assignment=first.manifest_path,
        reassignment_reason="machine_01 unavailable before dispatch",
    )
    first_rows = pd.read_csv(first.job_assignments_path, keep_default_na=False)
    second_manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    coordination = tmp_path / "coordination"
    activate_assignment(
        coordination,
        campaign_id=CAMPAIGN_ID,
        release_id="PILOT_S001_S002",
        assignment_id="ASSIGNMENT_V1",
        assignment_sha256=sha256_file(first.manifest_path),
        job_ids=tuple(first_rows.job_id.astype(str)),
    )
    old_lease = claim_job_lease(
        coordination,
        campaign_id=CAMPAIGN_ID,
        release_id="PILOT_S001_S002",
        assignment_id="ASSIGNMENT_V1",
        assignment_sha256=sha256_file(first.manifest_path),
        job_id="S001_PREFIX",
        machine_id="machine_01",
        ttl_seconds=60,
        heartbeat_seconds=10,
    )
    old_lease.release(status="REASSIGNMENT_READY")
    activate_assignment(
        coordination,
        campaign_id=CAMPAIGN_ID,
        release_id="PILOT_S001_S002",
        assignment_id="ASSIGNMENT_V2",
        assignment_sha256=sha256_file(second.manifest_path),
        job_ids=tuple(first_rows.job_id.astype(str)),
        expected_previous_assignment_sha256=second_manifest[
            "supersedes_assignment_sha256"
        ],
    )
    with pytest.raises(LeaseLostError, match="active assignment changed"):
        old_lease.check_now()


def test_assignment_rejects_unknown_seed_machine_and_tampering(tmp_path: Path) -> None:
    queue, release = _queue_and_release(tmp_path)
    configs = _machine_configs(tmp_path)
    common = dict(
        campaign_id=CAMPAIGN_ID,
        assignment_id="BAD",
        machine_configs_dir=configs,
        slot_mapping={"M01": "machine_01", "M02": "machine_02"},
    )
    with pytest.raises(CampaignAssignmentError, match="unknown seed"):
        build_campaign_assignment(
            queue,
            release,
            tmp_path / "bad_seed",
            seed_overrides={"S999": "machine_11"},
            **common,
        )
    with pytest.raises(CampaignAssignmentError, match="unknown machine"):
        build_campaign_assignment(
            queue,
            release,
            tmp_path / "bad_machine",
            seed_overrides={"S001": "machine_99"},
            **common,
        )

    result = build_campaign_assignment(
        queue,
        release,
        tmp_path / "valid",
        **common,
    )
    rows = pd.read_csv(result.job_assignments_path, keep_default_na=False)
    rows.loc[rows.job_id.eq("S001_CHILD"), "assigned_machine_id"] = "machine_02"
    rows.to_csv(result.job_assignments_path, index=False)
    with pytest.raises(CampaignAssignmentError, match="checksum"):
        load_campaign_assignment(
            queue,
            release,
            result.manifest_path,
            expected_campaign_id=CAMPAIGN_ID,
        )


def test_assignment_authorizes_exact_job_machine_and_worker_cli_requires_it(tmp_path: Path) -> None:
    queue, release = _queue_and_release(tmp_path)
    configs = _machine_configs(tmp_path)
    result = build_campaign_assignment(
        queue,
        release,
        tmp_path / "assignment",
        campaign_id=CAMPAIGN_ID,
        assignment_id="ASSIGNMENT_V1",
        machine_configs_dir=configs,
        slot_mapping={"M01": "machine_01", "M02": "machine_02"},
    )

    loaded = load_campaign_assignment(
        queue,
        release,
        result.manifest_path,
        expected_campaign_id=CAMPAIGN_ID,
        expected_machine_id="machine_01",
        expected_job_id="S001_CHILD",
    )
    assert loaded.assignment_id == "ASSIGNMENT_V1"
    assert loaded.assigned_machine("S001_CHILD") == "machine_01"
    with pytest.raises(CampaignAssignmentError, match="assigned to machine_01"):
        load_campaign_assignment(
            queue,
            release,
            result.manifest_path,
            expected_campaign_id=CAMPAIGN_ID,
            expected_machine_id="machine_02",
            expected_job_id="S001_CHILD",
        )

    parsed = parse_worker_args(
        [
            "--machine-config",
            "machine.yaml",
            "--job-id",
            "S001_CHILD",
            "--release",
            "release.json",
            "--assignment",
            "assignment.json",
            "--expected-release-id",
            "PILOT_S001_S002",
            "--expected-canonical-lock-sha256",
            "C" * 64,
        ]
    )
    assert parsed.assignment == "assignment.json"
    assert parsed.expected_release_id == "PILOT_S001_S002"


def test_assignment_machine_config_paths_are_portable_across_repo_roots(
    tmp_path: Path,
) -> None:
    source_repo = tmp_path / "source_repo"
    source_repo.mkdir()
    queue, release = _queue_and_release(source_repo)
    configs = _machine_configs(source_repo)
    result = build_campaign_assignment(
        queue,
        release,
        source_repo / "assignment",
        campaign_id=CAMPAIGN_ID,
        assignment_id="ASSIGNMENT_PORTABLE",
        machine_configs_dir=configs,
        slot_mapping={"M01": "machine_01", "M02": "machine_02"},
        repo_root=source_repo,
    )
    rows = pd.read_csv(result.job_assignments_path, keep_default_na=False)
    assert rows.machine_config_path.str.startswith("machines/").all()
    assert not rows.machine_config_path.map(lambda value: Path(value).is_absolute()).any()

    mirror_repo = tmp_path / "mirror_repo"
    mirror_configs = mirror_repo / "machines"
    mirror_configs.mkdir(parents=True)
    for source in configs.glob("*.yaml"):
        (mirror_configs / source.name).write_bytes(source.read_bytes())

    loaded = load_campaign_assignment(
        queue,
        release,
        result.manifest_path,
        expected_campaign_id=CAMPAIGN_ID,
        expected_machine_id="machine_01",
        expected_job_id="S001_PREFIX",
        repo_root=mirror_repo,
    )
    assert loaded.assigned_machine("S001_PREFIX") == "machine_01"


def test_worker_cli_rejects_duplicate_or_batched_job_ids() -> None:
    base = [
        "--machine-config",
        "machine.yaml",
        "--release",
        "release.json",
        "--assignment",
        "assignment.json",
        "--expected-release-id",
        "PILOT_S001_S002",
        "--expected-canonical-lock-sha256",
        "C" * 64,
    ]
    with pytest.raises(SystemExit):
        parse_worker_args(base + ["--job-id", "S001", "--job-id", "S002"])
    with pytest.raises(SystemExit):
        parse_worker_args(base + ["--job-id", "S001,S002"])
    with pytest.raises(SystemExit):
        parse_worker_args(base + ["--job-id", "S001", "--max-jobs", "2"])
