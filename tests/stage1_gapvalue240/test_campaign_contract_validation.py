from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.campaign_assignment import build_campaign_assignment
from stage1_gapvalue240.campaign_contract_validation import (
    build_source_tree_manifest,
    validate_assignment_reassignment,
    validate_source_tree_immutability,
    validate_standalone_entry,
)
from stage1_gapvalue240.errors import ValidationError
from stage1_gapvalue240.util import atomic_write_json, sha256_file
from tests.stage1_gapvalue240.test_campaign_assignment import (
    CAMPAIGN_ID,
    _machine_configs,
    _queue_and_release,
)


def _smoke(path: Path, job_id: str) -> Path:
    atomic_write_json(
        path,
        {
            "schema_version": "stage1.standalone_job_smoke.v1",
            "status": "PASS",
            "job_id": job_id,
            "controller_offline": True,
            "single_process": True,
            "exit_code": 0,
        },
    )
    return path


def test_standalone_entry_validates_one_command_per_released_job(tmp_path: Path) -> None:
    queue, release = _queue_and_release(tmp_path)
    configs = _machine_configs(tmp_path)
    assignment = build_campaign_assignment(
        queue,
        release,
        tmp_path / "assignment",
        campaign_id=CAMPAIGN_ID,
        assignment_id="ASSIGNMENT_V1",
        machine_configs_dir=configs,
        slot_mapping={"M01": "machine_01", "M02": "machine_02"},
        repo_root=tmp_path,
    )
    commands = pd.read_csv(assignment.standalone_commands_path, keep_default_na=False)
    smoke = _smoke(tmp_path / "standalone_smoke.json", str(commands.iloc[0].job_id))
    report = validate_standalone_entry(
        queue,
        release,
        assignment.manifest_path,
        repo_root=tmp_path,
        controller_offline_smoke_report=smoke,
        output_path=tmp_path / "STANDALONE_ENTRY_VALIDATION.json",
    )
    assert report["status"] == "PASS"
    assert report["command_count"] == len(commands)
    assert report["one_job_per_command"]
    assert report["controller_offline_smoke"] == "PASS"


def test_standalone_entry_rejects_batch_job_arguments(tmp_path: Path) -> None:
    queue, release = _queue_and_release(tmp_path)
    configs = _machine_configs(tmp_path)
    assignment = build_campaign_assignment(
        queue,
        release,
        tmp_path / "assignment",
        campaign_id=CAMPAIGN_ID,
        assignment_id="ASSIGNMENT_V1",
        machine_configs_dir=configs,
        slot_mapping={"M01": "machine_01", "M02": "machine_02"},
        repo_root=tmp_path,
    )
    commands = pd.read_csv(assignment.standalone_commands_path, keep_default_na=False)
    commands.loc[0, "command"] += " --job-id EXTRA_JOB"
    commands.loc[0, "command_sha256"] = "0" * 64
    commands.to_csv(assignment.standalone_commands_path, index=False)
    manifest = json.loads(assignment.manifest_path.read_text(encoding="utf-8"))
    manifest["standalone_commands_sha256"] = sha256_file(assignment.standalone_commands_path)
    assignment.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValidationError, match="standalone entry validation"):
        validate_standalone_entry(
            queue,
            release,
            assignment.manifest_path,
            repo_root=tmp_path,
            controller_offline_smoke_report=_smoke(tmp_path / "smoke.json", "S001_PREFIX"),
            output_path=tmp_path / "validation.json",
        )


def test_reassignment_validation_proves_only_placement_changed(tmp_path: Path) -> None:
    queue, release = _queue_and_release(tmp_path)
    configs = _machine_configs(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "worker.py").write_text("print('stable')\n", encoding="utf-8")
    before_manifest = build_source_tree_manifest(source, tmp_path / "source_before.json")
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
        reassignment_reason="machine_01 offline",
    )
    report = validate_assignment_reassignment(
        queue,
        release,
        first.manifest_path,
        second.manifest_path,
        source_root=source,
        source_manifest_before=before_manifest,
        output_path=tmp_path / "ASSIGNMENT_REASSIGNMENT_VALIDATION.json",
    )
    assert report["status"] == "PASS"
    assert report["scientific_identity_equal"]
    assert report["placement_changed"]
    assert report["old_assignment_preserved"]
    assert report["source_tree_unchanged"]


def test_source_tree_immutability_detects_changed_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    file = source / "worker.py"
    file.write_text("a = 1\n", encoding="utf-8")
    baseline = build_source_tree_manifest(source, tmp_path / "before.json")
    file.write_text("a = 2\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="source tree immutability"):
        validate_source_tree_immutability(
            source,
            baseline,
            tmp_path / "SOURCE_TREE_IMMUTABILITY_VALIDATION.json",
        )
