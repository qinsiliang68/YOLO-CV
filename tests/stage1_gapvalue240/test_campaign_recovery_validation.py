from __future__ import annotations

import json
from pathlib import Path

import pytest

from stage1_gapvalue240.campaign_recovery_validation import REQUIRED_SCENARIOS, validate_failure_recovery_evidence
from stage1_gapvalue240.errors import ValidationError
from stage1_gapvalue240.util import atomic_write_json


def _reports(tmp_path: Path) -> dict[str, Path]:
    result = {}
    extras = {
        "zero_epoch_oom": {"failed_attempt_preserved": True, "restart_from_base_checkpoint": True, "canonical_completion_count": 1},
        "process_kill_resume": {"resume_epoch_contiguous": True, "duplicate_epoch_count": 0, "canonical_parameters_unchanged": True},
        "bad_checkpoint": {"formal_start_blocked": True, "completion_published": False},
        "bad_sidecar": {"formal_start_blocked": True, "completion_published": False},
        "disk_preflight_failure": {"formal_start_blocked": True, "completion_published": False},
        "atomic_write_interruption": {"completion_published": False, "partial_artifacts_cleaned_or_quarantined": True},
        "controller_loss": {"worker_survived_controller_loss": True, "duplicate_worker_count": 0},
        "assignment_fencing": {"stale_holder_heartbeat_blocked": True, "stale_holder_publish_blocked": True, "new_holder_completed": True},
        "hot_spare_full_block_restart": {"full_block_restarted": True, "orphan_attempts_excluded": True, "canonical_completion_count": 1},
    }
    for name, schema in REQUIRED_SCENARIOS.items():
        path = tmp_path / f"{name}.json"
        atomic_write_json(path, {"schema_version": schema, "status": "PASS", **extras[name]})
        result[name] = path
    return result


def test_failure_recovery_validation_accepts_complete_evidence(tmp_path: Path) -> None:
    report = validate_failure_recovery_evidence(_reports(tmp_path), output_path=tmp_path / "out.json")
    assert report["status"] == "PASS"
    assert report["cross_machine_resume_enabled"] is False


def test_hot_spare_partial_attempt_is_rejected(tmp_path: Path) -> None:
    reports = _reports(tmp_path)
    payload = json.loads(reports["hot_spare_full_block_restart"].read_text())
    payload["orphan_attempts_excluded"] = False
    atomic_write_json(reports["hot_spare_full_block_restart"], payload, overwrite=True)
    with pytest.raises(ValidationError, match="failure/recovery"):
        validate_failure_recovery_evidence(reports, output_path=tmp_path / "out.json")


def test_cross_machine_resume_requires_full_state_validation(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="failure/recovery"):
        validate_failure_recovery_evidence(
            _reports(tmp_path), output_path=tmp_path / "out.json", allow_cross_machine_resume=True
        )
