"""Aggregate failure-injection and recovery evidence without accepting partial completion."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping

from .errors import ValidationError
from .util import atomic_write_json, sha256_file


FAILURE_VALIDATION_SCHEMA = "stage1.failure_injection_validation.v1"
REQUIRED_SCENARIOS = {
    "zero_epoch_oom": "stage1.failure_scenario.zero_epoch_oom.v1",
    "process_kill_resume": "stage1.failure_scenario.process_kill_resume.v1",
    "bad_checkpoint": "stage1.failure_scenario.bad_checkpoint.v1",
    "bad_sidecar": "stage1.failure_scenario.bad_sidecar.v1",
    "disk_preflight_failure": "stage1.failure_scenario.disk_preflight_failure.v1",
    "atomic_write_interruption": "stage1.failure_scenario.atomic_write_interruption.v1",
    "controller_loss": "stage1.failure_scenario.controller_loss.v1",
    "assignment_fencing": "stage1.failure_scenario.assignment_fencing.v1",
    "hot_spare_full_block_restart": "stage1.failure_scenario.hot_spare_full_block_restart.v1",
}


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"unreadable failure scenario report: {path}") from exc
    if not isinstance(payload, dict):
        raise ValidationError(f"failure scenario report is not an object: {path}")
    return payload


def validate_failure_recovery_evidence(
    scenario_reports: Mapping[str, str | Path],
    *,
    output_path: str | Path,
    allow_cross_machine_resume: bool = False,
    full_state_package_validation: str | Path | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    if set(scenario_reports) != set(REQUIRED_SCENARIOS):
        issues.append(
            f"scenario registry mismatch: missing={sorted(set(REQUIRED_SCENARIOS)-set(scenario_reports))}, "
            f"extra={sorted(set(scenario_reports)-set(REQUIRED_SCENARIOS))}"
        )
    validated: dict[str, Any] = {}
    for name, schema in REQUIRED_SCENARIOS.items():
        if name not in scenario_reports:
            continue
        path = Path(scenario_reports[name]).resolve()
        if not path.is_file():
            issues.append(f"missing scenario report: {name}")
            continue
        payload = _load(path)
        if payload.get("schema_version") != schema or payload.get("status") != "PASS":
            issues.append(f"scenario {name} schema/status mismatch")
            continue
        checks: dict[str, Any] = {}
        if name == "zero_epoch_oom":
            checks = {"failed_attempt_preserved": True, "restart_from_base_checkpoint": True, "canonical_completion_count": 1}
        elif name == "process_kill_resume":
            checks = {"resume_epoch_contiguous": True, "duplicate_epoch_count": 0, "canonical_parameters_unchanged": True}
        elif name in {"bad_checkpoint", "bad_sidecar", "disk_preflight_failure"}:
            checks = {"formal_start_blocked": True, "completion_published": False}
        elif name == "atomic_write_interruption":
            checks = {"completion_published": False, "partial_artifacts_cleaned_or_quarantined": True}
        elif name == "controller_loss":
            checks = {"worker_survived_controller_loss": True, "duplicate_worker_count": 0}
        elif name == "assignment_fencing":
            checks = {"stale_holder_heartbeat_blocked": True, "stale_holder_publish_blocked": True, "new_holder_completed": True}
        elif name == "hot_spare_full_block_restart":
            checks = {"full_block_restarted": True, "orphan_attempts_excluded": True, "canonical_completion_count": 1}
        mismatches = {
            key: {"expected": value, "observed": payload.get(key)}
            for key, value in checks.items()
            if payload.get(key) != value
        }
        if mismatches:
            issues.append(f"scenario {name} contract mismatch: {mismatches}")
        validated[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "schema_version": schema,
            "checks": checks,
        }
    state_package: dict[str, Any] | None = None
    if allow_cross_machine_resume:
        if full_state_package_validation is None:
            issues.append("cross-machine resume requested without full-state package validation")
        else:
            state_path = Path(full_state_package_validation).resolve()
            state_package = _load(state_path)
            expected = {
                "schema_version": "stage1.full_training_state_package_validation.v1",
                "status": "PASS",
                "model": "PASS",
                "ema": "PASS",
                "optimizer": "PASS",
                "scaler": "PASS",
                "rng": "PASS",
                "sampler": "PASS",
                "workspace": "PASS",
                "telemetry_boundary": "PASS",
            }
            mismatches = {k: {"expected": v, "observed": state_package.get(k)} for k, v in expected.items() if state_package.get(k) != v}
            if mismatches:
                issues.append(f"cross-machine full-state package is invalid: {mismatches}")
    report = {
        "schema_version": FAILURE_VALIDATION_SCHEMA,
        "status": "PASS" if not issues else "FAIL",
        "created_at_unix": time.time(),
        "issues": issues,
        "cross_machine_resume_enabled": bool(allow_cross_machine_resume and not issues),
        "cross_machine_resume_default": "DISABLED",
        "validated_scenarios": validated,
        "full_state_package_validation": state_package,
    }
    atomic_write_json(output_path, report, overwrite=True)
    if issues:
        raise ValidationError(f"failure/recovery validation failed; see {output_path}")
    return report
