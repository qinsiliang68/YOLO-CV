"""Fail-closed global audit for the Stage1 research-finalization package.

The audit never starts training, generates a queue, or opens a holdout.  It
only verifies already-recorded evidence, including the user-approved literature
scope freeze and the still-missing BudgetedReplay source carriers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "stage1.dynamic_replay.global_completion_audit.v2"
EXPECTED_MISSING_SOURCES = [
    "Stage1_BudgetedReplay_Learnability_20260809_v1.0.0.tar.gz",
    "Stage1_BudgetedReplay_Learnability_20260809_v1.0.0.zip",
    "stage1_budgeted_replay-1.0.0-py3-none-any.whl",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _evidence(path: Path, root: Path) -> dict[str, Any]:
    try:
        shown = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        shown = str(path.resolve())
    return {
        "path": shown,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _check(
    status: str,
    *,
    evidence: list[dict[str, Any]],
    details: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "status": status,
        "evidence": evidence,
        "details": dict(details),
    }


def _tripartite_check(
    *, repo_root: Path, experiment_root: Path
) -> tuple[bool, dict[str, Any]]:
    root = experiment_root / "01_field_audit" / "expert_review_reproductions"
    validation_path = root / "TRIPARTITE_CROSSWALK_VALIDATION_v2.json"
    receipt_path = root / "tripartite_reproduction_v2" / "EXECUTION_RECEIPT.json"
    budgeted_path = root / "expert_vs_v3_tripartite_v2.csv"
    dynamic_path = root / "dynamic_review_vs_v3_tripartite_v2.csv"
    paths = [validation_path, receipt_path, budgeted_path, dynamic_path]
    errors: list[str] = []
    if not all(path.is_file() for path in paths):
        errors.extend(str(path) for path in paths if not path.is_file())
        return False, {"errors": errors, "evidence": []}

    validation = _read_json(validation_path)
    receipt = _read_json(receipt_path)
    rows = _read_csv(budgeted_path) + _read_csv(dynamic_path)
    row_map = {row.get("requirement_id", ""): row for row in rows}
    result_map = {
        str(item.get("requirement_id", "")): item
        for item in receipt.get("results", [])
    }
    if validation.get("status") != "PASS" or validation.get("error_count") != 0:
        errors.append("tripartite schema validation is not PASS with zero errors")
    if validation.get("total_rows") != 46 or len(rows) != 46 or len(row_map) != 46:
        errors.append("tripartite row count or unique identity is not exactly 46")
    if (
        receipt.get("status") != "PASS"
        or receipt.get("row_count") != 46
        or receipt.get("nonzero_exit_count") != 0
        or receipt.get("commands_executed") is not True
    ):
        errors.append("tripartite execution receipt is not 46/46 exit-zero PASS")
    if set(row_map) != set(result_map):
        errors.append("matrix and execution-receipt requirement identities differ")

    for requirement_id, item in result_map.items():
        artifact = Path(str(item.get("path", "")))
        if not artifact.is_absolute():
            artifact = repo_root / artifact
        try:
            artifact.resolve().relative_to(repo_root.resolve())
        except ValueError:
            errors.append(f"{requirement_id}: result artifact is outside repository")
            continue
        if not artifact.is_file():
            errors.append(f"{requirement_id}: result artifact missing")
            continue
        actual_sha = _sha256(artifact)
        row = row_map.get(requirement_id, {})
        if actual_sha != str(item.get("sha256", "")).upper():
            errors.append(f"{requirement_id}: receipt SHA mismatch")
        if actual_sha != str(row.get("result_artifact_sha", "")).upper():
            errors.append(f"{requirement_id}: matrix SHA mismatch")
        if str(item.get("exit_code")) != str(row.get("exit_code")) or item.get(
            "exit_code"
        ) != 0:
            errors.append(f"{requirement_id}: exit-code mismatch or nonzero")
        if str(item.get("command", "")) != str(row.get("reproduction_command", "")):
            errors.append(f"{requirement_id}: command mismatch")

    return not errors, {
        "errors": errors,
        "row_count": len(rows),
        "executed_command_count": len(result_map),
        "evidence": [_evidence(path, repo_root) for path in paths],
    }


def _desktop_mirror_check(
    *, repo_root: Path, mirror: Path
) -> tuple[bool, dict[str, Any]]:
    validation_path = mirror / "PACKAGE_VALIDATION.json"
    manifest_path = mirror / "MANIFEST_SHA256.csv"
    errors: list[str] = []
    if not validation_path.is_file() or not manifest_path.is_file():
        return False, {
            "errors": ["Desktop validation or manifest is missing"],
            "evidence": [],
        }
    validation = _read_json(validation_path)
    rows = _read_csv(manifest_path)
    expected_manifest_sha = str(validation.get("manifest", {}).get("sha256", ""))
    if validation.get("status") != "PASS":
        errors.append("Desktop package validation status is not PASS")
    if _sha256(manifest_path) != expected_manifest_sha.upper():
        errors.append("Desktop manifest SHA does not match package validation")
    if len(rows) != 610 or validation.get("counts", {}).get("payload_files") != 610:
        errors.append("Desktop payload count is not exactly 610")
    for row in rows:
        relative = Path(str(row.get("relative_path", "")).replace("/", "\\"))
        destination = (mirror / relative).resolve()
        try:
            destination.relative_to(mirror.resolve())
        except ValueError:
            errors.append(f"manifest path escapes Desktop mirror: {relative}")
            continue
        if not destination.is_file():
            errors.append(f"Desktop payload missing: {relative}")
            continue
        if _sha256(destination) != str(row.get("sha256", "")).upper():
            errors.append(f"Desktop payload SHA mismatch: {relative}")
        if destination.stat().st_size != int(row.get("bytes", "-1")):
            errors.append(f"Desktop payload byte mismatch: {relative}")
        source = Path(str(row.get("source_path", "")))
        if not source.is_file():
            errors.append(f"registered source missing: {source}")
            continue
        if _sha256(source) != str(row.get("source_sha256", "")).upper():
            errors.append(f"registered source SHA mismatch: {source}")
    return not errors, {
        "errors": errors,
        "manifest_rows": len(rows),
        "evidence": [
            _evidence(validation_path, repo_root),
            _evidence(manifest_path, repo_root),
        ],
    }


def _file_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(candidate.is_file() for candidate in path.rglob("*"))


def build_global_completion_audit(
    *,
    repo_root: str | Path,
    experiment_root: str | Path,
    desktop_literature_mirror: str | Path,
) -> dict[str, Any]:
    """Return the current global audit without mutating any evidence."""

    repo = Path(repo_root).resolve()
    experiment = Path(experiment_root).resolve()
    mirror = Path(desktop_literature_mirror).resolve()
    checks: dict[str, dict[str, Any]] = {}

    expert_path = (
        experiment
        / "01_field_audit"
        / "expert_delivery_audit_v3"
        / "expert_v1_hash_validation.json"
    )
    expert = _read_json(expert_path)
    missing_sources = list(expert.get("missing_required_sources", []))
    archives = expert.get("archive_summaries", {})
    expert_assets_ok = (
        expert.get("expected_artifact_count") == 17
        and expert.get("observed_inventory_rows", 0) >= 17
        and expert.get("hash_or_archive_failure_count") == 0
        and archives
        and all(item.get("status") == "PASS" for item in archives.values())
    )
    checks["expert_assets_accounted_and_present_hashes_verified"] = _check(
        "PASS" if expert_assets_ok else "FAIL",
        evidence=[_evidence(expert_path, repo)],
        details={
            "expected_artifact_identities": expert.get("expected_artifact_count"),
            "observed_inventory_rows": expert.get("observed_inventory_rows"),
            "archive_member_rows": expert.get("archive_member_rows"),
            "hash_or_archive_failure_count": expert.get(
                "hash_or_archive_failure_count"
            ),
        },
    )
    source_missing_exact = missing_sources == EXPECTED_MISSING_SOURCES
    checks["expert_source_level_audit"] = _check(
        "NOT_TESTABLE_SOURCE_MISSING" if source_missing_exact else "FAIL",
        evidence=[_evidence(expert_path, repo)],
        details={
            "missing_required_sources": missing_sources,
            "report_or_excerpt_substitution_allowed": False,
        },
    )

    tripartite_ok, tripartite = _tripartite_check(
        repo_root=repo, experiment_root=experiment
    )
    checks["tripartite_46_rows_hash_bound"] = _check(
        "PASS" if tripartite_ok else "FAIL",
        evidence=tripartite.pop("evidence"),
        details=tripartite,
    )

    literature_root = experiment / "02_literature" / "review_500_300_100_v2"
    decision_path = literature_root / "USER_LITERATURE_SUFFICIENCY_DECISION_20260810.md"
    decision_text = decision_path.read_text(encoding="utf-8")
    literature_decision_ok = (
        "LITERATURE_EXPANSION_STOPPED" in decision_text
        and "Q/R/A/D signals remain hypotheses" in decision_text
        and "does not authorize formal training" in decision_text
    )
    checks["literature_user_scope_decision_registered"] = _check(
        "PASS" if literature_decision_ok else "FAIL",
        evidence=[_evidence(decision_path, repo)],
        details={
            "decision": "LITERATURE_EXPANSION_STOPPED",
            "legacy_formal_gate_overridden_for_current_scope": literature_decision_ok,
        },
    )
    legacy_literature_path = literature_root / "validation" / "COMPLETION_AUDIT.json"
    legacy_literature = _read_json(legacy_literature_path)
    checks["legacy_formal_literature_audit"] = _check(
        "SUPERSEDED_BY_USER_SCOPE_DECISION"
        if literature_decision_ok
        else "INCOMPLETE",
        evidence=[_evidence(legacy_literature_path, repo)],
        details={
            "observed_status": legacy_literature.get("status"),
            "still_reported_not_relabelled_pass": True,
        },
    )
    mirror_ok, mirror_result = _desktop_mirror_check(repo_root=repo, mirror=mirror)
    checks["literature_desktop_mirror_integrity"] = _check(
        "PASS" if mirror_ok else "FAIL",
        evidence=mirror_result.pop("evidence"),
        details=mirror_result,
    )

    prereg_path = experiment / "03_preregistration_v3" / "PREREGISTRATION_VALIDATION.json"
    prereg = _read_json(prereg_path)
    prereg_ok = (
        prereg.get("status") == "PASS"
        and prereg.get("scientific_status") == "PREREGISTERED_NOT_RUN"
        and prereg.get("candidate_effectiveness_claim") == "NOT_EVALUATED"
        and prereg.get("formal_training_authorized") is False
        and prereg.get("engineering_gate_authorized") is False
        and prereg.get("blind_holdout_authorized") is False
        and prereg.get("validation_generated_assignments") is False
    )
    checks["preregistration_v3_contract"] = _check(
        "PASS" if prereg_ok else "FAIL",
        evidence=[_evidence(prereg_path, repo)],
        details={
            "scientific_status": prereg.get("scientific_status"),
            "candidate_effectiveness_claim": prereg.get(
                "candidate_effectiveness_claim"
            ),
            "arm_ids": prereg.get("arm_ids"),
            "confirmation_seed_count": prereg.get("confirmation_seed_count"),
        },
    )

    suite_path = experiment / "08_reports" / "V3_TEST_SUITE_20260810.txt"
    suite_text = suite_path.read_text(encoding="utf-8-sig")
    passed_match = re.search(r"(\d+) passed", suite_text)
    suite_passed = int(passed_match.group(1)) if passed_match else 0
    suite_ok = "EXIT_CODE=0" in suite_text and suite_passed >= 230
    checks["v3_test_suite"] = _check(
        "PASS" if suite_ok else "FAIL",
        evidence=[_evidence(suite_path, repo)],
        details={"passed": suite_passed, "exit_code": 0 if suite_ok else None},
    )

    training_files = _file_count(experiment / "05_training_runs")
    formal_training_started = training_files != 0
    checks["formal_training_not_started"] = _check(
        "PASS" if not formal_training_started else "FAIL",
        evidence=[],
        details={"05_training_runs_file_count": training_files},
    )

    plan_path = experiment / "00_registry" / "experiment_plan.json"
    plan = _read_json(plan_path)
    active_queue_files = _file_count(experiment / "04_run_queue_v3")
    active_runtime_dirs = plan.get("active_runtime_dirs", {})
    execution_authorized = plan.get("execution_authorized")
    active_gate_or_assignments = not (
        active_queue_files == 0
        and active_runtime_dirs == {}
        and execution_authorized is False
    )
    checks["active_v3_gate_and_assignments_not_generated"] = _check(
        "PASS" if not active_gate_or_assignments else "FAIL",
        evidence=[_evidence(plan_path, repo)],
        details={
            "04_run_queue_v3_file_count": active_queue_files,
            "active_runtime_dirs": active_runtime_dirs,
            "execution_authorized": execution_authorized,
        },
    )

    v2_queue_path = experiment / "04_run_queue_v2" / "RUN_QUEUE_VALIDATION.json"
    legacy_release_path = (
        experiment / "04_run_queue" / "releases" / "PILOT_S001_S005_RELEASED.json"
    )
    v2_queue = _read_json(v2_queue_path)
    legacy_release = _read_json(legacy_release_path)
    historical_gate_detected = (
        v2_queue.get("engineering_gate_job_count", 0) > 0
        and legacy_release.get("release_status") == "RELEASED"
    )
    checks["historical_gate_assets_preserved_nonactive"] = _check(
        "PASS" if historical_gate_detected and not active_gate_or_assignments else "FAIL",
        evidence=[
            _evidence(v2_queue_path, repo),
            _evidence(legacy_release_path, repo),
        ],
        details={
            "engineering_gate_v2_jobs": v2_queue.get("engineering_gate_job_count"),
            "legacy_released_pilot_jobs": legacy_release.get("job_count"),
            "globally_erased_or_reported_absent": False,
            "current_v3_authorized": False,
        },
    )

    blind_v1_path = experiment / "03_preregistration" / "BLIND_HOLDOUT_STATUS.json"
    blind_v2_path = experiment / "03_preregistration_v2" / "BLIND_HOLDOUT_STATUS.json"
    blind_v1 = _read_json(blind_v1_path)
    blind_v2 = _read_json(blind_v2_path)
    evaluation_files = _file_count(experiment / "07_evaluation")
    blind_opened = not (
        blind_v1.get("status") == "UNBOUND"
        and blind_v2.get("status") == "UNBOUND"
        and prereg.get("blind_holdout_authorized") is False
        and evaluation_files == 0
    )
    checks["blind_holdout_not_opened"] = _check(
        "PASS" if not blind_opened else "FAIL",
        evidence=[
            _evidence(blind_v1_path, repo),
            _evidence(blind_v2_path, repo),
            _evidence(prereg_path, repo),
        ],
        details={
            "legacy_status": blind_v1.get("status"),
            "v2_status": blind_v2.get("status"),
            "v3_authorized": prereg.get("blind_holdout_authorized"),
            "07_evaluation_file_count": evaluation_files,
        },
    )
    candidate_not_claimed = prereg.get("candidate_effectiveness_claim") == "NOT_EVALUATED"
    checks["candidate_effectiveness_not_claimed"] = _check(
        "PASS" if candidate_not_claimed else "FAIL",
        evidence=[_evidence(prereg_path, repo)],
        details={"claim": prereg.get("candidate_effectiveness_claim")},
    )

    nonblocking = {
        "expert_source_level_audit",
        "legacy_formal_literature_audit",
    }
    controllable_failures = [
        check_id
        for check_id, item in checks.items()
        if check_id not in nonblocking and item["status"] != "PASS"
    ]
    blocking_conditions: list[dict[str, Any]] = []
    if source_missing_exact:
        blocking_conditions.append(
            {
                "code": "REPORT_ONLY_SOURCE_MISSING",
                "artifacts": missing_sources,
            }
        )
    if controllable_failures:
        blocking_conditions.append(
            {"code": "CONTROLLABLE_CHECKS_FAILED", "checks": controllable_failures}
        )

    if controllable_failures:
        status = "INCOMPLETE_EVIDENCE"
    elif source_missing_exact:
        status = "INCOMPLETE_SOURCE_MISSING"
    else:
        status = "PASS"
    completion_allowed = status == "PASS"

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment.name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "GLOBAL_RESEARCH_FINALIZATION",
        "status": status,
        "completion_allowed": completion_allowed,
        "candidate_effectiveness_claim": prereg.get(
            "candidate_effectiveness_claim", "NOT_EVALUATED"
        ),
        "literature_policy": {
            "current_scope": "USER_DECLARED_SUFFICIENT_STOP_EXPANSION",
            "legacy_500_300_100_audit_status": legacy_literature.get("status"),
            "legacy_status_relabelled_pass": False,
        },
        "prohibited_actions": {
            "formal_training_started": formal_training_started,
            "active_v3_engineering_gate_generated": active_gate_or_assignments,
            "active_v3_assignments_generated": active_gate_or_assignments,
            "blind_holdout_opened": blind_opened,
            "historical_gate_assets_detected": historical_gate_detected,
        },
        "blocking_conditions": blocking_conditions,
        "checks": checks,
    }


__all__ = ["EXPECTED_MISSING_SOURCES", "build_global_completion_audit"]
