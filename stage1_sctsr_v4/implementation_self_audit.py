from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .baseline_reference import TASKBOOK_BLOB_SHA
from .errors import ErrorCode, SctsrError
from .serialization import load_json, sha256_file, stable_digest


SELF_AUDIT_SCHEMA = "stage1.sctsr.implementation_self_audit.v1"
SELF_AUDIT_PASS = "SELF_AUDIT_PASS_IMPLEMENTATION_ONLY"
SELF_AUDIT_FAIL = "SELF_AUDIT_FAIL"
SELF_REVIEW_IDENTITY = "SELF_REVIEW_NOT_INDEPENDENT_REVIEW"
ALLOWED_STATUSES = ("PASS", "FAIL", "BLOCKED_WITH_REASON")
SIDE_EFFECT_FIELDS = (
    "formal_training_started",
    "engineering_gate_generated",
    "assignments_generated",
    "pilot_release_generated",
    "blind_holdout_opened",
    "selector_trained",
    "method_effectiveness_claimed",
    "val_target_available",
    "synthetic_registered_as_scientific",
)
LEGACY_FIELDS = (
    "legacy_engineering_gate_detected",
    "legacy_pilot_release_detected",
    "legacy_assignments_detected",
)
CHECK_FIELDS = {
    "check_id",
    "status",
    "requirement",
    "taskbook_line",
    "evidence_paths",
    "reproduction_command",
    "exit_code",
    "stdout_log_path",
    "stdout_log_bytes",
    "stdout_log_sha256",
    "stderr_log_path",
    "stderr_log_bytes",
    "stderr_log_sha256",
    "observed_result",
    "expected_result",
    "reviewed_source_files",
    "reviewed_test_files",
    "remaining_risk",
    "required_action_if_not_pass",
}
AUDIT_FIELDS = {
    "schema_version",
    "taskbook_path",
    "taskbook_bytes",
    "taskbook_sha256",
    "taskbook_blob_sha",
    "implementation_source_commit",
    "generated_at_utc",
    "generated_by",
    "reviewer_independence_claim",
    "statuses_allowed",
    "applicable_check_count",
    "pass_count",
    "fail_count",
    "blocked_count",
    "overall_status",
    "side_effects",
    "legacy_detected",
    "checks",
    "audit_digest",
}
_CHECK_PATTERN = re.compile(r"^- \[ \] (SA-\d{3})：(.*)$")
_FORBIDDEN_PLACEHOLDER = re.compile(r"(?:\bTODO\b|\bTBD\b|\bunknown\b|待补|同上|空字段)", re.IGNORECASE)


def parse_taskbook_self_audit(taskbook_path: str | Path) -> tuple[dict[str, Any], ...]:
    path = Path(taskbook_path).resolve()
    if not path.is_file():
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "SCTSR taskbook is missing", artifact_path=str(path))
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "SCTSR taskbook is not strict UTF-8") from exc
    requirements: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, 1):
        match = _CHECK_PATTERN.fullmatch(line)
        if match is None:
            continue
        check_id, requirement = match.groups()
        requirement = requirement.strip()
        if check_id in seen or not requirement:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Taskbook self-audit ID is duplicate or empty", observed=check_id)
        seen.add(check_id)
        requirements.append({"check_id": check_id, "requirement": requirement, "taskbook_line": line_number})
    if not requirements:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Taskbook contains no machine-auditable SA requirements")
    return tuple(requirements)


def _resolve_registered_file(root: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Self-audit contains an empty evidence path", failing_field=field)
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Self-audit evidence escapes its registered root", failing_field=field, artifact_path=str(path)) from exc
    if not path.is_file():
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Self-audit evidence file is missing", failing_field=field, artifact_path=str(path))
    return path


def _require_nonempty_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or _FORBIDDEN_PLACEHOLDER.search(value):
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Self-audit text is empty or contains a forbidden placeholder", failing_field=field)
    return value


def validate_implementation_self_audit(
    audit: Mapping[str, Any] | str | Path,
    *,
    taskbook_path: str | Path,
    evidence_root: str | Path,
) -> dict[str, Any]:
    raw: Any = load_json(audit) if not isinstance(audit, Mapping) else audit
    if not isinstance(raw, Mapping) or set(raw) != AUDIT_FIELDS:
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Implementation self-audit top-level schema is invalid")
    core = {key: value for key, value in raw.items() if key != "audit_digest"}
    if raw.get("schema_version") != SELF_AUDIT_SCHEMA or raw.get("audit_digest") != stable_digest(core):
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Implementation self-audit digest is invalid")
    taskbook = Path(taskbook_path).resolve()
    if (
        raw.get("taskbook_bytes") != taskbook.stat().st_size
        or raw.get("taskbook_sha256") != sha256_file(taskbook)
        or raw.get("taskbook_blob_sha") != TASKBOOK_BLOB_SHA
    ):
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Implementation self-audit binds a different taskbook")
    recorded_taskbook = Path(str(raw.get("taskbook_path", ""))).resolve()
    if recorded_taskbook != taskbook:
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Implementation self-audit taskbook path differs")
    commit = str(raw.get("implementation_source_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Implementation self-audit source commit is invalid")
    try:
        datetime.fromisoformat(str(raw.get("generated_at_utc", "")).replace("Z", "+00:00"))
    except ValueError as exc:
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Implementation self-audit timestamp is invalid") from exc
    _require_nonempty_text(raw.get("generated_by"), field="generated_by")
    if raw.get("reviewer_independence_claim") != SELF_REVIEW_IDENTITY:
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Self-audit improperly claims independent review")
    if raw.get("statuses_allowed") != list(ALLOWED_STATUSES):
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Self-audit status registry is invalid")
    side_effects = raw.get("side_effects")
    if not isinstance(side_effects, Mapping) or tuple(side_effects) != SIDE_EFFECT_FIELDS or any(value is not False for value in side_effects.values()):
        raise SctsrError(ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED, "Self-audit detected or obscured prohibited v4 side effects")
    legacy = raw.get("legacy_detected")
    if not isinstance(legacy, Mapping) or tuple(legacy) != LEGACY_FIELDS or any(type(value) is not bool for value in legacy.values()):
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Self-audit legacy detection schema is invalid")

    requirements = parse_taskbook_self_audit(taskbook)
    expected_by_id = {row["check_id"]: row for row in requirements}
    checks = raw.get("checks")
    if not isinstance(checks, list) or len(checks) != len(requirements):
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Self-audit does not contain exactly one row per taskbook SA check")
    root = Path(evidence_root).resolve()
    observed_ids: list[str] = []
    statuses: list[str] = []
    for index, row in enumerate(checks):
        if not isinstance(row, Mapping) or set(row) != CHECK_FIELDS:
            raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Self-audit check row schema is invalid", failing_field=f"checks[{index}]")
        check_id = str(row["check_id"])
        expected = expected_by_id.get(check_id)
        if expected is None or check_id in observed_ids:
            raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Self-audit check ID is missing, extra, or duplicate", observed=check_id)
        observed_ids.append(check_id)
        if row["requirement"] != expected["requirement"] or row["taskbook_line"] != expected["taskbook_line"]:
            raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Self-audit requirement text or taskbook line drifted", failing_field=check_id)
        status = str(row["status"])
        if status not in ALLOWED_STATUSES:
            raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Self-audit check status is invalid", failing_field=check_id)
        statuses.append(status)
        if type(row["exit_code"]) is not int or (status == "PASS" and row["exit_code"] != 0):
            raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "PASS self-audit check lacks a successful reproduction command", failing_field=check_id)
        for field in (
            "reproduction_command",
            "observed_result",
            "expected_result",
            "remaining_risk",
            "required_action_if_not_pass",
        ):
            _require_nonempty_text(row[field], field=f"{check_id}.{field}")
        for field in ("evidence_paths", "reviewed_source_files", "reviewed_test_files"):
            values = row[field]
            if not isinstance(values, list) or not values:
                raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Self-audit evidence list is empty", failing_field=f"{check_id}.{field}")
            for item_index, value in enumerate(values):
                _resolve_registered_file(root, value, field=f"{check_id}.{field}[{item_index}]")
        for stream in ("stdout", "stderr"):
            log = _resolve_registered_file(root, row[f"{stream}_log_path"], field=f"{check_id}.{stream}_log_path")
            expected_bytes = row[f"{stream}_log_bytes"]
            expected_sha = row[f"{stream}_log_sha256"]
            if type(expected_bytes) is not int or expected_bytes < 0 or log.stat().st_size != expected_bytes or sha256_file(log) != expected_sha:
                raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Self-audit log bytes or SHA mismatch", failing_field=f"{check_id}.{stream}")
    if observed_ids != [row["check_id"] for row in requirements]:
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Self-audit check order differs from the taskbook")
    counts = {
        "applicable_check_count": len(statuses),
        "pass_count": statuses.count("PASS"),
        "fail_count": statuses.count("FAIL"),
        "blocked_count": statuses.count("BLOCKED_WITH_REASON"),
    }
    if any(raw.get(field) != value for field, value in counts.items()):
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Self-audit status counts are inconsistent", observed=counts)
    expected_overall = SELF_AUDIT_PASS if counts["fail_count"] == counts["blocked_count"] == 0 else SELF_AUDIT_FAIL
    if raw.get("overall_status") != expected_overall:
        raise SctsrError(ErrorCode.CLOSEOUT_NOT_VALIDATED, "Self-audit overall status hides failed or blocked checks", expected=expected_overall)
    return {
        "status": "PASS" if expected_overall == SELF_AUDIT_PASS else "VALID_AUDIT_WITH_FAILURES",
        "overall_status": expected_overall,
        "check_count": len(statuses),
        "failed_check_ids": [row["check_id"] for row in checks if row["status"] == "FAIL"],
        "blocked_check_ids": [row["check_id"] for row in checks if row["status"] == "BLOCKED_WITH_REASON"],
        "audit_digest": raw["audit_digest"],
    }


def build_implementation_self_audit(
    *,
    taskbook_path: str | Path,
    implementation_source_commit: str,
    generated_at_utc: str,
    generated_by: str,
    checks: Sequence[Mapping[str, Any]],
    side_effects: Mapping[str, bool],
    legacy_detected: Mapping[str, bool],
) -> dict[str, Any]:
    taskbook = Path(taskbook_path).resolve()
    statuses = [str(row.get("status")) for row in checks]
    core = {
        "schema_version": SELF_AUDIT_SCHEMA,
        "taskbook_path": taskbook.as_posix(),
        "taskbook_bytes": taskbook.stat().st_size,
        "taskbook_sha256": sha256_file(taskbook),
        "taskbook_blob_sha": TASKBOOK_BLOB_SHA,
        "implementation_source_commit": implementation_source_commit,
        "generated_at_utc": generated_at_utc,
        "generated_by": generated_by,
        "reviewer_independence_claim": SELF_REVIEW_IDENTITY,
        "statuses_allowed": list(ALLOWED_STATUSES),
        "applicable_check_count": len(statuses),
        "pass_count": statuses.count("PASS"),
        "fail_count": statuses.count("FAIL"),
        "blocked_count": statuses.count("BLOCKED_WITH_REASON"),
        "overall_status": SELF_AUDIT_PASS if statuses.count("FAIL") == statuses.count("BLOCKED_WITH_REASON") == 0 else SELF_AUDIT_FAIL,
        "side_effects": dict(side_effects),
        "legacy_detected": dict(legacy_detected),
        "checks": [dict(row) for row in checks],
    }
    return {**core, "audit_digest": stable_digest(core)}
