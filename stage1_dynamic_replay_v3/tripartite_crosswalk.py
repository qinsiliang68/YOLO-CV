"""Fail-closed schema validation for the Stage1 tripartite code crosswalk.

The validator inspects recorded evidence only.  In particular, it never executes
the reproduction commands carried by a crosswalk row.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
from typing import Any, Mapping, Sequence
import uuid


CROSSWALK_FIELDS = (
    "requirement_id",
    "overall_status",
    "expert_source_status",
    "expert_claim_refs",
    "expert_source_refs",
    "v3_status",
    "v3_source_refs",
    "reproduction_command",
    "exit_code",
    "result_artifact_sha",
    "observed_result",
    "remaining_risk",
    "required_action",
)

STATUS_ORDER = (
    "CONFIRMED_PRESENT",
    "CONFIRMED_ABSENT",
    "PARTIALLY_MITIGATED",
    "CONTRADICTED_BY_EVIDENCE",
    "NOT_TESTABLE_SOURCE_MISSING",
    "NOT_APPLICABLE",
)
ALLOWED_STATUSES = frozenset(STATUS_ORDER)
SOURCE_MISSING_REFERENCE = "NOT_APPLICABLE_SOURCE_MISSING"
VALIDATION_SCHEMA = "stage1.tripartite_crosswalk_validation.v1"

_LINE_REFERENCE = re.compile(
    r"^(?P<path>[^:\r\n]+):(?P<start>[1-9]\d*)(?:-(?P<end>[1-9]\d*))?$"
)
_SHA256 = re.compile(r"^[0-9A-Fa-f]{64}$")
_PYTEST_COMMAND = re.compile(
    r"^uv run pytest "
    r"(?P<node>[A-Za-z0-9_.\\/\-]+\.py::[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]\r\n]+\])?)"
    r"(?: -q)?$"
)
_FORBIDDEN_COMMAND_TOKEN = re.compile(
    r"[\r\n;&|><]|\b(?:TODO|TBD|UNKNOWN)\b|待补|同上|\{[^{}]*\}",
    re.IGNORECASE,
)
_PLACEHOLDER = re.compile(
    r"^(?:TODO|TBD|UNKNOWN|待补|同上)$",
    re.IGNORECASE,
)
_BUDGETED_SOURCE_ROLE = re.compile(r"^budgeted_replay_(?:source_|wheel$)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        rows = [
            {str(key): "" if value is None else str(value) for key, value in row.items() if key is not None}
            for row in reader
        ]
    return headers, rows


def _error(
    errors: list[dict[str, str]],
    *,
    code: str,
    matrix_role: str,
    requirement_id: str,
    field: str,
    message: str,
) -> None:
    errors.append(
        {
            "code": code,
            "matrix_role": matrix_role,
            "requirement_id": requirement_id,
            "field": field,
            "message": message,
        }
    )


def _row_identity(row: Mapping[str, str], row_number: int) -> str:
    return (
        row.get("requirement_id", "").strip()
        or row.get("finding_id", "").strip()
        or f"ROW_{row_number:04d}"
    )


def _inside_repo(path: Path, repo_root: Path) -> bool:
    try:
        path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    return True


def _validate_line_reference(
    reference: str,
    *,
    repo_root: Path,
) -> tuple[str, str] | None:
    match = _LINE_REFERENCE.fullmatch(reference.strip())
    if match is None:
        return "INVALID_LINE_REFERENCE", "reference must be repo-relative path:start[-end]"
    relative = Path(match.group("path"))
    if relative.is_absolute() or ".." in relative.parts:
        return "INVALID_LINE_REFERENCE", "reference path must stay inside the repository"
    source = (repo_root / relative).resolve()
    if not _inside_repo(source, repo_root):
        return "INVALID_LINE_REFERENCE", "reference path escapes the repository"
    if not source.is_file():
        return "LINE_REFERENCE_FILE_MISSING", f"referenced file does not exist: {relative.as_posix()}"
    try:
        line_count = len(source.read_text(encoding="utf-8").splitlines())
    except UnicodeDecodeError:
        return "LINE_REFERENCE_NOT_TEXT", f"referenced file is not UTF-8 text: {relative.as_posix()}"
    start = int(match.group("start"))
    end = int(match.group("end") or start)
    if end < start or end > line_count:
        return (
            "LINE_REFERENCE_OUT_OF_BOUNDS",
            f"reference {reference!r} exceeds {line_count} text lines",
        )
    return None


def _validate_reproduction_command(
    command: str,
    *,
    repo_root: Path,
) -> str | None:
    if _FORBIDDEN_COMMAND_TOKEN.search(command) or re.search(r"<[^<>]*>", command):
        return "command contains a shell operator, placeholder, or unresolved token"
    pytest_match = _PYTEST_COMMAND.fullmatch(command)
    if pytest_match is not None:
        test_path = Path(pytest_match.group("node").split("::", 1)[0])
        if test_path.is_absolute() or ".." in test_path.parts:
            return "pytest node path must be repository-relative"
        resolved = (repo_root / test_path).resolve()
        if not _inside_repo(resolved, repo_root) or not resolved.is_file():
            return "pytest node must name an existing repository test file"
        return None

    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return "command quoting is invalid"
    if len(tokens) == 4 and tokens[:2] == ["rg", "-n"]:
        target = Path(tokens[-1])
        if target.is_absolute() or ".." in target.parts or any(char in tokens[-1] for char in "*?[]"):
            return "rg target must be one exact repository-relative file"
        resolved = (repo_root / target).resolve()
        if not _inside_repo(resolved, repo_root) or not resolved.is_file():
            return "rg target must name an existing repository file"
        return None
    return "only an exact uv run pytest node or rg -n query is permitted"


def _budgeted_source_missing(
    inventory_path: Path,
    errors: list[dict[str, str]],
) -> tuple[bool | None, dict[str, Any]]:
    summary: dict[str, Any] = {
        "path": str(inventory_path.resolve()),
        "sha256": "",
        "row_count": 0,
    }
    if not inventory_path.is_file():
        _error(
            errors,
            code="EXPERT_INVENTORY_MISSING",
            matrix_role="inventory",
            requirement_id="N/A",
            field="expert_inventory",
            message=f"expert inventory does not exist: {inventory_path}",
        )
        return None, summary
    summary["sha256"] = _sha256(inventory_path)
    try:
        headers, rows = _read_csv(inventory_path)
    except (OSError, csv.Error, UnicodeError) as exc:
        _error(
            errors,
            code="EXPERT_INVENTORY_UNREADABLE",
            matrix_role="inventory",
            requirement_id="N/A",
            field="expert_inventory",
            message=str(exc),
        )
        return None, summary
    summary["row_count"] = len(rows)
    required_headers = {"evidence_role", "required_source", "status"}
    missing_headers = sorted(required_headers - set(headers))
    if missing_headers:
        _error(
            errors,
            code="EXPERT_INVENTORY_SCHEMA_INVALID",
            matrix_role="inventory",
            requirement_id="N/A",
            field="expert_inventory",
            message=f"missing inventory fields: {missing_headers}",
        )
        return None, summary
    budgeted = [
        row
        for row in rows
        if row.get("required_source", "").strip().lower() == "true"
        and _BUDGETED_SOURCE_ROLE.match(row.get("evidence_role", "").strip())
    ]
    if not budgeted:
        _error(
            errors,
            code="BUDGETED_SOURCE_INVENTORY_MISSING",
            matrix_role="inventory",
            requirement_id="N/A",
            field="expert_inventory",
            message="inventory has no required BudgetedReplay source carrier",
        )
        return None, summary
    summary["budgeted_required_source_rows"] = len(budgeted)
    summary["budgeted_source_statuses"] = sorted(
        {row.get("status", "").strip() for row in budgeted}
    )
    return any(
        row.get("status", "").strip() == "REPORT_ONLY_SOURCE_MISSING"
        for row in budgeted
    ), summary


def _validate_matrix(
    *,
    role: str,
    path: Path,
    expected_rows: int,
    repo_root: Path,
    budgeted_missing: bool | None,
    errors: list[dict[str, str]],
    observed_requirement_ids: dict[str, tuple[str, int]],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    summary: dict[str, Any] = {
        "matrix_role": role,
        "path": str(path.resolve()),
        "sha256": "",
        "expected_rows": expected_rows,
        "row_count": 0,
        "headers": [],
    }
    if not path.is_file():
        _error(
            errors,
            code="MATRIX_MISSING",
            matrix_role=role,
            requirement_id="N/A",
            field="matrix",
            message=f"matrix does not exist: {path}",
        )
        return summary, []
    summary["sha256"] = _sha256(path)
    try:
        headers, rows = _read_csv(path)
    except (OSError, csv.Error, UnicodeError) as exc:
        _error(
            errors,
            code="MATRIX_UNREADABLE",
            matrix_role=role,
            requirement_id="N/A",
            field="matrix",
            message=str(exc),
        )
        return summary, []
    summary["headers"] = headers
    summary["row_count"] = len(rows)
    if headers != list(CROSSWALK_FIELDS):
        _error(
            errors,
            code="SCHEMA_FIELDS_MISMATCH",
            matrix_role=role,
            requirement_id="N/A",
            field="header",
            message=(
                f"expected exact fields {list(CROSSWALK_FIELDS)!r}; observed {headers!r}"
            ),
        )
    if len(rows) != expected_rows:
        _error(
            errors,
            code="ROW_COUNT_MISMATCH",
            matrix_role=role,
            requirement_id="N/A",
            field="row_count",
            message=f"expected {expected_rows} rows; observed {len(rows)}",
        )

    for row_number, row in enumerate(rows, start=2):
        requirement_id = _row_identity(row, row_number)
        for field in CROSSWALK_FIELDS:
            value = row.get(field, "").strip()
            if not value:
                _error(
                    errors,
                    code="MISSING_FIELD",
                    matrix_role=role,
                    requirement_id=requirement_id,
                    field=field,
                    message=f"row {row_number} has no non-empty {field}",
                )

        canonical_id = row.get("requirement_id", "").strip()
        if canonical_id:
            previous = observed_requirement_ids.get(canonical_id)
            if previous is not None:
                _error(
                    errors,
                    code="DUPLICATE_REQUIREMENT_ID",
                    matrix_role=role,
                    requirement_id=canonical_id,
                    field="requirement_id",
                    message=(
                        f"requirement_id duplicates {previous[0]} row {previous[1]}"
                    ),
                )
            else:
                observed_requirement_ids[canonical_id] = (role, row_number)

        for status_field in ("overall_status", "expert_source_status", "v3_status"):
            status = row.get(status_field, "").strip()
            if status and status not in ALLOWED_STATUSES:
                _error(
                    errors,
                    code="UNSUPPORTED_STATUS",
                    matrix_role=role,
                    requirement_id=requirement_id,
                    field=status_field,
                    message=f"unsupported status {status!r}",
                )
        legacy_status = row.get("status", "").strip()
        if not row.get("v3_status", "").strip() and legacy_status:
            if legacy_status not in ALLOWED_STATUSES:
                _error(
                    errors,
                    code="UNSUPPORTED_STATUS",
                    matrix_role=role,
                    requirement_id=requirement_id,
                    field="status",
                    message=f"unsupported legacy status {legacy_status!r}",
                )

        source_status = row.get("expert_source_status", "").strip()
        source_refs = row.get("expert_source_refs", "").strip()
        if role == "budgeted" and budgeted_missing is True:
            if source_status != "NOT_TESTABLE_SOURCE_MISSING":
                _error(
                    errors,
                    code="BUDGETED_SOURCE_MISSING_STATUS_REQUIRED",
                    matrix_role=role,
                    requirement_id=requirement_id,
                    field="expert_source_status",
                    message=(
                        "BudgetedReplay source carrier is report-only missing; "
                        "expert_source_status must be NOT_TESTABLE_SOURCE_MISSING"
                    ),
                )
            if source_refs != SOURCE_MISSING_REFERENCE:
                _error(
                    errors,
                    code="SOURCE_MISSING_REFERENCE_REQUIRED",
                    matrix_role=role,
                    requirement_id=requirement_id,
                    field="expert_source_refs",
                    message=(
                        f"missing source must use the exact token {SOURCE_MISSING_REFERENCE!r}"
                    ),
                )

        for reference_field in (
            "expert_claim_refs",
            "expert_source_refs",
            "v3_source_refs",
        ):
            raw_references = row.get(reference_field, "").strip()
            if not raw_references:
                continue
            if raw_references == SOURCE_MISSING_REFERENCE:
                if not (
                    reference_field == "expert_source_refs"
                    and source_status == "NOT_TESTABLE_SOURCE_MISSING"
                ):
                    _error(
                        errors,
                        code="INVALID_SOURCE_MISSING_REFERENCE",
                        matrix_role=role,
                        requirement_id=requirement_id,
                        field=reference_field,
                        message="source-missing token is legal only for missing expert source",
                    )
                continue
            for reference in raw_references.split(";"):
                failure = _validate_line_reference(reference, repo_root=repo_root)
                if failure is not None:
                    code, message = failure
                    _error(
                        errors,
                        code=code,
                        matrix_role=role,
                        requirement_id=requirement_id,
                        field=reference_field,
                        message=message,
                    )

        command = row.get("reproduction_command", "").strip()
        if command:
            command_error = _validate_reproduction_command(command, repo_root=repo_root)
            if command_error is not None:
                _error(
                    errors,
                    code="INVALID_REPRODUCTION_COMMAND",
                    matrix_role=role,
                    requirement_id=requirement_id,
                    field="reproduction_command",
                    message=command_error,
                )

        exit_code = row.get("exit_code", "").strip()
        if exit_code:
            try:
                parsed_exit_code = int(exit_code)
            except ValueError:
                parsed_exit_code = -1
            if str(parsed_exit_code) != exit_code or not 0 <= parsed_exit_code <= 255:
                _error(
                    errors,
                    code="INVALID_EXIT_CODE",
                    matrix_role=role,
                    requirement_id=requirement_id,
                    field="exit_code",
                    message="exit_code must be a canonical integer from 0 through 255",
                )

        result_sha = row.get("result_artifact_sha", "").strip()
        if result_sha and _SHA256.fullmatch(result_sha) is None:
            _error(
                errors,
                code="INVALID_SHA256",
                matrix_role=role,
                requirement_id=requirement_id,
                field="result_artifact_sha",
                message="result_artifact_sha must be exactly 64 hexadecimal characters",
            )

        for narrative_field in ("observed_result", "remaining_risk", "required_action"):
            value = row.get(narrative_field, "").strip()
            if value and _PLACEHOLDER.fullmatch(value):
                _error(
                    errors,
                    code="PLACEHOLDER_VALUE",
                    matrix_role=role,
                    requirement_id=requirement_id,
                    field=narrative_field,
                    message=f"placeholder {value!r} is not evidence",
                )
    return summary, rows


def validate_tripartite_crosswalks(
    *,
    repo_root: str | Path,
    budgeted_matrix: str | Path,
    dynamic_matrix: str | Path,
    expert_inventory: str | Path,
    expected_budgeted_rows: int = 31,
    expected_dynamic_rows: int = 15,
) -> dict[str, Any]:
    """Validate two crosswalk matrices without executing any recorded command."""

    root = Path(repo_root).resolve()
    errors: list[dict[str, str]] = []
    budgeted_missing, inventory_summary = _budgeted_source_missing(
        Path(expert_inventory), errors
    )
    requirement_ids: dict[str, tuple[str, int]] = {}
    summaries: list[dict[str, Any]] = []
    total_rows = 0
    for role, matrix, expected in (
        ("budgeted", Path(budgeted_matrix), expected_budgeted_rows),
        ("dynamic", Path(dynamic_matrix), expected_dynamic_rows),
    ):
        summary, rows = _validate_matrix(
            role=role,
            path=matrix,
            expected_rows=expected,
            repo_root=root,
            budgeted_missing=budgeted_missing,
            errors=errors,
            observed_requirement_ids=requirement_ids,
        )
        summaries.append(summary)
        total_rows += len(rows)

    return {
        "schema_version": VALIDATION_SCHEMA,
        "status": "PASS" if not errors else "FAIL",
        "validation_scope": "TRIPARTITE_CROSSWALK_ONLY",
        "scientific_result": False,
        "reproduction_commands_executed": False,
        "repo_root": str(root),
        "contract": {
            "required_fields": list(CROSSWALK_FIELDS),
            "allowed_statuses": list(STATUS_ORDER),
            "source_missing_reference": SOURCE_MISSING_REFERENCE,
            "reproduction_command_policy": (
                "exact uv run pytest node or exact rg -n query; validation never executes it"
            ),
            "result_artifact_sha_policy": "64 hexadecimal characters",
        },
        "budgeted_source_missing": budgeted_missing,
        "expert_inventory": inventory_summary,
        "input_matrices": summaries,
        "expected_total_rows": expected_budgeted_rows + expected_dynamic_rows,
        "total_rows": total_rows,
        "error_count": len(errors),
        "errors": errors,
    }


def write_validation_report(report: Mapping[str, Any], output_path: str | Path) -> None:
    """Atomically replace the reproducible validation report."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "ALLOWED_STATUSES",
    "CROSSWALK_FIELDS",
    "SOURCE_MISSING_REFERENCE",
    "validate_tripartite_crosswalks",
    "write_validation_report",
]
