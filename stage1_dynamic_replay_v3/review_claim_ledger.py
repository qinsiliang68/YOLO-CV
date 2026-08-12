from __future__ import annotations

import csv
import codecs
import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class ReviewLedgerError(ValueError):
    pass


_ID_PATTERN = re.compile(r"^(P0|H|M)-(\d{2})$")
_SEVERITY_BY_PREFIX = {
    "P0": "P0",
    "H": "High",
    "M": "Moderate",
}
_ASSESSMENT_STATUSES = {
    "CONFIRMED_ABSENT",
    "PARTIALLY_MITIGATED",
    "CONFIRMED_PRESENT",
    "NOT_APPLICABLE_CAPABILITY_ABSENT",
    "UNVERIFIED_SOURCE_MISSING",
}
_SOURCE_REF_PATTERN = re.compile(r"^(?P<path>[^:]+):(?P<start>\d+)(?:-(?P<end>\d+))?$")


@dataclass(frozen=True)
class ExpertFinding:
    finding_id: str
    severity: str
    title: str
    locations: tuple[str, ...]
    impact: str
    impact_supplied: bool
    evidence: str
    evidence_supplied: bool
    required_fix: str
    source_path: Path
    source_line: int


def _required_text(item: Mapping[str, object], field: str, finding_id: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ReviewLedgerError(f"{finding_id}: {field} must be non-empty text")
    return value.strip()


def _optional_narrative(
    item: Mapping[str, object], field: str, finding_id: str
) -> tuple[str, bool]:
    value = item.get(field)
    if value is None or value == "":
        return "NOT_SUPPLIED_IN_EXPERT_JSON", False
    if not isinstance(value, str):
        raise ReviewLedgerError(f"{finding_id}: {field} must be text when supplied")
    stripped = value.strip()
    if not stripped:
        return "NOT_SUPPLIED_IN_EXPERT_JSON", False
    return stripped, True


def _source_line(lines: Sequence[str], finding_id: str) -> int:
    needle = re.compile(rf'"id"\s*:\s*"{re.escape(finding_id)}"')
    matches = [index for index, line in enumerate(lines, start=1) if needle.search(line)]
    if len(matches) != 1:
        raise ReviewLedgerError(
            f"{finding_id}: expected exactly one source identity line, found {len(matches)}"
        )
    return matches[0]


def load_expert_findings(path: Path) -> list[ExpertFinding]:
    source_path = Path(path).resolve()
    text = source_path.read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict) or not isinstance(payload.get("findings"), list):
        raise ReviewLedgerError("expert findings JSON must contain a findings list")

    source_lines = text.splitlines()
    findings: list[ExpertFinding] = []
    seen: set[str] = set()
    for raw in payload["findings"]:
        if not isinstance(raw, dict):
            raise ReviewLedgerError("each finding must be an object")
        finding_id = _required_text(raw, "id", "UNKNOWN")
        match = _ID_PATTERN.fullmatch(finding_id)
        if match is None:
            raise ReviewLedgerError(f"invalid finding ID: {finding_id}")
        if finding_id in seen:
            raise ReviewLedgerError(f"duplicate finding ID: {finding_id}")
        seen.add(finding_id)

        severity = _required_text(raw, "severity", finding_id)
        expected_severity = _SEVERITY_BY_PREFIX[match.group(1)]
        if severity != expected_severity:
            raise ReviewLedgerError(
                f"{finding_id}: severity {severity!r} does not match {expected_severity!r}"
            )

        locations = raw.get("locations")
        if (
            not isinstance(locations, list)
            or not locations
            or not all(isinstance(item, str) and item.strip() for item in locations)
        ):
            raise ReviewLedgerError(f"{finding_id}: locations must be non-empty text list")

        impact, impact_supplied = _optional_narrative(raw, "impact", finding_id)
        evidence, evidence_supplied = _optional_narrative(raw, "evidence", finding_id)
        findings.append(
            ExpertFinding(
                finding_id=finding_id,
                severity=severity,
                title=_required_text(raw, "title", finding_id),
                locations=tuple(item.strip() for item in locations),
                impact=impact,
                impact_supplied=impact_supplied,
                evidence=evidence,
                evidence_supplied=evidence_supplied,
                required_fix=_required_text(raw, "required_fix", finding_id),
                source_path=source_path,
                source_line=_source_line(source_lines, finding_id),
            )
        )
    return findings


def write_expert_findings_ledger(
    findings: Iterable[ExpertFinding],
    output_path: Path,
) -> None:
    rows = list(findings)
    fieldnames = [
        "finding_id",
        "severity",
        "title",
        "expert_locations",
        "impact",
        "impact_supplied",
        "expert_evidence",
        "expert_evidence_supplied",
        "expert_required_fix",
        "source_json",
        "source_line",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for finding in rows:
        writer.writerow(
            {
                "finding_id": finding.finding_id,
                "severity": finding.severity,
                "title": finding.title,
                "expert_locations": "; ".join(finding.locations),
                "impact": finding.impact,
                "impact_supplied": str(finding.impact_supplied).lower(),
                "expert_evidence": finding.evidence,
                "expert_evidence_supplied": str(finding.evidence_supplied).lower(),
                "expert_required_fix": finding.required_fix,
                "source_json": finding.source_path.as_posix(),
                "source_line": finding.source_line,
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(buffer.getvalue(), encoding="utf-8-sig", newline="")
    temporary.replace(output)


def _validate_source_ref(reference: str, repo_root: Path) -> None:
    match = _SOURCE_REF_PATTERN.fullmatch(reference.strip())
    if match is None:
        raise ReviewLedgerError(f"invalid v3 source reference: {reference!r}")
    relative = Path(match.group("path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ReviewLedgerError(f"v3 source reference must be repo-relative: {reference!r}")
    source = repo_root / relative
    if not source.is_file():
        raise ReviewLedgerError(f"v3 source reference does not exist: {reference!r}")
    line_count = len(source.read_text(encoding="utf-8").splitlines())
    start = int(match.group("start"))
    end = int(match.group("end") or start)
    if start < 1 or end < start or end > line_count:
        raise ReviewLedgerError(
            f"v3 source reference is outside file bounds ({line_count} lines): {reference!r}"
        )


def validate_v3_assessments(
    findings: Sequence[ExpertFinding],
    assessments: Sequence[Mapping[str, str]],
    *,
    repo_root: Path,
) -> None:
    expected_ids = {finding.finding_id for finding in findings}
    observed_ids = [row.get("finding_id", "").strip() for row in assessments]
    if len(observed_ids) != len(set(observed_ids)):
        raise ReviewLedgerError("duplicate assessment IDs")
    if set(observed_ids) != expected_ids:
        raise ReviewLedgerError(
            "assessment IDs must exactly match expert finding IDs; "
            f"missing={sorted(expected_ids - set(observed_ids))}, "
            f"extra={sorted(set(observed_ids) - expected_ids)}"
        )

    required_fields = (
        "v3_status",
        "v3_source_refs",
        "local_reproduction",
        "observed_result",
        "remaining_risk",
        "required_action",
    )
    root = Path(repo_root).resolve()
    for row in assessments:
        finding_id = row["finding_id"].strip()
        for field in required_fields:
            if not row.get(field, "").strip():
                raise ReviewLedgerError(f"{finding_id}: missing assessment field {field}")
        status = row["v3_status"].strip()
        if status not in _ASSESSMENT_STATUSES:
            raise ReviewLedgerError(f"{finding_id}: unsupported v3 status {status!r}")
        for reference in row["v3_source_refs"].split(";"):
            _validate_source_ref(reference, root)


def validate_line_evidence_matrix(
    rows: Sequence[Mapping[str, str]],
    *,
    expected_ids: Sequence[str],
    repo_root: Path,
    reference_fields: Sequence[str],
) -> None:
    expected = list(expected_ids)
    observed = [row.get("finding_id", "").strip() for row in rows]
    if observed != expected or len(observed) != len(set(observed)):
        raise ReviewLedgerError(
            f"evidence matrix IDs must exactly match expected order: {observed} != {expected}"
        )
    root = Path(repo_root).resolve()
    for row in rows:
        finding_id = row["finding_id"].strip()
        status = row.get("status", "").strip()
        if status not in _ASSESSMENT_STATUSES:
            raise ReviewLedgerError(f"{finding_id}: unsupported matrix status {status!r}")
        for field in reference_fields:
            value = row.get(field, "").strip()
            if not value:
                raise ReviewLedgerError(f"{finding_id}: missing reference field {field}")
            for reference in value.split(";"):
                _validate_source_ref(reference, root)


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ReviewLedgerError(f"review source is outside repo root: {path}") from exc


def _publish_immutable_csv(output_path: Path, buffer: io.StringIO) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = codecs.BOM_UTF8 + buffer.getvalue().encode("utf-8")
    digest = hashlib.sha256(content).hexdigest().upper()
    if output.exists():
        if output.read_bytes() == content:
            return digest
        raise ReviewLedgerError(f"immutable review ledger already differs: {output}")
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def write_merged_review_ledger(
    findings: Sequence[ExpertFinding],
    assessments: Sequence[Mapping[str, str]],
    output_path: Path,
    *,
    repo_root: Path,
) -> str:
    validate_v3_assessments(findings, assessments, repo_root=repo_root)
    by_id = {row["finding_id"].strip(): row for row in assessments}
    fieldnames = [
        "finding_id",
        "severity",
        "title",
        "expert_locations",
        "impact",
        "impact_supplied",
        "expert_evidence",
        "expert_evidence_supplied",
        "expert_required_fix",
        "source_json",
        "source_line",
        "v3_status",
        "v3_source_refs",
        "local_reproduction",
        "observed_result",
        "remaining_risk",
        "required_action",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for finding in findings:
        assessment = by_id[finding.finding_id]
        writer.writerow(
            {
                "finding_id": finding.finding_id,
                "severity": finding.severity,
                "title": finding.title,
                "expert_locations": "; ".join(finding.locations),
                "impact": finding.impact,
                "impact_supplied": str(finding.impact_supplied).lower(),
                "expert_evidence": finding.evidence,
                "expert_evidence_supplied": str(finding.evidence_supplied).lower(),
                "expert_required_fix": finding.required_fix,
                "source_json": _repo_relative(finding.source_path, Path(repo_root)),
                "source_line": finding.source_line,
                **{field: assessment[field].strip() for field in fieldnames[11:]},
            }
        )
    return _publish_immutable_csv(Path(output_path), buffer)
