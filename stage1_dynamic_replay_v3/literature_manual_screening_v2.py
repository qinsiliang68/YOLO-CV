"""Structural validation for human primary-source screening decisions."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
import re
from typing import Sequence


class ManualScreeningError(RuntimeError):
    pass


@dataclass(frozen=True)
class ManualScreeningResult:
    status: str
    reviewed_count: int
    eligible_count: int
    excluded_count: int
    rows: tuple[dict[str, str], ...]


ELIGIBLE_REQUIRED_SCOPE = {"TITLE", "ABSTRACT", "PROBLEM", "METHOD_OVERVIEW", "CONCLUSION"}
EXCLUDED_REQUIRED_SCOPE = {"TITLE", "ABSTRACT", "PROBLEM"}
ALLOWED_DECISIONS = {"ELIGIBLE_BROAD", "EXCLUDE"}
ALLOWED_RELEVANCE_CLASSES = {
    "DIRECT_INTERVENTION",
    "DIRECT_MECHANISM",
    "STRICT_CONTROL_NEGATIVE",
    "TARGET_METRIC",
    "TRANSFER_COMPONENT",
    "EXCLUDED",
}
ALLOWED_AUTHORITIES = {"PRIMARY_PUBLISHER", "AUTHOR_HOSTED", "OFFICIAL_REPOSITORY"}
ALLOWED_RQ_IDS = {f"RQ{index}" for index in range(1, 9)}
PROSE_FIELDS = (
    "problem_summary_zh",
    "method_overview_zh",
    "conclusion_summary_zh",
    "critical_review_zh",
    "stage1_transfer_zh",
    "cannot_infer_zh",
)
REQUIRED_COLUMNS = {
    "queue_id",
    "decision",
    "canonical_title",
    "primary_url_checked",
    "source_authority",
    "checked_at",
    "reading_scope",
    "direct_rq_ids",
    "relevance_class",
    *PROSE_FIELDS,
    "exclusion_reason",
    "reviewer",
}
BANNED_PLACEHOLDERS = re.compile(
    r"(?:\b(?:todo|tbd|unknown)\b|待补|待确认|同上|未阅读|未核对)",
    flags=re.IGNORECASE,
)


def blind_order_queue(
    rows: list[dict[str, str]], *, frozen_seed: str
) -> list[dict[str, str]]:
    """Return a deterministic review order independent of discovery ranking metadata."""

    if not frozen_seed.strip():
        raise ManualScreeningError("frozen_seed is required")
    ordered: list[dict[str, str]] = []
    identities: set[str] = set()
    for row in rows:
        identity = "|".join(
            (
                _normalize_title(str(row.get("title", ""))),
                str(row.get("doi", "")).strip().casefold(),
                _normalize_title(str(row.get("authors", ""))),
            )
        )
        if identity in identities:
            raise ManualScreeningError(f"duplicate blind-order identity: {identity}")
        identities.add(identity)
        blind_key = hashlib.sha256(f"{frozen_seed}|{identity}".encode("utf-8")).hexdigest().upper()
        output = dict(row)
        output["blind_order_key"] = blind_key
        ordered.append(output)
    ordered.sort(key=lambda row: (row["blind_order_key"], row["queue_id"]))
    for rank, row in enumerate(ordered, start=1):
        row["blind_review_rank"] = str(rank)
    return ordered


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ManualScreeningError(f"CSV has no header: {path}")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ManualScreeningError(f"{path.name} lacks columns: {sorted(missing)}")
        return [dict(row) for row in reader]


def _normalize_title(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _validate_timestamp(value: str, queue_id: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ManualScreeningError(f"invalid checked_at for {queue_id}: {value}") from exc
    if parsed.tzinfo is None:
        raise ManualScreeningError(f"checked_at lacks timezone for {queue_id}")


def _validate_prose(
    row: dict[str, str], *, decision: str, full_scope_claimed: bool
) -> None:
    if decision == "EXCLUDE" and not full_scope_claimed:
        for field in ("method_overview_zh", "conclusion_summary_zh", "stage1_transfer_zh"):
            value = row[field].strip()
            if not value.startswith("NOT_APPLICABLE_WITH_REASON:"):
                raise ManualScreeningError(
                    f"{row['queue_id']} {field} must disclose that full-scope reading was not claimed"
                )
        prose_fields = ("problem_summary_zh", "critical_review_zh", "cannot_infer_zh")
    else:
        prose_fields = PROSE_FIELDS
    normalized: list[str] = []
    for field in prose_fields:
        value = row[field].strip()
        if BANNED_PLACEHOLDERS.search(value):
            raise ManualScreeningError(f"placeholder in {row['queue_id']} {field}")
        if len(value) < 15:
            raise ManualScreeningError(f"{row['queue_id']} {field} is too short")
        normalized.append(re.sub(r"\s+", "", value).casefold())
    if len(normalized) != len(set(normalized)):
        raise ManualScreeningError(f"reused prose fields in {row['queue_id']}")


def _validate_row(row: dict[str, str], queue_row: dict[str, str]) -> None:
    queue_id = row["queue_id"].strip()
    decision = row["decision"].strip()
    if decision not in ALLOWED_DECISIONS:
        raise ManualScreeningError(f"invalid decision for {queue_id}: {decision}")
    if _normalize_title(row["canonical_title"]) != _normalize_title(queue_row["title"]):
        raise ManualScreeningError(f"title identity mismatch for {queue_id}")
    if not row["primary_url_checked"].strip().startswith("https://"):
        raise ManualScreeningError(f"non-HTTPS primary source for {queue_id}")
    if row["source_authority"].strip() not in ALLOWED_AUTHORITIES:
        raise ManualScreeningError(f"invalid source authority for {queue_id}")
    _validate_timestamp(row["checked_at"].strip(), queue_id)
    scope = {part.strip() for part in row["reading_scope"].split(";") if part.strip()}
    required_scope = (
        ELIGIBLE_REQUIRED_SCOPE if decision == "ELIGIBLE_BROAD" else EXCLUDED_REQUIRED_SCOPE
    )
    if not required_scope.issubset(scope):
        raise ManualScreeningError(f"incomplete reading scope for {queue_id}: {sorted(scope)}")
    relevance = row["relevance_class"].strip()
    if relevance not in ALLOWED_RELEVANCE_CLASSES:
        raise ManualScreeningError(f"invalid relevance class for {queue_id}: {relevance}")

    rq_value = row["direct_rq_ids"].strip()
    if decision == "ELIGIBLE_BROAD":
        rq_ids = {part.strip() for part in rq_value.split(";") if part.strip()}
        if not rq_ids or not rq_ids.issubset(ALLOWED_RQ_IDS):
            raise ManualScreeningError(f"invalid direct RQ mapping for {queue_id}: {rq_value}")
        if relevance == "EXCLUDED":
            raise ManualScreeningError(f"eligible row marked EXCLUDED for {queue_id}")
        if not row["exclusion_reason"].startswith("NOT_APPLICABLE_WITH_REASON:"):
            raise ManualScreeningError(f"eligible row lacks N/A exclusion reason for {queue_id}")
    else:
        if relevance != "EXCLUDED":
            raise ManualScreeningError(f"excluded row has non-excluded relevance class for {queue_id}")
        if not rq_value.startswith("NOT_APPLICABLE_WITH_REASON:"):
            raise ManualScreeningError(f"excluded row has an RQ claim for {queue_id}")
        reason = row["exclusion_reason"].strip()
        if len(reason) < 8 or BANNED_PLACEHOLDERS.search(reason):
            raise ManualScreeningError(f"invalid exclusion reason for {queue_id}")

    if not row["reviewer"].strip():
        raise ManualScreeningError(f"blank reviewer for {queue_id}")
    _validate_prose(
        row,
        decision=decision,
        full_scope_claimed=ELIGIBLE_REQUIRED_SCOPE.issubset(scope),
    )


def merge_and_validate_manual_screening(
    *,
    queue_path: Path,
    decision_dir: Path,
    decision_paths: Sequence[Path] | None = None,
) -> ManualScreeningResult:
    with queue_path.open("r", encoding="utf-8-sig", newline="") as handle:
        queue_rows = [dict(row) for row in csv.DictReader(handle)]
    if not queue_rows or not {"queue_id", "title"}.issubset(queue_rows[0]):
        raise ManualScreeningError("manual queue is empty or lacks queue_id/title")
    queue_by_id: dict[str, dict[str, str]] = {}
    for row in queue_rows:
        queue_id = row["queue_id"].strip()
        if queue_id in queue_by_id:
            raise ManualScreeningError(f"duplicate queue_id: {queue_id}")
        queue_by_id[queue_id] = row

    if decision_paths is None:
        selected_paths = sorted(decision_dir.glob("*.csv"))
    else:
        selected_paths = sorted(
            (Path(path) for path in decision_paths),
            key=lambda path: path.name.casefold(),
        )
        if len(selected_paths) != len({path.resolve() for path in selected_paths}):
            raise ManualScreeningError("duplicate explicit decision path")
        decision_root = decision_dir.resolve()
        for path in selected_paths:
            if path.suffix.casefold() != ".csv":
                raise ManualScreeningError(f"decision file is not CSV: {path}")
            if path.resolve().parent != decision_root:
                raise ManualScreeningError(
                    f"decision file is outside decision directory: {path}"
                )
            if not path.is_file():
                raise ManualScreeningError(f"decision file does not exist: {path}")
    if not selected_paths:
        raise ManualScreeningError(f"no decision CSV files in {decision_dir}")
    decisions: dict[str, dict[str, str]] = {}
    for path in selected_paths:
        for row in _read_csv(path):
            queue_id = row["queue_id"].strip()
            if queue_id in decisions:
                raise ManualScreeningError(f"duplicate manual decision: {queue_id}")
            decisions[queue_id] = row
    if set(decisions) != set(queue_by_id):
        missing = sorted(set(queue_by_id) - set(decisions))
        extra = sorted(set(decisions) - set(queue_by_id))
        raise ManualScreeningError(f"manual decision set mismatch: missing={missing} extra={extra}")

    ordered: list[dict[str, str]] = []
    for queue_id in sorted(queue_by_id):
        row = decisions[queue_id]
        _validate_row(row, queue_by_id[queue_id])
        ordered.append(row)
    eligible = sum(row["decision"] == "ELIGIBLE_BROAD" for row in ordered)
    return ManualScreeningResult(
        status="PASS",
        reviewed_count=len(ordered),
        eligible_count=eligible,
        excluded_count=len(ordered) - eligible,
        rows=tuple(ordered),
    )
