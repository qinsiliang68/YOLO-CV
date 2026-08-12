"""Fail-closed validation for full-text SCREENED literature reviews.

The review records bind human-written method and experiment notes to immutable
PDF/text artifacts and exact page/line anchors.  Passing this module is still an
intermediate evidence gate; it does not by itself grant formal SCREENED credit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


class ScreenedReviewError(RuntimeError):
    """Raised when a SCREENED review cannot be tied to its primary full text."""

    def __init__(self, errors: str | Sequence[str]) -> None:
        values = (errors,) if isinstance(errors, str) else tuple(str(item) for item in errors)
        self.errors = values
        super().__init__("SCREENED review validation failed:\n- " + "\n- ".join(values))


@dataclass(frozen=True)
class ScreenedReviewResult:
    status: str
    reviewed_count: int
    eligible_count: int
    excluded_count: int
    records: tuple[Mapping[str, Any], ...]


REQUIRED_SECTIONS = {"METHODS", "EXPERIMENTS", "ABLATIONS", "LIMITATIONS"}
ALLOWED_SECTION_STATUS = {"READ", "NOT_REPORTED_BY_PAPER"}
ALLOWED_DECISIONS = {"SCREENED_ELIGIBLE", "EXCLUDE_SCREENED"}
ALLOWED_TRANSFER_CLASSES = {
    "REPLICATION",
    "INSPIRED_ADAPTATION",
    "MECHANISM_ONLY",
    "NOT_TRANSFERABLE",
}
ALLOWED_RQ_IDS = {f"RQ{index}" for index in range(1, 9)}
ALLOWED_MISSING = {"NOT_REPORTED_BY_PAPER"}
PLACEHOLDER_RE = re.compile(
    r"(?i)(?:^|\b)(?:TODO|TBD|UNKNOWN)(?:$|\b)|待补|待确认|同上|未阅读|未核对"
)

REQUIRED_RECORD_FIELDS = {
    "schema_version",
    "paper_id",
    "canonical_work_id",
    "title",
    "decision",
    "reviewed_at",
    "reviewer",
    "method_source",
    "text_source",
    "rq_ids",
    "section_evidence",
    "formulas",
    "algorithm_steps",
    "variables",
    "selection_timing",
    "refresh_rule",
    "budget",
    "random_baselines",
    "datasets",
    "models",
    "seed_count",
    "checkpoint_selection",
    "results",
    "ablations",
    "negative_results",
    "failure_conditions",
    "limitations",
    "transfer_class",
    "stage1_mechanism_zh",
    "stage1_non_inference_zh",
    "exclusion_reason",
}
REQUIRED_BUDGET_FIELDS = {
    "unit",
    "denominator",
    "unique_sample_definition",
    "repeat_definition",
    "cumulative_exposure_definition",
    "compute_cost",
}
REQUIRED_ANCHOR_FIELDS = {"page", "line", "quote", "paraphrase_zh"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _normalize_title(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _normalize_quote(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _safe_path(root: Path, value: Any, label: str, errors: list[str]) -> Path | None:
    raw = Path(str(value or ""))
    if not str(value or "").strip() or raw.is_absolute():
        errors.append(f"{label}: corpus-relative path required")
        return None
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        errors.append(f"{label}: path escapes corpus root")
        return None
    return resolved


def _require_mapping(value: Any, fields: set[str], label: str, errors: list[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{label}: mapping required")
        return None
    missing = sorted(fields - set(value))
    if missing:
        errors.append(f"{label}: missing fields {missing}")
    return value


def _walk_strings(value: Any, prefix: str = "") -> Sequence[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, str):
        found.append((prefix, value))
    elif isinstance(value, Mapping):
        for key, child in value.items():
            found.extend(_walk_strings(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_strings(child, f"{prefix}[{index}]"))
    return found


def _require_prose(value: Any, label: str, errors: list[str], *, minimum: int = 12) -> None:
    if not isinstance(value, str) or len(value.strip()) < minimum:
        errors.append(f"{label}: at least {minimum} characters required")
        return
    if PLACEHOLDER_RE.search(value):
        errors.append(f"{label}: placeholder is forbidden")


def _require_nonempty_list(value: Any, label: str, errors: list[str], *, min_items: int = 1) -> None:
    if not isinstance(value, list) or len(value) < min_items:
        errors.append(f"{label}: list with at least {min_items} item(s) required")
        return
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{label}[{index}]: non-empty string required")
        elif PLACEHOLDER_RE.search(item):
            errors.append(f"{label}[{index}]: placeholder is forbidden")


def _validate_artifact(
    *,
    root: Path,
    value: Any,
    expected_path: str,
    expected_bytes: str,
    expected_sha256: str,
    label: str,
    errors: list[str],
    require_pdf: bool,
) -> Path | None:
    artifact = _require_mapping(value, {"path", "bytes", "sha256"}, label, errors)
    if artifact is None:
        return None
    if str(artifact.get("path", "")).replace("\\", "/") != expected_path.replace("\\", "/"):
        errors.append(f"{label}.path: does not match frozen input")
    try:
        declared_bytes = int(artifact.get("bytes"))
        frozen_bytes = int(expected_bytes)
    except (TypeError, ValueError):
        errors.append(f"{label}.bytes: integer identity required")
        declared_bytes = frozen_bytes = -1
    declared_sha = str(artifact.get("sha256", "")).upper()
    if declared_bytes != frozen_bytes:
        errors.append(f"{label}.bytes: does not match frozen input")
    if declared_sha != expected_sha256.upper():
        errors.append(f"{label}.SHA: does not match frozen input")
    path = _safe_path(root, artifact.get("path"), f"{label}.path", errors)
    if path is None:
        return None
    if not path.is_file():
        errors.append(f"{label}: file missing: {path}")
        return path
    if path.stat().st_size != declared_bytes:
        errors.append(f"{label}.bytes: declared {declared_bytes}, observed {path.stat().st_size}")
    observed_sha = _sha256(path)
    if observed_sha != declared_sha:
        errors.append(f"{label}.SHA: declared {declared_sha}, observed {observed_sha}")
    if require_pdf and (path.suffix.casefold() != ".pdf" or path.read_bytes()[:5] != b"%PDF-"):
        errors.append(f"{label}: verified PDF required")
    return path


def _page_lines(text: str) -> list[list[str]]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return [page.splitlines() for page in normalized.split("\f")]


def _validate_anchor(
    anchor: Any,
    *,
    pages: list[list[str]],
    label: str,
    errors: list[str],
) -> None:
    value = _require_mapping(anchor, REQUIRED_ANCHOR_FIELDS, label, errors)
    if value is None:
        return
    try:
        page_number = int(value.get("page"))
        line_number = int(value.get("line"))
    except (TypeError, ValueError):
        errors.append(f"{label}: integer page and line required")
        return
    if page_number < 1 or page_number > len(pages):
        errors.append(f"{label}.page: {page_number} outside 1..{len(pages)}")
        return
    page = pages[page_number - 1]
    if line_number < 1 or line_number > len(page):
        errors.append(f"{label}.line: {line_number} outside 1..{len(page)}")
        return
    quote = _normalize_quote(value.get("quote"))
    if len(quote) < 8:
        errors.append(f"{label}.quote: at least 8 characters required")
    lower = max(0, line_number - 3)
    upper = min(len(page), line_number + 2)
    window = _normalize_quote(" ".join(page[lower:upper]))
    if quote and quote not in window:
        errors.append(f"{label}: anchor quote not found near declared page/line")
    _require_prose(value.get("paraphrase_zh"), f"{label}.paraphrase_zh", errors, minimum=12)


def _validate_sections(value: Any, *, pages: list[list[str]], paper_id: str, errors: list[str]) -> None:
    sections = _require_mapping(value, REQUIRED_SECTIONS, f"{paper_id}.section_evidence", errors)
    if sections is None:
        return
    for section_name in sorted(REQUIRED_SECTIONS):
        section = _require_mapping(
            sections.get(section_name),
            {"status", "pages", "anchors"},
            f"{paper_id}.section_evidence.{section_name}",
            errors,
        )
        if section is None:
            continue
        status = section.get("status")
        if status not in ALLOWED_SECTION_STATUS:
            errors.append(f"{paper_id}.section_evidence.{section_name}.status: invalid {status!r}")
        _require_prose(
            section.get("pages"),
            f"{paper_id}.section_evidence.{section_name}.pages",
            errors,
            minimum=1,
        )
        anchors = section.get("anchors")
        if not isinstance(anchors, list):
            errors.append(f"{paper_id}.section_evidence.{section_name}.anchors: list required")
            continue
        if status == "READ":
            if not anchors:
                errors.append(f"{paper_id}.section_evidence.{section_name}.anchors: one anchor required")
            for index, anchor in enumerate(anchors):
                _validate_anchor(
                    anchor,
                    pages=pages,
                    label=f"{paper_id}.section_evidence.{section_name}.anchors[{index}]",
                    errors=errors,
                )
        else:
            if section_name in {"METHODS", "EXPERIMENTS"}:
                errors.append(f"{paper_id}.section_evidence.{section_name}: eligible screening requires reading")
            if anchors:
                errors.append(f"{paper_id}.section_evidence.{section_name}.anchors: must be empty when not reported")
            _require_prose(
                section.get("absence_reason_zh"),
                f"{paper_id}.section_evidence.{section_name}.absence_reason_zh",
                errors,
                minimum=20,
            )


def _validate_record(
    *,
    root: Path,
    queue: Mapping[str, str],
    extraction: Mapping[str, str],
    record: Mapping[str, Any],
    errors: list[str],
) -> None:
    paper_id = str(record.get("paper_id", "")).strip() or "<missing-paper-id>"
    missing = sorted(REQUIRED_RECORD_FIELDS - set(record))
    if missing:
        errors.append(f"{paper_id}: missing record fields {missing}")
    if record.get("schema_version") != "3.0":
        errors.append(f"{paper_id}.schema_version: expected 3.0")
    if str(record.get("canonical_work_id", "")) != queue["canonical_work_id"]:
        errors.append(f"{paper_id}: canonical work identity mismatch")
    if _normalize_title(record.get("title")) != _normalize_title(queue["title"]):
        errors.append(f"{paper_id}: title identity mismatch")
    decision = record.get("decision")
    if decision not in ALLOWED_DECISIONS:
        errors.append(f"{paper_id}.decision: invalid {decision!r}")
    try:
        reviewed = datetime.fromisoformat(str(record.get("reviewed_at", "")))
        if reviewed.tzinfo is None:
            raise ValueError("timezone missing")
    except ValueError:
        errors.append(f"{paper_id}.reviewed_at: timezone-aware ISO timestamp required")
    _require_prose(record.get("reviewer"), f"{paper_id}.reviewer", errors, minimum=3)

    method_path = _validate_artifact(
        root=root,
        value=record.get("method_source"),
        # Queue paths are relative to their BROAD staging root.  The extraction
        # ledger publishes the same verified bytes under a corpus-relative path.
        expected_path=extraction["source_path"],
        expected_bytes=queue["method_source_bytes"],
        expected_sha256=queue["method_source_sha256"],
        label=f"{paper_id}.method_source",
        errors=errors,
        require_pdf=True,
    )
    text_path = _validate_artifact(
        root=root,
        value=record.get("text_source"),
        expected_path=extraction["text_path"],
        expected_bytes=extraction["text_bytes"],
        expected_sha256=extraction["text_sha256"],
        label=f"{paper_id}.text_source",
        errors=errors,
        require_pdf=False,
    )
    if extraction["source_sha256"].upper() != queue["method_source_sha256"].upper():
        errors.append(f"{paper_id}: extraction source SHA does not match method source")
    if int(extraction["source_bytes"]) != int(queue["method_source_bytes"]):
        errors.append(f"{paper_id}: extraction source bytes do not match method source")
    pages: list[list[str]] = []
    if text_path is not None and text_path.is_file():
        pages = _page_lines(text_path.read_text(encoding="utf-8", errors="replace"))
    if pages:
        _validate_sections(record.get("section_evidence"), pages=pages, paper_id=paper_id, errors=errors)

    rq_ids = record.get("rq_ids")
    if not isinstance(rq_ids, list) or not rq_ids or not set(map(str, rq_ids)).issubset(ALLOWED_RQ_IDS):
        errors.append(f"{paper_id}.rq_ids: non-empty subset of RQ1..RQ8 required")
    for field in (
        "formulas",
        "variables",
        "random_baselines",
        "datasets",
        "models",
        "ablations",
        "negative_results",
        "failure_conditions",
        "limitations",
    ):
        _require_nonempty_list(record.get(field), f"{paper_id}.{field}", errors)
    _require_nonempty_list(record.get("algorithm_steps"), f"{paper_id}.algorithm_steps", errors, min_items=2)
    for field in (
        "selection_timing",
        "refresh_rule",
        "checkpoint_selection",
        "stage1_mechanism_zh",
        "stage1_non_inference_zh",
    ):
        minimum = 50 if field.startswith("stage1_") else 12
        _require_prose(record.get(field), f"{paper_id}.{field}", errors, minimum=minimum)
    budget = _require_mapping(record.get("budget"), REQUIRED_BUDGET_FIELDS, f"{paper_id}.budget", errors)
    if budget is not None:
        for field in REQUIRED_BUDGET_FIELDS:
            _require_prose(budget.get(field), f"{paper_id}.budget.{field}", errors, minimum=6)
    seed_count = record.get("seed_count")
    if isinstance(seed_count, Mapping):
        seed_evidence = _require_mapping(
            seed_count,
            {"counts", "scope", "aggregation"},
            f"{paper_id}.seed_count",
            errors,
        )
        if seed_evidence is not None:
            _require_nonempty_list(
                seed_evidence.get("counts"),
                f"{paper_id}.seed_count.counts",
                errors,
            )
            _require_prose(
                seed_evidence.get("scope"),
                f"{paper_id}.seed_count.scope",
                errors,
                minimum=8,
            )
            _require_prose(
                seed_evidence.get("aggregation"),
                f"{paper_id}.seed_count.aggregation",
                errors,
                minimum=8,
            )
    elif seed_count != "NOT_REPORTED_BY_PAPER" and not (
        isinstance(seed_count, int) and not isinstance(seed_count, bool) and seed_count >= 1
    ):
        errors.append(
            f"{paper_id}.seed_count: positive integer, structured seed evidence, "
            "or NOT_REPORTED_BY_PAPER required"
        )
    if record.get("transfer_class") not in ALLOWED_TRANSFER_CLASSES:
        errors.append(f"{paper_id}.transfer_class: invalid value")

    results = record.get("results")
    if not isinstance(results, list) or not results:
        errors.append(f"{paper_id}.results: non-empty result list required")
    else:
        for index, result in enumerate(results):
            value = _require_mapping(
                result,
                {"claim", "locator", "value", "anchor"},
                f"{paper_id}.results[{index}]",
                errors,
            )
            if value is None:
                continue
            for field in ("claim", "locator", "value"):
                _require_prose(value.get(field), f"{paper_id}.results[{index}].{field}", errors, minimum=4)
            if pages:
                _validate_anchor(
                    value.get("anchor"),
                    pages=pages,
                    label=f"{paper_id}.results[{index}].anchor",
                    errors=errors,
                )

    exclusion = str(record.get("exclusion_reason", ""))
    if decision == "SCREENED_ELIGIBLE":
        if not exclusion.startswith("NOT_APPLICABLE_WITH_REASON:") or len(exclusion.split(":", 1)[-1]) < 8:
            errors.append(f"{paper_id}.exclusion_reason: explicit N/A reason required for eligible paper")
    else:
        _require_prose(exclusion, f"{paper_id}.exclusion_reason", errors, minimum=30)

    for field_path, value in _walk_strings(record):
        # The title is already required to match the frozen queue identity.
        # Scientific titles may legitimately contain words such as "unknown"
        # (for example, unknown label noise); treating that subject term as an
        # unfinished-field marker would reject valid primary-source identity.
        if field_path == "title":
            continue
        if PLACEHOLDER_RE.search(value):
            errors.append(f"{paper_id}.{field_path}: placeholder is forbidden")
    del method_path


def validate_screened_review_records(
    *,
    corpus_root: str | Path,
    queue_rows: Sequence[Mapping[str, str]],
    extraction_rows: Sequence[Mapping[str, str]],
    records: Sequence[Mapping[str, Any]],
    require_queue_coverage: bool = False,
) -> ScreenedReviewResult:
    root = Path(corpus_root).resolve()
    errors: list[str] = []
    queue_by_id = {str(row.get("paper_id", "")): row for row in queue_rows}
    extraction_by_id = {str(row.get("paper_id", "")): row for row in extraction_rows}
    if len(queue_by_id) != len(queue_rows):
        errors.append("screening queue contains duplicate paper IDs")
    if len(extraction_by_id) != len(extraction_rows):
        errors.append("extraction ledger contains duplicate paper IDs")
    if set(queue_by_id) != set(extraction_by_id):
        errors.append("screening queue and extraction ledger identities differ")
    record_by_id: dict[str, Mapping[str, Any]] = {}
    unique_prose: dict[str, dict[str, str]] = {
        "stage1_mechanism_zh": {},
        "stage1_non_inference_zh": {},
    }
    for record in records:
        paper_id = str(record.get("paper_id", ""))
        if paper_id in record_by_id:
            errors.append(f"duplicate review record: {paper_id}")
            continue
        record_by_id[paper_id] = record
        queue = queue_by_id.get(paper_id)
        extraction = extraction_by_id.get(paper_id)
        if queue is None or extraction is None:
            errors.append(f"{paper_id}: review identity absent from frozen queue")
            continue
        _validate_record(
            root=root,
            queue=queue,
            extraction=extraction,
            record=record,
            errors=errors,
        )
        for field in unique_prose:
            value = record.get(field)
            if not isinstance(value, str) or len(value.strip()) < 50:
                continue
            normalized = re.sub(r"\s+", "", value).casefold()
            previous = unique_prose[field].get(normalized)
            if previous is not None:
                errors.append(f"{paper_id}.{field}: prose reused verbatim from {previous}")
            else:
                unique_prose[field][normalized] = paper_id
    if require_queue_coverage and set(record_by_id) != set(queue_by_id):
        missing = sorted(set(queue_by_id) - set(record_by_id))
        extra = sorted(set(record_by_id) - set(queue_by_id))
        errors.append(f"review set does not cover queue exactly: missing={missing} extra={extra}")
    if errors:
        raise ScreenedReviewError(errors)
    ordered = tuple(record_by_id[paper_id] for paper_id in sorted(record_by_id))
    eligible = sum(record["decision"] == "SCREENED_ELIGIBLE" for record in ordered)
    return ScreenedReviewResult(
        status="PASS",
        reviewed_count=len(ordered),
        eligible_count=eligible,
        excluded_count=len(ordered) - eligible,
        records=ordered,
    )
