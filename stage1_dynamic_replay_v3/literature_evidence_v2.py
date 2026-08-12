"""Fail-closed contracts for the Stage1 500/300/100 literature corpus.

The validator treats files, hashes, and per-paper reading records as evidence
of work performed.  It never infers that a paper was read from a title count or
from a previous campaign ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse


SCHEMA_VERSION = "2.0"
TIERS = {"BROAD", "SCREENED", "DEEP"}
RELATIONS = {"SUPPORTED", "REFUTED", "MIXED"}
TRANSFER_CLASSES = {
    "REPLICATION",
    "INSPIRED_ADAPTATION",
    "MECHANISM_ONLY",
    "NOT_TRANSFERABLE",
}
RESEARCH_QUESTIONS = {f"RQ{index}" for index in range(1, 9)}
REQUIRED_BROAD_SCOPES = {
    "TITLE",
    "ABSTRACT",
    "PROBLEM",
    "METHOD_OVERVIEW",
    "CONCLUSION",
}
REQUIRED_SCREENED_SECTIONS = {"METHODS", "EXPERIMENTS", "ABLATIONS", "LIMITATIONS"}
ALLOWED_MISSING_MARKERS = {
    "NOT_ASSESSED_AT_BROAD_LEVEL",
    "NOT_REPORTED_BY_PAPER",
}
PLACEHOLDER_RE = re.compile(r"(?i)^\s*(?:TODO|TBD|UNKNOWN)\s*$|待补|待确认|同上")
MARKER = "<!-- STAGE1_EVIDENCE_V2 -->"
JSON_OPEN = "```json\n"
JSON_CLOSE = "\n```"

REGISTRY_FIELDS = (
    "paper_id",
    "tier",
    "canonical_work_id",
    "title",
    "authors",
    "year",
    "venue",
    "primary_url",
    "doi",
    "arxiv_id",
    "openreview_id",
    "note_path",
    "source_path",
    "source_sha256",
    "source_bytes",
)


class LiteratureEvidenceError(RuntimeError):
    """Raised when any corpus contract fails."""

    def __init__(self, errors: str | Sequence[str]) -> None:
        if isinstance(errors, str):
            values = (errors,)
        else:
            values = tuple(str(error) for error in errors)
        self.errors = values
        super().__init__("literature evidence validation failed:\n- " + "\n- ".join(values))


@dataclass(frozen=True)
class TierCounts:
    broad: int = 500
    screened: int = 300
    deep: int = 100

    def __post_init__(self) -> None:
        if self.broad < 1 or min(self.screened, self.deep) < 0:
            raise ValueError("broad count must be positive and nested tier counts non-negative")
        if not self.deep <= self.screened <= self.broad:
            raise ValueError("expected DEEP <= SCREENED <= BROAD")


@dataclass(frozen=True)
class PaperEvidence:
    paper_id: str
    tier: str
    title: str
    metadata: Mapping[str, Any]
    note_path: Path


@dataclass(frozen=True)
class CorpusValidationReport:
    status: str
    counts: Mapping[str, int]
    note_count: int
    errors: tuple[str, ...]
    papers: tuple[PaperEvidence, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "counts": dict(self.counts),
            "note_count": self.note_count,
            "paper_ids": [paper.paper_id for paper in self.papers],
            "errors": list(self.errors),
            "formal_training_started": False,
            "engineering_gate_generated": False,
            "blind_holdout_opened": False,
        }


@dataclass(frozen=True)
class CompletionAuditReport:
    status: str
    gates: Mapping[str, str]
    counts: Mapping[str, int]
    errors: tuple[str, ...]
    formal_training_started: bool = False
    engineering_gate_generated: bool = False
    blind_holdout_opened: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "gates": dict(self.gates),
            "counts": dict(self.counts),
            "errors": list(self.errors),
            "formal_training_started": self.formal_training_started,
            "engineering_gate_generated": self.engineering_gate_generated,
            "blind_holdout_opened": self.blind_holdout_opened,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value.casefold())


def _normalize_doi(value: str) -> str | None:
    clean = value.strip().casefold()
    if _is_missing_marker(clean.upper()):
        return None
    clean = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", clean)
    return clean.rstrip("/") or None


def _normalize_external_id(value: str, *, prefixes: Sequence[str]) -> str | None:
    clean = value.strip().casefold()
    if _is_missing_marker(clean.upper()):
        return None
    for prefix in prefixes:
        if clean.startswith(prefix):
            clean = clean[len(prefix) :]
    return clean.strip("/ ") or None


def _is_missing_marker(value: str) -> bool:
    return value in ALLOWED_MISSING_MARKERS or value.startswith("NOT_APPLICABLE_WITH_REASON:")


def _valid_missing_marker(value: str) -> bool:
    if value in ALLOWED_MISSING_MARKERS:
        return True
    if value.startswith("NOT_APPLICABLE_WITH_REASON:"):
        return bool(value.split(":", 1)[1].strip())
    return False


def _walk_strings(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_strings(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_strings(child, f"{prefix}[{index}]")


def _read_registry(path: Path, errors: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        errors.append(f"registry missing: {path}")
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        missing = [field for field in REGISTRY_FIELDS if field not in fields]
        if missing:
            errors.append(f"registry missing fields: {missing}")
        return [dict(row) for row in reader]


def _read_note_metadata(path: Path, paper_id: str, errors: list[str]) -> Mapping[str, Any] | None:
    if not path.is_file():
        errors.append(f"{paper_id}: note missing: {path}")
        return None
    text = path.read_text(encoding="utf-8")
    if text.count(MARKER) != 1 or text.count(JSON_OPEN) != 1:
        errors.append(f"{paper_id}: note must contain exactly one v2 JSON evidence block")
        return None
    try:
        raw = text.split(JSON_OPEN, 1)[1].split(JSON_CLOSE, 1)[0]
        metadata = json.loads(raw)
    except (IndexError, json.JSONDecodeError) as exc:
        errors.append(f"{paper_id}: invalid note JSON: {exc}")
        return None
    if not isinstance(metadata, Mapping):
        errors.append(f"{paper_id}: note JSON root must be an object")
        return None
    return metadata


def _safe_relative_path(root: Path, raw: Any, label: str, errors: list[str]) -> Path | None:
    value = str(raw or "").strip()
    if not value:
        errors.append(f"{label}: path is empty")
        return None
    path = Path(value)
    if path.is_absolute():
        errors.append(f"{label}: path must be corpus-relative, found absolute path {value}")
        return None
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        errors.append(f"{label}: path escapes corpus root: {value}")
        return None
    return resolved


def _validate_artifact(
    root: Path,
    artifact: Any,
    label: str,
    errors: list[str],
) -> Path | None:
    if not isinstance(artifact, Mapping):
        errors.append(f"{label}: artifact must be an object")
        return None
    for field in ("path", "kind", "bytes", "sha256"):
        if field not in artifact:
            errors.append(f"{label}.{field}: required")
    path = _safe_relative_path(root, artifact.get("path"), f"{label}.path", errors)
    if path is None:
        return None
    if not path.is_file():
        errors.append(f"{label}: artifact missing: {path}")
        return path
    try:
        expected_bytes = int(artifact.get("bytes"))
    except (TypeError, ValueError):
        errors.append(f"{label}.bytes: must be an integer")
        expected_bytes = -1
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        errors.append(f"{label}.bytes: expected {expected_bytes}, observed {actual_bytes}")
    expected_sha = str(artifact.get("sha256") or "").upper()
    actual_sha = _sha256(path)
    if not re.fullmatch(r"[0-9A-F]{64}", expected_sha):
        errors.append(f"{label}.sha256: invalid SHA-256 {expected_sha!r}")
    elif actual_sha != expected_sha:
        errors.append(f"{label}.sha256: expected {expected_sha}, observed {actual_sha}")
    return path


def _require_mapping_fields(
    value: Any,
    fields: Sequence[str],
    label: str,
    errors: list[str],
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{label}: must be an object")
        return None
    for field in fields:
        if field not in value:
            errors.append(f"{label}.{field}: required")
    return value


def _require_nonempty(value: Any, label: str, errors: list[str], *, min_length: int = 1) -> None:
    if isinstance(value, str):
        if len(value.strip()) < min_length:
            errors.append(f"{label}: must contain at least {min_length} characters")
    elif isinstance(value, list):
        if not value:
            errors.append(f"{label}: list must not be empty")
    elif value is None:
        errors.append(f"{label}: required")


def _parse_iso_datetime(value: Any, label: str, errors: list[str]) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        errors.append(f"{label}: must be an ISO-8601 datetime")
        return None
    if parsed.tzinfo is None:
        errors.append(f"{label}: timezone is required")
    return parsed


def _validate_placeholders(metadata: Mapping[str, Any], paper_id: str, errors: list[str]) -> None:
    for field, value in _walk_strings(metadata):
        if PLACEHOLDER_RE.search(value):
            errors.append(f"{paper_id}.{field}: forbidden placeholder {PLACEHOLDER_RE.search(value).group(0)!r}")
        if value.startswith("NOT_") and value not in TRANSFER_CLASSES and not _valid_missing_marker(value):
            errors.append(f"{paper_id}.{field}: invalid missing marker {value!r}")
        if value == "SOURCE_UNAVAILABLE_EXCLUDED":
            errors.append(f"{paper_id}.{field}: excluded sources cannot enter the counted corpus")


def _validate_broad(
    metadata: Mapping[str, Any],
    paper_id: str,
    errors: list[str],
    unique_prose: dict[str, dict[str, str]],
) -> None:
    rq_ids = metadata.get("rq_ids")
    if not isinstance(rq_ids, list) or not rq_ids:
        errors.append(f"{paper_id}.rq_ids: at least one research question is required")
    else:
        invalid = sorted(set(map(str, rq_ids)) - RESEARCH_QUESTIONS)
        if invalid:
            errors.append(f"{paper_id}.rq_ids: invalid IDs {invalid}")
    relation = metadata.get("relation")
    if relation not in RELATIONS:
        errors.append(f"{paper_id}.relation: expected one of {sorted(RELATIONS)}, observed {relation!r}")

    reading = _require_mapping_fields(
        metadata.get("reading"),
        (
            "read_at",
            "scopes",
            "sections_checked",
            "summary_zh",
            "critical_review_zh",
            "direct_relevance_chain",
            "supported_or_refuted",
            "transferable_mechanisms",
            "unsupported_inferences",
            "stage1_boundary",
        ),
        f"{paper_id}.reading",
        errors,
    )
    if reading is None:
        return
    _parse_iso_datetime(reading.get("read_at"), f"{paper_id}.reading.read_at", errors)
    scopes = reading.get("scopes")
    if not isinstance(scopes, list):
        errors.append(f"{paper_id}.reading.scopes: must be a list")
    else:
        missing = sorted(REQUIRED_BROAD_SCOPES - set(map(str, scopes)))
        if missing:
            errors.append(f"{paper_id}.reading.scopes: missing {missing}")
    _require_nonempty(reading.get("sections_checked"), f"{paper_id}.reading.sections_checked", errors)
    for field, minimum in (
        ("summary_zh", 80),
        ("critical_review_zh", 80),
        ("direct_relevance_chain", 60),
        ("supported_or_refuted", 60),
        ("stage1_boundary", 60),
    ):
        value = reading.get(field)
        _require_nonempty(value, f"{paper_id}.reading.{field}", errors, min_length=minimum)
        if isinstance(value, str) and len(value.strip()) >= minimum:
            normalized = _normalize_text(value)
            previous = unique_prose[field].get(normalized)
            if previous is not None:
                errors.append(
                    f"{paper_id}.reading.{field}: reused verbatim from {previous}; per-paper prose must be independent"
                )
            else:
                unique_prose[field][normalized] = paper_id
    for field in ("transferable_mechanisms", "unsupported_inferences"):
        _require_nonempty(reading.get(field), f"{paper_id}.reading.{field}", errors)


def _validate_screened(
    root: Path,
    metadata: Mapping[str, Any],
    paper_id: str,
    errors: list[str],
) -> None:
    screened = _require_mapping_fields(
        metadata.get("screened"),
        (
            "sections_checked",
            "method_source",
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
        ),
        f"{paper_id}.screened",
        errors,
    )
    if screened is None:
        return
    sections = screened.get("sections_checked")
    if not isinstance(sections, list):
        errors.append(f"{paper_id}.screened.sections_checked: must be a list")
    else:
        missing = sorted(REQUIRED_SCREENED_SECTIONS - set(map(str, sections)))
        if missing:
            errors.append(f"{paper_id}.screened.sections_checked: missing {missing}")
    _validate_artifact(root, screened.get("method_source"), f"{paper_id}.screened.method_source", errors)
    for field in (
        "formulas",
        "algorithm_steps",
        "variables",
        "random_baselines",
        "datasets",
        "models",
        "results",
        "ablations",
        "negative_results",
        "failure_conditions",
        "limitations",
    ):
        _require_nonempty(screened.get(field), f"{paper_id}.screened.{field}", errors)
    steps = screened.get("algorithm_steps")
    if isinstance(steps, list) and steps != ["NOT_REPORTED_BY_PAPER"] and len(steps) < 2:
        errors.append(f"{paper_id}.screened.algorithm_steps: require at least two steps")
    for field in ("selection_timing", "refresh_rule", "checkpoint_selection"):
        _require_nonempty(screened.get(field), f"{paper_id}.screened.{field}", errors, min_length=8)
    budget = _require_mapping_fields(
        screened.get("budget"),
        (
            "unit",
            "denominator",
            "unique_sample_definition",
            "repeat_definition",
            "cumulative_exposure_definition",
            "compute_cost",
        ),
        f"{paper_id}.screened.budget",
        errors,
    )
    if budget is not None:
        for field in budget:
            _require_nonempty(budget.get(field), f"{paper_id}.screened.budget.{field}", errors)
    seed_count = screened.get("seed_count")
    if not (isinstance(seed_count, int) and seed_count >= 1) and seed_count != "NOT_REPORTED_BY_PAPER":
        errors.append(f"{paper_id}.screened.seed_count: positive integer or NOT_REPORTED_BY_PAPER required")
    transfer = screened.get("transfer_class")
    if transfer not in TRANSFER_CLASSES:
        errors.append(
            f"{paper_id}.screened.transfer_class: expected one of {sorted(TRANSFER_CLASSES)}, observed {transfer!r}"
        )
    results = screened.get("results")
    if isinstance(results, list):
        for index, result in enumerate(results):
            result_map = _require_mapping_fields(
                result,
                ("claim", "locator", "value"),
                f"{paper_id}.screened.results[{index}]",
                errors,
            )
            if result_map is not None:
                for field in ("claim", "locator", "value"):
                    _require_nonempty(
                        result_map.get(field),
                        f"{paper_id}.screened.results[{index}].{field}",
                        errors,
                    )


def _pdf_page_count(path: Path) -> int:
    executable = shutil.which("pdfinfo")
    if executable is None:
        raise LiteratureEvidenceError("pdfinfo is required for strict PDF page-count validation")
    result = subprocess.run(
        [executable, str(path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise LiteratureEvidenceError(f"pdfinfo failed for {path}: {result.stderr.strip()}")
    match = re.search(r"(?m)^Pages:\s+(\d+)\s*$", result.stdout)
    if match is None:
        raise LiteratureEvidenceError(f"pdfinfo output has no page count for {path}")
    return int(match.group(1))


def _validate_deep(
    root: Path,
    metadata: Mapping[str, Any],
    paper_id: str,
    errors: list[str],
    *,
    inspect_pdf_pages: bool,
) -> None:
    deep = _require_mapping_fields(
        metadata.get("deep"),
        (
            "first_read_at",
            "full_text",
            "page_count",
            "section_coverage",
            "anchors",
            "formula_assumptions",
            "algorithm_complexity",
            "randomness",
            "data_roles",
            "leakage_risks",
            "budget_fairness",
            "seed_variation",
            "worst_case",
            "key_ablations",
            "limitations",
            "stage1_mapping",
            "counter_check",
        ),
        f"{paper_id}.deep",
        errors,
    )
    if deep is None:
        return
    _parse_iso_datetime(deep.get("first_read_at"), f"{paper_id}.deep.first_read_at", errors)
    pdf = _validate_artifact(root, deep.get("full_text"), f"{paper_id}.deep.full_text", errors)
    if pdf is not None and pdf.is_file():
        if pdf.suffix.casefold() != ".pdf" or not pdf.read_bytes()[:5].startswith(b"%PDF-"):
            errors.append(f"{paper_id}.deep.full_text: must be a PDF file with a PDF signature")
    try:
        page_count = int(deep.get("page_count"))
    except (TypeError, ValueError):
        page_count = -1
        errors.append(f"{paper_id}.deep.page_count: positive integer required")
    if page_count < 1:
        errors.append(f"{paper_id}.deep.page_count: positive integer required")
    elif inspect_pdf_pages and pdf is not None and pdf.is_file():
        try:
            observed_pages = _pdf_page_count(pdf)
        except LiteratureEvidenceError as exc:
            errors.extend(f"{paper_id}.deep.full_text: {message}" for message in exc.errors)
        else:
            if observed_pages != page_count:
                errors.append(
                    f"{paper_id}.deep.page_count: declared {page_count}, observed {observed_pages}"
                )

    coverage = deep.get("section_coverage")
    if not isinstance(coverage, list) or len(coverage) < 4:
        errors.append(f"{paper_id}.deep.section_coverage: at least four fully read sections required")
    else:
        for index, section in enumerate(coverage):
            section_map = _require_mapping_fields(
                section,
                ("section", "pages", "status"),
                f"{paper_id}.deep.section_coverage[{index}]",
                errors,
            )
            if section_map is not None and section_map.get("status") != "READ_FULLY":
                errors.append(f"{paper_id}.deep.section_coverage[{index}].status: READ_FULLY required")

    anchors = deep.get("anchors")
    if not isinstance(anchors, list) or len(anchors) < 3:
        errors.append(f"{paper_id}.deep.anchors: at least three page-level anchors required")
    else:
        for index, anchor in enumerate(anchors):
            anchor_map = _require_mapping_fields(
                anchor,
                ("page", "locator", "paraphrase"),
                f"{paper_id}.deep.anchors[{index}]",
                errors,
            )
            if anchor_map is None:
                continue
            try:
                page = int(anchor_map.get("page"))
            except (TypeError, ValueError):
                page = -1
            if page < 1 or (page_count > 0 and page > page_count):
                errors.append(
                    f"{paper_id}.deep.anchors[{index}].page: {page} outside 1..{page_count}"
                )
            _require_nonempty(anchor_map.get("locator"), f"{paper_id}.deep.anchors[{index}].locator", errors)
            _require_nonempty(
                anchor_map.get("paraphrase"),
                f"{paper_id}.deep.anchors[{index}].paraphrase",
                errors,
                min_length=10,
            )

    for field in (
        "formula_assumptions",
        "algorithm_complexity",
        "randomness",
        "data_roles",
        "leakage_risks",
        "budget_fairness",
        "seed_variation",
        "worst_case",
        "key_ablations",
        "limitations",
        "counter_check",
    ):
        _require_nonempty(deep.get(field), f"{paper_id}.deep.{field}", errors)
    mapping = _require_mapping_fields(
        deep.get("stage1_mapping"),
        ("fields", "interfaces", "cost", "code_mapping"),
        f"{paper_id}.deep.stage1_mapping",
        errors,
    )
    if mapping is not None:
        for field in mapping:
            _require_nonempty(mapping.get(field), f"{paper_id}.deep.stage1_mapping.{field}", errors)


def audit_corpus(
    corpus_root: str | Path,
    *,
    expected: TierCounts = TierCounts(),
    inspect_pdf_pages: bool = True,
) -> CorpusValidationReport:
    """Inspect a corpus and return every detected error without publishing PASS early."""

    root = Path(corpus_root).resolve()
    errors: list[str] = []
    rows = _read_registry(root / "CANONICAL_WORKS.csv", errors)
    counts = {
        "broad": len(rows),
        "screened": sum(row.get("tier") in {"SCREENED", "DEEP"} for row in rows),
        "deep": sum(row.get("tier") == "DEEP" for row in rows),
    }
    expected_values = {
        "broad": expected.broad,
        "screened": expected.screened,
        "deep": expected.deep,
    }
    for tier, expected_count in expected_values.items():
        if counts[tier] != expected_count:
            errors.append(f"count.{tier}: expected exactly {expected_count}, observed {counts[tier]}")

    observed_ids = [row.get("paper_id", "") for row in rows]
    expected_ids = [f"P{index:04d}" for index in range(1, expected.broad + 1)]
    if observed_ids != expected_ids:
        errors.append("paper_id sequence must be exactly P0001..P%04d in registry order" % expected.broad)
    if len(set(observed_ids)) != len(observed_ids):
        errors.append("paper_id: duplicate identities found")

    note_files = sorted((root / "notes").glob("P*.md")) if (root / "notes").is_dir() else []
    registered_note_names = {Path(row.get("note_path", "")).name for row in rows}
    observed_note_names = {path.name for path in note_files}
    missing_notes = sorted(registered_note_names - observed_note_names)
    extra_notes = sorted(observed_note_names - registered_note_names)
    if missing_notes:
        errors.append(f"note files missing: {missing_notes}")
    if extra_notes:
        errors.append(f"unregistered P*.md note files: {extra_notes}")

    duplicate_fields: dict[str, dict[str, str]] = {
        "canonical_work_id": {},
        "title": {},
        "doi": {},
        "arxiv_id": {},
        "openreview_id": {},
        "primary_url": {},
    }
    unique_prose = {
        "summary_zh": {},
        "critical_review_zh": {},
        "direct_relevance_chain": {},
        "supported_or_refuted": {},
        "stage1_boundary": {},
    }
    papers: list[PaperEvidence] = []

    for row_index, row in enumerate(rows, start=2):
        paper_id = row.get("paper_id") or f"REGISTRY_ROW_{row_index}"
        tier = row.get("tier", "")
        if tier not in TIERS:
            errors.append(f"{paper_id}.tier: invalid tier {tier!r}")

        identity_values = {
            "canonical_work_id": _normalize_text(row.get("canonical_work_id", "")),
            "title": _normalize_text(row.get("title", "")),
            "doi": _normalize_doi(row.get("doi", "")),
            "arxiv_id": _normalize_external_id(
                row.get("arxiv_id", ""),
                prefixes=("https://arxiv.org/abs/", "arxiv:"),
            ),
            "openreview_id": _normalize_external_id(
                row.get("openreview_id", ""),
                prefixes=("https://openreview.net/forum?id=",),
            ),
            "primary_url": row.get("primary_url", "").strip().casefold().rstrip("/"),
        }
        for field, normalized in identity_values.items():
            if not normalized:
                if field in {"canonical_work_id", "title", "primary_url"}:
                    errors.append(f"{paper_id}.{field}: required")
                continue
            previous = duplicate_fields[field].get(normalized)
            if previous is not None:
                errors.append(f"{paper_id}.{field}: duplicate canonical identity with {previous}")
            else:
                duplicate_fields[field][normalized] = paper_id

        parsed_url = urlparse(row.get("primary_url", ""))
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            errors.append(f"{paper_id}.primary_url: valid HTTP(S) primary URL required")

        note_path = _safe_relative_path(root, row.get("note_path"), f"{paper_id}.note_path", errors)
        source_path = _safe_relative_path(root, row.get("source_path"), f"{paper_id}.source_path", errors)
        if note_path is None:
            continue
        expected_note = (root / "notes" / f"{paper_id}.md").resolve()
        if note_path != expected_note:
            errors.append(f"{paper_id}.note_path: expected notes/{paper_id}.md")
        metadata = _read_note_metadata(note_path, paper_id, errors)
        if metadata is None:
            continue
        _validate_placeholders(metadata, paper_id, errors)
        if metadata.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"{paper_id}.schema_version: expected {SCHEMA_VERSION}")
        if metadata.get("paper_id") != paper_id:
            errors.append(f"{paper_id}.paper_id: note identity mismatch {metadata.get('paper_id')!r}")
        if metadata.get("tier") != tier:
            errors.append(f"{paper_id}.tier: note {metadata.get('tier')!r} != registry {tier!r}")
        identity = _require_mapping_fields(
            metadata.get("identity"),
            (
                "canonical_work_id",
                "title",
                "authors",
                "year",
                "venue",
                "primary_url",
                "doi",
                "arxiv_id",
                "openreview_id",
                "merged_versions",
            ),
            f"{paper_id}.identity",
            errors,
        )
        if identity is not None:
            comparisons = {
                "canonical_work_id": str(identity.get("canonical_work_id", "")),
                "title": str(identity.get("title", "")),
                "authors": "; ".join(map(str, identity.get("authors", [])))
                if isinstance(identity.get("authors"), list)
                else str(identity.get("authors", "")),
                "year": str(identity.get("year", "")),
                "venue": str(identity.get("venue", "")),
                "primary_url": str(identity.get("primary_url", "")),
                "doi": str(identity.get("doi", "")),
                "arxiv_id": str(identity.get("arxiv_id", "")),
                "openreview_id": str(identity.get("openreview_id", "")),
            }
            for field, value in comparisons.items():
                if value != str(row.get(field, "")):
                    errors.append(
                        f"{paper_id}.identity.{field}: note {value!r} != registry {row.get(field)!r}"
                    )
            _require_nonempty(identity.get("merged_versions"), f"{paper_id}.identity.merged_versions", errors)

        artifact = metadata.get("source_artifact")
        artifact_path = _validate_artifact(root, artifact, f"{paper_id}.source_artifact", errors)
        if isinstance(artifact, Mapping):
            if artifact_path is not None and source_path is not None and artifact_path != source_path:
                errors.append(f"{paper_id}.source_artifact.path: does not match registry source_path")
            expected_sha = str(row.get("source_sha256", "")).upper()
            expected_bytes_raw = row.get("source_bytes", "")
            if str(artifact.get("sha256", "")).upper() != expected_sha:
                errors.append(f"{paper_id}.source_artifact.sha256: does not match registry")
            if str(artifact.get("bytes", "")) != str(expected_bytes_raw):
                errors.append(f"{paper_id}.source_artifact.bytes: does not match registry")

        _validate_broad(metadata, paper_id, errors, unique_prose)
        if tier == "BROAD":
            if metadata.get("screened") != "NOT_ASSESSED_AT_BROAD_LEVEL":
                errors.append(f"{paper_id}.screened: BROAD paper must use NOT_ASSESSED_AT_BROAD_LEVEL")
            if metadata.get("deep") != "NOT_ASSESSED_AT_BROAD_LEVEL":
                errors.append(f"{paper_id}.deep: BROAD paper must use NOT_ASSESSED_AT_BROAD_LEVEL")
        elif tier == "SCREENED":
            _validate_screened(root, metadata, paper_id, errors)
            if metadata.get("deep") != "NOT_ASSESSED_AT_BROAD_LEVEL":
                errors.append(f"{paper_id}.deep: SCREENED paper must use NOT_ASSESSED_AT_BROAD_LEVEL")
        elif tier == "DEEP":
            _validate_screened(root, metadata, paper_id, errors)
            _validate_deep(root, metadata, paper_id, errors, inspect_pdf_pages=inspect_pdf_pages)

        papers.append(
            PaperEvidence(
                paper_id=paper_id,
                tier=tier,
                title=row.get("title", ""),
                metadata=metadata,
                note_path=note_path,
            )
        )

    return CorpusValidationReport(
        status="PASS" if not errors else "INCOMPLETE",
        counts=counts,
        note_count=len(note_files),
        errors=tuple(errors),
        papers=tuple(papers),
    )


def validate_corpus(
    corpus_root: str | Path,
    *,
    expected: TierCounts = TierCounts(),
    inspect_pdf_pages: bool = True,
) -> CorpusValidationReport:
    report = audit_corpus(
        corpus_root,
        expected=expected,
        inspect_pdf_pages=inspect_pdf_pages,
    )
    if report.errors:
        raise LiteratureEvidenceError(report.errors)
    return report


def _rank_ids(ids: Iterable[str], *, seed: str, scope: str) -> list[str]:
    return sorted(
        ids,
        key=lambda paper_id: hashlib.sha256(f"{seed}|{scope}|{paper_id}".encode("utf-8")).digest(),
    )


def deterministic_audit_ids(
    papers: Sequence[PaperEvidence],
    *,
    seed: str,
) -> dict[str, tuple[str, ...]]:
    """Return the fixed nested 10%/15%/20% audit sample."""

    broad_ids = {paper.paper_id for paper in papers}
    screened_ids = {paper.paper_id for paper in papers if paper.tier in {"SCREENED", "DEEP"}}
    deep_ids = {paper.paper_id for paper in papers if paper.tier == "DEEP"}
    deep_target = math.ceil(len(deep_ids) * 0.20)
    screened_target = math.ceil(len(screened_ids) * 0.15)
    broad_target = math.ceil(len(broad_ids) * 0.10)
    if deep_target > screened_target or screened_target > broad_target:
        raise LiteratureEvidenceError(
            "audit sample targets cannot be nested; adjust corpus tier counts or audit fractions"
        )

    selected_deep = _rank_ids(deep_ids, seed=seed, scope="deep")[:deep_target]
    selected_screened = list(selected_deep)
    screened_remaining = screened_ids - set(selected_screened)
    selected_screened.extend(
        _rank_ids(screened_remaining, seed=seed, scope="screened")[
            : screened_target - len(selected_screened)
        ]
    )
    selected_broad = list(selected_screened)
    broad_remaining = broad_ids - set(selected_broad)
    selected_broad.extend(
        _rank_ids(broad_remaining, seed=seed, scope="broad")[: broad_target - len(selected_broad)]
    )
    return {
        "broad": tuple(sorted(selected_broad)),
        "screened": tuple(sorted(selected_screened)),
        "deep": tuple(sorted(selected_deep)),
    }


def _read_required_csv(
    path: Path,
    required_fields: Sequence[str],
    errors: list[str],
    *,
    label: str,
) -> list[dict[str, str]]:
    if not path.is_file():
        errors.append(f"{label}: missing {path}")
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        missing = [field for field in required_fields if field not in fields]
        if missing:
            errors.append(f"{label}: missing fields {missing}")
        return [dict(row) for row in reader]


def _raise_if_errors(errors: list[str]) -> None:
    if errors:
        raise LiteratureEvidenceError(errors)


def validate_discovery_evidence(
    corpus_root: str | Path,
    papers: Sequence[PaperEvidence],
) -> dict[str, int]:
    """Validate exact queries, raw snapshots, candidate decisions, and exclusions."""

    root = Path(corpus_root).resolve()
    errors: list[str] = []
    query_rows = _read_required_csv(
        root / "discovery" / "QUERY_LOG.csv",
        (
            "query_id",
            "database",
            "exact_query",
            "searched_at",
            "result_start",
            "result_end",
            "raw_result_count",
            "snapshot_path",
            "snapshot_sha256",
            "snapshot_bytes",
        ),
        errors,
        label="discovery.query_log",
    )
    query_ids: set[str] = set()
    for index, row in enumerate(query_rows, start=2):
        query_id = row.get("query_id", "").strip() or f"QUERY_ROW_{index}"
        if query_id in query_ids:
            errors.append(f"{query_id}: duplicate query_id")
        query_ids.add(query_id)
        for field in ("database", "exact_query"):
            _require_nonempty(row.get(field), f"{query_id}.{field}", errors, min_length=3)
            if PLACEHOLDER_RE.search(row.get(field, "")):
                errors.append(f"{query_id}.{field}: forbidden placeholder")
        _parse_iso_datetime(row.get("searched_at"), f"{query_id}.searched_at", errors)
        try:
            result_start = int(row.get("result_start", ""))
            result_end = int(row.get("result_end", ""))
            raw_count = int(row.get("raw_result_count", ""))
        except ValueError:
            errors.append(f"{query_id}: result_start/result_end/raw_result_count must be integers")
        else:
            if result_start < 1 or result_end < result_start or raw_count < result_end:
                errors.append(f"{query_id}: invalid result range/count contract")
        snapshot = _safe_relative_path(
            root,
            row.get("snapshot_path"),
            f"{query_id}.snapshot_path",
            errors,
        )
        if snapshot is not None:
            if not snapshot.is_file():
                errors.append(f"{query_id}: raw query snapshot missing: {snapshot}")
            else:
                try:
                    expected_bytes = int(row.get("snapshot_bytes", ""))
                except ValueError:
                    errors.append(f"{query_id}.snapshot_bytes: integer required")
                else:
                    if snapshot.stat().st_size != expected_bytes:
                        errors.append(f"{query_id}.snapshot_bytes: size mismatch")
                expected_sha = row.get("snapshot_sha256", "").upper()
                if _sha256(snapshot) != expected_sha:
                    errors.append(f"{query_id}.snapshot_sha256: hash mismatch")

    candidate_rows = _read_required_csv(
        root / "discovery" / "CANDIDATE_LEDGER.csv",
        (
            "candidate_id",
            "title",
            "primary_url",
            "source_database",
            "query_ids",
            "decision",
            "canonical_paper_id",
            "exclusion_reason",
        ),
        errors,
        label="discovery.candidate_ledger",
    )
    seen_candidates: set[str] = set()
    included_ids: list[str] = []
    excluded = 0
    for index, row in enumerate(candidate_rows, start=2):
        candidate_id = row.get("candidate_id", "").strip() or f"CANDIDATE_ROW_{index}"
        if candidate_id in seen_candidates:
            errors.append(f"{candidate_id}: duplicate candidate_id")
        seen_candidates.add(candidate_id)
        _require_nonempty(row.get("title"), f"{candidate_id}.title", errors, min_length=5)
        parsed_url = urlparse(row.get("primary_url", ""))
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            errors.append(f"{candidate_id}.primary_url: valid HTTP(S) URL required")
        referenced_queries = {
            value.strip() for value in row.get("query_ids", "").split(";") if value.strip()
        }
        if not referenced_queries:
            errors.append(f"{candidate_id}.query_ids: at least one query required")
        unknown_queries = sorted(referenced_queries - query_ids)
        if unknown_queries:
            errors.append(f"{candidate_id}.query_ids: unknown query IDs {unknown_queries}")
        decision = row.get("decision")
        if decision == "INCLUDED":
            paper_id = row.get("canonical_paper_id", "")
            included_ids.append(paper_id)
            if not _valid_missing_marker(row.get("exclusion_reason", "")):
                errors.append(
                    f"{candidate_id}.exclusion_reason: included row must state NOT_APPLICABLE_WITH_REASON"
                )
        elif decision == "EXCLUDED":
            excluded += 1
            if not _valid_missing_marker(row.get("canonical_paper_id", "")):
                errors.append(
                    f"{candidate_id}.canonical_paper_id: excluded row must state NOT_APPLICABLE_WITH_REASON"
                )
            reason = row.get("exclusion_reason", "")
            _require_nonempty(reason, f"{candidate_id}.exclusion_reason", errors, min_length=20)
            if PLACEHOLDER_RE.search(reason):
                errors.append(f"{candidate_id}.exclusion_reason: forbidden placeholder")
        else:
            errors.append(f"{candidate_id}.decision: expected INCLUDED or EXCLUDED")

    expected_ids = {paper.paper_id for paper in papers}
    observed_ids = set(included_ids)
    if observed_ids != expected_ids or len(included_ids) != len(expected_ids):
        errors.append(
            "discovery.candidate_ledger: INCLUDED canonical IDs must match the corpus exactly once; "
            f"missing={sorted(expected_ids - observed_ids)}, extra={sorted(observed_ids - expected_ids)}"
        )
    _raise_if_errors(errors)
    return {
        "queries": len(query_rows),
        "candidates": len(candidate_rows),
        "included": len(included_ids),
        "excluded": excluded,
    }


def _expected_artifacts(papers: Sequence[PaperEvidence]) -> dict[tuple[str, str], Mapping[str, Any]]:
    expected: dict[tuple[str, str], Mapping[str, Any]] = {}
    for paper in papers:
        expected[(paper.paper_id, "BROAD_SOURCE")] = paper.metadata["source_artifact"]
        if paper.tier in {"SCREENED", "DEEP"}:
            expected[(paper.paper_id, "METHOD_SOURCE")] = paper.metadata["screened"]["method_source"]
        if paper.tier == "DEEP":
            expected[(paper.paper_id, "DEEP_FULL_TEXT")] = paper.metadata["deep"]["full_text"]
    return expected


def validate_source_acquisitions(
    corpus_root: str | Path,
    papers: Sequence[PaperEvidence],
) -> int:
    """Require a primary-source acquisition record for every counted artifact role."""

    root = Path(corpus_root).resolve()
    errors: list[str] = []
    rows = _read_required_csv(
        root / "SOURCE_ACQUISITION.csv",
        (
            "paper_id",
            "artifact_role",
            "path",
            "url",
            "retrieved_at",
            "http_status",
            "content_type",
            "bytes",
            "sha256",
            "retrieval_method",
            "source_authority",
        ),
        errors,
        label="source_acquisition",
    )
    expected = _expected_artifacts(papers)
    observed: dict[tuple[str, str], dict[str, str]] = {}
    allowed_authorities = {"PRIMARY_PUBLISHER", "AUTHOR_HOSTED", "OFFICIAL_REPOSITORY"}
    allowed_methods = {"HTTP_DOWNLOAD", "BROWSER_SAVE", "OFFICIAL_API_SNAPSHOT"}
    for index, row in enumerate(rows, start=2):
        key = (row.get("paper_id", ""), row.get("artifact_role", ""))
        label = f"{key[0] or 'ROW_'+str(index)}.{key[1] or 'artifact_role'}"
        if key in observed:
            errors.append(f"{label}: duplicate source acquisition role")
        observed[key] = row
        if key not in expected:
            errors.append(f"{label}: acquisition row is not required by the counted corpus")
            continue
        artifact = expected[key]
        for field in ("path", "sha256", "bytes"):
            if str(row.get(field, "")).upper() != str(artifact.get(field, "")).upper():
                errors.append(f"{label}.{field}: does not match note artifact identity")
        parsed_url = urlparse(row.get("url", ""))
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            errors.append(f"{label}.url: valid primary HTTP(S) URL required")
        _parse_iso_datetime(row.get("retrieved_at"), f"{label}.retrieved_at", errors)
        if row.get("http_status") != "200":
            errors.append(f"{label}.http_status: expected 200")
        _require_nonempty(row.get("content_type"), f"{label}.content_type", errors, min_length=4)
        if row.get("retrieval_method") not in allowed_methods:
            errors.append(f"{label}.retrieval_method: invalid method")
        if row.get("source_authority") not in allowed_authorities:
            errors.append(f"{label}.source_authority: secondary or unknown sources are not accepted")
    missing = sorted(set(expected) - set(observed))
    if missing:
        errors.append(f"source_acquisition: missing artifact roles {missing}")
    _raise_if_errors(errors)
    return len(rows)


def validate_random_audit(
    corpus_root: str | Path,
    papers: Sequence[PaperEvidence],
    *,
    seed: str,
) -> dict[str, int]:
    """Validate the fixed nested 10%/15%/20% manual audit receipt."""

    root = Path(corpus_root).resolve()
    errors: list[str] = []
    rows = _read_required_csv(
        root / "validation" / "RANDOM_AUDIT.csv",
        (
            "paper_id",
            "audit_tier",
            "audited_at",
            "identity_pass",
            "source_hash_pass",
            "relevance_pass",
            "reading_depth_pass",
            "locator_pass",
            "outcome",
            "audit_note",
        ),
        errors,
        label="random_audit",
    )
    expected = deterministic_audit_ids(papers, seed=seed)
    expected_keys = {
        (paper_id, tier.upper()) for tier, paper_ids in expected.items() for paper_id in paper_ids
    }
    observed_keys: list[tuple[str, str]] = []
    for index, row in enumerate(rows, start=2):
        key = (row.get("paper_id", ""), row.get("audit_tier", ""))
        observed_keys.append(key)
        label = f"random_audit.{key[0] or index}.{key[1]}"
        _parse_iso_datetime(row.get("audited_at"), f"{label}.audited_at", errors)
        for field in (
            "identity_pass",
            "source_hash_pass",
            "relevance_pass",
            "reading_depth_pass",
            "locator_pass",
            "outcome",
        ):
            if row.get(field) != "PASS":
                errors.append(f"{label}.{field}: PASS required, observed {row.get(field)!r}")
        _require_nonempty(row.get("audit_note"), f"{label}.audit_note", errors, min_length=40)
        if PLACEHOLDER_RE.search(row.get("audit_note", "")):
            errors.append(f"{label}.audit_note: forbidden placeholder")
    if set(observed_keys) != expected_keys or len(observed_keys) != len(expected_keys):
        errors.append(
            "random_audit: rows must match the deterministic nested sample exactly; "
            f"missing={sorted(expected_keys - set(observed_keys))}, "
            f"extra={sorted(set(observed_keys) - expected_keys)}"
        )
    _raise_if_errors(errors)
    return {
        "rows": len(rows),
        "broad": len(expected["broad"]),
        "screened": len(expected["screened"]),
        "deep": len(expected["deep"]),
    }


def validate_second_pass(
    corpus_root: str | Path,
    papers: Sequence[PaperEvidence],
    *,
    minimum: int = 30,
    required_rqs: set[str] | None = None,
    min_elapsed_hours: float = 24.0,
) -> dict[str, Any]:
    """Validate the time-separated re-read of the critical deep-paper subset."""

    if minimum < 1:
        raise ValueError("minimum must be positive")
    root = Path(corpus_root).resolve()
    errors: list[str] = []
    rows = _read_required_csv(
        root / "validation" / "SECOND_PASS_30.csv",
        (
            "rank",
            "paper_id",
            "priority_reason",
            "first_read_at",
            "second_read_at",
            "pdf_sha256",
            "sections_rechecked",
            "claims_confirmed",
            "claims_revised",
            "contradictions",
            "stage1_effect",
            "same_reviewer_disclosed",
            "outcome",
        ),
        errors,
        label="second_pass",
    )
    if len(rows) < minimum:
        errors.append(f"second_pass: expected at least {minimum} papers, observed {len(rows)}")
    deep_lookup = {paper.paper_id: paper for paper in papers if paper.tier == "DEEP"}
    observed_ids: list[str] = []
    observed_ranks: list[int] = []
    covered_rqs: set[str] = set()
    unique_text: dict[str, set[str]] = {
        "priority_reason": set(),
        "claims_confirmed": set(),
        "claims_revised": set(),
        "contradictions": set(),
        "stage1_effect": set(),
    }
    for index, row in enumerate(rows, start=2):
        paper_id = row.get("paper_id", "")
        observed_ids.append(paper_id)
        label = f"second_pass.{paper_id or index}"
        try:
            rank = int(row.get("rank", ""))
        except ValueError:
            rank = -1
            errors.append(f"{label}.rank: integer required")
        observed_ranks.append(rank)
        paper = deep_lookup.get(paper_id)
        if paper is None:
            errors.append(f"{label}: paper must belong to DEEP tier")
            continue
        covered_rqs.update(map(str, paper.metadata.get("rq_ids", [])))
        declared_first = row.get("first_read_at", "")
        expected_first = str(paper.metadata["deep"].get("first_read_at", ""))
        if declared_first != expected_first:
            errors.append(f"{label}.first_read_at: does not match deep note")
        first = _parse_iso_datetime(declared_first, f"{label}.first_read_at", errors)
        second = _parse_iso_datetime(row.get("second_read_at"), f"{label}.second_read_at", errors)
        if first is not None and second is not None:
            elapsed = (second - first).total_seconds() / 3600.0
            if elapsed < min_elapsed_hours:
                errors.append(
                    f"{label}: second pass must be separated by at least {min_elapsed_hours:g} hours; observed {elapsed:g}"
                )
        expected_sha = str(paper.metadata["deep"]["full_text"].get("sha256", "")).upper()
        if row.get("pdf_sha256", "").upper() != expected_sha:
            errors.append(f"{label}.pdf_sha256: does not match first-pass PDF")
        for field in unique_text:
            value = row.get(field, "")
            _require_nonempty(value, f"{label}.{field}", errors, min_length=20)
            normalized = _normalize_text(value)
            if normalized in unique_text[field]:
                errors.append(f"{label}.{field}: reused verbatim in second-pass records")
            unique_text[field].add(normalized)
            if PLACEHOLDER_RE.search(value):
                errors.append(f"{label}.{field}: forbidden placeholder")
        _require_nonempty(
            row.get("sections_rechecked"),
            f"{label}.sections_rechecked",
            errors,
            min_length=20,
        )
        if row.get("same_reviewer_disclosed", "").casefold() != "true":
            errors.append(f"{label}.same_reviewer_disclosed: true required")
        if row.get("outcome") != "PASS":
            errors.append(f"{label}.outcome: PASS required")
    if len(set(observed_ids)) != len(observed_ids):
        errors.append("second_pass.paper_id: duplicate papers")
    if sorted(observed_ranks) != list(range(1, len(rows) + 1)):
        errors.append("second_pass.rank: ranks must be contiguous 1..N")
    required = required_rqs if required_rqs is not None else set(RESEARCH_QUESTIONS)
    missing_rqs = sorted(required - covered_rqs)
    if missing_rqs:
        errors.append(f"second_pass: critical subset does not cover research questions {missing_rqs}")
    _raise_if_errors(errors)
    return {"papers": len(rows), "covered_rqs": sorted(covered_rqs)}


def audit_completion(
    corpus_root: str | Path,
    *,
    expected: TierCounts = TierCounts(),
    inspect_pdf_pages: bool = True,
    audit_seed: str = "stage1-literature-500-300-100-v2-20260809",
    second_pass_minimum: int = 30,
    second_pass_required_rqs: set[str] | None = None,
    second_pass_min_elapsed_hours: float = 24.0,
) -> CompletionAuditReport:
    """Run the only completion gate accepted for the 500/300/100 evidence set."""

    corpus = audit_corpus(
        corpus_root,
        expected=expected,
        inspect_pdf_pages=inspect_pdf_pages,
    )
    gates: dict[str, str] = {"corpus": "PASS" if not corpus.errors else "FAIL"}
    errors = list(corpus.errors)
    downstream = ("discovery", "source_acquisition", "random_audit", "second_pass")
    if corpus.errors:
        gates.update({name: "BLOCKED_BY_CORPUS" for name in downstream})
    else:
        checks = (
            ("discovery", lambda: validate_discovery_evidence(corpus_root, corpus.papers)),
            (
                "source_acquisition",
                lambda: validate_source_acquisitions(corpus_root, corpus.papers),
            ),
            (
                "random_audit",
                lambda: validate_random_audit(corpus_root, corpus.papers, seed=audit_seed),
            ),
            (
                "second_pass",
                lambda: validate_second_pass(
                    corpus_root,
                    corpus.papers,
                    minimum=second_pass_minimum,
                    required_rqs=(
                        second_pass_required_rqs
                        if second_pass_required_rqs is not None
                        else set(RESEARCH_QUESTIONS)
                    ),
                    min_elapsed_hours=second_pass_min_elapsed_hours,
                ),
            ),
        )
        for name, check in checks:
            try:
                check()
            except LiteratureEvidenceError as exc:
                gates[name] = "FAIL"
                errors.extend(f"{name}: {message}" for message in exc.errors)
            else:
                gates[name] = "PASS"
    status = "PASS" if not errors and all(value == "PASS" for value in gates.values()) else "INCOMPLETE"
    return CompletionAuditReport(
        status=status,
        gates=gates,
        counts=corpus.counts,
        errors=tuple(errors),
    )
