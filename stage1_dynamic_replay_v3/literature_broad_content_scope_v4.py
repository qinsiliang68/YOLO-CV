"""Fail-closed BROAD source-scope and canonical-version audit.

This audit is intentionally independent from tier selection and publication.  It
does not grant BROAD credit, rewrite notes, select reserves, or promote a staging
corpus.  A PASS requires source-bound evidence for every declared BROAD reading
scope and no unresolved canonical-version identity.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA_VERSION = "4.0"
REQUIRED_SCOPES = ("TITLE", "ABSTRACT", "PROBLEM", "METHOD_OVERVIEW", "CONCLUSION")
MISSING_PREFIXES = (
    "NOT_APPLICABLE_WITH_REASON:",
    "NOT_ASSESSED_AT_BROAD_LEVEL",
    "NOT_REPORTED_BY_PAPER",
)
_TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "via",
    "with",
}
_EVIDENCE_MARKER = "<!-- STAGE1_EVIDENCE_V2 -->"
_JSON_OPEN = "```json\n"
_JSON_CLOSE = "\n```"


@dataclass(frozen=True)
class ContentScopeFinding:
    severity: str
    code: str
    field: str
    message: str
    paper_id: str | None = None
    related_paper_ids: tuple[str, ...] = ()
    evidence: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "field": self.field,
            "message": self.message,
        }
        if self.paper_id is not None:
            payload["paper_id"] = self.paper_id
        if self.related_paper_ids:
            payload["related_paper_ids"] = list(self.related_paper_ids)
        if self.evidence:
            payload["evidence"] = dict(self.evidence)
        return payload


@dataclass(frozen=True)
class BroadContentScopeReport:
    status: str
    promotion_allowed: bool
    corpus_root: Path
    expected_count: int | None
    observed_count: int
    findings: tuple[ContentScopeFinding, ...]
    paper_ids: tuple[str, ...]
    source_kind_counts: Mapping[str, int]
    sections_checked_signatures: Mapping[str, int]

    def as_dict(self) -> dict[str, Any]:
        per_paper: dict[str, dict[str, Any]] = {}
        for paper_id in self.paper_ids:
            paper_findings = [
                finding.as_dict()
                for finding in self.findings
                if finding.paper_id == paper_id
            ]
            severities = {item["severity"] for item in paper_findings}
            paper_status = (
                "FAIL"
                if "FAIL" in severities
                else "REVIEW_REQUIRED"
                if "REVIEW_REQUIRED" in severities
                else "PASS"
            )
            per_paper[paper_id] = {
                "status": paper_status,
                "findings": paper_findings,
            }
        corpus_findings = [
            finding.as_dict() for finding in self.findings if finding.paper_id is None
        ]
        fail_findings = sum(item.severity == "FAIL" for item in self.findings)
        review_findings = sum(
            item.severity == "REVIEW_REQUIRED" for item in self.findings
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "promotion_allowed": self.promotion_allowed,
            "formal_broad_increment": 0,
            "formal_registry_published": False,
            "corpus_root": str(self.corpus_root),
            "expected_count": self.expected_count,
            "observed_count": self.observed_count,
            "counts": {
                "fail_findings": fail_findings,
                "review_required_findings": review_findings,
                "papers_failed": sum(
                    item["status"] == "FAIL" for item in per_paper.values()
                ),
                "papers_review_required": sum(
                    item["status"] == "REVIEW_REQUIRED"
                    for item in per_paper.values()
                ),
                "papers_passed": sum(
                    item["status"] == "PASS" for item in per_paper.values()
                ),
            },
            "required_scopes": list(REQUIRED_SCOPES),
            "source_kind_counts": dict(self.source_kind_counts),
            "sections_checked_signatures": dict(self.sections_checked_signatures),
            "corpus_findings": corpus_findings,
            "per_paper": per_paper,
        }


@dataclass(frozen=True)
class _SourceContent:
    kind: str
    text: str
    title_text: str
    abstract_text: str | None


SourceLoader = Callable[[Path], _SourceContent]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _is_missing(value: str) -> bool:
    clean = value.strip().casefold()
    return not clean or any(
        clean.startswith(prefix.casefold()) for prefix in MISSING_PREFIXES
    )


def _normalize_identifier(value: str) -> str | None:
    clean = value.strip().casefold()
    if _is_missing(clean):
        return None
    clean = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", clean).rstrip("/")
    return clean or None


def _arxiv_identity(row: Mapping[str, str]) -> str | None:
    values = (row.get("arxiv_id", ""), row.get("doi", ""), row.get("primary_url", ""))
    patterns = (
        r"(?i)(?:arxiv(?:\.org)?/(?:abs|pdf)/)([a-z-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?",
        r"(?i)(?:arxiv[:.\s])([a-z-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?",
        r"(?i)^([a-z-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?$",
    )
    for value in values:
        if _is_missing(value):
            continue
        for pattern in patterns:
            match = re.search(pattern, value)
            if match:
                return match.group(1).casefold()
    return None


def _title_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) >= 2 and token not in _TITLE_STOPWORDS
    )


def _title_coverage(title: str, source_text: str) -> float:
    expected = set(_title_tokens(title))
    if len(expected) < 2:
        return 0.0
    observed = set(re.findall(r"[a-z0-9]+", source_text.casefold()))
    return len(expected & observed) / len(expected)


def _author_identity(value: str) -> tuple[str, ...]:
    authors = []
    for raw in value.split(";"):
        normalized = re.sub(r"[^a-z0-9]+", "", raw.casefold())
        if normalized:
            authors.append(normalized)
    return tuple(sorted(set(authors)))


def _normalize_quote(value: str) -> str:
    return re.sub(r"[^\w]+", "", html.unescape(value).casefold(), flags=re.UNICODE)


def _strip_markup(value: str) -> str:
    without_code = re.sub(
        r"(?is)<(?:script|style).*?>.*?</(?:script|style)>", " ", value
    )
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"(?s)<[^>]+>", " ", without_code))).strip()


def _extract_html_abstract(raw: str) -> str | None:
    meta_tags = re.findall(r"(?is)<meta\b[^>]*>", raw)
    for tag in meta_tags:
        attributes = {
            key.casefold(): html.unescape(value)
            for key, _, value in re.findall(
                r"(?i)([\w:-]+)\s*=\s*(['\"])(.*?)\2", tag
            )
        }
        name = (attributes.get("name") or attributes.get("property") or "").casefold()
        if name in {
            "citation_abstract",
            "dc.description",
            "description",
            "og:description",
            "twitter:description",
        }:
            candidate = _strip_markup(attributes.get("content", ""))
            if len(candidate) >= 40:
                return candidate
    json_abstract = re.search(
        r'(?is)["\']abstract["\']\s*:\s*["\'](.{40,8000}?)["\']\s*[,}]',
        raw,
    )
    if json_abstract:
        return _strip_markup(json_abstract.group(1).replace(r"\n", " "))
    section = re.search(
        r"(?is)(?:<h[1-6][^>]*>|<strong[^>]*>)\s*abstract\s*</.*?>"
        r"(.{40,12000}?)(?=<h[1-6]\b|</section>|\Z)",
        raw,
    )
    if section:
        return _strip_markup(section.group(1))
    return None


def _extract_pdf_abstract(text: str) -> str | None:
    match = re.search(
        r"(?is)(?:^|\n)\s*abstract\s*[:\-—]?\s*(.{80,8000}?)"
        r"(?=\n\s*(?:\d+\.?\s*)?(?:introduction|keywords?|index terms)\b)",
        text,
    )
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    fallback = re.search(r"(?is)\babstract\b\s*[:\-—]?\s*(.{120,3000})", text)
    if fallback:
        return re.sub(r"\s+", " ", fallback.group(1)).strip()
    return None


def _default_source_loader(path: Path) -> _SourceContent:
    data = path.read_bytes()
    if data.startswith(b"%PDF-"):
        executable = shutil.which("pdftotext")
        if executable is None:
            raise RuntimeError("pdftotext is required for PDF content-scope validation")
        process = subprocess.run(
            [executable, "-layout", str(path), "-"],
            check=False,
            capture_output=True,
        )
        if process.returncode != 0:
            detail = process.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"pdftotext failed: {detail}")
        text = process.stdout.decode("utf-8", errors="replace")
        return _SourceContent(
            kind="PDF_FULL_TEXT",
            text=text,
            title_text=text,
            abstract_text=_extract_pdf_abstract(text),
        )

    raw = data.decode("utf-8", errors="strict")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        clean = _strip_markup(raw)
        return _SourceContent(
            kind="HTML",
            text=clean,
            title_text=clean,
            abstract_text=_extract_html_abstract(raw),
        )
    if isinstance(payload, Mapping) and isinstance(payload.get("message"), Mapping):
        message = payload["message"]
        titles = message.get("title")
        if isinstance(titles, list):
            title_text = " ".join(str(item) for item in titles)
        else:
            title_text = str(titles or "")
        abstract_raw = message.get("abstract")
        abstract = (
            _strip_markup(str(abstract_raw))
            if isinstance(abstract_raw, str) and abstract_raw.strip()
            else None
        )
        return _SourceContent(
            kind="CROSSREF_JSON",
            text=" ".join(value for value in (title_text, abstract or "") if value),
            title_text=title_text,
            abstract_text=abstract,
        )
    clean = _strip_markup(raw)
    return _SourceContent(
        kind="JSON_OR_TEXT",
        text=clean,
        title_text=clean,
        abstract_text=None,
    )


def _safe_path(root: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        raise ValueError("path must be corpus-relative")
    resolved = (root / path).resolve()
    resolved.relative_to(root)
    return resolved


def _read_registry(root: Path) -> list[dict[str, str]]:
    path = root / "CANONICAL_WORKS.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_note(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8", errors="strict")
    if text.count(_EVIDENCE_MARKER) != 1 or text.count(_JSON_OPEN) != 1:
        raise ValueError("note must contain exactly one STAGE1_EVIDENCE_V2 JSON block")
    raw = text.split(_JSON_OPEN, 1)[1].split(_JSON_CLOSE, 1)[0]
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("note JSON root must be an object")
    return payload


def _add_identity_group_findings(
    findings: list[ContentScopeFinding],
    rows: Sequence[Mapping[str, str]],
    *,
    key: Callable[[Mapping[str, str]], str | None],
    code: str,
    field: str,
) -> None:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        identity = key(row)
        if identity:
            grouped[identity].append(row["paper_id"])
    for identity, paper_ids in sorted(grouped.items()):
        if len(paper_ids) < 2:
            continue
        for paper_id in sorted(paper_ids):
            related = tuple(sorted(set(paper_ids) - {paper_id}))
            findings.append(
                ContentScopeFinding(
                    severity="FAIL",
                    code=code,
                    paper_id=paper_id,
                    field=field,
                    related_paper_ids=related,
                    message=f"canonical identity is shared with {', '.join(related)}",
                    evidence={"normalized_identity": identity},
                )
            )


def _add_near_version_findings(
    findings: list[ContentScopeFinding], rows: Sequence[Mapping[str, str]]
) -> None:
    explicit_pairs = {
        tuple(sorted((finding.paper_id, related)))
        for finding in findings
        if finding.paper_id is not None
        and finding.code
        in {"DUPLICATE_SOURCE_SHA256", "DUPLICATE_DOI", "DUPLICATE_ARXIV_ID"}
        for related in finding.related_paper_ids
    }
    for index, left in enumerate(rows):
        left_authors = _author_identity(left.get("authors", ""))
        left_tokens = _title_tokens(left.get("title", ""))
        if not left_authors or len(left_tokens) < 2:
            continue
        for right in rows[index + 1 :]:
            pair = tuple(sorted((left["paper_id"], right["paper_id"])))
            if pair in explicit_pairs:
                continue
            if left_authors != _author_identity(right.get("authors", "")):
                continue
            try:
                year_delta = abs(int(left.get("year", "")) - int(right.get("year", "")))
            except ValueError:
                continue
            if year_delta > 2:
                continue
            right_tokens = _title_tokens(right.get("title", ""))
            ratio = SequenceMatcher(None, " ".join(left_tokens), " ".join(right_tokens)).ratio()
            union = set(left_tokens) | set(right_tokens)
            jaccard = len(set(left_tokens) & set(right_tokens)) / max(1, len(union))
            if year_delta == 0 and ratio >= 0.97 and jaccard >= 0.70:
                severity = "FAIL"
                code = "CANONICAL_VERSION_DUPLICATE"
                message = "near-identical title with the same authors and year is a canonical version duplicate"
            elif ratio >= 0.82 and jaccard >= 0.65:
                severity = "REVIEW_REQUIRED"
                code = "CANONICAL_VERSION_REVIEW_REQUIRED"
                message = "similar title and matching authors/year window require explicit version resolution"
            else:
                continue
            evidence = {
                "title_similarity": round(ratio, 6),
                "title_token_jaccard": round(jaccard, 6),
                "year_delta": year_delta,
            }
            for paper_id, related in ((left["paper_id"], right["paper_id"]), (right["paper_id"], left["paper_id"])):
                findings.append(
                    ContentScopeFinding(
                        severity=severity,
                        code=code,
                        paper_id=paper_id,
                        field="canonical_version_identity",
                        related_paper_ids=(related,),
                        message=message,
                        evidence=evidence,
                    )
                )


def _validate_scope_evidence(
    *,
    paper_id: str,
    title: str,
    metadata: Mapping[str, Any],
    source: _SourceContent,
    findings: list[ContentScopeFinding],
) -> None:
    reading = metadata.get("reading")
    if not isinstance(reading, Mapping):
        findings.append(
            ContentScopeFinding(
                severity="FAIL",
                code="READING_OBJECT_MISSING",
                paper_id=paper_id,
                field="reading",
                message="note reading object is required",
            )
        )
        return
    declared = reading.get("scopes")
    declared_set = set(map(str, declared)) if isinstance(declared, list) else set()
    for scope in REQUIRED_SCOPES:
        if scope not in declared_set:
            findings.append(
                ContentScopeFinding(
                    severity="FAIL",
                    code="REQUIRED_SCOPE_NOT_DECLARED",
                    paper_id=paper_id,
                    field="reading.scopes",
                    message=f"required BROAD scope {scope} is not declared",
                )
            )

    coverage = _title_coverage(title, source.title_text or source.text)
    if coverage < 0.60:
        findings.append(
            ContentScopeFinding(
                severity="FAIL",
                code="SOURCE_TITLE_MISMATCH",
                paper_id=paper_id,
                field="reading.source_scope_evidence.TITLE",
                message="source bytes do not support the registered title identity",
                evidence={"title_token_coverage": round(coverage, 6)},
            )
        )
    if not source.abstract_text:
        findings.append(
            ContentScopeFinding(
                severity="FAIL",
                code="SOURCE_ABSTRACT_MISSING",
                paper_id=paper_id,
                field="reading.source_scope_evidence.ABSTRACT",
                message=(
                    "Crossref JSON has no message.abstract"
                    if source.kind == "CROSSREF_JSON"
                    else "source has no machine-verifiable primary abstract"
                ),
                evidence={"source_kind": source.kind},
            )
        )

    scope_evidence = reading.get("source_scope_evidence")
    if not isinstance(scope_evidence, Mapping):
        findings.append(
            ContentScopeFinding(
                severity="FAIL",
                code="FIXED_SECTIONS_CHECKED_NOT_CONTENT_EVIDENCE",
                paper_id=paper_id,
                field="reading.sections_checked",
                message=(
                    "sections_checked is a declaration only; field-specific source quotes and locators are required"
                ),
            )
        )
        scope_evidence = {}
    normalized_full = _normalize_quote(source.text)
    normalized_abstract = _normalize_quote(source.abstract_text or "")
    normalized_title = _normalize_quote(source.title_text or source.text)
    for scope in REQUIRED_SCOPES:
        field = f"reading.source_scope_evidence.{scope}"
        item = scope_evidence.get(scope)
        if not isinstance(item, Mapping):
            findings.append(
                ContentScopeFinding(
                    severity="FAIL",
                    code="SOURCE_SCOPE_EVIDENCE_MISSING",
                    paper_id=paper_id,
                    field=field,
                    message=f"{scope} requires a source-bound locator and quote",
                )
            )
            continue
        locator = str(item.get("locator") or "").strip()
        quote = str(item.get("quote") or "").strip()
        if len(locator) < 5:
            findings.append(
                ContentScopeFinding(
                    severity="FAIL",
                    code="SOURCE_SCOPE_LOCATOR_MISSING",
                    paper_id=paper_id,
                    field=f"{field}.locator",
                    message=f"{scope} locator must identify the source location",
                )
            )
        normalized_quote = _normalize_quote(quote)
        if len(normalized_quote) < 8:
            findings.append(
                ContentScopeFinding(
                    severity="FAIL",
                    code="SOURCE_SCOPE_QUOTE_MISSING",
                    paper_id=paper_id,
                    field=f"{field}.quote",
                    message=f"{scope} quote is too short or empty",
                )
            )
            continue
        if scope == "TITLE":
            searchable = normalized_title
        elif scope == "ABSTRACT" or source.kind in {"CROSSREF_JSON", "HTML"}:
            searchable = normalized_abstract
        else:
            searchable = normalized_full
        if normalized_quote not in searchable:
            findings.append(
                ContentScopeFinding(
                    severity="FAIL",
                    code="SOURCE_SCOPE_QUOTE_NOT_FOUND",
                    paper_id=paper_id,
                    field=f"{field}.quote",
                    message=f"{scope} quote is not present in the allowed source content",
                    evidence={"source_kind": source.kind, "locator": locator},
                )
            )


def audit_broad_content_scope(
    corpus_root: str | Path,
    *,
    expected_count: int | None = 500,
    source_loader: SourceLoader = _default_source_loader,
) -> BroadContentScopeReport:
    """Audit BROAD source scope and canonical identity without mutating the corpus."""

    root = Path(corpus_root).resolve()
    findings: list[ContentScopeFinding] = []
    try:
        rows = _read_registry(root)
    except (OSError, csv.Error) as exc:
        findings.append(
            ContentScopeFinding(
                severity="FAIL",
                code="REGISTRY_UNREADABLE",
                field="CANONICAL_WORKS.csv",
                message=str(exc),
            )
        )
        rows = []
    paper_ids = tuple(row.get("paper_id", "") for row in rows if row.get("paper_id"))
    if expected_count is not None and len(rows) != expected_count:
        findings.append(
            ContentScopeFinding(
                severity="FAIL",
                code="COUNT_MISMATCH",
                field="CANONICAL_WORKS.csv",
                message=f"expected exactly {expected_count} rows, observed {len(rows)}",
            )
        )

    _add_identity_group_findings(
        findings,
        rows,
        key=lambda row: (
            row.get("source_sha256", "").strip().upper()
            if re.fullmatch(r"[0-9A-Fa-f]{64}", row.get("source_sha256", "").strip())
            else None
        ),
        code="DUPLICATE_SOURCE_SHA256",
        field="source_sha256",
    )
    _add_identity_group_findings(
        findings,
        rows,
        key=lambda row: _normalize_identifier(row.get("doi", "")),
        code="DUPLICATE_DOI",
        field="doi",
    )
    _add_identity_group_findings(
        findings,
        rows,
        key=_arxiv_identity,
        code="DUPLICATE_ARXIV_ID",
        field="arxiv_id",
    )
    _add_near_version_findings(findings, rows)

    source_kind_counts: Counter[str] = Counter()
    sections_signatures: Counter[str] = Counter()
    for row in rows:
        paper_id = row.get("paper_id", "") or "REGISTRY_ROW_UNKNOWN"
        try:
            note_path = _safe_path(root, row.get("note_path", ""))
            metadata = _read_note(note_path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            findings.append(
                ContentScopeFinding(
                    severity="FAIL",
                    code="NOTE_UNREADABLE",
                    paper_id=paper_id,
                    field="note_path",
                    message=str(exc),
                )
            )
            continue
        reading = metadata.get("reading")
        if isinstance(reading, Mapping):
            sections = reading.get("sections_checked")
            signature = json.dumps(sections, ensure_ascii=False, sort_keys=True)
            sections_signatures[signature] += 1
        try:
            source_path = _safe_path(root, row.get("source_path", ""))
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            actual_bytes = source_path.stat().st_size
            actual_sha = _sha256(source_path)
            if str(actual_bytes) != str(row.get("source_bytes", "")):
                findings.append(
                    ContentScopeFinding(
                        severity="FAIL",
                        code="SOURCE_BYTES_MISMATCH",
                        paper_id=paper_id,
                        field="source_bytes",
                        message="source byte count does not match the registry",
                        evidence={"expected": row.get("source_bytes", ""), "observed": actual_bytes},
                    )
                )
            if actual_sha != row.get("source_sha256", "").strip().upper():
                findings.append(
                    ContentScopeFinding(
                        severity="FAIL",
                        code="SOURCE_SHA256_MISMATCH",
                        paper_id=paper_id,
                        field="source_sha256",
                        message="source SHA-256 does not match the registry",
                        evidence={"expected": row.get("source_sha256", ""), "observed": actual_sha},
                    )
                )
            source = source_loader(source_path)
        except Exception as exc:  # fail closed on extraction/tool/source errors
            findings.append(
                ContentScopeFinding(
                    severity="FAIL",
                    code="SOURCE_CONTENT_UNREADABLE",
                    paper_id=paper_id,
                    field="source_path",
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        source_kind_counts[source.kind] += 1
        _validate_scope_evidence(
            paper_id=paper_id,
            title=row.get("title", ""),
            metadata=metadata,
            source=source,
            findings=findings,
        )

    if sections_signatures:
        signature, count = sections_signatures.most_common(1)[0]
        if count == len(rows):
            findings.append(
                ContentScopeFinding(
                    severity="INFO",
                    code="UNIFORM_SECTIONS_CHECKED_DECLARATION",
                    field="reading.sections_checked",
                    message=(
                        "all notes share one sections_checked declaration; it is reported for audit but grants no content credit"
                    ),
                    evidence={"paper_count": count, "signature": json.loads(signature)},
                )
            )

    severities = {finding.severity for finding in findings}
    status = (
        "FAIL"
        if "FAIL" in severities
        else "REVIEW_REQUIRED"
        if "REVIEW_REQUIRED" in severities
        else "PASS"
    )
    return BroadContentScopeReport(
        status=status,
        promotion_allowed=status == "PASS",
        corpus_root=root,
        expected_count=expected_count,
        observed_count=len(rows),
        findings=tuple(findings),
        paper_ids=paper_ids,
        source_kind_counts=dict(source_kind_counts),
        sections_checked_signatures=dict(sections_signatures),
    )


__all__ = [
    "BroadContentScopeReport",
    "ContentScopeFinding",
    "REQUIRED_SCOPES",
    "audit_broad_content_scope",
]
