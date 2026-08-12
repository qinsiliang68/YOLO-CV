"""Validate broad-screen source bytes without granting reading credit."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import html
import json
from pathlib import Path
import re
from collections.abc import Sequence
from typing import Callable

from stage1_dynamic_replay_v3.literature_source_inventory_v2 import (
    PdfProbeResult,
    probe_pdf_with_poppler,
)


class BroadSourceValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class BroadSourceValidationResult:
    status: str
    expected_count: int
    verified_count: int
    failed_count: int
    rows: tuple[dict[str, object], ...]
    missing_ids: tuple[str, ...]


PdfProbe = Callable[[Path], PdfProbeResult]
_STOPWORDS = {
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _ledger_paths(value: Path | Sequence[Path]) -> tuple[Path, ...]:
    if isinstance(value, Path):
        return (value,)
    paths = tuple(value)
    if not paths:
        raise BroadSourceValidationError("at least one ledger path is required")
    return paths


def _optional_ledger_paths(value: Path | Sequence[Path] | None) -> tuple[Path, ...]:
    if value is None:
        return ()
    return _ledger_paths(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _safe_path(root: Path, value: str) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        raise BroadSourceValidationError(f"source path must be relative: {value}")
    resolved_root = root.resolve()
    resolved = (resolved_root / raw).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise BroadSourceValidationError(f"source path escapes corpus root: {value}") from exc
    return resolved


def _title_coverage(title: str, source_text: str) -> float:
    expected = {
        token
        for token in re.findall(r"[a-z0-9]+", title.casefold())
        if len(token) >= 2 and token not in _STOPWORDS
    }
    if len(expected) < 2:
        raise BroadSourceValidationError(f"title has too few identity tokens: {title!r}")
    observed = set(re.findall(r"[a-z0-9]+", source_text.casefold()))
    return len(expected.intersection(observed)) / len(expected)


def _html_text(data: bytes, paper_id: str) -> str:
    decoded = data.decode("utf-8", errors="replace")
    lowered = decoded.casefold()
    if any(marker in lowered for marker in ("access denied", "just a moment", "captcha")):
        raise BroadSourceValidationError(f"HTML source is an access/error page for {paper_id}")
    without_code = re.sub(r"(?is)<(?:script|style).*?>.*?</(?:script|style)>", " ", decoded)
    text = html.unescape(re.sub(r"(?s)<[^>]+>", " ", without_code))
    return re.sub(r"\s+", " ", text).strip()


def validate_broad_source_batch(
    *,
    corpus_root: Path,
    queue_path: Path,
    acquisition_ledger: Path | Sequence[Path],
    failure_ledger: Path | Sequence[Path],
    supersession_ledger: Path | Sequence[Path] | None = None,
    pdf_probe: PdfProbe = probe_pdf_with_poppler,
    minimum_title_coverage: float = 0.60,
) -> BroadSourceValidationResult:
    """Validate all acquired members and report, rather than hide, missing sources."""

    queue_rows = _read_csv(queue_path)
    acquisitions = [
        row
        for path in _ledger_paths(acquisition_ledger)
        for row in _read_csv(path)
    ]
    failures = [
        row
        for path in _ledger_paths(failure_ledger)
        for row in _read_csv(path)
    ]
    supersession_rows = [
        row
        for path in _optional_ledger_paths(supersession_ledger)
        for row in _read_csv(path)
    ]
    if not queue_rows or not {"queue_id", "title"}.issubset(queue_rows[0]):
        raise BroadSourceValidationError("queue is empty or lacks queue_id/title")
    queue: dict[str, dict[str, str]] = {}
    for row in queue_rows:
        paper_id = row["queue_id"].strip()
        if not paper_id or paper_id in queue:
            raise BroadSourceValidationError(f"blank or duplicate queue_id: {paper_id!r}")
        queue[paper_id] = row

    supersessions: dict[str, dict[str, str]] = {}
    required_supersession_fields = {
        "paper_id",
        "superseded_sha256",
        "canonical_sha256",
        "reason",
    }
    for row in supersession_rows:
        if not required_supersession_fields.issubset(row):
            raise BroadSourceValidationError("supersession ledger lacks required columns")
        paper_id = row["paper_id"].strip()
        if not paper_id or paper_id in supersessions:
            raise BroadSourceValidationError(
                f"blank or duplicate supersession paper_id: {paper_id!r}"
            )
        if paper_id not in queue:
            raise BroadSourceValidationError(
                f"supersession ledger contains ID outside queue: {paper_id}"
            )
        superseded_sha = row["superseded_sha256"].strip().upper()
        canonical_sha = row["canonical_sha256"].strip().upper()
        if (
            not re.fullmatch(r"[0-9A-F]{64}", superseded_sha)
            or not re.fullmatch(r"[0-9A-F]{64}", canonical_sha)
            or superseded_sha == canonical_sha
            or len(row["reason"].strip()) < 12
        ):
            raise BroadSourceValidationError(
                f"invalid supersession hash binding or reason for {paper_id}"
            )
        supersessions[paper_id] = {
            **row,
            "superseded_sha256": superseded_sha,
            "canonical_sha256": canonical_sha,
        }

    acquisition_groups: dict[str, list[dict[str, str]]] = {}
    for row in acquisitions:
        paper_id = row.get("paper_id", "").strip()
        acquisition_groups.setdefault(paper_id, []).append(row)

    acquired: dict[str, dict[str, str]] = {}
    applied_supersessions: dict[str, dict[str, str]] = {}
    for paper_id, rows in acquisition_groups.items():
        by_sha: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_sha.setdefault(row.get("sha256", "").upper(), []).append(row)
        if len(by_sha) == 1:
            acquired[paper_id] = rows[0]
            continue
        binding = supersessions.get(paper_id)
        if binding is None:
            raise BroadSourceValidationError(
                f"conflicting acquisitions for {paper_id}: source hashes differ"
            )
        expected_hashes = {
            binding["superseded_sha256"],
            binding["canonical_sha256"],
        }
        if set(by_sha) != expected_hashes:
            raise BroadSourceValidationError(
                f"supersession hash binding does not match acquisitions for {paper_id}"
            )
        acquired[paper_id] = by_sha[binding["canonical_sha256"]][0]
        applied_supersessions[paper_id] = binding
    unused_supersessions = sorted(set(supersessions) - set(applied_supersessions))
    if unused_supersessions:
        raise BroadSourceValidationError(
            f"supersession ledger has no conflicting acquisition to resolve: {unused_supersessions}"
        )
    failed_ids = {row.get("paper_id", "").strip() for row in failures}
    failed_ids.discard("")
    if (set(acquired) | set(failed_ids)) - set(queue):
        raise BroadSourceValidationError("acquisition/failure ledger contains IDs outside queue")

    missing = sorted(set(queue) - set(acquired))
    unresolved_failures = failed_ids - set(acquired)
    if set(missing) != unresolved_failures:
        raise BroadSourceValidationError(
            f"missing source IDs are not exactly documented by failure ledger: {missing}"
        )

    verified: list[dict[str, object]] = []
    for paper_id in sorted(acquired):
        row = acquired[paper_id]
        source_path = _safe_path(corpus_root, row["path"])
        receipt_path = _safe_path(corpus_root, row["receipt_path"])
        if not source_path.is_file() or not receipt_path.is_file():
            raise BroadSourceValidationError(f"source/receipt pair missing for {paper_id}")
        observed_bytes = source_path.stat().st_size
        observed_sha = _sha256(source_path)
        if observed_bytes != int(row["bytes"]) or observed_sha != row["sha256"].upper():
            raise BroadSourceValidationError(f"source byte/hash mismatch for {paper_id}")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BroadSourceValidationError(f"invalid receipt for {paper_id}") from exc
        receipt_row = receipt.get("ledger_row") or {}
        checks = {
            "paper_id": receipt.get("paper_id") == paper_id,
            "artifact_role": receipt.get("artifact_role") == "BROAD_SOURCE",
            "bytes": int(receipt.get("bytes", -1)) == observed_bytes,
            "sha256": str(receipt.get("sha256", "")).upper() == observed_sha,
            "ledger_path": receipt_row.get("path") == row["path"],
        }
        if failed_checks := [name for name, passed in checks.items() if not passed]:
            raise BroadSourceValidationError(
                f"receipt identity mismatch for {paper_id}: {failed_checks}"
            )

        data = source_path.read_bytes()
        if data.startswith(b"%PDF-"):
            probe = pdf_probe(source_path)
            source_text = probe.first_pages_text
            source_format = "PDF"
            page_count: int | str = probe.page_count
            probe_tool = probe.probe_tool
        else:
            source_text = _html_text(data, paper_id)
            source_format = "HTML"
            page_count = "NOT_APPLICABLE_WITH_REASON:HTML source"
            probe_tool = "HTML_TEXT_IDENTITY"
        coverage = _title_coverage(queue[paper_id]["title"], source_text)
        if coverage < minimum_title_coverage:
            raise BroadSourceValidationError(
                f"source title identity mismatch for {paper_id}: coverage={coverage:.3f}"
            )
        verified.append(
            {
                "paper_id": paper_id,
                "title": queue[paper_id]["title"],
                "path": row["path"],
                "bytes": observed_bytes,
                "sha256": observed_sha,
                "source_format": source_format,
                "page_count": page_count,
                "title_token_coverage": f"{coverage:.6f}",
                "source_authority": row["source_authority"],
                "source_url": row["url"],
                "receipt_path": row["receipt_path"],
                "probe_tool": probe_tool,
                "source_superseded": paper_id in applied_supersessions,
                "superseded_sha256": (
                    applied_supersessions[paper_id]["superseded_sha256"]
                    if paper_id in applied_supersessions
                    else "NOT_APPLICABLE_WITH_REASON:no source supersession"
                ),
                "supersession_reason": (
                    applied_supersessions[paper_id]["reason"]
                    if paper_id in applied_supersessions
                    else "NOT_APPLICABLE_WITH_REASON:no source supersession"
                ),
                "reading_credit_granted": False,
            }
        )

    status = "PASS" if not missing else "INCOMPLETE_ACQUISITION"
    return BroadSourceValidationResult(
        status=status,
        expected_count=len(queue),
        verified_count=len(verified),
        failed_count=len(missing),
        rows=tuple(verified),
        missing_ids=tuple(missing),
    )
