"""Hash-bound full-text extraction for literature review evidence.

Extraction is a mechanical preprocessing step. It never grants BROAD,
SCREENED, or DEEP reading credit.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Callable


class LiteratureTextExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class LiteratureTextExtractionResult:
    status: str
    extracted_count: int
    rows: tuple[dict[str, object], ...]


PdfExtractor = Callable[[Path], tuple[str, str]]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _safe_path(root: Path, value: str | Path) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        raise LiteratureTextExtractionError(f"path must be relative: {value}")
    resolved_root = root.resolve()
    resolved = (resolved_root / raw).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise LiteratureTextExtractionError(f"path escapes corpus root: {value}") from exc
    return resolved


def _extract_pdf(path: Path) -> tuple[str, str]:
    executable = shutil.which("pdftotext")
    if not executable:
        raise LiteratureTextExtractionError("pdftotext is required for full-text extraction")
    process = subprocess.run(
        [executable, "-layout", "-enc", "UTF-8", str(path), "-"],
        check=False,
        capture_output=True,
        timeout=180,
    )
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise LiteratureTextExtractionError(f"pdftotext failed for {path.name}: {detail}")
    return process.stdout.decode("utf-8", errors="replace"), "POPPLER_PDFTOTEXT_LAYOUT_UTF8"


def _extract_html(data: bytes, paper_id: str) -> tuple[str, str]:
    decoded = data.decode("utf-8", errors="replace")
    lowered = decoded.casefold()
    if any(marker in lowered for marker in ("access denied", "just a moment", "captcha")):
        raise LiteratureTextExtractionError(f"HTML source is an access/error page for {paper_id}")
    without_code = re.sub(r"(?is)<(?:script|style).*?>.*?</(?:script|style)>", " ", decoded)
    text = html.unescape(re.sub(r"(?s)<[^>]+>", " ", without_code))
    return re.sub(r"[ \t]+", " ", text).strip(), "HTML_TAG_STRIP_V1"


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def extract_literature_text_batch(
    *,
    corpus_root: Path,
    validation_path: Path,
    output_dir: Path,
    pdf_extractor: PdfExtractor = _extract_pdf,
    minimum_characters: int = 500,
) -> LiteratureTextExtractionResult:
    if minimum_characters < 1:
        raise LiteratureTextExtractionError("minimum_characters must be positive")
    output_root = _safe_path(corpus_root, output_dir)
    rows = _read_csv(validation_path)
    if not rows:
        raise LiteratureTextExtractionError("validated source ledger is empty")
    required = {"paper_id", "path", "bytes", "sha256", "source_format", "title"}
    if not required.issubset(rows[0]):
        raise LiteratureTextExtractionError(
            f"validated source ledger lacks columns: {sorted(required - set(rows[0]))}"
        )

    seen: set[str] = set()
    extracted: list[dict[str, object]] = []
    validation_sha = _sha256_bytes(validation_path.read_bytes())
    for row in rows:
        paper_id = row["paper_id"].strip()
        if not paper_id or paper_id in seen:
            raise LiteratureTextExtractionError(f"blank or duplicate paper_id: {paper_id!r}")
        seen.add(paper_id)
        source_path = _safe_path(corpus_root, row["path"])
        source_data = source_path.read_bytes()
        source_sha = _sha256_bytes(source_data)
        if len(source_data) != int(row["bytes"]) or source_sha != row["sha256"].upper():
            raise LiteratureTextExtractionError(f"source identity mismatch for {paper_id}")

        source_format = row["source_format"].upper()
        if source_format == "PDF":
            text, tool = pdf_extractor(source_path)
        elif source_format == "HTML":
            text, tool = _extract_html(source_data, paper_id)
        else:
            raise LiteratureTextExtractionError(
                f"unsupported source format for {paper_id}: {source_format}"
            )
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"
        visible_characters = len(re.sub(r"\s+", "", text))
        if visible_characters < minimum_characters:
            raise LiteratureTextExtractionError(
                f"extracted text too short for {paper_id}: {visible_characters} characters"
            )

        text_relative = output_dir / f"{paper_id}.txt"
        receipt_relative = output_dir / f"{paper_id}.txt.receipt.json"
        text_path = _safe_path(corpus_root, text_relative)
        receipt_path = _safe_path(corpus_root, receipt_relative)
        text_bytes = text.encode("utf-8")
        text_sha = _sha256_bytes(text_bytes)
        _write_atomic(text_path, text_bytes)
        receipt = {
            "schema_version": "1.0",
            "paper_id": paper_id,
            "title": row["title"],
            "source_path": row["path"],
            "source_bytes": len(source_data),
            "source_sha256": source_sha,
            "validation_ledger_sha256": validation_sha,
            "text_path": text_relative.as_posix(),
            "text_bytes": len(text_bytes),
            "text_sha256": text_sha,
            "visible_characters": visible_characters,
            "extraction_tool": tool,
            "reading_credit_granted": False,
        }
        _write_atomic(
            receipt_path,
            (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
        extracted.append(
            {
                "paper_id": paper_id,
                "title": row["title"],
                "source_path": row["path"],
                "source_bytes": len(source_data),
                "source_sha256": source_sha,
                "source_format": source_format,
                "text_path": text_relative.as_posix(),
                "text_bytes": len(text_bytes),
                "text_sha256": text_sha,
                "visible_characters": visible_characters,
                "extraction_tool": tool,
                "receipt_path": receipt_relative.as_posix(),
                "reading_credit_granted": False,
            }
        )
    return LiteratureTextExtractionResult(
        status="PASS",
        extracted_count=len(extracted),
        rows=tuple(extracted),
    )
