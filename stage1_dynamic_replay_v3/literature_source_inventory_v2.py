"""Fail-closed inventory validation for acquired primary literature PDFs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Callable, Iterable


class SourceInventoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class PdfProbeResult:
    page_count: int
    first_pages_text: str
    probe_tool: str


@dataclass(frozen=True)
class SourceInventoryResult:
    status: str
    expected_count: int
    verified_count: int
    rows: tuple[dict[str, object], ...]


PdfProbe = Callable[[Path], PdfProbeResult]

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
    "toward",
    "towards",
    "via",
    "with",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _safe_path(root: Path, value: str) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        raise SourceInventoryError(f"path must be relative: {value}")
    resolved_root = root.resolve()
    resolved = (resolved_root / raw).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise SourceInventoryError(f"path escapes corpus root: {value}") from exc
    return resolved


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _ledger_identity(value: str) -> str:
    clean = value.strip()
    if not clean:
        raise SourceInventoryError("blank paper_id")
    if clean.startswith("LEGACY-"):
        return clean
    if re.fullmatch(r"(?:RG|AX)[A-F0-9]{16}", clean):
        return clean
    return f"LEGACY-{clean}"


def _title_tokens(value: str) -> set[str]:
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) >= 2 and token not in _TITLE_STOPWORDS
    }
    if len(tokens) < 2:
        raise SourceInventoryError(f"title has too few identity tokens: {value!r}")
    return tokens


def _title_coverage(title: str, first_pages_text: str) -> float:
    expected = _title_tokens(title)
    observed = set(re.findall(r"[a-z0-9]+", first_pages_text.casefold()))
    return len(expected & observed) / len(expected)


def _resolve_pdfinfo() -> str:
    discovered = shutil.which("pdfinfo")
    if not discovered:
        raise SourceInventoryError("pdfinfo is required for source inventory validation")
    path = Path(discovered)
    if path.suffix.casefold() == ".cmd" and len(path.parents) >= 3:
        bundled_exe = path.parents[2] / "native" / "poppler" / "Library" / "bin" / "pdfinfo.exe"
        if bundled_exe.is_file():
            return str(bundled_exe)
    return discovered


def probe_pdf_with_poppler(path: Path) -> PdfProbeResult:
    pdfinfo = _resolve_pdfinfo()
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        raise SourceInventoryError("pdftotext is required for source inventory validation")

    info = subprocess.run(
        [pdfinfo, str(path)],
        check=False,
        capture_output=True,
        timeout=60,
    )
    if info.returncode != 0:
        detail = info.stderr.decode("utf-8", errors="replace").strip()
        raise SourceInventoryError(f"pdfinfo failed for {path.name}: {detail}")
    info_text = info.stdout.decode("utf-8", errors="replace")
    page_match = re.search(r"^Pages:\s+(\d+)\s*$", info_text, flags=re.MULTILINE)
    if not page_match or int(page_match.group(1)) < 1:
        raise SourceInventoryError(f"pdfinfo reported no valid page count for {path.name}")

    text = subprocess.run(
        [pdftotext, "-f", "1", "-l", "2", "-layout", str(path), "-"],
        check=False,
        capture_output=True,
        timeout=60,
    )
    if text.returncode != 0:
        detail = text.stderr.decode("utf-8", errors="replace").strip()
        raise SourceInventoryError(f"pdftotext failed for {path.name}: {detail}")
    first_pages_text = text.stdout.decode("utf-8", errors="replace").strip()
    if not first_pages_text:
        raise SourceInventoryError(f"first two PDF pages contain no extractable text: {path.name}")
    return PdfProbeResult(
        page_count=int(page_match.group(1)),
        first_pages_text=first_pages_text,
        probe_tool="POPPLER_PDFINFO_AND_PDFTOTEXT",
    )


def _expected_titles(path: Path) -> dict[str, str]:
    rows = _read_csv(path)
    required = {"paper_id", "title"}
    if not rows:
        raise SourceInventoryError("expected literature ledger is empty")
    if not required.issubset(rows[0]):
        raise SourceInventoryError(f"expected ledger lacks columns: {sorted(required - set(rows[0]))}")
    result: dict[str, str] = {}
    for row in rows:
        paper_id = _ledger_identity(row["paper_id"])
        if paper_id in result:
            raise SourceInventoryError(f"duplicate expected paper_id: {paper_id}")
        title = row["title"].strip()
        _title_tokens(title)
        result[paper_id] = title
    return result


def _acquisitions(paths: Iterable[Path]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    required = {
        "paper_id",
        "artifact_role",
        "path",
        "bytes",
        "sha256",
        "receipt_path",
        "source_authority",
        "url",
    }
    for path in paths:
        rows = _read_csv(path)
        if rows and not required.issubset(rows[0]):
            raise SourceInventoryError(f"{path.name} lacks columns: {sorted(required - set(rows[0]))}")
        for row in rows:
            paper_id = _ledger_identity(row["paper_id"])
            if paper_id in result:
                raise SourceInventoryError(f"duplicate acquisition paper_id across ledgers: {paper_id}")
            result[paper_id] = row
    return result


def validate_source_inventory(
    *,
    corpus_root: Path,
    acquisition_ledgers: Iterable[Path],
    expected_ledger: Path,
    expected_count: int,
    pdf_probe: PdfProbe = probe_pdf_with_poppler,
    minimum_title_coverage: float = 0.60,
) -> SourceInventoryResult:
    if expected_count < 1:
        raise SourceInventoryError("expected_count must be positive")
    if not 0.0 < minimum_title_coverage <= 1.0:
        raise SourceInventoryError("minimum_title_coverage must be in (0, 1]")

    expected = _expected_titles(expected_ledger)
    acquired = _acquisitions(acquisition_ledgers)
    if len(expected) != expected_count:
        raise SourceInventoryError(
            f"expected ledger count mismatch: expected={expected_count} observed={len(expected)}"
        )
    if set(acquired) != set(expected):
        missing = sorted(set(expected) - set(acquired))
        extra = sorted(set(acquired) - set(expected))
        raise SourceInventoryError(f"acquisition identity set mismatch: missing={missing} extra={extra}")

    verified: list[dict[str, object]] = []
    for paper_id in sorted(expected):
        row = acquired[paper_id]
        source_path = _safe_path(corpus_root, row["path"])
        receipt_path = _safe_path(corpus_root, row["receipt_path"])
        if not source_path.is_file():
            raise SourceInventoryError(f"source file missing for {paper_id}: {row['path']}")
        if not receipt_path.is_file():
            raise SourceInventoryError(f"receipt missing for {paper_id}: {row['receipt_path']}")
        with source_path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise SourceInventoryError(f"source is not a PDF for {paper_id}: {row['path']}")

        observed_bytes = source_path.stat().st_size
        observed_sha = _sha256(source_path)
        if observed_bytes != int(row["bytes"]):
            raise SourceInventoryError(f"source byte count mismatch for {paper_id}")
        if observed_sha != row["sha256"].strip().upper():
            raise SourceInventoryError(f"source hash mismatch for {paper_id}")

        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_row = receipt.get("ledger_row", {})
        identity_checks = {
            "paper_id": receipt.get("paper_id") == paper_id,
            "artifact_role": receipt.get("artifact_role") == row["artifact_role"],
            "bytes": int(receipt.get("bytes", -1)) == observed_bytes,
            "sha256": str(receipt.get("sha256", "")).upper() == observed_sha,
            "ledger_paper_id": receipt_row.get("paper_id") == paper_id,
            "ledger_path": receipt_row.get("path") == row["path"],
            "ledger_bytes": int(receipt_row.get("bytes", -1)) == observed_bytes,
            "ledger_sha256": str(receipt_row.get("sha256", "")).upper() == observed_sha,
        }
        failed_checks = [name for name, ok in identity_checks.items() if not ok]
        if failed_checks:
            raise SourceInventoryError(f"receipt identity mismatch for {paper_id}: {failed_checks}")

        probe = pdf_probe(source_path)
        coverage = _title_coverage(expected[paper_id], probe.first_pages_text)
        if coverage < minimum_title_coverage:
            raise SourceInventoryError(
                f"PDF title identity mismatch for {paper_id}: coverage={coverage:.3f}"
            )
        verified.append(
            {
                "paper_id": paper_id,
                "title": expected[paper_id],
                "path": row["path"],
                "bytes": observed_bytes,
                "sha256": observed_sha,
                "source_format": "PDF",
                "page_count": probe.page_count,
                "title_token_coverage": f"{coverage:.6f}",
                "title_identity_status": "PASS",
                "source_authority": row["source_authority"],
                "source_url": row["url"],
                "receipt_path": row["receipt_path"],
                "probe_tool": probe.probe_tool,
                "first_pages_text_sha256": hashlib.sha256(
                    probe.first_pages_text.encode("utf-8")
                ).hexdigest().upper(),
            }
        )

    return SourceInventoryResult(
        status="PASS",
        expected_count=expected_count,
        verified_count=len(verified),
        rows=tuple(verified),
    )
