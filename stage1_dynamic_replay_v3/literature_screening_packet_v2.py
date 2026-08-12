"""Create page-located review packets from hash-bound extracted paper text."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence


class ScreeningPacketError(RuntimeError):
    """Raised when an extracted paper or its review packet loses identity."""


@dataclass(frozen=True)
class ScreeningTextScan:
    page_count: int
    headings: tuple[tuple[str, int, int, str], ...]
    evidence_candidates: tuple[tuple[str, int, int, str], ...]


_HEADING_PATTERNS = {
    "METHOD": re.compile(
        r"^(?:\d+(?:\.\d+)*\.?\s*)?(?:materials?\s+and\s+methods?|methods?|methodology|"
        r"proposed\s+(?:method|approach|framework)|approach|algorithm|framework)\b",
        re.IGNORECASE,
    ),
    "EXPERIMENT": re.compile(
        r"^(?:\d+(?:\.\d+)*\.?\s*)?(?:experiments?|experimental\s+(?:setup|results?|evaluation)|evaluation)\b",
        re.IGNORECASE,
    ),
    "ABLATION": re.compile(
        r"^(?:\d+(?:\.\d+)*\.?\s*)?(?:ablation|component\s+analysis|sensitivity\s+analysis)\b",
        re.IGNORECASE,
    ),
    "LIMITATION": re.compile(
        r"^(?:\d+(?:\.\d+)*\.?\s*)?(?:limitations?|discussion|threats?\s+to\s+validity)\b",
        re.IGNORECASE,
    ),
    "CONCLUSION": re.compile(
        r"^(?:\d+(?:\.\d+)*\.?\s*)?(?:conclusions?|concluding\s+remarks?|summary)\b",
        re.IGNORECASE,
    ),
}
_EVIDENCE_PATTERNS = {
    "BUDGET": re.compile(
        r"\b(?:budget|coreset|subset size|sampling ratio|selection ratio|replay slots?|"
        r"examples? selected|samples? selected|top[- ]?k)\b",
        re.IGNORECASE,
    ),
    "RANDOM_BASELINE": re.compile(
        r"\b(?:random (?:selection|sampling|subset|baseline)|uniform (?:selection|sampling)|randomly selected)\b",
        re.IGNORECASE,
    ),
    "SEED": re.compile(
        r"\b(?:seeds?|independent runs?|"
        r"repeated (?:runs?|trials?)|mean\s*[±+/-])\b",
        re.IGNORECASE,
    ),
    "CHECKPOINT": re.compile(
        r"\b(?:checkpoint|early stopping|best epoch|last epoch|model selection)\b",
        re.IGNORECASE,
    ),
    "NEGATIVE_RESULT": re.compile(
        r"\b(?:does not outperform|did not outperform|no (?:significant )?improvement|"
        r"fails? to|failed to|worse than|underperform|limitation)\b",
        re.IGNORECASE,
    ),
}


def scan_screening_text(text: str, *, per_kind_limit: int = 20) -> ScreeningTextScan:
    if per_kind_limit < 1:
        raise ScreeningPacketError("per_kind_limit must be positive")
    pages = text.replace("\r\n", "\n").replace("\r", "\n").split("\f")
    if not pages:
        raise ScreeningPacketError("extracted text has no pages")
    headings: list[tuple[str, int, int, str]] = []
    evidence: list[tuple[str, int, int, str]] = []
    evidence_counts = {kind: 0 for kind in _EVIDENCE_PATTERNS}
    for page_number, page in enumerate(pages, start=1):
        for line_number, raw_line in enumerate(page.splitlines(), start=1):
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            if len(line) <= 160:
                for kind, pattern in _HEADING_PATTERNS.items():
                    if pattern.search(line):
                        headings.append((kind, page_number, line_number, line))
                        break
            for kind, pattern in _EVIDENCE_PATTERNS.items():
                if evidence_counts[kind] >= per_kind_limit or not pattern.search(line):
                    continue
                evidence.append((kind, page_number, line_number, line[:360]))
                evidence_counts[kind] += 1
    return ScreeningTextScan(
        page_count=len(pages),
        headings=tuple(headings),
        evidence_candidates=tuple(evidence),
    )


def verify_extracted_text(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None,
) -> str:
    if not path.is_file():
        raise ScreeningPacketError(f"extracted text missing: {path}")
    data = path.read_bytes()
    if expected_bytes is not None and len(data) != expected_bytes:
        raise ScreeningPacketError(
            f"text bytes mismatch for {path.name}: expected {expected_bytes}, observed {len(data)}"
        )
    digest = hashlib.sha256(data).hexdigest().upper()
    if digest != expected_sha256.upper():
        raise ScreeningPacketError(
            f"text SHA mismatch for {path.name}: expected {expected_sha256}, observed {digest}"
        )
    return data.decode("utf-8", errors="replace")


def _read_csv(path: Path, required: Sequence[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ScreeningPacketError(f"required CSV missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(set(required) - fields)
        if missing:
            raise ScreeningPacketError(f"{path.name} missing fields: {missing}")
        return [dict(row) for row in reader]


def _safe_corpus_path(root: Path, value: str) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        raise ScreeningPacketError(f"corpus path must be relative: {value}")
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ScreeningPacketError(f"corpus path escapes root: {value}") from exc
    return resolved


def _safe_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _packet_text(
    queue: Mapping[str, str],
    extraction: Mapping[str, str],
    scan: ScreeningTextScan,
) -> str:
    lines = [
        f"# {queue['paper_id']} - {queue['title']}",
        "",
        "Status: AUTO_LOCATED_NOT_READ",
        "",
        f"- Selection role: `{queue['selection_role']}`",
        f"- Queue RQ: `{queue['quota_rq']}`",
        f"- Full text: `{extraction['text_path']}`",
        f"- Source SHA-256: `{extraction['source_sha256']}`",
        f"- Text SHA-256: `{extraction['text_sha256']}`",
        f"- Extracted pages: `{scan.page_count}`",
        "- Reading credit: `false`",
        "",
        "## Candidate Section Headings",
        "",
        "| Kind | Page | Line | Text |",
        "|---|---:|---:|---|",
    ]
    lines.extend(
        f"| {kind} | {page} | {line} | {_safe_cell(text)} |"
        for kind, page, line, text in scan.headings
    )
    if not scan.headings:
        lines.append("| NONE | 0 | 0 | No heading matched automatically; manual full-text navigation required. |")
    lines.extend(
        [
            "",
            "## Candidate Evidence Lines",
            "",
            "| Kind | Page | Line | Text |",
            "|---|---:|---:|---|",
        ]
    )
    lines.extend(
        f"| {kind} | {page} | {line} | {_safe_cell(text)} |"
        for kind, page, line, text in scan.evidence_candidates
    )
    if not scan.evidence_candidates:
        lines.append("| NONE | 0 | 0 | No keyword evidence matched; manual full-text reading required. |")
    lines.extend(
        [
            "",
            "These locators are mechanical navigation aids. They are not screened evidence and do not grant reading credit.",
            "",
        ]
    )
    return "\n".join(lines)


def build_screening_packets(
    *,
    corpus_root: str | Path,
    screening_queue: str | Path,
    extraction_ledger: str | Path,
    output_root: str | Path,
    replace_existing: bool = False,
) -> Path:
    root = Path(corpus_root).resolve()
    queue_path = Path(screening_queue).resolve()
    extraction_path = Path(extraction_ledger).resolve()
    output = Path(output_root).resolve()
    if output.exists() and not replace_existing:
        raise ScreeningPacketError(f"packet output already exists: {output}")
    queue_rows = _read_csv(
        queue_path,
        ("paper_id", "title", "selection_role", "quota_rq"),
    )
    extraction_rows = _read_csv(
        extraction_path,
        (
            "paper_id",
            "text_path",
            "text_bytes",
            "text_sha256",
            "source_sha256",
        ),
    )
    queue_by_id = {row["paper_id"]: row for row in queue_rows}
    extraction_by_id = {row["paper_id"]: row for row in extraction_rows}
    if len(queue_by_id) != len(queue_rows) or len(extraction_by_id) != len(extraction_rows):
        raise ScreeningPacketError("duplicate paper identity in packet inputs")
    if set(queue_by_id) != set(extraction_by_id):
        raise ScreeningPacketError("screening queue and extraction ledger paper IDs differ")

    temp = output.parent / f".{output.name}.tmp"
    if temp.exists():
        raise ScreeningPacketError(f"stale packet temp exists: {temp}")
    (temp / "packets").mkdir(parents=True)
    index_rows: list[dict[str, Any]] = []
    try:
        for paper_id in sorted(queue_by_id):
            queue = queue_by_id[paper_id]
            extraction = extraction_by_id[paper_id]
            text_path = _safe_corpus_path(root, extraction["text_path"])
            text = verify_extracted_text(
                text_path,
                expected_sha256=extraction["text_sha256"],
                expected_bytes=int(extraction["text_bytes"]),
            )
            scan = scan_screening_text(text)
            packet_rel = Path("packets") / f"{paper_id}.md"
            packet = temp / packet_rel
            packet.write_text(_packet_text(queue, extraction, scan), encoding="utf-8")
            kind_counts = {
                kind: sum(item[0] == kind for item in scan.headings)
                for kind in _HEADING_PATTERNS
            }
            index_rows.append(
                {
                    "paper_id": paper_id,
                    "selection_role": queue["selection_role"],
                    "title": queue["title"],
                    "page_count": scan.page_count,
                    "heading_count": len(scan.headings),
                    "evidence_candidate_count": len(scan.evidence_candidates),
                    **{f"{kind.casefold()}_heading_count": count for kind, count in kind_counts.items()},
                    "packet_path": packet_rel.as_posix(),
                    "packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest().upper(),
                    "reading_credit_granted": False,
                }
            )
        fields = list(index_rows[0])
        with (temp / "SCREENING_PACKET_INDEX.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(index_rows)
        receipt = {
            "schema_version": "2.0",
            "status": "PASS",
            "created_at": datetime.now().astimezone().isoformat(),
            "packet_count": len(index_rows),
            "primary_count": sum(row["selection_role"] == "PRIMARY" for row in index_rows),
            "reserve_count": sum(row["selection_role"] == "RESERVE" for row in index_rows),
            "screening_queue_sha256": hashlib.sha256(queue_path.read_bytes()).hexdigest().upper(),
            "extraction_ledger_sha256": hashlib.sha256(extraction_path.read_bytes()).hexdigest().upper(),
            "formal_screened_increment": 0,
            "reading_credit_granted": False,
        }
        (temp / "SCREENING_PACKET_RECEIPT.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            backup = output.parent / f".{output.name}.previous"
            if backup.exists():
                raise ScreeningPacketError(f"stale packet backup exists: {backup}")
            output.rename(backup)
            try:
                temp.rename(output)
            except Exception:
                backup.rename(output)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(temp, output)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise
    return output
