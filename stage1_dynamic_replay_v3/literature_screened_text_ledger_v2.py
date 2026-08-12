"""Build the hash-bound PDF ledger used for SCREENED text extraction."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


class ScreenedTextLedgerError(RuntimeError):
    """Raised when a SCREENED text source is not a verified PDF identity."""


FIELDS = (
    "paper_id",
    "title",
    "path",
    "bytes",
    "sha256",
    "source_format",
    "selection_role",
    "reading_rank",
    "reading_credit_granted",
)


def build_screened_text_ledger_rows(
    screening_rows: Sequence[Mapping[str, Any]],
    *,
    broad_staging_relative: str,
) -> list[dict[str, str]]:
    base = PurePosixPath(broad_staging_relative)
    if base.is_absolute() or ".." in base.parts:
        raise ScreenedTextLedgerError(
            "broad_staging_relative must be a safe corpus-relative path"
        )
    seen: set[str] = set()
    output: list[dict[str, str]] = []
    for row in screening_rows:
        paper_id = str(row.get("paper_id", "")).strip()
        if not paper_id or paper_id in seen:
            raise ScreenedTextLedgerError(f"blank or duplicate paper_id: {paper_id!r}")
        seen.add(paper_id)
        source_format = str(
            row.get("method_source_format", row.get("broad_source_format", ""))
        ).strip().upper()
        source_path = PurePosixPath(
            str(row.get("method_source_path", row.get("broad_source_path", ""))).strip()
        )
        source_origin = str(row.get("method_source_origin", "BROAD_SOURCE")).strip()
        if source_format != "PDF" or source_path.suffix.casefold() != ".pdf":
            raise ScreenedTextLedgerError(f"{paper_id} is not backed by a verified PDF")
        digest = str(
            row.get("method_source_sha256", row.get("broad_source_sha256", ""))
        ).strip().upper()
        if len(digest) != 64 or any(ch not in "0123456789ABCDEF" for ch in digest):
            raise ScreenedTextLedgerError(f"{paper_id} has an invalid source SHA-256")
        try:
            size = int(
                str(
                    row.get("method_source_bytes", row.get("broad_source_bytes", ""))
                ).strip()
            )
        except ValueError as exc:
            raise ScreenedTextLedgerError(f"{paper_id} has invalid source bytes") from exc
        if size < 1:
            raise ScreenedTextLedgerError(f"{paper_id} has non-positive source bytes")
        output.append(
            {
                "paper_id": paper_id,
                "title": str(row.get("title", "")).strip(),
                "path": str(
                    source_path
                    if source_origin == "VERIFIED_OVERRIDE"
                    else base / source_path
                ),
                "bytes": str(size),
                "sha256": digest,
                "source_format": "PDF",
                "selection_role": str(row.get("selection_role", "")).strip(),
                "reading_rank": str(row.get("reading_rank", "")).strip(),
                "reading_credit_granted": "False",
            }
        )
    return output


def write_screened_text_ledger(
    screening_queue: str | Path,
    *,
    broad_staging_relative: str,
    output_ledger: str | Path,
    output_receipt: str | Path,
) -> int:
    queue_path = Path(screening_queue).resolve()
    if not queue_path.is_file():
        raise ScreenedTextLedgerError(f"screening queue missing: {queue_path}")
    with queue_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "paper_id",
        "title",
        "broad_source_path",
        "broad_source_format",
        "broad_source_sha256",
        "broad_source_bytes",
        "selection_role",
        "reading_rank",
    }
    if not rows or not required.issubset(rows[0]):
        raise ScreenedTextLedgerError(
            f"screening queue is empty or missing fields: {sorted(required - set(rows[0] if rows else {}))}"
        )
    output = build_screened_text_ledger_rows(
        rows,
        broad_staging_relative=broad_staging_relative,
    )
    ledger_path = Path(output_ledger).resolve()
    receipt_path = Path(output_receipt).resolve()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(output)
    os.replace(temporary, ledger_path)
    receipt = {
        "schema_version": "2.0",
        "status": "PASS",
        "source_count": len(output),
        "screening_queue_sha256": hashlib.sha256(queue_path.read_bytes()).hexdigest().upper(),
        "text_source_ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest().upper(),
        "reading_credit_granted": False,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_temp = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    receipt_temp.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(receipt_temp, receipt_path)
    return len(output)
