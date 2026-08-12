"""Migrate trusted SCREENED reviews across a canonical-ID-preserving re-freeze."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping


class ReviewMigrationError(RuntimeError):
    """Raised when a prior review cannot be rebound without changing evidence."""


@dataclass(frozen=True)
class ReviewMigrationResult:
    output_dir: Path
    migrated_count: int
    renamed_count: int
    mapping_sha256: str


MAPPING_FIELDS = (
    "canonical_work_id",
    "old_paper_id",
    "new_paper_id",
    "title",
    "method_source_sha256",
    "text_source_sha256",
    "content_review_reused",
    "evidence_provenance_class",
)

EVIDENCE_PROVENANCE_CLASS = "USER_ACCEPTED_INHERITED_EVIDENCE"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _read_csv(path: Path, key: str) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise ReviewMigrationError(f"required CSV missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows or key not in rows[0]:
        raise ReviewMigrationError(f"{path.name} is empty or lacks {key}")
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row[key].strip()
        if not value or value in indexed:
            raise ReviewMigrationError(f"blank or duplicate {key}: {value!r}")
        indexed[value] = row
    return indexed


def _artifact_matches(
    value: Mapping[str, Any], *, sha256: str, size: str, label: str
) -> None:
    if str(value.get("sha256", "")).upper() != sha256.upper():
        raise ReviewMigrationError(f"{label} SHA-256 changed")
    try:
        actual_size = int(value.get("bytes", -1))
        expected_size = int(size)
    except (TypeError, ValueError) as exc:
        raise ReviewMigrationError(f"{label} bytes are invalid") from exc
    if actual_size != expected_size:
        raise ReviewMigrationError(f"{label} bytes changed")


def migrate_screened_reviews(
    corpus_root: str | Path,
    *,
    source_review_dir: str | Path,
    screening_queue: str | Path,
    extraction_ledger: str | Path,
    output_relative: str | Path,
    replace_existing: bool = False,
) -> ReviewMigrationResult:
    """Rebind reviews by canonical_work_id while preserving reviewed content."""

    root = Path(corpus_root).resolve()
    source_dir = Path(source_review_dir).resolve()
    queue = _read_csv(Path(screening_queue).resolve(), "canonical_work_id")
    extraction = _read_csv(Path(extraction_ledger).resolve(), "paper_id")
    output_fragment = Path(output_relative)
    if output_fragment.is_absolute() or ".." in output_fragment.parts:
        raise ReviewMigrationError("output_relative must stay inside the corpus root")
    output = (root / output_fragment).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ReviewMigrationError("output path escapes corpus root") from exc
    if output.exists() and not replace_existing:
        raise ReviewMigrationError(f"migration output already exists: {output}")

    review_paths = sorted(source_dir.glob("P[0-9][0-9][0-9][0-9].json"))
    if not review_paths:
        raise ReviewMigrationError("source review directory contains no Pxxxx.json")
    temp = output.parent / f".{output.name}.tmp"
    if temp.exists():
        raise ReviewMigrationError(f"stale migration temp exists: {temp}")
    temp.mkdir(parents=True)
    mappings: list[dict[str, str]] = []
    seen_new_ids: set[str] = set()
    renamed_count = 0
    try:
        for review_path in review_paths:
            try:
                record = json.loads(review_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ReviewMigrationError(f"invalid review JSON: {review_path.name}") from exc
            canonical_work_id = str(record.get("canonical_work_id", "")).strip()
            old_paper_id = str(record.get("paper_id", "")).strip()
            if canonical_work_id not in queue:
                raise ReviewMigrationError(
                    f"{review_path.name} canonical work is absent from repaired queue"
                )
            queue_row = queue[canonical_work_id]
            new_paper_id = queue_row["paper_id"].strip()
            if queue_row.get("selection_role", "").strip() != "PRIMARY":
                raise ReviewMigrationError(
                    f"{canonical_work_id} is no longer a PRIMARY SCREENED work"
                )
            if new_paper_id in seen_new_ids:
                raise ReviewMigrationError(f"duplicate migrated paper ID: {new_paper_id}")
            seen_new_ids.add(new_paper_id)
            if new_paper_id not in extraction:
                raise ReviewMigrationError(f"missing repaired extraction for {new_paper_id}")
            extraction_row = extraction[new_paper_id]
            if str(record.get("title", "")).strip() != queue_row["title"].strip():
                raise ReviewMigrationError(f"{old_paper_id} title changed")
            _artifact_matches(
                record.get("method_source", {}),
                sha256=queue_row["method_source_sha256"],
                size=queue_row["method_source_bytes"],
                label=f"{old_paper_id} method source",
            )
            _artifact_matches(
                record.get("text_source", {}),
                sha256=extraction_row["text_sha256"],
                size=extraction_row["text_bytes"],
                label=f"{old_paper_id} extracted text",
            )

            migrated = dict(record)
            migrated["paper_id"] = new_paper_id
            migrated["method_source"] = {
                "path": extraction_row["source_path"],
                "bytes": int(extraction_row["source_bytes"]),
                "sha256": extraction_row["source_sha256"].upper(),
            }
            migrated["text_source"] = {
                "path": extraction_row["text_path"],
                "bytes": int(extraction_row["text_bytes"]),
                "sha256": extraction_row["text_sha256"].upper(),
            }
            migrated["migration_provenance"] = {
                "source_review": review_path.relative_to(root).as_posix(),
                "source_paper_id": old_paper_id,
                "migration_basis": "CANONICAL_WORK_ID_AND_BYTE_IDENTICAL_EVIDENCE",
                "content_review_reused": True,
                "evidence_provenance_class": EVIDENCE_PROVENANCE_CLASS,
                "independent_content_rereview_performed": False,
            }
            (temp / f"{new_paper_id}.json").write_text(
                json.dumps(migrated, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            renamed_count += int(old_paper_id != new_paper_id)
            mappings.append(
                {
                    "canonical_work_id": canonical_work_id,
                    "old_paper_id": old_paper_id,
                    "new_paper_id": new_paper_id,
                    "title": queue_row["title"],
                    "method_source_sha256": extraction_row["source_sha256"].upper(),
                    "text_source_sha256": extraction_row["text_sha256"].upper(),
                    "content_review_reused": "True",
                    "evidence_provenance_class": EVIDENCE_PROVENANCE_CLASS,
                }
            )

        mapping_path = temp / "MIGRATION_MAPPING_v4.csv"
        with mapping_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(MAPPING_FIELDS))
            writer.writeheader()
            writer.writerows(mappings)
        receipt = {
            "schema_version": "4.0",
            "status": "PASS",
            "created_at": datetime.now().astimezone().isoformat(),
            "migrated_count": len(mappings),
            "renamed_count": renamed_count,
            "content_review_reused": True,
            "evidence_provenance_class": EVIDENCE_PROVENANCE_CLASS,
            "independent_content_rereview_performed": False,
            "mapping_sha256": _sha256(mapping_path),
        }
        (temp / "MIGRATION_RECEIPT_v4.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            backup = output.parent / f".{output.name}.previous"
            if backup.exists():
                raise ReviewMigrationError(f"stale migration backup exists: {backup}")
            output.rename(backup)
            try:
                temp.rename(output)
            except Exception:
                backup.rename(output)
                raise
            shutil.rmtree(backup)
        else:
            temp.rename(output)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise

    return ReviewMigrationResult(
        output_dir=output,
        migrated_count=len(mappings),
        renamed_count=renamed_count,
        mapping_sha256=_sha256(output / "MIGRATION_MAPPING_v4.csv"),
    )
