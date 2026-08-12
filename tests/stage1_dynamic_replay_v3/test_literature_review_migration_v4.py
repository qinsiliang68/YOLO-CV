from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from stage1_dynamic_replay_v3.literature_review_migration_v4 import (
    ReviewMigrationError,
    migrate_screened_reviews,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "corpus"
    reviews = root / "reviews_v3"
    reviews.mkdir(parents=True)
    source = root / "sources" / "new.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"pdf bytes")
    text = root / "text" / "new.txt"
    text.parent.mkdir(parents=True)
    text.write_text("full paper text", encoding="utf-8")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest().upper()
    text_sha = hashlib.sha256(text.read_bytes()).hexdigest().upper()
    (reviews / "P0009.json").write_text(
        json.dumps(
            {
                "schema_version": "3.0",
                "paper_id": "P0009",
                "canonical_work_id": "CW1",
                "title": "Canonical study",
                "method_source": {
                    "path": "old/source.pdf",
                    "bytes": source.stat().st_size,
                    "sha256": source_sha,
                },
                "text_source": {
                    "path": "old/text.txt",
                    "bytes": text.stat().st_size,
                    "sha256": text_sha,
                },
                "decision": "SCREENED_ELIGIBLE",
            }
        ),
        encoding="utf-8",
    )
    queue = root / "queue.csv"
    _write_csv(
        queue,
        [
            {
                "paper_id": "P0011",
                "canonical_work_id": "CW1",
                "title": "Canonical study",
                "selection_role": "PRIMARY",
                "method_source_sha256": source_sha,
                "method_source_bytes": source.stat().st_size,
            }
        ],
    )
    extraction = root / "extraction.csv"
    _write_csv(
        extraction,
        [
            {
                "paper_id": "P0011",
                "source_path": "sources/new.pdf",
                "source_bytes": source.stat().st_size,
                "source_sha256": source_sha,
                "text_path": "text/new.txt",
                "text_bytes": text.stat().st_size,
                "text_sha256": text_sha,
            }
        ],
    )
    return root, reviews, queue, extraction


def test_migration_rebinds_paths_without_rewriting_review_content(tmp_path: Path) -> None:
    root, reviews, queue, extraction = _fixture(tmp_path)

    result = migrate_screened_reviews(
        root,
        source_review_dir=reviews,
        screening_queue=queue,
        extraction_ledger=extraction,
        output_relative=Path("reviews_v4"),
    )

    assert result.migrated_count == 1
    assert result.renamed_count == 1
    migrated = json.loads((result.output_dir / "P0011.json").read_text(encoding="utf-8"))
    assert migrated["paper_id"] == "P0011"
    assert migrated["decision"] == "SCREENED_ELIGIBLE"
    assert migrated["method_source"]["path"] == "sources/new.pdf"
    assert migrated["text_source"]["path"] == "text/new.txt"
    assert migrated["migration_provenance"]["source_paper_id"] == "P0009"
    assert migrated["migration_provenance"]["content_review_reused"] is True
    assert (
        migrated["migration_provenance"]["evidence_provenance_class"]
        == "USER_ACCEPTED_INHERITED_EVIDENCE"
    )
    assert (
        migrated["migration_provenance"]["independent_content_rereview_performed"]
        is False
    )

    receipt = json.loads(
        (result.output_dir / "MIGRATION_RECEIPT_v4.json").read_text(encoding="utf-8")
    )
    assert receipt["evidence_provenance_class"] == "USER_ACCEPTED_INHERITED_EVIDENCE"
    assert receipt["independent_content_rereview_performed"] is False


def test_migration_ignores_non_paper_json_receipts(tmp_path: Path) -> None:
    root, reviews, queue, extraction = _fixture(tmp_path)
    (reviews / "PROVENANCE.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )

    result = migrate_screened_reviews(
        root,
        source_review_dir=reviews,
        screening_queue=queue,
        extraction_ledger=extraction,
        output_relative=Path("reviews_v4"),
    )

    assert result.migrated_count == 1


def test_migration_rejects_changed_evidence_bytes(tmp_path: Path) -> None:
    root, reviews, queue, extraction = _fixture(tmp_path)
    record_path = reviews / "P0009.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["text_source"]["sha256"] = "0" * 64
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ReviewMigrationError, match="extracted text SHA-256 changed"):
        migrate_screened_reviews(
            root,
            source_review_dir=reviews,
            screening_queue=queue,
            extraction_ledger=extraction,
            output_relative=Path("reviews_v4"),
        )
