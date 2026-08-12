import csv
import hashlib
import json
from pathlib import Path

import pytest

from stage1_dynamic_replay_v3.literature_text_extraction_v2 import (
    LiteratureTextExtractionError,
    extract_literature_text_batch,
)


def _write_validation(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validation_row(root: Path, paper_id: str, suffix: str, data: bytes) -> dict[str, str]:
    relative = f"sources/{paper_id}{suffix}"
    source = root / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(data)
    return {
        "paper_id": paper_id,
        "title": f"Title for {paper_id}",
        "path": relative,
        "bytes": str(len(data)),
        "sha256": hashlib.sha256(data).hexdigest().upper(),
        "source_format": "PDF" if suffix == ".pdf" else "HTML",
        "page_count": "4" if suffix == ".pdf" else "NOT_APPLICABLE_WITH_REASON:HTML source",
        "title_token_coverage": "1.0",
        "source_authority": "PRIMARY_PUBLISHER",
        "source_url": f"https://example.test/{paper_id}",
        "receipt_path": f"{relative}.receipt.json",
        "probe_tool": "FIXTURE",
        "reading_credit_granted": "False",
    }


def test_extracts_pdf_and_html_with_hash_bound_receipts(tmp_path: Path) -> None:
    validation = tmp_path / "validation.csv"
    pdf = _validation_row(tmp_path, "P1", ".pdf", b"%PDF-fixture")
    html = _validation_row(
        tmp_path,
        "P2",
        ".html",
        b"<html><body><h1>Title</h1><p>Full article method and results.</p></body></html>",
    )
    _write_validation(validation, [pdf, html])

    result = extract_literature_text_batch(
        corpus_root=tmp_path,
        validation_path=validation,
        output_dir=Path("extracted/batch_001"),
        pdf_extractor=lambda _: ("PDF full text\fsecond page", "FIXTURE_PDFTEXT"),
        minimum_characters=10,
    )

    assert result.status == "PASS"
    assert result.extracted_count == 2
    assert {row["paper_id"] for row in result.rows} == {"P1", "P2"}
    for row in result.rows:
        output = tmp_path / str(row["text_path"])
        receipt = tmp_path / str(row["receipt_path"])
        assert output.is_file()
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        assert payload["source_sha256"] == row["source_sha256"]
        assert payload["text_sha256"] == row["text_sha256"]
        assert payload["reading_credit_granted"] is False


def test_rejects_source_changed_after_validation(tmp_path: Path) -> None:
    validation = tmp_path / "validation.csv"
    row = _validation_row(tmp_path, "P1", ".pdf", b"%PDF-original")
    _write_validation(validation, [row])
    (tmp_path / row["path"]).write_bytes(b"%PDF-replaced")

    with pytest.raises(LiteratureTextExtractionError, match="source identity mismatch"):
        extract_literature_text_batch(
            corpus_root=tmp_path,
            validation_path=validation,
            output_dir=Path("extracted/batch_001"),
            pdf_extractor=lambda _: ("valid extracted text", "FIXTURE_PDFTEXT"),
            minimum_characters=10,
        )


def test_rejects_error_page_html(tmp_path: Path) -> None:
    validation = tmp_path / "validation.csv"
    row = _validation_row(
        tmp_path,
        "P1",
        ".html",
        b"<html><body>Access Denied. CAPTCHA required.</body></html>",
    )
    _write_validation(validation, [row])

    with pytest.raises(LiteratureTextExtractionError, match="access/error page"):
        extract_literature_text_batch(
            corpus_root=tmp_path,
            validation_path=validation,
            output_dir=Path("extracted/batch_001"),
            minimum_characters=10,
        )
