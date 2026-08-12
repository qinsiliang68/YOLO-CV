import csv
import hashlib
import json
from pathlib import Path

import pytest

from stage1_dynamic_replay_v3.literature_source_inventory_v2 import (
    PdfProbeResult,
    SourceInventoryError,
    validate_source_inventory,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "corpus"
    pdf_path = root / "sources" / "P0001.pdf"
    pdf_path.parent.mkdir(parents=True)
    payload = b"%PDF-1.7\nfixture\n%%EOF\n"
    pdf_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest().upper()
    receipt_path = pdf_path.with_suffix(".pdf.receipt.json")
    receipt_path.write_text(
        json.dumps(
            {
                "paper_id": "LEGACY-P0001",
                "artifact_role": "DEEP_FULL_TEXT",
                "bytes": len(payload),
                "sha256": digest,
                "ledger_row": {
                    "paper_id": "LEGACY-P0001",
                    "path": "sources/P0001.pdf",
                    "bytes": len(payload),
                    "sha256": digest,
                },
            }
        ),
        encoding="utf-8",
    )
    acquisition = tmp_path / "acquisition.csv"
    _write_csv(
        acquisition,
        [
            {
                "paper_id": "LEGACY-P0001",
                "artifact_role": "DEEP_FULL_TEXT",
                "path": "sources/P0001.pdf",
                "url": "https://publisher.example/P0001.pdf",
                "retrieved_at": "2026-08-09T00:00:00+08:00",
                "http_status": 200,
                "content_type": "application/pdf",
                "bytes": len(payload),
                "sha256": digest,
                "retrieval_method": "HTTP_DOWNLOAD",
                "source_authority": "PRIMARY_PUBLISHER",
                "final_url": "https://publisher.example/P0001.pdf",
                "receipt_path": "sources/P0001.pdf.receipt.json",
                "reused_existing": False,
            }
        ],
    )
    expected = tmp_path / "expected.csv"
    _write_csv(expected, [{"paper_id": "P0001", "title": "Useful Training Samples"}])
    return root, acquisition, expected


def _probe(_: Path) -> PdfProbeResult:
    return PdfProbeResult(
        page_count=12,
        first_pages_text="Useful Training Samples Authors Abstract",
        probe_tool="TEST_PROBE",
    )


def test_inventory_verifies_exact_set_receipt_hash_pages_and_title(tmp_path: Path) -> None:
    root, acquisition, expected = _fixture(tmp_path)

    result = validate_source_inventory(
        corpus_root=root,
        acquisition_ledgers=[acquisition],
        expected_ledger=expected,
        expected_count=1,
        pdf_probe=_probe,
    )

    assert result.status == "PASS"
    assert result.expected_count == result.verified_count == 1
    assert result.rows[0]["paper_id"] == "LEGACY-P0001"
    assert result.rows[0]["source_format"] == "PDF"
    assert result.rows[0]["page_count"] == 12
    assert result.rows[0]["title_identity_status"] == "PASS"


@pytest.mark.parametrize("mutation", ["hash", "receipt", "expected_set", "title"])
def test_inventory_fails_closed_on_identity_breaks(tmp_path: Path, mutation: str) -> None:
    root, acquisition, expected = _fixture(tmp_path)
    probe = _probe
    if mutation == "hash":
        (root / "sources" / "P0001.pdf").write_bytes(b"%PDF-1.7\ntampered\n")
    elif mutation == "receipt":
        receipt = root / "sources" / "P0001.pdf.receipt.json"
        data = json.loads(receipt.read_text(encoding="utf-8"))
        data["paper_id"] = "LEGACY-P9999"
        receipt.write_text(json.dumps(data), encoding="utf-8")
    elif mutation == "expected_set":
        _write_csv(expected, [{"paper_id": "P9999", "title": "Useful Training Samples"}])
    else:
        probe = lambda _: PdfProbeResult(12, "Completely unrelated document", "TEST_PROBE")

    with pytest.raises(SourceInventoryError):
        validate_source_inventory(
            corpus_root=root,
            acquisition_ledgers=[acquisition],
            expected_ledger=expected,
            expected_count=1,
            pdf_probe=probe,
        )


def test_inventory_rejects_duplicate_paper_across_ledgers(tmp_path: Path) -> None:
    root, acquisition, expected = _fixture(tmp_path)

    with pytest.raises(SourceInventoryError, match="duplicate acquisition"):
        validate_source_inventory(
            corpus_root=root,
            acquisition_ledgers=[acquisition, acquisition],
            expected_ledger=expected,
            expected_count=1,
            pdf_probe=_probe,
        )


def test_inventory_preserves_native_queue_identity_in_receipts(tmp_path: Path) -> None:
    root, acquisition, expected = _fixture(tmp_path)
    native_id = "AX501F9ECF2E222F25"
    acquisition_rows = list(csv.DictReader(acquisition.open(encoding="utf-8")))
    acquisition_rows[0]["paper_id"] = native_id
    _write_csv(acquisition, acquisition_rows)
    _write_csv(expected, [{"paper_id": native_id, "title": "Useful Training Samples"}])
    receipt_path = root / "sources" / "P0001.pdf.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["paper_id"] = native_id
    receipt["ledger_row"]["paper_id"] = native_id
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    result = validate_source_inventory(
        corpus_root=root,
        acquisition_ledgers=[acquisition],
        expected_ledger=expected,
        expected_count=1,
        pdf_probe=_probe,
    )

    assert result.rows[0]["paper_id"] == native_id
