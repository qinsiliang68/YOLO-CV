import csv
import hashlib
import json
from pathlib import Path

import pytest

from stage1_dynamic_replay_v3.literature_broad_source_validation_v2 import (
    BroadSourceValidationError,
    validate_broad_source_batch,
)
from stage1_dynamic_replay_v3.literature_source_inventory_v2 import PdfProbeResult


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _source(
    root: Path,
    *,
    paper_id: str,
    suffix: str,
    data: bytes,
) -> dict[str, str]:
    relative = f"sources/{paper_id}{suffix}"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest().upper()
    receipt_relative = f"{relative}.receipt.json"
    row = {
        "paper_id": paper_id,
        "artifact_role": "BROAD_SOURCE",
        "path": relative,
        "url": f"https://primary.example/{paper_id}{suffix}",
        "retrieved_at": "2026-08-09T12:00:00+08:00",
        "http_status": "200",
        "content_type": "application/pdf" if suffix == ".pdf" else "text/html",
        "bytes": str(len(data)),
        "sha256": digest,
        "retrieval_method": "HTTP_DOWNLOAD",
        "source_authority": "PRIMARY_PUBLISHER",
        "final_url": f"https://primary.example/{paper_id}{suffix}",
        "receipt_path": receipt_relative,
        "reused_existing": "false",
    }
    receipt = {
        "paper_id": paper_id,
        "artifact_role": "BROAD_SOURCE",
        "bytes": len(data),
        "sha256": digest,
        "ledger_row": row,
    }
    (root / receipt_relative).write_text(json.dumps(receipt), encoding="utf-8")
    return row


def test_validates_pdf_and_html_but_reports_missing_batch_members(tmp_path: Path) -> None:
    queue = tmp_path / "queue.csv"
    acquisition = tmp_path / "acquisition.csv"
    failures = tmp_path / "failures.csv"
    _write_csv(
        queue,
        [
            {"queue_id": "RG1", "title": "Learning Useful Training Samples"},
            {"queue_id": "RG2", "title": "Coverage Aware Data Selection"},
            {"queue_id": "RG3", "title": "Unavailable Study"},
        ],
    )
    pdf_row = _source(tmp_path, paper_id="RG1", suffix=".pdf", data=b"%PDF-fixture")
    html_row = _source(
        tmp_path,
        paper_id="RG2",
        suffix=".html",
        data=(
            b"<html><head><title>Coverage Aware Data Selection</title></head>"
            b"<body>Primary article abstract and method.</body></html>"
        ),
    )
    _write_csv(acquisition, [pdf_row, html_row])
    _write_csv(
        failures,
        [
            {
                "paper_id": "RG3",
                "artifact_role": "BROAD_SOURCE",
                "url": "https://primary.example/RG3",
                "destination": "sources/RG3.html",
                "attempts": "2",
                "error": "HTTP 403",
            }
        ],
    )

    result = validate_broad_source_batch(
        corpus_root=tmp_path,
        queue_path=queue,
        acquisition_ledger=acquisition,
        failure_ledger=failures,
        pdf_probe=lambda _: PdfProbeResult(
            page_count=12,
            first_pages_text="Learning Useful Training Samples",
            probe_tool="FIXTURE",
        ),
    )

    assert result.status == "INCOMPLETE_ACQUISITION"
    assert result.expected_count == 3
    assert result.verified_count == 2
    assert result.failed_count == 1
    assert {row["source_format"] for row in result.rows} == {"PDF", "HTML"}


def test_rejects_wrong_title_even_when_hash_and_counts_match(tmp_path: Path) -> None:
    queue = tmp_path / "queue.csv"
    acquisition = tmp_path / "acquisition.csv"
    failures = tmp_path / "failures.csv"
    _write_csv(queue, [{"queue_id": "RG1", "title": "Expected Training Paper"}])
    row = _source(tmp_path, paper_id="RG1", suffix=".pdf", data=b"%PDF-fixture")
    _write_csv(acquisition, [row])
    failures.write_text(
        "paper_id,artifact_role,url,destination,attempts,error\n", encoding="utf-8"
    )

    with pytest.raises(BroadSourceValidationError, match="title identity"):
        validate_broad_source_batch(
            corpus_root=tmp_path,
            queue_path=queue,
            acquisition_ledger=acquisition,
            failure_ledger=failures,
            pdf_probe=lambda _: PdfProbeResult(
                page_count=5,
                first_pages_text="A Completely Different Article",
                probe_tool="FIXTURE",
            ),
        )


def test_later_success_supersedes_documented_earlier_failure(tmp_path: Path) -> None:
    queue = tmp_path / "queue.csv"
    first_acquisition = tmp_path / "acquisition_1.csv"
    first_failures = tmp_path / "failures_1.csv"
    correction_acquisition = tmp_path / "acquisition_2.csv"
    correction_failures = tmp_path / "failures_2.csv"
    _write_csv(queue, [{"queue_id": "RG1", "title": "Recovered Training Paper"}])
    first_acquisition.write_text(
        "paper_id,artifact_role,path,url,retrieved_at,http_status,content_type,bytes,sha256,"
        "retrieval_method,source_authority,final_url,receipt_path,reused_existing\n",
        encoding="utf-8",
    )
    _write_csv(
        first_failures,
        [
            {
                "paper_id": "RG1",
                "artifact_role": "BROAD_SOURCE",
                "url": "https://blocked.example/RG1",
                "destination": "sources/RG1.pdf",
                "attempts": "2",
                "error": "HTTP 403",
            }
        ],
    )
    recovered = _source(
        tmp_path,
        paper_id="RG1",
        suffix=".pdf",
        data=b"%PDF-recovered-fixture",
    )
    _write_csv(correction_acquisition, [recovered])
    correction_failures.write_text(
        "paper_id,artifact_role,url,destination,attempts,error\n", encoding="utf-8"
    )

    result = validate_broad_source_batch(
        corpus_root=tmp_path,
        queue_path=queue,
        acquisition_ledger=[first_acquisition, correction_acquisition],
        failure_ledger=[first_failures, correction_failures],
        pdf_probe=lambda _: PdfProbeResult(
            page_count=7,
            first_pages_text="Recovered Training Paper",
            probe_tool="FIXTURE",
        ),
    )

    assert result.status == "PASS"
    assert result.verified_count == 1
    assert result.failed_count == 0


def test_rejects_conflicting_successful_source_hashes(tmp_path: Path) -> None:
    queue = tmp_path / "queue.csv"
    acquisition_1 = tmp_path / "acquisition_1.csv"
    acquisition_2 = tmp_path / "acquisition_2.csv"
    failures = tmp_path / "failures.csv"
    _write_csv(queue, [{"queue_id": "RG1", "title": "Stable Source Identity"}])
    first = _source(tmp_path, paper_id="RG1", suffix=".pdf", data=b"%PDF-first")
    second = _source(tmp_path, paper_id="RG1", suffix=".html", data=b"Stable Source Identity")
    _write_csv(acquisition_1, [first])
    _write_csv(acquisition_2, [second])
    failures.write_text(
        "paper_id,artifact_role,url,destination,attempts,error\n", encoding="utf-8"
    )

    with pytest.raises(BroadSourceValidationError, match="conflicting acquisitions"):
        validate_broad_source_batch(
            corpus_root=tmp_path,
            queue_path=queue,
            acquisition_ledger=[acquisition_1, acquisition_2],
            failure_ledger=failures,
            pdf_probe=lambda _: PdfProbeResult(
                page_count=7,
                first_pages_text="Stable Source Identity",
                probe_tool="FIXTURE",
            ),
        )


def test_explicit_hash_bound_supersession_selects_corrected_source(tmp_path: Path) -> None:
    queue = tmp_path / "queue.csv"
    acquisition_1 = tmp_path / "acquisition_1.csv"
    acquisition_2 = tmp_path / "acquisition_2.csv"
    supersessions = tmp_path / "supersessions.csv"
    failures = tmp_path / "failures.csv"
    _write_csv(queue, [{"queue_id": "RG1", "title": "Corrected Source Identity"}])
    blocked = _source(
        tmp_path,
        paper_id="RG1",
        suffix=".html",
        data=b"<html><body>Access gateway without article identity</body></html>",
    )
    corrected = _source(
        tmp_path,
        paper_id="RG1",
        suffix=".pdf",
        data=b"%PDF-corrected",
    )
    _write_csv(acquisition_1, [blocked])
    _write_csv(acquisition_2, [corrected])
    _write_csv(
        supersessions,
        [
            {
                "paper_id": "RG1",
                "superseded_sha256": blocked["sha256"],
                "canonical_sha256": corrected["sha256"],
                "reason": "Initial HTTP 200 payload was an access page.",
            }
        ],
    )
    failures.write_text(
        "paper_id,artifact_role,url,destination,attempts,error\n", encoding="utf-8"
    )

    result = validate_broad_source_batch(
        corpus_root=tmp_path,
        queue_path=queue,
        acquisition_ledger=[acquisition_1, acquisition_2],
        failure_ledger=failures,
        supersession_ledger=supersessions,
        pdf_probe=lambda _: PdfProbeResult(
            page_count=4,
            first_pages_text="Corrected Source Identity",
            probe_tool="FIXTURE",
        ),
    )

    assert result.status == "PASS"
    assert result.rows[0]["sha256"] == corrected["sha256"]


def test_supersession_fails_closed_when_hash_binding_is_wrong(tmp_path: Path) -> None:
    queue = tmp_path / "queue.csv"
    acquisition_1 = tmp_path / "acquisition_1.csv"
    acquisition_2 = tmp_path / "acquisition_2.csv"
    supersessions = tmp_path / "supersessions.csv"
    failures = tmp_path / "failures.csv"
    _write_csv(queue, [{"queue_id": "RG1", "title": "Corrected Source Identity"}])
    first = _source(tmp_path, paper_id="RG1", suffix=".html", data=b"first source")
    second = _source(tmp_path, paper_id="RG1", suffix=".pdf", data=b"%PDF-second")
    _write_csv(acquisition_1, [first])
    _write_csv(acquisition_2, [second])
    _write_csv(
        supersessions,
        [
            {
                "paper_id": "RG1",
                "superseded_sha256": first["sha256"],
                "canonical_sha256": "0" * 64,
                "reason": "Claimed replacement does not match an acquired source.",
            }
        ],
    )
    failures.write_text(
        "paper_id,artifact_role,url,destination,attempts,error\n", encoding="utf-8"
    )

    with pytest.raises(BroadSourceValidationError, match="supersession hash binding"):
        validate_broad_source_batch(
            corpus_root=tmp_path,
            queue_path=queue,
            acquisition_ledger=[acquisition_1, acquisition_2],
            failure_ledger=failures,
            supersession_ledger=supersessions,
            pdf_probe=lambda _: PdfProbeResult(
                page_count=4,
                first_pages_text="Corrected Source Identity",
                probe_tool="FIXTURE",
            ),
        )
