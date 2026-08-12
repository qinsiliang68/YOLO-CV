import hashlib
import json
from pathlib import Path

import pytest

from stage1_dynamic_replay_v3.literature_source_acquisition_v2 import (
    SourceAcquisitionError,
    SourceRequest,
    acquire_source,
)


class _Response:
    def __init__(self, content: bytes, *, content_type: str, url: str, status: int = 200) -> None:
        self.content = content
        self.headers = {"content-type": content_type}
        self.url = url
        self.status_code = status
        self.text = content.decode("utf-8", errors="replace")


class _Session:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        return self.response


def test_acquire_pdf_is_atomic_hash_bound_and_idempotent(tmp_path: Path) -> None:
    pdf = b"%PDF-1.7\nprimary paper bytes\n%%EOF\n"
    url = "https://proceedings.example/paper.pdf"
    session = _Session(_Response(pdf, content_type="application/pdf", url=url))
    request = SourceRequest(
        paper_id="LEGACY-P009",
        artifact_role="DEEP_FULL_TEXT",
        url=url,
        destination="sources/legacy_core/P009.pdf",
        source_authority="PRIMARY_PUBLISHER",
    )

    first = acquire_source(request, corpus_root=tmp_path, session=session)
    second = acquire_source(request, corpus_root=tmp_path, session=session)

    destination = tmp_path / request.destination
    receipt = destination.with_suffix(".pdf.receipt.json")
    assert destination.read_bytes() == pdf
    assert receipt.is_file()
    assert first == second
    assert session.calls == 1
    assert first["sha256"] == hashlib.sha256(pdf).hexdigest().upper()
    assert first["bytes"] == len(pdf)
    assert first["reused_existing"] is False
    assert json.loads(receipt.read_text(encoding="utf-8"))["paper_id"] == "LEGACY-P009"
    assert not destination.with_suffix(".pdf.tmp").exists()


def test_acquire_pdf_rejects_html_error_page_without_publishing(tmp_path: Path) -> None:
    url = "https://publisher.example/paper.pdf"
    session = _Session(_Response(b"<html>blocked</html>", content_type="text/html", url=url))
    request = SourceRequest(
        paper_id="P0001",
        artifact_role="DEEP_FULL_TEXT",
        url=url,
        destination="sources/P0001.pdf",
        source_authority="PRIMARY_PUBLISHER",
    )

    with pytest.raises(SourceAcquisitionError, match="PDF signature"):
        acquire_source(request, corpus_root=tmp_path, session=session)

    assert not (tmp_path / request.destination).exists()


def test_acquire_source_rejects_path_escape_secondary_source_and_hash_drift(tmp_path: Path) -> None:
    url = "https://secondary.example/paper.pdf"
    response = _Response(b"%PDF-1.4\nbytes\n", content_type="application/pdf", url=url)
    with pytest.raises(SourceAcquisitionError, match="authority"):
        acquire_source(
            SourceRequest("P1", "DEEP_FULL_TEXT", url, "sources/P1.pdf", "SEARCH_RESULT"),
            corpus_root=tmp_path,
            session=_Session(response),
        )
    with pytest.raises(SourceAcquisitionError, match="escapes"):
        acquire_source(
            SourceRequest("P1", "DEEP_FULL_TEXT", url, "../P1.pdf", "AUTHOR_HOSTED"),
            corpus_root=tmp_path,
            session=_Session(response),
        )

    request = SourceRequest("P1", "DEEP_FULL_TEXT", url, "sources/P1.pdf", "AUTHOR_HOSTED")
    acquire_source(request, corpus_root=tmp_path, session=_Session(response))
    (tmp_path / request.destination).write_bytes(b"tampered")
    with pytest.raises(SourceAcquisitionError, match="existing source hash"):
        acquire_source(request, corpus_root=tmp_path, session=_Session(response))
