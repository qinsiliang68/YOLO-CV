import csv
import hashlib
import json
from pathlib import Path

from stage1_dynamic_replay_v3.literature_broad_content_scope_v4 import (
    audit_broad_content_scope,
)


SCOPES = ("TITLE", "ABSTRACT", "PROBLEM", "METHOD_OVERVIEW", "CONCLUSION")
REGISTRY_FIELDS = (
    "paper_id",
    "tier",
    "canonical_work_id",
    "title",
    "authors",
    "year",
    "venue",
    "primary_url",
    "doi",
    "arxiv_id",
    "openreview_id",
    "note_path",
    "source_path",
    "source_sha256",
    "source_bytes",
)


def _crossref_bytes(title: str, abstract: str | None) -> bytes:
    message: dict[str, object] = {"title": [title]}
    if abstract is not None:
        message["abstract"] = abstract
    return json.dumps(
        {"status": "ok", "message-type": "work", "message": message},
        ensure_ascii=False,
    ).encode("utf-8")


def _paper(
    root: Path,
    *,
    paper_id: str,
    title: str,
    authors: str = "Alice Smith; Bob Jones",
    year: int = 2024,
    doi: str = "NOT_APPLICABLE_WITH_REASON:no DOI",
    arxiv_id: str = "NOT_APPLICABLE_WITH_REASON:no arXiv",
    primary_url: str | None = None,
    source_bytes: bytes | None = None,
    abstract: str | None = (
        "We address a training data selection problem. Our method selects useful "
        "samples from model state. Experiments conclude that the method improves efficiency."
    ),
    include_scope_evidence: bool = True,
) -> dict[str, str]:
    primary_url = primary_url or f"https://primary.example/{paper_id}"
    source_bytes = source_bytes or _crossref_bytes(title, abstract)
    source_rel = f"sources/{paper_id}.html"
    source = root / source_rel
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(source_bytes)
    digest = hashlib.sha256(source_bytes).hexdigest().upper()

    evidence = None
    if include_scope_evidence:
        assert abstract is not None
        evidence = {
            "TITLE": {"locator": "Crossref message.title", "quote": title},
            "ABSTRACT": {
                "locator": "Crossref message.abstract",
                "quote": "We address a training data selection problem.",
            },
            "PROBLEM": {
                "locator": "Crossref message.abstract sentence 1",
                "quote": "training data selection problem",
            },
            "METHOD_OVERVIEW": {
                "locator": "Crossref message.abstract sentence 2",
                "quote": "Our method selects useful samples from model state.",
            },
            "CONCLUSION": {
                "locator": "Crossref message.abstract sentence 3",
                "quote": "Experiments conclude that the method improves efficiency.",
            },
        }
    reading: dict[str, object] = {
        "scopes": list(SCOPES),
        "sections_checked": [
            "Primary title and identity",
            "Primary abstract",
            "Research problem",
            "Method overview",
            "Conclusion",
        ],
    }
    if evidence is not None:
        reading["source_scope_evidence"] = evidence
    metadata = {
        "paper_id": paper_id,
        "tier": "BROAD",
        "identity": {"title": title},
        "source_artifact": {
            "path": source_rel,
            "kind": "OFFICIAL_LANDING_HTML",
            "bytes": len(source_bytes),
            "sha256": digest,
        },
        "reading": reading,
    }
    note = root / "notes" / f"{paper_id}.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "<!-- STAGE1_EVIDENCE_V2 -->\n```json\n"
        + json.dumps(metadata, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    return {
        "paper_id": paper_id,
        "tier": "BROAD",
        "canonical_work_id": f"CW{paper_id[1:]}",
        "title": title,
        "authors": authors,
        "year": str(year),
        "venue": "Fixture Venue",
        "primary_url": primary_url,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "openreview_id": "NOT_APPLICABLE_WITH_REASON:no OpenReview",
        "note_path": f"notes/{paper_id}.md",
        "source_path": source_rel,
        "source_sha256": digest,
        "source_bytes": str(len(source_bytes)),
    }


def _registry(root: Path, rows: list[dict[str, str]]) -> None:
    with (root / "CANONICAL_WORKS.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _codes(payload: dict[str, object], paper_id: str) -> set[str]:
    paper = payload["per_paper"][paper_id]
    return {item["code"] for item in paper["findings"]}


def test_pass_requires_source_bound_evidence_for_every_declared_scope(tmp_path: Path) -> None:
    _registry(
        tmp_path,
        [_paper(tmp_path, paper_id="P0001", title="Reliable Dynamic Sample Selection")],
    )

    report = audit_broad_content_scope(tmp_path, expected_count=1)
    payload = report.as_dict()

    assert report.status == "PASS"
    assert report.promotion_allowed is True
    assert payload["per_paper"]["P0001"]["status"] == "PASS"


def test_crossref_without_abstract_and_template_declaration_fail_closed(
    tmp_path: Path,
) -> None:
    _registry(
        tmp_path,
        [
            _paper(
                tmp_path,
                paper_id="P0001",
                title="Metadata Only Study",
                abstract=None,
                include_scope_evidence=False,
            )
        ],
    )

    payload = audit_broad_content_scope(tmp_path, expected_count=1).as_dict()
    codes = _codes(payload, "P0001")

    assert payload["status"] == "FAIL"
    assert payload["promotion_allowed"] is False
    assert "SOURCE_ABSTRACT_MISSING" in codes
    assert "FIXED_SECTIONS_CHECKED_NOT_CONTENT_EVIDENCE" in codes
    for scope in SCOPES:
        assert any(
            item["field"] == f"reading.source_scope_evidence.{scope}"
            for item in payload["per_paper"]["P0001"]["findings"]
        )


def test_crossref_without_abstract_rejects_claimed_non_title_quotes(
    tmp_path: Path,
) -> None:
    title = "Metadata Cannot Prove Full Reading"
    source_bytes = _crossref_bytes(title, abstract=None)
    _registry(
        tmp_path,
        [
            _paper(
                tmp_path,
                paper_id="P0001",
                title=title,
                source_bytes=source_bytes,
            )
        ],
    )

    payload = audit_broad_content_scope(tmp_path, expected_count=1).as_dict()
    findings = payload["per_paper"]["P0001"]["findings"]

    assert payload["status"] == "FAIL"
    assert "SOURCE_ABSTRACT_MISSING" in _codes(payload, "P0001")
    for scope in ("ABSTRACT", "PROBLEM", "METHOD_OVERVIEW", "CONCLUSION"):
        assert any(
            item["code"] == "SOURCE_SCOPE_QUOTE_NOT_FOUND"
            and item["field"] == f"reading.source_scope_evidence.{scope}.quote"
            for item in findings
        )


def test_registered_title_and_claimed_title_quote_must_match_source_bytes(
    tmp_path: Path,
) -> None:
    registered_title = "Reliable Replay Sample Selection"
    source_bytes = _crossref_bytes(
        "Unrelated Statistical Calibration Study",
        (
            "We address a training data selection problem. Our method selects useful "
            "samples from model state. Experiments conclude that the method improves "
            "efficiency."
        ),
    )
    _registry(
        tmp_path,
        [
            _paper(
                tmp_path,
                paper_id="P0001",
                title=registered_title,
                source_bytes=source_bytes,
            )
        ],
    )

    payload = audit_broad_content_scope(tmp_path, expected_count=1).as_dict()
    findings = payload["per_paper"]["P0001"]["findings"]

    assert payload["status"] == "FAIL"
    assert "SOURCE_TITLE_MISMATCH" in _codes(payload, "P0001")
    assert any(
        item["code"] == "SOURCE_SCOPE_QUOTE_NOT_FOUND"
        and item["field"] == "reading.source_scope_evidence.TITLE.quote"
        for item in findings
    )


def test_same_source_sha_doi_and_arxiv_are_per_paper_failures(tmp_path: Path) -> None:
    title = "One Canonical Study"
    shared = _crossref_bytes(title, "A problem. A method. A conclusion.")
    rows = [
        _paper(
            tmp_path,
            paper_id=paper_id,
            title=title,
            doi="10.1000/shared",
            arxiv_id="2401.01234",
            primary_url="https://arxiv.org/pdf/2401.01234",
            source_bytes=shared,
        )
        for paper_id in ("P0001", "P0002")
    ]
    _registry(tmp_path, rows)

    payload = audit_broad_content_scope(tmp_path, expected_count=2).as_dict()

    for paper_id in ("P0001", "P0002"):
        codes = _codes(payload, paper_id)
        assert "DUPLICATE_SOURCE_SHA256" in codes
        assert "DUPLICATE_DOI" in codes
        assert "DUPLICATE_ARXIV_ID" in codes
        assert payload["per_paper"][paper_id]["status"] == "FAIL"


def test_legal_missing_identifier_markers_do_not_form_duplicate_groups(
    tmp_path: Path,
) -> None:
    rows = [
        _paper(
            tmp_path,
            paper_id="P0001",
            title="Reliable Selection Under Label Noise",
            authors="Alice Smith; Bob Jones",
        ),
        _paper(
            tmp_path,
            paper_id="P0002",
            title="Coverage Control for Replay Buffers",
            authors="Carol White; David Brown",
        ),
    ]
    _registry(tmp_path, rows)

    payload = audit_broad_content_scope(tmp_path, expected_count=2).as_dict()

    assert payload["status"] == "PASS"
    assert "DUPLICATE_DOI" not in _codes(payload, "P0001")
    assert "DUPLICATE_DOI" not in _codes(payload, "P0002")
    assert "DUPLICATE_ARXIV_ID" not in _codes(payload, "P0001")
    assert "DUPLICATE_ARXIV_ID" not in _codes(payload, "P0002")


def test_near_identical_title_same_authors_and_year_is_clear_duplicate(
    tmp_path: Path,
) -> None:
    rows = [
        _paper(
            tmp_path,
            paper_id="P0001",
            title="Estimating Training Data Influence by Tracking Gradient Descent",
            doi="10.1000/tracking",
        ),
        _paper(
            tmp_path,
            paper_id="P0002",
            title="Estimating Training Data Influence by Tracing Gradient Descent",
            doi="10.1000/tracing",
        ),
    ]
    _registry(tmp_path, rows)

    payload = audit_broad_content_scope(tmp_path, expected_count=2).as_dict()

    assert payload["status"] == "FAIL"
    assert "CANONICAL_VERSION_DUPLICATE" in _codes(payload, "P0001")
    assert "CANONICAL_VERSION_DUPLICATE" in _codes(payload, "P0002")


def test_probable_version_pair_is_review_required_and_blocks_promotion(
    tmp_path: Path,
) -> None:
    authors = "Arpit Garg; Cuong Nguyen; Rafael Felix; Thanh Do; Gustavo Carneiro"
    rows = [
        _paper(
            tmp_path,
            paper_id="P0001",
            title="PASS Peer Agreement Based Sample Selection for Training with Noisy Labels",
            authors=authors,
            year=2023,
            doi="10.1000/pass-preprint",
        ),
        _paper(
            tmp_path,
            paper_id="P0002",
            title=(
                "PASS Peer Agreement Based Sample Selection for Training with Instance "
                "Dependent Noisy Labels"
            ),
            authors=authors,
            year=2025,
            doi="10.1000/pass-journal",
        ),
    ]
    _registry(tmp_path, rows)

    payload = audit_broad_content_scope(tmp_path, expected_count=2).as_dict()

    assert payload["status"] == "REVIEW_REQUIRED"
    assert payload["promotion_allowed"] is False
    assert "CANONICAL_VERSION_REVIEW_REQUIRED" in _codes(payload, "P0001")
    assert "CANONICAL_VERSION_REVIEW_REQUIRED" in _codes(payload, "P0002")
