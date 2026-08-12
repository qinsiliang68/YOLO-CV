from stage1_dynamic_replay_v3.literature_broad_source_queue_v2 import (
    build_broad_source_requests,
)


def _row(queue_id: str, *, primary: str, full_text: str) -> dict[str, str]:
    return {
        "queue_id": queue_id,
        "primary_url": primary,
        "full_text_url": full_text,
    }


def test_prefers_direct_full_text_and_assigns_stable_pdf_destination() -> None:
    rows = [
        _row(
            "RG001",
            primary="https://arxiv.org/abs/2301.00001",
            full_text="https://arxiv.org/pdf/2301.00001",
        )
    ]

    requests = build_broad_source_requests(rows, source_subdir="sources/broad_screen")

    assert requests[0]["paper_id"] == "RG001"
    assert requests[0]["url"] == "https://arxiv.org/pdf/2301.00001"
    assert requests[0]["destination"] == "sources/broad_screen/RG001.pdf"
    assert requests[0]["source_authority"] == "OFFICIAL_REPOSITORY"
    assert requests[0]["selection_reason"] == "DIRECT_FULL_TEXT_URL"


def test_converts_arxiv_abstract_page_when_full_text_is_missing() -> None:
    rows = [
        _row(
            "RG002",
            primary="http://arxiv.org/abs/2102.12345",
            full_text="NOT_REPORTED_BY_SOURCE",
        )
    ]

    requests = build_broad_source_requests(rows, source_subdir="sources/broad_screen")

    assert requests[0]["url"] == "https://arxiv.org/pdf/2102.12345"
    assert requests[0]["destination"].endswith(".pdf")
    assert requests[0]["selection_reason"] == "DERIVED_ARXIV_PDF_FROM_PRIMARY"


def test_falls_back_to_primary_landing_page_without_claiming_full_text() -> None:
    rows = [
        _row(
            "RG003",
            primary="https://doi.org/10.1000/example",
            full_text="NOT_REPORTED_BY_SOURCE",
        )
    ]

    requests = build_broad_source_requests(rows, source_subdir="sources/broad_screen")

    assert requests[0]["url"] == "https://doi.org/10.1000/example"
    assert requests[0]["destination"] == "sources/broad_screen/RG003.html"
    assert requests[0]["selection_reason"] == "PRIMARY_LANDING_PAGE_ONLY"
    assert requests[0]["full_text_claimed"] == "false"


def test_output_is_invariant_to_input_order_and_rejects_duplicate_queue_id() -> None:
    first = _row(
        "RG001",
        primary="https://arxiv.org/abs/2301.00001",
        full_text="https://arxiv.org/pdf/2301.00001",
    )
    second = _row(
        "RG002",
        primary="https://publisher.example/paper",
        full_text="NOT_REPORTED_BY_SOURCE",
    )

    forward = build_broad_source_requests([first, second], source_subdir="sources/broad")
    reverse = build_broad_source_requests([second, first], source_subdir="sources/broad")

    assert forward == reverse
    try:
        build_broad_source_requests([first, dict(first)], source_subdir="sources/broad")
    except ValueError as exc:
        assert "duplicate queue_id" in str(exc)
    else:
        raise AssertionError("duplicate queue IDs must fail closed")
