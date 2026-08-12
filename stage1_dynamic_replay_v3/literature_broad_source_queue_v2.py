"""Build deterministic primary-source acquisition requests for broad screening."""

from __future__ import annotations

from pathlib import PurePosixPath
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse, urlunparse


_OFFICIAL_REPOSITORIES = {
    "arxiv.org",
    "export.arxiv.org",
    "aclanthology.org",
    "www.aclweb.org",
    "openreview.net",
    "papers.nips.cc",
    "proceedings.neurips.cc",
    "proceedings.mlr.press",
    "raw.githubusercontent.com",
    "openaccess.thecvf.com",
    "pmc.ncbi.nlm.nih.gov",
    "zenodo.org",
    "hal.science",
}
_PRIMARY_PUBLISHERS = {
    "doi.org",
    "dx.doi.org",
    "ojs.aaai.org",
    "www.ijcai.org",
    "ieeexplore.ieee.org",
    "link.springer.com",
    "dl.acm.org",
    "www.frontiersin.org",
    "public-pages-files-2025.frontiersin.org",
    "www.mdpi.com",
    "direct.mit.edu",
    "www.nature.com",
    "onlinelibrary.wiley.com",
    "academic.oup.com",
    "www.sciencedirect.com",
    "projecteuclid.org",
    "journals.plos.org",
    "www.pnas.org",
}


def _usable(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and not text.startswith("NOT_")


def _https(value: str) -> str:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"invalid source URL: {value}")
    if parsed.scheme == "http":
        parsed = parsed._replace(scheme="https")
    return urlunparse(parsed)


def _arxiv_pdf(value: str) -> str | None:
    parsed = urlparse(value)
    if parsed.netloc.casefold() not in {"arxiv.org", "export.arxiv.org"}:
        return None
    match = re.fullmatch(r"/abs/([^/?#]+)", parsed.path.rstrip("/"))
    if not match:
        return None
    return f"https://arxiv.org/pdf/{match.group(1)}"


def _looks_like_pdf(value: str) -> bool:
    parsed = urlparse(value)
    path = parsed.path.casefold()
    return any(
        marker in path
        for marker in (
            ".pdf",
            "/pdf/",
            "/pdf",
            "/article/download/",
            "/content/pdf/",
            "/bitstream/",
            "/servlets/purl/",
        )
    ) or "type=printable" in parsed.query.casefold()


def _authority(value: str) -> str:
    domain = urlparse(value).netloc.casefold()
    if domain in _OFFICIAL_REPOSITORIES or domain.endswith(".edu") or domain.endswith(".ac.uk"):
        return "OFFICIAL_REPOSITORY"
    if domain in _PRIMARY_PUBLISHERS or domain.endswith(".org") and "doi" in domain:
        return "PRIMARY_PUBLISHER"
    return "AUTHOR_HOSTED"


def build_broad_source_requests(
    rows: Sequence[Mapping[str, Any]], *, source_subdir: str
) -> list[dict[str, str]]:
    """Select one primary source target per unresolved manual-review group."""

    subdir = PurePosixPath(source_subdir)
    if subdir.is_absolute() or ".." in subdir.parts:
        raise ValueError("source_subdir must be a safe relative path")
    seen: set[str] = set()
    requests: list[dict[str, str]] = []
    for row in rows:
        queue_id = str(row.get("queue_id", "")).strip()
        if not queue_id:
            raise ValueError("queue_id is required")
        if queue_id in seen:
            raise ValueError(f"duplicate queue_id: {queue_id}")
        seen.add(queue_id)
        primary = str(row.get("primary_url", "")).strip()
        full_text = str(row.get("full_text_url", "")).strip()
        if _usable(full_text):
            selected = _https(full_text)
            reason = "DIRECT_FULL_TEXT_URL"
            full_text_claimed = _looks_like_pdf(selected)
        elif _usable(primary) and (derived := _arxiv_pdf(_https(primary))) is not None:
            selected = derived
            reason = "DERIVED_ARXIV_PDF_FROM_PRIMARY"
            full_text_claimed = True
        elif _usable(primary):
            selected = _https(primary)
            reason = "PRIMARY_LANDING_PAGE_ONLY"
            full_text_claimed = False
        else:
            raise ValueError(f"{queue_id}: no usable primary source URL")
        suffix = ".pdf" if full_text_claimed else ".html"
        requests.append(
            {
                "paper_id": queue_id,
                "artifact_role": "BROAD_SOURCE",
                "url": selected,
                "destination": str(subdir / f"{queue_id}{suffix}"),
                "source_authority": _authority(selected),
                "selection_reason": reason,
                "full_text_claimed": str(full_text_claimed).lower(),
                "reading_credit_granted": "false",
            }
        )
    return sorted(requests, key=lambda row: row["paper_id"])
