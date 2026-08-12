"""Atomic, hash-bound acquisition of primary literature source bytes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


ALLOWED_AUTHORITIES = {"PRIMARY_PUBLISHER", "AUTHOR_HOSTED", "OFFICIAL_REPOSITORY"}
ALLOWED_ROLES = {"BROAD_SOURCE", "METHOD_SOURCE", "DEEP_FULL_TEXT", "SUPPLEMENT"}
USER_AGENT = "YOLO-CV-Stage1-Literature-Evidence/2.0"


class SourceAcquisitionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceRequest:
    paper_id: str
    artifact_role: str
    url: str
    destination: str
    source_authority: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _safe_destination(root: Path, value: str) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        raise SourceAcquisitionError(f"destination must be relative: {value}")
    destination = (root / raw).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as exc:
        raise SourceAcquisitionError(f"destination escapes corpus root: {value}") from exc
    return destination


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _validate_request(request: SourceRequest, root: Path) -> Path:
    if not request.paper_id.strip():
        raise SourceAcquisitionError("paper_id is required")
    if request.artifact_role not in ALLOWED_ROLES:
        raise SourceAcquisitionError(f"unsupported artifact role: {request.artifact_role}")
    if request.source_authority not in ALLOWED_AUTHORITIES:
        raise SourceAcquisitionError(
            f"source authority must be primary/official, observed {request.source_authority!r}"
        )
    parsed = urlparse(request.url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SourceAcquisitionError(f"invalid HTTP(S) source URL: {request.url}")
    return _safe_destination(root, request.destination)


def _load_existing(
    request: SourceRequest,
    destination: Path,
    receipt_path: Path,
) -> dict[str, Any] | None:
    if not destination.exists() and not receipt_path.exists():
        return None
    if not destination.is_file() or not receipt_path.is_file():
        raise SourceAcquisitionError(
            f"existing source/receipt pair is incomplete for {request.paper_id}: {destination}"
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceAcquisitionError(f"invalid existing source receipt: {receipt_path}") from exc
    for field, expected in (
        ("paper_id", request.paper_id),
        ("artifact_role", request.artifact_role),
        ("requested_url", request.url),
        ("source_authority", request.source_authority),
    ):
        if receipt.get(field) != expected:
            raise SourceAcquisitionError(
                f"existing source receipt {field} mismatch for {request.paper_id}"
            )
    data = destination.read_bytes()
    observed = _sha256(data)
    if receipt.get("sha256") != observed:
        raise SourceAcquisitionError(
            f"existing source hash mismatch for {request.paper_id}: "
            f"receipt={receipt.get('sha256')} observed={observed}"
        )
    row = receipt.get("ledger_row")
    if not isinstance(row, dict):
        raise SourceAcquisitionError(f"existing source receipt has no ledger row: {receipt_path}")
    return row


def acquire_source(
    request: SourceRequest,
    *,
    corpus_root: str | Path,
    session: Any | None = None,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Download one source or return its immutable verified acquisition record."""

    root = Path(corpus_root).resolve()
    destination = _validate_request(request, root)
    receipt_path = destination.with_suffix(destination.suffix + ".receipt.json")
    existing = _load_existing(request, destination, receipt_path)
    if existing is not None:
        return existing

    client = session or requests.Session()
    if hasattr(client, "headers"):
        client.headers["User-Agent"] = USER_AGENT
    response = client.get(request.url, timeout=timeout_seconds, allow_redirects=True)
    if int(response.status_code) != 200:
        raise SourceAcquisitionError(
            f"{request.paper_id}: HTTP {response.status_code} from {request.url}"
        )
    data = bytes(response.content)
    if not data:
        raise SourceAcquisitionError(f"{request.paper_id}: source response is empty")
    if destination.suffix.casefold() == ".pdf" and not data.startswith(b"%PDF-"):
        raise SourceAcquisitionError(
            f"{request.paper_id}: expected PDF signature but received "
            f"{response.headers.get('content-type', 'unknown')}"
        )
    digest = _sha256(data)
    retrieved_at = datetime.now(timezone.utc).astimezone().isoformat()
    relative = destination.relative_to(root).as_posix()
    ledger_row: dict[str, Any] = {
        "paper_id": request.paper_id,
        "artifact_role": request.artifact_role,
        "path": relative,
        "url": request.url,
        "retrieved_at": retrieved_at,
        "http_status": 200,
        "content_type": str(response.headers.get("content-type", "NOT_REPORTED_BY_SOURCE")),
        "bytes": len(data),
        "sha256": digest,
        "retrieval_method": "HTTP_DOWNLOAD",
        "source_authority": request.source_authority,
        "final_url": str(getattr(response, "url", request.url)),
        "receipt_path": receipt_path.relative_to(root).as_posix(),
        "reused_existing": False,
    }
    receipt = {
        "schema_version": "1.0",
        "paper_id": request.paper_id,
        "artifact_role": request.artifact_role,
        "requested_url": request.url,
        "final_url": ledger_row["final_url"],
        "source_authority": request.source_authority,
        "retrieved_at": retrieved_at,
        "bytes": len(data),
        "sha256": digest,
        "ledger_row": ledger_row,
    }
    _atomic_write(destination, data)
    try:
        _atomic_write(
            receipt_path,
            (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return ledger_row
