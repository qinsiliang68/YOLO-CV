"""Plan fail-closed full-method source acquisition for SCREENED reading."""

from __future__ import annotations

from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

from .literature_broad_source_queue_v2 import _authority, _looks_like_pdf


class ScreenedSourcePlanError(RuntimeError):
    """Raised when a SCREENED source plan cannot preserve evidence identity."""


PLAN_FIELDS = (
    "reading_rank",
    "selection_role",
    "paper_id",
    "queue_id",
    "title",
    "source_action",
    "artifact_role",
    "url",
    "destination",
    "source_authority",
    "resolution_reason",
    "broad_source_path",
    "broad_source_format",
    "method_source_path",
    "method_source_format",
    "method_source_sha256",
    "method_source_bytes",
    "method_source_origin",
    "primary_url",
    "full_text_url",
    "doi",
    "screened_credit",
)
REQUEST_FIELDS = (
    "paper_id",
    "artifact_role",
    "url",
    "destination",
    "source_authority",
)
INPUT_FIELDS = ("input_role", "path", "bytes", "sha256")


def _usable(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and not text.startswith("NOT_")


def _not_applicable(reason: str) -> str:
    return f"NOT_APPLICABLE_WITH_REASON:{reason}"


def _safe_subdir(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ScreenedSourcePlanError("source_subdir must be a safe relative path")
    return path


def build_screened_source_plan_rows(
    screening_rows: Sequence[Mapping[str, Any]],
    discovery_rows: Sequence[Mapping[str, Any]],
    *,
    source_subdir: str,
) -> list[dict[str, str]]:
    """Classify each reading item as verified reuse, direct PDF, or discovery."""

    subdir = _safe_subdir(source_subdir)
    discovery_by_id: dict[str, Mapping[str, Any]] = {}
    for row in discovery_rows:
        queue_id = str(row.get("queue_id", "")).strip()
        if not queue_id:
            raise ScreenedSourcePlanError("discovery row has no queue_id")
        if queue_id in discovery_by_id:
            raise ScreenedSourcePlanError(f"duplicate discovery queue_id: {queue_id}")
        discovery_by_id[queue_id] = row

    seen_papers: set[str] = set()
    seen_queues: set[str] = set()
    plan: list[dict[str, str]] = []
    for fallback_rank, row in enumerate(screening_rows, start=1):
        paper_id = str(row.get("paper_id", "")).strip()
        queue_id = str(row.get("queue_id", "")).strip()
        if not paper_id or not queue_id:
            raise ScreenedSourcePlanError("screening row requires paper_id and queue_id")
        if paper_id in seen_papers or queue_id in seen_queues:
            raise ScreenedSourcePlanError(
                f"duplicate screening identity: paper_id={paper_id} queue_id={queue_id}"
            )
        seen_papers.add(paper_id)
        seen_queues.add(queue_id)
        source_format = str(row.get("broad_source_format", "")).strip().upper()
        if source_format not in {"PDF", "HTML"}:
            raise ScreenedSourcePlanError(
                f"{paper_id}: unsupported broad source format {source_format!r}"
            )

        method_source_path = str(
            row.get("method_source_path", row.get("broad_source_path", ""))
        ).strip()
        method_source_format = str(
            row.get("method_source_format", source_format)
        ).strip().upper()
        method_source_origin = str(
            row.get("method_source_origin", "BROAD_SOURCE")
        ).strip()
        if method_source_format not in {"PDF", "HTML"}:
            raise ScreenedSourcePlanError(
                f"{paper_id}: unsupported method source format {method_source_format!r}"
            )

        discovery = discovery_by_id.get(queue_id)
        if method_source_format == "HTML" and discovery is None:
            raise ScreenedSourcePlanError(
                f"{queue_id}: HTML screening item has no discovery identity"
            )
        discovery = discovery or {}
        primary_url = str(discovery.get("primary_url", "")).strip()
        full_text_url = str(discovery.get("full_text_url", "")).strip()
        doi = str(discovery.get("doi", "")).strip()

        if method_source_format == "PDF":
            if method_source_origin == "VERIFIED_OVERRIDE":
                action = "REUSE_VERIFIED_METHOD_PDF"
                url = str(row.get("method_source_url", "")).strip() or _not_applicable(
                    "URL retained in method-source override ledger"
                )
                authority = str(
                    row.get("method_source_authority", "")
                ).strip() or _not_applicable(
                    "authority retained in method-source override ledger"
                )
                reason = "HASH_VALIDATED_METHOD_SOURCE_OVERRIDE"
            else:
                action = "REUSE_VERIFIED_BROAD_PDF"
                url = _not_applicable("verified BROAD PDF is reused")
                authority = _not_applicable(
                    "authority retained in BROAD acquisition ledger"
                )
                reason = "BROAD_SOURCE_IS_VERIFIED_PDF"
            destination = method_source_path
        elif _usable(full_text_url) and _looks_like_pdf(full_text_url):
            parsed = urlparse(full_text_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ScreenedSourcePlanError(
                    f"{queue_id}: malformed direct full-text URL {full_text_url!r}"
                )
            action = "ACQUIRE_DIRECT_PDF"
            url = full_text_url.replace("http://", "https://", 1)
            destination = str(subdir / f"{paper_id}.pdf")
            authority = _authority(url)
            reason = "DISCOVERY_METADATA_HAS_DIRECT_PDF_URL"
        else:
            action = "DISCOVERY_REQUIRED"
            url = _not_applicable("no verified direct PDF URL")
            destination = _not_applicable("full-text source must be discovered first")
            authority = _not_applicable("full-text source not selected")
            reason = (
                "FULL_TEXT_FIELD_IS_LANDING_PAGE"
                if _usable(full_text_url)
                else "NO_FULL_TEXT_URL_IN_DISCOVERY_METADATA"
            )

        plan.append(
            {
                "reading_rank": str(row.get("reading_rank", fallback_rank)),
                "selection_role": str(row.get("selection_role", "")).strip(),
                "paper_id": paper_id,
                "queue_id": queue_id,
                "title": str(row.get("title", "")).strip(),
                "source_action": action,
                "artifact_role": "METHOD_SOURCE",
                "url": url,
                "destination": destination,
                "source_authority": authority,
                "resolution_reason": reason,
                "broad_source_path": str(row.get("broad_source_path", "")).strip(),
                "broad_source_format": source_format,
                "method_source_path": method_source_path,
                "method_source_format": method_source_format,
                "method_source_sha256": str(
                    row.get("method_source_sha256", row.get("broad_source_sha256", ""))
                ).strip(),
                "method_source_bytes": str(
                    row.get("method_source_bytes", row.get("broad_source_bytes", ""))
                ).strip(),
                "method_source_origin": method_source_origin,
                "primary_url": primary_url or _not_applicable("discovery metadata unavailable"),
                "full_text_url": full_text_url
                or _not_applicable("not reported in discovery metadata"),
                "doi": doi or _not_applicable("DOI not reported in discovery metadata"),
                "screened_credit": "NOT_ASSESSED_AT_BROAD_LEVEL",
            }
        )
    return plan


def _read_csv(path: Path, required: Iterable[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ScreenedSourcePlanError(f"required CSV missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(set(required) - fields)
        if missing:
            raise ScreenedSourcePlanError(f"{path.name} missing fields: {missing}")
        return [dict(row) for row in reader]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_screened_source_plan(
    screening_queue: str | Path,
    *,
    discovery_paths: Sequence[str | Path],
    output_root: str | Path,
    source_subdir: str = "sources/screened_method_v2",
    replace_existing: bool = False,
) -> Path:
    """Write an atomic, hash-bound source plan without granting reading credit."""

    queue_path = Path(screening_queue).resolve()
    discovery_files = tuple(Path(path).resolve() for path in discovery_paths)
    output = Path(output_root).resolve()
    if not discovery_files:
        raise ScreenedSourcePlanError("at least one discovery input is required")
    if output.exists() and not replace_existing:
        raise ScreenedSourcePlanError(f"source plan output already exists: {output}")

    screening_rows = _read_csv(
        queue_path,
        (
            "reading_rank",
            "selection_role",
            "paper_id",
            "queue_id",
            "title",
            "broad_source_path",
            "broad_source_format",
        ),
    )
    discovery_rows: list[dict[str, str]] = []
    for path in discovery_files:
        discovery_rows.extend(
            _read_csv(path, ("queue_id", "primary_url", "full_text_url", "doi"))
        )
    plan = build_screened_source_plan_rows(
        screening_rows,
        discovery_rows,
        source_subdir=source_subdir,
    )

    temp = output.parent / f".{output.name}.tmp"
    if temp.exists():
        raise ScreenedSourcePlanError(f"stale source-plan temp exists: {temp}")
    temp.mkdir(parents=True)
    try:
        requests = [row for row in plan if row["source_action"] == "ACQUIRE_DIRECT_PDF"]
        unresolved = [row for row in plan if row["source_action"] == "DISCOVERY_REQUIRED"]
        reused = [
            row
            for row in plan
            if row["source_action"]
            in {"REUSE_VERIFIED_BROAD_PDF", "REUSE_VERIFIED_METHOD_PDF"}
        ]
        _write_csv(temp / "SCREENED_METHOD_SOURCE_PLAN.csv", plan, PLAN_FIELDS)
        _write_csv(
            temp / "SCREENED_METHOD_SOURCE_REQUESTS.csv",
            requests,
            REQUEST_FIELDS,
        )
        _write_csv(temp / "SCREENED_METHOD_SOURCE_DISCOVERY.csv", unresolved, PLAN_FIELDS)
        _write_csv(temp / "SCREENED_METHOD_SOURCE_REUSE.csv", reused, PLAN_FIELDS)
        input_rows = [
            {
                "input_role": "SCREENING_QUEUE" if path == queue_path else "DISCOVERY_BATCH",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (queue_path, *discovery_files)
        ]
        _write_csv(temp / "SOURCE_PLAN_INPUTS.csv", input_rows, INPUT_FIELDS)
        receipt = {
            "schema_version": "2.0",
            "status": "PASS",
            "created_at": datetime.now().astimezone().isoformat(),
            "planned_count": len(plan),
            "reused_verified_pdf_count": len(reused),
            "direct_pdf_request_count": len(requests),
            "discovery_required_count": len(unresolved),
            "formal_screened_increment": 0,
            "formal_training_started": False,
            "engineering_gate_generated": False,
            "blind_holdout_opened": False,
            "input_manifest_sha256": _sha256(temp / "SOURCE_PLAN_INPUTS.csv"),
        }
        (temp / "SOURCE_PLAN_RECEIPT.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            backup = output.parent / f".{output.name}.previous"
            if backup.exists():
                raise ScreenedSourcePlanError(f"stale source-plan backup exists: {backup}")
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
    return output
