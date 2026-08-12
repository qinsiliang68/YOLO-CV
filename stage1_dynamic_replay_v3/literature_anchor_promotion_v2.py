"""Promote validated supplemental anchors into one immutable BROAD input batch."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .literature_anchor_contract_v2 import validate_anchor_contract


class AnchorPromotionError(RuntimeError):
    """Raised when supplemental anchor evidence cannot be promoted exactly."""


REVIEW_INPUT_FIELDS = (
    "queue_id",
    "review_group_id",
    "queue_status",
    "identity_status",
    "title",
    "authors",
    "year",
    "venue",
    "primary_url",
    "full_text_url",
    "doi",
    "abstract",
    "rq_ids",
    "source_origins",
    "candidate_version_ids",
    "candidate_version_count",
    "discovery_decisions",
    "grouping_evidence",
    "v2_counted_tier",
    "blind_order_key",
    "blind_review_rank",
)
SOURCE_VALIDATION_FIELDS = (
    "paper_id",
    "title",
    "path",
    "bytes",
    "sha256",
    "source_format",
    "page_count",
    "title_token_coverage",
    "source_authority",
    "source_url",
    "receipt_path",
    "probe_tool",
    "source_superseded",
    "superseded_sha256",
    "supersession_reason",
    "reading_credit_granted",
)
DECISION_FIELDS = (
    "queue_id",
    "decision",
    "canonical_title",
    "primary_url_checked",
    "source_authority",
    "checked_at",
    "reading_scope",
    "direct_rq_ids",
    "relevance_class",
    "problem_summary_zh",
    "method_overview_zh",
    "conclusion_summary_zh",
    "critical_review_zh",
    "stage1_transfer_zh",
    "cannot_infer_zh",
    "exclusion_reason",
    "reviewer",
)


@dataclass(frozen=True)
class AnchorPromotionRows:
    review_input_rows: tuple[dict[str, Any], ...]
    source_validation_rows: tuple[dict[str, Any], ...]
    decision_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class AnchorPromotionResult:
    status: str
    promoted_count: int
    review_input_path: Path
    source_validation_path: Path
    decision_path: Path
    receipt_path: Path
    formal_broad_increment: int = 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise AnchorPromotionError(f"required anchor input missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _index(
    rows: Sequence[Mapping[str, str]],
    *,
    field: str,
    role: str,
) -> dict[str, Mapping[str, str]]:
    indexed: dict[str, Mapping[str, str]] = {}
    for row in rows:
        value = row.get(field, "").strip()
        if not value:
            raise AnchorPromotionError(f"{role} has blank {field}")
        if value in indexed:
            raise AnchorPromotionError(f"{role} has duplicate {field}: {value}")
        indexed[value] = row
    return indexed


def _required(row: Mapping[str, str], field: str, queue_id: str) -> str:
    value = row.get(field, "").strip()
    if not value:
        raise AnchorPromotionError(f"{queue_id} missing {field}")
    return value


def build_anchor_promotion_rows(
    *,
    contract_rows: Sequence[Mapping[str, str]],
    queue_rows: Sequence[Mapping[str, str]],
    source_rows: Sequence[Mapping[str, str]],
    decision_rows: Sequence[Mapping[str, str]],
) -> AnchorPromotionRows:
    """Cross-bind four evidence tables and build the standard batch rows."""

    contract_by_id = _index(contract_rows, field="queue_id", role="anchor contract")
    queue_by_id = _index(queue_rows, field="queue_id", role="anchor queue")
    source_by_id = _index(source_rows, field="paper_id", role="anchor sources")
    decision_by_id = _index(decision_rows, field="queue_id", role="anchor decisions")
    identity_sets = {
        "contract": set(contract_by_id),
        "queue": set(queue_by_id),
        "source": set(source_by_id),
        "decision": set(decision_by_id),
    }
    expected = identity_sets["contract"]
    if any(values != expected for values in identity_sets.values()):
        details = ", ".join(
            f"{name}={len(values)}" for name, values in identity_sets.items()
        )
        raise AnchorPromotionError(f"anchor identity sets differ: {details}")
    if not expected:
        raise AnchorPromotionError("anchor promotion set is empty")

    review_output: list[dict[str, Any]] = []
    source_output: list[dict[str, Any]] = []
    decision_output: list[dict[str, Any]] = []
    for rank, queue_id in enumerate(sorted(expected), start=1):
        contract = contract_by_id[queue_id]
        queue = queue_by_id[queue_id]
        source = source_by_id[queue_id]
        decision = decision_by_id[queue_id]
        title = _required(queue, "title", queue_id)
        authors = _required(queue, "authors", queue_id)
        year = _required(queue, "year", queue_id)
        if _normalize(title) != _normalize(_required(contract, "canonical_title", queue_id)):
            raise AnchorPromotionError(f"{queue_id} contract and queue titles differ")
        if authors != _required(contract, "canonical_authors", queue_id):
            raise AnchorPromotionError(f"{queue_id} contract and queue authors differ")
        if year != _required(contract, "year", queue_id):
            raise AnchorPromotionError(f"{queue_id} contract and queue years differ")
        if _normalize(title) != _normalize(_required(source, "title", queue_id)):
            raise AnchorPromotionError(f"{queue_id} source and queue titles differ")
        if _normalize(title) != _normalize(
            _required(decision, "canonical_title", queue_id)
        ):
            raise AnchorPromotionError(f"{queue_id} decision and queue titles differ")
        if decision.get("decision", "").strip() != "ELIGIBLE_BROAD":
            raise AnchorPromotionError(f"{queue_id} is not ELIGIBLE_BROAD")
        if _required(decision, "source_authority", queue_id) != _required(
            source, "source_authority", queue_id
        ):
            raise AnchorPromotionError(f"{queue_id} source authorities differ")

        review_output.append(
            {
                "queue_id": queue_id,
                "review_group_id": queue_id,
                "queue_status": "MANUAL_SCREEN_COMPLETE",
                "identity_status": "RESOLVED_PRIMARY",
                "title": title,
                "authors": authors,
                "year": year,
                "venue": "NOT_REPORTED_BY_SOURCE",
                "primary_url": _required(queue, "primary_url", queue_id),
                "full_text_url": _required(queue, "full_text_url", queue_id),
                "doi": "NOT_REPORTED_BY_SOURCE",
                "abstract": "NOT_ASSESSED_AT_BROAD_LEVEL",
                "rq_ids": _required(queue, "rq_ids", queue_id),
                "source_origins": "MANDATORY_ANCHOR_SUPPLEMENT_V2",
                "candidate_version_ids": queue_id,
                "candidate_version_count": 1,
                "discovery_decisions": "ELIGIBLE_BROAD",
                "grouping_evidence": "ANCHOR:" + _required(
                    contract, "anchor_id", queue_id
                ),
                "v2_counted_tier": (
                    "NOT_APPLICABLE_WITH_REASON:supplemental anchor not in prior BROAD"
                ),
                "blind_order_key": hashlib.sha256(
                    f"{queue_id}|anchor-promotion-v2".encode("utf-8")
                ).hexdigest().upper(),
                "blind_review_rank": rank,
            }
        )
        source_output.append(
            {
                field: source.get(field, "")
                for field in SOURCE_VALIDATION_FIELDS
            }
        )
        source_output[-1].update(
            {
                "source_superseded": "False",
                "superseded_sha256": (
                    "NOT_APPLICABLE_WITH_REASON:no source supersession"
                ),
                "supersession_reason": (
                    "NOT_APPLICABLE_WITH_REASON:no source supersession"
                ),
                "reading_credit_granted": "False",
            }
        )
        decision_output.append(
            {field: decision.get(field, "") for field in DECISION_FIELDS}
        )

    return AnchorPromotionRows(
        review_input_rows=tuple(review_output),
        source_validation_rows=tuple(source_output),
        decision_rows=tuple(decision_output),
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _verify_source_files(corpus_root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    root = corpus_root.resolve()
    for row in rows:
        queue_id = str(row["paper_id"])
        relative = Path(str(row["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise AnchorPromotionError(f"{queue_id} source path escapes corpus root")
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise AnchorPromotionError(
                f"{queue_id} source path escapes corpus root"
            ) from exc
        if not source.is_file():
            raise AnchorPromotionError(f"{queue_id} source file missing: {source}")
        if source.stat().st_size != int(str(row["bytes"])):
            raise AnchorPromotionError(f"{queue_id} source byte mismatch")
        if _sha256(source) != str(row["sha256"]).upper():
            raise AnchorPromotionError(f"{queue_id} source SHA mismatch")
        receipt = (root / Path(str(row["receipt_path"]))).resolve()
        if not receipt.is_file():
            raise AnchorPromotionError(f"{queue_id} source receipt missing")


def promote_anchor_batch(
    corpus_root: str | Path,
    *,
    output_discovery_root: str | Path | None = None,
    batch_number: int = 24,
) -> AnchorPromotionResult:
    """Publish a new standard manual batch only after all inputs cross-check."""

    if batch_number < 1:
        raise ValueError("batch_number must be positive")
    root = Path(corpus_root).resolve()
    discovery = root / "discovery"
    output = (
        Path(output_discovery_root).resolve()
        if output_discovery_root is not None
        else discovery
    )
    contract_path = discovery / "CORE_METHOD_ANCHORS_v2.csv"
    broad_membership_path = root / "staging" / "broad_freeze_v2" / "BROAD_500.csv"
    queue_path = discovery / "ANCHOR_REVIEW_QUEUE_v2.csv"
    source_path = discovery / "ANCHOR_SOURCE_INVENTORY_v2.csv"
    decision_path = (
        discovery
        / "manual_screen_anchor_validation_v2"
        / "ANCHOR_MANUAL_SCREENING_VALIDATED_v2.csv"
    )
    validate_anchor_contract(
        contract_path,
        expected_count=40,
        broad_membership_path=broad_membership_path,
    )
    contract_rows = [
        row
        for row in _read_csv(contract_path)
        if row["current_status"] != "BROAD_V2_ELIGIBLE"
    ]
    rows = build_anchor_promotion_rows(
        contract_rows=contract_rows,
        queue_rows=_read_csv(queue_path),
        source_rows=_read_csv(source_path),
        decision_rows=_read_csv(decision_path),
    )
    _verify_source_files(root, rows.source_validation_rows)

    batch_root = output / "manual_screen_batches_v2"
    decisions_root = output / "manual_screen_decisions_v2"
    batch_root.mkdir(parents=True, exist_ok=True)
    decisions_root.mkdir(parents=True, exist_ok=True)
    review_target = batch_root / f"review_input_{batch_number:03d}.csv"
    source_target = batch_root / f"source_validation_{batch_number:03d}.csv"
    decision_target = decisions_root / f"batch_{batch_number:03d}.csv"
    receipt_target = batch_root / f"anchor_promotion_{batch_number:03d}.json"
    targets = (review_target, source_target, decision_target, receipt_target)
    existing = [path for path in targets if path.exists()]
    if existing:
        raise AnchorPromotionError(f"promotion target already exists: {existing[0]}")

    temporary = tuple(path.with_suffix(path.suffix + ".tmp") for path in targets)
    published: list[Path] = []
    try:
        _write_csv(temporary[0], rows.review_input_rows, REVIEW_INPUT_FIELDS)
        _write_csv(
            temporary[1], rows.source_validation_rows, SOURCE_VALIDATION_FIELDS
        )
        _write_csv(temporary[2], rows.decision_rows, DECISION_FIELDS)
        receipt = {
            "schema_version": "2.0",
            "status": "PASS",
            "created_at": datetime.now().astimezone().isoformat(),
            "batch_number": batch_number,
            "output_count": len(rows.review_input_rows),
            "input_sha256": {
                path.relative_to(root).as_posix(): _sha256(path)
                for path in (
                    contract_path,
                    broad_membership_path,
                    queue_path,
                    source_path,
                    decision_path,
                )
            },
            "output_sha256": {
                review_target.name: _sha256(temporary[0]),
                source_target.name: _sha256(temporary[1]),
                decision_target.name: _sha256(temporary[2]),
            },
            "reading_credit_granted": False,
            "formal_broad_increment": 0,
            "formal_screened_increment": 0,
            "formal_deep_increment": 0,
        }
        temporary[3].write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for temp_path, target in zip(temporary, targets, strict=True):
            temp_path.replace(target)
            published.append(target)
    except Exception:
        for path in temporary:
            path.unlink(missing_ok=True)
        for path in published:
            path.unlink(missing_ok=True)
        raise

    return AnchorPromotionResult(
        status="PASS",
        promoted_count=len(rows.review_input_rows),
        review_input_path=review_target,
        source_validation_path=source_target,
        decision_path=decision_target,
        receipt_path=receipt_target,
    )
