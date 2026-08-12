"""Freeze a non-credit DEEP review queue from the canonical SCREENED pool."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import unicodedata
from typing import Any, Mapping, Sequence

from .literature_tier_freeze_v2 import (
    BroadCandidate,
    TierSelectionPolicy,
    select_broad_candidates,
)


class DeepReviewQueueError(RuntimeError):
    """Raised when the provisional DEEP queue violates its frozen contract."""


@dataclass(frozen=True)
class DeepReviewQueueResult:
    output_root: Path
    selected_count: int
    reserve_count: int
    mandatory_count: int
    current_review_count: int
    byte_identical_legacy_count: int
    ready_union_count: int


QUEUE_FIELDS = (
    "deep_rank",
    "selection_role",
    "paper_id",
    "queue_id",
    "canonical_work_id",
    "title",
    "quota_rq",
    "secondary_rqs",
    "directness",
    "relevance_class",
    "effect_relation",
    "selection_phase",
    "tie_break_key",
    "method_source_path",
    "method_source_sha256",
    "method_source_bytes",
    "current_screened_review",
    "legacy_deep_note",
    "ready_evidence_status",
    "formal_deep_credit",
)

LEGACY_FIELDS = (
    "legacy_paper_id",
    "paper_id",
    "canonical_work_id",
    "title",
    "current_method_source_path",
    "pdf_sha256",
    "legacy_note_path",
    "legacy_note_bytes",
    "legacy_note_sha256",
    "evidence_provenance_class",
    "independent_content_rereview_performed",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _read_csv(path: Path, required: Sequence[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise DeepReviewQueueError(f"required CSV missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(required) - set(reader.fieldnames or ()))
        if missing:
            raise DeepReviewQueueError(f"{path.name} missing fields: {missing}")
        return [dict(row) for row in reader]


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _normalized_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    normalized = normalized.replace("\\n", " ").replace("&", " and ")
    tokens = re.findall(r"[a-z0-9]+", normalized)
    aliases = {"maximally": "maximal"}
    return "".join(aliases.get(token, token) for token in tokens)


def _safe_source(
    root: Path,
    row: Mapping[str, str],
    *,
    staging_root: Path,
) -> Path:
    relative = Path(row["method_source_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise DeepReviewQueueError(
            f"{row['paper_id']} method source path escapes corpus root"
        )
    candidates = ((root / relative).resolve(), (staging_root / relative).resolve())
    for candidate in candidates:
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise DeepReviewQueueError(
                f"{row['paper_id']} method source path escapes corpus root"
            ) from exc
    source = next((candidate for candidate in candidates if candidate.is_file()), None)
    if source is None or not source.read_bytes()[:5].startswith(b"%PDF-"):
        raise DeepReviewQueueError(f"{row['paper_id']} method source is not a PDF")
    try:
        expected_bytes = int(row["method_source_bytes"])
    except ValueError as exc:
        raise DeepReviewQueueError(
            f"{row['paper_id']} method source bytes are invalid"
        ) from exc
    if source.stat().st_size != expected_bytes:
        raise DeepReviewQueueError(f"{row['paper_id']} method source byte mismatch")
    if _sha256(source) != row["method_source_sha256"].strip().upper():
        raise DeepReviewQueueError(f"{row['paper_id']} method source SHA mismatch")
    return source


def _direct_rqs(row: Mapping[str, str]) -> tuple[str, ...]:
    values = [row["quota_rq"]]
    secondary = row["secondary_rqs"]
    if not secondary.startswith("NOT_APPLICABLE_WITH_REASON:"):
        values.extend(value for value in secondary.split(";") if value)
    return tuple(sorted(set(values)))


def _current_reviews(
    directory: Path,
    screened_by_work: Mapping[str, Mapping[str, str]],
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    if not directory.is_dir():
        raise DeepReviewQueueError(f"screened review directory missing: {directory}")
    work_ids: set[str] = set()
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("P[0-9][0-9][0-9][0-9].json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DeepReviewQueueError(f"invalid screened review: {path.name}") from exc
        work_id = str(record.get("canonical_work_id", "")).strip()
        row = screened_by_work.get(work_id)
        if row is None:
            raise DeepReviewQueueError(
                f"{path.name} canonical work is outside SCREENED PRIMARY"
            )
        if work_id in work_ids:
            raise DeepReviewQueueError(f"duplicate current review work: {work_id}")
        if _normalized_title(str(record.get("title", ""))) != _normalized_title(
            row["title"]
        ):
            raise DeepReviewQueueError(f"{path.name} title mismatch")
        source = record.get("method_source")
        if not isinstance(source, Mapping):
            raise DeepReviewQueueError(f"{path.name} method source missing")
        try:
            review_bytes = int(source.get("bytes", -1))
        except (TypeError, ValueError) as exc:
            raise DeepReviewQueueError(f"{path.name} method source bytes invalid") from exc
        if (
            str(source.get("sha256", "")).upper()
            != row["method_source_sha256"].upper()
            or review_bytes != int(row["method_source_bytes"])
        ):
            raise DeepReviewQueueError(f"{path.name} method source identity changed")
        work_ids.add(work_id)
        records[work_id] = {
            "path": path,
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    return work_ids, records


def _legacy_note_path(
    raw: str,
    *,
    ledger_path: Path,
    legacy_note_dir: Path | None,
) -> Path:
    relative = Path(raw)
    if legacy_note_dir is not None:
        return (legacy_note_dir / relative.name).resolve()
    return (ledger_path.parent / relative).resolve()


def _legacy_rebindings(
    rows: Sequence[Mapping[str, str]],
    *,
    ledger_path: Path,
    legacy_note_dir: Path | None,
    screened_by_title: Mapping[str, Mapping[str, str]],
) -> tuple[set[str], list[dict[str, Any]]]:
    work_ids: set[str] = set()
    mappings: list[dict[str, Any]] = []
    for legacy in rows:
        current = screened_by_title.get(_normalized_title(legacy["title"]))
        if current is None:
            continue
        if legacy["pdf_sha256"].strip().upper() != current[
            "method_source_sha256"
        ].strip().upper():
            continue
        note = _legacy_note_path(
            legacy["note_path"],
            ledger_path=ledger_path,
            legacy_note_dir=legacy_note_dir,
        )
        if not note.is_file():
            continue
        work_id = current["canonical_work_id"]
        if work_id in work_ids:
            raise DeepReviewQueueError(f"duplicate legacy rebind work: {work_id}")
        work_ids.add(work_id)
        mappings.append(
            {
                "legacy_paper_id": legacy["paper_id"],
                "paper_id": current["paper_id"],
                "canonical_work_id": work_id,
                "title": current["title"],
                "current_method_source_path": current["method_source_path"],
                "pdf_sha256": current["method_source_sha256"].upper(),
                "legacy_note_path": note.as_posix(),
                "legacy_note_bytes": note.stat().st_size,
                "legacy_note_sha256": _sha256(note),
                "evidence_provenance_class": "USER_ACCEPTED_INHERITED_EVIDENCE",
                "independent_content_rereview_performed": "false",
            }
        )
    return work_ids, mappings


def _queue_row(
    *,
    rank: int,
    role: str,
    selected: Any,
    screened: Mapping[str, str],
    current_review_ids: set[str],
    legacy_ids: set[str],
) -> dict[str, Any]:
    work_id = selected.candidate.canonical_work_id
    current = work_id in current_review_ids
    legacy = work_id in legacy_ids
    return {
        "deep_rank": rank,
        "selection_role": role,
        "paper_id": screened["paper_id"],
        "queue_id": screened["queue_id"],
        "canonical_work_id": work_id,
        "title": screened["title"],
        "quota_rq": selected.quota_rq,
        "secondary_rqs": ";".join(
            rq for rq in selected.candidate.direct_rqs if rq != selected.quota_rq
        )
        or "NOT_APPLICABLE_WITH_REASON:no secondary RQ",
        "directness": selected.directness,
        "relevance_class": screened["relevance_class"],
        "effect_relation": screened["effect_relation"],
        "selection_phase": selected.selection_phase,
        "tie_break_key": selected.tie_break_key,
        "method_source_path": screened["method_source_path"],
        "method_source_sha256": screened["method_source_sha256"].upper(),
        "method_source_bytes": screened["method_source_bytes"],
        "current_screened_review": "HASH_BOUND_PRESENT" if current else "PENDING",
        "legacy_deep_note": "BYTE_IDENTICAL_PRESENT" if legacy else "NOT_AVAILABLE",
        "ready_evidence_status": (
            "INHERITED_EVIDENCE_PRESENT" if current or legacy else "FULL_REVIEW_PENDING"
        ),
        "formal_deep_credit": "false",
    }


def _input_artifact(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise DeepReviewQueueError(f"input artifact missing: {path}")
    return {
        "input_role": role,
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build_deep_review_queue(
    corpus_root: str | Path,
    *,
    canonical_registry_path: str | Path,
    screened_primary_path: str | Path,
    core_anchors_path: str | Path,
    screened_review_dir: str | Path,
    legacy_fulltext_ledger_path: str | Path,
    output_relative: str | Path,
    policy: TierSelectionPolicy,
    legacy_note_dir: str | Path | None = None,
    replace_existing: bool = False,
) -> DeepReviewQueueResult:
    """Select exactly ``policy.total`` DEEP review candidates without tier credit."""

    root = Path(corpus_root).resolve()
    canonical_path = Path(canonical_registry_path).resolve()
    screened_path = Path(screened_primary_path).resolve()
    anchors_path = Path(core_anchors_path).resolve()
    reviews_dir = Path(screened_review_dir).resolve()
    legacy_path = Path(legacy_fulltext_ledger_path).resolve()
    note_dir = Path(legacy_note_dir).resolve() if legacy_note_dir is not None else None
    if policy.tier_label != "DEEP":
        raise DeepReviewQueueError("deep review queue policy must use tier_label=DEEP")
    output_fragment = Path(output_relative)
    if output_fragment.is_absolute() or ".." in output_fragment.parts:
        raise DeepReviewQueueError("output_relative must stay inside the corpus root")
    output = (root / output_fragment).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise DeepReviewQueueError("output path escapes corpus root") from exc
    if output.exists() and not replace_existing:
        raise DeepReviewQueueError(f"deep queue output already exists: {output}")

    canonical_rows = _read_csv(
        canonical_path,
        ("paper_id", "canonical_work_id", "title", "authors", "year", "doi"),
    )
    screened_rows = _read_csv(
        screened_path,
        (
            "paper_id",
            "queue_id",
            "canonical_work_id",
            "title",
            "quota_rq",
            "secondary_rqs",
            "relevance_class",
            "effect_relation",
            "method_source_path",
            "method_source_sha256",
            "method_source_bytes",
        ),
    )
    anchor_rows = _read_csv(anchors_path, ("anchor_id", "canonical_work_id"))
    legacy_rows = _read_csv(
        legacy_path,
        ("paper_id", "title", "pdf_sha256", "note_path", "reading_status"),
    )
    if len(screened_rows) < policy.total:
        raise DeepReviewQueueError(
            f"SCREENED pool {len(screened_rows)} is smaller than DEEP total {policy.total}"
        )

    canonical_by_work: dict[str, dict[str, str]] = {}
    for row in canonical_rows:
        work_id = row["canonical_work_id"]
        if not work_id or work_id in canonical_by_work:
            raise DeepReviewQueueError(f"blank or duplicate canonical work: {work_id!r}")
        canonical_by_work[work_id] = row
    screened_by_work: dict[str, dict[str, str]] = {}
    screened_by_title: dict[str, dict[str, str]] = {}
    candidates: list[BroadCandidate] = []
    for row in screened_rows:
        work_id = row["canonical_work_id"]
        canonical = canonical_by_work.get(work_id)
        if canonical is None:
            raise DeepReviewQueueError(f"SCREENED work is outside canonical registry: {work_id}")
        if work_id in screened_by_work:
            raise DeepReviewQueueError(f"duplicate SCREENED canonical work: {work_id}")
        title_key = _normalized_title(row["title"])
        if not title_key or title_key in screened_by_title:
            raise DeepReviewQueueError(f"blank or duplicate SCREENED title: {row['title']!r}")
        if title_key != _normalized_title(canonical["title"]):
            raise DeepReviewQueueError(f"{work_id} canonical title mismatch")
        source = _safe_source(root, row, staging_root=canonical_path.parent)
        normalized_row = dict(row)
        normalized_row["method_source_path"] = source.relative_to(root).as_posix()
        screened_by_work[work_id] = normalized_row
        screened_by_title[title_key] = normalized_row
        try:
            year = int(canonical["year"])
        except ValueError as exc:
            raise DeepReviewQueueError(f"{work_id} canonical year invalid") from exc
        candidates.append(
            BroadCandidate(
                queue_id=normalized_row["queue_id"],
                canonical_work_id=work_id,
                title=normalized_row["title"],
                authors=canonical["authors"],
                year=year,
                direct_rqs=_direct_rqs(normalized_row),
                relevance_class=normalized_row["relevance_class"],
                doi=canonical["doi"],
                effect_relation=normalized_row["effect_relation"],
            )
        )

    current_review_ids, current_review_records = _current_reviews(
        reviews_dir, screened_by_work
    )
    legacy_ids, legacy_mappings = _legacy_rebindings(
        legacy_rows,
        ledger_path=legacy_path,
        legacy_note_dir=note_dir,
        screened_by_title=screened_by_title,
    )
    anchor_ids = {row["canonical_work_id"].strip() for row in anchor_rows}
    missing_anchors = sorted(anchor_ids - set(screened_by_work))
    if missing_anchors:
        raise DeepReviewQueueError(
            "core anchor is absent from SCREENED PRIMARY: " + ",".join(missing_anchors)
        )
    mandatory_ids = set(policy.mandatory_canonical_work_ids)
    mandatory_ids.update(anchor_ids)
    mandatory_ids.update(current_review_ids)
    mandatory_ids.update(legacy_ids)
    frozen_policy = replace(
        policy,
        mandatory_canonical_work_ids=tuple(sorted(mandatory_ids)),
    )
    selection = select_broad_candidates(candidates, frozen_policy)

    selected_rows = [
        _queue_row(
            rank=index,
            role="PRIMARY",
            selected=selected,
            screened=screened_by_work[selected.candidate.canonical_work_id],
            current_review_ids=current_review_ids,
            legacy_ids=legacy_ids,
        )
        for index, selected in enumerate(selection.selected, start=1)
    ]
    reserve_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(selection.reserves, start=1):
        screened = screened_by_work[candidate.canonical_work_id]
        reserve_rows.append(
            {
                "deep_rank": index,
                "selection_role": "RESERVE",
                "paper_id": screened["paper_id"],
                "queue_id": screened["queue_id"],
                "canonical_work_id": candidate.canonical_work_id,
                "title": screened["title"],
                "quota_rq": screened["quota_rq"],
                "secondary_rqs": screened["secondary_rqs"],
                "directness": candidate.directness,
                "relevance_class": screened["relevance_class"],
                "effect_relation": screened["effect_relation"],
                "selection_phase": "RESERVE",
                "tie_break_key": hashlib.sha256(
                    f"{candidate.canonical_work_id}|DEEP|{policy.frozen_seed}".encode()
                ).hexdigest().upper(),
                "method_source_path": screened["method_source_path"],
                "method_source_sha256": screened["method_source_sha256"].upper(),
                "method_source_bytes": screened["method_source_bytes"],
                "current_screened_review": (
                    "HASH_BOUND_PRESENT"
                    if candidate.canonical_work_id in current_review_ids
                    else "PENDING"
                ),
                "legacy_deep_note": (
                    "BYTE_IDENTICAL_PRESENT"
                    if candidate.canonical_work_id in legacy_ids
                    else "NOT_AVAILABLE"
                ),
                "ready_evidence_status": (
                    "INHERITED_EVIDENCE_PRESENT"
                    if candidate.canonical_work_id in current_review_ids | legacy_ids
                    else "FULL_REVIEW_PENDING"
                ),
                "formal_deep_credit": "false",
            }
        )

    temp = output.parent / f".{output.name}.tmp"
    if temp.exists():
        raise DeepReviewQueueError(f"stale deep queue temp exists: {temp}")
    temp.mkdir(parents=True)
    try:
        _write_csv(temp / "DEEP_PRIMARY_100.csv", selected_rows, QUEUE_FIELDS)
        _write_csv(temp / "DEEP_RESERVES_200.csv", reserve_rows, QUEUE_FIELDS)
        _write_csv(temp / "LEGACY_DEEP_REBINDING.csv", legacy_mappings, LEGACY_FIELDS)
        input_artifacts = [
            _input_artifact(canonical_path, "CANONICAL_REGISTRY"),
            _input_artifact(screened_path, "SCREENED_PRIMARY"),
            _input_artifact(anchors_path, "CORE_ANCHORS"),
            _input_artifact(legacy_path, "LEGACY_FULLTEXT_LEDGER"),
        ]
        input_artifacts.extend(
            {
                "input_role": "CURRENT_SCREENED_REVIEW",
                "path": value["path"].as_posix(),
                "bytes": value["bytes"],
                "sha256": value["sha256"],
            }
            for value in current_review_records.values()
        )
        receipt = {
            "schema_version": "4.0",
            "status": "PASS",
            "created_at": datetime.now().astimezone().isoformat(),
            "selected_count": len(selected_rows),
            "reserve_count": len(reserve_rows),
            "mandatory_count": len(mandatory_ids),
            "core_anchor_count": len(anchor_ids),
            "current_review_count": len(current_review_ids),
            "byte_identical_legacy_count": len(legacy_ids),
            "ready_union_count": len(current_review_ids | legacy_ids),
            "quota_counts": dict(selection.quota_counts),
            "selection_policy": {
                "total": frozen_policy.total,
                "minimum_per_rq": frozen_policy.minimum_per_rq,
                "maximum_per_rq": frozen_policy.maximum_per_rq,
                "maximum_transfer": frozen_policy.maximum_transfer,
                "minimum_counterevidence_per_rq": frozen_policy.minimum_counterevidence_per_rq,
                "mandatory_canonical_work_ids": list(
                    frozen_policy.mandatory_canonical_work_ids
                ),
                "tier_label": frozen_policy.tier_label,
                "frozen_seed": frozen_policy.frozen_seed,
            },
            "input_artifacts": input_artifacts,
            "formal_deep_increment": 0,
            "formal_training_started_by_this_builder": False,
            "engineering_gate_generated_by_this_builder": False,
            "blind_holdout_opened_by_this_builder": False,
            "global_runtime_state_assessed": False,
        }
        (temp / "DEEP_REVIEW_QUEUE_RECEIPT.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            backup = output.parent / f".{output.name}.previous"
            if backup.exists():
                raise DeepReviewQueueError(f"stale deep queue backup exists: {backup}")
            output.rename(backup)
            try:
                os.replace(temp, output)
            except Exception:
                os.replace(backup, output)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(temp, output)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise

    return DeepReviewQueueResult(
        output_root=output,
        selected_count=len(selected_rows),
        reserve_count=len(reserve_rows),
        mandatory_count=len(mandatory_ids),
        current_review_count=len(current_review_ids),
        byte_identical_legacy_count=len(legacy_ids),
        ready_union_count=len(current_review_ids | legacy_ids),
    )
