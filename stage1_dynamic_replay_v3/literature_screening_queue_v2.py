"""Build a non-credit SCREENED reading queue from validated BROAD staging."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

from .literature_tier_freeze_v2 import (
    BroadCandidate,
    TierSelectionError,
    TierSelectionPolicy,
    select_broad_candidates,
)


class ScreeningQueueError(RuntimeError):
    """Raised when a provisional screening queue violates its frozen contract."""


@dataclass(frozen=True)
class ScreeningQueueResult:
    status: str
    output_root: Path
    primary_count: int
    reserve_count: int
    reading_queue_count: int
    quota_counts: Mapping[str, int]
    eligible_count: int
    required_source_format: str | None
    formal_screened_increment: int = 0


QUEUE_FIELDS = (
    "reading_rank",
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
    "broad_source_path",
    "broad_source_format",
    "broad_source_sha256",
    "broad_source_bytes",
    "method_source_path",
    "method_source_format",
    "method_source_sha256",
    "method_source_bytes",
    "method_source_origin",
    "method_source_authority",
    "method_source_url",
    "method_source_receipt_path",
    "method_source_status",
    "screened_credit",
)


def _read_csv(path: Path, required: Sequence[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ScreeningQueueError(f"required CSV missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(set(required) - fields)
        if missing:
            raise ScreeningQueueError(f"{path.name} missing fields: {missing}")
        return [dict(row) for row in reader]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ScreeningQueueError(f"refusing to write an empty queue: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(QUEUE_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _input_artifact(
    path: Path,
    *,
    role: str,
    corpus_root: Path,
) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ScreeningQueueError(f"screening input missing: {resolved}")
    try:
        relative = resolved.relative_to(corpus_root.resolve()).as_posix()
    except ValueError as exc:
        raise ScreeningQueueError(
            f"screening input must stay inside corpus root: {resolved}"
        ) from exc
    return {
        "input_role": role,
        "path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _effect_relation(note_relation: str) -> str:
    return {
        "SUPPORTED": "SUPPORTED",
        "REFUTED": "NULL_NEGATIVE",
        "MIXED": "MIXED",
    }[note_relation]


def _direct_rqs(row: Mapping[str, str]) -> tuple[str, ...]:
    values = [row["quota_rq"]]
    secondary = row["secondary_rqs"]
    if not secondary.startswith("NOT_APPLICABLE_WITH_REASON:"):
        values.extend(part for part in secondary.split(";") if part)
    return tuple(sorted(set(values)))


def _source_format(path: str) -> str:
    return "PDF" if Path(path).suffix.casefold() == ".pdf" else "HTML"


def _normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _safe_corpus_path(corpus_root: Path, raw: str, work_id: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ScreeningQueueError(f"{work_id} method source path escapes corpus root")
    resolved = (corpus_root / relative).resolve()
    try:
        resolved.relative_to(corpus_root.resolve())
    except ValueError as exc:
        raise ScreeningQueueError(
            f"{work_id} method source path escapes corpus root"
        ) from exc
    return resolved


def _load_method_source_overrides(
    path: Path | None,
    *,
    corpus_root: Path,
    registry_by_work: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    rows = _read_csv(
        path,
        (
            "canonical_work_id",
            "title",
            "path",
            "bytes",
            "sha256",
            "source_authority",
            "source_url",
            "receipt_path",
            "override_reason",
            "reading_credit_granted",
        ),
    )
    overrides: dict[str, dict[str, str]] = {}
    for row in rows:
        work_id = row["canonical_work_id"].strip()
        if work_id in overrides:
            raise ScreeningQueueError(f"duplicate method source override: {work_id}")
        registry = registry_by_work.get(work_id)
        if registry is None:
            raise ScreeningQueueError(
                f"method source override is outside BROAD registry: {work_id}"
            )
        if _normalize_title(row["title"]) != _normalize_title(registry["title"]):
            raise ScreeningQueueError(f"{work_id} method source title mismatch")
        source = _safe_corpus_path(corpus_root, row["path"], work_id)
        if not source.is_file():
            raise ScreeningQueueError(f"{work_id} method source missing: {source}")
        try:
            expected_bytes = int(row["bytes"])
        except ValueError as exc:
            raise ScreeningQueueError(f"{work_id} invalid method source bytes") from exc
        if source.stat().st_size != expected_bytes:
            raise ScreeningQueueError(f"{work_id} method source byte mismatch")
        observed_sha = _sha256(source)
        if observed_sha != row["sha256"].strip().upper():
            raise ScreeningQueueError(f"{work_id} method source SHA mismatch")
        if _source_format(row["path"]) != "PDF" or not source.read_bytes().startswith(
            b"%PDF-"
        ):
            raise ScreeningQueueError(f"{work_id} method source override is not a PDF")
        receipt = _safe_corpus_path(corpus_root, row["receipt_path"], work_id)
        if not receipt.is_file():
            raise ScreeningQueueError(f"{work_id} method source receipt missing")
        try:
            receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ScreeningQueueError(
                f"{work_id} method source receipt is invalid JSON"
            ) from exc
        receipt_sha = str(receipt_payload.get("sha256", "")).upper()
        ledger = receipt_payload.get("ledger_row")
        ledger_sha = (
            str(ledger.get("sha256", "")).upper()
            if isinstance(ledger, Mapping)
            else ""
        )
        if observed_sha not in {receipt_sha, ledger_sha}:
            raise ScreeningQueueError(f"{work_id} method source receipt SHA mismatch")
        if row["reading_credit_granted"].strip().casefold() != "false":
            raise ScreeningQueueError(
                f"{work_id} method source override cannot grant reading credit"
            )
        overrides[work_id] = dict(row)
    return overrides


def _method_source(
    registry: Mapping[str, str],
    override: Mapping[str, str] | None,
) -> dict[str, str]:
    if override is None:
        return {
            "path": registry["source_path"],
            "format": _source_format(registry["source_path"]),
            "sha256": registry["source_sha256"],
            "bytes": registry["source_bytes"],
            "origin": "BROAD_SOURCE",
            "authority": "NOT_APPLICABLE_WITH_REASON:retained in BROAD acquisition ledger",
            "url": "NOT_APPLICABLE_WITH_REASON:retained in BROAD acquisition ledger",
            "receipt_path": "NOT_APPLICABLE_WITH_REASON:retained in BROAD acquisition ledger",
        }
    return {
        "path": override["path"],
        "format": _source_format(override["path"]),
        "sha256": override["sha256"].upper(),
        "bytes": override["bytes"],
        "origin": "VERIFIED_OVERRIDE",
        "authority": override["source_authority"],
        "url": override["source_url"],
        "receipt_path": override["receipt_path"],
    }


def _queue_row(
    *,
    rank: int,
    role: str,
    selected: Any,
    paper_id: str,
    registry: Mapping[str, str],
    method_source: Mapping[str, str],
) -> dict[str, Any]:
    candidate = selected.candidate
    source_format = _source_format(registry["source_path"])
    return {
        "reading_rank": rank,
        "selection_role": role,
        "paper_id": paper_id,
        "queue_id": candidate.queue_id,
        "canonical_work_id": candidate.canonical_work_id,
        "title": candidate.title,
        "quota_rq": selected.quota_rq,
        "secondary_rqs": ";".join(
            rq for rq in candidate.direct_rqs if rq != selected.quota_rq
        )
        or "NOT_APPLICABLE_WITH_REASON:no secondary RQ",
        "directness": selected.directness,
        "relevance_class": candidate.relevance_class,
        "effect_relation": candidate.effect_relation,
        "selection_phase": selected.selection_phase,
        "tie_break_key": selected.tie_break_key,
        "broad_source_path": registry["source_path"],
        "broad_source_format": source_format,
        "broad_source_sha256": registry["source_sha256"],
        "broad_source_bytes": registry["source_bytes"],
        "method_source_path": method_source["path"],
        "method_source_format": method_source["format"],
        "method_source_sha256": method_source["sha256"],
        "method_source_bytes": method_source["bytes"],
        "method_source_origin": method_source["origin"],
        "method_source_authority": method_source["authority"],
        "method_source_url": method_source["url"],
        "method_source_receipt_path": method_source["receipt_path"],
        "method_source_status": (
            "REUSE_VERIFIED_METHOD_PDF"
            if method_source["format"] == "PDF"
            else "FULL_TEXT_DISCOVERY_REQUIRED"
        ),
        "screened_credit": "NOT_ASSESSED_AT_BROAD_LEVEL",
    }


def build_screening_queue(
    broad_staging_root: str | Path,
    *,
    output_root: str | Path,
    policy: TierSelectionPolicy,
    reserve_read_count: int = 60,
    required_source_format: str | None = None,
    policy_source_paths: Sequence[Path] = (),
    method_source_override_path: Path | None = None,
    replace_existing: bool = False,
) -> ScreeningQueueResult:
    """Select a provisional reading queue; this function grants no tier credit."""

    broad_root = Path(broad_staging_root).resolve()
    output = Path(output_root).resolve()
    corpus_root = broad_root.parent.parent
    if policy.tier_label != "SCREENED":
        raise ScreeningQueueError("screening queue policy must use tier_label=SCREENED")
    if reserve_read_count < 0:
        raise ScreeningQueueError("reserve_read_count must be non-negative")
    source_gate = required_source_format.upper() if required_source_format else None
    if source_gate not in {None, "PDF", "HTML"}:
        raise ScreeningQueueError("required_source_format must be PDF, HTML, or omitted")
    if output.exists() and not replace_existing:
        raise ScreeningQueueError(f"screening queue output already exists: {output}")

    input_artifacts = [
        _input_artifact(
            broad_root / name,
            role="BROAD_STAGING_INPUT",
            corpus_root=corpus_root,
        )
        for name in ("BROAD_500.csv", "CANONICAL_WORKS.csv", "FREEZE_RECEIPT.json")
    ]
    input_artifacts.extend(
        _input_artifact(
            Path(path),
            role="SELECTION_POLICY_SOURCE",
            corpus_root=corpus_root,
        )
        for path in policy_source_paths
    )
    if method_source_override_path is not None:
        input_artifacts.append(
            _input_artifact(
                Path(method_source_override_path),
                role="METHOD_SOURCE_OVERRIDE",
                corpus_root=corpus_root,
            )
        )

    registry_rows = _read_csv(
        broad_root / "CANONICAL_WORKS.csv",
        (
            "paper_id",
            "canonical_work_id",
            "title",
            "authors",
            "year",
            "doi",
            "source_path",
            "source_sha256",
            "source_bytes",
        ),
    )
    broad_rows = _read_csv(
        broad_root / "BROAD_500.csv",
        (
            "paper_id",
            "queue_id",
            "canonical_work_id",
            "quota_rq",
            "secondary_rqs",
            "relevance_class",
            "relation",
        ),
    )
    registry_by_id = {row["paper_id"]: row for row in registry_rows}
    if len(registry_by_id) != len(registry_rows):
        raise ScreeningQueueError("broad staging registry has duplicate paper IDs")
    if {row["paper_id"] for row in broad_rows} != set(registry_by_id):
        raise ScreeningQueueError("BROAD_500 and canonical registry paper IDs disagree")
    registry_by_work = {
        row["canonical_work_id"]: row for row in registry_rows
    }
    if len(registry_by_work) != len(registry_rows):
        raise ScreeningQueueError("broad staging registry has duplicate canonical work IDs")
    method_overrides = _load_method_source_overrides(
        Path(method_source_override_path) if method_source_override_path else None,
        corpus_root=corpus_root,
        registry_by_work=registry_by_work,
    )

    candidates: list[BroadCandidate] = []
    paper_by_work: dict[str, str] = {}
    method_by_work: dict[str, dict[str, str]] = {}
    for row in broad_rows:
        registry = registry_by_id[row["paper_id"]]
        if registry["canonical_work_id"] != row["canonical_work_id"]:
            raise ScreeningQueueError(f"{row['paper_id']} canonical work identity mismatch")
        method_source = _method_source(
            registry,
            method_overrides.get(row["canonical_work_id"]),
        )
        if source_gate and method_source["format"] != source_gate:
            continue
        candidate = BroadCandidate(
            queue_id=row["queue_id"],
            canonical_work_id=row["canonical_work_id"],
            title=registry["title"],
            authors=registry["authors"],
            year=int(registry["year"]),
            direct_rqs=_direct_rqs(row),
            relevance_class=row["relevance_class"],
            doi=registry["doi"],
            effect_relation=_effect_relation(row["relation"]),
        )
        candidates.append(candidate)
        paper_by_work[candidate.canonical_work_id] = row["paper_id"]
        method_by_work[candidate.canonical_work_id] = method_source

    if len(candidates) < policy.total:
        raise ScreeningQueueError(
            f"{source_gate or 'all-source'} eligible candidates {len(candidates)} "
            f"are fewer than required {policy.total}"
        )
    try:
        selection = select_broad_candidates(candidates, policy)
    except TierSelectionError as exc:
        raise ScreeningQueueError(f"SCREENED source-gated selection failed: {exc}") from exc
    if reserve_read_count > len(selection.reserves):
        raise ScreeningQueueError(
            f"requested {reserve_read_count} reserves but only {len(selection.reserves)} exist"
        )

    temp = output.parent / f".{output.name}.tmp"
    if temp.exists():
        raise ScreeningQueueError(f"stale queue temp exists: {temp}")
    temp.mkdir(parents=True)
    try:
        primary_rows = [
            _queue_row(
                rank=index,
                role="PRIMARY",
                selected=item,
                paper_id=paper_by_work[item.candidate.canonical_work_id],
                registry=registry_by_id[paper_by_work[item.candidate.canonical_work_id]],
                method_source=method_by_work[item.candidate.canonical_work_id],
            )
            for index, item in enumerate(selection.selected, start=1)
        ]
        reserve_selected = []
        for candidate in selection.reserves[:reserve_read_count]:
            # A reserve has no quota assignment until it replaces a failed primary.
            reserve_selected.append(
                type(selection.selected[0])(
                    candidate=candidate,
                    quota_rq="NOT_APPLICABLE_WITH_REASON:assigned only if reserve replaces a primary",
                    directness=candidate.directness,
                    tie_break_key="RESERVE_" + candidate.canonical_work_id,
                    selection_phase="PROVISIONAL_RESERVE",
                )
            )
        reserve_rows = [
            _queue_row(
                rank=len(primary_rows) + index,
                role="RESERVE",
                selected=item,
                paper_id=paper_by_work[item.candidate.canonical_work_id],
                registry=registry_by_id[paper_by_work[item.candidate.canonical_work_id]],
                method_source=method_by_work[item.candidate.canonical_work_id],
            )
            for index, item in enumerate(reserve_selected, start=1)
        ]
        all_reserve_rows = []
        for index, candidate in enumerate(selection.reserves, start=1):
            paper_id = paper_by_work[candidate.canonical_work_id]
            registry = registry_by_id[paper_id]
            all_reserve_rows.append(
                {
                    "reading_rank": index,
                    "selection_role": "RESERVE_ALL",
                    "paper_id": paper_id,
                    "queue_id": candidate.queue_id,
                    "canonical_work_id": candidate.canonical_work_id,
                    "title": candidate.title,
                    "quota_rq": "NOT_APPLICABLE_WITH_REASON:assigned only if reserve replaces a primary",
                    "secondary_rqs": ";".join(candidate.direct_rqs),
                    "directness": candidate.directness,
                    "relevance_class": candidate.relevance_class,
                    "effect_relation": candidate.effect_relation,
                    "selection_phase": "PROVISIONAL_RESERVE",
                    "tie_break_key": "RESERVE_" + candidate.canonical_work_id,
                    "broad_source_path": registry["source_path"],
                    "broad_source_format": _source_format(registry["source_path"]),
                    "broad_source_sha256": registry["source_sha256"],
                    "broad_source_bytes": registry["source_bytes"],
                    "method_source_path": method_by_work[candidate.canonical_work_id]["path"],
                    "method_source_format": method_by_work[candidate.canonical_work_id]["format"],
                    "method_source_sha256": method_by_work[candidate.canonical_work_id]["sha256"],
                    "method_source_bytes": method_by_work[candidate.canonical_work_id]["bytes"],
                    "method_source_origin": method_by_work[candidate.canonical_work_id]["origin"],
                    "method_source_authority": method_by_work[candidate.canonical_work_id]["authority"],
                    "method_source_url": method_by_work[candidate.canonical_work_id]["url"],
                    "method_source_receipt_path": method_by_work[candidate.canonical_work_id]["receipt_path"],
                    "method_source_status": (
                        "REUSE_VERIFIED_METHOD_PDF"
                        if method_by_work[candidate.canonical_work_id]["format"] == "PDF"
                        else "FULL_TEXT_DISCOVERY_REQUIRED"
                    ),
                    "screened_credit": "NOT_ASSESSED_AT_BROAD_LEVEL",
                }
            )
        _write_csv(temp / "SCREENED_PRIMARY.csv", primary_rows)
        _write_csv(temp / "SCREENED_READING_QUEUE.csv", primary_rows + reserve_rows)
        if all_reserve_rows:
            _write_csv(temp / "SCREENED_ALL_RESERVES.csv", all_reserve_rows)
        receipt = {
            "schema_version": "2.0",
            "status": "PASS",
            "created_at": datetime.now().astimezone().isoformat(),
            "primary_count": len(primary_rows),
            "reserve_count": len(selection.reserves),
            "reserve_read_count": len(reserve_rows),
            "reading_queue_count": len(primary_rows) + len(reserve_rows),
            "quota_counts": dict(selection.quota_counts),
            "eligible_count": len(candidates),
            "required_source_format": source_gate
            or "NOT_APPLICABLE_WITH_REASON:no source-format gate",
            "method_source_override_count": len(method_overrides),
            "method_source_override_ids": sorted(method_overrides),
            "formal_screened_increment": 0,
            "selection_is_provisional_until_full_method_review": True,
            "selection_policy": {
                "total": policy.total,
                "minimum_per_rq": policy.minimum_per_rq,
                "maximum_per_rq": policy.maximum_per_rq,
                "maximum_transfer": policy.maximum_transfer,
                "minimum_counterevidence_per_rq": (
                    policy.minimum_counterevidence_per_rq
                ),
                "mandatory_canonical_work_ids": list(
                    policy.mandatory_canonical_work_ids
                ),
                "tier_label": policy.tier_label,
                "frozen_seed": policy.frozen_seed,
            },
            "input_artifacts": input_artifacts,
            "formal_training_started": False,
            "engineering_gate_generated": False,
            "blind_holdout_opened": False,
        }
        (temp / "SCREENING_QUEUE_RECEIPT.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            backup = output.parent / f".{output.name}.previous"
            if backup.exists():
                raise ScreeningQueueError(f"stale queue backup exists: {backup}")
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

    return ScreeningQueueResult(
        status="PASS",
        output_root=output,
        primary_count=len(selection.selected),
        reserve_count=len(selection.reserves),
        reading_queue_count=len(selection.selected) + reserve_read_count,
        quota_counts=dict(selection.quota_counts),
        eligible_count=len(candidates),
        required_source_format=source_gate,
    )
