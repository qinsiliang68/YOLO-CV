"""Lossless candidate-universe and manual-review queue construction.

Discovery records are versions, not canonical papers.  This module preserves
every source record and only emits conservative exact-title/DOI grouping hints
for later primary-source identity review.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


class CandidateUniverseError(RuntimeError):
    """Raised when candidate provenance or identity is incomplete."""


_RQ_IDS = {f"RQ{index}" for index in range(1, 9)}
_LEGACY_TOPIC_RQS: dict[str, tuple[str, ...]] = {
    "TRAINING_DYNAMICS": ("RQ1",),
    "DYNAMIC_REPLAY": ("RQ1", "RQ6"),
    "STATIC_SCORE_BASELINES": ("RQ2", "RQ7"),
    "INFLUENCE_ATTRIBUTION": ("RQ3",),
    "DATA_ATTRIBUTION": ("RQ3",),
    "NOISY_LABEL_HARD_CLEAN": ("RQ4",),
    "NOISY_LABELS": ("RQ4",),
    "DATA_SUBSET_PRUNING": ("RQ5",),
    "DATA_SUBSET_SELECTION": ("RQ5",),
    "ACTIVE_LEARNING": ("RQ5", "RQ7"),
    "REPLAY_AND_SCHEDULING": ("RQ6",),
    "EXPERIMENTAL_VARIANCE": ("RQ7", "RQ8"),
    "TRAINING_RANDOMNESS": ("RQ7", "RQ8"),
    "OPTIMIZATION_STABILITY": ("RQ7", "RQ8"),
    "OPTIMIZATION_AND_SEED_VARIANCE": ("RQ7", "RQ8"),
    "OPERATIONAL_TAIL": ("RQ8",),
    "OPERATIONAL_TAIL_AND_CALIBRATION": ("RQ8",),
}


def _normalize_title(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _split(value: Any) -> set[str]:
    return {
        item.strip()
        for item in str(value or "").split(";")
        if item.strip() and not item.strip().startswith("NOT_")
    }


def _normalize_doi(value: Any) -> str:
    clean = str(value or "").strip().casefold()
    clean = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", clean).rstrip("/")
    if not clean.startswith("10.") or "/" not in clean:
        return "NOT_REPORTED_BY_SOURCE"
    return clean


def _digest(prefix: str, value: str, length: int = 20) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:length].upper()
    return f"{prefix}{digest}"


def _required(row: Mapping[str, Any], field: str, source_record: str) -> str:
    value = str(row.get(field, "")).strip()
    if not value:
        raise CandidateUniverseError(f"{source_record}: missing required field {field}")
    return value


def _common_version(
    *,
    source_origin: str,
    source_record_id: str,
    title: str,
    authors: str,
    year: Any,
    venue: str,
    primary_url: str,
    full_text_url: str,
    doi: str,
    abstract: str,
    title_rq_ids: Iterable[str],
    query_rq_ids: Iterable[str],
    discovery_decision: str,
    discovery_reason: str,
    manual_review_required: bool,
    inherited_read_depth: str,
    source_query_ids: str,
) -> dict[str, str]:
    normalized_title = _normalize_title(title)
    if not normalized_title:
        raise CandidateUniverseError(f"{source_origin}:{source_record_id}: title normalizes empty")
    title_rqs = sorted(set(title_rq_ids))
    query_rqs = sorted(set(query_rq_ids))
    rqs = sorted(set(title_rqs).union(query_rqs))
    invalid_rqs = set(rqs) - _RQ_IDS
    if invalid_rqs:
        raise CandidateUniverseError(
            f"{source_origin}:{source_record_id}: invalid RQ IDs {sorted(invalid_rqs)}"
        )
    normalized_doi = _normalize_doi(doi)
    source_identity = f"{source_origin}\0{source_record_id}"
    return {
        "candidate_version_id": _digest("CV", source_identity),
        "source_origin": source_origin,
        "source_record_id": source_record_id,
        "title": title,
        "normalized_title": normalized_title,
        "authors": authors or "NOT_REPORTED_BY_SOURCE",
        "year": str(year or "NOT_REPORTED_BY_SOURCE"),
        "venue": venue or "NOT_REPORTED_BY_SOURCE",
        "primary_url": primary_url or "NOT_REPORTED_BY_SOURCE",
        "full_text_url": full_text_url or "NOT_REPORTED_BY_SOURCE",
        "doi": normalized_doi,
        "abstract": abstract or "NOT_REPORTED_BY_SOURCE",
        "title_rq_ids": ";".join(title_rqs)
        if title_rqs
        else "NOT_APPLICABLE_WITH_REASON:no title or legacy-topic RQ mapping",
        "query_rq_ids": ";".join(query_rqs)
        if query_rqs
        else "NOT_APPLICABLE_WITH_REASON:no query-derived RQ mapping",
        "proposed_rq_ids": ";".join(rqs)
        if rqs
        else "NOT_APPLICABLE_WITH_REASON:no preregistered RQ mapping yet",
        "discovery_decision": discovery_decision,
        "discovery_reason": discovery_reason,
        "manual_review_required": str(manual_review_required).lower(),
        "inherited_read_depth": inherited_read_depth,
        "source_query_ids": source_query_ids
        or "NOT_APPLICABLE_WITH_REASON:no database query identifier",
        "exact_title_group_hint": _digest("TITLE-", normalized_title, 16),
        "exact_doi_group_hint": (
            _digest("DOI-", normalized_doi, 16)
            if normalized_doi != "NOT_REPORTED_BY_SOURCE"
            else "NOT_APPLICABLE_WITH_REASON:no DOI supplied by source"
        ),
        "identity_status": "UNRESOLVED_MANUAL",
        "v2_reading_credit": "NONE_DISCOVERY_ONLY",
    }


def build_candidate_universe(
    openalex_rows: Sequence[Mapping[str, Any]],
    legacy_rows: Sequence[Mapping[str, Any]],
    targeted_rows: Sequence[Mapping[str, Any]],
    *,
    openalex_query_rq_map: Mapping[str, set[str]] | None = None,
) -> list[dict[str, str]]:
    """Preserve every discovery record as a stable candidate-version row."""

    versions: list[dict[str, str]] = []
    source_keys: set[tuple[str, str]] = set()

    def append(version: dict[str, str]) -> None:
        key = (version["source_origin"], version["source_record_id"])
        if key in source_keys:
            raise CandidateUniverseError(f"duplicate source record {key[0]}:{key[1]}")
        source_keys.add(key)
        versions.append(version)

    query_rq_map = openalex_query_rq_map or {}
    for row in openalex_rows:
        source_record = _required(row, "candidate_key", "OPENALEX_V1")
        decision = _required(row, "prefilter_decision", f"OPENALEX_V1:{source_record}")
        query_ids = _split(row.get("query_ids"))
        query_rqs = {
            rq_id
            for query_id in query_ids
            for rq_id in query_rq_map.get(query_id, set())
        }
        append(
            _common_version(
                source_origin="OPENALEX_V1",
                source_record_id=source_record,
                title=_required(row, "title", f"OPENALEX_V1:{source_record}"),
                authors=str(row.get("authors", "")),
                year=row.get("year"),
                venue=str(row.get("venue", "")),
                primary_url=str(row.get("primary_url", "")),
                full_text_url=str(row.get("full_text_url", "")),
                doi=str(row.get("doi", "")),
                abstract=str(row.get("abstract", "")),
                title_rq_ids=_split(row.get("rq_ids")),
                query_rq_ids=query_rqs,
                discovery_decision=decision,
                discovery_reason=str(row.get("prefilter_reason", "")),
                manual_review_required=decision == "MANUAL_SCREEN_REQUIRED",
                inherited_read_depth="NOT_APPLICABLE_WITH_REASON:new OpenAlex discovery",
                source_query_ids=";".join(sorted(query_ids)),
            )
        )

    for row in legacy_rows:
        source_record = _required(row, "evidence_id", "LEGACY_155")
        append(
            _common_version(
                source_origin="LEGACY_155",
                source_record_id=source_record,
                title=_required(row, "title", f"LEGACY_155:{source_record}"),
                authors=str(row.get("authors", "")),
                year=row.get("year"),
                venue=str(row.get("venue", "")),
                primary_url=str(row.get("primary_url", "")),
                full_text_url="NOT_REPORTED_BY_SOURCE",
                doi=str(row.get("doi", "")),
                abstract=str(row.get("abstract", "")),
                title_rq_ids=_LEGACY_TOPIC_RQS.get(str(row.get("topic", "")), ()),
                query_rq_ids=(),
                discovery_decision="LEGACY_REVALIDATION_REQUIRED",
                discovery_reason="Legacy evidence is retained as a candidate but receives no v2 reading credit.",
                manual_review_required=True,
                inherited_read_depth=str(row.get("screening_depth", ""))
                or "NOT_REPORTED_BY_SOURCE",
                source_query_ids="NOT_APPLICABLE_WITH_REASON:legacy candidate ledger",
            )
        )

    for row in targeted_rows:
        source_record = _required(row, "target_id", "TARGETED_PRIMARY_V1")
        screening_status = _required(
            row, "screening_status", f"TARGETED_PRIMARY_V1:{source_record}"
        )
        append(
            _common_version(
                source_origin="TARGETED_PRIMARY_V1",
                source_record_id=source_record,
                title=_required(row, "title", f"TARGETED_PRIMARY_V1:{source_record}"),
                authors=str(row.get("authors", "")),
                year=row.get("year"),
                venue=str(row.get("venue", "")),
                primary_url=str(row.get("primary_url", "")),
                full_text_url=str(row.get("full_text_url", "")),
                doi=str(row.get("doi_or_repository_id", "")),
                abstract=str(row.get("abstract", "")),
                title_rq_ids=_split(row.get("proposed_rq_ids")),
                query_rq_ids=(),
                discovery_decision=screening_status,
                discovery_reason=str(row.get("directness_hint", "")),
                manual_review_required=screening_status == "PENDING_PRIMARY_SOURCE_REVIEW",
                inherited_read_depth="NOT_APPLICABLE_WITH_REASON:new targeted discovery",
                source_query_ids=str(row.get("target_id", "")),
            )
        )

    expected_count = len(openalex_rows) + len(legacy_rows) + len(targeted_rows)
    if len(versions) != expected_count:
        raise CandidateUniverseError(
            f"candidate version loss: expected {expected_count}, observed {len(versions)}"
        )
    return sorted(versions, key=lambda row: row["candidate_version_id"])


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def build_manual_review_groups(
    universe_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Build unresolved review groups from exact title/DOI evidence only."""

    eligible = [
        {str(key): str(value) for key, value in row.items()}
        for row in universe_rows
        if str(row.get("manual_review_required", "")).casefold() == "true"
    ]
    ids = [row["candidate_version_id"] for row in eligible]
    if len(ids) != len(set(ids)):
        raise CandidateUniverseError("duplicate candidate_version_id in candidate universe")
    union_find = _UnionFind(ids)
    first_by_title: dict[str, str] = {}
    first_by_doi: dict[str, str] = {}
    for row in eligible:
        version_id = row["candidate_version_id"]
        title_key = row["normalized_title"]
        if title_key in first_by_title:
            union_find.union(version_id, first_by_title[title_key])
        else:
            first_by_title[title_key] = version_id
        doi = row["doi"]
        if doi != "NOT_REPORTED_BY_SOURCE":
            if doi in first_by_doi:
                union_find.union(version_id, first_by_doi[doi])
            else:
                first_by_doi[doi] = version_id

    components: dict[str, list[dict[str, str]]] = {}
    for row in eligible:
        components.setdefault(union_find.find(row["candidate_version_id"]), []).append(row)

    origin_preference = {"TARGETED_PRIMARY_V1": 0, "OPENALEX_V1": 1, "LEGACY_155": 2}
    groups: list[dict[str, str]] = []
    for members in components.values():
        members.sort(
            key=lambda row: (
                origin_preference.get(row["source_origin"], 99),
                row["candidate_version_id"],
            )
        )
        preferred = members[0]
        member_ids = sorted(row["candidate_version_id"] for row in members)
        group_id = _digest("RG", "\0".join(member_ids), 16)
        rqs = sorted(
            {
                rq
                for row in members
                for rq in _split(row.get("proposed_rq_ids"))
            }
        )
        title_hints = sorted({row["exact_title_group_hint"] for row in members})
        doi_hints = sorted(
            {
                row["exact_doi_group_hint"]
                for row in members
                if not row["exact_doi_group_hint"].startswith("NOT_")
            }
        )
        grouping_evidence = ";".join([*title_hints, *doi_hints])
        groups.append(
            {
                "queue_id": group_id,
                "review_group_id": group_id,
                "queue_status": "PENDING_PRIMARY_SOURCE_REVIEW",
                "identity_status": "UNRESOLVED_MANUAL",
                "title": preferred["title"],
                "authors": preferred["authors"],
                "year": preferred["year"],
                "venue": preferred["venue"],
                "primary_url": preferred["primary_url"],
                "full_text_url": preferred["full_text_url"],
                "doi": preferred["doi"],
                "abstract": preferred["abstract"],
                "rq_ids": ";".join(rqs)
                if rqs
                else "NOT_APPLICABLE_WITH_REASON:no preregistered RQ mapping yet",
                "source_origins": ";".join(sorted({row["source_origin"] for row in members})),
                "candidate_version_ids": ";".join(member_ids),
                "candidate_version_count": str(len(member_ids)),
                "discovery_decisions": ";".join(
                    sorted({row["discovery_decision"] for row in members})
                ),
                "grouping_evidence": grouping_evidence,
                "v2_counted_tier": "NOT_APPLICABLE_WITH_REASON:primary-source review not complete",
            }
        )
    return sorted(groups, key=lambda row: row["review_group_id"])


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise CandidateUniverseError(f"cannot read {path}: {exc}") from exc
    if not rows:
        raise CandidateUniverseError(f"input CSV is empty: {path}")
    return rows


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise CandidateUniverseError(f"refusing to publish empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _relative(path: Path, anchor: Path) -> str:
    return Path(os.path.relpath(path.resolve(), anchor.resolve())).as_posix()


def publish_candidate_universe(
    *,
    openalex_path: str | Path,
    legacy_path: str | Path,
    targeted_path: str | Path,
    query_plan_path: str | Path | None = None,
    universe_path: str | Path,
    manual_queue_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Build and atomically publish the lossless universe and review groups."""

    inputs = {
        "OPENALEX_V1": Path(openalex_path),
        "LEGACY_155": Path(legacy_path),
        "TARGETED_PRIMARY_V1": Path(targeted_path),
    }
    source_rows = {role: _read_csv(path) for role, path in inputs.items()}
    query_rq_map: dict[str, set[str]] = {}
    query_count = 0
    query_plan = Path(query_plan_path) if query_plan_path is not None else None
    if query_plan is not None:
        from stage1_dynamic_replay_v3.literature_discovery_v2 import load_query_plan

        specs = load_query_plan(query_plan)
        query_count = len(specs)
        query_rq_map = {spec.query_id: set(spec.rq_ids) for spec in specs}
    universe = build_candidate_universe(
        source_rows["OPENALEX_V1"],
        source_rows["LEGACY_155"],
        source_rows["TARGETED_PRIMARY_V1"],
        openalex_query_rq_map=query_rq_map,
    )
    groups = build_manual_review_groups(universe)
    if not groups:
        raise CandidateUniverseError("manual review queue is empty")

    universe_output = Path(universe_path)
    queue_output = Path(manual_queue_path)
    manifest_output = Path(manifest_path)
    _atomic_csv(universe_output, universe)
    _atomic_csv(queue_output, groups)

    anchor = manifest_output.parent
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "PASS",
        "candidate_version_count": len(universe),
        "manual_review_group_count": len(groups),
        "canonical_work_count": "NOT_APPLICABLE_WITH_REASON:identity review not complete",
        "inputs": [
            {
                "role": role,
                "path": _relative(path, anchor),
                "rows": len(source_rows[role]),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for role, path in inputs.items()
        ]
        + (
            [
                {
                    "role": "OPENALEX_QUERY_PLAN",
                    "path": _relative(query_plan, anchor),
                    "rows": query_count,
                    "bytes": query_plan.stat().st_size,
                    "sha256": _sha256(query_plan),
                }
            ]
            if query_plan is not None
            else []
        ),
        "outputs": [
            {
                "role": "CANDIDATE_VERSION_UNIVERSE",
                "path": _relative(universe_output, anchor),
                "rows": len(universe),
                "bytes": universe_output.stat().st_size,
                "sha256": _sha256(universe_output),
            },
            {
                "role": "UNRESOLVED_MANUAL_REVIEW_GROUPS",
                "path": _relative(queue_output, anchor),
                "rows": len(groups),
                "bytes": queue_output.stat().st_size,
                "sha256": _sha256(queue_output),
            },
        ],
        "selection_credit_granted": False,
        "reading_credit_granted": False,
        "formal_training_started": False,
        "engineering_gate_generated": False,
        "blind_holdout_opened": False,
        "note": (
            "Every input record is preserved as a candidate version. Exact title/DOI groups are "
            "unresolved review hints, not canonical works or reading evidence."
        ),
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_output.with_suffix(manifest_output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_output)
    return payload
