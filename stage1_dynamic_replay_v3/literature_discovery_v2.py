"""Reproducible high-recall discovery for the Stage1 literature v2 corpus."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping, Sequence

import requests


RESEARCH_QUESTIONS = {f"RQ{index}" for index in range(1, 9)}
OPENALEX_ENDPOINT = "https://api.openalex.org/works"
USER_AGENT = "YOLO-CV-Stage1-Literature-Evidence/2.0"

SECONDARY_TITLE_RE = re.compile(
    r"\b(?:survey|review|overview|tutorial|bibliometric|systematic review|taxonomy)\b",
    re.IGNORECASE,
)
ML_CONTEXT_RE = re.compile(
    r"\b(?:deep learning|machine learning|neural|model|training|train(?:ed|ing)?|"
    r"classifier|classification|dataset|gradient|active learning|continual learning|learner)\b",
    re.IGNORECASE,
)
TRAINING_SELECTION_CONTEXT_RE = re.compile(
    r"(?:training data|train(?:ing|ed)?\s+(?:examples?|samples?|subset)|"
    r"(?:examples?|samples?|subset)\s+(?:for|during|to)\s+train|"
    r"select(?:ed|ing|ion)?[^.]{0,80}(?:model|train|classifier|learner)|"
    r"data selection[^.]{0,80}(?:model|train|learning)|"
    r"(?:model|classifier|learner)[^.]{0,80}(?:train|select))",
    re.IGNORECASE,
)
TITLE_MECHANISMS: dict[str, tuple[re.Pattern[str], ...]] = {
    "RQ1": tuple(
        re.compile(value, re.IGNORECASE)
        for value in (
            r"training dynamics",
            r"learning dynamics",
            r"example forgetting",
            r"dataset cartograph",
            r"hard[- ]to[- ]learn",
            r"slow[- ]learn",
            r"memorization.*training|training.*memorization",
        )
    ),
    "RQ2": tuple(
        re.compile(value, re.IGNORECASE)
        for value in (
            r"reducible loss",
            r"irreducible loss",
            r"learnability",
            r"worth learning",
            r"not yet learn",
            r"data diet",
            r"sample importance|example importance|important examples",
            r"data valuation",
        )
    ),
    "RQ3": tuple(
        re.compile(value, re.IGNORECASE)
        for value in (
            r"gradient match",
            r"gradient align",
            r"influence function",
            r"training data influence",
            r"data attribution",
            r"reweight.*(?:example|sample)",
            r"data shapley",
            r"\btracin\b",
            r"\btrak\b",
        )
    ),
    "RQ4": tuple(
        re.compile(value, re.IGNORECASE)
        for value in (
            r"noisy label",
            r"label noise",
            r"mislabel",
            r"hard[- ]clean",
            r"confidence tracking",
            r"area under.*margin",
            r"detrimental.*sample",
            r"outlier gradient",
        )
    ),
    "RQ5": tuple(
        re.compile(value, re.IGNORECASE)
        for value in (
            r"data pruning",
            r"dataset pruning",
            r"coreset selection",
            r"data subset selection",
            r"training data selection",
            r"data selection",
            r"sample selection",
            r"dataset dedup",
            r"deduplicat.*training",
            r"coverage.*selection|selection.*coverage",
            r"diversity.*selection|selection.*diversity",
        )
    ),
    "RQ6": tuple(
        re.compile(value, re.IGNORECASE)
        for value in (
            r"experience replay",
            r"replay sample",
            r"replay sched",
            r"online batch selection",
            r"selective backprop",
            r"hard example mining",
            r"curriculum learning",
            r"self[- ]paced learning",
        )
    ),
    "RQ7": tuple(
        re.compile(value, re.IGNORECASE)
        for value in (
            r"random.*data selection|data selection.*random",
            r"training seed",
            r"seed variance",
            r"data order",
        )
    ),
    "RQ8": tuple(
        re.compile(value, re.IGNORECASE)
        for value in (
            r"neyman.*pearson.*classif",
            r"partial auc",
            r"recall[- ]constrain",
            r"false negative.*constrain",
        )
    ),
}
RQ5_GENERIC_TITLE_RE = re.compile(
    r"\b(?:training data selection|data selection|sample selection)\b",
    re.IGNORECASE,
)
LEGACY_TOPIC_RQS: dict[str, tuple[str, ...]] = {
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


class DiscoveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    database: str
    query: str
    rq_ids: tuple[str, ...]
    filters: str | None = None


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _normalize_doi(value: Any) -> str:
    if not value:
        return "NOT_REPORTED_BY_SOURCE"
    clean = str(value).strip().casefold()
    clean = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", clean)
    return clean.rstrip("/") or "NOT_REPORTED_BY_SOURCE"


def _openalex_short_id(value: Any) -> str:
    return str(value or "").rstrip("/").rsplit("/", 1)[-1]


def load_query_plan(path: str | Path) -> tuple[QuerySpec, ...]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiscoveryError(f"cannot read query plan {source}: {exc}") from exc
    if payload.get("schema_version") != "1.0":
        raise DiscoveryError("query plan schema_version must be 1.0")
    rows = payload.get("queries")
    if not isinstance(rows, list) or not rows:
        raise DiscoveryError("query plan must contain a non-empty queries list")
    seen: set[str] = set()
    specs: list[QuerySpec] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise DiscoveryError(f"query row {index} must be an object")
        query_id = str(row.get("query_id", "")).strip()
        if not re.fullmatch(r"[A-Z]{1,4}\d{3}", query_id):
            raise DiscoveryError(f"invalid query_id {query_id!r}")
        if query_id in seen:
            raise DiscoveryError(f"duplicate query_id {query_id}")
        seen.add(query_id)
        database = str(row.get("database", "")).strip().upper()
        if database != "OPENALEX":
            raise DiscoveryError(f"{query_id}: unsupported database {database!r}")
        query = str(row.get("query", "")).strip()
        if len(query) < 12:
            raise DiscoveryError(f"{query_id}: query is too short")
        rq_ids = tuple(str(value) for value in row.get("rq_ids", []))
        if not rq_ids or set(rq_ids) - RESEARCH_QUESTIONS:
            raise DiscoveryError(f"{query_id}: invalid or empty rq_ids {rq_ids}")
        filters = str(row.get("filters", "")).strip() or None
        specs.append(QuerySpec(query_id, database, query, rq_ids, filters))
    return tuple(specs)


def reconstruct_openalex_abstract(inverted_index: Any) -> str:
    if not isinstance(inverted_index, Mapping) or not inverted_index:
        return "NOT_REPORTED_BY_SOURCE"
    positions: dict[int, str] = {}
    for token, raw_positions in inverted_index.items():
        if not isinstance(raw_positions, list):
            raise DiscoveryError("OpenAlex abstract positions must be a list")
        for raw_position in raw_positions:
            try:
                position = int(raw_position)
            except (TypeError, ValueError) as exc:
                raise DiscoveryError("OpenAlex abstract position is not an integer") from exc
            if position in positions:
                raise DiscoveryError("OpenAlex abstract contains a duplicate token position")
            positions[position] = str(token)
    # Some OpenAlex records preserve sparse positions from an upstream source.
    # Ordering is still defined; only duplicate positions are ambiguous.
    return " ".join(positions[index] for index in sorted(positions))


def _candidate_from_work(work: Mapping[str, Any], *, query_id: str) -> dict[str, Any]:
    title = str(work.get("title") or work.get("display_name") or "").strip()
    if not title:
        raise DiscoveryError(f"{query_id}: OpenAlex work has no title")
    doi = _normalize_doi(work.get("doi"))
    normalized_title = _normalize_text(title)
    # Discovery merges exact normalized titles across preprint/proceedings/journal
    # records and preserves every DOI for later manual canonicalization.
    identity_key = f"title:{normalized_title}"
    authors = []
    for authorship in work.get("authorships") or []:
        if isinstance(authorship, Mapping):
            author = authorship.get("author")
            if isinstance(author, Mapping) and author.get("display_name"):
                authors.append(str(author["display_name"]).strip())
    primary_location = work.get("primary_location") or {}
    if not isinstance(primary_location, Mapping):
        primary_location = {}
    source = primary_location.get("source") or {}
    if not isinstance(source, Mapping):
        source = {}
    open_access = work.get("open_access") or {}
    if not isinstance(open_access, Mapping):
        open_access = {}
    openalex_id = _openalex_short_id(work.get("id"))
    if not openalex_id:
        raise DiscoveryError(f"{query_id}: OpenAlex work has no identity")
    return {
        "identity_key": identity_key,
        "title": title,
        "authors": tuple(authors),
        "year": work.get("publication_year") or "NOT_REPORTED_BY_SOURCE",
        "venue": str(source.get("display_name") or "NOT_REPORTED_BY_SOURCE"),
        "dois": {doi} if doi != "NOT_REPORTED_BY_SOURCE" else set(),
        "openalex_ids": {openalex_id},
        "primary_url": str(
            primary_location.get("landing_page_url")
            or work.get("doi")
            or work.get("id")
            or ""
        ),
        "full_text_url": str(open_access.get("oa_url") or "NOT_REPORTED_BY_SOURCE"),
        "abstract": reconstruct_openalex_abstract(work.get("abstract_inverted_index")),
        "query_ids": {query_id},
        "max_relevance_score": float(work.get("relevance_score") or 0.0),
        "cited_by_count": int(work.get("cited_by_count") or 0),
        "work_types": {str(work.get("type") or "NOT_REPORTED_BY_SOURCE")},
    }


def _merge_candidate(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    target["query_ids"].update(incoming["query_ids"])
    target["openalex_ids"].update(incoming["openalex_ids"])
    target["work_types"].update(incoming["work_types"])
    target["dois"].update(incoming["dois"])
    target["max_relevance_score"] = max(
        float(target["max_relevance_score"]), float(incoming["max_relevance_score"])
    )
    target["cited_by_count"] = max(int(target["cited_by_count"]), int(incoming["cited_by_count"]))
    target["authors"] = tuple(dict.fromkeys((*target["authors"], *incoming["authors"])))
    for field in ("primary_url", "full_text_url", "abstract", "venue"):
        current = str(target.get(field) or "")
        new = str(incoming.get(field) or "")
        if current in {"", "NOT_REPORTED_BY_SOURCE"} or (
            field == "abstract" and len(new) > len(current)
        ):
            target[field] = new


def build_candidate_inventory(
    snapshots: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize and deduplicate OpenAlex raw snapshots without deciding inclusion."""

    merged: dict[str, dict[str, Any]] = {}
    for descriptor in snapshots:
        query_id = str(descriptor.get("query_id", "")).strip()
        snapshot = Path(str(descriptor.get("snapshot_path", "")))
        try:
            payload = json.loads(snapshot.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DiscoveryError(f"cannot parse snapshot {snapshot}: {exc}") from exc
        works = payload.get("results")
        if not isinstance(works, list):
            raise DiscoveryError(f"{snapshot}: results must be a list")
        for work in works:
            if not isinstance(work, Mapping):
                raise DiscoveryError(f"{snapshot}: result is not an object")
            candidate = _candidate_from_work(work, query_id=query_id)
            key = candidate["identity_key"]
            if key in merged:
                _merge_candidate(merged[key], candidate)
            else:
                merged[key] = candidate

    rows: list[dict[str, Any]] = []
    for identity_key, candidate in merged.items():
        canonical_digest = hashlib.sha256(identity_key.encode("utf-8")).hexdigest()[:16].upper()
        rows.append(
            {
                "candidate_key": f"OA-{canonical_digest}",
                "title": candidate["title"],
                "authors": "; ".join(candidate["authors"]),
                "year": candidate["year"],
                "venue": candidate["venue"],
                "doi": ";".join(sorted(candidate["dois"])) or "NOT_REPORTED_BY_SOURCE",
                "openalex_ids": ";".join(sorted(candidate["openalex_ids"])),
                "primary_url": candidate["primary_url"],
                "full_text_url": candidate["full_text_url"],
                "abstract": candidate["abstract"],
                "query_ids": ";".join(sorted(candidate["query_ids"])),
                "max_relevance_score": candidate["max_relevance_score"],
                "cited_by_count": candidate["cited_by_count"],
                "work_types": ";".join(sorted(candidate["work_types"])),
                "source_database": "OpenAlex",
            }
        )
    return sorted(
        rows,
        key=lambda row: (-float(row["max_relevance_score"]), -int(row["cited_by_count"]), row["title"]),
    )


def triage_candidate_inventory(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Apply explicit relevance gates; no blended score is produced."""

    triaged: list[dict[str, Any]] = []
    for row in rows:
        title = str(row.get("title", "")).strip()
        abstract = str(row.get("abstract", "")).strip()
        rq_ids = [
            rq_id
            for rq_id, patterns in TITLE_MECHANISMS.items()
            if any(pattern.search(title) for pattern in patterns)
        ]
        generic_selection = bool(RQ5_GENERIC_TITLE_RE.search(title))
        if SECONDARY_TITLE_RE.search(title):
            decision = "EXCLUDED_SECONDARY_REVIEW"
            reason = "Title identifies a review/survey rather than a primary mechanism study."
            priority = "NOT_APPLICABLE"
        elif not rq_ids:
            decision = "EXCLUDED_NO_DIRECT_MECHANISM"
            reason = "Title does not directly name a preregistered sample-value or replay mechanism."
            priority = "NOT_APPLICABLE"
        elif set(rq_ids) == {"RQ1"} and not ML_CONTEXT_RE.search(abstract):
            decision = "EXCLUDED_NON_ML_DYNAMICS"
            reason = "Learning-dynamics wording is not about model-training sample behavior."
            priority = "NOT_APPLICABLE"
        elif generic_selection and not TRAINING_SELECTION_CONTEXT_RE.search(abstract):
            decision = "EXCLUDED_NO_TRAINING_SELECTION"
            reason = "Sample/data selection is not used to choose finite-budget model-training examples."
            priority = "NOT_APPLICABLE"
        elif not ML_CONTEXT_RE.search(abstract) and not set(rq_ids).intersection({"RQ8"}):
            decision = "EXCLUDED_NO_ML_TRAINING_CONTEXT"
            reason = "Abstract does not establish a machine-learning training or classification context."
            priority = "NOT_APPLICABLE"
        else:
            decision = "MANUAL_SCREEN_REQUIRED"
            if generic_selection and set(rq_ids) == {"RQ5"}:
                priority = "APPLICATION_TRANSFER"
                reason = (
                    "Primary application study explicitly selects training samples; manual review must "
                    "verify budget, random controls, and transferable mechanism."
                )
            else:
                priority = "CORE_MECHANISM"
                reason = (
                    "Title directly names a preregistered mechanism and the abstract establishes an ML "
                    "training context; primary-source manual screening is still required."
                )
        output = dict(row)
        output.update(
            {
                "rq_ids": ";".join(rq_ids) if rq_ids else "NOT_APPLICABLE_WITH_REASON:no direct RQ mapping",
                "prefilter_decision": decision,
                "priority_band": priority,
                "prefilter_reason": reason,
            }
        )
        triaged.append(output)
    return triaged


def _split_ids(value: Any) -> set[str]:
    return {
        item.strip()
        for item in str(value or "").split(";")
        if item.strip() and not item.startswith("NOT_")
    }


def build_manual_screen_queue(
    openalex_rows: Sequence[Mapping[str, Any]],
    legacy_rows: Sequence[Mapping[str, Any]],
    *,
    legacy_note_ids: set[str],
    legacy_source_ids: set[str],
) -> list[dict[str, Any]]:
    """Merge old and new candidates while explicitly revoking inherited read depth."""

    merged: dict[str, dict[str, Any]] = {}
    for row in openalex_rows:
        if row.get("prefilter_decision") != "MANUAL_SCREEN_REQUIRED":
            continue
        title = str(row.get("title", "")).strip()
        key = _normalize_text(title)
        merged[key] = {
            "title": title,
            "authors": str(row.get("authors", "")),
            "year": row.get("year", ""),
            "venue": str(row.get("venue", "")),
            "primary_url": str(row.get("primary_url", "")),
            "doi": str(row.get("doi", "")),
            "abstract": str(row.get("abstract", "")),
            "rq_ids_set": _split_ids(row.get("rq_ids")),
            "source_origins_set": {"OPENALEX"},
            "candidate_keys_set": {str(row.get("candidate_key", ""))},
            "legacy_ids_set": set(),
            "legacy_depths": set(),
            "legacy_note_present": False,
            "legacy_source_present": False,
            "new_priority_band": str(row.get("priority_band", "")),
            "prefilter_reason": str(row.get("prefilter_reason", "")),
        }

    for row in legacy_rows:
        evidence_id = str(row.get("evidence_id", "")).strip()
        title = str(row.get("title", "")).strip()
        key = _normalize_text(title)
        target = merged.get(key)
        if target is None:
            target = {
                "title": title,
                "authors": str(row.get("authors", "")),
                "year": row.get("year", ""),
                "venue": str(row.get("venue", "")),
                "primary_url": str(row.get("primary_url", "")),
                "doi": str(row.get("doi", "")) or "NOT_REPORTED_BY_SOURCE",
                "abstract": str(row.get("abstract", "")),
                "rq_ids_set": set(),
                "source_origins_set": set(),
                "candidate_keys_set": set(),
                "legacy_ids_set": set(),
                "legacy_depths": set(),
                "legacy_note_present": False,
                "legacy_source_present": False,
                "new_priority_band": "NOT_APPLICABLE",
                "prefilter_reason": "Legacy candidate requires complete v2 revalidation.",
            }
            merged[key] = target
        target["source_origins_set"].add("LEGACY_155")
        target["legacy_ids_set"].add(evidence_id)
        depth = str(row.get("screening_depth", "")).strip()
        if depth:
            target["legacy_depths"].add(depth)
        target["legacy_note_present"] = target["legacy_note_present"] or evidence_id in legacy_note_ids
        target["legacy_source_present"] = (
            target["legacy_source_present"] or evidence_id in legacy_source_ids
        )
        target["rq_ids_set"].update(LEGACY_TOPIC_RQS.get(str(row.get("topic", "")), ()))
        if not target["authors"]:
            target["authors"] = str(row.get("authors", ""))
        if not target["abstract"]:
            target["abstract"] = str(row.get("abstract", ""))
        if not target["primary_url"]:
            target["primary_url"] = str(row.get("primary_url", ""))

    depth_priority = {"DEEP_READ": 0, "METHOD_READ": 1, "ABSTRACT_SCREEN": 2}
    queue_rows: list[dict[str, Any]] = []
    for target in merged.values():
        legacy_depth = min(
            target["legacy_depths"],
            key=lambda value: depth_priority.get(value, 99),
            default="NOT_APPLICABLE_WITH_REASON:new discovery candidate",
        )
        if legacy_depth == "DEEP_READ":
            queue_band = "LEGACY_DEEP_REVALIDATION"
        elif legacy_depth == "METHOD_READ":
            queue_band = "LEGACY_METHOD_REVALIDATION"
        elif legacy_depth == "ABSTRACT_SCREEN":
            queue_band = "LEGACY_ABSTRACT_REVALIDATION"
        elif target["new_priority_band"] == "CORE_MECHANISM":
            queue_band = "NEW_CORE_MECHANISM"
        else:
            queue_band = "NEW_APPLICATION_TRANSFER"
        queue_rows.append(
            {
                "queue_id": "",
                "queue_band": queue_band,
                "queue_status": "PENDING_PRIMARY_SOURCE_REVIEW",
                "title": target["title"],
                "authors": target["authors"],
                "year": target["year"],
                "venue": target["venue"],
                "primary_url": target["primary_url"],
                "doi": target["doi"],
                "abstract": target["abstract"],
                "rq_ids": ";".join(sorted(target["rq_ids_set"])),
                "source_origins": ";".join(sorted(target["source_origins_set"])),
                "candidate_keys": ";".join(sorted(target["candidate_keys_set"])),
                "legacy_evidence_ids": ";".join(sorted(target["legacy_ids_set"])),
                "legacy_depth": legacy_depth,
                "legacy_note_present": str(bool(target["legacy_note_present"])).lower(),
                "legacy_source_bytes_present": str(bool(target["legacy_source_present"])).lower(),
                "prefilter_reason": target["prefilter_reason"],
                "v2_counted_tier": "NOT_APPLICABLE_WITH_REASON:primary-source review not complete",
            }
        )
    band_order = {
        "LEGACY_DEEP_REVALIDATION": 0,
        "LEGACY_METHOD_REVALIDATION": 1,
        "LEGACY_ABSTRACT_REVALIDATION": 2,
        "NEW_CORE_MECHANISM": 3,
        "NEW_APPLICATION_TRANSFER": 4,
    }
    queue_rows.sort(
        key=lambda row: (
            band_order[row["queue_band"]],
            row["rq_ids"],
            -int(row["year"] or 0),
            row["title"].casefold(),
        )
    )
    for index, row in enumerate(queue_rows, start=1):
        row["queue_id"] = f"LQ{index:04d}"
    return queue_rows


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def fetch_openalex_queries(
    specs: Sequence[QuerySpec],
    *,
    corpus_root: str | Path,
    per_page: int = 100,
    timeout_seconds: float = 60.0,
    force: bool = False,
    delay_seconds: float = 0.15,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Fetch one relevance-sorted raw result page per preregistered query."""

    if not 1 <= per_page <= 200:
        raise ValueError("per_page must be in 1..200")
    root = Path(corpus_root).resolve()
    raw_root = root / "discovery" / "raw" / "openalex_v1"
    snapshots: list[dict[str, str]] = []
    query_log: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    for spec in specs:
        snapshot = raw_root / f"{spec.query_id}.json"
        receipt = raw_root / f"{spec.query_id}.receipt.json"
        if snapshot.exists() and receipt.exists() and not force:
            receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
            raw_bytes = snapshot.read_bytes()
            if hashlib.sha256(raw_bytes).hexdigest().upper() != receipt_payload.get("sha256"):
                raise DiscoveryError(f"{spec.query_id}: existing snapshot hash mismatch")
            payload = json.loads(raw_bytes.decode("utf-8"))
            retrieved_at = receipt_payload["retrieved_at"]
        else:
            params: dict[str, Any] = {
                "search": spec.query,
                "per-page": per_page,
                "page": 1,
                "sort": "relevance_score:desc",
            }
            if spec.filters:
                params["filter"] = spec.filters
            response = session.get(OPENALEX_ENDPOINT, params=params, timeout=timeout_seconds)
            if response.status_code != 200:
                raise DiscoveryError(
                    f"{spec.query_id}: OpenAlex returned HTTP {response.status_code}: {response.text[:300]}"
                )
            raw_bytes = response.content
            try:
                payload = response.json()
            except requests.JSONDecodeError as exc:
                raise DiscoveryError(f"{spec.query_id}: OpenAlex response is not JSON") from exc
            retrieved_at = datetime.now(timezone.utc).astimezone().isoformat()
            digest = hashlib.sha256(raw_bytes).hexdigest().upper()
            _atomic_write_bytes(snapshot, raw_bytes)
            _atomic_write_bytes(
                receipt,
                (json.dumps(
                    {
                        "schema_version": "1.0",
                        "query_id": spec.query_id,
                        "endpoint": OPENALEX_ENDPOINT,
                        "query": spec.query,
                        "filters": spec.filters,
                        "retrieved_at": retrieved_at,
                        "http_status": response.status_code,
                        "bytes": len(raw_bytes),
                        "sha256": digest,
                    },
                    ensure_ascii=False,
                    indent=2,
                ) + "\n").encode("utf-8"),
            )
            if delay_seconds > 0:
                time.sleep(delay_seconds)
        results = payload.get("results")
        meta = payload.get("meta") or {}
        if not isinstance(results, list) or not isinstance(meta, Mapping):
            raise DiscoveryError(f"{spec.query_id}: malformed OpenAlex payload")
        digest = hashlib.sha256(raw_bytes).hexdigest().upper()
        relative_snapshot = snapshot.relative_to(root).as_posix()
        query_log.append(
            {
                "query_id": spec.query_id,
                "database": "OpenAlex",
                "exact_query": spec.query,
                "searched_at": retrieved_at,
                "result_start": 1,
                "result_end": len(results),
                "raw_result_count": int(meta.get("count") or len(results)),
                "snapshot_path": relative_snapshot,
                "snapshot_sha256": digest,
                "snapshot_bytes": len(raw_bytes),
            }
        )
        snapshots.append({"query_id": spec.query_id, "snapshot_path": str(snapshot)})
    return query_log, snapshots
