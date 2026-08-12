"""Deterministic, fail-closed tier selection for the Stage1 literature corpus."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import re
from typing import Iterable, Mapping, Sequence


RQS = tuple(f"RQ{index}" for index in range(1, 9))
EFFECT_RELATIONS = {"SUPPORTED", "NULL_NEGATIVE", "MIXED", "METHOD_ONLY"}
DIRECTNESS_BY_RELEVANCE = {
    "DIRECT_INTERVENTION": "D1_DIRECT_UTILITY",
    "STRICT_CONTROL_NEGATIVE": "D1_DIRECT_UTILITY",
    "DIRECT_MECHANISM": "D2_DIRECT_MECHANISM",
    "TARGET_METRIC": "D3_INFERENCE_CONTROL",
    "TRANSFER_COMPONENT": "D4_TRANSFER_ANALOG",
}
DIRECTNESS_RANK = {
    "D1_DIRECT_UTILITY": 1,
    "D2_DIRECT_MECHANISM": 2,
    "D3_INFERENCE_CONTROL": 3,
    "D4_TRANSFER_ANALOG": 4,
}
MERGE_BASES = {"EXACT_METADATA", "VERSION_IDENTITY"}
SHARED_IDENTITY_PATTERN = re.compile(
    r"^(?:ARXIV|DOI|OPENREVIEW|PMID|ISBN):[^\s|]+(?:\|(?:ARXIV|DOI|OPENREVIEW|PMID|ISBN):[^\s|]+)*$",
    re.IGNORECASE,
)


class TierSelectionError(RuntimeError):
    """Raised when a tier cannot be frozen without violating its contract."""


@dataclass(frozen=True)
class BroadCandidate:
    queue_id: str
    canonical_work_id: str
    title: str
    authors: str
    year: int
    direct_rqs: tuple[str, ...]
    relevance_class: str
    citation_count: int = 0
    doi: str = ""
    effect_relation: str = "SUPPORTED"
    merged_queue_ids: tuple[str, ...] = ()

    @property
    def directness(self) -> str:
        try:
            return DIRECTNESS_BY_RELEVANCE[self.relevance_class]
        except KeyError as exc:
            raise TierSelectionError(
                f"unsupported relevance class for {self.queue_id}: {self.relevance_class}"
            ) from exc


@dataclass(frozen=True)
class ExplicitMerge:
    alias_queue_id: str
    canonical_queue_id: str
    evidence: str
    merge_basis: str = "EXACT_METADATA"
    shared_identity: str = ""


@dataclass(frozen=True)
class TierSelectionPolicy:
    total: int = 500
    minimum_per_rq: int = 40
    maximum_per_rq: int = 100
    maximum_transfer: int = 100
    minimum_counterevidence_per_rq: int = 0
    mandatory_canonical_work_ids: tuple[str, ...] = ()
    tier_label: str = "BROAD"
    frozen_seed: str = "stage1-literature-tier-freeze-v2-20260810"

    def __post_init__(self) -> None:
        if self.total < 1:
            raise ValueError("total must be positive")
        if not 0 <= self.minimum_per_rq <= self.maximum_per_rq:
            raise ValueError("invalid RQ minimum/maximum")
        if self.minimum_per_rq * len(RQS) > self.total:
            raise ValueError("RQ minima exceed tier total")
        if self.maximum_per_rq * len(RQS) < self.total:
            raise ValueError("RQ maxima cannot hold tier total")
        if not 0 <= self.maximum_transfer <= self.total:
            raise ValueError("invalid transfer maximum")
        if not 0 <= self.minimum_counterevidence_per_rq <= self.minimum_per_rq:
            raise ValueError("invalid counterevidence minimum")
        if self.minimum_counterevidence_per_rq * len(RQS) > self.total:
            raise ValueError("counterevidence minima exceed tier total")
        if len(self.mandatory_canonical_work_ids) > self.total:
            raise ValueError("mandatory anchors exceed tier total")
        if len(set(self.mandatory_canonical_work_ids)) != len(
            self.mandatory_canonical_work_ids
        ):
            raise ValueError("mandatory canonical work IDs must be unique")
        if any(not work_id.strip() for work_id in self.mandatory_canonical_work_ids):
            raise ValueError("mandatory canonical work IDs cannot be empty")
        if self.tier_label not in {"BROAD", "SCREENED", "DEEP"}:
            raise ValueError("tier_label must be BROAD, SCREENED, or DEEP")
        if not self.frozen_seed:
            raise ValueError("frozen_seed is required")


@dataclass(frozen=True)
class SelectedBroadCandidate:
    candidate: BroadCandidate
    quota_rq: str
    directness: str
    tie_break_key: str
    selection_phase: str


@dataclass(frozen=True)
class BroadSelectionResult:
    selected: tuple[SelectedBroadCandidate, ...]
    reserves: tuple[BroadCandidate, ...]
    quota_counts: Mapping[str, int]


def _normalize_identity_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _normalized_dois(value: str) -> tuple[str, ...]:
    dois: set[str] = set()
    for raw in value.split(";"):
        clean = raw.strip().casefold()
        clean = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", clean).rstrip("/")
        if not clean or clean.startswith("not_"):
            continue
        dois.add(clean)
    return tuple(sorted(dois))


def _normalized_author_set(value: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            normalized
            for raw in value.split(";")
            if (normalized := _normalize_identity_text(raw))
        )
    )


def _tie_key(candidate: BroadCandidate, policy: TierSelectionPolicy) -> str:
    payload = f"{candidate.canonical_work_id}|{policy.tier_label}|{policy.frozen_seed}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def _candidate_order(candidate: BroadCandidate, policy: TierSelectionPolicy) -> tuple[int, str]:
    return DIRECTNESS_RANK[candidate.directness], _tie_key(candidate, policy)


def _validate_candidate(candidate: BroadCandidate) -> None:
    if not candidate.queue_id or not candidate.canonical_work_id:
        raise TierSelectionError("candidate queue_id and canonical_work_id are required")
    if not candidate.title.strip() or not candidate.authors.strip():
        raise TierSelectionError(f"candidate identity incomplete: {candidate.queue_id}")
    if candidate.relevance_class not in DIRECTNESS_BY_RELEVANCE:
        raise TierSelectionError(
            f"unsupported relevance class for {candidate.queue_id}: {candidate.relevance_class}"
        )
    if candidate.effect_relation not in EFFECT_RELATIONS:
        raise TierSelectionError(
            f"unsupported effect relation for {candidate.queue_id}: {candidate.effect_relation}"
        )
    rqs = set(candidate.direct_rqs)
    if not rqs or not rqs.issubset(RQS):
        raise TierSelectionError(f"invalid direct RQs for {candidate.queue_id}: {candidate.direct_rqs}")


def canonicalize_candidates(
    candidates: Iterable[BroadCandidate],
    *,
    merges: Sequence[ExplicitMerge],
) -> tuple[BroadCandidate, ...]:
    """Apply explicit version merges and reject unresolved identity collisions."""

    by_queue: dict[str, BroadCandidate] = {}
    for candidate in candidates:
        _validate_candidate(candidate)
        if candidate.queue_id in by_queue:
            raise TierSelectionError(f"duplicate queue ID: {candidate.queue_id}")
        by_queue[candidate.queue_id] = candidate

    aliases: dict[str, ExplicitMerge] = {}
    for merge in merges:
        if not merge.evidence.strip():
            raise TierSelectionError(f"merge evidence is empty for {merge.alias_queue_id}")
        if merge.alias_queue_id == merge.canonical_queue_id:
            raise TierSelectionError(f"self-merge is invalid: {merge.alias_queue_id}")
        if merge.alias_queue_id in aliases:
            raise TierSelectionError(f"duplicate merge alias: {merge.alias_queue_id}")
        if merge.merge_basis not in MERGE_BASES:
            raise TierSelectionError(
                f"unsupported merge basis for {merge.alias_queue_id}: {merge.merge_basis}"
            )
        if merge.merge_basis == "VERSION_IDENTITY" and not SHARED_IDENTITY_PATTERN.fullmatch(
            merge.shared_identity.strip()
        ):
            raise TierSelectionError(
                f"version merge shared identity is invalid for {merge.alias_queue_id}"
            )
        if merge.alias_queue_id not in by_queue or merge.canonical_queue_id not in by_queue:
            raise TierSelectionError(
                f"merge references an unknown queue ID: {merge.alias_queue_id} -> {merge.canonical_queue_id}"
            )
        aliases[merge.alias_queue_id] = merge

    for alias_id, merge in aliases.items():
        alias = by_queue[alias_id]
        canonical = by_queue[merge.canonical_queue_id]
        if merge.merge_basis == "EXACT_METADATA" and _normalize_identity_text(
            alias.title
        ) != _normalize_identity_text(canonical.title):
            raise TierSelectionError(
                f"explicit merge title mismatch: {alias_id} -> {merge.canonical_queue_id}"
            )
        if _normalized_author_set(alias.authors) != _normalized_author_set(canonical.authors):
            raise TierSelectionError(
                f"explicit merge author mismatch: {alias_id} -> {merge.canonical_queue_id}"
            )
        if merge.merge_basis == "EXACT_METADATA" and alias.year != canonical.year:
            raise TierSelectionError(
                f"explicit merge year mismatch: {alias_id} -> {merge.canonical_queue_id}"
            )
        by_queue[merge.canonical_queue_id] = replace(
            canonical,
            direct_rqs=tuple(sorted(set(canonical.direct_rqs) | set(alias.direct_rqs))),
            effect_relation=(
                canonical.effect_relation
                if canonical.effect_relation == alias.effect_relation
                else "MIXED"
            ),
            merged_queue_ids=tuple(sorted(set(canonical.merged_queue_ids) | {alias_id})),
        )
        del by_queue[alias_id]

    title_groups: dict[str, list[str]] = {}
    for candidate in by_queue.values():
        title_groups.setdefault(_normalize_identity_text(candidate.title), []).append(candidate.queue_id)
    unresolved = [ids for ids in title_groups.values() if len(ids) > 1]
    if unresolved:
        raise TierSelectionError(
            "duplicate normalized titles require explicit merge evidence: "
            + "; ".join(",".join(sorted(ids)) for ids in unresolved)
        )
    doi_groups: dict[str, list[str]] = {}
    for candidate in by_queue.values():
        for doi in _normalized_dois(candidate.doi):
            doi_groups.setdefault(doi, []).append(candidate.queue_id)
    duplicate_dois = [ids for ids in doi_groups.values() if len(ids) > 1]
    if duplicate_dois:
        raise TierSelectionError(
            "duplicate DOI identities require explicit merge evidence: "
            + "; ".join(",".join(sorted(ids)) for ids in duplicate_dois)
        )
    return tuple(sorted(by_queue.values(), key=lambda item: item.canonical_work_id))


def _choose_quota_rq(
    candidate: BroadCandidate,
    counts: Mapping[str, int],
    policy: TierSelectionPolicy,
) -> str | None:
    available = [rq for rq in candidate.direct_rqs if counts[rq] < policy.maximum_per_rq]
    if not available:
        return None
    return min(
        available,
        key=lambda rq: (
            counts[rq],
            hashlib.sha256(
                f"{candidate.canonical_work_id}|{rq}|{policy.frozen_seed}".encode("utf-8")
            ).hexdigest(),
        ),
    )


def select_broad_candidates(
    candidates: Iterable[BroadCandidate],
    policy: TierSelectionPolicy = TierSelectionPolicy(),
) -> BroadSelectionResult:
    """Select the exact BROAD tier by preregistered lexicographic rules."""

    materialized = tuple(candidates)
    if len(materialized) < policy.total:
        raise TierSelectionError(
            f"eligible canonical candidates {len(materialized)} are fewer than required {policy.total}"
        )
    seen_work_ids: set[str] = set()
    for candidate in materialized:
        _validate_candidate(candidate)
        if candidate.canonical_work_id in seen_work_ids:
            raise TierSelectionError(f"duplicate canonical work ID: {candidate.canonical_work_id}")
        seen_work_ids.add(candidate.canonical_work_id)

    availability = {rq: sum(rq in item.direct_rqs for item in materialized) for rq in RQS}
    for rq, count in availability.items():
        if count < policy.minimum_per_rq:
            raise TierSelectionError(
                f"{rq} has only {count} eligible candidates; minimum is {policy.minimum_per_rq}"
            )

    ordered = sorted(materialized, key=lambda item: _candidate_order(item, policy))
    rq_order = sorted(RQS, key=lambda rq: (availability[rq], rq))
    counts = {rq: 0 for rq in RQS}
    selected_by_id: dict[str, SelectedBroadCandidate] = {}
    transfer_count = 0

    candidate_by_work_id = {
        candidate.canonical_work_id: candidate for candidate in materialized
    }
    missing_mandatory = sorted(
        set(policy.mandatory_canonical_work_ids) - set(candidate_by_work_id)
    )
    if missing_mandatory:
        raise TierSelectionError(
            "mandatory anchor is absent from the eligible canonical pool: "
            + ",".join(missing_mandatory)
        )
    mandatory_candidates = sorted(
        (
            candidate_by_work_id[work_id]
            for work_id in policy.mandatory_canonical_work_ids
        ),
        key=lambda item: _candidate_order(item, policy),
    )
    mandatory_transfer_count = sum(
        candidate.directness == "D4_TRANSFER_ANALOG"
        for candidate in mandatory_candidates
    )
    if mandatory_transfer_count > policy.maximum_transfer:
        raise TierSelectionError(
            "mandatory anchor set exceeds the transfer cap: "
            f"{mandatory_transfer_count}>{policy.maximum_transfer}"
        )
    for candidate in mandatory_candidates:
        quota_rq = _choose_quota_rq(candidate, counts, policy)
        if quota_rq is None:
            raise TierSelectionError(
                "mandatory anchor cannot be assigned without exceeding an RQ maximum: "
                f"{candidate.canonical_work_id}"
            )
        selected_by_id[candidate.canonical_work_id] = SelectedBroadCandidate(
            candidate=candidate,
            quota_rq=quota_rq,
            directness=candidate.directness,
            tie_break_key=_tie_key(candidate, policy),
            selection_phase="MANDATORY_ANCHOR",
        )
        counts[quota_rq] += 1
        if candidate.directness == "D4_TRANSFER_ANALOG":
            transfer_count += 1

    counter_relations = {"NULL_NEGATIVE", "MIXED"}
    if policy.minimum_counterevidence_per_rq:
        counter_availability = {
            rq: sum(
                rq in item.direct_rqs and item.effect_relation in counter_relations
                for item in materialized
            )
            for rq in RQS
        }
        counter_rq_order = sorted(RQS, key=lambda rq: (counter_availability[rq], rq))
        for rq in counter_rq_order:
            counter_count = sum(
                selected.quota_rq == rq
                and selected.candidate.effect_relation in counter_relations
                for selected in selected_by_id.values()
            )
            for candidate in ordered:
                if counter_count >= policy.minimum_counterevidence_per_rq:
                    break
                if (
                    candidate.canonical_work_id in selected_by_id
                    or rq not in candidate.direct_rqs
                    or candidate.effect_relation not in counter_relations
                ):
                    continue
                if (
                    candidate.directness == "D4_TRANSFER_ANALOG"
                    and transfer_count >= policy.maximum_transfer
                ):
                    continue
                selected_by_id[candidate.canonical_work_id] = SelectedBroadCandidate(
                    candidate=candidate,
                    quota_rq=rq,
                    directness=candidate.directness,
                    tie_break_key=_tie_key(candidate, policy),
                    selection_phase="COUNTEREVIDENCE_MINIMUM",
                )
                counts[rq] += 1
                counter_count += 1
                if candidate.directness == "D4_TRANSFER_ANALOG":
                    transfer_count += 1
            if counter_count < policy.minimum_counterevidence_per_rq:
                raise TierSelectionError(
                    f"{rq} has insufficient counterevidence after identity and transfer constraints"
                )

    for rq in rq_order:
        for candidate in ordered:
            if counts[rq] >= policy.minimum_per_rq:
                break
            if candidate.canonical_work_id in selected_by_id or rq not in candidate.direct_rqs:
                continue
            if candidate.directness == "D4_TRANSFER_ANALOG" and transfer_count >= policy.maximum_transfer:
                continue
            selected_by_id[candidate.canonical_work_id] = SelectedBroadCandidate(
                candidate=candidate,
                quota_rq=rq,
                directness=candidate.directness,
                tie_break_key=_tie_key(candidate, policy),
                selection_phase="RQ_MINIMUM",
            )
            counts[rq] += 1
            if candidate.directness == "D4_TRANSFER_ANALOG":
                transfer_count += 1
        if counts[rq] < policy.minimum_per_rq:
            raise TierSelectionError(
                f"{rq} minimum cannot be met after respecting identity and transfer constraints"
            )

    for candidate in ordered:
        if len(selected_by_id) >= policy.total:
            break
        if candidate.canonical_work_id in selected_by_id:
            continue
        if candidate.directness == "D4_TRANSFER_ANALOG" and transfer_count >= policy.maximum_transfer:
            continue
        quota_rq = _choose_quota_rq(candidate, counts, policy)
        if quota_rq is None:
            continue
        selected_by_id[candidate.canonical_work_id] = SelectedBroadCandidate(
            candidate=candidate,
            quota_rq=quota_rq,
            directness=candidate.directness,
            tie_break_key=_tie_key(candidate, policy),
            selection_phase="GLOBAL_FILL",
        )
        counts[quota_rq] += 1
        if candidate.directness == "D4_TRANSFER_ANALOG":
            transfer_count += 1

    if len(selected_by_id) != policy.total:
        raise TierSelectionError(
            f"could select only {len(selected_by_id)} of required {policy.total} candidates"
        )
    for rq, count in counts.items():
        if not policy.minimum_per_rq <= count <= policy.maximum_per_rq:
            raise TierSelectionError(f"{rq} quota out of bounds: {count}")

    selected = tuple(
        sorted(
            selected_by_id.values(),
            key=lambda item: item.tie_break_key,
        )
    )
    selected_ids = set(selected_by_id)
    reserves = tuple(
        candidate for candidate in ordered if candidate.canonical_work_id not in selected_ids
    )
    return BroadSelectionResult(selected=selected, reserves=reserves, quota_counts=counts)
