from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

from .errors import ErrorCode, SctsrError
from .identity_pool import IdentityPool, IdentityRecord, identity_digest, make_pool
from .rate_spec import IDENTITY_POOL_RATE
from .terminal_field_guard import TerminalFieldGuard


def counter_hash(domain: str, seed: int, *parts: object) -> str:
    payload = "\0".join([domain, str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PoolBuildAudit:
    policy: str
    candidate_count: int
    selected_count: int
    excluded_count: int
    overlap_with_t_count: int
    overlap_with_t_rate: float
    quota_required: Mapping[str, int]
    quota_available: Mapping[str, int]
    terminal_field_guard_digest: str | None
    selection_seed: int
    candidate_universe_digest: str = "NOT_RECORDED"
    excluded_identity_digest: str = "NOT_RECORDED"
    selected_identity_digest: str = "NOT_RECORDED"
    exact_joint_stratum_quota: bool = False


@dataclass(frozen=True, slots=True)
class PoolBuildResult:
    pool: IdentityPool
    audit: PoolBuildAudit


def build_r1_global_random(
    base_records: Sequence[IdentityRecord],
    *,
    base_denominator: int,
    base_manifest_sha256: str,
    source_manifest_sha256: str,
    selection_seed: int,
    t_ids: set[str],
) -> PoolBuildResult:
    eligible = [record for record in base_records if record.base_manifest_membership]
    if len({r.sample_id for r in eligible}) != len(eligible):
        raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "Canonical R1 universe contains duplicate IDs")
    # R1 is global random: T identities are deliberately not excluded.
    count = IDENTITY_POOL_RATE.derive_count(base_denominator)
    if len(eligible) < count:
        raise SctsrError(ErrorCode.R1_UNIVERSE_NOT_GLOBAL, "Global random universe is smaller than required selection")
    selected = sorted(eligible, key=lambda r: (counter_hash("R1_selection", selection_seed, r.sample_id), r.sample_id))[:count]
    overlap = len({r.sample_id for r in selected} & t_ids)
    pool = make_pool(
        pool_id=f"R1_GLOBAL_RANDOM_{selection_seed}",
        pool_role="R1_GLOBAL_RANDOM",
        records=selected,
        base_denominator=base_denominator,
        base_manifest_sha256=base_manifest_sha256,
        source_manifest_path="CANONICAL_BASE",
        source_manifest_sha256=source_manifest_sha256,
        construction_seed=selection_seed,
        selection_semantic="GLOBAL_RANDOM_IDENTITY_NEUTRAL_NOT_ZERO_OVERLAP_COMPARATOR",
    )
    return PoolBuildResult(
        pool,
        PoolBuildAudit(
            policy="R1_GLOBAL_RANDOM",
            candidate_count=len(eligible),
            selected_count=len(selected),
            excluded_count=len(base_records) - len(eligible),
            overlap_with_t_count=overlap,
            overlap_with_t_rate=overlap / len(selected) if selected else 0.0,
            quota_required={},
            quota_available={},
            terminal_field_guard_digest=None,
            selection_seed=selection_seed,
            candidate_universe_digest=identity_digest(eligible),
            excluded_identity_digest=identity_digest(tuple(record for record in base_records if not record.base_manifest_membership)),
            selected_identity_digest=pool.spec.identity_digest,
            exact_joint_stratum_quota=False,
        ),
    )


def _stratum_token(record: IdentityRecord) -> str:
    return "|".join(map(str, record.stratum()))


def build_r2_matched_random(
    raw_rows: Sequence[Mapping[str, object]],
    *,
    t_pool: IdentityPool,
    base_denominator: int,
    base_manifest_sha256: str,
    source_manifest_sha256: str,
    selection_seed: int,
    guard: TerminalFieldGuard | None = None,
) -> PoolBuildResult:
    guard = guard or TerminalFieldGuard()
    projected = guard.project_rows(raw_rows)
    records = [IdentityRecord.from_mapping(row) for row in projected]
    t_ids = {record.sample_id for record in t_pool.records}
    candidates = [record for record in records if record.base_manifest_membership and record.sample_id not in t_ids]
    excluded = [record for record in records if not record.base_manifest_membership or record.sample_id in t_ids]
    if len({r.sample_id for r in candidates}) != len(candidates):
        raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "R2 candidate universe contains duplicate IDs")
    required = Counter(_stratum_token(record) for record in t_pool.records)
    available = Counter(_stratum_token(record) for record in candidates)
    shortages = {stratum: need - available.get(stratum, 0) for stratum, need in required.items() if available.get(stratum, 0) < need}
    if shortages:
        raise SctsrError(
            ErrorCode.R2_QUOTA_INFEASIBLE,
            "At least one exact pre-terminal R2 stratum lacks enough zero-overlap candidates",
            observed=shortages,
            required_action="Provide a new frozen asset or submit a specification change; matching may not relax",
        )
    by_stratum: dict[str, list[IdentityRecord]] = defaultdict(list)
    for record in candidates:
        by_stratum[_stratum_token(record)].append(record)
    selected: list[IdentityRecord] = []
    for stratum in sorted(required):
        ranked = sorted(
            by_stratum[stratum],
            key=lambda r: (counter_hash("R2_stratum_selection", selection_seed, stratum, r.sample_id), r.sample_id),
        )
        selected.extend(ranked[: required[stratum]])
    if {r.sample_id for r in selected} & t_ids:
        raise SctsrError(ErrorCode.R2_OVERLAPS_T, "R2 must have zero identity overlap with T")
    pool = make_pool(
        pool_id=f"R2_MATCHED_RANDOM_{selection_seed}",
        pool_role="R2_MATCHED_RANDOM",
        records=selected,
        base_denominator=base_denominator,
        base_manifest_sha256=base_manifest_sha256,
        source_manifest_path="PRETERMINAL_PROJECTED_BASE",
        source_manifest_sha256=source_manifest_sha256,
        construction_seed=selection_seed,
        selection_semantic="ZERO_OVERLAP_EXACT_PRETERMINAL_MATCHED_RANDOM",
    )
    if Counter(r.stratum() for r in selected) != Counter(r.stratum() for r in t_pool.records):
        raise SctsrError(ErrorCode.R2_QUOTA_INFEASIBLE, "R2 exact quota equality failed after selection")
    return PoolBuildResult(
        pool,
        PoolBuildAudit(
            policy="R2_MATCHED_RANDOM",
            candidate_count=len(candidates),
            selected_count=len(selected),
            excluded_count=len(records) - len(candidates),
            overlap_with_t_count=0,
            overlap_with_t_rate=0.0,
            quota_required=dict(required),
            quota_available=dict(available),
            terminal_field_guard_digest=guard.digest,
            selection_seed=selection_seed,
            candidate_universe_digest=identity_digest(candidates),
            excluded_identity_digest=identity_digest(excluded),
            selected_identity_digest=pool.spec.identity_digest,
            exact_joint_stratum_quota=True,
        ),
    )
