from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

from .errors import ErrorCode, SctsrError
from .identity_pool import IdentityPool, IdentityRecord, identity_digest, make_pool
from .r2_addendum import (
    R2_APPROVED_SELECTION_SEMANTIC,
    R2_MINIMUM_OOF_GROUP_DISPLACEMENT_POLICY,
    R2_RELAXED_FIELDS,
    R2_STRICT_EXACT_POLICY,
)
from .rate_spec import IDENTITY_POOL_RATE
from .serialization import stable_digest
from .terminal_field_guard import TerminalFieldGuard


def counter_hash(domain: str, seed: int, *parts: object) -> str:
    payload = "\0".join([domain, str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class R2GroupDisplacement:
    selected_sample_id: str
    y_true: int
    historical_dynamic_bucket: str
    oof_fold: int
    requested_oof_group_id: str
    selected_oof_group_id: str
    selection_counter_hash: str


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
    exact_coarse_stratum_quota: bool = False
    relaxed_fields: tuple[str, ...] = ()
    displacement_count: int = 0
    displacement_lower_bound: int = 0
    displacement_records: tuple[R2GroupDisplacement, ...] = ()
    displacement_ledger_digest: str = "NOT_APPLICABLE"


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


def _coarse_key(record: IdentityRecord) -> tuple[int, str, int]:
    return record.stratum()[:3]


def _key_token(key: Sequence[object]) -> str:
    return "|".join(map(str, key))


def build_minimum_oof_group_displacement_selection(
    candidates: Sequence[IdentityRecord],
    treatment_records: Sequence[IdentityRecord],
    *,
    selection_seed: int,
) -> tuple[tuple[IdentityRecord, ...], tuple[R2GroupDisplacement, ...], int]:
    """Select the capacity-optimal zero-overlap R2 under the approved addendum.

    Every available exact four-field cell is exhausted first. Only the
    unavoidable deficit is filled inside the same label/dynamic/fold cell.
    Pairing each fill row to one requested group slot makes all relaxed
    occurrences explicit and independently auditable.
    """

    required = Counter(record.stratum() for record in treatment_records)
    by_strict: dict[tuple[int, str, int, str], list[IdentityRecord]] = defaultdict(list)
    for record in candidates:
        by_strict[record.stratum()].append(record)

    selected: list[IdentityRecord] = []
    selected_ids: set[str] = set()
    shortages: dict[tuple[int, str, int, str], int] = {}
    for key in sorted(required, key=_key_token):
        ranked = sorted(
            by_strict.get(key, ()),
            key=lambda record: (
                counter_hash("R2_exact_capacity", selection_seed, _key_token(key), record.sample_id),
                record.sample_id,
            ),
        )
        take = min(required[key], len(ranked))
        exact_rows = ranked[:take]
        selected.extend(exact_rows)
        selected_ids.update(record.sample_id for record in exact_rows)
        if take < required[key]:
            shortages[key] = required[key] - take

    required_coarse = Counter(_coarse_key(record) for record in treatment_records)
    selected_coarse = Counter(_coarse_key(record) for record in selected)
    remaining_by_coarse: dict[tuple[int, str, int], list[IdentityRecord]] = defaultdict(list)
    for record in candidates:
        if record.sample_id not in selected_ids:
            remaining_by_coarse[_coarse_key(record)].append(record)

    displacement_records: list[R2GroupDisplacement] = []
    for coarse in sorted(required_coarse, key=_key_token):
        deficit = required_coarse[coarse] - selected_coarse.get(coarse, 0)
        if deficit <= 0:
            continue
        requested_slots = [
            key
            for key in sorted(shortages, key=_key_token)
            if key[:3] == coarse
            for _ in range(shortages[key])
        ]
        if len(requested_slots) != deficit:
            raise SctsrError(
                ErrorCode.R2_QUOTA_INFEASIBLE,
                "R2 relaxed-slot accounting differs from the exact-cell capacity deficit",
                observed=len(requested_slots),
                expected=deficit,
            )
        ranked = sorted(
            remaining_by_coarse.get(coarse, ()),
            key=lambda record: (
                counter_hash("R2_minimum_displacement_fill", selection_seed, _key_token(coarse), record.sample_id),
                record.sample_id,
            ),
        )
        if len(ranked) < deficit:
            raise SctsrError(
                ErrorCode.R2_QUOTA_INFEASIBLE,
                "Approved R2 addendum is infeasible even after relaxing only oof_group_id",
                observed={_key_token(coarse): deficit - len(ranked)},
                required_action="Stop; no second field may be relaxed without another owner addendum.",
            )
        fills = ranked[:deficit]
        for requested, record in zip(requested_slots, fills, strict=True):
            if record.oof_group_id == requested[3]:
                raise SctsrError(
                    ErrorCode.R2_QUOTA_INFEASIBLE,
                    "A purported displacement remained inside its exact group cell",
                    observed=record.sample_id,
                )
            displacement_records.append(
                R2GroupDisplacement(
                    selected_sample_id=record.sample_id,
                    y_true=record.y_true,
                    historical_dynamic_bucket=record.historical_dynamic_bucket,
                    oof_fold=record.oof_fold,
                    requested_oof_group_id=str(requested[3]),
                    selected_oof_group_id=record.oof_group_id,
                    selection_counter_hash=counter_hash(
                        "R2_minimum_displacement_fill", selection_seed, _key_token(coarse), record.sample_id
                    ),
                )
            )
        selected.extend(fills)
        selected_ids.update(record.sample_id for record in fills)
        selected_coarse[coarse] += deficit

    selected_tuple = tuple(selected)
    displacement_tuple = tuple(displacement_records)
    minimum = sum(shortages.values())
    if len(selected_tuple) != len(treatment_records) or len(selected_ids) != len(selected_tuple):
        raise SctsrError(ErrorCode.R2_QUOTA_INFEASIBLE, "Approved R2 addendum does not conserve unique count")
    if Counter(_coarse_key(record) for record in selected_tuple) != required_coarse:
        raise SctsrError(ErrorCode.R2_QUOTA_INFEASIBLE, "Approved R2 addendum changed an exact three-field quota")
    selected_strict = Counter(record.stratum() for record in selected_tuple)
    observed_displacement = sum(
        abs(required.get(key, 0) - selected_strict.get(key, 0)) for key in set(required) | set(selected_strict)
    ) // 2
    if observed_displacement != minimum or len(displacement_tuple) != minimum:
        raise SctsrError(
            ErrorCode.R2_QUOTA_INFEASIBLE,
            "Approved R2 construction did not attain the exact-cell capacity lower bound",
            observed={"distance": observed_displacement, "ledger_rows": len(displacement_tuple)},
            expected=minimum,
        )
    return selected_tuple, displacement_tuple, minimum


def build_r2_matched_random(
    raw_rows: Sequence[Mapping[str, object]],
    *,
    t_pool: IdentityPool,
    base_denominator: int,
    base_manifest_sha256: str,
    source_manifest_sha256: str,
    selection_seed: int,
    guard: TerminalFieldGuard | None = None,
    matching_policy: str = R2_STRICT_EXACT_POLICY,
) -> PoolBuildResult:
    if matching_policy not in {R2_STRICT_EXACT_POLICY, R2_MINIMUM_OOF_GROUP_DISPLACEMENT_POLICY}:
        raise SctsrError(
            ErrorCode.CONFIGURATION_MISMATCH,
            "R2 matching policy is not registered",
            observed=matching_policy,
        )
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
    if shortages and matching_policy == R2_STRICT_EXACT_POLICY:
        raise SctsrError(
            ErrorCode.R2_QUOTA_INFEASIBLE,
            "At least one exact pre-terminal R2 stratum lacks enough zero-overlap candidates",
            observed=shortages,
            required_action="Provide a new frozen asset or submit a specification change; matching may not relax",
        )
    displacement_records: tuple[R2GroupDisplacement, ...] = ()
    displacement_lower_bound = 0
    if matching_policy == R2_STRICT_EXACT_POLICY:
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
        selection_semantic = "ZERO_OVERLAP_EXACT_PRETERMINAL_MATCHED_RANDOM"
    else:
        selected_tuple, displacement_records, displacement_lower_bound = build_minimum_oof_group_displacement_selection(
            candidates,
            t_pool.records,
            selection_seed=selection_seed,
        )
        selected = list(selected_tuple)
        selection_semantic = R2_APPROVED_SELECTION_SEMANTIC
    if {r.sample_id for r in selected} & t_ids:
        raise SctsrError(ErrorCode.R2_OVERLAPS_T, "R2 must have zero identity overlap with T")
    pool = make_pool(
        pool_id=f"R2_MATCHED_RANDOM_{matching_policy}_{selection_seed}",
        pool_role="R2_MATCHED_RANDOM",
        records=selected,
        base_denominator=base_denominator,
        base_manifest_sha256=base_manifest_sha256,
        source_manifest_path="PRETERMINAL_PROJECTED_BASE",
        source_manifest_sha256=source_manifest_sha256,
        construction_seed=selection_seed,
        selection_semantic=selection_semantic,
    )
    exact_joint = Counter(r.stratum() for r in selected) == Counter(r.stratum() for r in t_pool.records)
    exact_coarse = Counter(_coarse_key(r) for r in selected) == Counter(_coarse_key(r) for r in t_pool.records)
    if matching_policy == R2_STRICT_EXACT_POLICY and not exact_joint:
        raise SctsrError(ErrorCode.R2_QUOTA_INFEASIBLE, "R2 exact quota equality failed after selection")
    if matching_policy == R2_MINIMUM_OOF_GROUP_DISPLACEMENT_POLICY and (exact_joint or not exact_coarse):
        raise SctsrError(ErrorCode.R2_QUOTA_INFEASIBLE, "Approved R2 addendum changed more or less than oof_group_id")
    displacement_digest = stable_digest(displacement_records) if displacement_records else "NOT_APPLICABLE"
    return PoolBuildResult(
        pool,
        PoolBuildAudit(
            policy=matching_policy,
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
            exact_joint_stratum_quota=exact_joint,
            exact_coarse_stratum_quota=exact_coarse,
            relaxed_fields=R2_RELAXED_FIELDS if displacement_records else (),
            displacement_count=len(displacement_records),
            displacement_lower_bound=displacement_lower_bound,
            displacement_records=displacement_records,
            displacement_ledger_digest=displacement_digest,
        ),
    )
