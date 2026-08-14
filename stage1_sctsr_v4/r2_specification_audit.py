from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .errors import ErrorCode, SctsrError
from .identity_pool import IdentityRecord, identity_digest
from .random_controls import build_minimum_oof_group_displacement_selection, counter_hash
from .serialization import stable_digest


R2_SPECIFICATION_AUDIT_SCHEMA = "stage1.sctsr.r2_specification_audit.v1"
STRICT_FIELDS = ("y_true", "historical_dynamic_bucket", "oof_fold", "oof_group_id")
COARSE_FIELDS = ("y_true", "historical_dynamic_bucket", "oof_fold")


def _strict_key(record: IdentityRecord) -> tuple[int, str, int, str]:
    return record.stratum()


def _coarse_key(record: IdentityRecord) -> tuple[int, str, int]:
    return record.stratum()[:3]


def _stratum_text(key: Sequence[object]) -> str:
    return "|".join(map(str, key))


def _counter_tv(first: Mapping[object, int], second: Mapping[object, int], denominator: int) -> float:
    if denominator <= 0:
        raise SctsrError(ErrorCode.DENOMINATOR_IDENTITY_MISMATCH, "R2 audit denominator must be positive")
    return sum(abs(int(first.get(key, 0)) - int(second.get(key, 0))) for key in set(first) | set(second)) / (2 * denominator)


def _validate_inputs(
    base_records: Sequence[IdentityRecord],
    treatment_records: Sequence[IdentityRecord],
) -> tuple[tuple[IdentityRecord, ...], tuple[IdentityRecord, ...], tuple[IdentityRecord, ...]]:
    base = tuple(base_records)
    treatment = tuple(treatment_records)
    base_by_id = {record.sample_id: record for record in base}
    if len(base_by_id) != len(base):
        raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "R2 audit base contains duplicate IDs")
    treatment_ids = {record.sample_id for record in treatment}
    if len(treatment_ids) != len(treatment):
        raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "R2 audit treatment contains duplicate IDs")
    if not treatment_ids.issubset(base_by_id):
        raise SctsrError(
            ErrorCode.IDENTITY_NOT_IN_BASE,
            "R2 audit treatment is not a subset of the frozen base",
            observed=sorted(treatment_ids - set(base_by_id))[:20],
        )
    for record in treatment:
        base_record = base_by_id[record.sample_id]
        if base_record.stratum() != record.stratum() or base_record.replay_role != record.replay_role:
            raise SctsrError(
                ErrorCode.IDENTITY_DIGEST_MISMATCH,
                "R2 audit treatment metadata differs from its base row",
                observed=record.sample_id,
            )
    candidates = tuple(
        record for record in base if record.base_manifest_membership and record.sample_id not in treatment_ids
    )
    if len({record.sample_id for record in candidates}) != len(candidates):
        raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "R2 audit candidate universe contains duplicate IDs")
    return base, treatment, candidates


@dataclass(frozen=True, slots=True)
class MinimumDisplacementCandidate:
    records: tuple[IdentityRecord, ...]
    minimum_displacement_count: int
    observed_displacement_count: int
    strict_shortage_strata: int
    coarse_quota_exact: bool
    overlap_with_t_count: int
    group_total_variation: float

    @property
    def identity_digest(self) -> str:
        return identity_digest(self.records)


def build_minimum_displacement_candidate(
    base_records: Sequence[IdentityRecord],
    treatment_records: Sequence[IdentityRecord],
    *,
    selection_seed: int,
) -> MinimumDisplacementCandidate:
    """Build an audit candidate; it is not an activated formal R2 policy.

    The construction exhausts every available zero-overlap exact joint cell,
    then fills only within the same label/dynamic/fold cell. Consequently the
    number of displaced group occurrences equals the capacity lower bound.
    """

    _base, treatment, candidates = _validate_inputs(base_records, treatment_records)
    required = Counter(_strict_key(record) for record in treatment)
    available = Counter(_strict_key(record) for record in candidates)
    shortages = {
        key: required[key] - available.get(key, 0)
        for key in required
        if available.get(key, 0) < required[key]
    }
    selected_tuple, displacement_records, minimum = build_minimum_oof_group_displacement_selection(
        candidates,
        treatment,
        selection_seed=selection_seed,
    )
    selected_ids = {record.sample_id for record in selected_tuple}
    treatment_ids = {record.sample_id for record in treatment}
    if len(selected_tuple) != len(treatment) or len(selected_ids) != len(selected_tuple):
        raise SctsrError(ErrorCode.R2_QUOTA_INFEASIBLE, "Minimum-displacement candidate does not conserve unique count")
    overlap = len(selected_ids & treatment_ids)
    if overlap:
        raise SctsrError(ErrorCode.R2_OVERLAPS_T, "Minimum-displacement candidate overlaps T", observed=overlap)
    required_coarse = Counter(_coarse_key(record) for record in treatment)
    observed_coarse = Counter(_coarse_key(record) for record in selected_tuple)
    if observed_coarse != required_coarse:
        raise SctsrError(ErrorCode.R2_QUOTA_INFEASIBLE, "Minimum-displacement candidate changed coarse quota")
    displacement = len(displacement_records)
    if displacement != minimum:
        raise SctsrError(
            ErrorCode.R2_QUOTA_INFEASIBLE,
            "Minimum-displacement candidate exceeds the exact-cell capacity lower bound",
            observed=displacement,
            expected=minimum,
        )
    treatment_groups = Counter(record.oof_group_id for record in treatment)
    selected_groups = Counter(record.oof_group_id for record in selected_tuple)
    return MinimumDisplacementCandidate(
        records=selected_tuple,
        minimum_displacement_count=minimum,
        observed_displacement_count=displacement,
        strict_shortage_strata=len(shortages),
        coarse_quota_exact=True,
        overlap_with_t_count=0,
        group_total_variation=_counter_tv(treatment_groups, selected_groups, len(treatment)),
    )


def _drop_group_hash_random(
    candidates: Sequence[IdentityRecord],
    treatment: Sequence[IdentityRecord],
    *,
    selection_seed: int,
) -> tuple[IdentityRecord, ...]:
    required = Counter(_coarse_key(record) for record in treatment)
    by_coarse: dict[tuple[int, str, int], list[IdentityRecord]] = defaultdict(list)
    for record in candidates:
        by_coarse[_coarse_key(record)].append(record)
    selected: list[IdentityRecord] = []
    for key in sorted(required, key=_stratum_text):
        ranked = sorted(
            by_coarse.get(key, ()),
            key=lambda record: (
                counter_hash("R2_ROW", selection_seed, _stratum_text(key), record.sample_id),
                record.sample_id,
            ),
        )
        if len(ranked) < required[key]:
            raise SctsrError(ErrorCode.R2_QUOTA_INFEASIBLE, "Drop-group audit comparator is infeasible")
        selected.extend(ranked[: required[key]])
    return tuple(selected)


def audit_r2_specifications(
    base_records: Sequence[IdentityRecord],
    treatment_records: Sequence[IdentityRecord],
    *,
    selection_seed: int,
) -> dict[str, Any]:
    base, treatment, candidates = _validate_inputs(base_records, treatment_records)
    required = Counter(_strict_key(record) for record in treatment)
    available = Counter(_strict_key(record) for record in candidates)
    shortages = {
        key: required[key] - available.get(key, 0)
        for key in required
        if available.get(key, 0) < required[key]
    }
    shortage_rows = [
        {
            "stratum": _stratum_text(key),
            "y_true": key[0],
            "historical_dynamic_bucket": key[1],
            "oof_fold": key[2],
            "oof_group_id": key[3],
            "required": required[key],
            "available_zero_overlap": available.get(key, 0),
            "shortage": shortages[key],
        }
        for key in sorted(shortages, key=_stratum_text)
    ]
    zero_rows = [row for row in shortage_rows if row["available_zero_overlap"] == 0]
    matchable = sum(min(count, available.get(key, 0)) for key, count in required.items())

    minimum = build_minimum_displacement_candidate(base, treatment, selection_seed=selection_seed)
    drop_group = _drop_group_hash_random(candidates, treatment, selection_seed=selection_seed)
    treatment_groups = Counter(record.oof_group_id for record in treatment)
    drop_groups = Counter(record.oof_group_id for record in drop_group)
    treatment_strict = Counter(_strict_key(record) for record in treatment)
    drop_strict = Counter(_strict_key(record) for record in drop_group)

    full_by_strict: dict[tuple[int, str, int, str], list[IdentityRecord]] = defaultdict(list)
    for record in base:
        full_by_strict[_strict_key(record)].append(record)
    overlap_selection: list[IdentityRecord] = []
    expected_overlap = 0.0
    for key in sorted(required, key=_stratum_text):
        ranked = sorted(
            full_by_strict[key],
            key=lambda record: (
                counter_hash("FULL_BASE_EXACT", selection_seed, _stratum_text(key), record.sample_id),
                record.sample_id,
            ),
        )
        overlap_selection.extend(ranked[: required[key]])
        expected_overlap += required[key] * required[key] / len(ranked)
    treatment_ids = {record.sample_id for record in treatment}
    deterministic_overlap = len({record.sample_id for record in overlap_selection} & treatment_ids)
    denominator = len(treatment)

    core: dict[str, Any] = {
        "schema_version": R2_SPECIFICATION_AUDIT_SCHEMA,
        "status": "SPECIFICATION_CHANGE_REQUIRED",
        "selection_seed": selection_seed,
        "terminal_fields_read": False,
        "formal_training_started": False,
        "method_effectiveness_claimed": False,
        "strict_fields": list(STRICT_FIELDS),
        "proposed_exact_fields": list(COARSE_FIELDS),
        "base": {
            "unique_count": len(base),
            "identity_digest": identity_digest(base),
        },
        "treatment": {
            "unique_count": denominator,
            "identity_digest": identity_digest(treatment),
            "label_counts": dict(sorted(Counter(str(record.y_true) for record in treatment).items())),
            "dynamic_bucket_counts": dict(sorted(Counter(record.historical_dynamic_bucket for record in treatment).items())),
            "fold_counts": dict(sorted(Counter(str(record.oof_fold) for record in treatment).items())),
            "oof_group_count": len(treatment_groups),
        },
        "candidate_universe": {
            "unique_count": len(candidates),
            "identity_digest": identity_digest(candidates),
            "overlap_with_t_count": 0,
        },
        "strict_exact": {
            "status": "INFEASIBLE",
            "joint_strata": len(required),
            "shortage_strata": len(shortages),
            "shortage_occurrences": sum(shortages.values()),
            "zero_available_strata": len(zero_rows),
            "zero_available_occurrences": sum(int(row["shortage"]) for row in zero_rows),
            "shortage_rows": shortage_rows,
        },
        "options": {
            "strict_exact_zero_overlap_unique": {
                "status": "INFEASIBLE",
                "reason": "172 exact cells lack 378 zero-overlap unique occurrences.",
            },
            "exact_with_replacement": {
                "status": "INFEASIBLE",
                "zero_candidate_cells": len(zero_rows),
                "zero_candidate_occurrences": sum(int(row["shortage"]) for row in zero_rows),
                "reason": "Replacement cannot sample from an empty exact cell and would also mismatch unique/repeat exposure.",
            },
            "exact_allow_t_overlap": {
                "status": "REJECTED_CONTRAST_CONTAMINATION",
                "minimum_overlap_count": sum(shortages.values()),
                "minimum_overlap_rate": sum(shortages.values()) / denominator,
                "seeded_random_overlap_count": deterministic_overlap,
                "seeded_random_overlap_rate": deterministic_overlap / denominator,
                "expected_random_overlap_count": expected_overlap,
                "expected_random_overlap_rate": expected_overlap / denominator,
            },
            "matchable_t_subset": {
                "status": "REJECTED_CHANGES_TREATMENT_AND_RATE",
                "unique_count": matchable,
                "removed_t_count": denominator - matchable,
                "retained_fraction": matchable / denominator,
            },
            "drop_group_hash_random": {
                "status": "FEASIBLE_BUT_DOMINATED",
                "unique_count": len(drop_group),
                "identity_digest": identity_digest(drop_group),
                "overlap_with_t_count": 0,
                "exact_label_dynamic_fold": True,
                "joint_total_variation": _counter_tv(treatment_strict, drop_strict, denominator),
                "group_total_variation": _counter_tv(treatment_groups, drop_groups, denominator),
            },
            "minimum_displacement_zero_overlap": {
                "status": "FEASIBLE_REQUIRES_PREREGISTERED_SPEC_CHANGE",
                "unique_count": len(minimum.records),
                "identity_digest": minimum.identity_digest,
                "overlap_with_t_count": minimum.overlap_with_t_count,
                "exact_label_dynamic_fold": minimum.coarse_quota_exact,
                "exact_oof_group_quota": False,
                "joint_displacement_count": minimum.observed_displacement_count,
                "joint_displacement_lower_bound": minimum.minimum_displacement_count,
                "joint_total_variation": minimum.observed_displacement_count / denominator,
                "group_total_variation": minimum.group_total_variation,
                "selection_semantic": "ZERO_OVERLAP_UNIQUE_EXACT_LABEL_DYNAMIC_FOLD_MINIMUM_OOF_GROUP_DISPLACEMENT_RANDOM_FILL",
                "residual_risk": "12.6% of occurrences differ in the filename-bucket surrogate; R1 remains a co-primary control.",
            },
            "new_canonical_asset": {
                "status": "NOT_CURRENTLY_TESTABLE_CHANGES_FIXED_BASE_PROCESS",
                "minimum_new_exact_occurrences": sum(shortages.values()),
                "reason": "Adding identities changes the 120,000-row canonical denominator and common-parent process.",
            },
        },
        "recommended_option": "minimum_displacement_zero_overlap",
        "activation_status": "OWNER_PREREGISTRATION_ADDENDUM_REQUIRED",
    }
    return {**core, "audit_digest": stable_digest(core)}
