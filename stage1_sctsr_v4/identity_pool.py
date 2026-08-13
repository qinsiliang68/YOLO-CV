from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .errors import ErrorCode, SctsrError
from .rate_spec import IDENTITY_POOL_RATE, ReplayRateSpec
from .serialization import stable_digest

OOF_GROUP_SEMANTIC = "FILENAME_BUCKET_SURROGATE_NOT_TRUE_VIDEO_ID"
T_SELECTION_SEMANTIC = "HISTORICAL_SIGN_REVERSAL_STRESS_SET_NOT_VALIDATED_SELECTOR"


@dataclass(frozen=True, slots=True)
class IdentityRecord:
    sample_id: str
    y_true: int
    replay_role: str
    historical_dynamic_bucket: str
    oof_fold: int
    oof_group_id: str
    group_source: str = "numeric_filename_bucket"
    base_manifest_membership: bool = True

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "IdentityRecord":
        return cls(
            sample_id=str(row["sample_id"]),
            y_true=int(row.get("y_true", row.get("label"))),
            replay_role=str(row.get("replay_role", "NORMAL_REPLAY")),
            historical_dynamic_bucket=str(row.get("historical_dynamic_bucket", row.get("dynamic_bucket", "UNSPECIFIED"))),
            oof_fold=int(row["oof_fold"]),
            oof_group_id=str(row["oof_group_id"]),
            group_source=str(row.get("group_source", "numeric_filename_bucket")),
            base_manifest_membership=bool(row.get("base_manifest_membership", True)),
        )

    def stratum(self) -> tuple[int, str, int, str]:
        return (self.y_true, self.historical_dynamic_bucket, self.oof_fold, self.oof_group_id)


def identity_digest(records: Sequence[IdentityRecord]) -> str:
    h = hashlib.sha256()
    for record in sorted(records, key=lambda r: (r.replay_role, r.sample_id)):
        h.update(record.replay_role.encode("utf-8"))
        h.update(b"\t")
        h.update(record.sample_id.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest().upper()


@dataclass(frozen=True, slots=True)
class FixedIdentityPoolSpec:
    pool_id: str
    pool_role: str
    rate: ReplayRateSpec
    base_manifest_sha256: str
    source_manifest_path: str
    source_manifest_sha256: str
    identity_digest: str
    identity_digest_algorithm: str
    unique_count_derived: int
    label_quota: Mapping[str, int]
    oof_fold_quota: Mapping[str, int]
    dynamic_bucket_quota: Mapping[str, int]
    oof_group_quota: Mapping[str, int]
    oof_group_semantic: str
    construction_seed: int | None
    selection_semantic: str


@dataclass(frozen=True, slots=True)
class IdentityPool:
    spec: FixedIdentityPoolSpec
    records: tuple[IdentityRecord, ...]

    def validate(self, *, base_denominator: int, base_ids: set[str] | None = None, t_ids: set[str] | None = None) -> None:
        ids = [r.sample_id for r in self.records]
        if len(ids) != len(set(ids)):
            raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "Identity pool contains duplicate IDs")
        if base_ids is not None and not set(ids).issubset(base_ids):
            raise SctsrError(ErrorCode.IDENTITY_NOT_IN_BASE, "Identity pool contains IDs outside canonical base", observed=sorted(set(ids) - base_ids)[:20])
        expected_count = self.spec.rate.derive_count(base_denominator)
        if len(ids) != expected_count or self.spec.unique_count_derived != expected_count:
            raise SctsrError(ErrorCode.RATE_NOT_INTEGRAL, "Identity pool count differs from rate-derived count", observed=len(ids), expected=expected_count)
        observed_digest = identity_digest(self.records)
        if observed_digest != self.spec.identity_digest:
            raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Identity pool digest mismatch", observed=observed_digest, expected=self.spec.identity_digest)
        if self.spec.oof_group_semantic != OOF_GROUP_SEMANTIC:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "OOF group semantic must declare filename-bucket surrogate")
        if self.spec.pool_role == "T_STRESS" and self.spec.selection_semantic != T_SELECTION_SEMANTIC:
            raise SctsrError(ErrorCode.T_POOL_ROLE_MISMATCH, "T pool may not be labelled as a validated selector")
        if self.spec.pool_role == "R2_MATCHED_RANDOM" and t_ids is not None and set(ids) & t_ids:
            raise SctsrError(ErrorCode.R2_OVERLAPS_T, "R2 must have zero identity overlap with T")
        if self.spec.identity_digest_algorithm != "SHA256_SORTED_REPLAY_ROLE_TAB_SAMPLE_ID_LF":
            raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Identity pool uses an unregistered digest algorithm", observed=self.spec.identity_digest_algorithm)
        if len(self.spec.base_manifest_sha256) != 64 or len(self.spec.source_manifest_sha256) != 64:
            raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Identity pool source bindings are not SHA-256 values")
        if any(not record.base_manifest_membership for record in self.records):
            raise SctsrError(ErrorCode.IDENTITY_NOT_IN_BASE, "Identity pool contains a record not bound to canonical base")
        if any(record.group_source != "numeric_filename_bucket" for record in self.records):
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "OOF group source must be the frozen filename-bucket surrogate")
        counters = {
            "label_quota": Counter(str(r.y_true) for r in self.records),
            "oof_fold_quota": Counter(str(r.oof_fold) for r in self.records),
            "dynamic_bucket_quota": Counter(r.historical_dynamic_bucket for r in self.records),
            "oof_group_quota": Counter(r.oof_group_id for r in self.records),
        }
        for field, counter in counters.items():
            if dict(counter) != dict(getattr(self.spec, field)):
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, f"{field} does not match pool records")


def make_pool(
    *,
    pool_id: str,
    pool_role: str,
    records: Sequence[IdentityRecord],
    base_denominator: int,
    base_manifest_sha256: str,
    source_manifest_path: str,
    source_manifest_sha256: str,
    construction_seed: int | None,
    selection_semantic: str,
    rate: ReplayRateSpec = IDENTITY_POOL_RATE,
) -> IdentityPool:
    records_tuple = tuple(records)
    spec = FixedIdentityPoolSpec(
        pool_id=pool_id,
        pool_role=pool_role,
        rate=rate,
        base_manifest_sha256=base_manifest_sha256.upper(),
        source_manifest_path=source_manifest_path,
        source_manifest_sha256=source_manifest_sha256.upper(),
        identity_digest=identity_digest(records_tuple),
        identity_digest_algorithm="SHA256_SORTED_REPLAY_ROLE_TAB_SAMPLE_ID_LF",
        unique_count_derived=rate.derive_count(base_denominator),
        label_quota=dict(Counter(str(r.y_true) for r in records_tuple)),
        oof_fold_quota=dict(Counter(str(r.oof_fold) for r in records_tuple)),
        dynamic_bucket_quota=dict(Counter(r.historical_dynamic_bucket for r in records_tuple)),
        oof_group_quota=dict(Counter(r.oof_group_id for r in records_tuple)),
        oof_group_semantic=OOF_GROUP_SEMANTIC,
        construction_seed=construction_seed,
        selection_semantic=selection_semantic,
    )
    pool = IdentityPool(spec, records_tuple)
    pool.validate(base_denominator=base_denominator)
    return pool


def partition_five_groups(pool: IdentityPool, *, base_denominator: int) -> dict[str, tuple[IdentityRecord, ...]]:
    expected_group_size = 5 * base_denominator // 1000
    if base_denominator * 5 % 1000:
        raise SctsrError(ErrorCode.RATE_NOT_INTEGRAL, "0.5% group size is not integral for base denominator")
    if len(pool.records) != expected_group_size * 5:
        raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "Pool cannot be divided into five equal 0.5% groups")
    ranked = sorted(
        pool.records,
        key=lambda record: (
            hashlib.sha256((pool.spec.identity_digest + "\0" + record.sample_id).encode("utf-8")).hexdigest(),
            record.sample_id,
        ),
    )
    groups: dict[str, list[IdentityRecord]] = {f"G{i}": [] for i in range(5)}
    for index, record in enumerate(ranked):
        groups[f"G{index % 5}"].append(record)
    if any(len(values) != expected_group_size for values in groups.values()):
        raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "Five-group partition is not equal-sized")
    union = {record.sample_id for values in groups.values() for record in values}
    if len(union) != len(pool.records):
        raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "Five-group partition is not mutually exclusive and exhaustive")
    return {name: tuple(values) for name, values in groups.items()}


def group_membership_digest(groups: Mapping[str, Sequence[IdentityRecord]]) -> str:
    return stable_digest({group: [record.sample_id for record in records] for group, records in sorted(groups.items())})
