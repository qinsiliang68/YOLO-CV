from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from .asset_registry import AssetRegistry, validate_asset_registry
from .dataset_content_ledger import load_registered_dataset_content_map
from .errors import ErrorCode, SctsrError
from .identity_pool import IdentityPool, IdentityRecord, T_SELECTION_SEMANTIC, make_pool
from .random_controls import PoolBuildResult, build_r1_global_random, build_r2_matched_random
from .r2_addendum import (
    R2_APPROVED_IDENTITY_DIGEST,
    R2_APPROVED_SELECTION_SEED,
    R2_MINIMUM_OOF_GROUP_DISPLACEMENT_POLICY,
)
from .serialization import stable_digest
from .terminal_field_guard import TerminalFieldGuard


T_CANONICAL_IDENTITY_DIGEST = "D9702F54DA3D9C7C4E27B657B7EC7A5FD235DEC72E2257AE5029E0C62D7482C7"
PRETERMINAL_FIELDS = (
    "sample_id",
    "y_true",
    "replay_role",
    "historical_dynamic_bucket",
    "oof_fold",
    "oof_group_id",
    "group_source",
    "base_manifest_membership",
    "source_manifest_identity",
)


@dataclass(frozen=True, slots=True)
class FormalPoolInputs:
    base_records: tuple[IdentityRecord, ...]
    t_pool: IdentityPool
    base_manifest_sha256: str
    preterminal_source_sha256: str
    asset_registry_digest: str
    t_content_unique_count: int
    t_content_identity_digest: str


def _asset(registry: AssetRegistry, asset_id: str):
    matches = [record for record in registry.assets if record.asset_id == asset_id]
    if len(matches) != 1:
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Required formal pool asset is not uniquely registered", observed=asset_id)
    return matches[0]


def _frame_records(frame: pl.DataFrame) -> tuple[IdentityRecord, ...]:
    return tuple(IdentityRecord.from_mapping(row) for row in frame.iter_rows(named=True))


def load_formal_pool_inputs(registry: AssetRegistry, repository_root: str | Path) -> FormalPoolInputs:
    root = Path(repository_root).resolve()
    validate_asset_registry(registry, root)
    guard = TerminalFieldGuard()
    guard.reject_if_config_mentions_forbidden(PRETERMINAL_FIELDS)

    oof = _asset(registry, "oof_assignments")
    dynamics = _asset(registry, "sample_dynamics_summary")
    t_manifest = _asset(registry, "t_stress_manifest")
    oof_path = root / oof.relative_path
    dynamics_path = root / dynamics.relative_path
    t_path = root / t_manifest.relative_path

    # Polars projection is applied at scan time. Terminal columns from the
    # dynamics and T source CSVs never enter the matching frame.
    oof_frame = (
        pl.scan_csv(oof_path)
        .select(
            pl.col("canonical_image_relpath").alias("sample_id"),
            pl.col("oof_y_true").alias("y_true"),
            pl.col("oof_fold"),
            pl.col("oof_group_id"),
            pl.col("oof_group_source").alias("group_source"),
        )
        .collect()
    )
    dynamics_frame = (
        pl.scan_csv(dynamics_path)
        .select(
            pl.col("sample_id"),
            pl.col("dynamic_bucket").alias("historical_dynamic_bucket"),
        )
        .collect()
    )
    base = oof_frame.join(dynamics_frame, on="sample_id", how="inner", validate="1:1")
    if base.height != registry.base_denominator or base["sample_id"].n_unique() != registry.base_denominator:
        raise SctsrError(ErrorCode.DENOMINATOR_IDENTITY_MISMATCH, "Pre-terminal base frame does not cover the canonical denominator", observed={"rows": base.height, "unique": base["sample_id"].n_unique()}, expected=registry.base_denominator)
    base = base.with_columns(
        pl.when(pl.col("y_true") == 0).then(pl.lit("normal_replay")).otherwise(pl.lit("defect_guard_replay")).alias("replay_role"),
        pl.lit(True).alias("base_manifest_membership"),
        pl.lit(registry.base_identity_digest).alias("source_manifest_identity"),
    ).select(PRETERMINAL_FIELDS)

    t_source = (
        pl.scan_csv(t_path)
        .select(
            "sample_id",
            "y_true",
            pl.col("replay_role").str.to_lowercase(),
            pl.col("dynamic_bucket").alias("historical_dynamic_bucket"),
            "oof_fold",
        )
        .collect()
    )
    t = t_source.join(
        oof_frame.select("sample_id", "oof_group_id", "group_source"),
        on="sample_id",
        how="inner",
        validate="1:1",
    ).with_columns(
        pl.lit(True).alias("base_manifest_membership"),
        pl.lit(t_manifest.sha256).alias("source_manifest_identity"),
    ).select(PRETERMINAL_FIELDS)
    if t.height != t_manifest.row_count:
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "T manifest did not join one-to-one with OOF identity", observed=t.height, expected=t_manifest.row_count)

    base_records = _frame_records(base)
    t_records = _frame_records(t)
    base_binding = stable_digest({asset_id: _asset(registry, asset_id).sha256 for asset_id in registry.base_asset_ids})
    preterminal_binding = stable_digest({"oof_assignments": oof.sha256, "sample_dynamics_summary": dynamics.sha256})
    t_pool = make_pool(
        pool_id="T_STRESS_RUN_010_FROZEN",
        pool_role="T_STRESS",
        records=t_records,
        base_denominator=registry.base_denominator,
        base_manifest_sha256=base_binding,
        source_manifest_path=t_manifest.relative_path,
        source_manifest_sha256=t_manifest.sha256,
        construction_seed=None,
        selection_semantic=T_SELECTION_SEMANTIC,
    )
    t_pool.validate(base_denominator=registry.base_denominator, base_ids={record.sample_id for record in base_records})
    if t_pool.spec.identity_digest != T_CANONICAL_IDENTITY_DIGEST:
        raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Registered T stress-set digest differs from the taskbook", observed=t_pool.spec.identity_digest, expected=T_CANONICAL_IDENTITY_DIGEST)
    content = load_registered_dataset_content_map(registry=registry, repository_root=root)
    t_content_rows = []
    for record in t_records:
        row = content.get(record.sample_id)
        if row is None:
            raise SctsrError(
                ErrorCode.DATASET_CONTENT_MISMATCH,
                "Registered T identity is absent from the frozen content ledger",
                observed=record.sample_id,
            )
        t_content_rows.append({"sample_id": record.sample_id, "image_sha256": str(row["image_sha256"]).upper()})
    unique_content = len({row["image_sha256"] for row in t_content_rows})
    if unique_content != len(t_records):
        raise SctsrError(
            ErrorCode.DATASET_CONTENT_MISMATCH,
            "Registered T stress set contains duplicate image bytes",
            observed={"identities": len(t_records), "unique_image_sha256": unique_content},
        )
    t_content_digest = stable_digest(sorted(t_content_rows, key=lambda row: row["sample_id"]))
    return FormalPoolInputs(
        base_records,
        t_pool,
        base_binding,
        preterminal_binding,
        registry.digest,
        unique_content,
        t_content_digest,
    )


def build_registered_r1(inputs: FormalPoolInputs, *, base_denominator: int, selection_seed: int) -> PoolBuildResult:
    return build_r1_global_random(
        inputs.base_records,
        base_denominator=base_denominator,
        base_manifest_sha256=inputs.base_manifest_sha256,
        source_manifest_sha256=inputs.preterminal_source_sha256,
        selection_seed=selection_seed,
        t_ids={record.sample_id for record in inputs.t_pool.records},
    )


def build_registered_r2(inputs: FormalPoolInputs, *, base_denominator: int, selection_seed: int) -> PoolBuildResult:
    if selection_seed != R2_APPROVED_SELECTION_SEED:
        raise SctsrError(
            ErrorCode.CONFIGURATION_MISMATCH,
            "Formal R2 construction must use the owner-approved frozen selection seed",
            failing_field="selection_seed",
            observed=selection_seed,
            expected=R2_APPROVED_SELECTION_SEED,
        )
    rows: list[dict[str, Any]] = [
        {
            "sample_id": record.sample_id,
            "y_true": record.y_true,
            "replay_role": record.replay_role,
            "historical_dynamic_bucket": record.historical_dynamic_bucket,
            "oof_fold": record.oof_fold,
            "oof_group_id": record.oof_group_id,
            "group_source": record.group_source,
            "base_manifest_membership": record.base_manifest_membership,
            "source_manifest_identity": inputs.preterminal_source_sha256,
        }
        for record in inputs.base_records
    ]
    result = build_r2_matched_random(
        rows,
        t_pool=inputs.t_pool,
        base_denominator=base_denominator,
        base_manifest_sha256=inputs.base_manifest_sha256,
        source_manifest_sha256=inputs.preterminal_source_sha256,
        selection_seed=selection_seed,
        guard=TerminalFieldGuard(),
        matching_policy=R2_MINIMUM_OOF_GROUP_DISPLACEMENT_POLICY,
    )
    if result.pool.spec.identity_digest != R2_APPROVED_IDENTITY_DIGEST:
        raise SctsrError(
            ErrorCode.IDENTITY_DIGEST_MISMATCH,
            "Materialized R2 identity differs from the owner-approved addendum",
            observed=result.pool.spec.identity_digest,
            expected=R2_APPROVED_IDENTITY_DIGEST,
        )
    return result
