from __future__ import annotations

import importlib
import csv
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

from .arm_spec import ArmId
from .asset_registry import load_asset_registry, load_registered_split_labels, validate_asset_registry
from .baseline_reference import MAIN_COMMIT, TASKBOOK_BLOB_SHA
from .branch_lineage import BranchLineage
from .columnar import read_columnar, validate_columnar_file
from .contracts import require_synthetic_or_authorized, validate_contract_files
from .errors import ErrorCode, SctsrError
from .formal_training import FormalIdentity
from .formal_pool_inputs import load_formal_pool_inputs
from .identity_pool import FixedIdentityPoolSpec, IdentityPool, IdentityRecord, partition_five_groups
from .dataset_adapter import DatasetIdentity, load_identity_manifest, validate_materialized_dataset_bytes
from .dataset_content_ledger import (
    load_registered_dataset_content_map,
    registered_dataset_manifest_asset_ids,
    validate_registered_dataset_content,
)
from .evidence_runtime import SampleEvidence
from .rate_spec import DenominatorRole, RateSemantic, ReplayRateSpec
from .r2_addendum import validate_approved_r2_build
from .schedule import SchedulePlan
from .selection_ledger import validate_selection_rows
from .seed_registry import SeedRegistry
from .serialization import atomic_write_json, load_json, sha256_file, stable_digest
from .source_identity import validate_source_tree_manifest
from .training_system import UpstreamBinding, bind_upstream, prepare_classification_overrides, validate_sctsr_adapter_import


FORMAL_AUTHORIZATION_INPUT_ROLES = (
    "release_authorization",
    "release_trust_policy",
    "source_tree_manifest",
    "contract",
    "arms",
    "asset_registry",
    "runtime_config",
    "seed_registry",
)


def build_external_file_binding(
    paths: Mapping[str, str | Path],
    *,
    required_roles: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Bind every external formal input by absolute path, size and SHA-256."""

    required = tuple(required_roles)
    if len(required) != len(set(required)) or set(paths) != set(required):
        raise SctsrError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "External formal-input roles do not exactly match the registered set",
            observed=sorted(paths),
            expected=sorted(required),
        )
    rows: dict[str, dict[str, Any]] = {}
    for role in sorted(required):
        path = Path(paths[role]).resolve()
        if not path.is_file():
            raise SctsrError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                "External formal input is missing",
                failing_field=role,
                artifact_path=str(path),
            )
        rows[role] = {
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    core = {
        "schema_version": "stage1.sctsr.external_file_binding.v1",
        "required_roles": sorted(required),
        "files": rows,
    }
    return {**core, "binding_digest": stable_digest(core)}


def validate_external_file_binding(
    binding: Mapping[str, Any],
    *,
    required_roles: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    required = sorted(required_roles)
    if set(binding) != {"schema_version", "required_roles", "files", "binding_digest"}:
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "External formal-input binding schema is invalid")
    core = {key: value for key, value in binding.items() if key != "binding_digest"}
    if (
        binding.get("schema_version") != "stage1.sctsr.external_file_binding.v1"
        or binding.get("required_roles") != required
        or binding.get("binding_digest") != stable_digest(core)
    ):
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "External formal-input binding identity is invalid")
    rows = binding.get("files")
    if not isinstance(rows, Mapping) or set(rows) != set(required):
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "External formal-input file roles are incomplete")
    checked: dict[str, dict[str, Any]] = {}
    for role in required:
        row = rows[role]
        if not isinstance(row, Mapping) or set(row) != {"path", "bytes", "sha256"}:
            raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "External formal-input file row is invalid", failing_field=role)
        path = Path(str(row["path"])).resolve()
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            raise SctsrError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                "External formal-input bytes changed after validation",
                failing_field=role,
                artifact_path=str(path),
            )
        checked[role] = dict(row)
    return {"status": "PASS", "binding_digest": binding["binding_digest"], "files": checked}


def validate_prepared_trainer_external_files(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Re-hash trainer-facing lock, model, overrides and identity bytes."""

    if not isinstance(binding, Mapping):
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Prepared trainer binding is not an object")
    core = {key: value for key, value in binding.items() if key != "binding_digest"}
    if binding.get("binding_digest") != stable_digest(core):
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Prepared trainer binding digest is invalid")
    identity = binding.get("identity_manifest_binding")
    if not isinstance(identity, Mapping):
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Prepared trainer identity-manifest binding is missing")
    identity_core = {key: value for key, value in identity.items() if key != "binding_digest"}
    if identity.get("binding_digest") != stable_digest(identity_core):
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Training identity-manifest binding digest is invalid")
    file_specs = {
        "canonical_training_lock": (
            binding.get("canonical_training_lock_path"),
            binding.get("canonical_training_lock_sha256"),
            None,
        ),
        "initial_checkpoint": (
            binding.get("initial_checkpoint_path"),
            binding.get("initial_checkpoint_sha256"),
            None,
        ),
        "trainer_overrides": (
            binding.get("trainer_overrides_path"),
            binding.get("trainer_overrides_sha256"),
            None,
        ),
        "identity_manifest": (
            identity.get("path"),
            identity.get("sha256"),
            identity.get("bytes"),
        ),
    }
    checked: dict[str, dict[str, Any]] = {}
    for role, (path_value, expected_sha, expected_bytes) in file_specs.items():
        path = Path(str(path_value)).resolve()
        if (
            not path.is_file()
            or not isinstance(expected_sha, str)
            or sha256_file(path) != expected_sha
            or (expected_bytes is not None and path.stat().st_size != expected_bytes)
        ):
            raise SctsrError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                "Prepared trainer external bytes changed after setup",
                failing_field=role,
                artifact_path=str(path),
            )
        checked[role] = {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": expected_sha}
    return {"status": "PASS", "files": checked, "binding_digest": binding["binding_digest"]}


def validate_formal_authorization_inputs_at_closeout(
    external_binding: Mapping[str, Any],
    *,
    repository_root: str | Path,
    expected_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute every scientific authorization digest without renewing it.

    Signature time/nonce policy is evaluated immediately before training.
    Closeout instead proves that the exact already-authorized bytes, source
    checkout, assets, contract, runtime and seed registry still exist and
    still derive the signed identities.
    """

    checked = validate_external_file_binding(
        external_binding,
        required_roles=FORMAL_AUTHORIZATION_INPUT_ROLES,
    )
    paths = {role: Path(row["path"]) for role, row in checked["files"].items()}
    root = Path(repository_root).resolve()
    source = validate_source_tree_manifest(paths["source_tree_manifest"], root, require_clean=True)
    contract = validate_contract_files(paths["contract"], paths["arms"])
    registry = load_asset_registry(paths["asset_registry"])
    assets = validate_asset_registry(registry, root, verify_large_files=True)
    runtime_raw = load_json(paths["runtime_config"])
    if not isinstance(runtime_raw, Mapping):
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Runtime policy must be a JSON object")
    runtime_digest = _validate_runtime_policy(runtime_raw)
    seed_raw = load_json(paths["seed_registry"])
    if not isinstance(seed_raw, Mapping):
        raise SctsrError(ErrorCode.SEED_REGISTRY_INVALID, "Seed registry must be a JSON object")
    seed_digest = SeedRegistry.from_mapping(seed_raw).digest
    recomputed = {
        "baseline_main_commit": MAIN_COMMIT.upper(),
        "taskbook_blob_sha": TASKBOOK_BLOB_SHA.upper(),
        "source_tree_digest": str(source["source_tree_digest"]).upper(),
        "contract_digest": contract.contract_digest.upper(),
        "asset_registry_digest": str(assets["registry_digest"]).upper(),
        "runtime_config_digest": runtime_digest.upper(),
        "seed_registry_digest": seed_digest.upper(),
    }
    if dict(expected_bindings) != recomputed:
        raise SctsrError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Closeout scientific inputs no longer derive the signed release bindings",
            observed=dict(expected_bindings),
            expected=recomputed,
        )
    return {
        "status": "PASS",
        "external_binding_digest": checked["binding_digest"],
        "recomputed_bindings": recomputed,
        "asset_registry": registry,
    }

_FORMAL_IDENTITY_FIELDS = {
    "schema_version",
    "training_seed",
    "canonical_training_lock_sha256",
    "initial_checkpoint_sha256",
    "base_manifest_sha256",
    "source_tree_digest",
    "runtime_config_digest",
    "asset_registry_digest",
    "contract_digest",
    "seed_registry_digest",
}


def _contained_file(path_value: Any, *, root: Path, role: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, f"{role} path is missing")
    path = Path(path_value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise SctsrError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"{role} path escapes its immutable artifact root",
            artifact_path=str(path),
        ) from exc
    if not path.is_file():
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, f"{role} file is missing", artifact_path=str(path))
    return path


def _pool_spec_from_mapping(raw: Mapping[str, Any]) -> FixedIdentityPoolSpec:
    expected = {field.name for field in fields(FixedIdentityPoolSpec)}
    if set(raw) != expected or not isinstance(raw.get("rate"), Mapping):
        raise SctsrError(
            ErrorCode.SCHEMA_VALIDATION_FAILED,
            "Identity-pool specification does not exactly match the registered schema",
            observed={"missing": sorted(expected - set(raw)), "extra": sorted(set(raw) - expected)},
        )
    rate_raw = raw["rate"]
    expected_rate_fields = {"numerator", "denominator", "semantic", "denominator_role"}
    # ReplayRateSpec.as_dict also publishes the derived canonical token.  It is
    # checked, never trusted as another configuration input.
    observed_rate_fields = set(rate_raw)
    if observed_rate_fields != expected_rate_fields and observed_rate_fields != expected_rate_fields | {"canonical_token"}:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Identity-pool rate schema is invalid")
    try:
        rate = ReplayRateSpec(
            int(rate_raw["numerator"]),
            int(rate_raw["denominator"]),
            RateSemantic(str(rate_raw["semantic"])),
            DenominatorRole(str(rate_raw["denominator_role"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Identity-pool rate cannot be parsed") from exc
    if "canonical_token" in rate_raw and rate_raw["canonical_token"] != rate.canonical_token():
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Identity-pool canonical rate token is inconsistent")
    return FixedIdentityPoolSpec(
        pool_id=str(raw["pool_id"]),
        pool_role=str(raw["pool_role"]),
        rate=rate,
        base_manifest_sha256=str(raw["base_manifest_sha256"]).upper(),
        source_manifest_path=str(raw["source_manifest_path"]),
        source_manifest_sha256=str(raw["source_manifest_sha256"]).upper(),
        identity_digest=str(raw["identity_digest"]).upper(),
        identity_digest_algorithm=str(raw["identity_digest_algorithm"]),
        unique_count_derived=int(raw["unique_count_derived"]),
        label_quota={str(key): int(value) for key, value in raw["label_quota"].items()},
        oof_fold_quota={str(key): int(value) for key, value in raw["oof_fold_quota"].items()},
        dynamic_bucket_quota={str(key): int(value) for key, value in raw["dynamic_bucket_quota"].items()},
        oof_group_quota={str(key): int(value) for key, value in raw["oof_group_quota"].items()},
        oof_group_semantic=str(raw["oof_group_semantic"]),
        construction_seed=None if raw["construction_seed"] is None else int(raw["construction_seed"]),
        selection_semantic=str(raw["selection_semantic"]),
    )


def _load_identity_pool_artifact(
    manifest_path: str | Path,
    *,
    expected_base_denominator: int,
    expected_base_manifest_sha256: str,
) -> tuple[IdentityPool, dict[str, tuple[IdentityRecord, ...]], Mapping[str, Any]]:
    path = Path(manifest_path).resolve()
    if not path.is_file() or path.name != "POOL_MANIFEST.json":
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Identity pool must be supplied by its canonical POOL_MANIFEST.json", artifact_path=str(path))
    root = path.parent
    raw = load_json(path)
    expected_fields = {
        "schema_version", "pool_spec", "membership", "selection", "quota_audit", "audit",
        "group_counts", "pool_digest", "semantic", "formal_training_started",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_fields:
        raise SctsrError(
            ErrorCode.SCHEMA_VALIDATION_FAILED,
            "Identity-pool manifest fields do not exactly match the registered schema",
            observed={
                "missing": sorted(expected_fields - set(raw) if isinstance(raw, Mapping) else expected_fields),
                "extra": sorted(set(raw) - expected_fields if isinstance(raw, Mapping) else []),
            },
            artifact_path=str(path),
        )
    if raw["schema_version"] != "stage1.sctsr.identity_pool_manifest.v1" or raw["formal_training_started"] is not False:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Identity-pool manifest state/schema is invalid", artifact_path=str(path))
    spec_raw = raw["pool_spec"]
    membership = raw["membership"]
    selection = raw["selection"]
    quota_binding = raw["quota_audit"]
    if not isinstance(spec_raw, Mapping) or not isinstance(membership, Mapping) or not isinstance(selection, Mapping) or not isinstance(quota_binding, Mapping):
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Identity-pool manifest contains a non-object binding")
    spec = _pool_spec_from_mapping(spec_raw)
    if spec.base_manifest_sha256 != expected_base_manifest_sha256.upper():
        raise SctsrError(
            ErrorCode.IDENTITY_DIGEST_MISMATCH,
            "Identity pool is bound to another canonical base manifest",
            observed=spec.base_manifest_sha256,
            expected=expected_base_manifest_sha256.upper(),
        )
    membership_path = _contained_file(membership.get("path"), root=root, role="Identity membership")
    report = validate_columnar_file(
        membership_path,
        expected_rows=int(membership.get("row_count", -1)),
        expected_schema_version="stage1.sctsr.identity_group_membership.v1",
        expected_schema_digest=str(membership.get("schema_digest", "")),
        expected_sha256=str(membership.get("sha256", "")),
        allow_synthetic_portable_fallback=False,
    )
    if membership.get("canonical_parquet") is not True or membership.get("storage_format") != "PARQUET_ZSTD" or membership.get("compression") != "ZSTD":
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Formal identity membership is not canonical Zstd Parquet")
    if membership.get("epoch") != 0 or not isinstance(membership.get("run_id"), str):
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Identity membership lacks its run/epoch partition identity")
    rows = read_columnar(membership_path)
    expected_row_fields = {field.name for field in fields(IdentityRecord)} | {"identity_group", "pool_id", "pool_role"}
    groups: dict[str, list[IdentityRecord]] = {f"G{index}": [] for index in range(5)}
    for index, row in enumerate(rows):
        if set(row) != expected_row_fields:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Identity membership row fields are not exact", failing_field=f"row[{index}]")
        if row["pool_id"] != spec.pool_id or row["pool_role"] != spec.pool_role or row["identity_group"] not in groups:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Identity membership row disagrees with the pool manifest", failing_field=f"row[{index}]")
        groups[str(row["identity_group"])].append(IdentityRecord.from_mapping(row))
    pool = IdentityPool(spec, tuple(record for values in groups.values() for record in values))
    pool.validate(base_denominator=expected_base_denominator)
    expected_groups = partition_five_groups(pool, base_denominator=expected_base_denominator)
    for name in groups:
        if {record.sample_id for record in groups[name]} != {record.sample_id for record in expected_groups[name]}:
            raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Identity group assignment is not the deterministic five-way partition", failing_field=name)
    group_counts = {name: len(values) for name, values in groups.items()}
    if raw["group_counts"] != group_counts or int(report["row_count"]) != spec.unique_count_derived:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Identity-pool group counts are inconsistent", observed=raw["group_counts"], expected=group_counts)
    if raw["pool_digest"] != spec.identity_digest or raw["semantic"] != spec.selection_semantic:
        raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Identity-pool manifest digest/semantic differs from its specification")
    selection_path = _contained_file(selection.get("path"), root=root, role="Selection ledger")
    selection_report = validate_columnar_file(
        selection_path,
        expected_rows=expected_base_denominator,
        expected_schema_version="stage1.sctsr.selection_ledger.v1",
        expected_schema_digest=str(selection.get("schema_digest", "")),
        expected_sha256=str(selection.get("sha256", "")),
        allow_synthetic_portable_fallback=False,
    )
    if selection.get("canonical_parquet") is not True or selection.get("epoch") != 0:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Formal selection evidence is not canonical epoch=0000 Parquet")
    selection_rows = read_columnar(selection_path)
    validate_selection_rows(selection_rows, r2=spec.pool_role == "R2_MATCHED_RANDOM")
    selected_from_ledger = {str(row["candidate_sample_id"]) for row in selection_rows if bool(row["selected"])}
    if selected_from_ledger != {record.sample_id for record in pool.records}:
        raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Selection ledger selected IDs differ from identity membership")
    if set(quota_binding) != {"path", "bytes", "sha256"}:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Quota-audit byte binding is incomplete")
    quota_path = _contained_file(quota_binding["path"], root=root, role="Quota audit")
    if quota_path.stat().st_size != int(quota_binding["bytes"]) or sha256_file(quota_path) != quota_binding["sha256"]:
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Quota-audit byte binding differs from disk")
    quota = load_json(quota_path)
    expected_quota = {
        "schema_version": "stage1.sctsr.identity_pool_quota_audit.v1",
        "pool_role": spec.pool_role,
        "pool_digest": spec.identity_digest,
        "group_counts": group_counts,
    }
    if any(quota.get(field) != value for field, value in expected_quota.items()):
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Quota audit differs from identity-pool bytes")
    if spec.pool_role == "R2_MATCHED_RANDOM":
        if quota.get("terminal_field_status") != "TERMINAL_FIELDS_NOT_LOADED" or quota.get("overlap_with_t") != 0:
            raise SctsrError(ErrorCode.TERMINAL_FIELD_ACCESS_FORBIDDEN, "R2 quota audit does not prove terminal-field exclusion and zero overlap")
        if expected_base_denominator == 120_000 and not isinstance(raw.get("audit"), Mapping):
            raise SctsrError(ErrorCode.R2_QUOTA_INFEASIBLE, "Formal R2 artifact lacks its approved-addendum construction audit")
        if expected_base_denominator == 120_000:
            if quota.get("pool_build_audit") != raw["audit"]:
                raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "R2 quota audit and pool manifest disagree on construction evidence")
            validate_approved_r2_build(spec_raw, raw["audit"])
    return pool, {name: tuple(values) for name, values in groups.items()}, {
        "manifest_path": path.as_posix(),
        "manifest_sha256": sha256_file(path),
        "membership_sha256": membership["sha256"],
        "selection_sha256": selection["sha256"],
        "selection_rows": int(selection_report["row_count"]),
        "quota_audit_sha256": quota_binding["sha256"],
    }


def validate_identity_pool_artifacts(
    manifest_paths: list[str | Path] | tuple[str | Path, ...],
    *,
    schedule: SchedulePlan,
    expected_base_denominator: int,
    expected_base_manifest_sha256: str,
) -> dict[str, Any]:
    """Bind a materialized schedule to immutable, role-correct pool artifacts."""

    if schedule.base_denominator != expected_base_denominator:
        raise SctsrError(ErrorCode.DENOMINATOR_IDENTITY_MISMATCH, "Schedule and formal base denominator differ")
    expected_roles = {
        ArmId.NR: set(),
        ArmId.R1_U: {"R1_GLOBAL_RANDOM"},
        ArmId.R2_U: {"R2_MATCHED_RANDOM"},
        ArmId.T_U: {"T_STRESS"},
        ArmId.R2_F: {"R2_MATCHED_RANDOM"},
        ArmId.T_F: {"T_STRESS"},
        ArmId.T_TO_R2_AT_160: {"T_STRESS", "R2_MATCHED_RANDOM"},
        ArmId.T_TO_NR_AT_160: {"T_STRESS"},
    }[schedule.arm_id]
    if len(manifest_paths) != len(expected_roles):
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Branch identity-pool count differs from its arm", observed=len(manifest_paths), expected=len(expected_roles))
    loaded: dict[str, tuple[IdentityPool, dict[str, tuple[IdentityRecord, ...]], Mapping[str, Any]]] = {}
    for path in manifest_paths:
        item = _load_identity_pool_artifact(
            path,
            expected_base_denominator=expected_base_denominator,
            expected_base_manifest_sha256=expected_base_manifest_sha256,
        )
        role = item[0].spec.pool_role
        if role in loaded:
            raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Branch received duplicate identity-pool roles", observed=role)
        loaded[role] = item
    if set(loaded) != expected_roles:
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Branch identity-pool roles differ from its arm", observed=sorted(loaded), expected=sorted(expected_roles))
    if expected_roles:
        if schedule.arm_id is ArmId.T_TO_R2_AT_160:
            expected_digest = stable_digest({
                "primary": loaded["T_STRESS"][0].spec.identity_digest,
                "fallback": loaded["R2_MATCHED_RANDOM"][0].spec.identity_digest,
            })
        else:
            expected_digest = next(iter(loaded.values()))[0].spec.identity_digest
        if schedule.identity_pool_digest != expected_digest:
            raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Schedule identity digest differs from supplied pool artifacts", observed=schedule.identity_pool_digest, expected=expected_digest)
    elif schedule.identity_pool_digest != "NONE" or schedule.total_occurrences != 0:
        raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "NR schedule is not identity-free")
    ids_by_role = {role: {record.sample_id for record in item[0].records} for role, item in loaded.items()}
    if "T_STRESS" in ids_by_role and "R2_MATCHED_RANDOM" in ids_by_role and ids_by_role["T_STRESS"] & ids_by_role["R2_MATCHED_RANDOM"]:
        raise SctsrError(ErrorCode.R2_OVERLAPS_T, "Supplied T and R2 pool artifacts overlap")
    used: dict[str, set[str]] = {role: set() for role in loaded}
    policy_role = {"T_STRESS": "T_STRESS", "R1_GLOBAL_RANDOM": "R1_GLOBAL_RANDOM", "R2_MATCHED_RANDOM": "R2_MATCHED_RANDOM"}
    for epoch in schedule.epochs:
        if not epoch.sample_ids:
            continue
        role = policy_role.get(epoch.identity_policy)
        if role not in loaded or not set(epoch.sample_ids).issubset(ids_by_role[role]):
            raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Schedule occurrence IDs are not supplied by the declared role pool", arm_id=schedule.arm_id.value, epoch=epoch.epoch)
        used[role].update(epoch.sample_ids)
    if any(used[role] != ids for role, ids in ids_by_role.items()):
        raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "Schedule does not use each supplied identity pool exactly as registered")
    bindings = {role: dict(item[2]) for role, item in loaded.items()}
    return {
        "status": "PASS",
        "arm_id": schedule.arm_id.value,
        "pool_roles": sorted(loaded),
        "pool_digests": {role: loaded[role][0].spec.identity_digest for role in sorted(loaded)},
        "artifact_bindings": bindings,
        "binding_digest": stable_digest(bindings),
    }


def validate_parent_artifact_index(
    *,
    parent_checkpoint: str | Path,
    parent_artifact_index: str | Path,
) -> dict[str, Any]:
    """Re-audit the complete formal parent and bind the selected E120 bytes."""

    checkpoint = Path(parent_checkpoint).resolve()
    index = Path(parent_artifact_index).resolve()
    if not checkpoint.is_file():
        raise SctsrError(ErrorCode.PARENT_SHA_MISMATCH, "Parent checkpoint is missing", artifact_path=str(checkpoint))
    if index.name != "ARTIFACT_INDEX.json" or not index.is_file():
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Parent artifact index must be the canonical ARTIFACT_INDEX.json", artifact_path=str(index))
    parent_root = index.parent.resolve()
    try:
        checkpoint.relative_to(parent_root)
    except ValueError as exc:
        raise SctsrError(ErrorCode.BRANCH_LINEAGE_MISMATCH, "Parent checkpoint is outside the supplied parent artifact root") from exc
    from .run_validation import validate_run_tree

    report = validate_run_tree(parent_root, allow_synthetic_portable_fallback=False)
    checkpoint_sha = sha256_file(checkpoint)
    if report.get("semantic") != "formal" or report.get("run_role") != "COMMON_PARENT":
        raise SctsrError(ErrorCode.BRANCH_LINEAGE_MISMATCH, "Supplied artifact index is not a formally completed common parent", observed=report)
    if report.get("fixed_endpoint_checkpoint_sha256") != checkpoint_sha:
        raise SctsrError(
            ErrorCode.PARENT_SHA_MISMATCH,
            "Selected parent checkpoint differs from the re-audited E120 endpoint",
            observed=checkpoint_sha,
            expected=report.get("fixed_endpoint_checkpoint_sha256"),
        )
    binding = {
        "schema_version": "stage1.sctsr.parent_artifact_binding.v1",
        "parent_root": parent_root.as_posix(),
        "artifact_index_path": index.as_posix(),
        "artifact_index_sha256": sha256_file(index),
        "parent_checkpoint_path": checkpoint.as_posix(),
        "parent_checkpoint_sha256": checkpoint_sha,
        "parent_manifest_sha256": report["manifest_sha256"],
        "parent_artifact_digest": report["artifact_digest"],
        "parent_receipt_chain_digest": report["receipt_chain_digest"],
        "validated_epoch_transactions": report["epoch_transaction_count"],
        "validation_status": "PASS",
    }
    return {**binding, "binding_digest": stable_digest(binding)}


def validate_training_identity_manifest(
    identity_manifest: str | Path,
    *,
    base_records: tuple[IdentityRecord, ...],
    pool_manifest_paths: tuple[str | Path, ...] | list[str | Path],
    schedule: SchedulePlan | None,
    base_denominator: int,
    base_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate every trainer-facing identity row against pre-terminal assets.

    The manifest controls metadata and physical-file lookup but cannot change
    the canonical identity/label/dynamics/OOF universe or the deterministic
    group membership supplied by the selected arm's pool artifacts.
    """

    path = Path(identity_manifest).resolve()
    if not path.is_file() or path.suffix.lower() != ".csv":
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Training identity manifest must be a readable CSV", artifact_path=str(path))
    if len(base_records) != base_denominator:
        raise SctsrError(ErrorCode.DENOMINATOR_IDENTITY_MISMATCH, "Validated pre-terminal base record count changed")
    expected = {record.sample_id: record for record in base_records}
    if len(expected) != base_denominator:
        raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "Validated pre-terminal base contains duplicate identities")

    expected_group: dict[str, str] = {}
    expected_pool_role: dict[str, str] = {}
    if pool_manifest_paths:
        if schedule is None:
            raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Identity-pool annotations require a schedule")
        for manifest_path in pool_manifest_paths:
            pool, groups, _ = _load_identity_pool_artifact(
                manifest_path,
                expected_base_denominator=base_denominator,
                expected_base_manifest_sha256=base_manifest_sha256,
            )
            for group, records in groups.items():
                for record in records:
                    if record.sample_id in expected_group:
                        raise SctsrError(ErrorCode.R2_OVERLAPS_T, "Trainer identity annotation receives the same ID from multiple treatment pools", observed=record.sample_id)
                    expected_group[record.sample_id] = group
                    expected_pool_role[record.sample_id] = record.replay_role
    elif schedule is not None and schedule.arm_id is not ArmId.NR:
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Replay branch identity manifest lacks its pool artifacts")

    identities = load_identity_manifest(path)
    if len(identities) != base_denominator:
        raise SctsrError(ErrorCode.DENOMINATOR_IDENTITY_MISMATCH, "Training identity manifest does not contain exactly the canonical denominator", observed=len(identities), expected=base_denominator)
    observed_ids: set[str] = set()
    for row in identities:
        if row.sample_id in observed_ids:
            raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "Training identity manifest contains a duplicate ID", observed=row.sample_id)
        observed_ids.add(row.sample_id)
        registered = expected.get(row.sample_id)
        if registered is None:
            raise SctsrError(ErrorCode.IDENTITY_NOT_IN_BASE, "Training identity row is outside canonical base", observed=row.sample_id)
        core_observed = (
            row.y_true,
            row.replay_role,
            row.historical_dynamic_bucket,
            row.oof_fold,
            row.oof_group_id,
        )
        core_expected = (
            registered.y_true,
            registered.replay_role,
            registered.historical_dynamic_bucket,
            registered.oof_fold,
            registered.oof_group_id,
        )
        if core_observed != core_expected:
            raise SctsrError(
                ErrorCode.PREDICTION_IDENTITY_MISMATCH,
                "Training identity metadata differs from validated pre-terminal assets",
                observed={"sample_id": row.sample_id, "values": core_observed},
                expected=core_expected,
            )
        expected_identity_group = expected_group.get(row.sample_id, "UNASSIGNED")
        if row.identity_group != expected_identity_group:
            raise SctsrError(
                ErrorCode.IDENTITY_DIGEST_MISMATCH,
                "Training identity group differs from the supplied deterministic pool partition",
                observed={"sample_id": row.sample_id, "group": row.identity_group},
                expected=expected_identity_group,
            )
        if row.sample_id in expected_pool_role and row.replay_role != expected_pool_role[row.sample_id]:
            raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Training identity replay role differs from pool artifact", observed=row.sample_id)
        if Path(row.source_path).name.casefold() != Path(row.sample_id).name.casefold():
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Training identity source filename differs from canonical sample ID", observed=row.source_path, expected=row.sample_id)
        if row.oof_reference_probability is not None and not (math.isfinite(row.oof_reference_probability) and 0 <= row.oof_reference_probability <= 1):
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "OOF reference probability is invalid", observed=row.sample_id)
        SampleEvidence(
            sample_id=row.sample_id,
            y_true=row.y_true,
            replay_role=row.replay_role,
            oof_fold=row.oof_fold,
            oof_group_id=row.oof_group_id,
            historical_dynamic_bucket=row.historical_dynamic_bucket,
            identity_group=row.identity_group,
            oof_reference_probability=row.oof_reference_probability,
            oof_reference_reason=row.oof_reference_reason,
            rho_candidate_signal=row.rho_candidate_signal,
            rho_reason=row.rho_reason,
        ).validate()
    if observed_ids != set(expected):
        raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Training identity manifest does not exactly cover canonical base")
    payload = {
        "schema_version": "stage1.sctsr.training_identity_manifest_binding.v1",
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": len(identities),
        "base_manifest_sha256": base_manifest_sha256,
        "annotated_pool_identity_count": len(expected_group),
        "schedule_digest": None if schedule is None else schedule.plan_digest,
    }
    return {**payload, "binding_digest": stable_digest(payload)}


def _manifest_filename_label_pairs(path: Path, *, label: int) -> set[tuple[str, int]]:
    pairs: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "canonical_image_relpath" not in set(reader.fieldnames or ()):
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Data-role manifest lacks canonical_image_relpath", artifact_path=str(path))
        for row in reader:
            pair = (Path(str(row["canonical_image_relpath"])).name.casefold(), label)
            if not pair[0] or pair in pairs:
                raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "Data-role manifest has an ambiguous filename/label", artifact_path=str(path), observed=pair)
            pairs.add(pair)
    return pairs


def validate_prepared_trainer_datasets(
    trainer: Any,
    *,
    registry: Any,
    repository_root: Path,
    dataset_root: Path,
    evidence_root: Path,
) -> dict[str, Any]:
    """Prove upstream train/validation loaders expose only registered roles."""

    train_dataset = getattr(getattr(trainer, "train_loader", None), "dataset", None)
    train_identities = getattr(train_dataset, "identities", None)
    if train_identities is None or len(train_dataset) != registry.base_denominator or len(train_identities) != registry.base_denominator:
        raise SctsrError(ErrorCode.DENOMINATOR_IDENTITY_MISMATCH, "Prepared train loader is not the canonical identity-augmenting 120k base")
    if len(getattr(trainer, "train_loader", ())) != 938:
        raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "Prepared train loader does not contain exactly 938 base steps")
    expected_labels = load_registered_split_labels(registry, repository_root, "val_model")
    expected = {(Path(sample_id).name.casefold(), label) for sample_id, label in expected_labels.items()}
    val_dataset = getattr(getattr(trainer, "test_loader", None), "dataset", None)
    val_identities = getattr(val_dataset, "identities", None)
    samples = getattr(val_dataset, "samples", None)
    if not isinstance(samples, (list, tuple)) or not isinstance(val_identities, (list, tuple)):
        raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Prepared validation loader does not expose filtered auditable identities")
    observed = [(Path(str(row[0])).name.casefold(), int(row[1])) for row in samples]
    observed_labels = {str(row.sample_id): int(row.y_true) for row in val_identities}
    if len(observed) != len(set(observed)) or set(observed) != expected or observed_labels != expected_labels:
        raise SctsrError(
            ErrorCode.TEST_ACCESS_FORBIDDEN,
            "Upstream validation loader is not exactly the registered val_model/study role",
            observed={"rows": len(observed), "unexpected": sorted(set(observed) - expected)[:20], "missing": sorted(expected - set(observed))[:20]},
            expected={"rows": len(expected), "role": "val_model/study"},
        )
    content = load_registered_dataset_content_map(registry=registry, repository_root=repository_root)
    train_content_binding = validate_materialized_dataset_bytes(
        train_dataset,
        content,
        role="train",
        dataset_root=dataset_root,
        evidence_path=evidence_root / "train_materialized_files.parquet",
    )
    val_content_binding = validate_materialized_dataset_bytes(
        val_dataset,
        content,
        role="val_model",
        dataset_root=dataset_root,
        evidence_path=evidence_root / "val_model_materialized_files.parquet",
    )
    return {
        "status": "PASS",
        "train_rows": len(train_dataset),
        "train_steps": len(trainer.train_loader),
        "val_model_rows": len(observed),
        "val_role": "VAL_MODEL_STUDY_ONLY_NOT_METHOD_SELECTION",
        "train_materialized_content_binding": train_content_binding,
        "val_model_materialized_content_binding": val_content_binding,
        "test_accessed": False,
        "blind_holdout_opened": False,
        "dataset_binding_digest": stable_digest({"train_ids": [row.sample_id for row in train_identities], "val_pairs": sorted(observed)}),
    }


def validate_binary_classification_contract(trainer: Any, data_root: str | Path) -> dict[str, Any]:
    """Fail before optimizer step zero unless every binary identity agrees."""

    import torch

    expected_classes = ["no_target", "target_defect"]
    expected_class_to_idx = {"no_target": 0, "target_defect": 1}
    expected_model_names = {0: "no_target", 1: "target_defect"}
    root = Path(data_root).resolve()
    trainer_data = getattr(trainer, "data", None)
    if not isinstance(trainer_data, Mapping):
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Prepared trainer has no classification data contract")
    split_roots: dict[str, Path] = {}
    for role in ("train", "val"):
        split_value = trainer_data.get(role)
        if not isinstance(split_value, (str, Path)):
            raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Prepared trainer lacks a binary split root", failing_field=role)
        split_root = Path(split_value).resolve()
        try:
            split_root.relative_to(root)
        except ValueError as exc:
            raise SctsrError(
                ErrorCode.CONFIGURATION_MISMATCH,
                "Prepared classification split escapes its authorized data root",
                failing_field=role,
                observed=split_root.as_posix(),
                expected=root.as_posix(),
            ) from exc
        observed_directories = sorted(path.name for path in split_root.iterdir() if path.is_dir()) if split_root.is_dir() else []
        if observed_directories != expected_classes:
            raise SctsrError(
                ErrorCode.CONFIGURATION_MISMATCH,
                "Classification split directories are not exactly the frozen two classes",
                failing_field=role,
                observed=observed_directories,
                expected=expected_classes,
            )
        split_roots[role] = split_root

    datasets = {
        "train": getattr(getattr(trainer, "train_loader", None), "dataset", None),
        "val": getattr(getattr(trainer, "test_loader", None), "dataset", None),
    }
    class_counts: dict[str, dict[str, int]] = {}
    dataset_identities: dict[str, Any] = {}
    for role, dataset in datasets.items():
        classes = list(getattr(dataset, "classes", []))
        class_to_idx = dict(getattr(dataset, "class_to_idx", {}))
        samples = getattr(dataset, "samples", None)
        if classes != expected_classes or class_to_idx != expected_class_to_idx or not isinstance(samples, (list, tuple)):
            raise SctsrError(
                ErrorCode.CONFIGURATION_MISMATCH,
                "Prepared dataset class identity is not the frozen binary mapping",
                failing_field=role,
                observed={"classes": classes, "class_to_idx": class_to_idx},
                expected={"classes": expected_classes, "class_to_idx": expected_class_to_idx},
            )
        counts = {"0": 0, "1": 0}
        for sample in samples:
            if not isinstance(sample, (list, tuple)) or len(sample) < 2 or type(sample[1]) is not int or sample[1] not in {0, 1}:
                raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Prepared dataset contains a non-binary label", failing_field=role)
            counts[str(sample[1])] += 1
        if min(counts.values()) <= 0:
            raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Prepared dataset has an empty binary class", failing_field=role, observed=counts)
        class_counts[role] = counts
        dataset_identities[role] = {
            "classes": classes,
            "class_to_idx": class_to_idx,
            "derived_nc": len(classes),
            "row_count": len(samples),
        }

    raw_data_names = trainer_data.get("names")
    data_names = dict(raw_data_names) if isinstance(raw_data_names, Mapping) else {}
    if data_names != expected_model_names:
        raise SctsrError(
            ErrorCode.CONFIGURATION_MISMATCH,
            "Trainer dataset names are not the frozen binary mapping",
            observed=data_names,
            expected=expected_model_names,
        )
    if trainer_data.get("nc") != 2:
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Trainer dataset nc is not two", observed=trainer_data.get("nc"), expected=2)
    model = getattr(trainer, "model", None)
    raw_names = getattr(model, "names", None)
    model_names = dict(raw_names) if isinstance(raw_names, Mapping) else {}
    if model_names != expected_model_names:
        raise SctsrError(
            ErrorCode.CONFIGURATION_MISMATCH,
            "Prepared model names are not the frozen binary mapping",
            observed=model_names,
            expected=expected_model_names,
        )
    model_nc = getattr(model, "nc", len(model_names))
    if type(model_nc) is not int or model_nc != 2:
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Prepared model nc is not two", observed=model_nc, expected=2)
    linear_heads = [module for module in model.modules() if isinstance(module, torch.nn.Linear)] if hasattr(model, "modules") else []
    if not linear_heads or int(linear_heads[-1].out_features) != 2:
        raise SctsrError(
            ErrorCode.CONFIGURATION_MISMATCH,
            "Final classification head does not expose exactly two outputs",
            observed=None if not linear_heads else int(linear_heads[-1].out_features),
            expected=2,
        )
    core = {
        "schema_version": "stage1.sctsr.binary_classification_binding.v1",
        "status": "PASS",
        "data_root": root.as_posix(),
        "split_roots": {role: path.as_posix() for role, path in split_roots.items()},
        "expected_classes": expected_classes,
        "class_to_idx": expected_class_to_idx,
        "class_counts": class_counts,
        "datasets": dataset_identities,
        "trainer_data_nc": 2,
        "trainer_data_names": data_names,
        "model_names": model_names,
        "model_nc": model_nc,
        "head_type": f"{type(linear_heads[-1]).__module__}.{type(linear_heads[-1]).__qualname__}",
        "head_out_features": int(linear_heads[-1].out_features),
    }
    return {**core, "binary_contract_digest": stable_digest(core)}


def load_formal_identity(path: str | Path) -> FormalIdentity:
    raw = load_json(path)
    if not isinstance(raw, Mapping) or set(raw) != _FORMAL_IDENTITY_FIELDS:
        raise SctsrError(
            ErrorCode.CONFIGURATION_MISMATCH,
            "Formal identity fields do not exactly match the registered schema",
            observed={
                "missing": sorted(_FORMAL_IDENTITY_FIELDS - set(raw) if isinstance(raw, Mapping) else _FORMAL_IDENTITY_FIELDS),
                "extra": sorted(set(raw) - _FORMAL_IDENTITY_FIELDS if isinstance(raw, Mapping) else []),
            },
        )
    if raw.get("schema_version") != "stage1.sctsr.formal_identity.v1" or type(raw.get("training_seed")) is not int:
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Formal identity schema or seed type is invalid")
    identity = FormalIdentity(
        training_seed=raw["training_seed"],
        canonical_training_lock_sha256=str(raw["canonical_training_lock_sha256"]).upper(),
        initial_checkpoint_sha256=str(raw["initial_checkpoint_sha256"]).upper(),
        base_manifest_sha256=str(raw["base_manifest_sha256"]).upper(),
        source_tree_digest=str(raw["source_tree_digest"]).upper(),
        runtime_config_digest=str(raw["runtime_config_digest"]).upper(),
        asset_registry_digest=str(raw["asset_registry_digest"]).upper(),
        contract_digest=str(raw["contract_digest"]).upper(),
        seed_registry_digest=str(raw["seed_registry_digest"]).upper(),
    )
    identity.validate(formal=True)
    return identity


def load_lineage(path: str | Path) -> BranchLineage:
    raw = load_json(path)
    if not isinstance(raw, Mapping):
        raise SctsrError(ErrorCode.BRANCH_LINEAGE_MISMATCH, "Branch lineage must be a JSON object")
    try:
        return BranchLineage(**raw)
    except TypeError as exc:
        raise SctsrError(ErrorCode.BRANCH_LINEAGE_MISMATCH, "Branch lineage schema is invalid") from exc


def _validate_runtime_policy(raw: Mapping[str, Any]) -> str:
    expected = {
        "schema_version": "stage1.sctsr.runtime_policy.v1",
        "base_denominator": 120_000,
        "base_steps_per_epoch": 938,
        "canonical_batch_size": 128,
        "optimizer_steps_locked_to_base": True,
        "replay_in_dataset_length": False,
        "replay_gradient_retained": True,
        "scheduler_advances_on_replay": False,
        "ema_advances_on_replay": False,
        "warmup_locked_to_base_steps": True,
        "global_rng_restored_after_replay": True,
        "minimum_resume_free_bytes": 21_474_836_480,
        "batchnorm_buffers_restored_after_replay": True,
        "oom_policy": "ABORT_FIXED_CONTRACT",
        "world_size": 1,
        "formal_endpoint_epoch": 200,
        "formal_endpoint_split_role": "val_op",
        "formal_endpoint_model_variant": "EMA",
        "formal_endpoint_batch_size": 128,
        "execution_mode_default": "synthetic",
    }
    mismatch = {key: {"observed": raw.get(key), "expected": value} for key, value in expected.items() if raw.get(key) != value}
    if mismatch:
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Runtime policy differs from the frozen formal process", observed=mismatch)
    if raw.get("forbidden_checkpoint_names") != ["best.pt"]:
        raise SctsrError(ErrorCode.BEST_PT_FORBIDDEN, "Runtime policy does not uniquely forbid best.pt")
    if set(raw.get("formal_data_roles_forbidden", [])) != {"test", "blind_holdout"}:
        raise SctsrError(ErrorCode.TEST_ACCESS_FORBIDDEN, "Runtime policy does not seal test and blind holdout")
    return stable_digest(raw)


def prepare_formal_authorization(
    *,
    repository_root: str | Path,
    identity: FormalIdentity,
    release_authorization: str | Path,
    release_trust_policy: str | Path,
    source_tree_manifest: str | Path,
    contract_path: str | Path,
    arms_path: str | Path,
    asset_registry_path: str | Path,
    runtime_config_path: str | Path,
    seed_registry_path: str | Path,
) -> dict[str, Any]:
    """Validate every prepared-run byte before upstream trainer construction."""

    identity.validate(formal=True)
    root = Path(repository_root).resolve()
    source = validate_source_tree_manifest(source_tree_manifest, root, require_clean=True)
    contract = validate_contract_files(contract_path, arms_path)
    asset_registry = load_asset_registry(asset_registry_path)
    asset = validate_asset_registry(asset_registry, root, verify_large_files=True)
    assets_by_id = {record.asset_id: record for record in asset_registry.assets}
    required_asset_ids = {"canonical_training_lock", "initial_checkpoint"}
    if not required_asset_ids.issubset(assets_by_id):
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Formal asset registry lacks lock or initialization bytes")
    formal_pool_inputs = load_formal_pool_inputs(asset_registry, root)
    asset_identity_mismatch = {
        "canonical_training_lock_sha256": {
            "identity": identity.canonical_training_lock_sha256,
            "asset": assets_by_id["canonical_training_lock"].sha256,
        },
        "initial_checkpoint_sha256": {
            "identity": identity.initial_checkpoint_sha256,
            "asset": assets_by_id["initial_checkpoint"].sha256,
        },
        "base_manifest_sha256": {
            "identity": identity.base_manifest_sha256,
            "asset": formal_pool_inputs.base_manifest_sha256,
        },
    }
    asset_identity_mismatch = {
        field: values for field, values in asset_identity_mismatch.items() if values["identity"] != values["asset"]
    }
    if asset_identity_mismatch:
        raise SctsrError(
            ErrorCode.IDENTITY_DIGEST_MISMATCH,
            "Formal identity lock/checkpoint/base bytes differ from the validated asset registry",
            observed=asset_identity_mismatch,
        )
    runtime_raw = load_json(runtime_config_path)
    if not isinstance(runtime_raw, Mapping):
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Runtime policy must be a JSON object")
    runtime_digest = _validate_runtime_policy(runtime_raw)
    seed_raw = load_json(seed_registry_path)
    if not isinstance(seed_raw, Mapping):
        raise SctsrError(ErrorCode.SEED_REGISTRY_INVALID, "Seed registry must be a JSON object")
    seeds = SeedRegistry.from_mapping(seed_raw)
    seed_digest = seeds.digest
    expected_bindings = {
        "baseline_main_commit": MAIN_COMMIT.upper(),
        "taskbook_blob_sha": TASKBOOK_BLOB_SHA.upper(),
        "source_tree_digest": str(source["source_tree_digest"]).upper(),
        "contract_digest": contract.contract_digest.upper(),
        "asset_registry_digest": str(asset["registry_digest"]).upper(),
        "runtime_config_digest": runtime_digest.upper(),
        "seed_registry_digest": seed_digest.upper(),
    }
    prepared = {
        "source_tree_digest": identity.source_tree_digest,
        "contract_digest": identity.contract_digest,
        "asset_registry_digest": identity.asset_registry_digest,
        "runtime_config_digest": identity.runtime_config_digest,
        "seed_registry_digest": identity.seed_registry_digest,
    }
    mismatches = {
        field: {"identity": prepared[field], "validated": expected_bindings[field]}
        for field in prepared
        if prepared[field] != expected_bindings[field]
    }
    if mismatches:
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Formal identity differs from validated prepared-run inputs", observed=mismatches)
    require_synthetic_or_authorized(
        "formal",
        release_authorization,
        trust_policy=release_trust_policy,
        expected_bindings=expected_bindings,
    )
    seeds.validate(
        formal=True,
        release_authorization_verified=True,
        expected_registry_digest=expected_bindings["seed_registry_digest"],
    )
    allowed_seeds = set(seeds.discovery_seeds) | set(seeds.confirmation_seeds)
    if identity.training_seed not in allowed_seeds:
        raise SctsrError(
            ErrorCode.SEED_REGISTRY_INVALID,
            "Prepared training seed is not a release-frozen discovery or confirmation seed",
            observed=identity.training_seed,
        )
    release = load_json(release_authorization)
    formal_input_binding = build_external_file_binding(
        {
            "release_authorization": release_authorization,
            "release_trust_policy": release_trust_policy,
            "source_tree_manifest": source_tree_manifest,
            "contract": contract_path,
            "arms": arms_path,
            "asset_registry": asset_registry_path,
            "runtime_config": runtime_config_path,
            "seed_registry": seed_registry_path,
        },
        required_roles=FORMAL_AUTHORIZATION_INPUT_ROLES,
    )
    return {
        "status": "PASS",
        "expected_bindings": expected_bindings,
        "release_id": release["release_id"],
        "key_id": release["key_id"],
        "release_manifest_sha256": sha256_file(release_authorization),
        "trust_policy_sha256": sha256_file(release_trust_policy),
        "source_manifest_sha256": sha256_file(source_tree_manifest),
        "runtime_environment_digest": source["runtime_environment_digest"],
        "runtime_environment": source["runtime_environment"],
        "contract_path": Path(contract_path).resolve().as_posix(),
        "asset_registry_path": Path(asset_registry_path).resolve().as_posix(),
        "canonical_training_lock_path": (root / assets_by_id["canonical_training_lock"].relative_path).resolve().as_posix(),
        "initial_checkpoint_path": (root / assets_by_id["initial_checkpoint"].relative_path).resolve().as_posix(),
        "base_identity_digest": asset_registry.base_identity_digest,
        "base_manifest_sha256": formal_pool_inputs.base_manifest_sha256,
        "runtime_config_path": Path(runtime_config_path).resolve().as_posix(),
        "seed_registry_path": Path(seed_registry_path).resolve().as_posix(),
        "formal_input_binding": formal_input_binding,
    }


def build_prepared_trainer(
    *,
    repository_root: str | Path,
    identity_manifest: str | Path,
    trainer_overrides_path: str | Path,
    identity: FormalIdentity,
    output_root: str | Path,
    asset_registry_path: str | Path,
    schedule: SchedulePlan | None = None,
    identity_pool_manifests: tuple[str | Path, ...] | list[str | Path] = (),
) -> tuple[Any, UpstreamBinding, dict[str, Any]]:
    """Construct, but do not train, a byte- and role-bound upstream trainer."""

    identity.validate(formal=True)
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Formal run output root must not exist before upstream setup", artifact_path=str(output))
    binding = bind_upstream(root)
    registry = load_asset_registry(asset_registry_path)
    validate_asset_registry(registry, root, verify_large_files=True)
    pool_inputs = load_formal_pool_inputs(registry, root)
    assets = {record.asset_id: record for record in registry.assets}
    lock_path = (root / assets["canonical_training_lock"].relative_path).resolve()
    model_path = (root / assets["initial_checkpoint"].relative_path).resolve()
    expected_identity = {
        "canonical_training_lock_sha256": sha256_file(lock_path),
        "initial_checkpoint_sha256": sha256_file(model_path),
        "base_manifest_sha256": pool_inputs.base_manifest_sha256,
        "asset_registry_digest": registry.digest,
    }
    identity_values = {
        "canonical_training_lock_sha256": identity.canonical_training_lock_sha256,
        "initial_checkpoint_sha256": identity.initial_checkpoint_sha256,
        "base_manifest_sha256": identity.base_manifest_sha256,
        "asset_registry_digest": identity.asset_registry_digest,
    }
    mismatch = {field: {"identity": identity_values[field], "validated": expected} for field, expected in expected_identity.items() if identity_values[field] != expected}
    if mismatch:
        raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Prepared trainer identity differs from validated formal assets", observed=mismatch)
    manifest_binding = validate_training_identity_manifest(
        identity_manifest,
        base_records=pool_inputs.base_records,
        pool_manifest_paths=identity_pool_manifests,
        schedule=schedule,
        base_denominator=registry.base_denominator,
        base_manifest_sha256=pool_inputs.base_manifest_sha256,
    )
    overrides_raw = load_json(trainer_overrides_path)
    if not isinstance(overrides_raw, Mapping):
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Trainer overrides must be a JSON object")
    overrides = dict(overrides_raw)
    placeholder_fields = sorted(field for field, value in overrides.items() if isinstance(value, str) and "REPLACE_WITH" in value)
    if placeholder_fields:
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Trainer overrides still contain template placeholders", observed=placeholder_fields)
    clean = prepare_classification_overrides(binding, overrides)
    if type(clean.get("seed")) is not int or clean["seed"] != identity.training_seed:
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Trainer seed differs from the signed formal identity", observed=clean.get("seed"), expected=identity.training_seed)
    supplied_model = Path(str(clean["model"]))
    supplied_model = (root / supplied_model).resolve() if not supplied_model.is_absolute() else supplied_model.resolve()
    if supplied_model != model_path or sha256_file(supplied_model) != identity.initial_checkpoint_sha256:
        raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Trainer model is not the registered yolo11l initialization", artifact_path=str(supplied_model))
    data_root = Path(str(clean["data"]))
    data_root = (root / data_root).resolve() if not data_root.is_absolute() else data_root.resolve()
    if not data_root.is_dir():
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Prepared classification data root is missing", artifact_path=str(data_root))
    dataset_content_binding = validate_registered_dataset_content(
        registry=registry,
        repository_root=root,
        dataset_root=data_root,
        required_manifest_asset_ids=registered_dataset_manifest_asset_ids(registry),
        verify_physical_files=True,
    )
    project = Path(str(clean["project"]))
    project = (root / project).resolve() if not project.is_absolute() else project.resolve()
    if project != output or clean.get("name") != "trainer" or clean.get("exist_ok") is not False or clean.get("resume") is not False:
        raise SctsrError(
            ErrorCode.CONFIGURATION_MISMATCH,
            "Trainer output/name/resume binding differs from the fresh formal-run contract",
            observed={"project": project.as_posix(), "name": clean.get("name"), "exist_ok": clean.get("exist_ok"), "resume": clean.get("resume")},
            expected={"project": output.as_posix(), "name": "trainer", "exist_ok": False, "resume": False},
        )
    device = str(clean.get("device"))
    if not device.isdigit():
        raise SctsrError(ErrorCode.DISTRIBUTED_MODE_NOT_SUPPORTED_IN_V4_PHASE1, "Formal phase 1 requires exactly one numeric CUDA device", observed=clean.get("device"))
    clean.update({"model": model_path.as_posix(), "data": data_root.as_posix(), "project": output.as_posix()})
    yolo_root = root / "YOLOv11"
    integration_root = root / "integrations" / "ultralytics"
    for value in (str(yolo_root), str(root), str(integration_root)):
        if value not in sys.path:
            sys.path.insert(0, value)
    module = importlib.import_module("sctsr_classification_trainer")
    adapter_binding = validate_sctsr_adapter_import(binding, module)
    val_model_identities = tuple(
        DatasetIdentity(sample_id=sample_id, y_true=label, source_path=sample_id)
        for sample_id, label in sorted(load_registered_split_labels(registry, root, "val_model").items())
    )
    try:
        trainer = module.SctsrClassificationTrainer(
            overrides=clean,
            identity_manifest=identity_manifest,
            validation_identities=val_model_identities,
            training_seed=identity.training_seed,
        )
        # Use the frozen upstream setup routine, but never enter upstream
        # _do_train, which can auto-reduce batch and final-evaluate best.pt.
        trainer._setup_train()
        trainer.accumulate = 1
        binary_contract_binding = validate_binary_classification_contract(trainer, data_root)
        dataset_binding = validate_prepared_trainer_datasets(
            trainer,
            registry=registry,
            repository_root=root,
            dataset_root=data_root,
            evidence_root=output / "trainer" / "sctsr_materialized_bindings",
        )
    except BaseException as exc:
        if output.exists():
            suffix = stable_digest({"output": output.as_posix(), "exception": type(exc).__name__, "message": str(exc)})[:12]
            quarantine = output.with_name(f"{output.name}.setup_failed.{suffix}")
            attempt = 1
            while quarantine.exists():
                quarantine = output.with_name(f"{output.name}.setup_failed.{suffix}.{attempt}")
                attempt += 1
            try:
                output.rename(quarantine)
                atomic_write_json(
                    quarantine / "SETUP_FAILURE.json",
                    {
                        "schema_version": "stage1.sctsr.prepared_trainer_setup_failure.v1",
                        "status": "QUARANTINED_NOT_FORMAL_TRAINING",
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "original_output_root": output.as_posix(),
                        "quarantine_root": quarantine.as_posix(),
                        "formal_training_started": False,
                    },
                )
            except OSError:
                # The incomplete root has no RUN_MANIFEST and therefore can
                # never validate or resume. Preserve the original exception;
                # an operator can quarantine it after open handles close.
                pass
        raise
    trainer_binding = {
        "schema_version": "stage1.sctsr.prepared_trainer_binding.v1",
        "upstream_binding_digest": binding.binding_digest,
        "adapter_import_binding": adapter_binding,
        "canonical_training_lock_path": lock_path.as_posix(),
        "canonical_training_lock_sha256": sha256_file(lock_path),
        "initial_checkpoint_path": model_path.as_posix(),
        "initial_checkpoint_sha256": sha256_file(model_path),
        "trainer_overrides_path": Path(trainer_overrides_path).resolve().as_posix(),
        "trainer_overrides_sha256": sha256_file(trainer_overrides_path),
        "resolved_overrides_digest": stable_digest(clean),
        "scientific_overrides_digest": stable_digest(
            {
                key: value
                for key, value in clean.items()
                if key not in {"project", "name", "exist_ok", "resume"}
            }
        ),
        "identity_manifest_binding": manifest_binding,
        "dataset_binding": dataset_binding,
        "dataset_content_binding": dataset_content_binding,
        "binary_classification_binding": binary_contract_binding,
        "output_root": output.as_posix(),
        "training_seed": identity.training_seed,
        "formal_training_started": False,
    }
    return trainer, binding, {**trainer_binding, "binding_digest": stable_digest(trainer_binding)}
