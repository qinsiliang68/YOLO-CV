from __future__ import annotations

import csv
import hashlib

import pytest

from stage1_sctsr_v4.asset_registry import (
    AssetRegistry,
    build_split_identity_bundle,
    load_registered_split_labels,
    validate_asset_registry,
)
from stage1_sctsr_v4.serialization import atomic_write_json, load_json
from stage1_sctsr_v4.errors import ErrorCode, SctsrError


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "y_true", "split"))
        writer.writeheader()
        writer.writerows(rows)


def _registry(path, **overrides):
    expected_digest = hashlib.sha256(b"a|0\nb|1\n").hexdigest().upper()
    record = {
        "asset_id": "base",
        "relative_path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        "bytes": path.stat().st_size,
        "role": "CANONICAL_BASE_MANIFEST",
        "required": True,
        "row_count": 2,
        "identity_column": "sample_id",
        "label_column": "y_true",
        "identity_digest": expected_digest,
        "identity_digest_algorithm": "SHA256_SORTED_SAMPLE_ID_PIPE_LABEL_LF",
        "split_column": "split",
        "allowed_split_values": ["train"],
        "identity_universe": "CANONICAL_BASE_FULL",
    }
    record.update(overrides)
    return AssetRegistry.from_mapping(
        {
            "schema_version": "stage1.sctsr.asset_registry.v1",
            "base_denominator": 2,
            "base_identity_digest": expected_digest,
            "base_identity_digest_algorithm": "SHA256_SORTED_SAMPLE_ID_PIPE_LABEL_LF",
            "base_asset_ids": ["base"],
            "val_target_available": False,
            "assets": [record],
        }
    )


def test_sa_041_asset_row_count_is_verified_from_csv(tmp_path):
    path = tmp_path / "base.csv"
    _write_csv(path, [{"sample_id": "a", "y_true": 0, "split": "train"}, {"sample_id": "b", "y_true": 1, "split": "train"}])

    with pytest.raises(SctsrError) as exc:
        validate_asset_registry(_registry(path, row_count=3), tmp_path)

    assert exc.value.code is ErrorCode.ASSET_VALIDATION_FAILED


def test_sa_042_asset_identity_digest_is_recomputed(tmp_path):
    path = tmp_path / "base.csv"
    _write_csv(path, [{"sample_id": "a", "y_true": 0, "split": "train"}, {"sample_id": "b", "y_true": 1, "split": "train"}])

    with pytest.raises(SctsrError) as exc:
        validate_asset_registry(_registry(path, identity_digest="0" * 64), tmp_path)

    assert exc.value.code is ErrorCode.IDENTITY_DIGEST_MISMATCH


def test_sa_043_asset_split_role_is_verified_from_rows(tmp_path):
    path = tmp_path / "base.csv"
    _write_csv(path, [{"sample_id": "a", "y_true": 0, "split": "train"}, {"sample_id": "b", "y_true": 1, "split": "test"}])

    with pytest.raises(SctsrError) as exc:
        validate_asset_registry(_registry(path), tmp_path)

    assert exc.value.code is ErrorCode.ASSET_VALIDATION_FAILED


def test_sa_044_base_denominator_is_derived_from_registered_identity(tmp_path):
    path = tmp_path / "base.csv"
    _write_csv(path, [{"sample_id": "a", "y_true": 0, "split": "train"}, {"sample_id": "b", "y_true": 1, "split": "train"}])
    registry = _registry(path)
    object.__setattr__(registry, "base_denominator", 120_000)

    with pytest.raises(SctsrError) as exc:
        validate_asset_registry(registry, tmp_path)

    assert exc.value.code is ErrorCode.DENOMINATOR_IDENTITY_MISMATCH


def test_registered_validation_split_bundle_is_sha_and_identity_bound(tmp_path):
    base = tmp_path / "base.csv"
    split = tmp_path / "val_op.csv"
    _write_csv(base, [{"sample_id": "base-a", "y_true": 0, "split": "train"}])
    _write_csv(
        split,
        [
            {"sample_id": "normal-a", "y_true": 0, "split": "val_op"},
            {"sample_id": "defect-a", "y_true": 1, "split": "val_op"},
        ],
    )

    def record(path, asset_id, role, row_count, identity_digest, split_value, universe):
        return {
            "asset_id": asset_id,
            "relative_path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
            "bytes": path.stat().st_size,
            "role": role,
            "row_count": row_count,
            "identity_column": "sample_id",
            "label_column": "y_true",
            "identity_digest": identity_digest,
            "identity_digest_algorithm": "SHA256_SORTED_SAMPLE_ID_PIPE_LABEL_LF",
            "split_column": "split",
            "allowed_split_values": [split_value],
            "identity_universe": universe,
        }

    base_digest = hashlib.sha256(b"base-a|0\n").hexdigest().upper()
    split_digest = hashlib.sha256(b"defect-a|1\nnormal-a|0\n").hexdigest().upper()
    raw_registry = {
            "schema_version": "stage1.sctsr.asset_registry.v1",
            "base_denominator": 1,
            "base_identity_digest": base_digest,
            "base_identity_digest_algorithm": "SHA256_SORTED_SAMPLE_ID_PIPE_LABEL_LF",
            "base_asset_ids": ["base"],
            "split_asset_ids": {"val_op": ["val_op"]},
            "val_target_available": False,
            "assets": [
                record(base, "base", "CANONICAL_BASE_MANIFEST", 1, base_digest, "train", "CANONICAL_BASE_FULL"),
                record(split, "val_op", "VAL_OP_ENDPOINT_ONLY", 2, split_digest, "val_op", "VAL_OP_COMPONENT"),
            ],
        }
    registry_path = tmp_path / "asset_registry.json"
    atomic_write_json(registry_path, raw_registry)
    registry = AssetRegistry.from_mapping(raw_registry)
    report = validate_asset_registry(registry, tmp_path)
    assert report["registered_splits"]["val_op"]["row_count"] == 2
    assert load_registered_split_labels(registry, tmp_path, "val_op") == {"normal-a": 0, "defect-a": 1}

    bundle_path = tmp_path / "VAL_OP_SPLIT_BUNDLE.json"
    bundle = build_split_identity_bundle(
        registry,
        repository_root=tmp_path,
        registry_path=registry_path,
        split_role="val_op",
        output_path=bundle_path,
    )
    assert load_json(bundle_path)["bundle_digest"] == bundle["bundle_digest"]
    assert bundle["sample_label_identity_digest"] == split_digest


def test_test_split_may_not_be_registered_for_sctsr_v4(tmp_path):
    path = tmp_path / "base.csv"
    _write_csv(path, [{"sample_id": "a", "y_true": 0, "split": "train"}, {"sample_id": "b", "y_true": 1, "split": "train"}])
    registry = _registry(path)
    object.__setattr__(registry, "split_asset_ids", (("test", ("base",)),))
    with pytest.raises(SctsrError) as caught:
        validate_asset_registry(registry, tmp_path)
    assert caught.value.code is ErrorCode.TEST_ACCESS_FORBIDDEN
