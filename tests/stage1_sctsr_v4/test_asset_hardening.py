from __future__ import annotations

import csv
import hashlib

import pytest

from stage1_sctsr_v4.asset_registry import AssetRegistry, validate_asset_registry
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
