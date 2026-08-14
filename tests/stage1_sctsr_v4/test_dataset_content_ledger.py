from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

from stage1_sctsr_v4.asset_registry import AssetRecord, AssetRegistry
from stage1_sctsr_v4.dataset_content_ledger import (
    DATASET_CONTENT_DIGEST_ALGORITHM,
    DATASET_CONTENT_SCHEMA_VERSION,
    build_dataset_content_ledger,
    validate_registered_dataset_content,
)
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.serialization import sha256_file


def _write_png(path: Path, colour: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), colour).save(path, format="PNG")


def _write_manifest(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("split", "canonical_image_relpath"), lineterminator="\n")
        writer.writeheader()
        for split, relative in rows:
            writer.writerow({"split": split, "canonical_image_relpath": relative})


def _source_registry(repository: Path, manifest: Path, *, row_count: int) -> AssetRegistry:
    relative = manifest.relative_to(repository).as_posix()
    return AssetRegistry(
        schema_version="stage1.sctsr.asset_registry.v1",
        base_denominator=row_count,
        assets=(
            AssetRecord(
                asset_id="canonical_base_defect_manifest",
                relative_path=relative,
                sha256=sha256_file(manifest),
                bytes=manifest.stat().st_size,
                role="CANONICAL_BASE_COMPONENT_DEFECT",
                row_count=row_count,
                identity_column="canonical_image_relpath",
                constant_label=1,
                split_column="split",
                allowed_split_values=("train",),
            ),
        ),
        base_asset_ids=("canonical_base_defect_manifest",),
    )


def _with_ledger(source: AssetRegistry, repository: Path, ledger: Path, receipt: dict) -> AssetRegistry:
    return AssetRegistry(
        schema_version=source.schema_version,
        base_denominator=source.base_denominator,
        assets=source.assets
        + (
            AssetRecord(
                asset_id="dataset_content_ledger",
                relative_path=ledger.relative_to(repository).as_posix(),
                sha256=sha256_file(ledger),
                bytes=ledger.stat().st_size,
                role="DATASET_CONTENT_LEDGER_ALL_REGISTERED_NON_TEST_IMAGES",
                row_count=int(receipt["row_count"]),
                identity_digest=str(receipt["content_identity_digest"]),
                identity_digest_algorithm=DATASET_CONTENT_DIGEST_ALGORITHM,
            ),
        ),
        base_asset_ids=source.base_asset_ids,
    )


def test_dataset_content_ledger_rejects_same_filename_and_label_with_wrong_bytes(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    data_root = repository / "dataset"
    image = data_root / "Det/images/train/00000001.png"
    _write_png(image, (1, 2, 3))
    manifest = repository / "manifests/train.csv"
    _write_manifest(manifest, [("train", "Det/images/train/00000001.png")])
    source = _source_registry(repository, manifest, row_count=1)
    ledger = repository / "assets/dataset_content.parquet"
    receipt = build_dataset_content_ledger(
        registry=source,
        repository_root=repository,
        dataset_root=data_root,
        output_path=ledger,
    )
    registry = _with_ledger(source, repository, ledger, receipt)

    # The old filename/label-only binding accepted this valid replacement.
    _write_png(image, (3, 2, 1))

    with pytest.raises(SctsrError) as error:
        validate_registered_dataset_content(
            registry=registry,
            repository_root=repository,
            dataset_root=data_root,
            required_manifest_asset_ids=("canonical_base_defect_manifest",),
            verify_physical_files=True,
        )
    assert error.value.code is ErrorCode.DATASET_CONTENT_MISMATCH


def test_dataset_content_ledger_is_zstd_parquet_and_binds_exact_manifest_universe(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    data_root = repository / "dataset"
    rows = [
        ("train", "Det/images/train/00000001.png"),
        ("train", "Det/images/train/00000002.png"),
    ]
    for index, (_, relative) in enumerate(rows, start=1):
        _write_png(data_root / relative, (index, index, index))
    manifest = repository / "manifests/train.csv"
    _write_manifest(manifest, rows)
    source = _source_registry(repository, manifest, row_count=2)
    ledger = repository / "assets/dataset_content.parquet"
    receipt = build_dataset_content_ledger(
        registry=source,
        repository_root=repository,
        dataset_root=data_root,
        output_path=ledger,
    )
    registry = _with_ledger(source, repository, ledger, receipt)

    report = validate_registered_dataset_content(
        registry=registry,
        repository_root=repository,
        dataset_root=data_root,
        required_manifest_asset_ids=("canonical_base_defect_manifest",),
        verify_physical_files=True,
    )

    assert receipt["schema_version"] == DATASET_CONTENT_SCHEMA_VERSION
    assert receipt["storage_format"] == "PARQUET_ZSTD"
    assert receipt["compression"] == "ZSTD"
    assert receipt["row_count"] == 2
    assert report["status"] == "PASS"
    assert report["physical_files_verified"] == 2
    assert report["manifest_asset_ids"] == ["canonical_base_defect_manifest"]


def test_dataset_content_ledger_rejects_missing_manifest_identity(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    data_root = repository / "dataset"
    image = data_root / "Det/images/train/00000001.png"
    _write_png(image, (1, 1, 1))
    manifest = repository / "manifests/train.csv"
    _write_manifest(manifest, [("train", "Det/images/train/00000001.png")])
    source = _source_registry(repository, manifest, row_count=1)
    ledger = repository / "assets/dataset_content.parquet"
    receipt = build_dataset_content_ledger(
        registry=source,
        repository_root=repository,
        dataset_root=data_root,
        output_path=ledger,
    )
    registry = _with_ledger(source, repository, ledger, receipt)

    _write_manifest(
        manifest,
        [
            ("train", "Det/images/train/00000001.png"),
            ("train", "Det/images/train/00000002.png"),
        ],
    )

    with pytest.raises(SctsrError) as error:
        validate_registered_dataset_content(
            registry=registry,
            repository_root=repository,
            dataset_root=data_root,
            required_manifest_asset_ids=("canonical_base_defect_manifest",),
            verify_physical_files=False,
        )
    assert error.value.code in {ErrorCode.ASSET_VALIDATION_FAILED, ErrorCode.DATASET_CONTENT_MISMATCH}


def test_formal_dataset_content_validation_requires_registered_ledger(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    data_root = repository / "dataset"
    image = data_root / "Det/images/train/00000001.png"
    _write_png(image, (1, 1, 1))
    manifest = repository / "manifests/train.csv"
    _write_manifest(manifest, [("train", "Det/images/train/00000001.png")])
    registry = _source_registry(repository, manifest, row_count=1)

    with pytest.raises(SctsrError) as error:
        validate_registered_dataset_content(
            registry=registry,
            repository_root=repository,
            dataset_root=data_root,
            required_manifest_asset_ids=("canonical_base_defect_manifest",),
            verify_physical_files=True,
        )
    assert error.value.code is ErrorCode.DATASET_CONTENT_LEDGER_REQUIRED


def test_formal_dataset_content_ledger_rejects_extra_identity_outside_registered_universe(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    data_root = repository / "dataset"
    two_rows = [
        ("train", "Det/images/train/00000001.png"),
        ("train", "Det/images/train/00000002.png"),
    ]
    for index, (_, relative) in enumerate(two_rows, start=1):
        _write_png(data_root / relative, (index, index, index))
    manifest = repository / "manifests/train.csv"
    _write_manifest(manifest, two_rows)
    build_source = _source_registry(repository, manifest, row_count=2)
    ledger = repository / "assets/dataset_content.parquet"
    receipt = build_dataset_content_ledger(
        registry=build_source,
        repository_root=repository,
        dataset_root=data_root,
        output_path=ledger,
    )

    # Freeze a new manifest universe with only one registered identity while
    # retaining the old two-row ledger. Formal validation must reject the
    # unregistered extra row instead of silently ignoring it.
    _write_manifest(manifest, [two_rows[0]])
    current_source = _source_registry(repository, manifest, row_count=1)
    registry = _with_ledger(current_source, repository, ledger, receipt)

    with pytest.raises(SctsrError) as error:
        validate_registered_dataset_content(
            registry=registry,
            repository_root=repository,
            dataset_root=data_root,
            required_manifest_asset_ids=("canonical_base_defect_manifest",),
            verify_physical_files=False,
        )
    assert error.value.code is ErrorCode.DATASET_CONTENT_MISMATCH
