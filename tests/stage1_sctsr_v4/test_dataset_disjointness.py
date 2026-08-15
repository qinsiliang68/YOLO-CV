from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

from stage1_sctsr_v4.asset_registry import (
    AssetRecord,
    AssetRegistry,
    load_registered_split_labels,
)
from stage1_sctsr_v4.columnar import read_columnar
from stage1_sctsr_v4.dataset_content_ledger import (
    DATASET_CONTENT_DIGEST_ALGORITHM,
    build_dataset_content_ledger,
    validate_registered_dataset_content,
)
from stage1_sctsr_v4.dataset_disjointness import (
    CONTENT_EXCLUSION_FIELDS,
    derive_content_exclusion_rows,
    write_content_exclusion_assets,
)
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.prediction_runtime import load_registered_image_records
from stage1_sctsr_v4.serialization import sha256_file


def _write_png(path: Path, colour: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (7, 5), colour).save(path, format="PNG")


def _write_manifest(path: Path, split: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("split", "canonical_image_relpath"),
            lineterminator="\n",
        )
        writer.writeheader()
        for relative in rows:
            writer.writerow({"split": split, "canonical_image_relpath": relative})


def _manifest_record(
    repository: Path,
    path: Path,
    *,
    asset_id: str,
    role: str,
    label: int,
    split: str,
) -> AssetRecord:
    with path.open("r", encoding="utf-8", newline="") as handle:
        row_count = sum(1 for _ in csv.DictReader(handle))
    return AssetRecord(
        asset_id=asset_id,
        relative_path=path.relative_to(repository).as_posix(),
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        role=role,
        row_count=row_count,
        identity_column="canonical_image_relpath",
        constant_label=label,
        split_column="split",
        allowed_split_values=(split,),
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
        split_asset_ids=source.split_asset_ids,
        content_exclusion_asset_id=source.content_exclusion_asset_id,
    )


def test_same_image_bytes_across_train_and_val_fail_without_exact_overlay(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    dataset = tmp_path / "dataset"
    base_image = "Det/images/train/base.png"
    leaked_image = "Det/images/val_op/leaked.png"
    clean_image = "Det/images/val_op/clean.png"
    _write_png(dataset / base_image, (1, 2, 3))
    (dataset / leaked_image).parent.mkdir(parents=True, exist_ok=True)
    (dataset / leaked_image).write_bytes((dataset / base_image).read_bytes())
    _write_png(dataset / clean_image, (9, 8, 7))

    base_manifest = repository / "manifests/base.csv"
    op_manifest = repository / "manifests/op.csv"
    _write_manifest(base_manifest, "train", [base_image])
    _write_manifest(op_manifest, "val_op", [leaked_image, clean_image])
    source = AssetRegistry(
        schema_version="stage1.sctsr.asset_registry.v1",
        base_denominator=1,
        assets=(
            _manifest_record(
                repository,
                base_manifest,
                asset_id="base",
                role="CANONICAL_BASE_COMPONENT_DEFECT",
                label=1,
                split="train",
            ),
            _manifest_record(
                repository,
                op_manifest,
                asset_id="op",
                role="VAL_OP_COMPONENT_DEFECT_ENDPOINT_ONLY_NOT_SELECTION",
                label=1,
                split="val_op",
            ),
        ),
        base_asset_ids=("base",),
        split_asset_ids=(("val_op", ("op",)),),
    )
    ledger = repository / "assets/content.parquet"
    receipt = build_dataset_content_ledger(
        registry=source,
        repository_root=repository,
        dataset_root=dataset,
        output_path=ledger,
    )
    registry = _with_ledger(source, repository, ledger, receipt)

    with pytest.raises(SctsrError) as caught:
        validate_registered_dataset_content(
            registry=registry,
            repository_root=repository,
            dataset_root=dataset,
            verify_physical_files=True,
        )
    assert caught.value.code is ErrorCode.DATASET_SPLIT_CONTENT_LEAKAGE


def test_exact_derived_overlay_removes_leak_and_drives_effective_split(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    dataset = tmp_path / "dataset"
    base_image = "Det/images/train/base.png"
    leaked_image = "Det/images/val_op/leaked.png"
    clean_image = "Det/images/val_op/clean.png"
    _write_png(dataset / base_image, (1, 2, 3))
    (dataset / leaked_image).parent.mkdir(parents=True, exist_ok=True)
    (dataset / leaked_image).write_bytes((dataset / base_image).read_bytes())
    _write_png(dataset / clean_image, (9, 8, 7))
    base_manifest = repository / "manifests/base.csv"
    op_manifest = repository / "manifests/op.csv"
    _write_manifest(base_manifest, "train", [base_image])
    _write_manifest(op_manifest, "val_op", [leaked_image, clean_image])
    base_record = _manifest_record(
        repository,
        base_manifest,
        asset_id="base",
        role="CANONICAL_BASE_COMPONENT_DEFECT",
        label=1,
        split="train",
    )
    op_record = _manifest_record(
        repository,
        op_manifest,
        asset_id="op",
        role="VAL_OP_COMPONENT_DEFECT_ENDPOINT_ONLY_NOT_SELECTION",
        label=1,
        split="val_op",
    )
    source = AssetRegistry(
        schema_version="stage1.sctsr.asset_registry.v1",
        base_denominator=1,
        assets=(base_record, op_record),
        base_asset_ids=("base",),
        split_asset_ids=(("val_op", ("op",)),),
    )
    ledger = repository / "assets/content.parquet"
    receipt = build_dataset_content_ledger(
        registry=source,
        repository_root=repository,
        dataset_root=dataset,
        output_path=ledger,
    )
    ledger_rows = read_columnar(ledger)
    exclusions, audit = derive_content_exclusion_rows(ledger_rows)
    assert audit["excluded_evaluation_occurrences"] == 1
    assert exclusions[0]["sample_id"] == leaked_image

    exclusion_path = repository / "assets/content_exclusions.csv"
    with exclusion_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTENT_EXCLUSION_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(exclusions)
    exclusion_record = AssetRecord(
        asset_id="dataset_content_exclusions",
        relative_path=exclusion_path.relative_to(repository).as_posix(),
        sha256=sha256_file(exclusion_path),
        bytes=exclusion_path.stat().st_size,
        role="SCTSR_V4_EFFECTIVE_EVALUATION_CONTENT_EXCLUSIONS",
        row_count=1,
        identity_column="sample_id",
        label_column="y_true",
    )
    registered = AssetRegistry(
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
            exclusion_record,
        ),
        base_asset_ids=source.base_asset_ids,
        split_asset_ids=source.split_asset_ids,
        content_exclusion_asset_id="dataset_content_exclusions",
    )

    report = validate_registered_dataset_content(
        registry=registered,
        repository_root=repository,
        dataset_root=dataset,
        verify_physical_files=True,
    )
    labels = load_registered_split_labels(registered, repository, "val_op")
    endpoint_records = load_registered_image_records(
        registered,
        repository_root=repository,
        dataset_root=dataset,
        split_role="val_op",
    )
    assert report["status"] == "PASS"
    assert report["effective_split_counts"] == {"val_op": 1}
    assert report["cross_role_content_collisions_after_exclusion"] == 0
    assert labels == {clean_image: 1}
    assert [row.sample_id for row in endpoint_records] == [clean_image]


def test_overlay_must_equal_deterministic_fail_closed_derivation(tmp_path: Path) -> None:
    rows = [
        {
            "sample_id": "Det/images/train/a.png",
            "split_role": "train",
            "y_true": 1,
            "image_sha256": "A" * 64,
        },
        {
            "sample_id": "Det/images/val_op/b.png",
            "split_role": "val_op",
            "y_true": 1,
            "image_sha256": "A" * 64,
        },
    ]
    exclusions, audit = derive_content_exclusion_rows(rows)
    assert [row["sample_id"] for row in exclusions] == ["Det/images/val_op/b.png"]
    assert audit["base_internal_duplicate_groups"] == 0
    assert exclusions[0]["reason"] == "CONTENT_SHA_COLLIDES_WITH_HIGHER_PRIORITY_ROLE"


def test_content_exclusion_asset_writer_is_lf_stable_and_immutable(tmp_path: Path) -> None:
    rows = [
        {"sample_id": "Det/images/train/a.png", "split_role": "train", "y_true": 1, "image_sha256": "A" * 64},
        {"sample_id": "Det/images/val_op/a-copy.png", "split_role": "val_op", "y_true": 1, "image_sha256": "A" * 64},
    ]
    output = tmp_path / "content_exclusions.csv"
    receipt = tmp_path / "content_exclusions.receipt.json"

    report = write_content_exclusion_assets(
        rows,
        output_path=output,
        receipt_path=receipt,
        source_ledger_sha256="B" * 64,
    )

    assert report["row_count"] == 1
    assert b"\r\n" not in output.read_bytes()
    assert report["sha256"] == sha256_file(output)
    with pytest.raises(SctsrError) as caught:
        write_content_exclusion_assets(
            rows,
            output_path=output,
            receipt_path=receipt,
            source_ledger_sha256="B" * 64,
        )
    assert caught.value.code is ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE
