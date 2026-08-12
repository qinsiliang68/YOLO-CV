"""Exact manifest and live image-byte identity for Stage1 v3 inputs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable


class AssetIdentityError(ValueError):
    """Raised when manifests or live image bytes differ from the frozen input identity."""


@dataclass(frozen=True)
class ManifestImageRecord:
    sample_id: str
    filename: str
    y_true: int
    image_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class DatasetIdentity:
    status: str
    row_count: int
    defect_count: int
    normal_count: int
    image_content_digest: str
    total_image_bytes: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _read_class_manifest(
    path: Path,
    *,
    dataset_root: Path,
    expected_count: int,
    expected_label: int,
) -> list[ManifestImageRecord]:
    if not path.is_file():
        if expected_count == 0:
            return []
        raise AssetIdentityError(f"manifest is missing: {path}")
    records: list[ManifestImageRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"canonical_image_relpath", "Filename", "Defect"}
        if not required.issubset(reader.fieldnames or ()):
            raise AssetIdentityError(f"manifest columns are invalid: {path}")
        for row in reader:
            sample_id = str(row["canonical_image_relpath"]).strip().replace("\\", "/")
            filename = Path(str(row["Filename"])).name
            try:
                label = int(row["Defect"])
            except (TypeError, ValueError) as exc:
                raise AssetIdentityError(f"manifest label is invalid for {sample_id}") from exc
            if label != expected_label:
                raise AssetIdentityError(
                    f"manifest label differs from class contract for {sample_id}: {label} != {expected_label}"
                )
            image = (dataset_root / Path(sample_id)).resolve()
            try:
                image.relative_to(dataset_root)
            except ValueError as exc:
                raise AssetIdentityError(f"manifest image path escapes dataset root: {sample_id}") from exc
            if image.name != filename:
                raise AssetIdentityError(
                    f"manifest filename differs from canonical path for {sample_id}: {filename}"
                )
            if not image.is_file():
                raise AssetIdentityError(f"manifest image is missing: {image}")
            records.append(ManifestImageRecord(sample_id, filename, label, image, path))
    if len(records) != expected_count:
        raise AssetIdentityError(
            f"manifest row count mismatch for {path}: expected {expected_count}, observed {len(records)}"
        )
    return records


def load_manifest_records(
    *,
    dataset_root: str | Path,
    defect_manifest: str | Path,
    normal_manifest: str | Path,
    expected_defect: int = 60_000,
    expected_normal: int = 60_000,
) -> tuple[ManifestImageRecord, ...]:
    root = Path(dataset_root).resolve()
    rows = _read_class_manifest(
        Path(defect_manifest).resolve(),
        dataset_root=root,
        expected_count=int(expected_defect),
        expected_label=1,
    ) + _read_class_manifest(
        Path(normal_manifest).resolve(),
        dataset_root=root,
        expected_count=int(expected_normal),
        expected_label=0,
    )
    identifiers = [row.sample_id for row in rows]
    paths = [row.image_path for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise AssetIdentityError("manifest sample identity is duplicated across classes")
    if len(paths) != len(set(paths)):
        raise AssetIdentityError("manifest image path is duplicated across classes")
    return tuple(rows)


def build_live_dataset_identity(records: Iterable[ManifestImageRecord]) -> DatasetIdentity:
    rows = tuple(records)
    digest = hashlib.sha256()
    total_bytes = 0
    for row in sorted(rows, key=lambda item: item.sample_id):
        if not row.image_path.is_file():
            raise AssetIdentityError(f"live image is missing: {row.image_path}")
        size = row.image_path.stat().st_size
        file_digest = _sha256(row.image_path)
        digest.update(f"{row.sample_id}|{row.y_true}|{size}|{file_digest}\n".encode("utf-8"))
        total_bytes += size
    return DatasetIdentity(
        status="PASS",
        row_count=len(rows),
        defect_count=sum(row.y_true == 1 for row in rows),
        normal_count=sum(row.y_true == 0 for row in rows),
        image_content_digest=digest.hexdigest().upper(),
        total_image_bytes=total_bytes,
    )


def validate_live_dataset_identity(
    records: Iterable[ManifestImageRecord], *, expected_digest: str
) -> DatasetIdentity:
    identity = build_live_dataset_identity(records)
    if identity.image_content_digest.lower() != str(expected_digest).strip().lower():
        raise AssetIdentityError(
            "live image content identity mismatch: "
            f"expected {expected_digest}, observed {identity.image_content_digest}"
        )
    return identity


__all__ = [
    "AssetIdentityError",
    "DatasetIdentity",
    "ManifestImageRecord",
    "build_live_dataset_identity",
    "load_manifest_records",
    "validate_live_dataset_identity",
]
