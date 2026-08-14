from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, UnidentifiedImageError

from .asset_registry import AssetRecord, AssetRegistry
from .columnar import StreamingZstdParquetWriter, read_columnar, validate_columnar_file
from .errors import ErrorCode, SctsrError
from .serialization import sha256_file, stable_digest


DATASET_CONTENT_SCHEMA_VERSION = "stage1.sctsr.dataset_content_ledger.v1"
DATASET_CONTENT_DIGEST_ALGORITHM = "SHA256_SORTED_DATASET_CONTENT_ROWS_V1"
DATASET_CONTENT_ASSET_ID = "dataset_content_ledger"
DATASET_CONTENT_ROLE = "DATASET_CONTENT_LEDGER_ALL_REGISTERED_NON_TEST_IMAGES"


@dataclass(frozen=True, slots=True)
class DatasetContentRow:
    sample_id: str
    split_role: str
    y_true: int
    manifest_asset_id: str
    manifest_sha256: str
    canonical_image_relpath: str
    image_bytes: int
    image_sha256: str
    image_width: int
    image_height: int
    image_mode: str
    image_format: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "split_role": self.split_role,
            "y_true": self.y_true,
            "manifest_asset_id": self.manifest_asset_id,
            "manifest_sha256": self.manifest_sha256,
            "canonical_image_relpath": self.canonical_image_relpath,
            "image_bytes": self.image_bytes,
            "image_sha256": self.image_sha256,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_mode": self.image_mode,
            "image_format": self.image_format,
        }


@dataclass(frozen=True, slots=True)
class _ExpectedIdentity:
    sample_id: str
    split_role: str
    y_true: int
    manifest_asset_id: str
    manifest_sha256: str
    canonical_image_relpath: str


def dataset_content_schema() -> Any:
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - exercised by the dependency gate.
        raise SctsrError(ErrorCode.COLUMNAR_ENGINE_UNAVAILABLE, "Dataset content ledger requires the locked PyArrow dependency") from exc
    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("split_role", pa.string(), nullable=False),
            pa.field("y_true", pa.int8(), nullable=False),
            pa.field("manifest_asset_id", pa.string(), nullable=False),
            pa.field("manifest_sha256", pa.string(), nullable=False),
            pa.field("canonical_image_relpath", pa.string(), nullable=False),
            pa.field("image_bytes", pa.int64(), nullable=False),
            pa.field("image_sha256", pa.string(), nullable=False),
            pa.field("image_width", pa.int32(), nullable=False),
            pa.field("image_height", pa.int32(), nullable=False),
            pa.field("image_mode", pa.string(), nullable=False),
            pa.field("image_format", pa.string(), nullable=False),
        ]
    )


def registered_dataset_manifest_asset_ids(registry: AssetRegistry) -> tuple[str, ...]:
    ordered: list[str] = []
    for asset_id in registry.base_asset_ids:
        if asset_id not in ordered:
            ordered.append(asset_id)
    for split_role, asset_ids in registry.split_asset_ids:
        if split_role.lower() in {"test", "blind_test", "blind_holdout", "holdout_test"}:
            raise SctsrError(ErrorCode.TEST_ACCESS_FORBIDDEN, "Dataset content ledger may not include test or blind identities")
        for asset_id in asset_ids:
            if asset_id not in ordered:
                ordered.append(asset_id)
    if not ordered:
        raise SctsrError(ErrorCode.DATASET_CONTENT_LEDGER_REQUIRED, "Asset registry declares no canonical dataset manifests")
    return tuple(ordered)


def _normalised_relative_image_path(value: Any, *, artifact_path: Path) -> str:
    token = str(value or "").replace("\\", "/").strip()
    pure = PurePosixPath(token)
    if not token or pure.is_absolute() or ".." in pure.parts or pure.parts[0].endswith(":"):
        raise SctsrError(
            ErrorCode.DATASET_CONTENT_MISMATCH,
            "Dataset manifest contains an unsafe or empty canonical image path",
            observed=token,
            artifact_path=str(artifact_path),
        )
    canonical = pure.as_posix()
    if canonical != token:
        raise SctsrError(
            ErrorCode.DATASET_CONTENT_MISMATCH,
            "Dataset manifest image path is not canonical POSIX-relative form",
            observed=token,
            expected=canonical,
            artifact_path=str(artifact_path),
        )
    return canonical


def _manifest_records(
    registry: AssetRegistry,
    repository_root: Path,
    manifest_asset_ids: Sequence[str],
) -> tuple[list[_ExpectedIdentity], list[dict[str, Any]]]:
    by_id = {record.asset_id: record for record in registry.assets}
    expected: list[_ExpectedIdentity] = []
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for asset_id in manifest_asset_ids:
        record = by_id.get(asset_id)
        if record is None:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Dataset manifest asset is absent from the registry", observed=asset_id)
        if record.identity_column != "canonical_image_relpath" or Path(record.relative_path).suffix.lower() != ".csv":
            raise SctsrError(
                ErrorCode.ASSET_VALIDATION_FAILED,
                "Dataset content source must be a registered canonical-image CSV manifest",
                observed=asset_id,
            )
        path = (repository_root / record.relative_path).resolve()
        try:
            path.relative_to(repository_root)
        except ValueError as exc:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Dataset manifest escapes repository root", artifact_path=str(path)) from exc
        if not path.is_file() or path.stat().st_size != record.bytes or sha256_file(path) != record.sha256:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Dataset manifest bytes differ from the asset registry", artifact_path=str(path))
        count = 0
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            required = {record.identity_column}
            if record.constant_label is None:
                if not record.label_column:
                    raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Dataset manifest has no registered binary label source", observed=asset_id)
                required.add(record.label_column)
            if record.split_column:
                required.add(record.split_column)
            missing = sorted(required - fields)
            if missing:
                raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Dataset manifest columns are missing", observed=missing, artifact_path=str(path))
            for row in reader:
                count += 1
                relative = _normalised_relative_image_path(row[record.identity_column], artifact_path=path)
                if relative in seen:
                    raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "Registered dataset manifests contain the same canonical image twice", observed=relative)
                seen.add(relative)
                raw_label = record.constant_label if record.constant_label is not None else row[record.label_column or ""]
                try:
                    label = int(raw_label)
                except (TypeError, ValueError) as exc:
                    raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Dataset manifest label is not an integer", observed=raw_label) from exc
                if label not in (0, 1):
                    raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Dataset content ledger supports only binary labels", observed=label)
                split = str(row.get(record.split_column or "", "UNREGISTERED"))
                if record.allowed_split_values and split not in record.allowed_split_values:
                    raise SctsrError(
                        ErrorCode.ASSET_VALIDATION_FAILED,
                        "Dataset manifest row has a split role outside its registry declaration",
                        observed=split,
                        expected=record.allowed_split_values,
                    )
                expected.append(
                    _ExpectedIdentity(
                        sample_id=relative,
                        split_role=split,
                        y_true=label,
                        manifest_asset_id=asset_id,
                        manifest_sha256=record.sha256,
                        canonical_image_relpath=relative,
                    )
                )
        if record.row_count is not None and count != record.row_count:
            raise SctsrError(
                ErrorCode.ASSET_VALIDATION_FAILED,
                "Dataset manifest row count differs from the registry",
                observed=count,
                expected=record.row_count,
                artifact_path=str(path),
            )
        evidence.append(
            {
                "asset_id": asset_id,
                "relative_path": record.relative_path,
                "bytes": record.bytes,
                "sha256": record.sha256,
                "row_count": count,
            }
        )
    return expected, evidence


def _physical_path(dataset_root: Path, relative: str) -> Path:
    path = (dataset_root / PurePosixPath(relative)).resolve()
    try:
        path.relative_to(dataset_root)
    except ValueError as exc:
        raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Canonical image path escapes the prepared dataset root", observed=relative) from exc
    return path


def _image_metadata(path: Path) -> tuple[int, int, str, str]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            mode = str(image.mode)
            image_format = str(image.format or "UNKNOWN").upper()
            image.verify()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Registered dataset image cannot be decoded and verified", artifact_path=str(path)) from exc
    if width <= 0 or height <= 0 or not mode or image_format == "UNKNOWN":
        raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Registered dataset image metadata is invalid", artifact_path=str(path))
    return width, height, mode, image_format


def _digest_row(row: Mapping[str, Any]) -> bytes:
    values = (
        row["sample_id"],
        row["split_role"],
        int(row["y_true"]),
        row["manifest_asset_id"],
        row["manifest_sha256"],
        row["canonical_image_relpath"],
        int(row["image_bytes"]),
        row["image_sha256"],
        int(row["image_width"]),
        int(row["image_height"]),
        row["image_mode"],
        row["image_format"],
    )
    return ("\x1f".join(str(value) for value in values) + "\n").encode("utf-8")


def dataset_content_identity_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    ordered = sorted((dict(row) for row in rows), key=lambda row: str(row["sample_id"]))
    digest = hashlib.sha256()
    for row in ordered:
        digest.update(_digest_row(row))
    return digest.hexdigest().upper()


def build_dataset_content_ledger(
    *,
    registry: AssetRegistry,
    repository_root: str | Path,
    dataset_root: str | Path,
    output_path: str | Path,
    manifest_asset_ids: Sequence[str] | None = None,
    batch_rows: int = 2_048,
) -> dict[str, Any]:
    """Build one immutable, decoded, byte-level ledger from registered manifests.

    This is an asset-construction operation, not a training operation.  It
    never discovers files by glob: every physical image must be named by one
    registered manifest row.
    """

    repository = Path(repository_root).resolve()
    data = Path(dataset_root).resolve()
    if not data.is_dir():
        raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Dataset root is missing", artifact_path=str(data))
    ids = tuple(manifest_asset_ids or registered_dataset_manifest_asset_ids(registry))
    expected, manifest_evidence = _manifest_records(registry, repository, ids)
    destination = Path(output_path)
    digest_rows: list[dict[str, Any]] = []
    image_bytes_total = 0
    writer = StreamingZstdParquetWriter(
        destination,
        schema_version=DATASET_CONTENT_SCHEMA_VERSION,
        schema=dataset_content_schema(),
        require_run_epoch_partition=False,
    )
    pending: list[dict[str, Any]] = []
    try:
        for identity in expected:
            path = _physical_path(data, identity.canonical_image_relpath)
            if not path.is_file():
                raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Registered dataset image is missing", artifact_path=str(path))
            image_bytes = path.stat().st_size
            image_sha256 = sha256_file(path)
            width, height, mode, image_format = _image_metadata(path)
            row = DatasetContentRow(
                sample_id=identity.sample_id,
                split_role=identity.split_role,
                y_true=identity.y_true,
                manifest_asset_id=identity.manifest_asset_id,
                manifest_sha256=identity.manifest_sha256,
                canonical_image_relpath=identity.canonical_image_relpath,
                image_bytes=image_bytes,
                image_sha256=image_sha256,
                image_width=width,
                image_height=height,
                image_mode=mode,
                image_format=image_format,
            ).as_dict()
            image_bytes_total += image_bytes
            pending.append(row)
            digest_rows.append(row)
            if len(pending) >= batch_rows:
                writer.append(pending)
                pending.clear()
        if pending:
            writer.append(pending)
        manifest = writer.close()
    except BaseException:
        writer.abort()
        writer.temp.unlink(missing_ok=True)
        raise
    content_digest = dataset_content_identity_digest(digest_rows)
    receipt = {
        "schema_version": DATASET_CONTENT_SCHEMA_VERSION,
        "status": "BUILT_NOT_FORMAL_TRAINING",
        "ledger_path": destination.resolve().as_posix(),
        "row_count": manifest.row_count,
        "bytes": manifest.bytes,
        "sha256": manifest.sha256,
        "storage_format": manifest.storage_format,
        "compression": manifest.compression,
        "schema_digest": manifest.schema_digest,
        "parquet_metadata_digest": manifest.parquet_metadata_digest,
        "content_identity_digest_algorithm": DATASET_CONTENT_DIGEST_ALGORITHM,
        "content_identity_digest": content_digest,
        "manifest_asset_ids": list(ids),
        "manifest_evidence": manifest_evidence,
        "image_bytes_total": image_bytes_total,
        "decoded_image_count": len(digest_rows),
        "generation_dataset_root": data.as_posix(),
        "formal_training_started": False,
        "blind_holdout_opened": False,
        "test_accessed": False,
    }
    return {**receipt, "receipt_digest": stable_digest(receipt)}


def _validate_ledger_rows(rows: Sequence[Mapping[str, Any]], record: AssetRecord) -> tuple[dict[str, Mapping[str, Any]], str]:
    expected_fields = set(dataset_content_schema().names)
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(rows):
        if set(raw) != expected_fields:
            raise SctsrError(
                ErrorCode.SCHEMA_VALIDATION_FAILED,
                "Dataset content ledger row fields differ from the registered schema",
                failing_field=f"row[{index}]",
            )
        row = dict(raw)
        sample_id = _normalised_relative_image_path(row["sample_id"], artifact_path=Path(record.relative_path))
        if row["canonical_image_relpath"] != sample_id:
            raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Dataset ledger sample ID and canonical image path differ", observed=sample_id)
        if sample_id in by_id:
            raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "Dataset content ledger contains a duplicate sample ID", observed=sample_id)
        if int(row["y_true"]) not in (0, 1) or int(row["image_bytes"]) <= 0:
            raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Dataset content ledger row has invalid label or byte count", observed=sample_id)
        for field in ("manifest_sha256", "image_sha256"):
            token = str(row[field]).upper()
            if len(token) != 64 or any(character not in "0123456789ABCDEF" for character in token):
                raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Dataset content ledger contains an invalid SHA-256", failing_field=field, observed=sample_id)
        by_id[sample_id] = row
    digest = dataset_content_identity_digest(rows)
    if record.row_count is None or len(rows) != record.row_count:
        raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Dataset content ledger row count differs from its registry record", observed=len(rows), expected=record.row_count)
    if record.identity_digest_algorithm != DATASET_CONTENT_DIGEST_ALGORITHM or digest != str(record.identity_digest or "").upper():
        raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Dataset content identity digest differs from the registry", observed=digest, expected=record.identity_digest)
    return by_id, digest


def validate_registered_dataset_content(
    *,
    registry: AssetRegistry,
    repository_root: str | Path,
    dataset_root: str | Path,
    required_manifest_asset_ids: Sequence[str] | None = None,
    verify_physical_files: bool = True,
) -> dict[str, Any]:
    """Fail closed when physical images differ from the frozen byte ledger."""

    repository = Path(repository_root).resolve()
    data = Path(dataset_root).resolve()
    records = [record for record in registry.assets if record.asset_id == DATASET_CONTENT_ASSET_ID]
    if len(records) != 1:
        raise SctsrError(ErrorCode.DATASET_CONTENT_LEDGER_REQUIRED, "Formal SCTSR execution requires exactly one registered dataset content ledger")
    record = records[0]
    if record.role != DATASET_CONTENT_ROLE:
        raise SctsrError(ErrorCode.DATASET_CONTENT_LEDGER_REQUIRED, "Registered dataset content ledger has the wrong evidence role", observed=record.role)
    ledger = (repository / record.relative_path).resolve()
    try:
        ledger.relative_to(repository)
    except ValueError as exc:
        raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Dataset content ledger escapes repository root", artifact_path=str(ledger)) from exc
    if not ledger.is_file() or ledger.stat().st_size != record.bytes or sha256_file(ledger) != record.sha256:
        raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Dataset content ledger bytes differ from the asset registry", artifact_path=str(ledger))
    columnar = validate_columnar_file(
        ledger,
        expected_rows=record.row_count,
        expected_schema_version=DATASET_CONTENT_SCHEMA_VERSION,
        expected_sha256=record.sha256,
    )
    rows = read_columnar(ledger)
    ledger_by_id, content_digest = _validate_ledger_rows(rows, record)
    ids = tuple(required_manifest_asset_ids or registered_dataset_manifest_asset_ids(registry))
    expected, manifest_evidence = _manifest_records(registry, repository, ids)
    expected_ids = {identity.sample_id for identity in expected}
    missing = sorted(expected_ids - set(ledger_by_id))
    if missing:
        raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Dataset content ledger omits registered manifest identities", observed=missing[:20])
    physical_files_verified = 0
    physical_bytes_verified = 0
    for identity in expected:
        row = ledger_by_id[identity.sample_id]
        semantic = (
            row["sample_id"],
            row["split_role"],
            int(row["y_true"]),
            row["manifest_asset_id"],
            str(row["manifest_sha256"]).upper(),
            row["canonical_image_relpath"],
        )
        expected_semantic = (
            identity.sample_id,
            identity.split_role,
            identity.y_true,
            identity.manifest_asset_id,
            identity.manifest_sha256,
            identity.canonical_image_relpath,
        )
        if semantic != expected_semantic:
            raise SctsrError(
                ErrorCode.DATASET_CONTENT_MISMATCH,
                "Dataset content ledger semantic row differs from the registered manifest",
                observed={"sample_id": identity.sample_id, "values": semantic},
                expected=expected_semantic,
            )
        if verify_physical_files:
            path = _physical_path(data, identity.canonical_image_relpath)
            if not path.is_file():
                raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Physical dataset image is missing", artifact_path=str(path))
            observed_bytes = path.stat().st_size
            if observed_bytes != int(row["image_bytes"]):
                raise SctsrError(
                    ErrorCode.DATASET_CONTENT_MISMATCH,
                    "Physical dataset image byte count differs from the frozen ledger",
                    artifact_path=str(path),
                    observed=observed_bytes,
                    expected=int(row["image_bytes"]),
                )
            observed_sha = sha256_file(path)
            if observed_sha != str(row["image_sha256"]).upper():
                raise SctsrError(
                    ErrorCode.DATASET_CONTENT_MISMATCH,
                    "Physical dataset image SHA-256 differs from the frozen ledger",
                    artifact_path=str(path),
                    observed=observed_sha,
                    expected=row["image_sha256"],
                )
            physical_files_verified += 1
            physical_bytes_verified += observed_bytes
    return {
        "schema_version": "stage1.sctsr.dataset_content_validation.v1",
        "status": "PASS",
        "dataset_root": data.as_posix(),
        "ledger_path": ledger.as_posix(),
        "ledger_bytes": record.bytes,
        "ledger_sha256": record.sha256,
        "ledger_row_count": len(rows),
        "content_identity_digest": content_digest,
        "manifest_asset_ids": list(ids),
        "manifest_evidence": manifest_evidence,
        "physical_verification_enabled": verify_physical_files,
        "physical_files_verified": physical_files_verified,
        "physical_bytes_verified": physical_bytes_verified,
        "storage_format": "PARQUET_ZSTD",
        "compression": columnar["compression"],
        "test_accessed": False,
        "blind_holdout_opened": False,
        "validation_digest": stable_digest(
            {
                "ledger_sha256": record.sha256,
                "content_identity_digest": content_digest,
                "manifest_asset_ids": list(ids),
                "physical_files_verified": physical_files_verified,
                "physical_bytes_verified": physical_bytes_verified,
            }
        ),
    }
