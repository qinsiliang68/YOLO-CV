from __future__ import annotations

import json
import os
import re
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import ErrorCode, SctsrError
from .serialization import _fsync_directory, atomic_write_bytes, canonical_json_bytes, sha256_file, stable_digest

PORTABLE_MAGIC = b"SCTSR_SYNTHETIC_COLUMNAR_V1\n"
_RUN_PARTITION = re.compile(r"^run_id=(?P<value>[^/\\=]+)$")
_EPOCH_PARTITION = re.compile(r"^epoch=(?P<value>\d{4})$")


@dataclass(frozen=True, slots=True)
class ColumnarManifest:
    path: str
    schema_version: str
    schema_digest: str
    row_count: int
    bytes: int
    sha256: str
    storage_format: str
    canonical_parquet: bool
    compression: str
    run_id: str | None
    epoch: int | None
    parquet_metadata_digest: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class StreamingZstdParquetWriter:
    """Append strict record batches and publish one immutable Parquet part."""

    def __init__(
        self,
        path: str | Path,
        *,
        schema_version: str,
        schema: Any,
        require_run_epoch_partition: bool = True,
    ) -> None:
        engine = _pyarrow_modules()
        if engine is None:
            raise SctsrError(ErrorCode.COLUMNAR_ENGINE_UNAVAILABLE, "Streaming evidence requires PyArrow")
        self.pa, self.pq = engine
        self.destination = Path(path)
        self.run_id, self.epoch = partition_identity(self.destination, required=require_run_epoch_partition)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        if self.destination.exists():
            raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Streaming Parquet destination already exists", artifact_path=str(self.destination))
        self.schema_version = schema_version
        metadata = dict(schema.metadata or {})
        metadata[b"sctsr_schema_version"] = schema_version.encode("utf-8")
        metadata[b"sctsr_semantic"] = b"CANONICAL_ZSTD_PARQUET"
        self.schema = schema.with_metadata(metadata)
        # The epoch generation directory is already unique and immutable.  A
        # UUID in every temporary filename only pushes otherwise valid Windows
        # paths beyond MAX_PATH for PyArrow's native file opener.  Refuse a
        # stale sibling instead of inventing a second writer identity.
        self.temp = self.destination.with_name(f".{self.destination.name}.inprogress")
        if self.temp.exists():
            raise SctsrError(
                ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE,
                "Streaming Parquet temporary path already exists",
                artifact_path=str(self.temp),
            )
        self.writer = self.pq.ParquetWriter(self.temp, self.schema, compression="zstd", use_dictionary=True, write_statistics=True)
        self.row_count = 0
        self.closed = False

    def append(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if self.closed:
            raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Cannot append to a closed Parquet writer")
        if not rows:
            return
        _validate_exact_columns(rows, self.schema)
        try:
            table = self.pa.Table.from_pylist([dict(row) for row in rows], schema=self.schema)
            self.writer.write_table(table)
        except (TypeError, ValueError, OverflowError, self.pa.ArrowException) as exc:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Streaming rows cannot be represented by the strict Arrow schema") from exc
        self.row_count += len(rows)

    def close(self) -> ColumnarManifest:
        if self.closed:
            raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Streaming Parquet writer was already closed")
        self.writer.close()
        self.closed = True
        if self.row_count == 0:
            self.temp.unlink(missing_ok=True)
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Canonical streaming Parquet partition may not be empty")
        with self.temp.open("r+b") as handle:
            os.fsync(handle.fileno())
        report = _parquet_report(self.temp)
        if int(report["num_rows"]) != self.row_count or report["schema_version"] != self.schema_version:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Streaming Parquet close validation failed")
        os.replace(self.temp, self.destination)
        _fsync_directory(self.destination.parent)
        return ColumnarManifest(
            path=self.destination.as_posix(),
            schema_version=self.schema_version,
            schema_digest=report["schema_digest"],
            row_count=self.row_count,
            bytes=self.destination.stat().st_size,
            sha256=sha256_file(self.destination),
            storage_format="PARQUET_ZSTD",
            canonical_parquet=True,
            compression="ZSTD",
            run_id=self.run_id,
            epoch=self.epoch,
            parquet_metadata_digest=report["parquet_metadata_digest"],
        )

    def abort(self) -> None:
        if not self.closed:
            self.writer.close()
            self.closed = True

    def __enter__(self) -> "StreamingZstdParquetWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.abort()
        elif not self.closed:
            self.close()
        return False


def _pyarrow_modules():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        return pa, pq
    except ImportError:
        return None


def parquet_engine_available() -> bool:
    return _pyarrow_modules() is not None


def partition_identity(path: str | Path, *, required: bool = False) -> tuple[str | None, int | None]:
    parts = Path(path).parts
    run_parts = [match.group("value") for part in parts if (match := _RUN_PARTITION.match(part))]
    epoch_parts = [int(match.group("value")) for part in parts if (match := _EPOCH_PARTITION.match(part))]
    if len(run_parts) > 1 or len(epoch_parts) > 1:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Ambiguous run/epoch Parquet partition path", artifact_path=str(path))
    run_id = run_parts[0] if run_parts else None
    epoch = epoch_parts[0] if epoch_parts else None
    if required and (run_id is None or epoch is None):
        raise SctsrError(
            ErrorCode.SCHEMA_VALIDATION_FAILED,
            "Large evidence tables must be partitioned by run_id=<id>/epoch=<eeee>",
            artifact_path=str(path),
        )
    return run_id, epoch


def _schema_descriptor(schema: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": bool(field.nullable),
            "metadata": {
                key.decode("utf-8", "replace"): value.decode("utf-8", "replace")
                for key, value in sorted((field.metadata or {}).items())
            },
        }
        for field in schema
    ]


def schema_digest(schema: Any) -> str:
    return stable_digest(_schema_descriptor(schema))


def _validate_exact_columns(rows: Sequence[Mapping[str, Any]], schema: Any) -> None:
    expected = tuple(schema.names)
    expected_set = set(expected)
    if not rows:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Canonical Parquet partitions may not be empty")
    for index, row in enumerate(rows):
        observed = set(row)
        if observed != expected_set:
            raise SctsrError(
                ErrorCode.SCHEMA_VALIDATION_FAILED,
                "Row fields do not exactly match the registered Parquet schema",
                failing_field=f"row[{index}]",
                observed={"missing": sorted(expected_set - observed), "extra": sorted(observed - expected_set)},
                expected=list(expected),
            )


def _parquet_report(path: Path) -> dict[str, Any]:
    engine = _pyarrow_modules()
    if engine is None:
        raise SctsrError(ErrorCode.COLUMNAR_ENGINE_UNAVAILABLE, "PyArrow is required to inspect canonical Parquet")
    _, pq = engine
    try:
        with path.open("rb") as handle:
            parquet_file = pq.ParquetFile(handle)
            metadata = parquet_file.metadata
            compressions = {
                str(metadata.row_group(group).column(column).compression).upper()
                for group in range(metadata.num_row_groups)
                for column in range(metadata.row_group(group).num_columns)
            }
            arrow_schema = parquet_file.schema_arrow
    except Exception as exc:
        raise SctsrError(
            ErrorCode.SCHEMA_VALIDATION_FAILED,
            "Parquet file is unreadable or incomplete",
            artifact_path=str(path),
        ) from exc
    if metadata.num_rows and compressions != {"ZSTD"}:
        raise SctsrError(
            ErrorCode.SCHEMA_VALIDATION_FAILED,
            "Canonical Parquet must use Zstd for every physical column",
            observed=sorted(compressions),
            expected=["ZSTD"],
            artifact_path=str(path),
        )
    semantic = (arrow_schema.metadata or {}).get(b"sctsr_semantic")
    version = (arrow_schema.metadata or {}).get(b"sctsr_schema_version")
    if semantic != b"CANONICAL_ZSTD_PARQUET" or version is None:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "SCTSR Parquet metadata is missing", artifact_path=str(path))
    descriptor = {
        "created_by": metadata.created_by,
        "num_columns": metadata.num_columns,
        "num_rows": metadata.num_rows,
        "num_row_groups": metadata.num_row_groups,
        "schema_version": version.decode("utf-8"),
        "schema_digest": schema_digest(arrow_schema),
        "compression": "ZSTD",
    }
    descriptor["parquet_metadata_digest"] = stable_digest(descriptor)
    return descriptor


def write_zstd_parquet(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    schema_version: str,
    schema: Any | None = None,
    require_run_epoch_partition: bool = False,
    allow_synthetic_portable_fallback: bool | None = None,
) -> ColumnarManifest:
    destination = Path(path)
    run_id, epoch = partition_identity(destination, required=require_run_epoch_partition)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Canonical Parquet destination already exists", artifact_path=str(destination))
    engine = _pyarrow_modules()
    metadata_digest: str | None = None
    if engine is not None:
        pa, pq = engine
        if schema is not None:
            _validate_exact_columns(rows, schema)
            try:
                table = pa.Table.from_pylist([dict(row) for row in rows], schema=schema)
            except (TypeError, ValueError, OverflowError, pa.ArrowException) as exc:
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Rows cannot be represented by the strict Arrow schema") from exc
        else:
            if not rows:
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Cannot infer a schema from an empty partition")
            table = pa.Table.from_pylist([dict(row) for row in rows])
        metadata = dict(table.schema.metadata or {})
        metadata[b"sctsr_schema_version"] = schema_version.encode("utf-8")
        metadata[b"sctsr_semantic"] = b"CANONICAL_ZSTD_PARQUET"
        table = table.replace_schema_metadata(metadata)
        temp = destination.with_name(f".{destination.name}.inprogress")
        if temp.exists():
            raise SctsrError(
                ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE,
                "Parquet temporary path already exists",
                artifact_path=str(temp),
            )
        try:
            pq.write_table(table, temp, compression="zstd", use_dictionary=True, write_statistics=True)
            with temp.open("r+b") as handle:
                os.fsync(handle.fileno())
            report = _parquet_report(temp)
            if int(report["num_rows"]) != len(rows) or report["schema_version"] != schema_version:
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Written Parquet metadata does not match the requested partition")
            os.replace(temp, destination)
            _fsync_directory(destination.parent)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        storage_format = "PARQUET_ZSTD"
        compression = "ZSTD"
        observed_schema_digest = report["schema_digest"]
        metadata_digest = report["parquet_metadata_digest"]
    else:
        allow = allow_synthetic_portable_fallback
        if allow is None:
            allow = os.environ.get("SCTSR_ALLOW_SYNTHETIC_COLUMNAR_FALLBACK") == "1"
        if not allow:
            raise SctsrError(
                ErrorCode.COLUMNAR_ENGINE_UNAVAILABLE,
                "PyArrow is required for canonical Zstd Parquet; portable fallback is synthetic-only and disabled",
                required_action="Install the locked pyarrow dependency in the Python 3.11/3.12 project environment",
            )
        payload = canonical_json_bytes(
            {
                "schema_version": schema_version,
                "semantic": "SYNTHETIC_PORTABLE_COLUMNAR_NOT_PARQUET",
                "rows": [dict(row) for row in rows],
            }
        )
        atomic_write_bytes(destination, PORTABLE_MAGIC + zlib.compress(payload, level=9))
        storage_format = "SYNTHETIC_PORTABLE_COLUMNAR_NOT_PARQUET"
        compression = "ZLIB_SYNTHETIC_ONLY"
        observed_schema_digest = stable_digest(sorted({key for row in rows for key in row}))
    return ColumnarManifest(
        path=destination.as_posix(),
        schema_version=schema_version,
        schema_digest=observed_schema_digest,
        row_count=len(rows),
        bytes=destination.stat().st_size,
        sha256=sha256_file(destination),
        storage_format=storage_format,
        canonical_parquet=storage_format == "PARQUET_ZSTD",
        compression=compression,
        run_id=run_id,
        epoch=epoch,
        parquet_metadata_digest=metadata_digest,
    )


def read_columnar(path: str | Path, *, allow_synthetic_portable_fallback: bool = False) -> list[dict[str, Any]]:
    source = Path(path)
    prefix = source.read_bytes()[: len(PORTABLE_MAGIC)]
    if prefix == PORTABLE_MAGIC:
        if not allow_synthetic_portable_fallback:
            raise SctsrError(ErrorCode.SYNTHETIC_RESULT_MISLABELLED, "Portable synthetic columnar file may not be read as canonical Parquet")
        payload = json.loads(zlib.decompress(source.read_bytes()[len(PORTABLE_MAGIC) :]).decode("utf-8"))
        if payload.get("semantic") != "SYNTHETIC_PORTABLE_COLUMNAR_NOT_PARQUET":
            raise SctsrError(ErrorCode.SYNTHETIC_RESULT_MISLABELLED, "Portable columnar semantic marker is missing")
        return list(payload["rows"])
    engine = _pyarrow_modules()
    if engine is None:
        raise SctsrError(ErrorCode.COLUMNAR_ENGINE_UNAVAILABLE, "PyArrow is required to read canonical Parquet")
    _, pq = engine
    try:
        return pq.read_table(source).to_pylist()
    except Exception as exc:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Canonical Parquet cannot be read", artifact_path=str(source)) from exc


def validate_columnar_file(
    path: str | Path,
    *,
    expected_rows: int | None = None,
    expected_schema_version: str | None = None,
    expected_schema_digest: str | None = None,
    expected_sha256: str | None = None,
    allow_synthetic_portable_fallback: bool = False,
) -> dict[str, Any]:
    source = Path(path)
    observed_sha = sha256_file(source)
    if expected_sha256 is not None and observed_sha != expected_sha256:
        raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Columnar SHA-256 mismatch", observed=observed_sha, expected=expected_sha256)
    prefix = source.read_bytes()[: len(PORTABLE_MAGIC)]
    if prefix == PORTABLE_MAGIC:
        rows = read_columnar(source, allow_synthetic_portable_fallback=allow_synthetic_portable_fallback)
        if expected_rows is not None and len(rows) != expected_rows:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Columnar row count mismatch", observed=len(rows), expected=expected_rows)
        return {
            "status": "PASS_SYNTHETIC_ONLY",
            "row_count": len(rows),
            "sha256": observed_sha,
            "rows_digest": stable_digest(rows),
            "compression": "ZLIB_SYNTHETIC_ONLY",
        }
    try:
        with source.open("rb") as handle:
            start = handle.read(4)
            handle.seek(-4, os.SEEK_END)
            end = handle.read(4)
        if start != b"PAR1" or end != b"PAR1":
            raise ValueError("Parquet boundary magic is missing")
    except (OSError, ValueError) as exc:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Parquet file is truncated or has invalid boundary magic", artifact_path=str(source)) from exc
    report = _parquet_report(source)
    if expected_rows is not None and int(report["num_rows"]) != expected_rows:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Columnar row count mismatch", observed=report["num_rows"], expected=expected_rows)
    if expected_schema_version is not None and report["schema_version"] != expected_schema_version:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Columnar schema version mismatch", observed=report["schema_version"], expected=expected_schema_version)
    if expected_schema_digest is not None and report["schema_digest"] != expected_schema_digest:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Columnar schema digest mismatch", observed=report["schema_digest"], expected=expected_schema_digest)
    return {
        "status": "PASS",
        "row_count": int(report["num_rows"]),
        "sha256": observed_sha,
        "schema_version": report["schema_version"],
        "schema_digest": report["schema_digest"],
        "compression": report["compression"],
        "parquet_metadata_digest": report["parquet_metadata_digest"],
    }
