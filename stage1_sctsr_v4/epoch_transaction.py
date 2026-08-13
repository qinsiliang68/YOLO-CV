from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .columnar import validate_columnar_file
from .errors import ErrorCode, SctsrError
from .serialization import _fsync_directory, atomic_write_bytes, atomic_write_json, canonical_json_bytes, sha256_file, stable_digest


@dataclass(frozen=True, slots=True)
class GenerationIdentity:
    parent_sha256: str
    arm_id: str
    training_seed: int
    source_tree_digest: str
    contract_digest: str
    asset_registry_digest: str
    rng_state_digest: str
    previous_generation_digest: str

    def validate(self) -> None:
        for name in (
            "parent_sha256", "source_tree_digest", "contract_digest", "asset_registry_digest",
            "rng_state_digest", "previous_generation_digest",
        ):
            value = str(getattr(self, name))
            if len(value) != 64:
                raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Generation identity digest is not SHA-256", failing_field=name, observed=value)
        if not self.arm_id or self.training_seed < 0:
            raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Generation arm/seed identity is invalid")


def _read_receipt_chain(path: Path) -> tuple[list[dict[str, Any]], str]:
    if not path.exists():
        return [], "0" * 64
    rows: list[dict[str, Any]] = []
    previous = "0" * 64
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.endswith("\n"):
                    raise ValueError("receipt line lacks newline terminator")
                row = json.loads(line)
                claimed = str(row.pop("receipt_digest"))
                if row.get("previous_receipt_digest") != previous or stable_digest(row) != claimed:
                    raise ValueError(f"receipt chain mismatch at line {line_number}")
                row["receipt_digest"] = claimed
                rows.append(row)
                previous = claimed
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Epoch receipt chain is corrupt", artifact_path=str(path)) from exc
    return rows, previous


def validate_receipt_chain(path: str | Path) -> dict[str, Any]:
    rows, digest = _read_receipt_chain(Path(path))
    return {"status": "PASS", "row_count": len(rows), "receipt_chain_digest": digest, "sha256": sha256_file(path) if Path(path).exists() else None}


@dataclass
class EpochTransaction:
    root: Path
    run_id: str
    epoch: int
    generation: int
    quarantine_root: Path | None = None
    identity: GenerationIdentity | None = None
    required_relative_paths: Sequence[str] = ()
    inprogress: Path = field(init=False)
    complete: Path = field(init=False)
    _committed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.quarantine_root = Path(self.quarantine_root or self.root.parent / "09_quarantine")
        if not self.run_id or not 1 <= int(self.epoch) <= 200 or int(self.generation) < 1:
            raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Epoch transaction identity is invalid")
        if self.identity is not None:
            self.identity.validate()
        stem = f"epoch_{self.epoch:04d}.generation_{self.generation}"
        self.inprogress = self.root / (stem + ".inprogress")
        self.complete = self.root / (stem + ".complete")

    @property
    def receipt_path(self) -> Path:
        return self.root.parent / "08_receipts" / "epoch_receipts.jsonl"

    @property
    def artifact_index_path(self) -> Path:
        return self.root.parent / "ARTIFACT_INDEX.json"

    @property
    def recovery_pointer_path(self) -> Path:
        return self.root.parent / "ROLLING_RECOVERY_POINTER.json"

    def begin(self) -> "EpochTransaction":
        if self.inprogress.exists() or self.complete.exists():
            raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Epoch generation path already exists", artifact_path=str(self.inprogress))
        self.inprogress.mkdir(parents=True)
        atomic_write_json(
            self.inprogress / "TRANSACTION_IDENTITY.json",
            {
                "schema_version": "stage1.sctsr.epoch_transaction_identity.v1",
                "status": "INPROGRESS",
                "run_id": self.run_id,
                "epoch": self.epoch,
                "generation": self.generation,
                "identity": asdict(self.identity) if self.identity is not None else None,
                "identity_reason": "PRESENT" if self.identity is not None else "UNBOUND_SYNTHETIC_UNIT_ONLY",
            },
        )
        return self

    def path_for(self, relative_path: str) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Transaction path must be relative and contained")
        path = self.inprogress / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, relative_path: str, value: Any) -> Path:
        path = self.path_for(relative_path)
        atomic_write_json(path, value)
        return path

    def write_bytes(self, relative_path: str, value: bytes) -> Path:
        return atomic_write_bytes(self.path_for(relative_path), value)

    def _validate_files(self) -> list[dict[str, Any]]:
        actual_relative = {path.relative_to(self.inprogress).as_posix() for path in self.inprogress.rglob("*") if path.is_file()}
        missing = set(self.required_relative_paths) - actual_relative
        if missing:
            raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Epoch transaction is missing required artifacts", observed=sorted(missing))
        records: list[dict[str, Any]] = []
        for path in sorted(self.inprogress.rglob("*")):
            if not path.is_file() or path.name == "GENERATION_MANIFEST.json":
                continue
            relative = path.relative_to(self.inprogress).as_posix()
            if path.name.endswith((".inprogress", ".tmp")):
                raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Epoch transaction contains an unpublished temporary file", artifact_path=str(path))
            if path.suffix.lower() == ".json":
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Epoch transaction contains half-written JSON", artifact_path=str(path)) from exc
            elif path.suffix.lower() == ".parquet":
                try:
                    report = validate_columnar_file(path)
                except SctsrError as exc:
                    raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Epoch transaction contains half-written or invalid Parquet", artifact_path=str(path)) from exc
                if report["compression"] != "ZSTD":
                    raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Epoch Parquet is not Zstd", artifact_path=str(path))
            records.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        if not records:
            raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Epoch transaction contains no evidence files")
        return records

    def _append_receipt(self, manifest: Mapping[str, Any], manifest_sha: str) -> str:
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        _, previous = _read_receipt_chain(self.receipt_path)
        row = {
            "schema_version": "stage1.sctsr.epoch_receipt.v1",
            "status": "COMPLETE",
            "path": self.complete.as_posix(),
            "run_id": self.run_id,
            "epoch": self.epoch,
            "generation": self.generation,
            "generation_digest": manifest["generation_digest"],
            "generation_manifest_sha256": manifest_sha,
            "previous_receipt_digest": previous,
        }
        row["receipt_digest"] = stable_digest(row)
        encoded = canonical_json_bytes(row)
        with self.receipt_path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(self.receipt_path.parent)
        return str(row["receipt_digest"])

    def _update_artifact_index(self, manifest: Mapping[str, Any], manifest_sha: str, receipt_digest: str) -> None:
        if self.artifact_index_path.exists():
            try:
                index = json.loads(self.artifact_index_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Logical artifact index is corrupt", artifact_path=str(self.artifact_index_path)) from exc
            entries = list(index.get("epoch_generations", []))
        else:
            entries = []
        key = (self.run_id, self.epoch, self.generation)
        if any((row["run_id"], int(row["epoch"]), int(row["generation"])) == key for row in entries):
            raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Logical artifact index would overwrite an epoch generation")
        entries.append(
            {
                "run_id": self.run_id,
                "epoch": self.epoch,
                "generation": self.generation,
                "complete_path": self.complete.as_posix(),
                "generation_digest": manifest["generation_digest"],
                "generation_manifest_sha256": manifest_sha,
                "receipt_digest": receipt_digest,
            }
        )
        entries.sort(key=lambda row: (row["run_id"], int(row["epoch"]), int(row["generation"])))
        atomic_write_json(
            self.artifact_index_path,
            {
                "schema_version": "stage1.sctsr.epoch_artifact_index.v1",
                "epoch_generations": entries,
                "epoch_generation_index_digest": stable_digest(entries),
            },
        )

    def commit(self, validator: Callable[[Path], None] | None = None) -> dict[str, Any]:
        if not self.inprogress.exists():
            raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "No inprogress transaction to commit")
        try:
            if validator:
                validator(self.inprogress)
            files = self._validate_files()
            telemetry_files = [row for row in files if "/telemetry/" in f"/{row['path']}" and row["path"].endswith(".parquet")]
            manifest: dict[str, Any] = {
                "schema_version": "stage1.sctsr.epoch_generation.v2",
                "status": "VALIDATED_READY_TO_PUBLISH",
                "run_id": self.run_id,
                "epoch": self.epoch,
                "generation": self.generation,
                "identity": asdict(self.identity) if self.identity is not None else None,
                "identity_reason": "PRESENT" if self.identity is not None else "UNBOUND_SYNTHETIC_UNIT_ONLY",
                "files": files,
                "file_count": len(files),
                "telemetry_partition_sha256": telemetry_files[0]["sha256"] if len(telemetry_files) == 1 else None,
                "telemetry_partition_reason": "PRESENT" if len(telemetry_files) == 1 else "NOT_PRESENT_SYNTHETIC_UNIT_ONLY" if not telemetry_files else "MULTIPLE_TELEMETRY_PARTITIONS_FORBIDDEN",
            }
            if len(telemetry_files) > 1:
                raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Epoch generation contains multiple telemetry partitions")
            manifest["generation_digest"] = stable_digest(manifest)
            manifest_path = self.inprogress / "GENERATION_MANIFEST.json"
            atomic_write_json(manifest_path, manifest)
            manifest_sha = sha256_file(manifest_path)
            os.replace(self.inprogress, self.complete)
            _fsync_directory(self.complete.parent)
            self._committed = True
            receipt_digest = self._append_receipt(manifest, manifest_sha)
            self._update_artifact_index(manifest, manifest_sha, receipt_digest)
            pointer = {
                "schema_version": "stage1.sctsr.rolling_recovery_pointer.v2",
                "run_id": self.run_id,
                "epoch": self.epoch,
                "generation": self.generation,
                "complete_path": self.complete.as_posix(),
                "generation_digest": manifest["generation_digest"],
                "generation_manifest_sha256": manifest_sha,
                "receipt_path": self.receipt_path.as_posix(),
                "receipt_chain_digest": receipt_digest,
                "identity": asdict(self.identity) if self.identity is not None else None,
            }
            atomic_write_json(self.recovery_pointer_path, pointer)
            return {
                **manifest,
                "generation_manifest_sha256": manifest_sha,
                "receipt_chain_digest": receipt_digest,
            }
        except BaseException:
            if self.inprogress.exists():
                self.abort("COMMIT_VALIDATION_OR_PUBLICATION_FAILED")
            raise

    def abort(self, reason: str) -> Path:
        if not self.inprogress.exists():
            return self.quarantine_root / "NOTHING_TO_QUARANTINE"
        self.quarantine_root.mkdir(parents=True, exist_ok=True)
        suffix = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.{uuid.uuid4().hex[:12]}"
        target = self.quarantine_root / f"{self.inprogress.name}.quarantined.{suffix}"
        os.replace(self.inprogress, target)
        atomic_write_json(
            target / "QUARANTINE_RECEIPT.json",
            {
                "schema_version": "stage1.sctsr.quarantine_receipt.v1",
                "status": "QUARANTINED",
                "reason": reason,
                "source": self.inprogress.as_posix(),
                "target": target.as_posix(),
                "run_id": self.run_id,
                "epoch": self.epoch,
                "generation": self.generation,
            },
        )
        _fsync_directory(target.parent)
        return target

    def __enter__(self) -> "EpochTransaction":
        return self.begin()

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None and not self._committed:
            self.abort(f"{exc_type.__name__}: {exc}")
        return False
