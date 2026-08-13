from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .columnar import validate_columnar_file
from .epoch_transaction import validate_receipt_chain
from .errors import ErrorCode, SctsrError
from .serialization import atomic_write_json, sha256_file, stable_digest

KEY_CHECKPOINT_EPOCHS = frozenset({120, 140, 150, 160, 180, 200})


@dataclass(frozen=True, slots=True)
class ResumeIdentity:
    logical_run_id: str
    parent_sha256: str
    arm_id: str
    training_seed: int
    source_tree_digest: str
    contract_digest: str
    asset_registry_digest: str
    generation_chain_digest: str
    rng_state_digest: str = "NOT_BOUND_SYNTHETIC_UNIT_ONLY"
    receipt_chain_digest: str = "NOT_BOUND_SYNTHETIC_UNIT_ONLY"

    @property
    def digest(self) -> str:
        return stable_digest(asdict(self))

    def validate(self, expected: "ResumeIdentity") -> None:
        for name in self.__dataclass_fields__:
            if getattr(self, name) != getattr(expected, name):
                raise SctsrError(
                    ErrorCode.RESUME_GENERATION_MISMATCH,
                    "Resume identity mismatch",
                    failing_field=name,
                    observed=getattr(self, name),
                    expected=getattr(expected, name),
                )


def _parse_complete(path: Path) -> tuple[int, int] | None:
    name = path.name
    try:
        stem = name.removesuffix(".complete")
        epoch_part, generation_part = stem.split(".generation_")
        return int(epoch_part.removeprefix("epoch_")), int(generation_part)
    except (ValueError, AttributeError):
        return None


def validate_complete_generation(path: str | Path) -> dict[str, Any]:
    complete = Path(path)
    manifest_path = complete / "GENERATION_MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Generation manifest is missing or corrupt", artifact_path=str(manifest_path)) from exc
    claimed_digest = manifest.get("generation_digest")
    digest_payload = dict(manifest)
    digest_payload.pop("generation_digest", None)
    if claimed_digest != stable_digest(digest_payload):
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Generation manifest digest mismatch", artifact_path=str(manifest_path))
    parsed = _parse_complete(complete)
    if parsed is None or parsed != (int(manifest.get("epoch", -1)), int(manifest.get("generation", -1))):
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Generation directory identity differs from manifest")
    listed = {str(row["path"]): row for row in manifest.get("files", [])}
    actual = {
        item.relative_to(complete).as_posix()
        for item in complete.rglob("*")
        if item.is_file() and item.name != "GENERATION_MANIFEST.json"
    }
    if set(listed) != actual or int(manifest.get("file_count", -1)) != len(actual):
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Generation file inventory mismatch", observed={"missing": sorted(set(listed) - actual), "extra": sorted(actual - set(listed))})
    for relative, record in listed.items():
        item = complete / relative
        if item.stat().st_size != int(record["bytes"]) or sha256_file(item) != record["sha256"]:
            raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Generation artifact bytes/SHA mismatch", artifact_path=str(item))
        if item.suffix.lower() == ".json":
            try:
                json.loads(item.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Generation JSON is corrupt", artifact_path=str(item)) from exc
        elif item.suffix.lower() == ".parquet":
            validate_columnar_file(item, expected_sha256=record["sha256"])
    telemetry = [record for relative, record in listed.items() if "/telemetry/" in f"/{relative}" and relative.endswith(".parquet")]
    expected_telemetry = manifest.get("telemetry_partition_sha256")
    if telemetry and (len(telemetry) != 1 or telemetry[0]["sha256"] != expected_telemetry):
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Generation telemetry SHA binding mismatch")
    return {
        "status": "PASS",
        "path": complete.as_posix(),
        "run_id": manifest["run_id"],
        "epoch": int(manifest["epoch"]),
        "generation": int(manifest["generation"]),
        "generation_digest": claimed_digest,
        "generation_manifest_sha256": sha256_file(manifest_path),
        "identity": manifest.get("identity"),
    }


def find_last_complete_epoch(
    transaction_root: str | Path,
    *,
    fail_on_corrupt_latest: bool = True,
) -> dict[str, Any] | None:
    root = Path(transaction_root)
    found = []
    for path in root.glob("epoch_*.generation_*.complete"):
        parsed = _parse_complete(path)
        if parsed:
            found.append((*parsed, path))
    if not found:
        return None
    ordered = sorted(found, key=lambda item: (item[0], item[1]), reverse=True)
    errors = []
    for position, (epoch, generation, path) in enumerate(ordered):
        try:
            report = validate_complete_generation(path)
            return report
        except SctsrError as exc:
            errors.append({"path": path.as_posix(), "code": exc.code.value})
            if position == 0 and fail_on_corrupt_latest:
                raise
    raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "No complete generation passed validation", observed=errors)


def quarantine_inprogress(transaction_root: str | Path, quarantine_root: str | Path, *, reason: str) -> list[str]:
    root = Path(transaction_root)
    qroot = Path(quarantine_root)
    qroot.mkdir(parents=True, exist_ok=True)
    moved = []
    for path in sorted(root.glob("*.inprogress")):
        suffix = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.{uuid.uuid4().hex[:12]}"
        target = qroot / f"{path.name}.quarantined.{suffix}"
        os.replace(path, target)
        moved.append(target.as_posix())
        atomic_write_json(
            target / "QUARANTINE_RECEIPT.json",
            {"reason": reason, "original_path": path.as_posix(), "target_path": target.as_posix(), "status": "QUARANTINED"},
        )
    return moved


def validate_recovery_pointer(path: str | Path) -> dict[str, Any]:
    pointer_path = Path(path)
    try:
        data = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Recovery pointer is missing or corrupt", artifact_path=str(pointer_path)) from exc
    required = {
        "run_id", "epoch", "generation", "complete_path", "generation_digest", "generation_manifest_sha256",
        "receipt_path", "receipt_chain_digest", "identity",
    }
    if not required.issubset(data):
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Recovery pointer is incomplete", observed=sorted(required - set(data)))
    complete = Path(data["complete_path"])
    if not complete.is_absolute():
        complete = (pointer_path.parent / complete).resolve()
    report = validate_complete_generation(complete)
    comparisons = {
        "run_id": report["run_id"],
        "epoch": report["epoch"],
        "generation": report["generation"],
        "generation_digest": report["generation_digest"],
        "generation_manifest_sha256": report["generation_manifest_sha256"],
    }
    for field, observed in comparisons.items():
        if data[field] != observed:
            raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Recovery pointer does not bind the complete generation", failing_field=field, observed=data[field], expected=observed)
    receipt_path = Path(data["receipt_path"])
    if not receipt_path.is_absolute():
        receipt_path = (pointer_path.parent / receipt_path).resolve()
    receipt = validate_receipt_chain(receipt_path)
    if receipt["receipt_chain_digest"] != data["receipt_chain_digest"]:
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Recovery pointer receipt-chain digest mismatch")
    return {**data, "complete_path": complete.as_posix(), "validation_status": "PASS"}


def prepare_resume(
    *,
    pointer_path: str | Path,
    transaction_root: str | Path,
    quarantine_root: str | Path,
    expected_identity: ResumeIdentity,
    minimum_free_bytes: int,
    disk_path: str | Path,
) -> dict[str, Any]:
    quarantined = quarantine_inprogress(transaction_root, quarantine_root, reason="RESUME_PRECHECK_PARTIAL")
    pointer = validate_recovery_pointer(pointer_path)
    identity = pointer.get("identity")
    if not isinstance(identity, dict):
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Recovery pointer has no formal generation identity")
    observed = ResumeIdentity(
        logical_run_id=str(pointer["run_id"]),
        parent_sha256=str(identity["parent_sha256"]),
        arm_id=str(identity["arm_id"]),
        training_seed=int(identity["training_seed"]),
        source_tree_digest=str(identity["source_tree_digest"]),
        contract_digest=str(identity["contract_digest"]),
        asset_registry_digest=str(identity["asset_registry_digest"]),
        generation_chain_digest=str(pointer["generation_digest"]),
        rng_state_digest=str(identity["rng_state_digest"]),
        receipt_chain_digest=str(pointer["receipt_chain_digest"]),
    )
    observed.validate(expected_identity)
    free = shutil.disk_usage(Path(disk_path)).free
    if free < minimum_free_bytes:
        raise SctsrError(ErrorCode.DISK_SPACE_PRECHECK_FAILED, "Insufficient disk space for safe epoch resume", observed=free, expected=minimum_free_bytes)
    last = find_last_complete_epoch(transaction_root, fail_on_corrupt_latest=True)
    if last is None or last["epoch"] != int(pointer["epoch"]) or last["generation"] != int(pointer["generation"]):
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Recovery pointer is not the last complete generation")
    return {"status": "PASS", "resume_epoch": last["epoch"] + 1, "quarantined_partial_paths": quarantined, "identity_digest": observed.digest}


def retained_checkpoint_epochs(complete_epochs: Iterable[int]) -> set[int]:
    epochs = sorted({int(epoch) for epoch in complete_epochs})
    if not epochs:
        return set()
    return (set(epochs) & set(KEY_CHECKPOINT_EPOCHS)) | set(epochs[-2:])
