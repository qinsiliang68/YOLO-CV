from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .checkpointing import load_checkpoint
from .columnar import read_columnar, validate_columnar_file
from .epoch_transaction import reconcile_epoch_publications, validate_receipt_chain
from .errors import ErrorCode, SctsrError
from .evidence_runtime import ReplayHistoryState
from .filesystem import windows_safe_resolved_path
from .occurrence_ledger import validate_occurrence_rows
from .serialization import atomic_write_json, sha256_file, stable_digest

KEY_CHECKPOINT_EPOCHS = frozenset({120, 140, 150, 160, 180, 200})


@dataclass(slots=True)
class FormalResumeContext:
    """Fully revalidated state needed to continue one formal run.

    The context deliberately stores paths and digests rather than a loaded
    model payload.  The runner reloads and validates the checkpoint immediately
    before restoring the prepared trainer, so a file changed between preflight
    and execution is rejected.
    """

    run_root: str
    run_id: str
    arm_id: str
    training_seed: int
    epoch_start: int
    epoch_end: int
    last_complete_epoch: int
    resume_epoch: int
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_rng_digest: str
    global_step: int
    required_free_bytes: int
    previous_generation_digest: str
    receipt_chain_digest: str
    history: ReplayHistoryState
    quarantined_partial_paths: tuple[str, ...]
    terminal_epoch_complete: bool

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "run_root": self.run_root,
            "run_id": self.run_id,
            "arm_id": self.arm_id,
            "training_seed": self.training_seed,
            "epoch_start": self.epoch_start,
            "epoch_end": self.epoch_end,
            "last_complete_epoch": self.last_complete_epoch,
            "resume_epoch": self.resume_epoch,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_rng_digest": self.checkpoint_rng_digest,
            "global_step": self.global_step,
            "required_free_bytes": self.required_free_bytes,
            "previous_generation_digest": self.previous_generation_digest,
            "receipt_chain_digest": self.receipt_chain_digest,
            "history": self.history.snapshot(),
            "quarantined_partial_paths": list(self.quarantined_partial_paths),
            "terminal_epoch_complete": self.terminal_epoch_complete,
        }
        return {**payload, "resume_context_digest": stable_digest(payload)}


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
    # Callers retain ordinary registered paths in manifests.  Normalize at the
    # filesystem boundary so Win32 can enumerate, move, and receipt transaction
    # directories whose resolved path exceeds MAX_PATH.
    root = windows_safe_resolved_path(transaction_root)
    qroot = windows_safe_resolved_path(quarantine_root)
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


def _canonical_windows_path(path: str | Path) -> Path:
    raw = str(path)
    if os.name == "nt":
        if raw.upper().startswith("//?/UNC/"):
            raw = "//" + raw[8:]
        elif raw.startswith("//?/"):
            raw = raw[4:]
        elif raw.startswith("\\\\?\\UNC\\"):
            raw = "\\\\" + raw[8:]
        elif raw.startswith("\\\\?\\"):
            raw = raw[4:]
    return Path(raw).resolve()


def _require_contained(path: Path, root: Path, *, role: str) -> Path:
    resolved = path.resolve()
    try:
        _canonical_windows_path(resolved).relative_to(_canonical_windows_path(root))
    except ValueError as exc:
        raise SctsrError(
            ErrorCode.RESUME_GENERATION_MISMATCH,
            f"{role} escapes the registered run root",
            artifact_path=str(resolved),
        ) from exc
    return resolved


def _load_receipt_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                rows.append(json.loads(line))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SctsrError(
            ErrorCode.RESUME_GENERATION_MISMATCH,
            "Epoch receipt chain cannot be decoded for resume",
            artifact_path=str(path),
        ) from exc
    return rows


def _replay_history_after_rows(
    history: ReplayHistoryState,
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    epoch: int,
) -> None:
    validate_occurrence_rows(rows)
    for index, row in enumerate(rows):
        if row["run_id"] != run_id or int(row["epoch"]) != epoch:
            raise SctsrError(
                ErrorCode.RESUME_GENERATION_MISMATCH,
                "Occurrence partition identity differs from the resumed run",
                failing_field=f"occurrence[{index}]",
            )
        cumulative_before = int(row["cumulative_replay_count_before"])
        cumulative_after = int(row["cumulative_replay_count_after"])
        if cumulative_before != history.cumulative_occurrences:
            raise SctsrError(
                ErrorCode.RESUME_GENERATION_MISMATCH,
                "Occurrence cumulative replay history is discontinuous",
                failing_field=f"occurrence[{index}].cumulative_replay_count_before",
                observed=cumulative_before,
                expected=history.cumulative_occurrences,
            )
        if row["occurrence_role"] == "BASE":
            if cumulative_after != history.cumulative_occurrences:
                raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Base occurrence advances replay history")
            continue
        sample_id = str(row["sample_id"])
        count_before = int(history.counts.get(sample_id, 0))
        previous_epoch = history.last_epoch.get(sample_id)
        if int(row["replay_count_before"]) != count_before or int(row["replay_count_after"]) != count_before + 1:
            raise SctsrError(
                ErrorCode.RESUME_GENERATION_MISMATCH,
                "Per-sample replay history is discontinuous",
                failing_field=f"occurrence[{index}].replay_count",
            )
        if row["last_replay_epoch"] != previous_epoch:
            raise SctsrError(
                ErrorCode.RESUME_GENERATION_MISMATCH,
                "Occurrence last replay epoch differs from reconstructed history",
                failing_field=f"occurrence[{index}].last_replay_epoch",
                observed=row["last_replay_epoch"],
                expected=previous_epoch,
            )
        history.counts[sample_id] = count_before + 1
        history.last_epoch[sample_id] = epoch
        history.cumulative_occurrences += 1
        if cumulative_after != history.cumulative_occurrences:
            raise SctsrError(
                ErrorCode.RESUME_GENERATION_MISMATCH,
                "Replay occurrence cumulative-after value is inconsistent",
                failing_field=f"occurrence[{index}].cumulative_replay_count_after",
            )


def prepare_formal_resume_context(
    *,
    run_root: str | Path,
    expected_run_id: str,
    expected_arm_id: str,
    expected_training_seed: int,
    expected_source_tree_digest: str,
    expected_contract_digest: str,
    expected_asset_registry_digest: str,
    expected_previous_checkpoint_sha256: str,
    expected_previous_generation_digest: str,
    epoch_start: int,
    epoch_end: int,
    minimum_free_bytes: int,
    allow_terminal_epoch_for_finalization: bool = False,
    validated_preview: FormalResumeContext | None = None,
    _mutate: bool = True,
) -> FormalResumeContext:
    """Re-audit an interrupted formal run and return its exact continuation.

    Every completed generation is validated in order, including its immutable
    file inventory, checkpoint bytes and full payload, generation ancestry,
    RNG evidence, receipt chain, and reconstructed replay history.  Partials
    are moved only after the immutable identity checks pass, preventing a
    caller with the wrong run identity from quarantining somebody else's work.
    """

    canonical_root = Path(run_root).resolve()
    root = windows_safe_resolved_path(canonical_root)
    transaction_root = root / "03_epoch_transactions"
    quarantine_root = root / "09_quarantine"
    if not root.is_dir() or not transaction_root.is_dir():
        raise SctsrError(
            ErrorCode.RESUME_GENERATION_MISMATCH,
            "Formal resume root or transaction directory is missing",
            artifact_path=str(root),
        )
    if not (1 <= int(epoch_start) <= int(epoch_end) <= 200):
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Formal resume epoch range is invalid")

    if validated_preview is not None:
        if not _mutate:
            raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "A validated resume preview may only be consumed by the fenced mutating phase")
        preview_identity = {
            "run_root": (_canonical_windows_path(validated_preview.run_root), _canonical_windows_path(canonical_root)),
            "run_id": (validated_preview.run_id, expected_run_id),
            "arm_id": (validated_preview.arm_id, expected_arm_id),
            "training_seed": (validated_preview.training_seed, expected_training_seed),
            "epoch_start": (validated_preview.epoch_start, epoch_start),
            "epoch_end": (validated_preview.epoch_end, epoch_end),
        }
        mismatch = {
            field: {"preview": values[0], "requested": values[1]}
            for field, values in preview_identity.items()
            if values[0] != values[1]
        }
        if mismatch:
            raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Fenced resume request differs from its validated preview", observed=mismatch)
        quarantined = tuple(quarantine_inprogress(transaction_root, quarantine_root, reason="FORMAL_RESUME_PRECHECK_PARTIAL"))
        pointer = validate_recovery_pointer(root / "ROLLING_RECOVERY_POINTER.json")
        expected_complete = _canonical_windows_path(Path(validated_preview.checkpoint_path).parents[1])
        if any(
            (
                pointer.get("run_id") != validated_preview.run_id,
                int(pointer.get("epoch", -1)) != validated_preview.last_complete_epoch,
                int(pointer.get("generation", -1)) != 1,
                pointer.get("generation_digest") != validated_preview.previous_generation_digest,
                pointer.get("receipt_chain_digest") != validated_preview.receipt_chain_digest,
                _canonical_windows_path(pointer.get("complete_path", "")) != expected_complete,
            )
        ):
            raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Canonical resume state changed after its read-only preview")
        if shutil.disk_usage(root).free < validated_preview.required_free_bytes:
            raise SctsrError(
                ErrorCode.DISK_SPACE_PRECHECK_FAILED,
                "Insufficient disk space for safe formal resume",
                observed=shutil.disk_usage(root).free,
                expected=validated_preview.required_free_bytes,
            )
        return replace(validated_preview, quarantined_partial_paths=quarantined)

    if _mutate:
        reconcile_epoch_publications(
            transaction_root,
            quarantine_root,
            expected_run_id=expected_run_id,
            expected_identity={
                "arm_id": expected_arm_id,
                "training_seed": expected_training_seed,
                "source_tree_digest": expected_source_tree_digest,
                "contract_digest": expected_contract_digest,
                "asset_registry_digest": expected_asset_registry_digest,
            },
        )
    pointer_path = root / "ROLLING_RECOVERY_POINTER.json"
    pointer = validate_recovery_pointer(pointer_path)
    pointer_complete = _require_contained(Path(pointer["complete_path"]), transaction_root, role="Recovery pointer")
    receipt_path = Path(pointer["receipt_path"])
    if not receipt_path.is_absolute():
        receipt_path = (pointer_path.parent / receipt_path).resolve()
    receipt_path = _require_contained(receipt_path, root, role="Receipt chain")

    discovered: list[tuple[int, int, Path]] = []
    for complete in transaction_root.glob("epoch_*.generation_*.complete"):
        parsed = _parse_complete(complete)
        if parsed is not None:
            discovered.append((parsed[0], parsed[1], complete.resolve()))
    discovered.sort(key=lambda item: (item[0], item[1]))
    if not discovered:
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Formal resume found no completed epoch")
    epochs = [epoch for epoch, generation, _ in discovered if generation == 1]
    if len(epochs) != len(discovered) or epochs != list(range(epoch_start, epochs[-1] + 1)):
        raise SctsrError(
            ErrorCode.RESUME_GENERATION_MISMATCH,
            "Completed resume generations are not one contiguous generation-1 prefix",
            observed=[(epoch, generation) for epoch, generation, _ in discovered],
            expected=f"epochs {epoch_start}..last, generation 1",
        )
    last_epoch = epochs[-1]
    if last_epoch > epoch_end or (last_epoch == epoch_end and not allow_terminal_epoch_for_finalization):
        raise SctsrError(
            ErrorCode.RESUME_GENERATION_MISMATCH,
            "Run already contains its terminal completed epoch and may not be resumed",
            observed=last_epoch,
            expected=f"less than {epoch_end}",
        )

    previous_checkpoint = expected_previous_checkpoint_sha256
    previous_generation = expected_previous_generation_digest
    history = ReplayHistoryState()
    validated: list[dict[str, Any]] = []
    last_checkpoint_path: Path | None = None
    last_checkpoint_sha: str | None = None
    last_checkpoint_rng: str | None = None
    last_global_step: int | None = None
    largest_generation_bytes = 0

    for epoch, generation, complete in discovered:
        report = validate_complete_generation(complete)
        if report["run_id"] != expected_run_id or report["epoch"] != epoch or report["generation"] != generation:
            raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Completed generation identity differs from the requested run")
        manifest_path = complete / "GENERATION_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        largest_generation_bytes = max(
            largest_generation_bytes,
            manifest_path.stat().st_size + sum(int(row.get("bytes", 0)) for row in manifest.get("files", [])),
        )
        identity = manifest.get("identity")
        if not isinstance(identity, dict):
            raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Formal generation has no bound identity")
        expected_identity = {
            "parent_sha256": previous_checkpoint,
            "arm_id": expected_arm_id,
            "training_seed": expected_training_seed,
            "source_tree_digest": expected_source_tree_digest,
            "contract_digest": expected_contract_digest,
            "asset_registry_digest": expected_asset_registry_digest,
            "previous_generation_digest": previous_generation,
        }
        mismatch = {
            field: {"observed": identity.get(field), "expected": value}
            for field, value in expected_identity.items()
            if identity.get(field) != value
        }
        if mismatch:
            raise SctsrError(
                ErrorCode.RESUME_GENERATION_MISMATCH,
                "Formal generation ancestry or static identity mismatch",
                observed=mismatch,
            )
        checkpoint_rows = [row for row in manifest.get("files", []) if str(row.get("path", "")).endswith(".pt")]
        occurrence_rows = [row for row in manifest.get("files", []) if "/occurrence/" in f"/{row.get('path', '')}" and str(row.get("path", "")).endswith(".parquet")]
        if len(checkpoint_rows) != 1 or len(occurrence_rows) != 1:
            raise SctsrError(
                ErrorCode.RESUME_GENERATION_MISMATCH,
                "Resume generation must contain exactly one checkpoint and occurrence partition",
                observed={"checkpoint": len(checkpoint_rows), "occurrence": len(occurrence_rows)},
            )
        checkpoint_path = _require_contained(complete / checkpoint_rows[0]["path"], complete, role="Epoch checkpoint")
        checkpoint_sha = str(checkpoint_rows[0]["sha256"])
        payload = load_checkpoint(checkpoint_path, expected_sha256=checkpoint_sha, expected_epoch=epoch)
        payload_expected = {
            "training_seed": expected_training_seed,
            "source_tree_digest": expected_source_tree_digest,
            "asset_registry_digest": expected_asset_registry_digest,
            "global_step": epoch * 938,
            "base_sampler_generation": epoch,
        }
        payload_mismatch = {
            field: {"observed": payload.get(field), "expected": value}
            for field, value in payload_expected.items()
            if payload.get(field) != value
        }
        if payload_mismatch:
            raise SctsrError(
                ErrorCode.RESUME_GENERATION_MISMATCH,
                "Resume checkpoint payload differs from the formal trajectory",
                observed=payload_mismatch,
            )
        checkpoint_rng = payload["rng_state"].digest()
        summary_path = complete / "EPOCH_EVIDENCE_SUMMARY.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Epoch evidence summary is missing or corrupt") from exc
        rng_evidence = summary.get("rng_evidence")
        if not isinstance(rng_evidence, dict) or any(
            (
                rng_evidence.get("recorder_epoch_start_digest") != identity.get("rng_state_digest"),
                rng_evidence.get("runtime_epoch_start_digest") != identity.get("rng_state_digest"),
                rng_evidence.get("runtime_epoch_end_digest") != checkpoint_rng,
                rng_evidence.get("finalize_entry_digest") != checkpoint_rng,
                summary.get("checkpoint_sha256") != checkpoint_sha,
            )
        ):
            raise SctsrError(
                ErrorCode.RESUME_GENERATION_MISMATCH,
                "Epoch RNG/checkpoint evidence is not a closed continuation boundary",
                observed=rng_evidence,
            )
        occurrence_path = _require_contained(complete / occurrence_rows[0]["path"], complete, role="Occurrence partition")
        rows = read_columnar(occurrence_path)
        _replay_history_after_rows(history, rows, run_id=expected_run_id, epoch=epoch)
        if summary.get("history_after_epoch") != history.snapshot():
            raise SctsrError(
                ErrorCode.RESUME_GENERATION_MISMATCH,
                "Epoch evidence summary replay history differs from occurrence bytes",
                observed=summary.get("history_after_epoch"),
                expected=history.snapshot(),
            )
        validated.append(report)
        previous_checkpoint = checkpoint_sha
        previous_generation = str(report["generation_digest"])
        last_checkpoint_path = checkpoint_path
        last_checkpoint_sha = checkpoint_sha
        last_checkpoint_rng = checkpoint_rng
        last_global_step = int(payload["global_step"])

    assert last_checkpoint_path is not None
    assert last_checkpoint_sha is not None
    assert last_checkpoint_rng is not None
    assert last_global_step is not None
    if _canonical_windows_path(pointer_complete) != _canonical_windows_path(discovered[-1][2]) or int(pointer["epoch"]) != last_epoch:
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Recovery pointer is not the last contiguous complete generation")
    chain = validate_receipt_chain(receipt_path)
    receipts = _load_receipt_rows(receipt_path)
    if chain["row_count"] != len(validated) or len(receipts) != len(validated):
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Receipt-chain length differs from completed generations")
    for receipt, report in zip(receipts, validated, strict=True):
        if any(
            (
                receipt.get("run_id") != expected_run_id,
                int(receipt.get("epoch", -1)) != report["epoch"],
                int(receipt.get("generation", -1)) != report["generation"],
                receipt.get("generation_digest") != report["generation_digest"],
                receipt.get("generation_manifest_sha256") != report["generation_manifest_sha256"],
            )
        ):
            raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Receipt row differs from its completed generation")
    if chain["receipt_chain_digest"] != pointer["receipt_chain_digest"]:
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Pointer does not bind the complete receipt chain")
    try:
        generation_index_path = (
            root / "ARTIFACT_INDEX_GENERATIONS.json"
            if (root / "ARTIFACT_INDEX_GENERATIONS.json").is_file()
            else root / "ARTIFACT_INDEX.json"
        )
        generation_index = json.loads(generation_index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Mutable generation index is missing or corrupt") from exc
    index_rows = generation_index.get("epoch_generations")
    if (
        generation_index.get("schema_version") != "stage1.sctsr.epoch_artifact_index.v1"
        or not isinstance(index_rows, list)
        or generation_index.get("epoch_generation_index_digest") != stable_digest(index_rows)
        or len(index_rows) != len(validated)
    ):
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Generation index does not bind the complete prefix")
    for index_row, report, (_, _, complete) in zip(index_rows, validated, discovered, strict=True):
        if any(
            (
                index_row.get("run_id") != expected_run_id,
                int(index_row.get("epoch", -1)) != report["epoch"],
                int(index_row.get("generation", -1)) != report["generation"],
                _canonical_windows_path(index_row.get("complete_path", "")) != _canonical_windows_path(complete),
                index_row.get("generation_digest") != report["generation_digest"],
                index_row.get("generation_manifest_sha256") != report["generation_manifest_sha256"],
                index_row.get("receipt_digest") != receipts[report["epoch"] - epoch_start].get("receipt_digest"),
            )
        ):
            raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Generation index row differs from immutable epoch evidence")
    remaining_epochs = epoch_end - last_epoch
    estimated_remaining_bytes = math.ceil(largest_generation_bytes * remaining_epochs * 1.25) + 2 * last_checkpoint_path.stat().st_size
    required_free_bytes = max(int(minimum_free_bytes), estimated_remaining_bytes)
    observed_free_bytes = shutil.disk_usage(root).free
    if observed_free_bytes < required_free_bytes:
        raise SctsrError(
            ErrorCode.DISK_SPACE_PRECHECK_FAILED,
            "Insufficient disk space for safe formal resume",
            observed=observed_free_bytes,
            expected=required_free_bytes,
        )

    quarantined = (
        tuple(quarantine_inprogress(transaction_root, quarantine_root, reason="FORMAL_RESUME_PRECHECK_PARTIAL"))
        if _mutate
        else ()
    )
    # Recheck the canonical pointer after the full read-only audit and, for the
    # mutating phase, after moving only uncommitted siblings.
    pointer_after = validate_recovery_pointer(pointer_path)
    if pointer_after["generation_digest"] != pointer["generation_digest"] or pointer_after["receipt_chain_digest"] != pointer["receipt_chain_digest"]:
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Canonical recovery pointer changed during resume preflight")
    return FormalResumeContext(
        run_root=canonical_root.as_posix(),
        run_id=expected_run_id,
        arm_id=expected_arm_id,
        training_seed=expected_training_seed,
        epoch_start=epoch_start,
        epoch_end=epoch_end,
        last_complete_epoch=last_epoch,
        resume_epoch=last_epoch + 1,
        checkpoint_path=_canonical_windows_path(last_checkpoint_path).as_posix(),
        checkpoint_sha256=last_checkpoint_sha,
        checkpoint_rng_digest=last_checkpoint_rng,
        global_step=last_global_step,
        required_free_bytes=required_free_bytes,
        previous_generation_digest=previous_generation,
        receipt_chain_digest=str(chain["receipt_chain_digest"]),
        history=history,
        quarantined_partial_paths=quarantined,
        terminal_epoch_complete=last_epoch == epoch_end,
    )


def _inspect_formal_resume_context_from_frozen_prefix(
    *,
    run_root: str | Path,
    expected_run_id: str,
    expected_arm_id: str,
    expected_training_seed: int,
    expected_source_tree_digest: str,
    expected_contract_digest: str,
    expected_asset_registry_digest: str,
    expected_previous_checkpoint_sha256: str,
    expected_previous_generation_digest: str,
    epoch_start: int,
    epoch_end: int,
    minimum_free_bytes: int,
    allow_terminal_epoch_for_finalization: bool = False,
) -> FormalResumeContext:
    """Validate the frozen manifest chain and fully audit only its last checkpoint."""

    canonical_root = Path(run_root).resolve()
    root = windows_safe_resolved_path(canonical_root)
    transaction_root = root / "03_epoch_transactions"
    if not root.is_dir() or not transaction_root.is_dir() or not (1 <= int(epoch_start) <= int(epoch_end) <= 200):
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Formal resume root or epoch range is invalid")
    discovered = sorted(
        (parsed[0], parsed[1], complete.resolve())
        for complete in transaction_root.glob("epoch_*.generation_*.complete")
        if (parsed := _parse_complete(complete)) is not None
    )
    if not discovered:
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Formal resume found no completed epoch")
    epochs = [epoch for epoch, generation, _complete in discovered if generation == 1]
    if len(epochs) != len(discovered) or epochs != list(range(epoch_start, epochs[-1] + 1)):
        raise SctsrError(
            ErrorCode.RESUME_GENERATION_MISMATCH,
            "Completed resume generations are not one contiguous generation-1 prefix",
            observed=[(epoch, generation) for epoch, generation, _complete in discovered],
        )
    last_epoch = epochs[-1]
    if last_epoch > epoch_end or (last_epoch == epoch_end and not allow_terminal_epoch_for_finalization):
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Run already contains its terminal completed epoch and may not be resumed")

    pointer = validate_recovery_pointer(root / "ROLLING_RECOVERY_POINTER.json")
    if (
        pointer.get("run_id") != expected_run_id
        or int(pointer.get("epoch", -1)) != last_epoch
        or int(pointer.get("generation", -1)) != 1
        or _canonical_windows_path(pointer.get("complete_path", "")) != _canonical_windows_path(discovered[-1][2])
    ):
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Recovery pointer is not the last contiguous complete generation")
    receipt_path = Path(pointer["receipt_path"])
    if not receipt_path.is_absolute():
        receipt_path = (root / receipt_path).resolve()
    receipt_path = _require_contained(receipt_path, root, role="Receipt chain")
    receipts = _load_receipt_rows(receipt_path)
    generation_index_path = root / "ARTIFACT_INDEX_GENERATIONS.json"
    if not generation_index_path.is_file():
        generation_index_path = root / "ARTIFACT_INDEX.json"
    generation_index = json.loads(generation_index_path.read_text(encoding="utf-8"))
    index_rows = generation_index.get("epoch_generations")
    if (
        generation_index.get("schema_version") != "stage1.sctsr.epoch_artifact_index.v1"
        or not isinstance(index_rows, list)
        or generation_index.get("epoch_generation_index_digest") != stable_digest(index_rows)
        or len(index_rows) != len(discovered)
        or len(receipts) != len(discovered)
    ):
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Frozen generation index or receipt-chain length is invalid")

    previous_checkpoint = expected_previous_checkpoint_sha256
    previous_generation = expected_previous_generation_digest
    largest_generation_bytes = 0
    last_manifest: dict[str, Any] | None = None
    for (epoch, generation, complete), index_row, receipt in zip(discovered, index_rows, receipts, strict=True):
        manifest_path = complete / "GENERATION_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest_payload = dict(manifest)
        claimed_digest = digest_payload.pop("generation_digest", None)
        if (
            claimed_digest != stable_digest(digest_payload)
            or manifest.get("run_id") != expected_run_id
            or int(manifest.get("epoch", -1)) != epoch
            or int(manifest.get("generation", -1)) != generation
            or sha256_file(manifest_path) != index_row.get("generation_manifest_sha256")
            or index_row.get("run_id") != expected_run_id
            or int(index_row.get("epoch", -1)) != epoch
            or int(index_row.get("generation", -1)) != generation
            or _canonical_windows_path(index_row.get("complete_path", "")) != _canonical_windows_path(complete)
            or index_row.get("generation_digest") != claimed_digest
            or receipt.get("run_id") != expected_run_id
            or int(receipt.get("epoch", -1)) != epoch
            or int(receipt.get("generation", -1)) != generation
            or receipt.get("generation_digest") != claimed_digest
            or receipt.get("generation_manifest_sha256") != index_row.get("generation_manifest_sha256")
            or receipt.get("receipt_digest") != index_row.get("receipt_digest")
        ):
            raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Frozen generation manifest/index/receipt binding is inconsistent", observed=epoch)
        identity = manifest.get("identity")
        expected_identity = {
            "parent_sha256": previous_checkpoint,
            "arm_id": expected_arm_id,
            "training_seed": expected_training_seed,
            "source_tree_digest": expected_source_tree_digest,
            "contract_digest": expected_contract_digest,
            "asset_registry_digest": expected_asset_registry_digest,
            "previous_generation_digest": previous_generation,
        }
        if not isinstance(identity, dict) or any(identity.get(field) != value for field, value in expected_identity.items()):
            raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Frozen generation ancestry or scientific identity changed", observed=epoch)
        checkpoint_rows = [row for row in manifest.get("files", []) if str(row.get("path", "")).endswith(".pt")]
        if len(checkpoint_rows) != 1:
            raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Frozen generation does not bind exactly one checkpoint", observed=epoch)
        previous_checkpoint = str(checkpoint_rows[0]["sha256"])
        previous_generation = str(claimed_digest)
        largest_generation_bytes = max(
            largest_generation_bytes,
            manifest_path.stat().st_size + sum(int(row.get("bytes", 0)) for row in manifest.get("files", [])),
        )
        last_manifest = manifest

    assert last_manifest is not None
    checkpoint_rows = [row for row in last_manifest["files"] if str(row.get("path", "")).endswith(".pt")]
    checkpoint_path = _require_contained(discovered[-1][2] / checkpoint_rows[0]["path"], discovered[-1][2], role="Epoch checkpoint")
    checkpoint_sha = str(checkpoint_rows[0]["sha256"])
    payload = load_checkpoint(checkpoint_path, expected_sha256=checkpoint_sha, expected_epoch=last_epoch)
    payload_expected = {
        "training_seed": expected_training_seed,
        "source_tree_digest": expected_source_tree_digest,
        "asset_registry_digest": expected_asset_registry_digest,
        "global_step": last_epoch * 938,
        "base_sampler_generation": last_epoch,
    }
    if any(payload.get(field) != value for field, value in payload_expected.items()):
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Last resume checkpoint payload differs from the formal trajectory")
    checkpoint_rng = payload["rng_state"].digest()
    summary = json.loads((discovered[-1][2] / "EPOCH_EVIDENCE_SUMMARY.json").read_text(encoding="utf-8"))
    history_payload = summary.get("history_after_epoch")
    if not isinstance(history_payload, dict) or not isinstance(history_payload.get("counts"), dict) or not isinstance(history_payload.get("last_epoch"), dict):
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Last epoch replay history is missing")
    history = ReplayHistoryState(
        counts={str(key): int(value) for key, value in history_payload["counts"].items()},
        last_epoch={str(key): int(value) for key, value in history_payload["last_epoch"].items()},
        cumulative_occurrences=int(history_payload.get("cumulative_occurrences", -1)),
    )
    if (
        set(history.counts) != set(history.last_epoch)
        or any(value <= 0 for value in history.counts.values())
        or any(value < epoch_start or value > last_epoch for value in history.last_epoch.values())
        or sum(history.counts.values()) != history.cumulative_occurrences
        or history.snapshot() != history_payload
    ):
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Last epoch replay history is internally inconsistent")
    rng_evidence = summary.get("rng_evidence")
    last_identity = last_manifest.get("identity", {})
    if (
        not isinstance(rng_evidence, dict)
        or rng_evidence.get("recorder_epoch_start_digest") != last_identity.get("rng_state_digest")
        or rng_evidence.get("runtime_epoch_start_digest") != last_identity.get("rng_state_digest")
        or rng_evidence.get("runtime_epoch_end_digest") != checkpoint_rng
        or rng_evidence.get("finalize_entry_digest") != checkpoint_rng
        or summary.get("checkpoint_sha256") != checkpoint_sha
    ):
        raise SctsrError(ErrorCode.RESUME_GENERATION_MISMATCH, "Last epoch RNG/checkpoint evidence is not a closed continuation boundary")
    remaining_epochs = epoch_end - last_epoch
    required_free_bytes = max(
        int(minimum_free_bytes),
        math.ceil(largest_generation_bytes * remaining_epochs * 1.25) + 2 * checkpoint_path.stat().st_size,
    )
    observed_free_bytes = shutil.disk_usage(root).free
    if observed_free_bytes < required_free_bytes:
        raise SctsrError(ErrorCode.DISK_SPACE_PRECHECK_FAILED, "Insufficient disk space for safe formal resume", observed=observed_free_bytes, expected=required_free_bytes)
    return FormalResumeContext(
        run_root=canonical_root.as_posix(),
        run_id=expected_run_id,
        arm_id=expected_arm_id,
        training_seed=expected_training_seed,
        epoch_start=epoch_start,
        epoch_end=epoch_end,
        last_complete_epoch=last_epoch,
        resume_epoch=last_epoch + 1,
        checkpoint_path=_canonical_windows_path(checkpoint_path).as_posix(),
        checkpoint_sha256=checkpoint_sha,
        checkpoint_rng_digest=checkpoint_rng,
        global_step=int(payload["global_step"]),
        required_free_bytes=required_free_bytes,
        previous_generation_digest=previous_generation,
        receipt_chain_digest=str(pointer["receipt_chain_digest"]),
        history=history,
        quarantined_partial_paths=(),
        terminal_epoch_complete=last_epoch == epoch_end,
    )


def inspect_formal_resume_context(**kwargs: Any) -> FormalResumeContext:
    """Validate the frozen prefix read-only, fully rechecking only its endpoint."""

    return _inspect_formal_resume_context_from_frozen_prefix(**kwargs)


def prepare_resume(
    *,
    pointer_path: str | Path,
    transaction_root: str | Path,
    quarantine_root: str | Path,
    expected_identity: ResumeIdentity,
    minimum_free_bytes: int,
    disk_path: str | Path,
) -> dict[str, Any]:
    reconcile_epoch_publications(
        transaction_root,
        quarantine_root,
        expected_run_id=expected_identity.logical_run_id,
        expected_identity={
            "arm_id": expected_identity.arm_id,
            "training_seed": expected_identity.training_seed,
            "source_tree_digest": expected_identity.source_tree_digest,
            "contract_digest": expected_identity.contract_digest,
            "asset_registry_digest": expected_identity.asset_registry_digest,
        },
    )
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
    quarantined = quarantine_inprogress(transaction_root, quarantine_root, reason="RESUME_PRECHECK_PARTIAL")
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
