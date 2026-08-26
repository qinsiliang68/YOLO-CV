from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .errors import ErrorCode, SctsrError
from .serialization import atomic_write_json, load_json, sha256_file, stable_digest


FORMAL_COMPLETION_SCHEMA = "stage1.sctsr.formal_completion_receipt.v1"
FORMAL_COMPLETION_FILENAME = "FORMAL_COMPLETION_RECEIPT.json"
_RUN_STATE = {
    "COMMON_PARENT": (
        "PARENT_RECEIPT.json",
        "stage1.sctsr.formal_parent_receipt.v3",
        "FORMAL_PARENT_EPOCHS_COMPLETE_PENDING_FINALIZATION",
        "FORMAL_PARENT_COMPLETE",
        120,
    ),
    "BRANCH": (
        "BRANCH_RECEIPT.json",
        "stage1.sctsr.formal_branch_receipt.v3",
        "FORMAL_BRANCH_ENDPOINT_COMPLETE_PENDING_COMMIT",
        "FORMAL_BRANCH_COMPLETE",
        200,
    ),
}


def _failure(message: str, *, artifact_path: str | None = None, observed: Any = None) -> SctsrError:
    return SctsrError(
        ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE,
        message,
        artifact_path=artifact_path,
        observed=observed,
        required_action="Do not treat this run as complete; resume finalization from the latest immutable epoch.",
    )


def _validated_artifact_index_payload(index: Mapping[str, Any]) -> None:
    files = index.get("files")
    if index.get("schema_version") != "stage1.sctsr.artifact_index.v1" or not isinstance(files, list):
        raise _failure("Formal artifact index schema is invalid")
    paths: list[str] = []
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {"path", "bytes", "sha256"}:
            raise _failure("Formal artifact index row fields are invalid", observed=row)
        relative = Path(str(row["path"]))
        digest = str(row["sha256"])
        if relative.is_absolute() or relative.drive or ".." in relative.parts or int(row["bytes"]) < 0:
            raise _failure("Formal artifact index row escapes its run root", observed=row)
        try:
            if len(digest) != 64:
                raise ValueError
            bytes.fromhex(digest)
        except ValueError as exc:
            raise _failure("Formal artifact index row SHA-256 is invalid", observed=row) from exc
        paths.append(relative.as_posix())
    if paths != sorted(paths) or len(paths) != len(set(paths)) or index.get("artifact_index_digest") != stable_digest(files):
        raise _failure("Formal artifact index ordering or digest is invalid")


def _validated_inputs(
    root: Path,
    run_role: str,
    *,
    artifact_index_already_validated: bool = False,
) -> dict[str, Any]:
    if run_role not in _RUN_STATE:
        raise _failure("Formal completion run role is invalid", observed=run_role)
    state_name, state_schema, pending_status, _complete_status, terminal_epoch = _RUN_STATE[run_role]
    required = {
        "run_state": root / state_name,
        "run_manifest": root / "RUN_MANIFEST.json",
        "artifact_index": root / "ARTIFACT_INDEX.json",
        "generation_index": root / "ARTIFACT_INDEX_GENERATIONS.json",
    }
    if run_role == "BRANCH":
        required["endpoint_receipt"] = root / "08_receipts" / "FORMAL_ENDPOINT_RECEIPT.json"
    missing = sorted(role for role, path in required.items() if not path.is_file())
    if missing:
        raise _failure("Formal completion lacks required final artifacts", observed=missing)
    state = load_json(required["run_state"])
    if (
        not isinstance(state, Mapping)
        or state.get("schema_version") != state_schema
        or state.get("status") != pending_status
        or int(state.get("epoch_end", -1)) != terminal_epoch
    ):
        raise _failure("Formal run state is not ready for atomic completion", artifact_path=required["run_state"].as_posix(), observed=state)
    manifest = load_json(required["run_manifest"])
    if manifest.get("execution_mode") != "formal" or manifest.get("run_role") != run_role:
        raise _failure("Formal run manifest role or mode is invalid", observed=manifest)
    generation_index = load_json(required["generation_index"])
    if generation_index.get("schema_version") != "stage1.sctsr.epoch_artifact_index.v1":
        raise _failure("Formal generation index schema is invalid")
    if run_role == "BRANCH":
        endpoint = load_json(required["endpoint_receipt"])
        if endpoint.get("status") != "FORMAL_ENDPOINT_COMPLETE_NOT_METHOD_SELECTION":
            raise _failure("Formal endpoint receipt is not complete", observed=endpoint.get("status"))
    observed_index = load_json(required["artifact_index"])
    _validated_artifact_index_payload(observed_index)
    if not artifact_index_already_validated:
        from .run_validation import build_artifact_index

        expected_index = build_artifact_index(root)
        if observed_index != expected_index:
            raise _failure(
                "Formal exhaustive artifact index is stale or incomplete",
                artifact_path=required["artifact_index"].as_posix(),
                observed={
                    "registered_digest": observed_index.get("artifact_index_digest"),
                    "current_digest": expected_index.get("artifact_index_digest"),
                },
            )
    return {"paths": required, "state": dict(state), "manifest": dict(manifest), "artifact_index": observed_index}


def publish_formal_completion(
    run_root: str | Path,
    *,
    run_role: str,
    run_id: str,
    arm_id: str,
    training_seed: int,
    terminal_epoch: int,
    fixed_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Atomically publish the sole canonical completion fact for one run."""

    root = Path(run_root).resolve()
    marker = root / FORMAL_COMPLETION_FILENAME
    if marker.exists():
        raise _failure("Formal completion receipt is immutable and already exists", artifact_path=marker.as_posix())
    validated = _validated_inputs(root, run_role)
    state_name, _state_schema, _pending, complete_status, expected_terminal = _RUN_STATE[run_role]
    if int(terminal_epoch) != expected_terminal:
        raise _failure("Formal completion terminal epoch is invalid", observed=terminal_epoch)
    if len(str(fixed_checkpoint_sha256)) != 64:
        raise _failure("Formal completion checkpoint digest is not SHA-256", observed=fixed_checkpoint_sha256)
    try:
        bytes.fromhex(str(fixed_checkpoint_sha256))
    except ValueError as exc:
        raise _failure("Formal completion checkpoint digest is not hexadecimal", observed=fixed_checkpoint_sha256) from exc
    state = validated["state"]
    manifest = validated["manifest"]
    expected_identity = {
        "run_id": str(run_id),
        "arm_id": str(arm_id),
        "training_seed": int(training_seed),
    }
    manifest_mismatch = {
        field: {"observed": manifest.get(field), "expected": value}
        for field, value in expected_identity.items()
        if manifest.get(field) != value
    }
    state_run_id = state.get("parent_id") if run_role == "COMMON_PARENT" else state.get("logical_run_id")
    state_mismatch = {
        "run_id": {"observed": state_run_id, "expected": str(run_id)},
        "arm_id": {"observed": state.get("arm_id", "COMMON_PARENT_NR"), "expected": str(arm_id)},
        "training_seed": {"observed": state.get("training_seed"), "expected": int(training_seed)},
    }
    state_mismatch = {field: values for field, values in state_mismatch.items() if values["observed"] != values["expected"]}
    if manifest_mismatch or state_mismatch:
        raise _failure(
            "Formal completion identity differs across run state and manifest",
            observed={"manifest": manifest_mismatch, "run_state": state_mismatch},
        )
    state_checkpoint = (
        state.get("checkpoint_sha256")
        if run_role == "COMMON_PARENT"
        else (state.get("fixed_formal_endpoint") or {}).get("sha256")
    )
    if str(state_checkpoint).upper() != str(fixed_checkpoint_sha256).upper():
        raise _failure("Formal completion checkpoint differs from terminal run state", observed=state_checkpoint)
    paths: Mapping[str, Path] = validated["paths"]
    core: dict[str, Any] = {
        "schema_version": FORMAL_COMPLETION_SCHEMA,
        "status": complete_status,
        "run_role": run_role,
        "run_id": str(run_id),
        "arm_id": str(arm_id),
        "training_seed": int(training_seed),
        "terminal_epoch": int(terminal_epoch),
        "fixed_checkpoint_sha256": str(fixed_checkpoint_sha256).upper(),
        "epoch_receipt_digest": state.get("epoch_receipt_digest"),
        "run_state_path": state_name,
        "run_state_sha256": sha256_file(paths["run_state"]),
        "run_manifest_sha256": sha256_file(paths["run_manifest"]),
        "artifact_index_sha256": sha256_file(paths["artifact_index"]),
        "artifact_index_digest": validated["artifact_index"]["artifact_index_digest"],
        "generation_index_sha256": sha256_file(paths["generation_index"]),
        "endpoint_receipt_sha256": (
            sha256_file(paths["endpoint_receipt"]) if "endpoint_receipt" in paths else None
        ),
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    receipt = {**core, "completion_digest": stable_digest(core)}
    atomic_write_json(marker, receipt)
    return dict(validate_formal_completion(root, expected_run_role=run_role)["receipt"])


def _validate_formal_completion(
    run_root: str | Path,
    *,
    expected_run_role: str,
    artifact_index_already_validated: bool,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    marker = root / FORMAL_COMPLETION_FILENAME
    if not marker.is_file() or marker.is_symlink():
        raise _failure("Formal completion receipt is missing or indirect", artifact_path=marker.as_posix())
    receipt = load_json(marker)
    required = {
        "schema_version",
        "status",
        "run_role",
        "run_id",
        "arm_id",
        "training_seed",
        "terminal_epoch",
        "fixed_checkpoint_sha256",
        "epoch_receipt_digest",
        "run_state_path",
        "run_state_sha256",
        "run_manifest_sha256",
        "artifact_index_sha256",
        "artifact_index_digest",
        "generation_index_sha256",
        "endpoint_receipt_sha256",
        "completed_at_utc",
        "completion_digest",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required:
        raise _failure("Formal completion receipt schema fields are invalid", observed=receipt)
    core = {key: value for key, value in receipt.items() if key != "completion_digest"}
    _state_name, _schema, _pending, complete_status, terminal_epoch = _RUN_STATE.get(
        expected_run_role,
        (None, None, None, None, None),
    )
    if any(
        (
            receipt.get("schema_version") != FORMAL_COMPLETION_SCHEMA,
            receipt.get("run_role") != expected_run_role,
            receipt.get("status") != complete_status,
            receipt.get("terminal_epoch") != terminal_epoch,
            receipt.get("completion_digest") != stable_digest(core),
        )
    ):
        raise _failure("Formal completion receipt identity or digest is invalid", observed=receipt)
    validated = _validated_inputs(
        root,
        expected_run_role,
        artifact_index_already_validated=artifact_index_already_validated,
    )
    state = validated["state"]
    manifest = validated["manifest"]
    state_run_id = state.get("parent_id") if expected_run_role == "COMMON_PARENT" else state.get("logical_run_id")
    identity_mismatch = {
        "manifest.run_id": (manifest.get("run_id"), receipt.get("run_id")),
        "manifest.arm_id": (manifest.get("arm_id"), receipt.get("arm_id")),
        "manifest.training_seed": (manifest.get("training_seed"), receipt.get("training_seed")),
        "run_state.run_id": (state_run_id, receipt.get("run_id")),
        "run_state.arm_id": (state.get("arm_id", "COMMON_PARENT_NR"), receipt.get("arm_id")),
        "run_state.training_seed": (state.get("training_seed"), receipt.get("training_seed")),
        "run_state.epoch_receipt_digest": (state.get("epoch_receipt_digest"), receipt.get("epoch_receipt_digest")),
    }
    identity_mismatch = {
        field: {"observed": values[0], "expected": values[1]}
        for field, values in identity_mismatch.items()
        if values[0] != values[1]
    }
    if identity_mismatch:
        raise _failure("Formal completion identity no longer matches bound run artifacts", observed=identity_mismatch)
    paths: Mapping[str, Path] = validated["paths"]
    comparisons = {
        "run_state_sha256": sha256_file(paths["run_state"]),
        "run_manifest_sha256": sha256_file(paths["run_manifest"]),
        "artifact_index_sha256": sha256_file(paths["artifact_index"]),
        "artifact_index_digest": validated["artifact_index"]["artifact_index_digest"],
        "generation_index_sha256": sha256_file(paths["generation_index"]),
        "endpoint_receipt_sha256": sha256_file(paths["endpoint_receipt"]) if "endpoint_receipt" in paths else None,
    }
    mismatch = {
        field: {"receipt": receipt.get(field), "observed": value}
        for field, value in comparisons.items()
        if receipt.get(field) != value
    }
    if mismatch:
        raise _failure("Formal completion receipt no longer binds current artifacts", observed=mismatch)
    return {"status": "PASS", "receipt_path": marker.as_posix(), "receipt_sha256": sha256_file(marker), "receipt": dict(receipt)}


def validate_formal_completion(
    run_root: str | Path,
    *,
    expected_run_role: str,
) -> dict[str, Any]:
    """Exhaustively revalidate a completed formal run."""

    return _validate_formal_completion(
        run_root,
        expected_run_role=expected_run_role,
        artifact_index_already_validated=False,
    )


def _validate_formal_completion_with_prevalidated_artifact_index(
    run_root: str | Path,
    *,
    expected_run_role: str,
) -> dict[str, Any]:
    """Recheck the small signed bindings after this process audited the full tree."""

    return _validate_formal_completion(
        run_root,
        expected_run_role=expected_run_role,
        artifact_index_already_validated=True,
    )
