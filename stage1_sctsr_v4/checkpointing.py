from __future__ import annotations

import os
import tempfile
import dataclasses
import hashlib
import math
import struct
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .errors import ErrorCode, SctsrError
from .rng_isolation import RngSnapshot, capture_global_rng
from .serialization import _fsync_directory, sha256_file

REQUIRED_CHECKPOINT_KEYS = {
    "schema_version", "model_state", "ema_state", "ema_updates", "optimizer_state",
    "scheduler_state", "scaler_state", "rng_state", "epoch", "global_step",
    "base_sampler_generation", "canonical_training_lock_sha256", "initial_checkpoint_sha256",
    "base_manifest_sha256", "training_seed", "source_tree_digest", "runtime_config_digest",
    "asset_registry_digest", "checkpoint_payload_digest",
}

CHECKPOINT_DIGEST_ALGORITHM = "SHA256_TYPED_RECURSIVE_FULL_CONTENT_V1"
SHA_IDENTITY_FIELDS = (
    "canonical_training_lock_sha256",
    "initial_checkpoint_sha256",
    "base_manifest_sha256",
    "source_tree_digest",
    "runtime_config_digest",
    "asset_registry_digest",
)


def _write_length_prefixed(hasher: Any, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)


def _mapping_key_token(key: Any) -> bytes:
    if key is None:
        return b"n:"
    if isinstance(key, bool):
        return b"b:1" if key else b"b:0"
    if isinstance(key, int):
        return b"i:" + str(key).encode("ascii")
    if isinstance(key, str):
        return b"s:" + key.encode("utf-8")
    raise SctsrError(
        ErrorCode.PARENT_CHECKPOINT_INCOMPLETE,
        "Checkpoint mapping contains an unsupported key type",
        observed=type(key).__qualname__,
    )


def _update_checkpoint_hash(hasher: Any, value: Any) -> None:
    if dataclasses.is_dataclass(value):
        hasher.update(b"D")
        _write_length_prefixed(hasher, f"{type(value).__module__}.{type(value).__qualname__}".encode("utf-8"))
        for field in dataclasses.fields(value):
            _write_length_prefixed(hasher, field.name.encode("utf-8"))
            _update_checkpoint_hash(hasher, getattr(value, field.name))
        return
    if value is None:
        hasher.update(b"N")
        return
    if isinstance(value, bool):
        hasher.update(b"B1" if value else b"B0")
        return
    if isinstance(value, int):
        hasher.update(b"I")
        _write_length_prefixed(hasher, str(value).encode("ascii"))
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SctsrError(ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, "Checkpoint contains non-finite floating state")
        hasher.update(b"F")
        hasher.update(struct.pack("!d", value))
        return
    if isinstance(value, str):
        hasher.update(b"S")
        _write_length_prefixed(hasher, value.encode("utf-8"))
        return
    if isinstance(value, bytes):
        hasher.update(b"Y")
        _write_length_prefixed(hasher, value)
        return
    if isinstance(value, torch.Tensor):
        if value.layout is not torch.strided:
            raise SctsrError(ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, "Checkpoint contains an unsupported non-strided tensor", observed=str(value.layout))
        # ``Tensor.view(torch.uint8)`` rejects zero-dimensional tensors when
        # the element size changes, and NumPy cannot represent every PyTorch
        # dtype (notably bfloat16).  A contiguous clone owns an exact-sized,
        # offset-zero storage, so hashing that storage covers every value byte
        # without either limitation.
        tensor = value.detach().cpu().contiguous().clone()
        hasher.update(b"T")
        _write_length_prefixed(hasher, str(tensor.dtype).encode("ascii"))
        _update_checkpoint_hash(hasher, tuple(int(item) for item in tensor.shape))
        raw = bytes(tensor.untyped_storage())
        _write_length_prefixed(hasher, raw)
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(b"A")
        _write_length_prefixed(hasher, array.dtype.str.encode("ascii"))
        _update_checkpoint_hash(hasher, tuple(int(item) for item in array.shape))
        _write_length_prefixed(hasher, array.tobytes(order="C"))
        return
    if isinstance(value, np.generic):
        _update_checkpoint_hash(hasher, value.item())
        return
    if isinstance(value, Mapping):
        hasher.update(b"M")
        ordered = sorted(((_mapping_key_token(key), key) for key in value), key=lambda item: item[0])
        _update_checkpoint_hash(hasher, len(ordered))
        for token, key in ordered:
            _write_length_prefixed(hasher, token)
            _update_checkpoint_hash(hasher, value[key])
        return
    if isinstance(value, tuple):
        hasher.update(b"Q")
        _update_checkpoint_hash(hasher, len(value))
        for item in value:
            _update_checkpoint_hash(hasher, item)
        return
    if isinstance(value, list):
        hasher.update(b"L")
        _update_checkpoint_hash(hasher, len(value))
        for item in value:
            _update_checkpoint_hash(hasher, item)
        return
    raise SctsrError(
        ErrorCode.PARENT_CHECKPOINT_INCOMPLETE,
        "Checkpoint contains an unsupported value type",
        observed=f"{type(value).__module__}.{type(value).__qualname__}",
    )


def checkpoint_payload_digest(payload: Mapping[str, Any]) -> str:
    hasher = hashlib.sha256()
    _update_checkpoint_hash(
        hasher,
        {key: value for key, value in payload.items() if key != "checkpoint_payload_digest"},
    )
    return hasher.hexdigest().upper()


def _ema_state(ema: Any) -> tuple[Mapping[str, Any], int]:
    if ema is None:
        return {}, 0
    if hasattr(ema, "state_dict"):
        state = ema.state_dict()
    elif hasattr(ema, "ema") and hasattr(ema.ema, "state_dict"):
        state = {"ema_model_state": ema.ema.state_dict(), "updates": int(getattr(ema, "updates", 0))}
    else:
        state = {}
    updates = int(getattr(ema, "updates", state.get("updates", 0)))
    return state, updates


def build_checkpoint_payload(
    *, model: torch.nn.Module, ema: Any, optimizer: torch.optim.Optimizer,
    scheduler: Any, scaler: Any, epoch: int, global_step: int,
    base_sampler_generation: int, canonical_training_lock_sha256: str,
    initial_checkpoint_sha256: str, base_manifest_sha256: str, training_seed: int,
    source_tree_digest: str, runtime_config_digest: str, asset_registry_digest: str,
) -> dict[str, Any]:
    ema_state, ema_updates = _ema_state(ema)
    payload: dict[str, Any] = {
        "schema_version": "stage1.sctsr.checkpoint.v1",
        "model_state": model.state_dict(),
        "ema_state": ema_state,
        "ema_updates": ema_updates,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": {} if scheduler is None else scheduler.state_dict(),
        "scaler_state": {} if scaler is None else scaler.state_dict(),
        "rng_state": capture_global_rng(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "base_sampler_generation": int(base_sampler_generation),
        "canonical_training_lock_sha256": canonical_training_lock_sha256.upper(),
        "initial_checkpoint_sha256": initial_checkpoint_sha256.upper(),
        "base_manifest_sha256": base_manifest_sha256.upper(),
        "training_seed": int(training_seed),
        "source_tree_digest": source_tree_digest.upper(),
        "runtime_config_digest": runtime_config_digest.upper(),
        "asset_registry_digest": asset_registry_digest.upper(),
    }
    payload["checkpoint_payload_digest"] = checkpoint_payload_digest(payload)
    return payload


def validate_checkpoint_payload(payload: Mapping[str, Any], *, expected_epoch: int | None = None) -> None:
    missing = sorted(REQUIRED_CHECKPOINT_KEYS - set(payload))
    if missing:
        raise SctsrError(ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, "Checkpoint payload is incomplete", observed=missing)
    if payload.get("schema_version") != "stage1.sctsr.checkpoint.v1":
        raise SctsrError(ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, "Unknown checkpoint schema")
    if expected_epoch is not None and int(payload["epoch"]) != expected_epoch:
        raise SctsrError(ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, "Checkpoint epoch mismatch", observed=payload["epoch"], expected=expected_epoch)
    for field in SHA_IDENTITY_FIELDS:
        value = str(payload.get(field, ""))
        if len(value) != 64 or value.upper() != value or any(char not in "0123456789ABCDEF" for char in value):
            raise SctsrError(ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, "Checkpoint identity is not a canonical SHA-256", failing_field=field, observed=value)
    if int(payload["epoch"]) < 1 or int(payload["global_step"]) < 0 or int(payload["base_sampler_generation"]) < 1:
        raise SctsrError(ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, "Checkpoint epoch/step/generation values are invalid")
    state_contract = {
        "model_state": (Mapping, True),
        "ema_state": (Mapping, True),
        "optimizer_state": (Mapping, True),
        "scheduler_state": (Mapping, True),
        "scaler_state": (Mapping, False),
    }
    for field, (expected_type, must_be_nonempty) in state_contract.items():
        state = payload[field]
        if not isinstance(state, expected_type) or (must_be_nonempty and not state):
            raise SctsrError(
                ErrorCode.PARENT_CHECKPOINT_INCOMPLETE,
                "Checkpoint training state is missing or empty",
                failing_field=field,
            )
    optimizer_state = payload["optimizer_state"]
    if "state" not in optimizer_state or not optimizer_state.get("param_groups"):
        raise SctsrError(ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, "Optimizer state is structurally incomplete", failing_field="optimizer_state")
    if not isinstance(payload["rng_state"], RngSnapshot):
        raise SctsrError(ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, "Checkpoint RNG state has the wrong schema", failing_field="rng_state")
    if int(payload["ema_updates"]) < 0:
        raise SctsrError(ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, "EMA update count is invalid", failing_field="ema_updates")
    observed_digest = str(payload.get("checkpoint_payload_digest", "")).upper()
    expected_digest = checkpoint_payload_digest(payload)
    if observed_digest != expected_digest:
        raise SctsrError(
            ErrorCode.IDENTITY_DIGEST_MISMATCH,
            "Checkpoint payload digest does not match full checkpoint content",
            failing_field="checkpoint_payload_digest",
            observed=observed_digest,
            expected=expected_digest,
        )


def save_checkpoint_atomic(path: str | Path, payload: Mapping[str, Any]) -> str:
    validate_checkpoint_payload(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise SctsrError(ErrorCode.CHILD_MUTATED_PARENT, "Checkpoint publication may not overwrite an existing generation", artifact_path=str(destination))
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        with open(temp_name, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        verification = torch.load(temp_name, map_location="cpu", weights_only=False)
        if not isinstance(verification, Mapping):
            raise SctsrError(ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, "Written checkpoint is not a mapping")
        validate_checkpoint_payload(verification, expected_epoch=int(payload["epoch"]))
        os.replace(temp_name, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        try: os.unlink(temp_name)
        except FileNotFoundError: pass
        raise
    return sha256_file(destination)


def load_checkpoint(path: str | Path, *, expected_sha256: str | None = None, expected_epoch: int | None = None) -> dict[str, Any]:
    checkpoint = Path(path)
    if checkpoint.name.lower() == "best.pt":
        raise SctsrError(ErrorCode.BEST_PT_FORBIDDEN, "best.pt may not be used by SCTSR v4")
    observed_sha = sha256_file(checkpoint)
    if expected_sha256 is not None and observed_sha != expected_sha256.upper():
        raise SctsrError(ErrorCode.PARENT_SHA_MISMATCH, "Checkpoint SHA mismatch", artifact_path=str(checkpoint), observed=observed_sha, expected=expected_sha256.upper())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise SctsrError(ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, "Checkpoint is not a mapping payload")
    validate_checkpoint_payload(payload, expected_epoch=expected_epoch)
    return payload
