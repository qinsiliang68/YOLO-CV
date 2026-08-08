"""Immutable optimizer-free checkpoint packages for dedicated inference nodes."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

from .errors import ValidationError
from .util import atomic_write_json, sha256_file


class CheckpointPackageError(ValidationError):
    """Raised when a key checkpoint cannot be packaged without changing weights."""


@dataclass(frozen=True)
class CheckpointPackageSpec:
    run_id: str
    arm_id: str
    source_job_id: str
    source_machine_id: str
    logical_epoch: int
    source_checkpoint: Path
    output_dir: Path
    yolo_root: Path

    def __post_init__(self) -> None:
        for name in ("source_checkpoint", "output_dir", "yolo_root"):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        if self.logical_epoch <= 0:
            raise CheckpointPackageError("logical epoch must be positive")
        for name in ("run_id", "arm_id", "source_job_id", "source_machine_id"):
            if not str(getattr(self, name)).strip():
                raise CheckpointPackageError(f"{name} must not be empty")


@dataclass(frozen=True)
class CheckpointPackageResult:
    status: str
    skipped: bool
    checkpoint_path: Path
    manifest_path: Path
    source_checkpoint_sha256: str
    package_checkpoint_sha256: str


def model_state_digest(model: Any) -> str:
    """Hash names, shapes, dtypes, and exact bytes of every model state tensor."""

    if model is None or not hasattr(model, "state_dict"):
        raise CheckpointPackageError("checkpoint weight object has no state_dict")
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name]
        if not hasattr(tensor, "detach"):
            raise CheckpointPackageError(f"model state is not a tensor: {name}")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(dtype=__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest().upper()


def _activate_local_yolo(yolo_root: Path) -> None:
    if not yolo_root.is_dir():
        return
    from .campaign_dynamic_training import _activate_local_ultralytics

    _activate_local_ultralytics(yolo_root)


def _load(path: Path, yolo_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    _activate_local_yolo(yolo_root)
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise CheckpointPackageError("source checkpoint is not a dictionary")
    return payload


def _validate_epoch(payload: dict[str, Any], logical_epoch: int) -> int:
    try:
        internal = int(payload["epoch"])
    except Exception as exc:
        raise CheckpointPackageError("source checkpoint has no valid internal epoch") from exc
    if internal + 1 != logical_epoch:
        raise CheckpointPackageError(
            f"source checkpoint logical epoch {internal + 1} != requested logical epoch {logical_epoch}"
        )
    return internal


def _validate_existing(
    spec: CheckpointPackageSpec,
    source_sha: str,
) -> CheckpointPackageResult | None:
    output = spec.output_dir
    if not output.exists():
        return None
    files = list(output.iterdir()) if output.is_dir() else []
    if not files:
        return None
    checkpoint = output / f"checkpoint_epoch_{spec.logical_epoch:04d}.pt"
    manifest_path = output / "checkpoint_package_manifest.json"
    if not checkpoint.is_file() or not manifest_path.is_file():
        raise CheckpointPackageError(f"checkpoint package is half-published: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package_sha = sha256_file(checkpoint)
    expected = {
        "status": "COMPLETE",
        "run_id": spec.run_id,
        "arm_id": spec.arm_id,
        "source_job_id": spec.source_job_id,
        "source_machine_id": spec.source_machine_id,
        "logical_epoch": spec.logical_epoch,
        "source_checkpoint_sha256": source_sha,
        "package_checkpoint_sha256": package_sha,
    }
    mismatch = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatch:
        if "source_checkpoint_sha256" in mismatch:
            raise CheckpointPackageError(f"source checkpoint changed: {mismatch}")
        raise CheckpointPackageError(f"existing checkpoint package identity mismatch: {mismatch}")
    payload = _load(checkpoint, spec.yolo_root)
    _validate_epoch(payload, spec.logical_epoch)
    digest = model_state_digest(payload.get("model"))
    if digest != manifest.get("model_state_digest"):
        raise CheckpointPackageError("existing checkpoint package model-state digest mismatch")
    return CheckpointPackageResult(
        "PASS", True, checkpoint, manifest_path, source_sha, package_sha
    )


def build_checkpoint_package(spec: CheckpointPackageSpec) -> CheckpointPackageResult:
    """Strip optimizer state while retaining exact serialized EMA tensor values."""

    source_sha = sha256_file(spec.source_checkpoint)
    existing = _validate_existing(spec, source_sha)
    if existing is not None:
        return existing

    source: dict[str, Any] | None = None
    model = None
    staging = spec.output_dir.parent / f".{spec.output_dir.name}.{uuid.uuid4().hex}.tmpdir"
    try:
        source = _load(spec.source_checkpoint, spec.yolo_root)
        internal_epoch = _validate_epoch(source, spec.logical_epoch)
        weight_source = "ema" if source.get("ema") is not None else "model"
        source_model = source.get(weight_source)
        if source_model is None:
            raise CheckpointPackageError("source checkpoint has neither EMA nor model weights")
        source_digest = model_state_digest(source_model)
        model = copy.deepcopy(source_model).cpu()
        if hasattr(model, "criterion"):
            model.criterion = None
        if hasattr(model, "args") and not isinstance(model.args, dict):
            model.args = dict(model.args)
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "parameters"):
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        package_digest = model_state_digest(model)
        if package_digest != source_digest:
            raise CheckpointPackageError("optimizer stripping changed serialized model tensor values")

        package = dict(source)
        package.update(
            {
                "model": model,
                "ema": None,
                "optimizer": None,
                "scaler": None,
                "best_fitness": source.get("best_fitness"),
                "campaign_checkpoint_package": {
                    "schema_version": "stage1.checkpoint_package.v1",
                    "run_id": spec.run_id,
                    "arm_id": spec.arm_id,
                    "source_job_id": spec.source_job_id,
                    "source_machine_id": spec.source_machine_id,
                    "logical_epoch": spec.logical_epoch,
                    "source_checkpoint_sha256": source_sha,
                    "weight_source": weight_source,
                    "numerical_weight_transform": "NONE",
                    "model_state_digest": package_digest,
                },
            }
        )
        staging.mkdir(parents=True)
        checkpoint = staging / f"checkpoint_epoch_{spec.logical_epoch:04d}.pt"
        import torch

        torch.save(package, checkpoint)
        reloaded = _load(checkpoint, spec.yolo_root)
        _validate_epoch(reloaded, spec.logical_epoch)
        if model_state_digest(reloaded.get("model")) != package_digest:
            raise CheckpointPackageError("serialized package failed exact model-state validation")
        package_sha = sha256_file(checkpoint)
        manifest_path = staging / "checkpoint_package_manifest.json"
        atomic_write_json(
            manifest_path,
            {
                "schema_version": "stage1.checkpoint_package.v1",
                "status": "COMPLETE",
                "run_id": spec.run_id,
                "arm_id": spec.arm_id,
                "source_job_id": spec.source_job_id,
                "source_machine_id": spec.source_machine_id,
                "logical_epoch": spec.logical_epoch,
                "source_checkpoint_internal_epoch": internal_epoch,
                "source_checkpoint": str(spec.source_checkpoint),
                "source_checkpoint_sha256": source_sha,
                "source_checkpoint_size_bytes": spec.source_checkpoint.stat().st_size,
                "package_checkpoint": checkpoint.name,
                "package_checkpoint_sha256": package_sha,
                "package_checkpoint_size_bytes": checkpoint.stat().st_size,
                "weight_source": weight_source,
                "numerical_weight_transform": "NONE",
                "model_state_digest": package_digest,
            },
        )
        if spec.output_dir.exists():
            spec.output_dir.rmdir()
        os.replace(staging, spec.output_dir)
        return CheckpointPackageResult(
            "PASS",
            False,
            spec.output_dir / checkpoint.name,
            spec.output_dir / manifest_path.name,
            source_sha,
            package_sha,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        del model
        del source
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


__all__ = [
    "CheckpointPackageError",
    "CheckpointPackageResult",
    "CheckpointPackageSpec",
    "build_checkpoint_package",
    "model_state_digest",
]
