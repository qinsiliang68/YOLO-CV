from __future__ import annotations

import importlib
import platform
import json
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from .contract import Contract
from .errors import ValidationError
from .util import atomic_write_json, sha256_file


AUDIT_SCHEMA = "stage1_gapvalue240.training_execution_audit.v1"
RESUME_MODE = "native_approximate"


def _major_minor(value: str) -> tuple[int, int]:
    parts: list[int] = []
    for token in str(value).split("."):
        digits = "".join(character for character in token if character.isdigit())
        if digits:
            parts.append(int(digits))
        if len(parts) == 2:
            break
    return tuple(parts[:2]) if len(parts) == 2 else (-1, -1)


def validate_formal_environment(contract: Contract, yolo_root: str | Path) -> dict[str, Any]:
    """Validate runtime versions inside the disposable training process."""

    root = Path(yolo_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    sys.path.insert(0, str(root))
    modules = {
        "numpy": importlib.import_module("numpy"),
        "pandas": importlib.import_module("pandas"),
        "sklearn": importlib.import_module("sklearn"),
        "torch": importlib.import_module("torch"),
        "ultralytics": importlib.import_module("ultralytics"),
        "polars": importlib.import_module("polars"),
    }
    ultralytics_path = Path(modules["ultralytics"].__file__).resolve()
    try:
        ultralytics_path.relative_to(root)
    except ValueError as exc:
        raise ValidationError(f"Ultralytics resolved outside frozen YOLO root: {ultralytics_path}") from exc
    expected = contract.data["environment"]
    actual = {
        "python": platform.python_version(),
        "numpy": str(modules["numpy"].__version__),
        "pandas": str(modules["pandas"].__version__),
        "scikit_learn": str(modules["sklearn"].__version__),
        "pytorch": str(modules["torch"].__version__),
        "ultralytics": str(modules["ultralytics"].__version__),
        "polars": str(modules["polars"].__version__),
        "cuda_build": str(modules["torch"].version.cuda),
        "ultralytics_module": str(ultralytics_path),
    }
    checks = {}
    for key in ("python", "numpy", "pandas", "scikit_learn", "pytorch", "ultralytics", "polars", "cuda_build"):
        checks[key] = {
            "expected": str(expected[key]),
            "actual": actual[key],
            "ok": _major_minor(actual[key]) == _major_minor(str(expected[key])),
        }
    issues = [key for key, check in checks.items() if not check["ok"]]
    if issues:
        raise ValidationError(f"Formal environment major/minor mismatch: {issues}: {checks}")
    return {"status": "PASS", "actual": actual, "checks": checks}


@dataclass(frozen=True)
class FormalTrainingSpec:
    dataset_dir: Path
    checkpoint: Path
    output_dir: Path
    yolo_root: Path
    epochs: int
    batch: int
    imgsz: int
    seed: int
    device: str
    workers: int
    expected_steps_per_epoch: int
    patience: int = 0
    deterministic: bool = True
    cache: bool = False
    resume_checkpoint: Path | None = None
    segment_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("dataset_dir", "checkpoint", "output_dir", "yolo_root"):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)).resolve())
        if self.resume_checkpoint is not None:
            object.__setattr__(self, "resume_checkpoint", Path(self.resume_checkpoint).resolve())
        if self.epochs <= 0 or self.batch <= 0 or self.imgsz <= 0 or self.expected_steps_per_epoch <= 0:
            raise ValidationError("Formal training dimensions and expected steps must be positive")
        if self.patience != 0 or self.deterministic is not True or self.cache is not False:
            raise ValidationError("Formal training requires patience=0, deterministic=true, cache=false")

    @classmethod
    def from_contract(
        cls,
        contract: Contract,
        *,
        dataset_dir: str | Path,
        checkpoint: str | Path,
        output_dir: str | Path,
        yolo_root: str | Path,
        training_seed: int,
        budget: int,
        device: str | int,
        workers: int,
        resume_checkpoint: str | Path | None = None,
        segment_id: str | None = None,
    ) -> "FormalTrainingSpec":
        training = contract.data["training"]
        step_key = f"B{int(budget)}"
        if step_key not in training["expected_steps"]:
            raise ValidationError(f"Contract has no expected steps for budget {budget}")
        return cls(
            dataset_dir=Path(dataset_dir),
            checkpoint=Path(checkpoint),
            output_dir=Path(output_dir),
            yolo_root=Path(yolo_root),
            epochs=int(training["epochs"]),
            batch=int(training["batch_size"]),
            imgsz=int(training["image_size"]),
            seed=int(training_seed),
            device=str(device),
            workers=int(workers),
            expected_steps_per_epoch=int(training["expected_steps"][step_key]),
            patience=0,
            deterministic=bool(training["deterministic"]),
            cache=False,
            resume_checkpoint=Path(resume_checkpoint) if resume_checkpoint else None,
            segment_id=segment_id,
        )

    @property
    def configured_args(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch": self.batch,
            "imgsz": self.imgsz,
            "patience": self.patience,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "cache": self.cache,
            "model": str(self.checkpoint),
        }


@dataclass(frozen=True)
class FormalTrainingResult:
    trainer_dir: Path
    results_csv: Path
    args_yaml: Path
    best_checkpoint: Path
    stable_last: Path
    audit_path: Path
    resolved_args_path: Path


def _local_yolo(yolo_root: Path):
    if not yolo_root.is_dir():
        raise FileNotFoundError(f"Missing local YOLOv11 source: {yolo_root}")
    sys.path.insert(0, str(yolo_root))
    module = importlib.import_module("ultralytics")
    module_path = Path(module.__file__).resolve()
    try:
        module_path.relative_to(yolo_root)
    except ValueError as exc:
        raise RuntimeError(f"Ultralytics resolved outside frozen YOLO root: {module_path}") from exc
    return module.YOLO


def _default_checkpoint_validator(path: Path, resume: bool, yolo_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    sys.path.insert(0, str(yolo_root))
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValidationError(f"Checkpoint cannot be loaded: {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ValidationError(f"Checkpoint root is not a mapping: {path}")
    if resume:
        epoch = int(checkpoint.get("epoch", -1))
        if epoch < 0 or checkpoint.get("optimizer") is None:
            raise ValidationError(f"Checkpoint is not resumable (epoch/optimizer missing): {path}")
    return {
        "epoch": int(checkpoint.get("epoch", -1)),
        "train_args": checkpoint.get("train_args") if isinstance(checkpoint.get("train_args"), dict) else {},
    }


def _is_finite(value: Any) -> bool:
    if value is None:
        return False
    try:
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        return bool(np.isfinite(np.asarray(value, dtype=float)).all())
    except Exception:
        return False


class TrainingAuditRecorder:
    def __init__(self, spec: FormalTrainingSpec, path: Path):
        self.spec = spec
        self.path = path
        self.resumed = spec.resume_checkpoint is not None
        previous = None
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if not self.resumed:
                raise ValidationError(f"Training audit already exists; explicit resume is required: {path}")
            self._validate_previous(previous)
        elif self.resumed:
            raise ValidationError(f"Resume requested without prior training audit: {path}")

        configured = spec.configured_args
        observed = list(previous["observed_steps_per_epoch"]) if previous else []
        segments = list(previous["resume_segments"]) if previous else []
        for segment in segments:
            if segment.get("status") == "RUNNING":
                segment["status"] = "INTERRUPTED"
        requested_segment_id = spec.segment_id or uuid.uuid4().hex
        existing_ids = {str(segment.get("segment_id")) for segment in segments}
        segment_id = requested_segment_id
        suffix = 2
        while segment_id in existing_ids:
            segment_id = f"{requested_segment_id}__{suffix}"
            suffix += 1
        self.segment: dict[str, Any] = {
            "segment_id": segment_id,
            "resumed": self.resumed,
            "resume_checkpoint_sha256": sha256_file(spec.resume_checkpoint) if self.resumed else None,
            "start_epoch": None,
            "end_epoch": None,
            "observed_steps_per_epoch": [],
            "optimizer_steps": 0,
            "loss_finite": True,
            "started_at": time.time(),
            "ended_at": None,
            "status": "RUNNING",
        }
        segments.append(self.segment)
        self.audit: dict[str, Any] = {
            "schema_version": AUDIT_SCHEMA,
            "expected_epochs": spec.epochs,
            "completed_epochs": int(previous["completed_epochs"]) if previous else 0,
            "expected_steps_per_epoch": spec.expected_steps_per_epoch,
            "observed_steps_per_epoch": observed,
            "optimizer_steps_total": int(previous["optimizer_steps_total"]) if previous else 0,
            "effective_batch_size": int(previous["effective_batch_size"]) if previous else spec.batch,
            "configured_args": configured,
            "loss_finite": bool(previous["loss_finite"]) if previous else True,
            "resume_mode": RESUME_MODE,
            "resume_count": int(previous["resume_count"]) + (1 if self.resumed else 0) if previous else 0,
            "resume_segments": segments,
        }
        self.current_batches = 0
        self.current_epoch: int | None = None
        self._optimizer_patched = False
        self._write()

    def _validate_previous(self, previous: dict[str, Any]) -> None:
        if previous.get("schema_version") != AUDIT_SCHEMA:
            raise ValidationError("Cannot resume from an incompatible training audit schema")
        for key, expected in (
            ("expected_epochs", self.spec.epochs),
            ("expected_steps_per_epoch", self.spec.expected_steps_per_epoch),
            ("configured_args", self.spec.configured_args),
        ):
            if previous.get(key) != expected:
                raise ValidationError(f"Resume audit mismatch for {key}: {previous.get(key)!r} != {expected!r}")

    def _write(self) -> None:
        atomic_write_json(self.path, self.audit, overwrite=True)

    def on_train_start(self, trainer) -> None:
        start_epoch = int(getattr(trainer, "start_epoch", 0)) + 1
        self.segment["start_epoch"] = start_epoch
        self.audit["effective_batch_size"] = int(getattr(trainer, "batch_size", self.spec.batch))
        if not self._optimizer_patched:
            original = trainer.optimizer_step

            def counted_optimizer_step(*args, **kwargs):
                result = original(*args, **kwargs)
                self.segment["optimizer_steps"] += 1
                self.audit["optimizer_steps_total"] += 1
                return result

            trainer.optimizer_step = counted_optimizer_step
            self._optimizer_patched = True
        self._write()

    def on_train_epoch_start(self, trainer) -> None:
        self.current_epoch = int(trainer.epoch) + 1
        self.current_batches = 0

    def on_train_batch_end(self, _trainer) -> None:
        self.current_batches += 1

    def on_train_epoch_end(self, trainer) -> None:
        epoch = int(trainer.epoch) + 1
        if self.current_epoch != epoch:
            self.current_epoch = epoch
        while len(self.audit["observed_steps_per_epoch"]) < epoch:
            self.audit["observed_steps_per_epoch"].append(None)
        self.audit["observed_steps_per_epoch"][epoch - 1] = self.current_batches
        self.segment["observed_steps_per_epoch"].append({"epoch": epoch, "steps": self.current_batches})
        self.segment["end_epoch"] = epoch
        finite = _is_finite(getattr(trainer, "tloss", None)) and _is_finite(getattr(trainer, "loss", None))
        self.segment["loss_finite"] = bool(self.segment["loss_finite"] and finite)
        self.audit["loss_finite"] = bool(self.audit["loss_finite"] and finite)
        observed = self.audit["observed_steps_per_epoch"]
        completed = 0
        for count in observed:
            if count is None:
                break
            completed += 1
        self.audit["completed_epochs"] = completed
        self.audit["effective_batch_size"] = int(getattr(trainer, "batch_size", self.spec.batch))
        self._write()

    def complete(self) -> None:
        self.segment["status"] = "COMPLETED"
        self.segment["ended_at"] = time.time()
        self._write()

    def fail(self, error: BaseException) -> None:
        self.segment["status"] = "FAILED"
        self.segment["ended_at"] = time.time()
        self.segment["error"] = f"{type(error).__name__}: {error}"
        self._write()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temp)
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def _atomic_hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        os.link(source, temp)
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


class StableCheckpointManager:
    def __init__(self, state_dir: Path, *, allow_existing_epoch: bool = False):
        self.state_dir = state_dir
        self.allow_existing_epoch = allow_existing_epoch
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def on_model_save(self, trainer) -> None:
        source = Path(trainer.last).resolve()
        if not source.is_file():
            raise ValidationError(f"Ultralytics on_model_save did not produce last.pt: {source}")
        epoch = int(trainer.epoch) + 1
        versioned = self.state_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        if versioned.exists() and not self.allow_existing_epoch:
            raise ValidationError(f"Refusing to overwrite stable checkpoint segment: {versioned}")
        os.replace(source, versioned)
        _atomic_hardlink(versioned, self.state_dir / "last.pt")
        final_epoch = epoch >= int(trainer.epochs)
        if final_epoch:
            # Keep Ultralytics' own finalization independent: it strips optimizer state in-place.
            _atomic_copy(versioned, source)
        for old in self.state_dir.glob("checkpoint_epoch_*.pt"):
            if old != versioned:
                old.unlink()


def _validate_output_artifacts(spec: FormalTrainingSpec, trainer_dir: Path, audit: dict[str, Any]) -> FormalTrainingResult:
    paths = {
        "results_csv": trainer_dir / "results.csv",
        "args_yaml": trainer_dir / "args.yaml",
        "best_checkpoint": trainer_dir / "weights/best.pt",
        "stable_last": spec.output_dir / "training_state/last.pt",
        "audit_path": spec.output_dir / "training_execution_audit.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValidationError(f"Formal trainer missing required artifacts: {missing}")
    rows = len(pd.read_csv(paths["results_csv"]))
    if rows != spec.epochs:
        raise ValidationError(f"Training did not complete exactly {spec.epochs} epochs: results.csv rows={rows}")
    if audit["completed_epochs"] != spec.epochs:
        raise ValidationError(f"Callback audit completed_epochs mismatch: {audit['completed_epochs']} != {spec.epochs}")
    observed = audit["observed_steps_per_epoch"]
    if observed != [spec.expected_steps_per_epoch] * spec.epochs:
        raise ValidationError(
            "Observed dataloader steps differ from contract: "
            f"expected={spec.expected_steps_per_epoch}, observed={observed[:5]}..."
        )
    if audit["effective_batch_size"] != spec.batch:
        raise ValidationError(f"Effective batch changed during training: {audit['effective_batch_size']} != {spec.batch}")
    if audit["loss_finite"] is not True:
        raise ValidationError("Training loss contains NaN or Inf")
    resolved = yaml.safe_load(paths["args_yaml"].read_text(encoding="utf-8"))
    for key, expected in {
        "epochs": spec.epochs,
        "batch": spec.batch,
        "imgsz": spec.imgsz,
        "patience": 0,
        "seed": spec.seed,
        "deterministic": True,
        "cache": False,
    }.items():
        if resolved.get(key) != expected:
            raise ValidationError(f"Resolved Ultralytics arg mismatch for {key}: {resolved.get(key)!r} != {expected!r}")
    optimization_keys = (
        "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
        "warmup_momentum", "warmup_bias_lr",
    )
    augmentation_keys = (
        "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
        "perspective", "flipud", "fliplr", "bgr", "mosaic", "mixup", "cutmix",
        "copy_paste", "auto_augment", "erasing",
    )
    missing_resolved = [key for key in (*optimization_keys, *augmentation_keys) if key not in resolved]
    if missing_resolved:
        raise ValidationError(f"Resolved Ultralytics args omit required optimizer/augmentation fields: {missing_resolved}")
    resolved_args_path = spec.output_dir / "resolved_training_args.json"
    atomic_write_json(
        resolved_args_path,
        {
            "schema_version": "stage1_gapvalue240.resolved_training_args.v1",
            "args_yaml_sha256": sha256_file(paths["args_yaml"]),
            "optimization": {key: resolved[key] for key in optimization_keys},
            "augmentation": {key: resolved[key] for key in augmentation_keys},
            "resolved_args": resolved,
        },
        overwrite=True,
    )
    return FormalTrainingResult(trainer_dir=trainer_dir, resolved_args_path=resolved_args_path, **paths)


def run_formal_training(
    spec: FormalTrainingSpec,
    *,
    yolo_factory: Callable[[str], Any] | None = None,
    checkpoint_validator: Callable[[Path, bool], Any] | None = None,
) -> FormalTrainingResult:
    """Run one isolated formal training worker with no scientific CLI overrides."""

    if not spec.dataset_dir.is_dir():
        raise FileNotFoundError(spec.dataset_dir)
    for relative in ("train/no_target", "train/target_defect", "val/no_target", "val/target_defect"):
        if not (spec.dataset_dir / relative).is_dir():
            raise ValidationError(f"Staged classification directory missing: {spec.dataset_dir / relative}")
    if not spec.checkpoint.is_file():
        raise FileNotFoundError(spec.checkpoint)
    model_path = spec.resume_checkpoint or spec.checkpoint
    if checkpoint_validator is None:
        checkpoint_info = _default_checkpoint_validator(model_path, spec.resume_checkpoint is not None, spec.yolo_root)
    else:
        checkpoint_info = checkpoint_validator(model_path, spec.resume_checkpoint is not None)
    trainer_dir = spec.output_dir / "trainer"
    if trainer_dir.exists() and spec.resume_checkpoint is None:
        raise ValidationError(f"Trainer output exists; overwrite is forbidden: {trainer_dir}")
    if spec.resume_checkpoint is not None and not trainer_dir.is_dir():
        raise ValidationError(f"Resume requires the existing stable trainer directory: {trainer_dir}")
    if spec.resume_checkpoint is not None and isinstance(checkpoint_info, dict):
        train_args = checkpoint_info.get("train_args") or {}
        if train_args.get("project") is not None and train_args.get("name") is not None:
            checkpoint_dir = (Path(str(train_args["project"])) / str(train_args["name"])).resolve()
            if checkpoint_dir != trainer_dir.resolve():
                raise ValidationError(
                    f"Resume checkpoint belongs to a different output directory: {checkpoint_dir} != {trainer_dir}"
                )
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    recorder = TrainingAuditRecorder(spec, spec.output_dir / "training_execution_audit.json")
    checkpoint_manager = StableCheckpointManager(
        spec.output_dir / "training_state",
        allow_existing_epoch=spec.resume_checkpoint is not None,
    )
    factory = yolo_factory or _local_yolo(spec.yolo_root)

    model = factory(str(model_path))
    model.add_callback("on_train_start", recorder.on_train_start)
    model.add_callback("on_train_epoch_start", recorder.on_train_epoch_start)
    model.add_callback("on_train_batch_end", recorder.on_train_batch_end)
    model.add_callback("on_train_epoch_end", recorder.on_train_epoch_end)
    model.add_callback("on_model_save", checkpoint_manager.on_model_save)
    kwargs: dict[str, Any] = {
        "data": str(spec.dataset_dir),
        "epochs": spec.epochs,
        "imgsz": spec.imgsz,
        "batch": spec.batch,
        "workers": spec.workers,
        "device": spec.device,
        "project": str(spec.output_dir),
        "name": "trainer",
        "exist_ok": spec.resume_checkpoint is not None,
        "seed": spec.seed,
        "deterministic": spec.deterministic,
        "cache": spec.cache,
        "patience": spec.patience,
        "save": True,
        "save_period": -1,
        "val": True,
        "plots": False,
        "verbose": True,
        "task": "classify",
    }
    if spec.resume_checkpoint is not None:
        kwargs["resume"] = str(spec.resume_checkpoint)
    try:
        model.train(**kwargs)
        recorder.complete()
        return _validate_output_artifacts(spec, trainer_dir, recorder.audit)
    except Exception as exc:
        recorder.fail(exc)
        raise
