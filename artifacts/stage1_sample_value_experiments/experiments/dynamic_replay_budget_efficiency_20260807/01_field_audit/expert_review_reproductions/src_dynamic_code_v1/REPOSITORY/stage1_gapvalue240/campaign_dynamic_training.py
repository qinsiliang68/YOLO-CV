"""Segmented, branchable training overlay for the dynamic replay campaign."""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import random
import shutil
import sys
import time
import uuid
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from .campaign_canonical_lock import (
    CanonicalLockError,
    CanonicalTrainingLock,
    build_train_kwargs,
    load_canonical_training_lock,
    validate_formal_training_dimensions,
    validate_resolved_training_args,
)
from .errors import ValidationError
from .campaign_process_telemetry import (
    ProcessTelemetryCollector,
    ProcessTelemetryInstaller,
    ProcessTelemetrySpec,
    validate_process_telemetry_epoch,
)
from .util import atomic_write_json, sha256_file


AUDIT_SCHEMA = "stage1.dynamic_training_audit.v2"
RESUME_MODE = "SEGMENTED_DETERMINISTIC_RESET"


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        os.link(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _finite(value: Any) -> bool:
    if value is None:
        return False
    try:
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        return bool(np.isfinite(np.asarray(value, dtype=float)).all())
    except Exception:
        return False


def _loss_value(value: Any) -> float | None:
    if not _finite(value):
        return None
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return float(np.asarray(value, dtype=float).mean())


def _state_hash(value: Any) -> str:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy().tobytes()
        elif isinstance(value, tuple):
            value = repr(value).encode("utf-8")
        elif not isinstance(value, bytes):
            value = repr(value).encode("utf-8")
        return hashlib.sha256(value).hexdigest().upper()
    except Exception:
        return "UNAVAILABLE"


def _rng_digests() -> dict[str, str]:
    values = {
        "python_rng_sha256": _state_hash(random.getstate()),
        "numpy_rng_sha256": _state_hash(np.random.get_state()),
        "torch_cpu_rng_sha256": "UNAVAILABLE",
        "torch_cuda_rng_sha256": "UNAVAILABLE",
    }
    try:
        import torch

        values["torch_cpu_rng_sha256"] = _state_hash(torch.get_rng_state())
        if torch.cuda.is_available():
            values["torch_cuda_rng_sha256"] = _state_hash(torch.cuda.get_rng_state_all())
    except Exception:
        pass
    return values


def _resource_snapshot() -> dict[str, Any]:
    values: dict[str, Any] = {
        "rss_bytes": None,
        "cpu_util_pct": None,
        "child_process_count": None,
        "cuda_allocated_bytes": None,
        "cuda_reserved_bytes": None,
        "cuda_peak_allocated_bytes": None,
        "cuda_peak_reserved_bytes": None,
    }
    try:
        import psutil

        process = psutil.Process(os.getpid())
        values["rss_bytes"] = int(process.memory_info().rss)
        values["cpu_util_pct"] = float(process.cpu_percent(interval=None))
        values["child_process_count"] = len(process.children(recursive=True))
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            values.update(
                {
                    "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
                    "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
                    "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                }
            )
    except Exception:
        pass
    return values


@dataclass(frozen=True)
class SmokeFailureInjection:
    """Controlled local-only failure used to validate restart and resume behavior."""

    mode: str
    target_epoch: int
    target_batch: int | None = None
    marker_path: Path | None = None
    continue_marker_path: Path | None = None
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.mode not in {
            "OOM_AT_BATCH_START",
            "PAUSE_AT_EPOCH_START",
            "TELEMETRY_WRITE_INTERRUPTION",
        }:
            raise ValidationError(f"unsupported smoke failure injection mode: {self.mode}")
        if self.target_epoch <= 0 or self.timeout_seconds <= 0:
            raise ValidationError("failure injection epoch and timeout must be positive")
        for name in ("marker_path", "continue_marker_path"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).resolve())
        if self.mode == "OOM_AT_BATCH_START":
            if self.target_batch is None or self.target_batch <= 0:
                raise ValidationError("OOM failure injection requires a positive target batch")
            if self.marker_path is not None or self.continue_marker_path is not None:
                raise ValidationError("OOM failure injection does not accept marker paths")
        elif self.mode == "PAUSE_AT_EPOCH_START":
            if self.target_batch is not None:
                raise ValidationError("epoch pause failure injection does not accept a target batch")
            if self.marker_path is None or self.continue_marker_path is None:
                raise ValidationError("epoch pause failure injection requires reached and continue markers")
        else:
            if self.target_batch is not None:
                raise ValidationError("telemetry write failure does not accept a target batch")
            if self.marker_path is not None or self.continue_marker_path is not None:
                raise ValidationError("telemetry write failure does not accept marker paths")


@dataclass(frozen=True)
class DynamicTrainingSpec:
    run_id: str
    arm_id: str
    schedule_id: str
    selection_digest: str
    dataset_dir: Path
    checkpoint: Path
    output_dir: Path
    yolo_root: Path
    total_epochs: int
    segment_start_epoch: int
    segment_end_epoch: int
    batch: int
    imgsz: int
    seed: int
    device: str
    workers: int
    expected_steps_per_epoch: int
    retained_checkpoint_epochs: tuple[int, ...]
    execution_mode: str
    resume_checkpoint: Path | None = None
    segment_id: str | None = None
    patience: int = 0
    deterministic: bool = True
    cache: bool = False
    process_telemetry: ProcessTelemetrySpec | None = None
    active_selection_digest: str | None = None
    canonical_lock_path: Path | None = None
    canonical_lock_file_sha256: str | None = None
    smoke_canonical_overrides: tuple[str, ...] = ()
    smoke_failure_injection: SmokeFailureInjection | None = None
    runtime_health_check: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        for name in ("dataset_dir", "checkpoint", "output_dir", "yolo_root"):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        if self.resume_checkpoint is not None:
            object.__setattr__(self, "resume_checkpoint", Path(self.resume_checkpoint).resolve())
        if self.canonical_lock_path is not None:
            object.__setattr__(self, "canonical_lock_path", Path(self.canonical_lock_path).resolve())
        if not self.run_id or not self.arm_id or not self.schedule_id:
            raise ValidationError("dynamic training identity fields must be non-empty")
        if len(self.selection_digest) != 64:
            raise ValidationError("selection_digest must be a 64-character SHA-256 identity")
        if self.active_selection_digest is not None and len(self.active_selection_digest) != 64:
            raise ValidationError("active_selection_digest must be a 64-character SHA-256 identity")
        if not (1 <= self.segment_start_epoch <= self.segment_end_epoch <= self.total_epochs):
            raise ValidationError("invalid dynamic training segment bounds")
        if self.segment_start_epoch == 1 and self.resume_checkpoint is not None:
            raise ValidationError("the first segment may not use a resume checkpoint")
        if self.segment_start_epoch > 1 and self.resume_checkpoint is None:
            raise ValidationError("a later segment requires a resume checkpoint")
        if min(self.batch, self.imgsz, self.expected_steps_per_epoch) <= 0:
            raise ValidationError("batch, image size, and expected steps must be positive")
        if self.execution_mode not in {"FORMAL", "SMOKE"}:
            raise ValidationError("execution_mode must be FORMAL or SMOKE")
        if self.smoke_failure_injection is not None:
            if self.execution_mode != "SMOKE":
                raise ValidationError("failure injection is SMOKE-only and forbidden in formal training")
            injection = self.smoke_failure_injection
            if injection.target_epoch > self.total_epochs:
                raise ValidationError("failure injection epoch exceeds the configured training horizon")
            for path in (injection.marker_path, injection.continue_marker_path):
                if path is None:
                    continue
                try:
                    path.relative_to(self.output_dir)
                except ValueError as exc:
                    raise ValidationError(
                        "failure injection marker must stay inside the registered output"
                    ) from exc
        if self.patience != 0 or self.deterministic is not True or self.cache is not False:
            raise ValidationError("dynamic campaign requires patience=0, deterministic=true, cache=false")
        normalized_smoke_overrides = tuple(sorted(map(str, self.smoke_canonical_overrides)))
        object.__setattr__(self, "smoke_canonical_overrides", normalized_smoke_overrides)
        if len(set(normalized_smoke_overrides)) != len(normalized_smoke_overrides):
            raise ValidationError("duplicate smoke canonical override")
        has_lock_path = self.canonical_lock_path is not None
        has_lock_sha = bool(self.canonical_lock_file_sha256)
        if has_lock_path != has_lock_sha:
            raise ValidationError("canonical lock path and SHA must be provided together")
        if self.execution_mode == "FORMAL" and not has_lock_path:
            raise ValidationError("formal dynamic training requires a canonical lock path and SHA")
        if has_lock_path:
            assert self.canonical_lock_path is not None
            assert self.canonical_lock_file_sha256 is not None
            if not self.canonical_lock_path.is_file():
                raise ValidationError(f"canonical lock is missing: {self.canonical_lock_path}")
            actual_lock_sha = sha256_file(self.canonical_lock_path)
            if actual_lock_sha != str(self.canonical_lock_file_sha256).upper():
                raise ValidationError(
                    "canonical lock file SHA mismatch: "
                    f"{actual_lock_sha} != {str(self.canonical_lock_file_sha256).upper()}"
                )
            try:
                lock = load_canonical_training_lock(self.canonical_lock_path)
                if self.execution_mode == "FORMAL":
                    validate_formal_training_dimensions(
                        lock,
                        batch=self.batch,
                        workers=self.workers,
                        imgsz=self.imgsz,
                        epochs=self.total_epochs,
                    )
                else:
                    allowed = {"epochs", "batch", "imgsz", "workers"}
                    unsupported = sorted(set(normalized_smoke_overrides) - allowed)
                    if unsupported:
                        raise ValidationError(
                            f"unsupported smoke canonical override: {unsupported}"
                        )
                    observed = {
                        "epochs": self.total_epochs,
                        "batch": self.batch,
                        "imgsz": self.imgsz,
                        "workers": self.workers,
                    }
                    actual_overrides = {
                        key
                        for key, value in observed.items()
                        if value != lock.immutable_args[key]
                    }
                    if actual_overrides != set(normalized_smoke_overrides):
                        raise ValidationError(
                            "declared smoke overrides do not match canonical dimension drift: "
                            f"declared={sorted(normalized_smoke_overrides)}, "
                            f"actual={sorted(actual_overrides)}"
                        )
            except CanonicalLockError as exc:
                raise ValidationError(str(exc)) from exc
            expected_checkpoint_sha = str(lock.data["initial_checkpoint"]["sha256"]).upper()
            if not self.checkpoint.is_file() or sha256_file(self.checkpoint) != expected_checkpoint_sha:
                raise ValidationError("base checkpoint differs from the canonical lock")
        elif normalized_smoke_overrides:
            raise ValidationError("smoke canonical overrides require a canonical lock")
        invalid = [epoch for epoch in self.retained_checkpoint_epochs if not 1 <= epoch <= self.total_epochs]
        if invalid:
            raise ValidationError(f"retained checkpoint epochs out of range: {invalid}")
        if self.process_telemetry is not None:
            telemetry = self.process_telemetry
            expected_identity = (self.run_id, self.arm_id, self.segment_id_resolved)
            observed_identity = (telemetry.run_id, telemetry.arm_id, telemetry.segment_id)
            if observed_identity != expected_identity:
                raise ValidationError(
                    f"process telemetry identity {observed_identity} != dynamic telemetry identity {expected_identity}"
                )
            if telemetry.output_dir != self.output_dir / "process_telemetry":
                raise ValidationError("process telemetry output must be inside the registered run output")

    @property
    def segment_id_resolved(self) -> str:
        return self.segment_id or f"{self.run_id}_E{self.segment_start_epoch:03d}_{self.segment_end_epoch:03d}"

    @property
    def active_selection_digest_resolved(self) -> str:
        return self.active_selection_digest or self.selection_digest


@dataclass(frozen=True)
class DynamicTrainingResult:
    trainer_dir: Path
    results_csv: Path
    args_yaml: Path
    best_checkpoint: Path
    stable_last: Path
    audit_path: Path
    resolved_args_path: Path
    completed_epoch: int
    is_final: bool
    process_telemetry_dir: Path | None = None


def _effective_canonical_lock(
    spec: DynamicTrainingSpec,
    lock: CanonicalTrainingLock,
) -> CanonicalTrainingLock:
    """Return canonical args with only the declared smoke dimensions replaced."""

    if spec.execution_mode != "SMOKE":
        return lock
    immutable = lock.immutable_args
    observed = {
        "epochs": spec.total_epochs,
        "batch": spec.batch,
        "imgsz": spec.imgsz,
        "workers": spec.workers,
    }
    for key in spec.smoke_canonical_overrides:
        immutable[key] = observed[key]
    data = dict(lock.data)
    data["immutable_args"] = immutable
    return CanonicalTrainingLock(lock.path, data, lock.file_sha256)


@dataclass(frozen=True)
class BranchWorkspaceResult:
    child_output_dir: Path
    resume_checkpoint: Path
    lineage_path: Path


def _default_checkpoint_validator(path: Path, resume: bool, _yolo_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    _activate_local_ultralytics(_yolo_root)
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValidationError(f"checkpoint cannot be loaded: {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ValidationError(f"checkpoint root is not a mapping: {path}")
    epoch = int(checkpoint.get("epoch", -1))
    if resume and (epoch < 0 or checkpoint.get("optimizer") is None):
        raise ValidationError(f"checkpoint is not resumable: {path}")
    return {
        "epoch": epoch,
        "train_args": checkpoint.get("train_args") if isinstance(checkpoint.get("train_args"), dict) else {},
    }


def _activate_local_ultralytics(yolo_root: Path):
    yolo_root = Path(yolo_root).resolve()
    if not yolo_root.is_dir():
        raise FileNotFoundError(f"frozen YOLO root is missing: {yolo_root}")
    existing = sys.modules.get("ultralytics")
    existing_path = getattr(existing, "__file__", None) if existing is not None else None
    existing_file = Path(existing_path).resolve() if existing_path else None
    if existing_file is not None:
        try:
            existing_file.relative_to(yolo_root)
        except ValueError:
            for name in [key for key in sys.modules if key == "ultralytics" or key.startswith("ultralytics.")]:
                del sys.modules[name]
    root_text = str(yolo_root)
    sys.path[:] = [value for value in sys.path if str(Path(value).resolve()) != root_text]
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    module = importlib.import_module("ultralytics")
    module_path = Path(module.__file__).resolve()
    try:
        module_path.relative_to(yolo_root)
    except ValueError as exc:
        raise ValidationError(f"Ultralytics resolved outside frozen YOLO root: {module_path}") from exc
    return module


def _local_yolo(yolo_root: Path):
    module = _activate_local_ultralytics(yolo_root)
    return module.YOLO


def _local_campaign_trainer(yolo_root: Path):
    _activate_local_ultralytics(yolo_root)
    from ultralytics.data import ClassificationDataset
    from ultralytics.models.yolo.classify.train import ClassificationTrainer

    class TraceableClassificationDataset(ClassificationDataset):
        def __getitem__(self, index: int) -> dict[str, Any]:
            sample = super().__getitem__(index)
            sample["im_file"] = str(self.samples[index][0])
            return sample

    class RelocatableCampaignClassificationTrainer(ClassificationTrainer):
        def build_dataset(self, img_path: str, mode: str = "train", batch=None):
            return TraceableClassificationDataset(
                root=img_path,
                args=self.args,
                augment=mode == "train",
                prefix=mode,
            )

        def check_resume(self, overrides):
            super().check_resume(overrides)
            if self.resume:
                for key in ("project", "name", "data", "exist_ok"):
                    if key in overrides:
                        setattr(self.args, key, overrides[key])
                self.args.save_dir = None

    RelocatableCampaignClassificationTrainer.__name__ = "RelocatableCampaignClassificationTrainer"
    return RelocatableCampaignClassificationTrainer


class DynamicAuditRecorder:
    def __init__(self, spec: DynamicTrainingSpec, path: Path):
        self.spec = spec
        self.path = path
        self.segment: dict[str, Any]
        if path.is_file():
            if spec.resume_checkpoint is None:
                raise ValidationError(f"existing dynamic audit requires explicit resume: {path}")
            audit = json.loads(path.read_text(encoding="utf-8"))
            self._validate_previous(audit)
        else:
            if spec.segment_start_epoch != 1:
                raise ValidationError(f"later segment has no prior dynamic audit: {path}")
            audit = {
                "schema_version": AUDIT_SCHEMA,
                "run_id": spec.run_id,
                "arm_id": spec.arm_id,
                "schedule_id": spec.schedule_id,
                "selection_digest": spec.selection_digest,
                "total_epochs": spec.total_epochs,
                "completed_epochs": 0,
                "batch": spec.batch,
                "imgsz": spec.imgsz,
                "seed": spec.seed,
                "execution_mode": spec.execution_mode,
                "canonical_lock_file_sha256": (
                    str(spec.canonical_lock_file_sha256).upper()
                    if spec.canonical_lock_file_sha256
                    else None
                ),
                "smoke_canonical_overrides": list(spec.smoke_canonical_overrides),
                "resume_mode": RESUME_MODE,
                "expected_steps_by_epoch": [None] * spec.total_epochs,
                "observed_steps_by_epoch": [None] * spec.total_epochs,
                "optimizer_steps_total": 0,
                "segments": [],
                "epoch_records": [],
                "branch_lineage": None,
                "loss_finite": True,
            }
        expected = audit["expected_steps_by_epoch"]
        for epoch in range(spec.segment_start_epoch, spec.segment_end_epoch + 1):
            previous = expected[epoch - 1]
            if previous not in (None, spec.expected_steps_per_epoch):
                raise ValidationError(f"expected step contract conflicts at epoch {epoch}")
            expected[epoch - 1] = spec.expected_steps_per_epoch
        for segment in audit["segments"]:
            if segment.get("status") == "RUNNING":
                segment["status"] = "INTERRUPTED"
        self.segment = {
            "segment_id": spec.segment_id_resolved,
            "start_epoch": spec.segment_start_epoch,
            "planned_end_epoch": spec.segment_end_epoch,
            "end_epoch": None,
            "expected_steps_per_epoch": spec.expected_steps_per_epoch,
            "resume_checkpoint_sha256": sha256_file(spec.resume_checkpoint) if spec.resume_checkpoint else None,
            "active_selection_digest": spec.active_selection_digest_resolved,
            "optimizer_steps": 0,
            "started_at_unix": time.time(),
            "ended_at_unix": None,
            "status": "RUNNING",
            "resource_start": _resource_snapshot(),
            "resource_end": None,
        }
        if any(str(row.get("segment_id")) == spec.segment_id_resolved for row in audit["segments"]):
            raise ValidationError(f"duplicate segment_id: {spec.segment_id_resolved}")
        audit["segments"].append(self.segment)
        self.audit = audit
        self.current_epoch: int | None = None
        self.current_batches = 0
        self.current_optimizer_start = int(audit["optimizer_steps_total"])
        self.current_batch_starts = 0
        self.current_batch_started_at: float | None = None
        self.previous_batch_ended_at: float | None = None
        self.current_epoch_started_at: float | None = None
        self.current_epoch_started_unix: float | None = None
        self.current_train_ended_at: float | None = None
        self.current_eval_started_at: float | None = None
        self.current_eval_seconds = 0.0
        self.current_model_save_started_at: float | None = None
        self.current_train_compute_seconds = 0.0
        self.current_interbatch_wait_seconds = 0.0
        self._optimizer_patched = False
        self._write()

    def _validate_previous(self, audit: dict[str, Any]) -> None:
        if audit.get("schema_version") != AUDIT_SCHEMA:
            raise ValidationError("incompatible dynamic training audit schema")
        checks = {
            "run_id": self.spec.run_id,
            "arm_id": self.spec.arm_id,
            "schedule_id": self.spec.schedule_id,
            "selection_digest": self.spec.selection_digest,
            "total_epochs": self.spec.total_epochs,
            "batch": self.spec.batch,
            "imgsz": self.spec.imgsz,
            "seed": self.spec.seed,
            "execution_mode": self.spec.execution_mode,
            "canonical_lock_file_sha256": (
                str(self.spec.canonical_lock_file_sha256).upper()
                if self.spec.canonical_lock_file_sha256
                else None
            ),
            "smoke_canonical_overrides": list(self.spec.smoke_canonical_overrides),
        }
        mismatches = {key: (audit.get(key), value) for key, value in checks.items() if audit.get(key) != value}
        if mismatches:
            raise ValidationError(f"dynamic resume identity mismatch: {mismatches}")
        if int(audit.get("completed_epochs", -1)) != self.spec.segment_start_epoch - 1:
            raise ValidationError(
                "dynamic resume completed epoch does not match segment start: "
                f"{audit.get('completed_epochs')} != {self.spec.segment_start_epoch - 1}"
            )

    def _write(self) -> None:
        atomic_write_json(self.path, self.audit, overwrite=True)

    def on_train_start(self, trainer) -> None:
        observed_start = int(getattr(trainer, "start_epoch", 0)) + 1
        if observed_start != self.spec.segment_start_epoch:
            raise ValidationError(
                f"trainer start epoch {observed_start} != registered {self.spec.segment_start_epoch}"
            )
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
        self.current_batch_starts = 0
        self.current_batch_started_at = None
        self.current_epoch_started_at = time.perf_counter()
        self.current_epoch_started_unix = time.time()
        self.current_train_ended_at = None
        self.current_eval_started_at = None
        self.current_eval_seconds = 0.0
        self.current_model_save_started_at = None
        self.previous_batch_ended_at = self.current_epoch_started_at
        self.current_train_compute_seconds = 0.0
        self.current_interbatch_wait_seconds = 0.0
        self.current_optimizer_start = int(self.audit["optimizer_steps_total"])
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def on_train_batch_start(self, _trainer) -> None:
        now = time.perf_counter()
        if self.current_batch_started_at is not None:
            raise ValidationError("training batch start observed before previous batch end")
        if self.previous_batch_ended_at is not None:
            self.current_interbatch_wait_seconds += max(0.0, now - self.previous_batch_ended_at)
        self.current_batch_started_at = now
        self.current_batch_starts += 1

    def on_train_batch_end(self, _trainer) -> None:
        now = time.perf_counter()
        if self.current_batch_started_at is not None:
            self.current_train_compute_seconds += max(0.0, now - self.current_batch_started_at)
            self.current_batch_started_at = None
        self.previous_batch_ended_at = now
        self.current_batches += 1

    def on_train_epoch_end(self, trainer) -> None:
        epoch = int(trainer.epoch) + 1
        if epoch < self.spec.segment_start_epoch or epoch > self.spec.segment_end_epoch:
            raise ValidationError(f"observed epoch outside registered segment: {epoch}")
        if self.current_batches != self.spec.expected_steps_per_epoch:
            raise ValidationError(
                f"dataloader steps at epoch {epoch}: {self.current_batches} != {self.spec.expected_steps_per_epoch}"
            )
        if self.current_batch_starts not in {0, self.current_batches}:
            raise ValidationError(
                f"batch timing callbacks at epoch {epoch}: {self.current_batch_starts} != {self.current_batches}"
            )
        epoch_wall = (
            max(0.0, time.perf_counter() - self.current_epoch_started_at)
            if self.current_epoch_started_at is not None
            else 0.0
        )
        self.current_train_ended_at = time.perf_counter()
        train_ended_unix = time.time()
        timed_total = self.current_train_compute_seconds + self.current_interbatch_wait_seconds
        self.audit["observed_steps_by_epoch"][epoch - 1] = self.current_batches
        self.audit["completed_epochs"] = epoch
        self.segment["end_epoch"] = epoch
        finite = _finite(getattr(trainer, "tloss", None)) and _finite(getattr(trainer, "loss", None))
        self.audit["loss_finite"] = bool(self.audit["loss_finite"] and finite)
        self.audit["epoch_records"].append(
            {
                "epoch": epoch,
                "epoch_started_at_unix": self.current_epoch_started_unix,
                "train_ended_at_unix": train_ended_unix,
                "epoch_ended_at_unix": None,
                "observed_steps": self.current_batches,
                "train_loss": _loss_value(getattr(trainer, "tloss", None)),
                "optimizer_steps_total": int(self.audit["optimizer_steps_total"]),
                "optimizer_steps_epoch": int(self.audit["optimizer_steps_total"])
                - self.current_optimizer_start,
                "batch_start_count": self.current_batch_starts,
                "epoch_train_wall_seconds": epoch_wall,
                "train_compute_seconds": self.current_train_compute_seconds,
                "interbatch_wait_seconds": self.current_interbatch_wait_seconds,
                "unattributed_epoch_seconds": max(0.0, epoch_wall - timed_total),
                "eval_seconds": 0.0,
                "checkpoint_seconds": 0.0,
                "write_seconds": 0.0,
                "queue_idle_seconds": 0.0,
                "step_time_mean_seconds": (
                    timed_total / self.current_batches if self.current_batches else 0.0
                ),
                "dataloader_wait_fraction": (
                    self.current_interbatch_wait_seconds / timed_total if timed_total > 0 else 0.0
                ),
                "param_groups": [],
                **_resource_snapshot(),
                **_rng_digests(),
            }
        )
        self._write()

    def on_val_start(self, _trainer) -> None:
        self.current_eval_started_at = time.perf_counter()

    def on_val_end(self, _trainer) -> None:
        if self.current_eval_started_at is not None:
            self.current_eval_seconds += max(
                0.0, time.perf_counter() - self.current_eval_started_at
            )
            self.current_eval_started_at = None

    def on_model_save(self, _trainer) -> None:
        # Registered before DynamicCheckpointManager.  The interval through
        # on_fit_epoch_end includes the checkpoint publication callbacks.
        self.current_model_save_started_at = time.perf_counter()

    def on_fit_epoch_end(self, trainer) -> None:
        epoch = int(trainer.epoch) + 1
        records = [row for row in self.audit["epoch_records"] if int(row["epoch"]) == epoch]
        if not records:
            if (
                int(self.audit["completed_epochs"]) == self.spec.segment_end_epoch
                and epoch == self.spec.segment_end_epoch + 1
            ):
                return
            raise ValidationError(f"fit epoch has no training audit record: {epoch}")
        groups = []
        optimizer = getattr(trainer, "optimizer", None)
        for index, group in enumerate(getattr(optimizer, "param_groups", [])):
            groups.append(
                {
                    "index": index,
                    "lr": float(group.get("lr", math.nan)),
                    "momentum": float(group.get("momentum", math.nan)) if "momentum" in group else None,
                    "betas": list(group.get("betas", ())) if "betas" in group else None,
                    "weight_decay": float(group.get("weight_decay", 0.0)),
                }
            )
        records[-1]["param_groups"] = groups
        records[-1]["eval_seconds"] = self.current_eval_seconds
        records[-1]["checkpoint_seconds"] = (
            max(0.0, time.perf_counter() - self.current_model_save_started_at)
            if self.current_model_save_started_at is not None
            else 0.0
        )
        records[-1]["epoch_ended_at_unix"] = time.time()
        records[-1]["post_train_epoch_seconds"] = (
            max(0.0, time.perf_counter() - self.current_train_ended_at)
            if self.current_train_ended_at is not None
            else 0.0
        )
        scaler = getattr(trainer, "scaler", None)
        try:
            records[-1]["scaler_scale"] = float(scaler.get_scale())
        except Exception:
            records[-1]["scaler_scale"] = None
        self._write()

    def complete(self) -> None:
        if int(self.audit["completed_epochs"]) != self.spec.segment_end_epoch:
            raise ValidationError("dynamic segment returned before its registered boundary")
        self.segment["status"] = "COMPLETED"
        self.segment["ended_at_unix"] = time.time()
        self.segment["duration_seconds"] = self.segment["ended_at_unix"] - self.segment["started_at_unix"]
        self.segment["resource_end"] = _resource_snapshot()
        self._write()

    def fail(self, error: BaseException) -> None:
        self.segment["status"] = "FAILED"
        self.segment["ended_at_unix"] = time.time()
        self.segment["duration_seconds"] = self.segment["ended_at_unix"] - self.segment["started_at_unix"]
        self.segment["resource_end"] = _resource_snapshot()
        self.segment["error"] = f"{type(error).__name__}: {error}"
        self._write()


class StopAtSegmentBoundary:
    def __init__(self, end_epoch: int):
        self.end_epoch = end_epoch

    def on_train_epoch_end(self, trainer) -> None:
        epoch = int(trainer.epoch) + 1
        if epoch == self.end_epoch:
            trainer.stop = True
        elif epoch > self.end_epoch:
            raise ValidationError(f"trainer crossed segment boundary {self.end_epoch}: {epoch}")


class SmokeFailureInjector:
    def __init__(self, spec: SmokeFailureInjection):
        self.spec = spec
        self.current_epoch: int | None = None
        self.current_batch = 0

    def on_train_epoch_start(self, trainer) -> None:
        self.current_epoch = int(trainer.epoch) + 1
        self.current_batch = 0
        if (
            self.spec.mode != "PAUSE_AT_EPOCH_START"
            or self.current_epoch != self.spec.target_epoch
        ):
            return
        assert self.spec.marker_path is not None
        assert self.spec.continue_marker_path is not None
        atomic_write_json(
            self.spec.marker_path,
            {
                "schema_version": "stage1.smoke_failure_pause.v1",
                "status": "PAUSED_AT_EPOCH_START",
                "epoch": self.current_epoch,
                "pid": os.getpid(),
                "reached_at_unix": time.time(),
            },
            overwrite=True,
        )
        deadline = time.monotonic() + self.spec.timeout_seconds
        while not self.spec.continue_marker_path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"smoke failure pause timed out at epoch {self.current_epoch}"
                )
            time.sleep(0.05)

    def on_train_batch_start(self, _trainer) -> None:
        self.current_batch += 1
        if (
            self.spec.mode == "OOM_AT_BATCH_START"
            and self.current_epoch == self.spec.target_epoch
            and self.current_batch == self.spec.target_batch
        ):
            import torch

            raise torch.cuda.OutOfMemoryError(
                f"injected smoke OOM at epoch {self.current_epoch}, batch {self.current_batch}"
            )


class DynamicCheckpointManager:
    def __init__(self, state_dir: Path, *, retained_epochs: tuple[int, ...], segment_end: int, final_epoch: int):
        self.state_dir = state_dir
        self.retained = set(int(epoch) for epoch in retained_epochs) | {int(segment_end)}
        self.segment_end = int(segment_end)
        self.final_epoch = int(final_epoch)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def on_model_save(self, trainer) -> None:
        source = Path(trainer.last).resolve()
        if not source.is_file():
            raise ValidationError(f"on_model_save produced no resumable last.pt: {source}")
        epoch = int(trainer.epoch) + 1
        versioned = self.state_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        if versioned.exists():
            raise ValidationError(f"refusing to overwrite checkpoint epoch: {versioned}")
        os.replace(source, versioned)
        _atomic_hardlink(versioned, self.state_dir / "last.pt")
        if epoch >= self.final_epoch:
            _atomic_copy(versioned, source)
        atomic_write_json(
            self.state_dir / f"checkpoint_epoch_{epoch:04d}.json",
            {
                "epoch": epoch,
                "sha256": sha256_file(versioned),
                "retained": epoch in self.retained,
                "segment_boundary": epoch == self.segment_end,
                "resumable_expected": True,
            },
            overwrite=False,
        )
        for old in self.state_dir.glob("checkpoint_epoch_*.pt"):
            old_epoch = int(old.stem.rsplit("_", 1)[-1])
            if old_epoch not in self.retained and old != versioned:
                old.unlink()
                old.with_suffix(".json").unlink(missing_ok=True)


def _resolved_args(spec: DynamicTrainingSpec, trainer_dir: Path) -> Path:
    args_path = trainer_dir / "args.yaml"
    resolved = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    if not isinstance(resolved, dict):
        raise ValidationError(f"resolved dynamic training args are not a mapping: {args_path}")
    canonical_validation = None
    if spec.canonical_lock_path is not None:
        assert spec.canonical_lock_path is not None
        lock = _effective_canonical_lock(
            spec,
            load_canonical_training_lock(spec.canonical_lock_path),
        )
        model_path = spec.resume_checkpoint or spec.checkpoint
        expected_runtime = {
            "data": str(spec.dataset_dir),
            "device": spec.device,
            "exist_ok": spec.segment_start_epoch > 1,
            "model": str(model_path),
            "project": str(spec.output_dir),
            "resume": str(spec.resume_checkpoint) if spec.resume_checkpoint is not None else False,
            "save_dir": str(trainer_dir),
            "seed": spec.seed,
        }
        try:
            canonical_validation = validate_resolved_training_args(
                lock,
                resolved,
                expected_runtime=expected_runtime,
            )
            if spec.execution_mode == "SMOKE":
                canonical_validation = {
                    **canonical_validation,
                    "status": "PASS_WITH_DECLARED_SMOKE_OVERRIDES",
                    "declared_smoke_overrides": list(spec.smoke_canonical_overrides),
                }
        except CanonicalLockError as exc:
            raise ValidationError(str(exc)) from exc
    else:
        expected = {
            "epochs": spec.total_epochs,
            "batch": spec.batch,
            "imgsz": spec.imgsz,
            "patience": 0,
            "seed": spec.seed,
            "deterministic": True,
            "cache": False,
        }
        mismatches = {
            key: (resolved.get(key), value)
            for key, value in expected.items()
            if resolved.get(key) != value
        }
        if mismatches:
            raise ValidationError(f"resolved dynamic training args mismatch: {mismatches}")
    output = spec.output_dir / "resolved_training_args.json"
    atomic_write_json(
        output,
        {
            "schema_version": "stage1.dynamic_resolved_training_args.v2",
            "args_yaml_sha256": sha256_file(args_path),
            "execution_mode": spec.execution_mode,
            "canonical_lock_file_sha256": (
                str(spec.canonical_lock_file_sha256).upper()
                if spec.canonical_lock_file_sha256
                else None
            ),
            "canonical_lock_validation": canonical_validation,
            "resolved_args": resolved,
        },
        overwrite=True,
    )
    return output


def _validate_outputs(spec: DynamicTrainingSpec, recorder: DynamicAuditRecorder) -> DynamicTrainingResult:
    trainer = spec.output_dir / "trainer"
    paths = {
        "results_csv": trainer / "results.csv",
        "args_yaml": trainer / "args.yaml",
        "best_checkpoint": trainer / "weights/best.pt",
        "stable_last": spec.output_dir / "training_state/last.pt",
        "audit_path": spec.output_dir / "dynamic_training_audit.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValidationError(f"dynamic trainer missing required artifacts: {missing}")
    rows = len(pd.read_csv(paths["results_csv"]))
    if rows != spec.segment_end_epoch:
        raise ValidationError(f"results rows {rows} != completed epoch {spec.segment_end_epoch}")
    audit = recorder.audit
    if int(audit["completed_epochs"]) != spec.segment_end_epoch:
        raise ValidationError("dynamic audit completed epoch mismatch")
    for epoch in range(1, spec.segment_end_epoch + 1):
        expected = audit["expected_steps_by_epoch"][epoch - 1]
        observed = audit["observed_steps_by_epoch"][epoch - 1]
        if expected is None or expected != observed:
            raise ValidationError(f"incomplete or mismatched step audit at epoch {epoch}: {expected} != {observed}")
    if audit["loss_finite"] is not True:
        raise ValidationError("dynamic training loss contains NaN or Inf")
    for epoch in set(spec.retained_checkpoint_epochs) | {spec.segment_end_epoch}:
        if epoch <= spec.segment_end_epoch:
            checkpoint = spec.output_dir / f"training_state/checkpoint_epoch_{epoch:04d}.pt"
            if not checkpoint.is_file():
                raise ValidationError(f"required key checkpoint is missing: {checkpoint}")
    if spec.segment_end_epoch == spec.total_epochs and not (trainer / "weights/last.pt").is_file():
        raise ValidationError("final dynamic segment did not preserve trainer weights/last.pt")
    if spec.process_telemetry is not None:
        for epoch in range(spec.segment_start_epoch, spec.segment_end_epoch + 1):
            validate_process_telemetry_epoch(spec.process_telemetry, epoch)
    resolved = _resolved_args(spec, trainer)
    return DynamicTrainingResult(
        trainer_dir=trainer,
        resolved_args_path=resolved,
        completed_epoch=spec.segment_end_epoch,
        is_final=spec.segment_end_epoch == spec.total_epochs,
        process_telemetry_dir=(spec.process_telemetry.output_dir if spec.process_telemetry else None),
        **paths,
    )


def run_dynamic_training_segment(
    spec: DynamicTrainingSpec,
    *,
    yolo_factory: Callable[[str], Any] | None = None,
    campaign_trainer_class: Any | None = None,
    checkpoint_validator: Callable[[Path, bool], dict[str, Any]] | None = None,
) -> DynamicTrainingResult:
    """Run exactly one registered segment and leave a resumable boundary checkpoint."""

    if not spec.dataset_dir.is_dir():
        raise FileNotFoundError(spec.dataset_dir)
    for relative in ("train/no_target", "train/target_defect", "val/no_target", "val/target_defect"):
        if not (spec.dataset_dir / relative).is_dir():
            raise ValidationError(f"staged classification directory missing: {relative}")
    model_path = spec.resume_checkpoint or spec.checkpoint
    validator = checkpoint_validator
    info = (
        validator(model_path, spec.resume_checkpoint is not None)
        if validator is not None
        else _default_checkpoint_validator(model_path, spec.resume_checkpoint is not None, spec.yolo_root)
    )
    if spec.resume_checkpoint is not None:
        checkpoint_epoch = int(info.get("epoch", -999))
        if checkpoint_epoch + 2 != spec.segment_start_epoch:
            raise ValidationError(
                "checkpoint epoch does not match segment start: "
                f"checkpoint logical={checkpoint_epoch + 1}, segment starts={spec.segment_start_epoch}"
            )
    trainer_dir = spec.output_dir / "trainer"
    if spec.segment_start_epoch == 1 and trainer_dir.exists():
        raise ValidationError(f"first segment output already exists: {trainer_dir}")
    if spec.segment_start_epoch > 1 and not trainer_dir.is_dir():
        raise ValidationError(f"resume segment requires existing trainer workspace: {trainer_dir}")
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    after_parquet_publish = None
    if (
        spec.smoke_failure_injection is not None
        and spec.smoke_failure_injection.mode == "TELEMETRY_WRITE_INTERRUPTION"
    ):
        failure_epoch = spec.smoke_failure_injection.target_epoch

        def fail_after_parquet_publish(epoch: int) -> None:
            if int(epoch) == failure_epoch:
                raise OSError(f"injected smoke telemetry write interruption at epoch {epoch}")

        after_parquet_publish = fail_after_parquet_publish
    telemetry = (
        ProcessTelemetryCollector(
            spec.process_telemetry,
            after_parquet_publish=after_parquet_publish,
        )
        if spec.process_telemetry
        else None
    )
    if telemetry is not None and campaign_trainer_class is not None:
        raise ValidationError("custom campaign trainer cannot bypass the registered process telemetry overlay")
    recorder = DynamicAuditRecorder(spec, spec.output_dir / "dynamic_training_audit.json")
    stopper = StopAtSegmentBoundary(spec.segment_end_epoch)
    failure_injector = (
        SmokeFailureInjector(spec.smoke_failure_injection)
        if spec.smoke_failure_injection is not None
        and spec.smoke_failure_injection.mode
        in {"OOM_AT_BATCH_START", "PAUSE_AT_EPOCH_START"}
        else None
    )
    checkpoints = DynamicCheckpointManager(
        spec.output_dir / "training_state",
        retained_epochs=spec.retained_checkpoint_epochs,
        segment_end=spec.segment_end_epoch,
        final_epoch=spec.total_epochs,
    )
    factory = yolo_factory or _local_yolo(spec.yolo_root)
    trainer_class = campaign_trainer_class or _local_campaign_trainer(spec.yolo_root)
    model = None
    try:
        model = factory(str(model_path))
        if spec.runtime_health_check is not None:
            def check_runtime_health(_trainer) -> None:
                assert spec.runtime_health_check is not None
                spec.runtime_health_check()

            model.add_callback("on_train_start", check_runtime_health)
            model.add_callback("on_train_epoch_start", check_runtime_health)
            model.add_callback("on_train_batch_start", check_runtime_health)
        if telemetry is not None:
            installer = ProcessTelemetryInstaller(telemetry)
            model.add_callback("on_train_start", installer.on_train_start)
        model.add_callback("on_train_start", recorder.on_train_start)
        model.add_callback("on_train_epoch_start", recorder.on_train_epoch_start)
        if telemetry is not None:
            model.add_callback("on_train_epoch_start", telemetry.on_train_epoch_start)
        if failure_injector is not None:
            model.add_callback("on_train_epoch_start", failure_injector.on_train_epoch_start)
        model.add_callback("on_train_batch_start", recorder.on_train_batch_start)
        if failure_injector is not None:
            model.add_callback("on_train_batch_start", failure_injector.on_train_batch_start)
        model.add_callback("on_train_batch_end", recorder.on_train_batch_end)
        model.add_callback("on_train_epoch_end", recorder.on_train_epoch_end)
        model.add_callback("on_val_start", recorder.on_val_start)
        model.add_callback("on_val_end", recorder.on_val_end)
        model.add_callback("on_train_epoch_end", stopper.on_train_epoch_end)
        if telemetry is not None:
            model.add_callback("on_train_epoch_end", telemetry.on_train_epoch_end)
        model.add_callback("on_model_save", recorder.on_model_save)
        model.add_callback("on_model_save", checkpoints.on_model_save)
        model.add_callback("on_fit_epoch_end", recorder.on_fit_epoch_end)
        if telemetry is not None:
            model.add_callback("on_fit_epoch_end", telemetry.on_fit_epoch_end)
        if spec.canonical_lock_path is not None:
            assert spec.canonical_lock_path is not None
            lock = _effective_canonical_lock(
                spec,
                load_canonical_training_lock(spec.canonical_lock_path),
            )
            kwargs = build_train_kwargs(
                lock,
                {
                    "data": str(spec.dataset_dir),
                    "device": spec.device,
                    "exist_ok": spec.segment_start_epoch > 1,
                    "model": str(model_path),
                    "project": str(spec.output_dir),
                    "resume": (
                        str(spec.resume_checkpoint) if spec.resume_checkpoint is not None else False
                    ),
                    "save_dir": str(trainer_dir),
                    "seed": spec.seed,
                },
            )
            kwargs["trainer"] = trainer_class
        else:
            kwargs = {
                "data": str(spec.dataset_dir),
                "epochs": spec.total_epochs,
                "imgsz": spec.imgsz,
                "batch": spec.batch,
                "workers": spec.workers,
                "device": spec.device,
                "project": str(spec.output_dir),
                "name": "trainer",
                "exist_ok": spec.segment_start_epoch > 1,
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
                "trainer": trainer_class,
            }
            if spec.resume_checkpoint is not None:
                kwargs["resume"] = str(spec.resume_checkpoint)
        model.train(**kwargs)
        recorder.complete()
        return _validate_outputs(spec, recorder)
    except Exception as exc:
        recorder.fail(exc)
        raise
    finally:
        if model is not None:
            del model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()


def clone_branch_workspace(
    parent_output_dir: str | Path,
    child_output_dir: str | Path,
    *,
    branch_run_id: str,
    branch_arm_id: str,
    schedule_id: str,
    selection_digest: str,
    branch_epoch: int,
) -> BranchWorkspaceResult:
    """Clone metadata and hardlink immutable prefix checkpoints into one child arm."""

    parent = Path(parent_output_dir).resolve()
    child = Path(child_output_dir).resolve()
    if child.exists():
        raise FileExistsError(child)
    audit_path = parent / "dynamic_training_audit.json"
    if not audit_path.is_file():
        raise ValidationError(f"parent dynamic audit is missing: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema_version") != AUDIT_SCHEMA or int(audit.get("completed_epochs", -1)) != branch_epoch:
        raise ValidationError("parent audit is not complete at the requested branch epoch")
    parent_checkpoint = parent / f"training_state/checkpoint_epoch_{branch_epoch:04d}.pt"
    if not parent_checkpoint.is_file():
        raise ValidationError(f"parent branch checkpoint is missing: {parent_checkpoint}")
    required = [parent / "trainer/results.csv", parent / "trainer/args.yaml", parent / "trainer/weights/best.pt"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValidationError(f"parent workspace is incomplete: {missing}")
    if len(pd.read_csv(required[0])) != branch_epoch:
        raise ValidationError("parent results do not end at branch epoch")

    child.parent.mkdir(parents=True, exist_ok=True)
    staging = child.parent / f".{child.name}.{uuid.uuid4().hex}.tmpdir"
    try:
        (staging / "trainer/weights").mkdir(parents=True)
        (staging / "training_state").mkdir(parents=True)
        _atomic_copy(required[0], staging / "trainer/results.csv")
        _atomic_copy(required[1], staging / "trainer/args.yaml")
        _atomic_copy(required[2], staging / "trainer/weights/best.pt")
        for source in sorted((parent / "training_state").glob("checkpoint_epoch_*.pt")):
            epoch = int(source.stem.rsplit("_", 1)[-1])
            if epoch <= branch_epoch:
                _atomic_hardlink(source, staging / "training_state" / source.name)
                sidecar = source.with_suffix(".json")
                if sidecar.is_file():
                    _atomic_copy(sidecar, staging / "training_state" / sidecar.name)
        inherited_telemetry: list[dict[str, Any]] = []
        parent_telemetry = parent / "process_telemetry"
        if parent_telemetry.is_dir():
            (staging / "process_telemetry").mkdir(parents=True, exist_ok=True)
            for source in sorted(parent_telemetry.glob("epoch_*_process_telemetry.parquet")):
                try:
                    epoch = int(source.name.split("_", 2)[1])
                except (IndexError, ValueError) as exc:
                    raise ValidationError(f"invalid process telemetry filename: {source}") from exc
                if epoch > branch_epoch:
                    continue
                sidecar = source.with_suffix(".json")
                if not sidecar.is_file():
                    raise ValidationError(f"process telemetry sidecar is missing: {sidecar}")
                destination = staging / "process_telemetry" / source.name
                _atomic_hardlink(source, destination)
                _atomic_copy(sidecar, destination.with_suffix(".json"))
                inherited_telemetry.append(
                    {
                        "epoch": epoch,
                        "parent_parquet": str(source),
                        "sha256": sha256_file(source),
                        "link_mode": "hardlink_exact_parent_telemetry",
                    }
                )
        resume = staging / "training_state/last.pt"
        _atomic_hardlink(parent_checkpoint, resume)
        lineage = {
            "parent_run_id": audit["run_id"],
            "parent_arm_id": audit["arm_id"],
            "parent_output_dir": str(parent),
            "parent_checkpoint": str(parent_checkpoint),
            "parent_checkpoint_sha256": sha256_file(parent_checkpoint),
            "branch_epoch": branch_epoch,
            "branch_run_id": branch_run_id,
            "branch_arm_id": branch_arm_id,
            "link_mode": "hardlink_exact_checkpoint",
            "resume_rng_policy": RESUME_MODE,
            "inherited_process_telemetry": inherited_telemetry,
        }
        child_audit = dict(audit)
        child_audit.update(
            {
                "run_id": branch_run_id,
                "arm_id": branch_arm_id,
                "schedule_id": schedule_id,
                "selection_digest": selection_digest,
                "branch_lineage": lineage,
            }
        )
        atomic_write_json(staging / "dynamic_training_audit.json", child_audit, overwrite=False)
        atomic_write_json(staging / "branch_lineage.json", lineage, overwrite=False)
        os.replace(staging, child)
        return BranchWorkspaceResult(
            child_output_dir=child,
            resume_checkpoint=child / "training_state/last.pt",
            lineage_path=child / "branch_lineage.json",
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


__all__ = [
    "AUDIT_SCHEMA",
    "BranchWorkspaceResult",
    "DynamicTrainingResult",
    "SmokeFailureInjection",
    "DynamicTrainingSpec",
    "clone_branch_workspace",
    "run_dynamic_training_segment",
]
