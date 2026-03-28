from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import torch


EPOCH_METRIC_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "top1",
    "top5",
    "lr_pg0",
    "lr_pg1",
    "lr_pg2",
    "epoch_time_seconds",
    "elapsed_seconds",
    "batch_size",
    "accumulate",
    "effective_batch",
    "gpu_mem_reserved_gb",
    "gpu_mem_peak_reserved_gb",
    "gpu_mem_peak_allocated_gb",
    "fitness",
    "best_fitness_so_far",
    "best_epoch_so_far",
    "possible_early_stop",
    "stop_triggered",
    "is_best_epoch",
]


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPOCH_METRIC_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _label_loss(trainer: Any) -> dict[str, Any]:
    with torch.no_grad():
        try:
            return trainer.label_loss_items(trainer.tloss)
        except Exception:
            return {}


def register_classification_material_callbacks(model: Any) -> None:
    def on_train_start(trainer: Any) -> None:
        trainer._raw_epoch_metrics_path = Path(trainer.save_dir) / "epoch_metrics.csv"
        trainer._raw_epoch_metrics_last_epoch = 0
        trainer._raw_epoch_metrics_peak_reserved_gb = 0.0
        trainer._raw_epoch_metrics_peak_allocated_gb = 0.0

    def on_train_epoch_start(trainer: Any) -> None:
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    def on_fit_epoch_end(trainer: Any) -> None:
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        last_epoch = int(getattr(trainer, "_raw_epoch_metrics_last_epoch", 0))
        if epoch <= last_epoch:
            return

        metrics = getattr(trainer, "metrics", {}) or {}
        losses = _label_loss(trainer)
        lrs = getattr(trainer, "lr", {}) or {}

        gpu_reserved = None
        gpu_peak_reserved = None
        gpu_peak_allocated = None
        if torch.cuda.is_available():
            try:
                gpu_reserved = torch.cuda.memory_reserved() / 1e9
                gpu_peak_reserved = torch.cuda.max_memory_reserved() / 1e9
                gpu_peak_allocated = torch.cuda.max_memory_allocated() / 1e9
            except Exception:
                pass

        if gpu_peak_reserved is not None:
            trainer._raw_epoch_metrics_peak_reserved_gb = max(
                float(getattr(trainer, "_raw_epoch_metrics_peak_reserved_gb", 0.0)),
                gpu_peak_reserved,
            )
        if gpu_peak_allocated is not None:
            trainer._raw_epoch_metrics_peak_allocated_gb = max(
                float(getattr(trainer, "_raw_epoch_metrics_peak_allocated_gb", 0.0)),
                gpu_peak_allocated,
            )

        best_epoch = int(getattr(trainer.stopper, "best_epoch", 0) or 0)
        best_fitness = _to_float(getattr(trainer.stopper, "best_fitness", None))
        current_fitness = _to_float(getattr(trainer, "fitness", None))

        row = {
            "epoch": epoch,
            "train_loss": _to_float(losses.get("train/loss")),
            "val_loss": _to_float(metrics.get("val/loss")),
            "top1": _to_float(metrics.get("metrics/accuracy_top1")),
            "top5": _to_float(metrics.get("metrics/accuracy_top5")),
            "lr_pg0": _to_float(lrs.get("lr/pg0")),
            "lr_pg1": _to_float(lrs.get("lr/pg1")),
            "lr_pg2": _to_float(lrs.get("lr/pg2")),
            "epoch_time_seconds": _to_float(getattr(trainer, "epoch_time", None)),
            "elapsed_seconds": _to_float(time.time() - getattr(trainer, "train_time_start", time.time())),
            "batch_size": int(getattr(trainer, "batch_size", 0) or 0),
            "accumulate": int(getattr(trainer, "accumulate", 0) or 0),
            "effective_batch": int((getattr(trainer, "batch_size", 0) or 0) * (getattr(trainer, "accumulate", 0) or 0)),
            "gpu_mem_reserved_gb": gpu_reserved,
            "gpu_mem_peak_reserved_gb": gpu_peak_reserved,
            "gpu_mem_peak_allocated_gb": gpu_peak_allocated,
            "fitness": current_fitness,
            "best_fitness_so_far": best_fitness,
            "best_epoch_so_far": best_epoch,
            "possible_early_stop": int(bool(getattr(trainer.stopper, "possible_stop", False))),
            "stop_triggered": int(bool(getattr(trainer, "stop", False))),
            "is_best_epoch": int(best_epoch == epoch),
        }

        _append_row(Path(trainer._raw_epoch_metrics_path), row)
        trainer._raw_epoch_metrics_last_epoch = epoch

    def on_train_end(trainer: Any) -> None:
        output_path = Path(trainer.save_dir) / "training_runtime.json"
        payload = {
            "train_time_seconds": _to_float(time.time() - getattr(trainer, "train_time_start", time.time())),
            "epochs_completed": int(getattr(trainer, "epoch", 0)) + 1,
            "best_epoch": int(getattr(trainer.stopper, "best_epoch", 0) or 0),
            "best_fitness": _to_float(getattr(trainer.stopper, "best_fitness", None)),
            "peak_gpu_mem_reserved_gb": _to_float(getattr(trainer, "_raw_epoch_metrics_peak_reserved_gb", None)),
            "peak_gpu_mem_allocated_gb": _to_float(getattr(trainer, "_raw_epoch_metrics_peak_allocated_gb", None)),
            "validator_speed_ms": getattr(getattr(trainer, "validator", None), "speed", {}) or {},
        }
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)
