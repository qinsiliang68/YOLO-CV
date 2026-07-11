from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from stage1_gapvalue240.contract import load_contract
from stage1_gapvalue240.formal_trainer import FormalTrainingSpec, run_formal_training


ROOT = Path(__file__).resolve().parents[2]


class FakeYOLO:
    constructed_with: list[str] = []
    train_calls: list[dict] = []

    def __init__(self, model_path: str):
        self.constructed_with.append(model_path)
        self.callbacks: dict[str, list] = {}
        self.trainer = None

    def add_callback(self, event: str, callback) -> None:
        self.callbacks.setdefault(event, []).append(callback)

    def _emit(self, event: str, trainer) -> None:
        for callback in self.callbacks.get(event, []):
            callback(trainer)

    def train(self, **kwargs):
        self.train_calls.append(kwargs)
        save_dir = Path(kwargs["project"]) / kwargs["name"]
        weights = save_dir / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        start_epoch = 1 if kwargs.get("resume") else 0
        trainer = SimpleNamespace(
            start_epoch=start_epoch,
            epoch=start_epoch,
            epochs=kwargs["epochs"],
            batch_size=kwargs["batch"],
            save_dir=save_dir,
            last=weights / "last.pt",
            tloss=np.asarray([0.25]),
            loss=np.asarray([0.25]),
        )
        trainer.optimizer_step = lambda: None
        self.trainer = trainer
        self._emit("on_train_start", trainer)
        results = []
        if start_epoch:
            results.append({"epoch": 1, "train/loss": 0.3})
        for epoch in range(start_epoch, kwargs["epochs"]):
            trainer.epoch = epoch
            self._emit("on_train_epoch_start", trainer)
            for _ in range(3):
                trainer.optimizer_step()
                self._emit("on_train_batch_end", trainer)
            self._emit("on_train_epoch_end", trainer)
            trainer.last.write_bytes(f"checkpoint-{epoch + 1}".encode())
            self._emit("on_model_save", trainer)
            results.append({"epoch": epoch + 1, "train/loss": 0.25})
        pd.DataFrame(results).to_csv(save_dir / "results.csv", index=False)
        resolved = {
            "optimizer": "auto", "lr0": 0.01, "lrf": 0.01, "momentum": 0.937,
            "weight_decay": 0.0005, "warmup_epochs": 3.0, "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
            "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0,
            "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5, "bgr": 0.0,
            "mosaic": 1.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
            "auto_augment": "randaugment", "erasing": 0.4, "crop_fraction": 1.0,
            **kwargs,
        }
        (save_dir / "args.yaml").write_text(yaml.safe_dump(resolved), encoding="utf-8")
        (weights / "best.pt").write_bytes(b"best")
        return None


def _test_spec(tmp_path: Path, *, resume_checkpoint: Path | None = None) -> FormalTrainingSpec:
    checkpoint = tmp_path / "yolo11l-cls.pt"
    checkpoint.write_bytes(b"base")
    dataset = tmp_path / "dataset"
    for path in (dataset / "train/no_target", dataset / "train/target_defect", dataset / "val/no_target", dataset / "val/target_defect"):
        path.mkdir(parents=True, exist_ok=True)
    return FormalTrainingSpec(
        dataset_dir=dataset,
        checkpoint=checkpoint,
        output_dir=tmp_path / "output",
        yolo_root=ROOT / "YOLOv11",
        epochs=2,
        batch=4,
        imgsz=32,
        seed=123,
        device="0",
        workers=0,
        expected_steps_per_epoch=3,
        resume_checkpoint=resume_checkpoint,
        segment_id="segment-test",
    )


def test_contract_factory_locks_formal_scientific_parameters(tmp_path):
    contract = load_contract(ROOT / "configs/stage1_gapvalue240/EXPERIMENT_CONTRACT.yaml")
    checkpoint = tmp_path / "exact-checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    spec = FormalTrainingSpec.from_contract(
        contract,
        dataset_dir=tmp_path / "dataset",
        checkpoint=checkpoint,
        output_dir=tmp_path / "output",
        yolo_root=ROOT / "YOLOv11",
        training_seed=999,
        budget=600,
        device="0",
        workers=8,
    )

    assert spec.checkpoint == checkpoint.resolve()
    assert spec.epochs == 200
    assert spec.batch == 128
    assert spec.imgsz == 224
    assert spec.seed == 999
    assert spec.expected_steps_per_epoch == 943
    assert spec.patience == 0
    assert spec.deterministic is True
    assert spec.cache is False


def test_formal_training_passes_locked_args_and_writes_exact_audit(tmp_path):
    FakeYOLO.constructed_with.clear()
    FakeYOLO.train_calls.clear()
    spec = _test_spec(tmp_path)

    result = run_formal_training(
        spec,
        yolo_factory=FakeYOLO,
        checkpoint_validator=lambda _path, _resume: None,
    )

    call = FakeYOLO.train_calls[-1]
    assert FakeYOLO.constructed_with[-1] == str(spec.checkpoint)
    assert call["epochs"] == 2
    assert call["batch"] == 4
    assert call["imgsz"] == 32
    assert call["patience"] == 0
    assert call["deterministic"] is True
    assert call["cache"] is False
    assert call["seed"] == 123
    assert call["project"] == str(spec.output_dir)
    assert call["name"] == "trainer"
    assert "resume" not in call

    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert set(audit) == {
        "schema_version",
        "expected_epochs",
        "completed_epochs",
        "expected_steps_per_epoch",
        "observed_steps_per_epoch",
        "optimizer_steps_total",
        "effective_batch_size",
        "configured_args",
        "loss_finite",
        "resume_mode",
        "resume_count",
        "resume_segments",
    }
    assert audit["completed_epochs"] == 2
    assert audit["observed_steps_per_epoch"] == [3, 3]
    assert audit["optimizer_steps_total"] == 6
    assert audit["effective_batch_size"] == 4
    assert audit["loss_finite"] is True
    assert audit["resume_mode"] == "native_approximate"
    assert audit["resume_count"] == 0
    assert audit["configured_args"] == {
        "epochs": 2,
        "batch": 4,
        "imgsz": 32,
        "patience": 0,
        "seed": 123,
        "deterministic": True,
        "cache": False,
        "model": str(spec.checkpoint),
    }
    assert result.results_csv == spec.output_dir / "trainer/results.csv"
    assert result.args_yaml == spec.output_dir / "trainer/args.yaml"
    assert result.stable_last == spec.output_dir / "training_state/last.pt"
    assert result.stable_last.read_bytes() == b"checkpoint-2"
    assert (spec.output_dir / "trainer/weights/last.pt").exists()
    resolved = json.loads(result.resolved_args_path.read_text(encoding="utf-8"))
    assert resolved["optimization"]["optimizer"] == "auto"
    assert resolved["optimization"]["lr0"] == 0.01
    assert resolved["augmentation"]["auto_augment"] == "randaugment"


def test_resume_uses_exact_checkpoint_and_appends_segment_metadata(tmp_path):
    FakeYOLO.constructed_with.clear()
    FakeYOLO.train_calls.clear()
    initial = _test_spec(tmp_path)
    first = run_formal_training(initial, yolo_factory=FakeYOLO, checkpoint_validator=lambda _p, _r: None)
    resumed = _test_spec(tmp_path, resume_checkpoint=first.stable_last)

    second = run_formal_training(resumed, yolo_factory=FakeYOLO, checkpoint_validator=lambda _p, _r: None)

    call = FakeYOLO.train_calls[-1]
    assert FakeYOLO.constructed_with[-1] == str(first.stable_last)
    assert call["resume"] == str(first.stable_last)
    audit = json.loads(second.audit_path.read_text(encoding="utf-8"))
    assert audit["resume_count"] == 1
    assert len(audit["resume_segments"]) == 2
    assert audit["resume_segments"][-1]["resumed"] is True
    assert audit["resume_segments"][-1]["resume_checkpoint_sha256"]


def test_formal_worker_direct_cli_exposes_only_runtime_and_frozen_run_inputs():
    script = ROOT / "scripts/stage1_gapvalue240/formal_train_worker.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout
    assert "--staging-root" in result.stdout
    assert "--resume-checkpoint" in result.stdout
    assert "--epochs" not in result.stdout
    assert "--batch" not in result.stdout
    assert "--imgsz" not in result.stdout
