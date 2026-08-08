from __future__ import annotations

import json
import importlib
import os
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from stage1_gapvalue240.campaign_dynamic_training import (
    DynamicTrainingSpec,
    SmokeFailureInjection,
    _local_campaign_trainer,
    clone_branch_workspace,
    run_dynamic_training_segment,
)
from stage1_gapvalue240.campaign_process_telemetry import ProcessTelemetrySpec
from stage1_gapvalue240.errors import ValidationError
from stage1_gapvalue240.util import sha256_file


ROOT = Path(__file__).resolve().parents[2]


def test_local_campaign_trainer_and_dataset_are_spawn_picklable() -> None:
    trainer_class = _local_campaign_trainer(ROOT / "YOLOv11")
    overlay = importlib.import_module("stage1_gapvalue240.campaign_ultralytics_overlay")

    assert pickle.loads(pickle.dumps(trainer_class)) is trainer_class
    assert pickle.loads(pickle.dumps(overlay.TraceableClassificationDataset)) is (
        overlay.TraceableClassificationDataset
    )
    assert "<locals>" not in trainer_class.__qualname__
    assert "<locals>" not in overlay.TraceableClassificationDataset.__qualname__


class FakeSegmentYOLO:
    train_calls: list[dict] = []
    resume_start_epoch = 0
    steps_per_epoch = 3
    emit_final_eval_callback = False

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.callbacks: dict[str, list] = {}

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
        (weights / "best.pt").write_bytes(b"best-prefix")
        start_epoch = self.resume_start_epoch if kwargs.get("resume") else 0
        trainer = SimpleNamespace(
            start_epoch=start_epoch,
            epoch=start_epoch,
            epochs=kwargs["epochs"],
            batch_size=kwargs["batch"],
            save_dir=save_dir,
            last=weights / "last.pt",
            best=weights / "best.pt",
            tloss=np.asarray([0.25]),
            loss=np.asarray([0.25]),
            stop=False,
            optimizer=SimpleNamespace(
                param_groups=[{"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005}]
            ),
            scaler=SimpleNamespace(get_scale=lambda: 1024.0),
        )
        trainer.optimizer_step = lambda: None
        self._emit("on_train_start", trainer)
        results_path = save_dir / "results.csv"
        results = pd.read_csv(results_path).to_dict("records") if results_path.is_file() else []
        for epoch in range(start_epoch, kwargs["epochs"]):
            trainer.epoch = epoch
            trainer.stop = False
            self._emit("on_train_epoch_start", trainer)
            for _ in range(self.steps_per_epoch):
                self._emit("on_train_batch_start", trainer)
                trainer.optimizer_step()
                self._emit("on_train_batch_end", trainer)
            self._emit("on_train_epoch_end", trainer)
            trainer.last.write_bytes(f"checkpoint-{epoch + 1}".encode())
            self._emit("on_model_save", trainer)
            self._emit("on_fit_epoch_end", trainer)
            results.append({"epoch": epoch + 1, "train/loss": 0.25})
            if trainer.stop:
                break
        pd.DataFrame(results).to_csv(results_path, index=False)
        if self.emit_final_eval_callback:
            trainer.epoch += 1
            self._emit("on_fit_epoch_end", trainer)
            trainer.epoch -= 1
        resolved = {
            "optimizer": "auto",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "bgr": 0.0,
            "mosaic": 1.0,
            "mixup": 0.0,
            "cutmix": 0.0,
            "copy_paste": 0.0,
            "auto_augment": "randaugment",
            "erasing": 0.4,
            "model": self.model_path,
            "save_dir": str(save_dir),
            **kwargs,
        }
        resolved.pop("trainer", None)
        (save_dir / "args.yaml").write_text(yaml.safe_dump(resolved), encoding="utf-8")


def _spec(
    tmp_path: Path,
    *,
    start: int,
    end: int,
    steps: int,
    output: Path | None = None,
    resume: Path | None = None,
    arm: str = "T_DYNAMIC_DECAY",
) -> DynamicTrainingSpec:
    checkpoint = tmp_path / "yolo11l-cls.pt"
    checkpoint.write_bytes(b"base")
    dataset = tmp_path / f"dataset_{start}_{end}"
    for relative in (
        "train/no_target",
        "train/target_defect",
        "val/no_target",
        "val/target_defect",
    ):
        (dataset / relative).mkdir(parents=True, exist_ok=True)
    return DynamicTrainingSpec(
        run_id="DRBE_S001_TEST",
        arm_id=arm,
        schedule_id=f"SCHEDULE_{arm}",
        selection_digest="A" * 64,
        dataset_dir=dataset,
        checkpoint=checkpoint,
        output_dir=output or (tmp_path / "output"),
        yolo_root=ROOT / "YOLOv11",
        total_epochs=5,
        segment_start_epoch=start,
        segment_end_epoch=end,
        batch=4,
        imgsz=32,
        seed=123,
        device="0",
        workers=0,
        expected_steps_per_epoch=steps,
        retained_checkpoint_epochs=(2, 3, 5),
        resume_checkpoint=resume,
        segment_id=f"segment_{start}_{end}",
        execution_mode="SMOKE",
    )


def _validator(epoch: int):
    return lambda _path, _resume: {"epoch": epoch, "train_args": {}}


def test_stops_at_segment_boundary_and_resumes_with_variable_steps(tmp_path: Path) -> None:
    FakeSegmentYOLO.train_calls.clear()
    FakeSegmentYOLO.resume_start_epoch = 0
    FakeSegmentYOLO.steps_per_epoch = 3
    first_spec = _spec(tmp_path, start=1, end=3, steps=3)

    first = run_dynamic_training_segment(
        first_spec,
        yolo_factory=FakeSegmentYOLO,
        campaign_trainer_class=object,
        checkpoint_validator=_validator(-1),
    )

    assert first.completed_epoch == 3
    assert first.is_final is False
    assert first.stable_last.read_bytes() == b"checkpoint-3"
    assert (first_spec.output_dir / "training_state/checkpoint_epoch_0002.pt").is_file()
    assert (first_spec.output_dir / "training_state/checkpoint_epoch_0003.pt").is_file()
    assert not (first_spec.output_dir / "training_state/checkpoint_epoch_0001.pt").exists()
    audit = json.loads(first.audit_path.read_text(encoding="utf-8"))
    assert audit["completed_epochs"] == 3
    assert audit["expected_steps_by_epoch"] == [3, 3, 3, None, None]
    assert audit["observed_steps_by_epoch"] == [3, 3, 3, None, None]
    assert audit["segments"][-1]["status"] == "COMPLETED"
    assert audit["segments"][-1]["end_epoch"] == 3
    assert audit["segments"][-1]["active_selection_digest"] == "A" * 64
    assert [row["optimizer_steps_epoch"] for row in audit["epoch_records"]] == [3, 3, 3]
    assert all("rss_bytes" in row for row in audit["epoch_records"])
    assert all(row["batch_start_count"] == 3 for row in audit["epoch_records"])
    assert all(row["train_compute_seconds"] >= 0 for row in audit["epoch_records"])
    assert all(row["interbatch_wait_seconds"] >= 0 for row in audit["epoch_records"])
    assert all(row["step_time_mean_seconds"] >= 0 for row in audit["epoch_records"])
    assert audit["segments"][-1]["duration_seconds"] >= 0

    FakeSegmentYOLO.resume_start_epoch = 3
    FakeSegmentYOLO.steps_per_epoch = 2
    second_spec = _spec(
        tmp_path,
        start=4,
        end=5,
        steps=2,
        output=first_spec.output_dir,
        resume=first.stable_last,
    )
    second_spec = DynamicTrainingSpec(
        **{**second_spec.__dict__, "active_selection_digest": "C" * 64}
    )
    second = run_dynamic_training_segment(
        second_spec,
        yolo_factory=FakeSegmentYOLO,
        campaign_trainer_class=object,
        checkpoint_validator=_validator(2),
    )

    assert second.completed_epoch == 5
    assert second.is_final is True
    audit = json.loads(second.audit_path.read_text(encoding="utf-8"))
    assert audit["expected_steps_by_epoch"] == [3, 3, 3, 2, 2]
    assert audit["observed_steps_by_epoch"] == [3, 3, 3, 2, 2]
    assert audit["optimizer_steps_total"] == 13
    assert len(audit["segments"]) == 2
    assert [segment["active_selection_digest"] for segment in audit["segments"]] == ["A" * 64, "C" * 64]
    assert len(pd.read_csv(second.results_csv)) == 5
    assert (second_spec.output_dir / "training_state/checkpoint_epoch_0005.pt").is_file()
    assert (second_spec.output_dir / "trainer/weights/last.pt").is_file()


def test_resume_epoch_must_match_registered_segment_start(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        start=4,
        end=5,
        steps=2,
        resume=tmp_path / "wrong.pt",
    )
    spec.resume_checkpoint.write_bytes(b"wrong")
    spec.output_dir.mkdir(parents=True)
    (spec.output_dir / "dynamic_training_audit.json").write_text(
        json.dumps(
            {
                "schema_version": "stage1.dynamic_training_audit.v1",
                "run_id": spec.run_id,
                "arm_id": spec.arm_id,
                "schedule_id": spec.schedule_id,
                "selection_digest": spec.selection_digest,
                "total_epochs": 5,
                "completed_epochs": 3,
                "batch": 4,
                "imgsz": 32,
                "seed": 123,
                "expected_steps_by_epoch": [3, 3, 3, None, None],
                "observed_steps_by_epoch": [3, 3, 3, None, None],
                "optimizer_steps_total": 9,
                "segments": [],
                "epoch_records": [],
                "branch_lineage": None,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="checkpoint epoch does not match segment start"):
        run_dynamic_training_segment(
            spec,
            yolo_factory=FakeSegmentYOLO,
            campaign_trainer_class=object,
            checkpoint_validator=_validator(1),
        )


def test_clones_shared_prefix_as_hardlinked_branch_lineage(tmp_path: Path) -> None:
    FakeSegmentYOLO.resume_start_epoch = 0
    FakeSegmentYOLO.steps_per_epoch = 3
    parent_spec = _spec(tmp_path, start=1, end=3, steps=3, arm="T_SHARED_PREFIX")
    parent = run_dynamic_training_segment(
        parent_spec,
        yolo_factory=FakeSegmentYOLO,
        campaign_trainer_class=object,
        checkpoint_validator=_validator(-1),
    )
    telemetry = parent_spec.output_dir / "process_telemetry"
    telemetry.mkdir(parents=True)
    (telemetry / "epoch_0003_process_telemetry.parquet").write_bytes(b"telemetry")
    (telemetry / "epoch_0003_process_telemetry.json").write_text(
        '{"status":"COMPLETE","epoch":3}', encoding="utf-8"
    )
    source_bytes = parent.stable_last.read_bytes()
    child = tmp_path / "child"

    result = clone_branch_workspace(
        parent_spec.output_dir,
        child,
        branch_run_id="DRBE_S001_T_DYNAMIC_DECAY",
        branch_arm_id="T_DYNAMIC_DECAY",
        schedule_id="SCHEDULE_T_DYNAMIC_DECAY",
        selection_digest="B" * 64,
        branch_epoch=3,
    )

    assert result.resume_checkpoint == child / "training_state/last.pt"
    assert os.path.samefile(result.resume_checkpoint, parent.stable_last)
    assert result.resume_checkpoint.read_bytes() == source_bytes
    assert os.path.samefile(
        child / "process_telemetry/epoch_0003_process_telemetry.parquet",
        telemetry / "epoch_0003_process_telemetry.parquet",
    )
    assert len(pd.read_csv(child / "trainer/results.csv")) == 3
    child_audit = json.loads((child / "dynamic_training_audit.json").read_text(encoding="utf-8"))
    assert child_audit["run_id"] == "DRBE_S001_T_DYNAMIC_DECAY"
    assert child_audit["arm_id"] == "T_DYNAMIC_DECAY"
    assert child_audit["branch_lineage"]["parent_checkpoint_sha256"]
    assert child_audit["branch_lineage"]["branch_epoch"] == 3
    assert parent.stable_last.read_bytes() == source_bytes

    with pytest.raises(FileExistsError):
        clone_branch_workspace(
            parent_spec.output_dir,
            child,
            branch_run_id="duplicate",
            branch_arm_id="T_DYNAMIC_DECAY",
            schedule_id="SCHEDULE_T_DYNAMIC_DECAY",
            selection_digest="B" * 64,
            branch_epoch=3,
        )


def test_dynamic_training_rejects_mismatched_process_telemetry_identity(tmp_path: Path) -> None:
    spec = _spec(tmp_path, start=1, end=3, steps=3)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    normal = manifests / "normal.csv"
    defect = manifests / "defect.csv"
    replay = manifests / "replay.csv"
    monitor = manifests / "monitor.csv"
    pd.DataFrame([{"canonical_image_relpath": "n.jpg", "Filename": "n.jpg"}]).to_csv(normal, index=False)
    pd.DataFrame([{"canonical_image_relpath": "d.jpg", "Filename": "d.jpg"}]).to_csv(defect, index=False)
    pd.DataFrame(columns=["staged_filename", "sample_id", "y_true", "replay_role"]).to_csv(
        replay, index=False
    )
    pd.DataFrame([{"sample_id": "n.jpg", "monitor_group": "A02_NORMAL"}]).to_csv(monitor, index=False)
    telemetry = ProcessTelemetrySpec(
        run_id="DIFFERENT_RUN",
        arm_id=spec.arm_id,
        segment_id=spec.segment_id_resolved,
        output_dir=spec.output_dir / "process_telemetry",
        base_normal_manifest=normal,
        base_defect_manifest=defect,
        replay_identity_manifest=replay,
        monitor_manifest=monitor,
        expected_epoch_samples=2,
        expected_replay_samples=0,
    )
    with pytest.raises(ValidationError, match="telemetry identity"):
        DynamicTrainingSpec(**{**spec.__dict__, "process_telemetry": telemetry})


def test_dynamic_audit_ignores_ultralytics_final_eval_fit_callback(tmp_path: Path) -> None:
    FakeSegmentYOLO.resume_start_epoch = 0
    FakeSegmentYOLO.steps_per_epoch = 3
    FakeSegmentYOLO.emit_final_eval_callback = True
    try:
        result = run_dynamic_training_segment(
            _spec(tmp_path, start=1, end=3, steps=3),
            yolo_factory=FakeSegmentYOLO,
            campaign_trainer_class=object,
            checkpoint_validator=_validator(-1),
        )
    finally:
        FakeSegmentYOLO.emit_final_eval_callback = False
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert len(audit["epoch_records"]) == 3


def test_formal_segment_uses_and_audits_the_full_canonical_lock(tmp_path: Path) -> None:
    lock = ROOT / "configs/stage1_gapvalue240/CANONICAL_TRAINING_LOCK_v1.json"
    FakeSegmentYOLO.train_calls.clear()
    FakeSegmentYOLO.resume_start_epoch = 0
    FakeSegmentYOLO.steps_per_epoch = 3
    base = _spec(tmp_path, start=1, end=3, steps=3)
    spec = DynamicTrainingSpec(
        **{
            **base.__dict__,
            "total_epochs": 200,
            "batch": 128,
            "imgsz": 224,
            "workers": 4,
            "checkpoint": ROOT / "yolo11l-cls.pt",
            "retained_checkpoint_epochs": (3, 120, 140, 150, 160, 180, 200),
            "execution_mode": "FORMAL",
            "canonical_lock_path": lock,
            "canonical_lock_file_sha256": sha256_file(lock),
        }
    )

    result = run_dynamic_training_segment(
        spec,
        yolo_factory=FakeSegmentYOLO,
        campaign_trainer_class=object,
        checkpoint_validator=_validator(-1),
    )

    kwargs = FakeSegmentYOLO.train_calls[-1]
    assert kwargs["batch"] == 128
    assert kwargs["workers"] == 4
    assert kwargs["optimizer"] == "auto"
    assert kwargs["lr0"] == 0.01
    assert kwargs["auto_augment"] == "randaugment"
    assert kwargs["resume"] is False
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["execution_mode"] == "FORMAL"
    assert audit["canonical_lock_file_sha256"] == sha256_file(lock)
    resolved = json.loads(result.resolved_args_path.read_text(encoding="utf-8"))
    assert resolved["canonical_lock_validation"]["status"] == "PASS"
    assert resolved["canonical_lock_file_sha256"] == sha256_file(lock)


def test_formal_segment_rejects_lock_hash_or_dimension_drift(tmp_path: Path) -> None:
    lock = ROOT / "configs/stage1_gapvalue240/CANONICAL_TRAINING_LOCK_v1.json"
    base = _spec(tmp_path, start=1, end=3, steps=3)
    values = {
        **base.__dict__,
        "total_epochs": 200,
        "batch": 128,
        "imgsz": 224,
        "workers": 4,
        "checkpoint": ROOT / "yolo11l-cls.pt",
        "retained_checkpoint_epochs": (3, 120, 140, 150, 160, 180, 200),
        "execution_mode": "FORMAL",
        "canonical_lock_path": lock,
        "canonical_lock_file_sha256": "0" * 64,
    }
    with pytest.raises(ValidationError, match="canonical lock file SHA"):
        DynamicTrainingSpec(**values)

    values["canonical_lock_file_sha256"] = sha256_file(lock)
    values["workers"] = 8
    with pytest.raises(ValidationError, match="workers"):
        DynamicTrainingSpec(**values)


def test_smoke_can_shorten_dimensions_but_inherits_every_other_canonical_field(
    tmp_path: Path,
) -> None:
    lock = ROOT / "configs/stage1_gapvalue240/CANONICAL_TRAINING_LOCK_v1.json"
    FakeSegmentYOLO.train_calls.clear()
    FakeSegmentYOLO.resume_start_epoch = 0
    FakeSegmentYOLO.steps_per_epoch = 3
    base = _spec(tmp_path, start=1, end=3, steps=3)
    spec = DynamicTrainingSpec(
        **{
            **base.__dict__,
            "checkpoint": ROOT / "yolo11l-cls.pt",
            "canonical_lock_path": lock,
            "canonical_lock_file_sha256": sha256_file(lock),
            "smoke_canonical_overrides": ("epochs", "batch", "imgsz", "workers"),
        }
    )

    result = run_dynamic_training_segment(
        spec,
        yolo_factory=FakeSegmentYOLO,
        campaign_trainer_class=object,
        checkpoint_validator=_validator(-1),
    )

    kwargs = FakeSegmentYOLO.train_calls[-1]
    assert (kwargs["epochs"], kwargs["batch"], kwargs["imgsz"], kwargs["workers"]) == (
        5,
        4,
        32,
        0,
    )
    assert kwargs["optimizer"] == "auto"
    assert kwargs["lr0"] == 0.01
    assert kwargs["warmup_epochs"] == 3.0
    assert kwargs["auto_augment"] == "randaugment"
    assert kwargs["amp"] is True
    resolved = json.loads(result.resolved_args_path.read_text(encoding="utf-8"))
    assert resolved["canonical_lock_validation"]["status"] == "PASS_WITH_DECLARED_SMOKE_OVERRIDES"
    assert resolved["canonical_lock_validation"]["declared_smoke_overrides"] == [
        "batch",
        "epochs",
        "imgsz",
        "workers",
    ]


def test_smoke_canonical_overrides_must_exactly_match_dimensions(tmp_path: Path) -> None:
    lock = ROOT / "configs/stage1_gapvalue240/CANONICAL_TRAINING_LOCK_v1.json"
    base = _spec(tmp_path, start=1, end=3, steps=3)
    values = {
        **base.__dict__,
        "checkpoint": ROOT / "yolo11l-cls.pt",
        "canonical_lock_path": lock,
        "canonical_lock_file_sha256": sha256_file(lock),
        "smoke_canonical_overrides": ("epochs", "batch", "imgsz"),
    }
    with pytest.raises(ValidationError, match="declared smoke overrides"):
        DynamicTrainingSpec(**values)

    values["smoke_canonical_overrides"] = ("epochs", "batch", "imgsz", "workers", "lr0")
    with pytest.raises(ValidationError, match="unsupported smoke canonical override"):
        DynamicTrainingSpec(**values)


def test_smoke_oom_injection_fails_before_first_epoch_is_committed(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    FakeSegmentYOLO.resume_start_epoch = 0
    FakeSegmentYOLO.steps_per_epoch = 3
    base = _spec(tmp_path, start=1, end=3, steps=3)
    spec = DynamicTrainingSpec(
        **{
            **base.__dict__,
            "smoke_failure_injection": SmokeFailureInjection(
                mode="OOM_AT_BATCH_START",
                target_epoch=1,
                target_batch=2,
            ),
        }
    )

    with pytest.raises(torch.cuda.OutOfMemoryError, match="injected smoke OOM"):
        run_dynamic_training_segment(
            spec,
            yolo_factory=FakeSegmentYOLO,
            campaign_trainer_class=object,
            checkpoint_validator=_validator(-1),
        )

    audit = json.loads((spec.output_dir / "dynamic_training_audit.json").read_text(encoding="utf-8"))
    assert audit["completed_epochs"] == 0
    assert audit["segments"][-1]["status"] == "FAILED"
    assert not (spec.output_dir / "training_state/last.pt").exists()


def test_smoke_pause_injection_publishes_reached_marker(tmp_path: Path) -> None:
    FakeSegmentYOLO.resume_start_epoch = 0
    FakeSegmentYOLO.steps_per_epoch = 3
    base = _spec(tmp_path, start=1, end=1, steps=3)
    reached = base.output_dir / "failure_injection/reached.json"
    release = base.output_dir / "failure_injection/continue.marker"
    release.parent.mkdir(parents=True)
    release.write_text("continue", encoding="utf-8")
    spec = DynamicTrainingSpec(
        **{
            **base.__dict__,
            "smoke_failure_injection": SmokeFailureInjection(
                mode="PAUSE_AT_EPOCH_START",
                target_epoch=1,
                marker_path=reached,
                continue_marker_path=release,
                timeout_seconds=1.0,
            ),
        }
    )

    run_dynamic_training_segment(
        spec,
        yolo_factory=FakeSegmentYOLO,
        campaign_trainer_class=object,
        checkpoint_validator=_validator(-1),
    )

    marker = json.loads(reached.read_text(encoding="utf-8"))
    assert marker["status"] == "PAUSED_AT_EPOCH_START"
    assert marker["epoch"] == 1


def test_failure_injection_is_smoke_only_and_confined_to_output(tmp_path: Path) -> None:
    base = _spec(tmp_path, start=1, end=3, steps=3)
    with pytest.raises(ValidationError, match="SMOKE-only"):
        DynamicTrainingSpec(
            **{
                **base.__dict__,
                "execution_mode": "FORMAL",
                "canonical_lock_path": ROOT / "configs/stage1_gapvalue240/CANONICAL_TRAINING_LOCK_v1.json",
                "canonical_lock_file_sha256": sha256_file(
                    ROOT / "configs/stage1_gapvalue240/CANONICAL_TRAINING_LOCK_v1.json"
                ),
                "checkpoint": ROOT / "yolo11l-cls.pt",
                "total_epochs": 200,
                "batch": 128,
                "imgsz": 224,
                "workers": 4,
                "retained_checkpoint_epochs": (3, 120, 140, 150, 160, 180, 200),
                "smoke_failure_injection": SmokeFailureInjection(
                    mode="OOM_AT_BATCH_START", target_epoch=1, target_batch=1
                ),
            }
        )


def test_runtime_health_guard_is_checked_before_training_batches(tmp_path: Path) -> None:
    FakeSegmentYOLO.resume_start_epoch = 0
    FakeSegmentYOLO.steps_per_epoch = 3
    checks = []

    def guard() -> None:
        checks.append(len(checks) + 1)

    base = _spec(tmp_path, start=1, end=2, steps=3)
    spec = DynamicTrainingSpec(**{**base.__dict__, "runtime_health_check": guard})
    run_dynamic_training_segment(
        spec,
        yolo_factory=FakeSegmentYOLO,
        campaign_trainer_class=object,
        checkpoint_validator=_validator(-1),
    )

    assert len(checks) >= 8

    with pytest.raises(ValidationError, match="inside the registered output"):
        DynamicTrainingSpec(
            **{
                **base.__dict__,
                "smoke_failure_injection": SmokeFailureInjection(
                    mode="PAUSE_AT_EPOCH_START",
                    target_epoch=1,
                    marker_path=tmp_path / "outside/reached.json",
                    continue_marker_path=tmp_path / "outside/continue.marker",
                ),
            }
        )
