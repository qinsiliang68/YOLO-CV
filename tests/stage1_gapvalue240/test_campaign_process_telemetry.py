from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch
import torch.nn.functional as F

from stage1_gapvalue240.campaign_process_telemetry import (
    ProcessTelemetryInstaller,
    ProcessTelemetryCollector,
    ProcessTelemetrySpec,
    TelemetryCriterion,
)
from stage1_gapvalue240.errors import ValidationError


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _spec(tmp_path: Path, *, run_id: str = "RUN_A") -> ProcessTelemetrySpec:
    normal = _write_manifest(
        tmp_path / "base_normal.csv",
        [
            {
                "canonical_image_relpath": "normal/n0.jpg",
                "Filename": "n0.jpg",
            },
            {
                "canonical_image_relpath": "normal/n1.jpg",
                "Filename": "n1.jpg",
            },
        ],
    )
    defect = _write_manifest(
        tmp_path / "base_defect.csv",
        [
            {
                "canonical_image_relpath": "defect/d0.jpg",
                "Filename": "d0.jpg",
            }
        ],
    )
    replay = _write_manifest(
        tmp_path / "replay.csv",
        [
            {
                "staged_filename": "replay__RUN_A__00001__n0.jpg",
                "sample_id": "normal/n0.jpg",
                "y_true": 0,
                "replay_role": "normal_replay",
            }
        ],
    )
    monitor = _write_manifest(
        tmp_path / "monitor.csv",
        [
            {"sample_id": "normal/n0.jpg", "monitor_group": "A02_NORMAL"},
            {"sample_id": "defect/d0.jpg", "monitor_group": "WEAK_DEFECT"},
        ],
    )
    return ProcessTelemetrySpec(
        run_id=run_id,
        arm_id="T_DYNAMIC_DECAY",
        segment_id=f"{run_id}_E141_150",
        output_dir=tmp_path / run_id / "process_telemetry",
        base_normal_manifest=normal,
        base_defect_manifest=defect,
        replay_identity_manifest=replay,
        monitor_manifest=monitor,
        expected_epoch_samples=4,
        expected_replay_samples=1,
        target_defect_class_index=1,
    )


def _record_epoch(collector: ProcessTelemetryCollector, *, reverse: bool = False) -> Path:
    collector.start_epoch(141)
    names = [
        "C:/stage/train/no_target/n0.jpg",
        "C:/stage/train/target_defect/d0.jpg",
        "C:/stage/train/no_target/n1.jpg",
        "C:/stage/train/no_target/replay__RUN_A__00001__n0.jpg",
    ]
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5], [1.0, 0.0], [0.2, 0.8]])
    labels = torch.tensor([0, 1, 0, 0])
    images = torch.arange(4 * 3 * 16 * 16, dtype=torch.float32).reshape(4, 3, 16, 16) / 4096.0
    if reverse:
        order = torch.tensor([3, 2, 1, 0])
        names = [names[index] for index in order.tolist()]
        logits = logits[order]
        labels = labels[order]
        images = images[order]
    collector.record_batch(logits, {"cls": labels, "im_file": names, "img": images})
    return collector.finish_epoch(141)


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.DataFrame(pl.read_parquet(path).to_dicts())


def test_process_telemetry_records_roles_monitored_samples_and_realized_digests(tmp_path: Path) -> None:
    collector = ProcessTelemetryCollector(_spec(tmp_path))
    output = _record_epoch(collector)

    frame = _read_parquet(output)
    assert set(frame.record_type) == {"EPOCH", "ROLE", "SAMPLE"}

    epoch = frame.loc[frame.record_type == "EPOCH"].iloc[0]
    assert epoch.total_exposure_count == 4
    assert epoch.batch_count == 1
    assert len(epoch.minibatch_order_sha256) == 64
    assert len(epoch.augmentation_realization_sha256) == 64

    roles = frame.loc[frame.record_type == "ROLE"].set_index("exposure_role")
    assert int(roles.loc["base_normal", "total_exposure_count"]) == 2
    assert int(roles.loc["base_defect", "total_exposure_count"]) == 1
    assert int(roles.loc["normal_replay", "total_exposure_count"]) == 1

    samples = frame.loc[frame.record_type == "SAMPLE"].set_index("sample_id")
    selected = samples.loc["normal/n0.jpg"]
    assert int(selected.total_exposure_count) == 2
    assert int(selected.base_exposure_count) == 1
    assert int(selected.replay_normal_exposure_count) == 1
    assert selected.monitor_groups == "A02_NORMAL"
    assert len(selected.augmentation_realization_sha256) == 64

    weak_defect = samples.loc["defect/d0.jpg"]
    assert int(weak_defect.total_exposure_count) == 1
    assert int(weak_defect.base_exposure_count) == 1
    assert weak_defect.monitor_groups == "WEAK_DEFECT"

    expected_loss = F.cross_entropy(
        torch.tensor([[2.0, -1.0], [0.2, 0.8]]),
        torch.tensor([0, 0]),
        reduction="none",
    ).mean()
    assert np.isclose(float(selected.loss_mean), float(expected_loss), rtol=1e-6)

    sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["status"] == "COMPLETE"
    assert sidecar["row_count"] == len(frame)
    assert sidecar["epoch"] == 141
    assert sidecar["parquet_sha256"]


def test_process_telemetry_batch_order_digest_changes_but_role_statistics_do_not(tmp_path: Path) -> None:
    first = ProcessTelemetryCollector(_spec(tmp_path / "first", run_id="RUN_A"))
    first_path = _record_epoch(first, reverse=False)
    second_spec = _spec(tmp_path / "second", run_id="RUN_A")
    second = ProcessTelemetryCollector(second_spec)
    second_path = _record_epoch(second, reverse=True)

    a = _read_parquet(first_path)
    b = _read_parquet(second_path)
    a_epoch = a.loc[a.record_type == "EPOCH"].iloc[0]
    b_epoch = b.loc[b.record_type == "EPOCH"].iloc[0]
    assert a_epoch.minibatch_order_sha256 != b_epoch.minibatch_order_sha256
    assert a_epoch.augmentation_realization_sha256 != b_epoch.augmentation_realization_sha256

    columns = ["exposure_role", "total_exposure_count", "loss_mean", "p_defect_mean"]
    pd.testing.assert_frame_equal(
        a.loc[a.record_type == "ROLE", columns].sort_values("exposure_role").reset_index(drop=True),
        b.loc[b.record_type == "ROLE", columns].sort_values("exposure_role").reset_index(drop=True),
    )


def test_augmentation_statistics_stream_without_retaining_all_image_values(tmp_path: Path) -> None:
    collector = ProcessTelemetryCollector(_spec(tmp_path))
    collector.start_epoch(141)
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5], [1.0, 0.0], [0.2, 0.8]])
    images = torch.arange(4 * 3 * 16 * 16, dtype=torch.float32).reshape(4, 3, 16, 16) / 4096.0
    collector.record_batch(
        logits,
        {
            "cls": torch.tensor([0, 1, 0, 0]),
            "im_file": [
                "C:/stage/train/no_target/n0.jpg",
                "C:/stage/train/target_defect/d0.jpg",
                "C:/stage/train/no_target/n1.jpg",
                "C:/stage/train/no_target/replay__RUN_A__00001__n0.jpg",
            ],
            "img": images,
        },
    )

    assert not hasattr(collector, "pooled_values")
    assert collector.augmentation_value_count == 4 * 8 * 8
    assert collector.augmentation_value_sum > 0


def test_telemetry_criterion_preserves_original_training_loss_exactly(tmp_path: Path) -> None:
    collector = ProcessTelemetryCollector(_spec(tmp_path))
    collector.start_epoch(141)

    class OriginalCriterion:
        def __call__(self, preds, batch):
            value = F.cross_entropy(preds, batch["cls"], reduction="mean") * 1.25
            return value, value.detach()

    wrapped = TelemetryCriterion(OriginalCriterion(), collector)
    logits = torch.tensor([[1.5, -0.5], [0.0, 1.0]], requires_grad=True)
    batch = {
        "cls": torch.tensor([0, 1]),
        "im_file": [
            "C:/stage/train/no_target/n0.jpg",
            "C:/stage/train/target_defect/d0.jpg",
        ],
        "img": torch.ones(2, 3, 16, 16),
    }
    expected = OriginalCriterion()(logits, batch)
    observed = wrapped(logits, batch)

    assert torch.equal(observed[0], expected[0])
    assert torch.equal(observed[1], expected[1])
    observed[0].backward()
    assert logits.grad is not None
    checkpoint_copy = copy.deepcopy(wrapped)
    assert not isinstance(checkpoint_copy, TelemetryCriterion)
    copied = checkpoint_copy(logits.detach(), batch)
    assert torch.equal(copied[0], expected[0])


def test_process_telemetry_fails_on_unknown_or_duplicate_epoch_data(tmp_path: Path) -> None:
    collector = ProcessTelemetryCollector(_spec(tmp_path))
    collector.start_epoch(141)
    with pytest.raises(ValidationError, match="not present in the frozen identity index"):
        collector.record_batch(
            torch.tensor([[1.0, 0.0]]),
            {
                "cls": torch.tensor([0]),
                "im_file": ["C:/stage/train/no_target/unknown.jpg"],
                "img": torch.ones(1, 3, 8, 8),
            },
        )

    complete = ProcessTelemetryCollector(_spec(tmp_path / "complete"))
    _record_epoch(complete)
    duplicate = ProcessTelemetryCollector(_spec(tmp_path / "complete"))
    with pytest.raises(ValidationError, match="already complete"):
        duplicate.start_epoch(141)


def test_process_telemetry_rejects_incomplete_replay_identity_contract(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    replay = pd.read_csv(spec.replay_identity_manifest)
    replay.loc[0, "replay_role"] = "base_normal"
    replay.to_csv(spec.replay_identity_manifest, index=False)
    with pytest.raises(ValidationError, match="replay_role"):
        ProcessTelemetryCollector(spec)


def test_process_telemetry_ignores_ultralytics_final_eval_fit_callback(tmp_path: Path) -> None:
    collector = ProcessTelemetryCollector(_spec(tmp_path))
    _record_epoch(collector)

    # Ultralytics increments trainer.epoch once and emits on_fit_epoch_end again
    # while validating best.pt after training has already finished.
    collector.on_fit_epoch_end(SimpleNamespace(epoch=141))


def test_process_telemetry_installer_does_not_replace_model_initializer(tmp_path: Path) -> None:
    collector = ProcessTelemetryCollector(_spec(tmp_path))

    class OriginalCriterion:
        def __call__(self, preds, batch):
            value = F.cross_entropy(preds, batch["cls"])
            return value, value.detach()

    class Model:
        criterion = None

        def init_criterion(self):
            return OriginalCriterion()

    model = Model()
    initializer = model.init_criterion.__func__
    trainer = SimpleNamespace(model=model, data={"names": {0: "no_target", 1: "target_defect"}})

    ProcessTelemetryInstaller(collector).on_train_start(trainer)

    assert isinstance(model.criterion, TelemetryCriterion)
    assert model.init_criterion.__func__ is initializer
    checkpoint_model = copy.deepcopy(model)
    assert not isinstance(checkpoint_model.criterion, TelemetryCriterion)


def test_process_telemetry_write_interruption_never_publishes_partial_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    collector = ProcessTelemetryCollector(spec)
    collector.start_epoch(141)
    names = [
        "C:/stage/train/no_target/n0.jpg",
        "C:/stage/train/target_defect/d0.jpg",
        "C:/stage/train/no_target/n1.jpg",
        "C:/stage/train/no_target/replay__RUN_A__00001__n0.jpg",
    ]
    collector.record_batch(
        torch.tensor([[2.0, -1.0], [-0.5, 1.5], [1.0, 0.0], [0.2, 0.8]]),
        {
            "cls": torch.tensor([0, 1, 0, 0]),
            "im_file": names,
            "img": torch.ones(4, 3, 16, 16),
        },
    )

    def interrupted_write(_frame, path, **_kwargs):
        Path(path).write_bytes(b"partial parquet")
        raise OSError("injected disk interruption")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", interrupted_write)
    with pytest.raises(OSError, match="injected disk interruption"):
        collector.finish_epoch(141)

    parquet = spec.output_dir / "epoch_0141_process_telemetry.parquet"
    assert not parquet.exists()
    assert not parquet.with_suffix(".json").exists()


def test_process_telemetry_interruption_after_parquet_publish_rolls_back_epoch(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)

    def interrupt_after_publish(epoch: int) -> None:
        raise OSError(f"injected after parquet publish at epoch {epoch}")

    collector = ProcessTelemetryCollector(
        spec,
        after_parquet_publish=interrupt_after_publish,
    )
    with pytest.raises(OSError, match="after parquet publish"):
        _record_epoch(collector)

    parquet = spec.output_dir / "epoch_0141_process_telemetry.parquet"
    assert not parquet.exists()
    assert not parquet.with_suffix(".json").exists()
    assert not list(spec.output_dir.glob("*.tmp"))


def test_bad_completed_sidecar_is_rejected_without_overwrite(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    output = _record_epoch(ProcessTelemetryCollector(spec))
    original_sha = __import__("hashlib").sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".json").write_text("{broken", encoding="utf-8")

    with pytest.raises(ValidationError, match="sidecar is unreadable"):
        ProcessTelemetryCollector(spec).start_epoch(141)

    assert __import__("hashlib").sha256(output.read_bytes()).hexdigest() == original_sha
