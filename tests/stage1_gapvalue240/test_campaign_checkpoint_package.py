from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from stage1_gapvalue240.campaign_checkpoint_package import (
    CheckpointPackageError,
    CheckpointPackageSpec,
    build_checkpoint_package,
    model_state_digest,
)


def _source_checkpoint(path: Path, *, scale: float = 1.0) -> Path:
    model = torch.nn.Linear(3, 2).float()
    ema = torch.nn.Linear(3, 2).half()
    with torch.no_grad():
        model.weight.fill_(0.25 * scale)
        model.bias.fill_(0.5 * scale)
        ema.weight.fill_(1.25 * scale)
        ema.bias.fill_(1.5 * scale)
    torch.save(
        {
            "epoch": 119,
            "model": model,
            "ema": ema,
            "optimizer": {"state": {"large": torch.ones(10_000)}},
            "scaler": {"scale": 1024},
            "updates": 123,
            "train_args": {"imgsz": 224, "batch": 128},
            "version": "test",
        },
        path,
    )
    return path


def _spec(tmp_path: Path, source: Path) -> CheckpointPackageSpec:
    return CheckpointPackageSpec(
        run_id="DRBE_S001_T_SHARED_PREFIX",
        arm_id="T_SHARED_PREFIX",
        source_job_id="JOB_S001_TP_E001_140",
        source_machine_id="machine_05",
        logical_epoch=120,
        source_checkpoint=source,
        output_dir=tmp_path / "epoch_0120",
        yolo_root=tmp_path / "YOLOv11",
    )


def test_checkpoint_package_preserves_serialized_ema_values_and_strips_training_state(
    tmp_path: Path,
) -> None:
    source = _source_checkpoint(tmp_path / "source.pt")
    source_bytes = source.read_bytes()
    source_payload = torch.load(source, map_location="cpu", weights_only=False)
    expected_digest = model_state_digest(source_payload["ema"])

    result = build_checkpoint_package(_spec(tmp_path, source))

    assert result.status == "PASS"
    assert result.skipped is False
    assert source.read_bytes() == source_bytes
    packaged = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert packaged["optimizer"] is None
    assert packaged["scaler"] is None
    assert packaged["ema"] is None
    assert packaged["epoch"] == 119
    assert packaged["model"].weight.dtype == torch.float16
    assert model_state_digest(packaged["model"]) == expected_digest

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE"
    assert manifest["logical_epoch"] == 120
    assert manifest["source_checkpoint_internal_epoch"] == 119
    assert manifest["weight_source"] == "ema"
    assert manifest["numerical_weight_transform"] == "NONE"
    assert manifest["model_state_digest"] == expected_digest
    assert manifest["source_checkpoint_size_bytes"] > manifest["package_checkpoint_size_bytes"]

    rerun = build_checkpoint_package(_spec(tmp_path, source))
    assert rerun.skipped is True


def test_checkpoint_package_rejects_changed_source_and_wrong_epoch(tmp_path: Path) -> None:
    source = _source_checkpoint(tmp_path / "source.pt")
    wrong = CheckpointPackageSpec(**{**_spec(tmp_path, source).__dict__, "logical_epoch": 121})
    with pytest.raises(CheckpointPackageError, match="logical epoch"):
        build_checkpoint_package(wrong)

    build_checkpoint_package(_spec(tmp_path, source))
    _source_checkpoint(source, scale=2.0)
    with pytest.raises(CheckpointPackageError, match="source checkpoint"):
        build_checkpoint_package(_spec(tmp_path, source))


def test_checkpoint_package_rejects_half_published_output(tmp_path: Path) -> None:
    source = _source_checkpoint(tmp_path / "source.pt")
    output = tmp_path / "epoch_0120"
    output.mkdir()
    (output / "checkpoint_epoch_0120.pt").write_bytes(b"partial")

    with pytest.raises(CheckpointPackageError, match="half-published"):
        build_checkpoint_package(_spec(tmp_path, source))
