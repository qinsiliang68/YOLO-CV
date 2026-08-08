from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch
import torch.nn.functional as F

from stage1_gapvalue240.campaign_gradient_probe import (
    GradientProbeError,
    GradientProbeSpec,
    compute_last_layer_gradient_metrics,
    run_gradient_probe,
)


def _flat_gradient(feature: torch.Tensor, logits: torch.Tensor, label: int) -> torch.Tensor:
    weight = torch.zeros((2, feature.numel()), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    adjusted = logits + weight @ feature + bias
    loss = F.cross_entropy(adjusted.unsqueeze(0), torch.tensor([label]))
    grad_weight, grad_bias = torch.autograd.grad(loss, (weight, bias))
    return torch.cat([grad_weight.reshape(-1), grad_bias])


def test_analytic_last_layer_gradients_match_autograd_and_leave_one_out() -> None:
    features = np.array(
        [
            [1.0, 0.5, -0.5],
            [0.2, 1.5, 0.3],
            [-0.7, 0.1, 1.2],
            [0.9, -0.4, 0.8],
        ],
        dtype=np.float64,
    )
    logits = np.array(
        [[1.0, -0.2], [0.4, 0.1], [-0.5, 0.9], [0.2, 1.1]],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    normal_target = np.array([True, True, False, False])
    defect_target = np.array([False, False, True, True])

    result = compute_last_layer_gradient_metrics(
        features,
        logits,
        labels,
        normal_target_mask=normal_target,
        defect_target_mask=defect_target,
    )

    gradients = torch.stack(
        [
            _flat_gradient(
                torch.tensor(features[index]),
                torch.tensor(logits[index]),
                int(labels[index]),
            )
            for index in range(len(labels))
        ]
    ).numpy()
    normal_mean = gradients[normal_target].mean(axis=0)
    defect_mean = gradients[defect_target].mean(axis=0)

    np.testing.assert_allclose(result.metrics.gradient_norm, np.linalg.norm(gradients, axis=1), rtol=1e-12)
    np.testing.assert_allclose(result.metrics.normal_target_dot, gradients @ normal_mean, rtol=1e-12)
    np.testing.assert_allclose(result.metrics.defect_target_dot, gradients @ defect_mean, rtol=1e-12)
    for index in range(2):
        leave_one_out = gradients[normal_target & (np.arange(len(labels)) != index)].mean(axis=0)
        expected_dot = gradients[index] @ leave_one_out
        expected_cosine = expected_dot / (
            np.linalg.norm(gradients[index]) * np.linalg.norm(leave_one_out)
        )
        assert np.isclose(result.metrics.loc[index, "normal_target_dot_self_excluded"], expected_dot)
        assert np.isclose(result.metrics.loc[index, "normal_target_cosine_self_excluded"], expected_cosine)


def test_gradient_metrics_keep_direction_axes_separate_without_composite_weight() -> None:
    features = np.eye(4, dtype=np.float64)
    logits = np.array([[0.1, 0.9], [0.8, 0.2], [0.7, 0.3], [0.2, 0.8]])
    labels = np.array([0, 0, 1, 1])
    result = compute_last_layer_gradient_metrics(
        features,
        logits,
        labels,
        normal_target_mask=np.array([True, True, False, False]),
        defect_target_mask=np.array([False, False, True, True]),
    )

    assert "normal_target_dot" in result.metrics
    assert "defect_target_dot" in result.metrics
    assert "gradient_value_score" not in result.metrics
    assert set(result.metrics.alignment_quadrant) <= {
        "HELP_BOTH",
        "HELP_NORMAL_HARM_DEFECT",
        "HARM_NORMAL_HELP_DEFECT",
        "HARM_BOTH",
        "NEUTRAL_AXIS",
    }


def _probe_spec(tmp_path: Path) -> GradientProbeSpec:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    candidate = tmp_path / "candidates.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "normal/n1.jpg",
                "y_true": 0,
                "image_path": "normal/n1.jpg",
                "normal_target_member": True,
                "defect_target_member": False,
                "candidate_groups": "A02",
            },
            {
                "sample_id": "normal/n2.jpg",
                "y_true": 0,
                "image_path": "normal/n2.jpg",
                "normal_target_member": True,
                "defect_target_member": False,
                "candidate_groups": "A02;R1",
            },
            {
                "sample_id": "defect/d1.jpg",
                "y_true": 1,
                "image_path": "defect/d1.jpg",
                "normal_target_member": False,
                "defect_target_member": True,
                "candidate_groups": "B04",
            },
            {
                "sample_id": "defect/d2.jpg",
                "y_true": 1,
                "image_path": "defect/d2.jpg",
                "normal_target_member": False,
                "defect_target_member": True,
                "candidate_groups": "B04",
            },
        ]
    ).to_csv(candidate, index=False)
    dataset = tmp_path / "dataset"
    for relative in ("normal/n1.jpg", "normal/n2.jpg", "defect/d1.jpg", "defect/d2.jpg"):
        path = dataset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    return GradientProbeSpec(
        run_id="RUN_A",
        arm_id="T_DYNAMIC_DECAY",
        checkpoint_epoch=160,
        checkpoint=checkpoint,
        candidate_manifest=candidate,
        dataset_root=dataset,
        output_dir=tmp_path / "gradient_probe",
        yolo_root=tmp_path / "YOLOv11",
        gpu_id="0",
        batch=4,
        workers=0,
        imgsz=224,
        accepted_defect_names=("defect", "target_defect"),
        save_feature_payload=True,
    )


def test_gradient_probe_atomically_publishes_scalars_features_and_manifest(tmp_path: Path) -> None:
    spec = _probe_spec(tmp_path)
    calls = []

    def fake_extractor(**kwargs):
        calls.append(tuple(kwargs["sample_ids"]))
        return (
            np.array([[1.0, 0.0], [0.5, 1.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
            np.array([[1.0, 0.0], [0.5, 0.1], [0.1, 1.0], [0.2, 0.8]], dtype=np.float32),
            {"extractor": "fake", "class_names": {"0": "normal", "1": "defect"}},
        )

    result = run_gradient_probe(spec, feature_extractor=fake_extractor)

    assert result.status == "PASS"
    assert result.skipped is False
    assert result.row_count == 4
    assert len(calls) == 1
    scalars = pl.read_parquet(spec.output_dir / "gradient_probe_scalars.parquet")
    assert {"gradient_norm", "normal_target_dot", "defect_target_dot", "alignment_quadrant"} <= set(
        scalars.columns
    )
    payload = np.load(spec.output_dir / "gradient_feature_payload.npz")
    assert payload["features"].shape == (4, 2)
    manifest = json.loads((spec.output_dir / "gradient_probe_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE"
    assert manifest["row_count"] == 4

    rerun = run_gradient_probe(spec, feature_extractor=fake_extractor)
    assert rerun.skipped is True
    assert len(calls) == 1


def test_gradient_probe_rejects_half_published_outputs(tmp_path: Path) -> None:
    spec = _probe_spec(tmp_path)
    spec.output_dir.mkdir(parents=True)
    (spec.output_dir / "gradient_probe_scalars.parquet").write_bytes(b"partial")

    with pytest.raises(GradientProbeError, match="half-published"):
        run_gradient_probe(spec, feature_extractor=lambda **_: None)
