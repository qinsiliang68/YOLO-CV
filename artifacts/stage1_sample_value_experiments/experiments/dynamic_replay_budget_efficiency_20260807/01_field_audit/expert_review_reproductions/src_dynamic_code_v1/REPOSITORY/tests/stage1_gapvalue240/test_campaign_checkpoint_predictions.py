from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.campaign_checkpoint_predictions import (
    CampaignCheckpointPredictionSpec,
    CampaignPredictionError,
    build_causal_probe_manifests,
    run_key_checkpoint_predictions,
)


def _manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_build_causal_probe_manifests_resolves_exact_train_rows(tmp_path: Path) -> None:
    normal = _manifest(
        tmp_path / "normal.csv",
        [
            {"canonical_image_relpath": "normal/n1.jpg", "image_path": "normal/n1.jpg"},
            {"canonical_image_relpath": "normal/n2.jpg", "image_path": "normal/n2.jpg"},
        ],
    )
    defect = _manifest(
        tmp_path / "defect.csv",
        [{"canonical_image_relpath": "defect/d1.jpg", "image_path": "defect/d1.jpg"}],
    )
    monitor = _manifest(
        tmp_path / "monitor.csv",
        [
            {"sample_id": "normal/n2.jpg", "y_true": 0, "monitor_group": "A02"},
            {"sample_id": "defect/d1.jpg", "y_true": 1, "monitor_group": "B04"},
        ],
    )

    result = build_causal_probe_manifests(
        monitor,
        normal,
        defect,
        tmp_path / "probe",
    )

    normal_probe = pd.read_csv(result.normal_manifest)
    defect_probe = pd.read_csv(result.defect_manifest)
    assert normal_probe.canonical_image_relpath.tolist() == ["normal/n2.jpg"]
    assert defect_probe.canonical_image_relpath.tolist() == ["defect/d1.jpg"]
    assert result.normal_count == 1
    assert result.defect_count == 1


def _spec(tmp_path: Path) -> CampaignCheckpointPredictionSpec:
    manifests = tmp_path / "manifests"
    normal_train = _manifest(
        manifests / "normal_train.csv",
        [{"canonical_image_relpath": "normal/n1.jpg", "image_path": "normal/n1.jpg"}],
    )
    defect_train = _manifest(
        manifests / "defect_train.csv",
        [{"canonical_image_relpath": "defect/d1.jpg", "image_path": "defect/d1.jpg"}],
    )
    monitor = _manifest(
        manifests / "monitor.csv",
        [
            {"sample_id": "normal/n1.jpg", "y_true": 0, "monitor_group": "A02"},
            {"sample_id": "defect/d1.jpg", "y_true": 1, "monitor_group": "B04"},
        ],
    )
    val_normal = _manifest(
        manifests / "val_normal.csv",
        [{"canonical_image_relpath": "normal/vn.jpg", "image_path": "normal/vn.jpg"}],
    )
    val_defect = _manifest(
        manifests / "val_defect.csv",
        [{"canonical_image_relpath": "defect/vd.jpg", "image_path": "defect/vd.jpg"}],
    )
    state = tmp_path / "run/training_state"
    state.mkdir(parents=True)
    (state / "checkpoint_epoch_0120.pt").write_bytes(b"checkpoint-120")
    (state / "checkpoint_epoch_0140.pt").write_bytes(b"checkpoint-140")
    return CampaignCheckpointPredictionSpec(
        run_id="RUN_A",
        arm_id="T_SHARED_PREFIX",
        job_id="JOB_A",
        checkpoint_epochs=(120, 140),
        training_state_dir=state,
        output_dir=tmp_path / "run/key_checkpoint_predictions",
        dataset_root=tmp_path / "dataset",
        normal_train_manifest=normal_train,
        defect_train_manifest=defect_train,
        monitor_manifest=monitor,
        val_op_normal_manifest=val_normal,
        val_op_defect_manifest=val_defect,
        yolo_root=tmp_path / "YOLOv11",
        python_executable="python",
        gpu_id="0",
        batch=128,
        workers=0,
        imgsz=224,
        accepted_defect_names=("defect",),
    )


def test_key_checkpoint_predictions_publish_only_after_validation(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    calls: list[tuple[int, str]] = []

    def fake_runner(**kwargs):
        split_name = kwargs["split_name"]
        epoch = kwargs["epoch"]
        defect = pd.read_csv(kwargs["defect_manifest"])
        normal = pd.read_csv(kwargs["normal_manifest"])
        rows = [
            *(
                {"sample_id": row.canonical_image_relpath, "y_true": 1, "score": 0.8}
                for row in defect.itertuples(index=False)
            ),
            *(
                {"sample_id": row.canonical_image_relpath, "y_true": 0, "score": 0.2}
                for row in normal.itertuples(index=False)
            ),
        ]
        pd.DataFrame(rows).to_csv(kwargs["temporary_output"], index=False)
        Path(kwargs["result_json"]).write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
        Path(kwargs["log_path"]).write_text("ok", encoding="utf-8")
        calls.append((epoch, split_name))
        return {"status": "PASS", "exit_code": 0}

    result = run_key_checkpoint_predictions(spec, prediction_runner=fake_runner)

    assert result.status == "PASS"
    assert calls == [
        (120, "val_op"),
        (120, "causal_train_probe"),
        (140, "val_op"),
        (140, "causal_train_probe"),
    ]
    for epoch in (120, 140):
        epoch_dir = spec.output_dir / f"epoch_{epoch:04d}"
        assert (epoch_dir / "val_op_predictions.csv").is_file()
        assert (epoch_dir / "causal_train_probe_predictions.csv").is_file()
        enriched = pd.read_csv(epoch_dir / "val_op_predictions.csv")
        assert {
            "score_raw",
            "p_defect",
            "p_normal",
            "probability_margin",
            "log_odds_defect",
            "entropy_nats",
            "predicted_y",
        } <= set(enriched.columns)
        assert enriched.loc[0, "p_defect"] == enriched.loc[0, "score"]
        assert json.loads((epoch_dir / "checkpoint_prediction_manifest.json").read_text())["status"] == "COMPLETE"

    rerun = run_key_checkpoint_predictions(spec, prediction_runner=fake_runner)
    assert rerun.status == "PASS"
    assert len(calls) == 4


def test_key_checkpoint_predictions_do_not_accept_half_published_output(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    half = spec.output_dir / "epoch_0120"
    half.mkdir(parents=True)
    (half / "val_op_predictions.csv").write_text("sample_id,y_true,score\n", encoding="utf-8")

    with pytest.raises(CampaignPredictionError, match="half-published"):
        run_key_checkpoint_predictions(spec, prediction_runner=lambda **_: {})
