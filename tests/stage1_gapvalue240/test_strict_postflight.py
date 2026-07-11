from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from stage1_gapvalue240.errors import ValidationError
from stage1_gapvalue240.metrics import operational_metrics
from stage1_gapvalue240.util import sha256_file
from stage1_gapvalue240.validation import (
    strict_postflight,
    verify_permanent_artifact_manifest,
    write_permanent_artifact_manifest,
)


def _manifest(path: Path, ids: list[str]) -> Path:
    pd.DataFrame({"canonical_image_relpath": ids, "Filename": [f"{x}.png" for x in ids]}).to_csv(path, index=False)
    return path


def _valid_attempt(tmp_path: Path, epochs: int = 3) -> tuple[Path, dict]:
    attempt = tmp_path / "attempt_x.inprogress"
    for rel in ("02_logs", "03_checkpoints", "04_predictions", "05_metrics", "07_validation"):
        (attempt / rel).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"epoch": range(1, epochs + 1), "train/loss": [0.3, 0.2, 0.1][:epochs]}).to_csv(
        attempt / "02_logs/epoch_training_metrics.csv", index=False
    )
    audit = {
        "schema_version": 1,
        "expected_epochs": epochs,
        "completed_epochs": epochs,
        "expected_steps_per_epoch": 2,
        "observed_steps_per_epoch": [2] * epochs,
        "optimizer_steps_total": 2 * epochs,
        "effective_batch_size": 128,
        "configured_args": {
            "epochs": epochs, "batch": 128, "imgsz": 224, "patience": 0, "seed": 123,
            "deterministic": True, "cache": False, "model": "yolo11l-cls.pt",
        },
        "loss_finite": True,
        "resume_mode": "native_approximate",
        "resume_count": 0,
        "resume_segments": [],
    }
    (attempt / "02_logs/training_execution_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    args = {
        **audit["configured_args"],
        "optimizer": "auto", "lr0": 0.01, "lrf": 0.01, "momentum": 0.937,
        "weight_decay": 0.0005, "warmup_epochs": 3.0, "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0,
        "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5, "bgr": 0.0,
        "mosaic": 1.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
        "auto_augment": "randaugment", "erasing": 0.4,
    }
    args_path = attempt / "02_logs/args.yaml"
    args_path.write_text(yaml.safe_dump(args), encoding="utf-8")
    optimization_keys = {"optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "warmup_momentum", "warmup_bias_lr"}
    augmentation_keys = {"hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr", "bgr", "mosaic", "mixup", "cutmix", "copy_paste", "auto_augment", "erasing"}
    (attempt / "02_logs/resolved_training_args.json").write_text(json.dumps({
        "args_yaml_sha256": sha256_file(args_path),
        "resolved_args": args,
        "optimization": {key: args[key] for key in optimization_keys},
        "augmentation": {key: args[key] for key in augmentation_keys},
    }), encoding="utf-8")
    for name in ("best.pt", "last.pt"):
        (attempt / "03_checkpoints" / name).write_bytes(b"valid checkpoint fixture")

    cal_ids = ["cal_d", "cal_n"]
    op_ids = ["op_d1", "op_d2", "op_n1", "op_n2"]
    cal = pd.DataFrame({"sample_id": cal_ids, "y_true": [1, 0], "score": [0.9, 0.1]})
    op = pd.DataFrame({"sample_id": op_ids, "y_true": [1, 1, 0, 0], "score": [0.9, 0.8, 0.2, 0.1]})
    cal.to_csv(attempt / "04_predictions/val_cal_predictions.csv", index=False)
    op.to_csv(attempt / "04_predictions/val_op_predictions.csv", index=False)
    metrics, _ = operational_metrics(op)
    (attempt / "05_metrics/operational_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    manifests = tmp_path / "manifests"
    manifests.mkdir()
    expected = {
        "epochs": epochs,
        "steps_per_epoch": 2,
        "batch_size": 128,
        "imgsz": 224,
        "seed": 123,
        "model_filename": "yolo11l-cls.pt",
        "val_cal_defect_manifest": str(_manifest(manifests / "cal_d.csv", ["cal_d"])),
        "val_cal_normal_manifest": str(_manifest(manifests / "cal_n.csv", ["cal_n"])),
        "val_op_defect_manifest": str(_manifest(manifests / "op_d.csv", ["op_d1", "op_d2"])),
        "val_op_normal_manifest": str(_manifest(manifests / "op_n.csv", ["op_n1", "op_n2"])),
    }
    return attempt, expected


def test_strict_postflight_accepts_complete_training_and_predictions(tmp_path):
    attempt, expected = _valid_attempt(tmp_path)
    checked = []
    report = strict_postflight(attempt, attempt / "07_validation/postflight_report.json", expected,
                               checkpoint_validator=lambda path: checked.append(path.name))
    assert report["status"] == "PASS"
    assert checked == ["best.pt", "last.pt"]


def test_strict_postflight_rejects_early_stopping(tmp_path):
    attempt, expected = _valid_attempt(tmp_path, epochs=3)
    pd.read_csv(attempt / "02_logs/epoch_training_metrics.csv").iloc[:2].to_csv(
        attempt / "02_logs/epoch_training_metrics.csv", index=False
    )
    with pytest.raises(ValidationError, match="Strict postflight failed"):
        strict_postflight(attempt, attempt / "07_validation/postflight_report.json", expected)
    report = json.loads((attempt / "07_validation/postflight_report.json").read_text())
    assert any("epoch rows" in issue for issue in report["issues"])


def test_strict_postflight_rejects_prediction_identity_drift(tmp_path):
    attempt, expected = _valid_attempt(tmp_path)
    pred = pd.read_csv(attempt / "04_predictions/val_op_predictions.csv")
    pred.loc[0, "sample_id"] = "wrong_id"
    pred.to_csv(attempt / "04_predictions/val_op_predictions.csv", index=False)
    with pytest.raises(ValidationError):
        strict_postflight(attempt, attempt / "07_validation/postflight_report.json", expected)
    report = json.loads((attempt / "07_validation/postflight_report.json").read_text())
    assert any("val_op prediction identities" in issue for issue in report["issues"])


def test_artifact_manifest_never_hashes_staging_or_work_images(tmp_path):
    attempt = tmp_path / "attempt.inprogress"
    permanent = attempt / "05_metrics/metrics.json"
    permanent.parent.mkdir(parents=True)
    permanent.write_text("{}", encoding="utf-8")
    for relative in ("work/train/image.jpg", "staging/replay.jpg", "training_state/last.pt"):
        path = attempt / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"large temporary payload")
    output = attempt / "07_validation/artifact_manifest.csv"

    rows = write_permanent_artifact_manifest(attempt, output)

    assert [row["relative_path"] for row in rows] == ["05_metrics/metrics.json"]
    assert output.exists()
    assert verify_permanent_artifact_manifest(attempt, output)["artifact_count"] == 1
    permanent.write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(ValidationError, match="artifact manifest differs"):
        verify_permanent_artifact_manifest(attempt, output)
