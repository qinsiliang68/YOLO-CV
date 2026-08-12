from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from stage1_dynamic_replay_v3.asset_identity import ManifestImageRecord
from stage1_dynamic_replay_v3.evaluation import (
    PredictionArtifactError,
    PredictionRow,
    compute_raw_safety_frontier,
    publish_prediction_artifact,
    validate_prediction_artifact,
)


def test_raw_frontier_is_tie_safe_and_monotone() -> None:
    rows = (
        PredictionRow("d-low", 1, 0.4, 200),
        PredictionRow("d-high", 1, 0.9, 200),
        PredictionRow("n-low", 0, 0.1, 200),
        PredictionRow("n-tie", 0, 0.4, 200),
        PredictionRow("n-high", 0, 0.8, 200),
    )
    metrics = compute_raw_safety_frontier(rows, max_fn=1, target_tn=1)
    assert [point.fn_budget for point in metrics.frontier] == [0, 1]
    assert [point.tn for point in metrics.frontier] == [1, 3]
    assert metrics.tn_at_fn_max == 3
    assert metrics.fn_at_target_tn == 0
    assert metrics.normalized_auc == pytest.approx(2 / 3)


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["canonical_image_relpath", "Filename", "Defect"])
        writer.writeheader()
        writer.writerows(rows)


def test_prediction_artifact_binds_manifests_and_checkpoint(tmp_path: Path) -> None:
    defect_manifest = tmp_path / "defect.csv"
    normal_manifest = tmp_path / "normal.csv"
    _write_manifest(defect_manifest, [{"canonical_image_relpath": "d.png", "Filename": "d.png", "Defect": 1}])
    _write_manifest(normal_manifest, [{"canonical_image_relpath": "n.png", "Filename": "n.png", "Defect": 0}])
    checkpoint = tmp_path / "epoch.pt"
    checkpoint.write_bytes(b"checkpoint")
    expected = (
        ManifestImageRecord("d.png", "d.png", 1, tmp_path / "d.png", defect_manifest),
        ManifestImageRecord("n.png", "n.png", 0, tmp_path / "n.png", normal_manifest),
    )
    output = tmp_path / "epoch_0200_predictions.csv"
    artifact = publish_prediction_artifact(
        rows=(PredictionRow("d.png", 1, 0.8, 200), PredictionRow("n.png", 0, 0.2, 200)),
        expected_records=expected,
        defect_manifest=defect_manifest,
        normal_manifest=normal_manifest,
        checkpoint=checkpoint,
        output_path=output,
    )
    assert validate_prediction_artifact(output, artifact.sidecar_path, expected_records=expected).status == "PASS"
    payload = json.loads(artifact.sidecar_path.read_text(encoding="utf-8"))
    assert payload["defect_manifest_sha256"]
    assert payload["normal_manifest_sha256"]
    assert payload["expected_sample_label_digest"] == payload["observed_sample_label_digest"]


def test_prediction_artifact_rejects_wrong_ids(tmp_path: Path) -> None:
    expected = (ManifestImageRecord("right", "right.png", 1, tmp_path / "right.png", tmp_path / "d.csv"),)
    with pytest.raises(PredictionArtifactError, match="identity"):
        publish_prediction_artifact(
            rows=(PredictionRow("wrong", 1, 0.8, 200),),
            expected_records=expected,
            defect_manifest=tmp_path / "d.csv",
            normal_manifest=tmp_path / "n.csv",
            checkpoint=tmp_path / "epoch.pt",
            output_path=tmp_path / "predictions.csv",
        )
