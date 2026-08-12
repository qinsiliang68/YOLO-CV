from __future__ import annotations

import csv
from pathlib import Path

import polars as pl
import pytest

from stage1_dynamic_replay_v3.oof_reference import (
    OOFReferenceError,
    binary_cross_entropy,
    build_oof_epoch200_reference,
)


def _write(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_oof_binary_cross_entropy_is_self_contained_and_boundary_safe() -> None:
    assert binary_cross_entropy(1, 0.8) == pytest.approx(0.2231435513)
    assert binary_cross_entropy(0, 0.2) == pytest.approx(0.2231435513)
    assert binary_cross_entropy(1, 0.0) > 20.0
    assert binary_cross_entropy(0, 1.0) > 20.0
    with pytest.raises(ValueError, match="binary label"):
        binary_cross_entropy(2, 0.5)
    with pytest.raises(ValueError, match="finite"):
        binary_cross_entropy(1, float("nan"))


def test_oof_reference_binds_exact_training_identity(tmp_path: Path) -> None:
    defect = tmp_path / "defect.csv"
    normal = tmp_path / "normal.csv"
    _write(defect, ["canonical_image_relpath", "Defect"], [{"canonical_image_relpath": "d.png", "Defect": 1}])
    _write(normal, ["canonical_image_relpath", "Defect"], [{"canonical_image_relpath": "n.png", "Defect": 0}])
    jobs = tmp_path / "jobs"
    _write(
        jobs / "job-a" / "fold_00" / "epoch_200_predictions.csv",
        ["sample_id", "y_true", "oof_fold", "epoch", "p_defect_raw"],
        [{"sample_id": "d.png", "y_true": 1, "oof_fold": "00", "epoch": 200, "p_defect_raw": 0.8}],
    )
    _write(
        jobs / "job-b" / "fold_01" / "epoch_200_predictions.csv",
        ["sample_id", "y_true", "oof_fold", "epoch", "p_defect_raw"],
        [{"sample_id": "n.png", "y_true": 0, "oof_fold": "01", "epoch": 200, "p_defect_raw": 0.2}],
    )

    result = build_oof_epoch200_reference(
        jobs_root=jobs,
        defect_manifest=defect,
        normal_manifest=normal,
        output_path=tmp_path / "reference.parquet",
        expected_rows=2,
        expected_defect=1,
        expected_normal=1,
        expected_fold_count=2,
    )

    table = pl.read_parquet(result.output_path).sort("sample_id")
    assert result.status == "PASS"
    assert table.get_column("sample_id").to_list() == ["d.png", "n.png"]
    assert table.get_column("oof_ce").is_finite().all()
    assert result.sidecar_path.is_file()


def test_oof_reference_rejects_wrong_sample_identity(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write(manifest, ["canonical_image_relpath", "Defect"], [{"canonical_image_relpath": "right.png", "Defect": 1}])
    prediction = tmp_path / "jobs" / "job" / "fold_00" / "epoch_200_predictions.csv"
    _write(
        prediction,
        ["sample_id", "y_true", "oof_fold", "epoch", "p_defect_raw"],
        [{"sample_id": "wrong.png", "y_true": 1, "oof_fold": "00", "epoch": 200, "p_defect_raw": 0.9}],
    )

    with pytest.raises(OOFReferenceError, match="identity"):
        build_oof_epoch200_reference(
            jobs_root=tmp_path / "jobs",
            defect_manifest=manifest,
            normal_manifest=tmp_path / "empty.csv",
            output_path=tmp_path / "reference.parquet",
            expected_rows=1,
            expected_defect=1,
            expected_normal=0,
            expected_fold_count=1,
        )
