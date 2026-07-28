from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.deep_subgroups import (
    load_subgroup_metadata,
    summarize_shift_subgroups,
)


def _write_manifests(
    attempt: Path,
    *,
    include_optional: bool = True,
    duplicate_id: bool = False,
) -> None:
    root = attempt / "01_manifests" / "frozen_inputs"
    root.mkdir(parents=True)
    normal = {
        "canonical_image_relpath": ["Det/images/normal_val_op/n1.png"],
    }
    defect = {
        "canonical_image_relpath": [
            "Det/images/normal_val_op/n1.png"
            if duplicate_id
            else "Det/images/val_op/d1.png"
        ],
    }
    if include_optional:
        normal.update(
            {
                "WaterLevel": [0],
                "target_labels": [""],
                "source_csv_path": [r"C:\source\SewerML_Train.csv"],
                "PF": [0],
                "DE": [0],
                "Defect": [0],
            }
        )
        defect.update(
            {
                "WaterLevel": [2],
                "target_labels": ["PF;DE"],
                "source_csv_path": [r"C:\source\SewerML_Train.csv"],
                "PF": [1],
                "DE": [1],
                "Defect": [1],
            }
        )
    pd.DataFrame(normal).to_csv(root / "val_op_normal_manifest.csv", index=False)
    pd.DataFrame(defect).to_csv(root / "val_op_defect_manifest.csv", index=False)


def test_load_subgroup_metadata_preserves_identity_fields_and_availability(tmp_path):
    attempt = tmp_path / "attempt"
    _write_manifests(attempt)

    metadata = load_subgroup_metadata(attempt)

    assert metadata["sample_id"].tolist() == [
        "Det/images/normal_val_op/n1.png",
        "Det/images/val_op/d1.png",
    ]
    assert metadata["y_true"].tolist() == [0, 1]
    assert metadata["source_basename"].tolist() == [
        "SewerML_Train.csv",
        "SewerML_Train.csv",
    ]
    assert metadata.loc[metadata.y_true == 1, "PF"].item() == 1
    availability = metadata.attrs["field_availability"]
    assert availability["canonical_image_relpath"] is True
    assert availability["WaterLevel"] is True
    assert availability["target_labels"] is True
    assert availability["source_csv_path"] is True
    assert availability["source_basename"] is True
    assert availability["primary_defect_class"] is False
    assert availability["category_columns"] == ["PF", "DE", "Defect"]


def test_load_subgroup_metadata_marks_missing_fields_without_guessing(tmp_path):
    attempt = tmp_path / "attempt"
    _write_manifests(attempt, include_optional=False)

    metadata = load_subgroup_metadata(attempt)

    availability = metadata.attrs["field_availability"]
    assert availability["WaterLevel"] is False
    assert availability["target_labels"] is False
    assert availability["source_csv_path"] is False
    assert availability["source_basename"] is False
    assert availability["category_columns"] == []
    assert metadata["WaterLevel"].isna().all()
    assert metadata["target_labels"].isna().all()
    assert metadata["source_basename"].isna().all()


def test_load_subgroup_metadata_rejects_duplicate_ids_across_labels(tmp_path):
    attempt = tmp_path / "attempt"
    _write_manifests(attempt, duplicate_id=True)

    with pytest.raises(ValueError, match="unique"):
        load_subgroup_metadata(attempt)


def test_summarize_shift_subgroups_reports_control_and_available_dimensions(tmp_path):
    attempt = tmp_path / "attempt"
    _write_manifests(attempt)
    metadata = load_subgroup_metadata(attempt)
    shifts = pd.DataFrame(
        {
            "sample_id": metadata["sample_id"],
            "y_true": [0, 1],
            "control": ["R1", "R1"],
            "triad_id": ["TRIAD_004", "TRIAD_004"],
            "raw_shift": [-0.2, 0.3],
            "calibrated_shift": [-0.1, 0.2],
        }
    )

    summary = summarize_shift_subgroups(shifts, metadata)

    overall = summary[
        (summary.control == "R1")
        & (summary.subgroup_dimension == "overall")
    ].iloc[0]
    assert overall["n"] == 2
    assert overall["raw_beneficial_rate"] == 1.0
    assert overall["raw_mean_shift"] == pytest.approx(0.05)
    assert overall["calibrated_beneficial_rate"] == 1.0
    water = summary[
        (summary.subgroup_dimension == "WaterLevel")
        & (summary.subgroup_value == "2")
    ].iloc[0]
    assert water["n"] == 1
    assert water["raw_mean_shift"] == pytest.approx(0.3)
    target = summary[
        (summary.subgroup_dimension == "target_labels")
        & (summary.subgroup_value == "PF;DE")
    ].iloc[0]
    assert target["n"] == 1
    defect_class = summary[
        (summary.subgroup_dimension == "defect_class")
        & (summary.subgroup_value == "PF")
    ].iloc[0]
    assert defect_class["n"] == 1
    assert summary.attrs["field_availability"]["primary_defect_class"] is False


def test_summarize_shift_subgroups_accepts_frame_list_and_validates_labels(tmp_path):
    attempt = tmp_path / "attempt"
    _write_manifests(attempt)
    metadata = load_subgroup_metadata(attempt)
    first = pd.DataFrame(
        {
            "sample_id": metadata["sample_id"],
            "y_true": [0, 1],
            "control": ["R1", "R1"],
            "raw_shift": [-0.1, 0.2],
            "calibrated_shift": [-0.1, 0.1],
        }
    )
    second = first.copy()
    second["control"] = "R2"

    summary = summarize_shift_subgroups([first, second], metadata)

    assert set(summary["control"]) == {"R1", "R2"}
    wrong = first.copy()
    wrong.loc[0, "y_true"] = 1
    with pytest.raises(ValueError, match="label"):
        summarize_shift_subgroups(wrong, metadata)
