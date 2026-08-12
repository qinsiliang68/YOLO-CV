from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stage1_gapvalue240.deep_analysis import CanonicalInputError
from stage1_gapvalue240.reconciliation_analysis import (
    build_unified_triad_outcomes,
    compute_late_overfit_features,
    controlled_cluster_regression,
    leave_group_out_predictions,
    load_expert_package,
)


def _expert_zip(path: Path, *, corrupt_manifest: bool = False) -> Path:
    root = "expert/"
    outcomes = pd.DataFrame(
        {
            "triad_id": ["T1"],
            "strong_positive": [True],
            "cost_effective": [True],
            "harmful": [False],
            "outcome_class": ["strong_positive"],
            "delta_TN_R1": [10],
            "delta_TN_R2": [11],
            "delta_FN_R1": [0],
            "delta_FN_R2": [-1],
        }
    ).to_csv(index=False).encode()
    report = b"# report\n"
    records = [
        ("tables/triad_outcome_classes.csv", outcomes),
        ("GOOD_COHORT_PATTERN_REPORT_CN.md", report),
    ]
    manifest = pd.DataFrame(
        [
            {
                "path": name,
                "size_bytes": len(data),
                "sha256": (
                    "0" * 64
                    if corrupt_manifest and index == 0
                    else hashlib.sha256(data).hexdigest().upper()
                ),
            }
            for index, (name, data) in enumerate(records)
        ]
    ).to_csv(index=False).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in records:
            archive.writestr(root + name, data)
        archive.writestr(root + "FILE_MANIFEST.csv", manifest)
    return path


def test_expert_package_requires_valid_internal_manifest(tmp_path: Path) -> None:
    loaded = load_expert_package(_expert_zip(tmp_path / "ok.zip"))
    assert loaded["validation"]["status"] == "PASS"
    assert len(loaded["tables"]["triad_outcome_classes"]) == 1

    with pytest.raises(CanonicalInputError, match="manifest"):
        load_expert_package(
            _expert_zip(tmp_path / "bad.zip", corrupt_manifest=True)
        )


def test_unified_outcomes_keep_relative_and_absolute_gates_separate() -> None:
    expert = pd.DataFrame(
        {
            "triad_id": ["robust", "local", "secondary", "relative", "harmful", "mixed"],
            "condition_id": [f"C{i}" for i in range(6)],
            "training_seed": range(6),
            "strong_positive": [True, True, True, True, False, False],
            "cost_effective": [True, True, True, True, False, False],
            "harmful": [False, False, False, False, True, False],
            "outcome_class": ["strong_positive"] * 4 + ["harmful", "mixed"],
        }
    )
    v5 = pd.DataFrame(
        {
            "experiment_family": ["240"] * 6,
            "triad_id": expert["triad_id"],
            "outcome_cohort": [
                "ROBUST_SAFE_DOUBLE_GATE",
                "LOCAL_PARETO_DOUBLE_GATE",
                "SECONDARY_CONTROLLED",
                "MIXED_OR_INCONCLUSIVE",
                "JOINTLY_HARMFUL",
                "MIXED_OR_INCONCLUSIVE",
            ],
            "run_id": [f"R{i}" for i in range(6)],
            "condition_id": [f"C{i}" for i in range(6)],
            "training_seed": range(6),
        }
    )
    result = build_unified_triad_outcomes(expert, v5).set_index("triad_id")
    assert result.loc["robust", "unified_outcome"] == "ROBUST_ABSOLUTE_GAIN"
    assert result.loc["local", "unified_outcome"] == "LOCAL_ABSOLUTE_PARETO"
    assert result.loc["secondary", "unified_outcome"] == "CONTROLLED_SECONDARY_GAIN"
    assert result.loc["relative", "unified_outcome"] == "RELATIVE_ONLY_WIN"
    assert result.loc["harmful", "unified_outcome"] == "JOINTLY_HARMFUL"
    assert result.loc["mixed", "unified_outcome"] == "MIXED_OR_INCONCLUSIVE"
    assert result.loc["local", "condition_id"] == "C1"
    assert result.loc["local", "training_seed"] == 1


def test_late_overfit_matches_frozen_formula() -> None:
    rows = []
    values = {
        "T": {121: 1.0, 200: 0.7},
        "R1": {121: 1.0, 200: 0.8},
        "R2": {121: 1.0, 200: 0.9},
    }
    for arm, epochs in values.items():
        for epoch, loss in epochs.items():
            rows.append(
                {
                    "triad_id": "T1",
                    "condition_id": "C1",
                    "training_seed": 7,
                    "arm": arm,
                    "epoch": epoch,
                    "train/loss": loss,
                    "metrics/accuracy_top1": 0.5 + epoch / 1000,
                    "val/loss": loss + 0.1,
                }
            )
    result = compute_late_overfit_features(
        pd.DataFrame(rows), start_epoch=121, cutoffs=(200,)
    )
    assert result.loc[0, "late_overfit"] == pytest.approx(0.15)


def test_leave_group_out_predictions_never_train_on_heldout_condition() -> None:
    rows = []
    for group_index in range(6):
        for replicate in range(2):
            rows.append(
                {
                    "triad_id": f"T{group_index}_{replicate}",
                    "condition_id": f"C{group_index}",
                    "training_seed": replicate,
                    "label": group_index % 2,
                    "late_overfit": float(group_index % 2) + replicate * 0.01,
                    "top1_shift": -float(group_index % 2),
                }
            )
    predictions, summary = leave_group_out_predictions(
        pd.DataFrame(rows),
        label_column="label",
        feature_columns=("late_overfit", "top1_shift"),
        group_column="condition_id",
        model_id="test",
    )
    assert predictions["leakage_free"].all()
    assert predictions["triad_id"].nunique() == 12
    assert 0.5 <= summary["auc"] <= 1.0


def test_controlled_cluster_regression_preserves_feature_direction() -> None:
    rows = []
    for condition in range(12):
        for seed in (1, 2):
            feature = condition / 10 + seed / 100
            rows.append(
                {
                    "condition_id": f"C{condition}",
                    "training_seed": seed,
                    "phase": "A" if condition < 6 else "B",
                    "budget": 600 if condition % 2 else 3000,
                    "late_overfit": feature,
                    "outcome": -2 * feature + condition / 100,
                }
            )
    result = controlled_cluster_regression(
        pd.DataFrame(rows),
        feature_column="late_overfit",
        outcome_column="outcome",
        cluster_column="condition_id",
        iterations=300,
    )
    assert result["standardized_coefficient"] < 0
    assert result["cluster_bootstrap_hi"] < 0
