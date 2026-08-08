from __future__ import annotations

import pandas as pd

from stage1_gapvalue240.training_dynamics_patterns import (
    audit_learning_rate_groups,
    build_divergence_timeline,
    build_paired_epoch_deltas,
)


def _curves() -> pd.DataFrame:
    rows = []
    for triad, cohort, offset in (
        ("GOOD", "DUAL_IMPROVEMENT", 0.0),
        ("HARM", "DUAL_HARM", 0.0),
    ):
        for arm in ("T", "R1", "R2"):
            for epoch in (1, 121, 150, 200):
                control_loss = 1.0 - 0.001 * epoch
                if arm == "T" and triad == "HARM" and epoch > 121:
                    loss = control_loss - 0.01 * (epoch - 121) / 79
                else:
                    loss = control_loss
                rows.append(
                    {
                        "run_slot": f"{triad}_{arm}",
                        "triad_id": triad,
                        "arm": arm,
                        "epoch": epoch,
                        "train/loss": loss,
                        "val/loss": 0.5,
                        "metrics/accuracy_top1": 0.9,
                        "metrics/accuracy_top5": 1.0,
                        **{f"lr/pg{i}": 0.01 - epoch * 0.00001 for i in range(8)},
                    }
                )
    return pd.DataFrame(rows)


def test_paired_epoch_deltas_measure_late_extra_fitting() -> None:
    outcomes = pd.DataFrame(
        {
            "triad_id": ["GOOD", "HARM"],
            "exclusive_cohort": ["DUAL_IMPROVEMENT", "DUAL_HARM"],
            "training_seed": [1, 2],
            "condition_slot": ["A01", "A02"],
        }
    )

    paired = build_paired_epoch_deltas(_curves(), outcomes, baseline_epoch=121)
    final = paired.query("epoch == 200").set_index(["triad_id", "control_arm"])

    assert final.loc[("GOOD", "R1"), "extra_train_loss_decline"] == 0
    assert abs(final.loc[("HARM", "R2"), "extra_train_loss_decline"] - 0.01) < 1e-12
    assert paired.loc[paired["epoch"] < 121, "extra_train_loss_decline"].isna().all()
    assert len(paired) == 16


def test_divergence_timeline_uses_one_row_per_triad_epoch() -> None:
    outcomes = pd.DataFrame(
        {
            "triad_id": ["GOOD", "HARM"],
            "exclusive_cohort": ["DUAL_IMPROVEMENT", "DUAL_HARM"],
            "training_seed": [1, 2],
            "condition_slot": ["A01", "A02"],
        }
    )
    paired = build_paired_epoch_deltas(_curves(), outcomes, baseline_epoch=121)

    timeline = build_divergence_timeline(paired)
    row = timeline.loc[
        (timeline["epoch"] == 200)
        & (timeline["feature"] == "extra_train_loss_decline")
    ].iloc[0]

    assert row["positive_n"] == 1
    assert row["negative_n"] == 1
    assert row["harm_prediction_auc"] == 1.0


def test_lr_audit_marks_active_and_scheduler_only_groups() -> None:
    curves = _curves()
    curves["budget"] = 600
    optimizer = pd.DataFrame(
        [
            {"run_slot": slot, "group_index": index, "active": index in {3, 5, 7}}
            for slot in curves["run_slot"].unique()
            for index in range(8)
        ]
    )
    metadata = pd.DataFrame(
        {
            "run_slot": curves["run_slot"].unique(),
            "budget": [600] * curves["run_slot"].nunique(),
        }
    )

    audit = audit_learning_rate_groups(curves, optimizer, metadata)

    assert set(audit.loc[audit["active_all_runs"], "group_index"]) == {3, 5, 7}
    assert set(audit.loc[audit["scheduler_only_all_runs"], "group_index"]) == {
        0,
        1,
        2,
        4,
        6,
    }
    assert audit["max_abs_second_difference_after_120"].max() < 1e-12
