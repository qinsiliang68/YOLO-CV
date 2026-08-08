from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.stage1_gapvalue240.analyze_goal_raw_frontier import (
    publish_tables_atomic,
    publish_val_cal_tables_atomic,
)
from stage1_gapvalue240.raw_frontier_analysis import (
    RawFrontierError,
    analyze_canonical_pairs,
    analyze_canonical_val_cal,
    build_control_reference,
    build_control_reference_from_index,
    compact_frontier,
    exact_np_frontier,
    load_canonical_prediction_index,
    platt_calibration_audit,
    paired_frontier_summary,
    paired_tail_shifts,
    probability_diagnostics,
    raw_calibrated_ranking_audit,
    read_val_op_predictions,
)


def _predictions(
    raw: list[float],
    calibrated: list[float] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["n1", "n2", "d1", "d2"],
            "y_true": [0, 0, 1, 1],
            "score": calibrated if calibrated is not None else raw,
            "score_raw": raw,
        }
    )


def _write_run(
    root: Path,
    *,
    run_slot: str,
    triad_id: str,
    arm: str,
    scores: list[float],
) -> dict[str, object]:
    package = "P01"
    attempt_id = f"attempt_{run_slot}_{arm}"
    attempt = (
        root
        / f"stage1_gapvalue240_{package}_upload"
        / "runs"
        / run_slot
        / attempt_id
    )
    prediction_path = attempt / "04_predictions" / "val_op_predictions.csv"
    prediction_path.parent.mkdir(parents=True)
    _predictions(scores).to_csv(prediction_path, index=False)
    return {
        "run_slot": run_slot,
        "triad_id": triad_id,
        "phase": "A",
        "condition_id": "A01",
        "method": "method",
        "budget": 600,
        "guard_ratio": 0.0,
        "arm": arm,
        "training_seed": 11,
        "selection_seed": 12,
        "package": package,
        "machine_id": "machine_01",
        "attempt_id": attempt_id,
        "resume_count": 0,
        "selection_sha256": "a" * 64,
        "release_ref": "release",
        "release_commit": "b" * 40,
        "input_snapshot_id": "c" * 64,
    }


def test_canonical_index_is_inventory_driven_and_requires_complete_triads(
    tmp_path: Path,
) -> None:
    rows = [
        _write_run(
            tmp_path,
            run_slot=f"RUN_{index:03d}",
            triad_id="TRIAD_001",
            arm=arm,
            scores=[0.1, 0.2, 0.8, 0.9],
        )
        for index, arm in enumerate(("T", "R1", "R2"), start=1)
    ]
    inventory = tmp_path / "inventory.csv"
    pd.DataFrame(rows).to_csv(inventory, index=False)

    index = load_canonical_prediction_index(
        inventory,
        tmp_path,
        expected_runs=3,
        expected_triads=1,
    )

    assert index["run_slot"].tolist() == ["RUN_001", "RUN_002", "RUN_003"]
    assert index["arm"].tolist() == ["T", "R1", "R2"]
    assert index["prediction_path"].map(lambda value: Path(value).exists()).all()

    invalid = pd.DataFrame(rows)
    invalid.loc[invalid.arm == "R2", "arm"] = "R1"
    invalid.to_csv(inventory, index=False)
    with pytest.raises(RawFrontierError, match="exactly one T, R1 and R2"):
        load_canonical_prediction_index(
            inventory,
            tmp_path,
            expected_runs=3,
            expected_triads=1,
        )


def test_probability_diagnostics_recomputes_both_score_spaces() -> None:
    predictions = _predictions(
        raw=[0.1, 0.4, 0.6, 0.9],
        calibrated=[0.05, 0.2, 0.8, 0.95],
    )

    diagnostics = probability_diagnostics(predictions, ece_bins=2).set_index(
        "score_type"
    )

    assert diagnostics.loc["raw", "auroc"] == pytest.approx(1.0)
    assert diagnostics.loc["raw", "auprc"] == pytest.approx(1.0)
    assert diagnostics.loc["raw", "brier"] == pytest.approx(0.085)
    assert diagnostics.loc["raw", "ece"] == pytest.approx(0.25)
    assert diagnostics.loc["calibrated", "brier"] == pytest.approx(0.02125)
    assert diagnostics.loc["calibrated", "ece"] == pytest.approx(0.125)
    expected_logloss = -np.mean(
        np.log([1 - 0.05, 1 - 0.2, 0.8, 0.95])
    )
    assert diagnostics.loc["calibrated", "log_loss"] == pytest.approx(
        expected_logloss
    )


def test_exact_np_frontier_uses_greater_equal_and_whole_tie_groups() -> None:
    predictions = pd.DataFrame(
        {
            "sample_id": ["d1", "d2", "n1", "n2"],
            "y_true": [1, 1, 0, 0],
            # At threshold 0.5, d2 and n1 must enter together under score >= t.
            "score_raw": [0.9, 0.5, 0.5, 0.1],
        }
    )

    frontier = exact_np_frontier(predictions, score_column="score_raw")

    assert frontier["fn_budget"].tolist() == [0, 1, 2]
    assert frontier["actual_fn"].tolist() == [0, 1, 2]
    assert frontier["TN"].tolist() == [1, 2, 2]
    at_zero = frontier.loc[frontier.fn_budget == 0].iloc[0]
    assert at_zero.threshold == pytest.approx(0.5)
    assert at_zero.tie_group_size == 2
    at_one = frontier.loc[frontier.fn_budget == 1].iloc[0]
    assert at_one.threshold == pytest.approx(0.9)

    compact = compact_frontier(frontier)
    assert compact["fn_budget"].tolist() == [0, 1, 2]
    assert compact["next_fn_budget_exclusive"].tolist() == [1, 2, 3]


def test_control_reference_is_exact_control_only_median_with_stable_tails(
    tmp_path: Path,
) -> None:
    paths: list[Path] = []
    for index, scores in enumerate(
        (
            [0.9, 0.2, 0.1, 0.8],
            [0.7, 0.4, 0.3, 0.6],
            [0.8, 0.3, 0.2, 0.7],
        )
    ):
        path = tmp_path / f"control_{index}.csv"
        _predictions(scores).to_csv(path, index=False)
        paths.append(path)

    reference = build_control_reference(
        paths,
        tn_target=1,
        fn_limit=1,
        normal_tail_fraction=0.5,
        defect_tail_fraction=0.5,
    ).set_index("sample_id")

    assert reference.loc["n1", "control_median_score_raw"] == pytest.approx(0.8)
    assert reference.loc["n2", "control_median_score_raw"] == pytest.approx(0.3)
    assert reference.loc["d1", "control_median_score_raw"] == pytest.approx(0.2)
    assert reference.loc["d2", "control_median_score_raw"] == pytest.approx(0.7)
    assert reference.loc["n1", "operational_tail"]
    assert reference.loc["d1", "operational_tail"]
    assert reference["control_run_count"].eq(3).all()
    assert reference["reference_source_arms"].eq("R1,R2").all()


def test_paired_tail_shifts_and_frontier_dominance_preserve_direction() -> None:
    control = _predictions(
        raw=[0.8, 0.4, 0.2, 0.7],
        calibrated=[0.7, 0.3, 0.3, 0.8],
    )
    treatment = _predictions(
        raw=[0.6, 0.3, 0.3, 0.8],
        calibrated=[0.5, 0.2, 0.4, 0.9],
    )
    reference = pd.DataFrame(
        {
            "sample_id": ["n1", "n2", "d1", "d2"],
            "y_true": [0, 0, 1, 1],
            "operational_tail": [True, False, True, False],
            "distribution_tail": [True, False, True, False],
        }
    )

    summaries, sample_shifts = paired_tail_shifts(
        treatment,
        control,
        reference,
    )

    assert sample_shifts.set_index("sample_id").loc["n1", "raw_shift"] == pytest.approx(
        -0.2
    )
    assert sample_shifts.set_index("sample_id").loc["d1", "raw_shift"] == pytest.approx(
        0.1
    )
    raw_operational = summaries[
        (summaries.scope == "operational") & (summaries.score_type == "raw")
    ]
    assert raw_operational["beneficial_rate"].eq(1.0).all()

    treatment_frontier = exact_np_frontier(treatment, score_column="score_raw")
    control_frontier = exact_np_frontier(control, score_column="score_raw")
    result = paired_frontier_summary(
        treatment_frontier,
        control_frontier,
        safe_fn_limit=1,
    )
    assert result["same_fn_budget_comparison"]
    assert result["safe_frontier_dominant"]
    assert result["delta_TN_at_FN0"] >= 0
    assert result["delta_TN_at_FN1"] >= 0
    assert math.isfinite(result["safe_mean_delta_TN"])


def test_prediction_reader_rejects_unreviewed_columns(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    frame = _predictions([0.1, 0.2, 0.8, 0.9])
    frame["unexpected"] = 1
    frame.to_csv(path, index=False)

    with pytest.raises(RawFrontierError, match="schema must be exactly"):
        read_val_op_predictions(
            path,
            expected_rows=4,
            expected_normal=2,
            expected_defect=2,
        )


def test_streamed_canonical_pair_analysis_covers_each_run_and_control(
    tmp_path: Path,
) -> None:
    rows = [
        _write_run(
            tmp_path,
            run_slot=f"RUN_{index:03d}",
            triad_id="TRIAD_001",
            arm=arm,
            scores=scores,
        )
        for index, (arm, scores) in enumerate(
            (
                ("T", [0.1, 0.2, 0.8, 0.9]),
                ("R1", [0.2, 0.3, 0.7, 0.8]),
                ("R2", [0.3, 0.4, 0.6, 0.7]),
            ),
            start=1,
        )
    ]
    inventory_path = tmp_path / "inventory.csv"
    pd.DataFrame(rows).to_csv(inventory_path, index=False)
    index = load_canonical_prediction_index(
        inventory_path,
        tmp_path,
        expected_runs=3,
        expected_triads=1,
    )
    reference = build_control_reference_from_index(
        index,
        expected_controls=2,
        tn_target=1,
        fn_limit=1,
        normal_tail_fraction=0.5,
        defect_tail_fraction=0.5,
        expected_rows=4,
        expected_normal=2,
        expected_defect=2,
    )
    sink_calls: list[tuple[str, str, int]] = []

    tables = analyze_canonical_pairs(
        index,
        reference,
        safe_fn_limit=1,
        expected_rows=4,
        expected_normal=2,
        expected_defect=2,
        frontier_sink=lambda run_slot, score_type, frame: sink_calls.append(
            (run_slot, score_type, len(frame))
        ),
    )

    assert len(tables["run_probability_metrics"]) == 6
    assert len(tables["frontier_equivalence_audit"]) == 3
    assert len(tables["paired_frontier_dominance"]) == 4
    assert len(tables["paired_tail_shift_summary"]) == 24
    assert set(tables["paired_frontier_dominance"]["control_arm"]) == {"R1", "R2"}
    assert tables["run_probability_metrics"]["input_snapshot_id"].eq("c" * 64).all()
    assert tables["paired_frontier_dominance"]["treatment_machine_id"].eq(
        "machine_01"
    ).all()
    assert tables["paired_frontier_dominance"]["control_machine_id"].eq(
        "machine_01"
    ).all()
    assert len(sink_calls) == 6

    leaked = reference.copy()
    leaked["reference_source_treatment_count"] = 1
    with pytest.raises(RawFrontierError, match="must contain zero treatment runs"):
        analyze_canonical_pairs(
            index,
            leaked,
            safe_fn_limit=1,
            expected_rows=4,
            expected_normal=2,
            expected_defect=2,
        )


def test_atomic_publisher_requires_inprogress_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    tables = {
        "raw_frontier_run_probability_metrics.csv": pd.DataFrame({"x": [1, 2]}),
        "raw_frontier_equivalence_audit.csv": pd.DataFrame({"x": [3]}),
        "raw_frontier_paired_dominance.csv": pd.DataFrame({"x": [4]}),
        "raw_frontier_paired_tail_shift_summary.csv": pd.DataFrame({"x": [5]}),
        "control_reference.csv": pd.DataFrame({"x": [6, 7, 8]}),
    }
    summary = {"status": "COMPLETE", "canonical_runs": 3}
    invalid = tmp_path / "report" / "tables"
    with pytest.raises(RawFrontierError, match=r"\.inprogress"):
        publish_tables_atomic(invalid, tables, summary, overwrite=False)

    output = tmp_path / "report.inprogress" / "tables"
    published = publish_tables_atomic(output, tables, summary, overwrite=False)

    assert set(published) == set(tables) | {"raw_frontier_analysis_summary.json"}
    assert pd.read_csv(output / "control_reference.csv").x.tolist() == [6, 7, 8]
    with pytest.raises(RawFrontierError, match="already exist"):
        publish_tables_atomic(output, tables, summary, overwrite=False)

    replacement = dict(tables)
    replacement["control_reference.csv"] = pd.DataFrame({"x": [9]})
    publish_tables_atomic(output, replacement, summary, overwrite=True)
    assert pd.read_csv(output / "control_reference.csv").x.tolist() == [9]
    assert not list(output.glob("*.tmp"))


def test_val_cal_ranking_and_platt_audits_recompute_saved_transform() -> None:
    raw = np.array([0.1, 0.4, 0.6, 0.9])
    predictions = _predictions(raw.tolist(), calibrated=raw.tolist())
    platt = {
        "coefficient": 1.0,
        "intercept": 0.0,
        "source_prevalence": 0.5,
        "deployment_prevalence": 0.5,
        "clip_low": 1e-7,
        "clip_high": 0.9999999,
    }

    calibration = platt_calibration_audit(predictions, platt)
    ranking = raw_calibrated_ranking_audit(predictions)

    assert calibration["coefficient_positive"]
    assert calibration["max_abs_transform_residual"] < 1e-12
    assert calibration["source_prevalence_abs_error"] == pytest.approx(0.0)
    assert ranking["monotonic_violation_count"] == 0
    assert ranking["raw_calibrated_frontier_exact"]

    reordered = predictions.copy()
    reordered["score"] = [0.1, 0.7, 0.6, 0.9]
    broken = raw_calibrated_ranking_audit(reordered)
    assert broken["monotonic_violation_count"] > 0
    assert not broken["monotonic_nondecreasing"]


def test_val_cal_atomic_publisher_has_separate_names(tmp_path: Path) -> None:
    output = tmp_path / "report.inprogress" / "tables"
    tables = {
        "val_cal_probability_metrics.csv": pd.DataFrame({"x": [1]}),
        "val_cal_ranking_frontier_audit.csv": pd.DataFrame({"x": [2]}),
        "val_cal_platt_audit.csv": pd.DataFrame({"x": [3]}),
    }

    published = publish_val_cal_tables_atomic(
        output,
        tables,
        {"status": "COMPLETE"},
        overwrite=False,
    )

    assert set(published) == set(tables) | {"val_cal_analysis_summary.json"}
    assert not (output / "raw_frontier_analysis_summary.json").exists()


def test_streamed_val_cal_analysis_covers_predictions_and_platt(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    parameters = {
        "coefficient": 1.0,
        "intercept": 0.0,
        "source_prevalence": 0.5,
        "deployment_prevalence": 0.5,
        "clip_low": 1e-7,
        "clip_high": 0.9999999,
    }
    for index, arm in enumerate(("T", "R1", "R2"), start=1):
        prediction = tmp_path / f"{arm}_val_cal.csv"
        platt = tmp_path / f"{arm}_platt.json"
        _predictions([0.1, 0.4, 0.6, 0.9]).to_csv(prediction, index=False)
        platt.write_text(json.dumps(parameters), encoding="utf-8")
        rows.append(
            {
                "run_slot": f"RUN_{index:03d}",
                "triad_id": "TRIAD_001",
                "arm": arm,
                "phase": "A",
                "condition_id": "A01",
                "method": "method",
                "budget": 600,
                "guard_ratio": 0.0,
                "training_seed": 11,
                "selection_seed": 12 + index,
                "machine_id": "machine_01",
                "resume_count": 0,
                "input_snapshot_id": "c" * 64,
                "val_cal_prediction_path": str(prediction),
                "platt_calibration_path": str(platt),
            }
        )

    tables = analyze_canonical_val_cal(
        pd.DataFrame(rows),
        expected_rows=4,
        expected_normal=2,
        expected_defect=2,
        ece_bins=2,
    )

    assert len(tables["val_cal_probability_metrics"]) == 6
    assert len(tables["val_cal_ranking_frontier_audit"]) == 3
    assert len(tables["val_cal_platt_audit"]) == 3
    assert tables["val_cal_ranking_frontier_audit"][
        "raw_calibrated_frontier_exact"
    ].all()
    assert tables["val_cal_platt_audit"][
        "transform_recomputed_exact_1e12"
    ].all()
