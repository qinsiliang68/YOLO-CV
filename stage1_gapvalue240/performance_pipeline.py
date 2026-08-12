"""Read-only orchestration for baseline and paired-control frontiers."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .deep_analysis import CanonicalInputError
from .performance_frontier import (
    REAL_GAIN_CLASSES,
    build_method_repeatability,
    classify_frontier_against_reference,
    frontier_from_predictions,
)


def _pair_record(
    *,
    experiment_family: str,
    comparison_id: str,
    treatment_id: str,
    control_id: str,
    control_label: str,
    treatment_frontier: pd.DataFrame,
    control_frontier: pd.DataFrame,
    baseline_fn: int,
    controlled_margins: Sequence[int] = (1, 2, 5),
) -> dict[str, Any]:
    result = classify_frontier_against_reference(
        treatment_frontier,
        control_frontier,
        baseline_fn=baseline_fn,
        controlled_margins=controlled_margins,
    )
    return {
        "experiment_family": experiment_family,
        "comparison_id": comparison_id,
        "treatment_id": treatment_id,
        "control_id": control_id,
        "control": control_label,
        **result,
        "real_gain": result["performance_class"] in REAL_GAIN_CLASSES,
    }


def build_240_control_gates(
    matrix: pd.DataFrame,
    frontiers: dict[str, pd.DataFrame],
    *,
    baseline_fn: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require every Treatment to beat both same-seed R1 and R2."""

    required = {"run_slot", "triad_id", "condition_id", "arm", "training_seed"}
    missing = sorted(required.difference(matrix.columns))
    if missing:
        raise CanonicalInputError(f"240 matrix missing columns: {missing}")
    pair_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for triad_id, group in matrix.groupby("triad_id", sort=True):
        if set(group["arm"].astype(str)) != {"T", "R1", "R2"} or len(group) != 3:
            raise CanonicalInputError(f"{triad_id} is not one T/R1/R2 triad")
        treatment = group.loc[group["arm"] == "T"].iloc[0]
        treatment_id = str(treatment["run_slot"])
        current: list[dict[str, Any]] = []
        for control_label in ("R1", "R2"):
            control = group.loc[group["arm"] == control_label].iloc[0]
            control_id = str(control["run_slot"])
            if treatment_id not in frontiers or control_id not in frontiers:
                raise CanonicalInputError(
                    f"Missing frontier for {treatment_id}/{control_id}"
                )
            record = _pair_record(
                experiment_family="240",
                comparison_id=str(triad_id),
                treatment_id=treatment_id,
                control_id=control_id,
                control_label=control_label,
                treatment_frontier=frontiers[treatment_id],
                control_frontier=frontiers[control_id],
                baseline_fn=baseline_fn,
            )
            current.append(record)
            pair_rows.append(record)
        pair_classes = [row["performance_class"] for row in current]
        gate_rows.append(
            {
                "experiment_family": "240",
                "triad_id": str(triad_id),
                "run_id": treatment_id,
                "condition_id": str(treatment["condition_id"]),
                "training_seed": int(treatment["training_seed"]),
                "paired_control_pass": bool(all(row["real_gain"] for row in current)),
                "paired_safe_frontier_pass": bool(
                    all(row["safe_frontier_dominant"] for row in current)
                ),
                "paired_full_frontier_pass": bool(
                    all(row["full_frontier_dominant"] for row in current)
                ),
                "paired_operating_harmful": bool(
                    all(row["delta_TN_at_baseline_fn"] < 0 for row in current)
                ),
                "min_control_delta_TN_at_baseline_fn": int(
                    min(row["delta_TN_at_baseline_fn"] for row in current)
                ),
                "max_control_delta_TN_at_baseline_fn": int(
                    max(row["delta_TN_at_baseline_fn"] for row in current)
                ),
                "all_controls_dominated": bool(
                    all(value == "DOMINATED" for value in pair_classes)
                ),
                "control_classes": "|".join(pair_classes),
            }
        )
    return pd.DataFrame(pair_rows), pd.DataFrame(gate_rows)


def _median_frontier(controls: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if len(controls) < 2:
        raise CanonicalInputError("Median control frontier requires at least two controls")
    ordered = sorted(controls)
    budgets = controls[ordered[0]]["fn_budget"].to_numpy(dtype=int)
    tn_values: list[np.ndarray] = []
    for name in ordered:
        frame = controls[name]
        if not np.array_equal(frame["fn_budget"].to_numpy(dtype=int), budgets):
            raise CanonicalInputError("Random-control frontier grids differ")
        tn_values.append(frame["TN"].to_numpy(dtype=float))
    median_tn = np.median(np.vstack(tn_values), axis=0).astype(int)
    normal_count = int(controls[ordered[0]]["TN"].iloc[-1] + controls[ordered[0]]["FP"].iloc[-1])
    return pd.DataFrame(
        {
            "fn_budget": budgets,
            "actual_fn": budgets,
            "TN": median_tn,
            "FP": normal_count - median_tn,
            "threshold": np.full(len(budgets), np.nan),
        }
    )


def build_120_control_gates(
    treatment_id: str,
    treatment_frontier: pd.DataFrame,
    controls: dict[str, pd.DataFrame],
    *,
    baseline_fn: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare one historical HN run with each RN and their median frontier."""

    if len(controls) < 2:
        raise CanonicalInputError(f"{treatment_id} has fewer than two RN controls")
    pair_rows: list[dict[str, Any]] = []
    for control_id, control_frontier in sorted(controls.items()):
        pair_rows.append(
            _pair_record(
                experiment_family="120",
                comparison_id=treatment_id,
                treatment_id=treatment_id,
                control_id=control_id,
                control_label=control_id,
                treatment_frontier=treatment_frontier,
                control_frontier=control_frontier,
                baseline_fn=baseline_fn,
                controlled_margins=(5, 10, 25),
            )
        )
    median_record = _pair_record(
        experiment_family="120",
        comparison_id=treatment_id,
        treatment_id=treatment_id,
        control_id="RN_MEDIAN",
        control_label="RN_MEDIAN",
        treatment_frontier=treatment_frontier,
        control_frontier=_median_frontier(controls),
        baseline_fn=baseline_fn,
        controlled_margins=(5, 10, 25),
    )
    pair_rows.append(median_record)
    individual = [row for row in pair_rows if row["control"] != "RN_MEDIAN"]
    wins = sum(bool(row["real_gain"]) for row in individual)
    median_delta = int(median_record["delta_TN_at_baseline_fn"])
    negative_individual = sum(
        int(row["delta_TN_at_baseline_fn"]) < 0 for row in individual
    )
    return pd.DataFrame(pair_rows), {
        "experiment_family": "120",
        "run_id": treatment_id,
        "condition_id": treatment_id,
        "training_seed": treatment_id,
        "wins_individual_controls": wins,
        "available_individual_controls": len(individual),
        "median_control_pass": bool(median_record["real_gain"]),
        "paired_control_pass": bool(wins >= 2 and median_record["real_gain"]),
        "paired_safe_frontier_pass": bool(
            sum(bool(row["safe_frontier_dominant"]) for row in individual) >= 2
            and median_record["safe_frontier_dominant"]
        ),
        "paired_full_frontier_pass": bool(
            sum(bool(row["full_frontier_dominant"]) for row in individual) >= 2
            and median_record["full_frontier_dominant"]
        ),
        "paired_operating_harmful": bool(
            negative_individual >= 2 and median_delta < 0
        ),
        "min_control_delta_TN_at_baseline_fn": int(
            min(row["delta_TN_at_baseline_fn"] for row in pair_rows)
        ),
        "max_control_delta_TN_at_baseline_fn": int(
            max(row["delta_TN_at_baseline_fn"] for row in pair_rows)
        ),
        "all_controls_dominated": bool(
            all(row["performance_class"] == "DOMINATED" for row in pair_rows)
        ),
        "control_classes": "|".join(row["performance_class"] for row in pair_rows),
    }


def classify_outcome_cohorts(rows: pd.DataFrame) -> pd.DataFrame:
    """Create outcome-first cohorts without blending absolute and paired gates."""

    required = {
        "run_id",
        "performance_class",
        "paired_control_pass",
        "all_controls_dominated",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise CanonicalInputError(f"Outcome gate table missing columns: {missing}")
    result = rows.copy()
    if "safe_frontier_dominant" not in result:
        result["safe_frontier_dominant"] = False
    if "paired_safe_frontier_pass" not in result:
        result["paired_safe_frontier_pass"] = False
    if "paired_operating_harmful" not in result:
        result["paired_operating_harmful"] = result["all_controls_dominated"].astype(bool)
    result["robust_double_gate_pass"] = (
        result["safe_frontier_dominant"].astype(bool)
        & result["paired_safe_frontier_pass"].astype(bool)
    )
    result["outcome_cohort"] = "MIXED_OR_INCONCLUSIVE"
    real = result["performance_class"].isin(REAL_GAIN_CLASSES)
    controlled = result["performance_class"].str.startswith("CONTROLLED_GAIN_")
    result.loc[real & result["paired_control_pass"].astype(bool), "outcome_cohort"] = (
        "LOCAL_PARETO_DOUBLE_GATE"
    )
    result.loc[
        real & result["paired_control_pass"].astype(bool) & result["robust_double_gate_pass"],
        "outcome_cohort",
    ] = (
        "ROBUST_SAFE_DOUBLE_GATE"
    )
    result.loc[real & ~result["paired_control_pass"].astype(bool), "outcome_cohort"] = (
        "ABSOLUTE_ONLY"
    )
    result.loc[controlled & result["paired_control_pass"].astype(bool), "outcome_cohort"] = (
        "SECONDARY_CONTROLLED"
    )
    result.loc[
        (pd.to_numeric(result["delta_TN_at_baseline_fn"], errors="coerce") < 0)
        & result["paired_operating_harmful"].astype(bool),
        "outcome_cohort",
    ] = "JOINTLY_HARMFUL"
    result.loc[
        (pd.to_numeric(result["delta_TN_at_baseline_fn"], errors="coerce") >= 0)
        & result["paired_operating_harmful"].astype(bool),
        "outcome_cohort",
    ] = "PAIRED_HARMFUL_ONLY"
    return result


def _read_predictions(
    path: Path,
    *,
    id_column: str,
    raw_column: str,
    calibrated_column: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise CanonicalInputError(f"Missing prediction file: {path}")
    frame = pd.read_csv(
        path,
        usecols=[id_column, "y_true", raw_column, calibrated_column],
        dtype={id_column: "string"},
    ).rename(
        columns={
            id_column: "sample_id",
            raw_column: "score_raw",
            calibrated_column: "score_calibrated",
        }
    )
    if (
        len(frame) != 120000
        or frame["sample_id"].nunique() != 120000
        or frame["y_true"].value_counts().to_dict() != {0: 100000, 1: 20000}
    ):
        raise CanonicalInputError(f"Prediction identity/count check failed: {path}")
    return frame


def _baseline_assets(baseline_root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    matches = sorted(
        baseline_root.glob("node-*/eval_1to5_full_yolo11l_*_best")
    )
    if len(matches) != 1:
        raise CanonicalInputError(
            f"Expected one yolo11l zero-replay baseline, found {len(matches)}"
        )
    run = matches[0]
    metrics_path = baseline_root / "metrics_summary.csv"
    metrics = pd.read_csv(metrics_path)
    metrics = metrics.loc[metrics["model"].astype(str) == "l"].copy()
    outputs: dict[str, pd.DataFrame] = {}
    selected_fn: dict[str, int] = {}
    for split in ("val_op", "test"):
        predictions = _read_predictions(
            run / f"predictions_{split}.csv",
            id_column="canonical_image_relpath",
            raw_column="p_defect_raw",
            calibrated_column="p_defect_operational",
        )
        raw = frontier_from_predictions(predictions, score_column="score_raw")
        calibrated = frontier_from_predictions(
            predictions, score_column="score_calibrated"
        )
        if not np.array_equal(raw["TN"].to_numpy(), calibrated["TN"].to_numpy()):
            raise CanonicalInputError(
                f"Baseline raw/calibrated frontiers differ for {split}"
            )
        outputs[split] = raw
        rows = metrics.loc[metrics["split"].astype(str) == split]
        if len(rows) != 1:
            raise CanonicalInputError(f"Baseline metrics missing unique {split} row")
        selected_fn[split] = int(rows.iloc[0]["fn"])
    return outputs, selected_fn


def _absolute_summary(
    *,
    experiment_family: str,
    run_id: str,
    split: str,
    frontier: pd.DataFrame,
    baseline: pd.DataFrame,
    baseline_fn: int,
    metadata: dict[str, Any],
    capability: str,
    controlled_margins: Sequence[int],
) -> dict[str, Any]:
    result = classify_frontier_against_reference(
        frontier,
        baseline,
        baseline_fn=baseline_fn,
        controlled_margins=controlled_margins,
    )
    return {
        "experiment_family": experiment_family,
        "run_id": run_id,
        "evaluation_split": split,
        "frontier_capability": capability,
        **metadata,
        **result,
    }


def _frontier_from_120_rows(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values("fn_budget", kind="stable")
    return pd.DataFrame(
        {
            "fn_budget": ordered["fn_budget"].astype(int),
            "actual_fn": ordered["FN_actual"].astype(int),
            "TN": ordered["TN_at_budget"].astype(int),
            "FP": 100000 - ordered["TN_at_budget"].astype(int),
            "threshold": ordered["threshold_for_budget"].astype(float),
        }
    ).reset_index(drop=True)


def _aggregate_mechanism_table(
    frame: pd.DataFrame,
    *,
    groups: Sequence[str],
    numeric_columns: Sequence[str],
    statistical_unit: str,
) -> pd.DataFrame:
    available = [column for column in numeric_columns if column in frame.columns]
    if not available:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(groups), dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = {column: value for column, value in zip(groups, keys)}
        record["row_count"] = len(group)
        record["statistical_unit"] = statistical_unit
        for column in available:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            record[f"mean_{column}"] = float(values.mean()) if len(values) else np.nan
            record[f"median_{column}"] = float(values.median()) if len(values) else np.nan
        records.append(record)
    return pd.DataFrame(records)


def _mechanism_tables(
    v3_report_dir: Path,
    triad_cohorts: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    table_root = v3_report_dir / "tables"
    mapping = triad_cohorts[["triad_id", "outcome_cohort"]].drop_duplicates()
    if mapping["triad_id"].duplicated().any():
        raise CanonicalInputError("Triad outcome cohort mapping is not unique")

    training = pd.read_csv(table_root / "training_window_features.csv")
    training = training.drop(columns=["cohort_code", "cohort_label"], errors="ignore")
    training = training.merge(mapping, on="triad_id", validate="many_to_one")
    training_summary = _aggregate_mechanism_table(
        training,
        groups=("outcome_cohort", "control"),
        numeric_columns=(
            "mean_delta_train_loss_e001_040",
            "mean_delta_train_loss_e041_120",
            "mean_delta_train_loss_e121_160",
            "mean_delta_train_loss_e161_200",
            "mean_delta_val_loss_e161_200",
            "mean_delta_top1_e161_200",
            "train_loss_extra_drop_epoch121_to_200",
            "val_loss_late_rebound",
        ),
        statistical_unit="triad_control_pair",
    )

    selection = pd.read_csv(table_root / "selection_run_operational_summary.csv")
    selection = selection.loc[(selection["arm"] == "T") & (selection["scope"] == "normal")]
    selection = selection.merge(mapping, on="triad_id", validate="many_to_one")
    selection_summary = _aggregate_mechanism_table(
        selection,
        groups=("outcome_cohort",),
        numeric_columns=(
            "mean_operational_error_rate",
            "mean_operational_forgetting_count",
            "mean_score_direction_changes",
            "mean_operational_correction",
            "mean_error_rate_late_161_200",
            "share_corrected",
            "share_persistent_wrong",
            "share_deteriorating",
        ),
        statistical_unit="treatment_selection_run",
    )

    tails = pd.read_csv(table_root / "prediction_tail_detail.csv")
    tails = tails.merge(mapping, on="triad_id", validate="many_to_one")
    tail_summary = _aggregate_mechanism_table(
        tails,
        groups=("outcome_cohort", "control", "label", "scope", "score_type"),
        numeric_columns=(
            "mean_shift",
            "median_shift",
            "beneficial_rate",
            "harmed_rate",
        ),
        statistical_unit="triad_control_tail",
    )

    composition = pd.read_csv(table_root / "treatment_selection_composition.csv")
    composition = composition.drop(columns=["cohort_code"], errors="ignore")
    composition = composition.merge(mapping, on="triad_id", validate="many_to_one")
    composition_summary = (
        composition.groupby(
            ["outcome_cohort", "y_true", "replay_role", "dynamic_bucket"],
            dropna=False,
            as_index=False,
        )["count"]
        .sum()
        .sort_values(["outcome_cohort", "y_true", "count"], ascending=[True, True, False])
    )
    totals = composition_summary.groupby(["outcome_cohort", "y_true"])["count"].transform("sum")
    composition_summary["share_within_cohort_label"] = composition_summary["count"] / totals
    return {
        "training_process_contrasts": training_summary,
        "selection_dynamic_contrasts": selection_summary,
        "prediction_tail_mechanism": tail_summary,
        "selection_composition_by_outcome": composition_summary,
    }


def _metric_from_group(
    frame: pd.DataFrame,
    *,
    cohort: str,
    control: str,
    column: str,
) -> float:
    rows = frame.loc[
        (frame["outcome_cohort"] == cohort) & (frame["control"] == control),
        column,
    ]
    return float(rows.iloc[0]) if len(rows) == 1 else float("nan")


def _build_hypothesis_registry(
    all_runs: pd.DataFrame,
    designed: pd.DataFrame,
    repeatability: pd.DataFrame,
    mechanisms: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    robust_240 = designed.loc[
        (designed["experiment_family"] == "240")
        & (designed["outcome_cohort"] == "ROBUST_SAFE_DOUBLE_GATE")
    ]
    local_240 = designed.loc[
        (designed["experiment_family"] == "240")
        & (designed["outcome_cohort"] == "LOCAL_PARETO_DOUBLE_GATE")
    ]
    harmful_240 = designed.loc[
        (designed["experiment_family"] == "240")
        & (designed["outcome_cohort"] == "JOINTLY_HARMFUL")
    ]
    robust_repeatable = repeatability.loc[
        repeatability["robust_repeatability_class"] == "REPEATABLE_ROBUST"
    ]
    training = mechanisms["training_process_contrasts"]
    tails = mechanisms["prediction_tail_mechanism"]
    local_late_r1 = _metric_from_group(
        training,
        cohort="LOCAL_PARETO_DOUBLE_GATE",
        control="R1",
        column="mean_train_loss_extra_drop_epoch121_to_200",
    )
    harmful_late_r1 = _metric_from_group(
        training,
        cohort="JOINTLY_HARMFUL",
        control="R1",
        column="mean_train_loss_extra_drop_epoch121_to_200",
    )
    defect_rows = tails.loc[
        (tails["label"] == "defect")
        & (tails["scope"] == "operational")
        & (tails["score_type"] == "raw")
    ]
    local_defect = _metric_from_group(
        defect_rows,
        cohort="LOCAL_PARETO_DOUBLE_GATE",
        control="R1",
        column="mean_mean_shift",
    )
    harmful_defect = _metric_from_group(
        defect_rows,
        cohort="JOINTLY_HARMFUL",
        control="R1",
        column="mean_mean_shift",
    )
    random_real = int(
        (
            (all_runs["experiment_family"] == "240")
            & all_runs["arm"].isin(["R1", "R2"])
            & all_runs["performance_class"].isin(REAL_GAIN_CLASSES)
        ).sum()
    )
    treatment_real = int(
        (
            (all_runs["experiment_family"] == "240")
            & (all_runs["arm"] == "T")
            & all_runs["performance_class"].isin(REAL_GAIN_CLASSES)
        ).sum()
    )
    return pd.DataFrame(
        [
            {
                "hypothesis_id": "H01",
                "hypothesis": "240-run contains a repeatably robust safe-frontier method",
                "status": "NOT_SUPPORTED",
                "evidence": f"robust Treatment runs={len(robust_240)}; repeatable robust conditions={len(robust_repeatable)}",
                "boundary": "Absolute baseline comparisons use a different seed; paired R1/R2 remain the attribution evidence.",
            },
            {
                "hypothesis_id": "H02",
                "hypothesis": "Some 240-run Treatments create real local Pareto improvement",
                "status": "SUPPORTED" if len(local_240) else "NOT_SUPPORTED",
                "evidence": f"local Pareto double-gate runs={len(local_240)}; operating-region harmful runs={len(harmful_240)}",
                "boundary": "Local Pareto gain is not whole-safe-range dominance and did not repeat across seeds.",
            },
            {
                "hypothesis_id": "H03",
                "hypothesis": "Late extra training-loss compression separates good and harmful 240 cohorts",
                "status": "SUPPORTED" if local_late_r1 < harmful_late_r1 else "INCONCLUSIVE",
                "evidence": f"R1 mean late extra drop: local={local_late_r1:.6g}, harmful={harmful_late_r1:.6g}",
                "boundary": "Outcome-defined exploratory contrast; local cohort has only three triads.",
            },
            {
                "hypothesis_id": "H04",
                "hypothesis": "Protecting the weakest defect tail distinguishes local gains from harmful outcomes",
                "status": "SUPPORTED" if local_defect > harmful_defect else "INCONCLUSIVE",
                "evidence": f"R1 operational weak-defect raw mean shift: local={local_defect:.6g}, harmful={harmful_defect:.6g}",
                "boundary": "Post-training mechanism evidence, not a ready-made training-time selection score.",
            },
            {
                "hypothesis_id": "H05",
                "hypothesis": "Fixed designed ranking clearly beats random replay in absolute baseline gains",
                "status": "NOT_SUPPORTED",
                "evidence": f"240 real-gain runs: Treatment={treatment_real}, random R1/R2={random_real}",
                "boundary": "Counts are descriptive because conditions and seeds are not exchangeable across all runs.",
            },
            {
                "hypothesis_id": "H06",
                "hypothesis": "Historical confidence-HN provides a repeatable robust training rule",
                "status": "NOT_SUPPORTED",
                "evidence": "40-run has one local Pareto HN and no robust HN; 120-run has three coarse robust points but no replicated seed confirmation.",
                "boundary": "120-run raw predictions are unavailable locally; its safe dominance is only audited at FN70..120 in steps of five.",
            },
        ]
    )


def run_performance_frontier_analysis(
    *,
    extracted_root: str | Path,
    inventory_path: str | Path,
    matrix_path: str | Path,
    baseline_root: str | Path,
    run40_root: str | Path,
    run120_root: str | Path,
    v3_report_dir: str | Path,
    output_dir: str | Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Analyze all currently available 240/40/120-run performance frontiers."""

    notify = progress or (lambda _message: None)
    extracted = Path(extracted_root).resolve()
    inventory_file = Path(inventory_path).resolve()
    matrix_file = Path(matrix_path).resolve()
    baseline_path = Path(baseline_root).resolve()
    run40_path = Path(run40_root).resolve()
    run120_path = Path(run120_root).resolve()
    v3_path = Path(v3_report_dir).resolve()
    output = Path(output_dir).resolve()
    for source in (extracted, baseline_path, run40_path, run120_path, v3_path):
        try:
            output.relative_to(source)
        except ValueError:
            continue
        raise CanonicalInputError(f"Output must not be inside read-only source: {source}")
    if output.exists() or output.with_name(output.name + ".inprogress").exists():
        raise FileExistsError(f"Refusing to overwrite analysis output: {output}")

    notify("Loading zero-replay yolo11l baseline")
    baselines, baseline_fn = _baseline_assets(baseline_path)
    baseline_tables = {
        f"baseline_frontier_{split}": frame.assign(evaluation_split=split)
        for split, frame in baselines.items()
    }
    inventory = pd.read_csv(inventory_file, dtype={"run_slot": str})
    matrix = pd.read_csv(matrix_file, dtype={"run_slot": str})
    if len(inventory) != 240 or len(matrix) != 240:
        raise CanonicalInputError("240-run inventory/matrix must each contain 240 rows")
    if set(inventory["run_slot"]) != set(matrix["run_slot"]):
        raise CanonicalInputError("240-run inventory and matrix run slots differ")
    matrix = matrix.merge(
        inventory[
            [
                "run_slot",
                "package",
                "attempt_id",
                "machine_id",
                "resume_count",
                "input_snapshot_id",
            ]
        ],
        on="run_slot",
        validate="one_to_one",
    )

    all_summaries: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    frontiers_240: dict[str, pd.DataFrame] = {}
    baseline_val_ids: set[str] | None = None
    for index, row in enumerate(matrix.sort_values("run_slot").itertuples(index=False), start=1):
        attempt = (
            extracted
            / f"stage1_gapvalue240_{row.package}_upload"
            / "runs"
            / str(row.run_slot)
            / str(row.attempt_id)
        )
        predictions = _read_predictions(
            attempt / "04_predictions/val_op_predictions.csv",
            id_column="sample_id",
            raw_column="score_raw",
            calibrated_column="score",
        )
        identities = set(predictions["sample_id"].astype(str))
        if baseline_val_ids is None:
            baseline_predictions = _read_predictions(
                next(baseline_path.glob("node-*/eval_1to5_full_yolo11l_*_best"))
                / "predictions_val_op.csv",
                id_column="canonical_image_relpath",
                raw_column="p_defect_raw",
                calibrated_column="p_defect_operational",
            )
            baseline_val_ids = set(baseline_predictions["sample_id"].astype(str))
        if identities != baseline_val_ids:
            raise CanonicalInputError(f"{row.run_slot} sample identities differ from baseline val_op")
        raw = frontier_from_predictions(predictions, score_column="score_raw")
        calibrated = frontier_from_predictions(predictions, score_column="score_calibrated")
        raw_tn = raw["TN"].to_numpy(dtype=int)
        calibrated_tn = calibrated["TN"].to_numpy(dtype=int)
        exact = bool(np.array_equal(raw_tn, calibrated_tn))
        frontiers_240[str(row.run_slot)] = raw
        all_summaries.append(
            _absolute_summary(
                experiment_family="240",
                run_id=str(row.run_slot),
                split="val_op",
                frontier=raw,
                baseline=baselines["val_op"],
                baseline_fn=baseline_fn["val_op"],
                metadata={
                    "condition_id": str(row.condition_id),
                    "method": str(row.method),
                    "budget": int(row.budget),
                    "arm": str(row.arm),
                    "training_seed": int(row.training_seed),
                    "machine_id": str(row.machine_id),
                    "resume_count": int(row.resume_count),
                    "input_snapshot_id": str(row.input_snapshot_id),
                    "source_path": str(attempt / "04_predictions/val_op_predictions.csv"),
                },
                capability="integer_fn_0_20000",
                controlled_margins=(1, 2, 5),
            )
        )
        calibration_rows.append(
            {
                "experiment_family": "240",
                "run_id": str(row.run_slot),
                "raw_calibrated_frontier_exact": exact,
                "differing_fn_budgets": int(np.count_nonzero(raw_tn != calibrated_tn)),
                "max_abs_delta_TN": int(np.max(np.abs(raw_tn - calibrated_tn))),
            }
        )
        if index % 20 == 0:
            notify(f"Processed {index}/240 GapValue runs")

    pairs_240, gates_240 = build_240_control_gates(
        matrix, frontiers_240, baseline_fn=baseline_fn["val_op"]
    )

    notify("Loading 40-run HN/RN full predictions")
    integrity = pd.read_csv(run40_path / "local_integrity_review_20260630.csv")
    if len(integrity) != 40 or set(integrity["run"].astype(str)) != {
        *(f"HN-{index:02d}" for index in range(1, 21)),
        *(f"RN-{index:02d}" for index in range(1, 21)),
    }:
        raise CanonicalInputError("40-run integrity table is incomplete")
    frontiers_40: dict[str, pd.DataFrame] = {}
    baseline_test_predictions = _read_predictions(
        next(baseline_path.glob("node-*/eval_1to5_full_yolo11l_*_best"))
        / "predictions_test.csv",
        id_column="canonical_image_relpath",
        raw_column="p_defect_raw",
        calibrated_column="p_defect_operational",
    )
    baseline_test_ids = set(baseline_test_predictions["sample_id"].astype(str))
    for index, row in enumerate(integrity.sort_values("run").itertuples(index=False), start=1):
        prediction_path = Path(str(row.material_dir)) / "eval" / "predictions_test.csv"
        predictions = _read_predictions(
            prediction_path,
            id_column="canonical_image_relpath",
            raw_column="p_defect_raw",
            calibrated_column="p_defect_operational",
        )
        if set(predictions["sample_id"].astype(str)) != baseline_test_ids:
            raise CanonicalInputError(f"{row.run} sample identities differ from baseline test")
        raw = frontier_from_predictions(predictions, score_column="score_raw")
        calibrated = frontier_from_predictions(predictions, score_column="score_calibrated")
        raw_tn = raw["TN"].to_numpy(dtype=int)
        calibrated_tn = calibrated["TN"].to_numpy(dtype=int)
        exact = bool(np.array_equal(raw_tn, calibrated_tn))
        frontiers_40[str(row.run)] = raw
        all_summaries.append(
            _absolute_summary(
                experiment_family="40",
                run_id=str(row.run),
                split="development_benchmark_120k",
                frontier=raw,
                baseline=baselines["test"],
                baseline_fn=baseline_fn["test"],
                metadata={
                    "condition_id": f"{row.group}_Q{int(row.q_percent):02d}_B{int(row.selected_unique)}",
                    "method": "Confidence-HN" if str(row.group) == "HN" else "Random-Normal",
                    "budget": int(row.selected_unique),
                    "arm": str(row.group),
                    "training_seed": 20260606,
                    "machine_id": Path(str(row.material_dir)).parts[-3],
                    "resume_count": 0,
                    "input_snapshot_id": "stage1_phase1_hn_rn_20260623",
                    "source_path": str(prediction_path),
                },
                capability="integer_fn_0_20000",
                controlled_margins=(1, 2, 5),
            )
        )
        calibration_rows.append(
            {
                "experiment_family": "40",
                "run_id": str(row.run),
                "raw_calibrated_frontier_exact": exact,
                "differing_fn_budgets": int(np.count_nonzero(raw_tn != calibrated_tn)),
                "max_abs_delta_TN": int(np.max(np.abs(raw_tn - calibrated_tn))),
            }
        )
        if index % 10 == 0:
            notify(f"Processed {index}/40 HN/RN runs")

    pair_rows_40: list[dict[str, Any]] = []
    gate_rows_40: list[dict[str, Any]] = []
    for q in range(1, 21):
        treatment_id = f"HN-{q:02d}"
        control_id = f"RN-{q:02d}"
        pair = _pair_record(
            experiment_family="40",
            comparison_id=f"Q{q:02d}",
            treatment_id=treatment_id,
            control_id=control_id,
            control_label="RN",
            treatment_frontier=frontiers_40[treatment_id],
            control_frontier=frontiers_40[control_id],
            baseline_fn=baseline_fn["test"],
        )
        pair_rows_40.append(pair)
        gate_rows_40.append(
            {
                "experiment_family": "40",
                "triad_id": f"Q{q:02d}",
                "run_id": treatment_id,
                "condition_id": f"HN_Q{q:02d}_B{q * 600}",
                "training_seed": 20260606,
                "paired_control_pass": bool(pair["real_gain"]),
                "paired_safe_frontier_pass": bool(pair["safe_frontier_dominant"]),
                "paired_full_frontier_pass": bool(pair["full_frontier_dominant"]),
                "paired_operating_harmful": bool(
                    pair["delta_TN_at_baseline_fn"] < 0
                ),
                "min_control_delta_TN_at_baseline_fn": int(
                    pair["delta_TN_at_baseline_fn"]
                ),
                "max_control_delta_TN_at_baseline_fn": int(
                    pair["delta_TN_at_baseline_fn"]
                ),
                "all_controls_dominated": pair["performance_class"] == "DOMINATED",
                "control_classes": pair["performance_class"],
            }
        )
    pairs_40 = pd.DataFrame(pair_rows_40)
    gates_40 = pd.DataFrame(gate_rows_40)

    notify("Loading 120-run audited coarse frontiers")
    dist_root = run120_path / "score_distribution_120runs_20260707"
    frontier_120_rows = pd.read_csv(
        dist_root / "run_probability_distribution_tail_tradeoff_120k.csv"
    )
    summary_120 = pd.read_csv(
        dist_root / "run_probability_distribution_summary_120k_valid_only.csv"
    )
    if summary_120["run_id"].nunique() != 111:
        raise CanonicalInputError("Expected 111 valid historical 120-run results")
    budgets_120 = sorted(frontier_120_rows["fn_budget"].astype(int).unique())
    if budgets_120 != list(range(70, 121, 5)):
        raise CanonicalInputError(f"Unexpected 120-run FN grid: {budgets_120}")
    baseline_120 = baselines["test"].loc[
        baselines["test"]["fn_budget"].isin(budgets_120)
    ].reset_index(drop=True)
    frontiers_120: dict[str, pd.DataFrame] = {}
    summary_lookup = summary_120.set_index("run_id")
    for run_id, group in frontier_120_rows.groupby("run_id", sort=True):
        frontier = _frontier_from_120_rows(group)
        frontiers_120[str(run_id)] = frontier
        meta = summary_lookup.loc[str(run_id)]
        all_summaries.append(
            _absolute_summary(
                experiment_family="120",
                run_id=str(run_id),
                split="development_benchmark_120k",
                frontier=frontier,
                baseline=baseline_120,
                baseline_fn=baseline_fn["test"],
                metadata={
                    "condition_id": str(run_id),
                    "method": "Confidence-HN" if str(meta["family"]) == "HN" else "Random-Normal",
                    "budget": int(str(run_id).split("-")[-1]),
                    "arm": str(meta["family"]),
                    "training_seed": np.nan,
                    "machine_id": np.nan,
                    "resume_count": np.nan,
                    "input_snapshot_id": "stage1_phase1_hn_band_20260628_120runs",
                    "source_path": "audited_5_fn_step_frontier_table",
                },
                capability="coarse_fn_70_120_step5",
                controlled_margins=(5, 10, 25),
            )
        )
        calibration_rows.append(
            {
                "experiment_family": "120",
                "run_id": str(run_id),
                "raw_calibrated_frontier_exact": np.nan,
                "differing_fn_budgets": np.nan,
                "max_abs_delta_TN": np.nan,
            }
        )

    expected_120 = {
        *(f"HN1-{index:02d}" for index in range(1, 21)),
        *(f"HN2-{index:02d}" for index in range(1, 11)),
        *(f"RN1{letter}-{index:02d}" for letter in "ABC" for index in range(1, 21)),
        *(f"RN2{letter}-{index:02d}" for letter in "ABC" for index in range(1, 11)),
    }
    missing_120 = sorted(expected_120.difference(frontiers_120))
    if len(missing_120) != 9:
        raise CanonicalInputError(f"Expected 9 missing historical runs, found {missing_120}")

    pair_frames_120: list[pd.DataFrame] = []
    gate_rows_120: list[dict[str, Any]] = []
    for group_name, maximum in (("HN1", 20), ("HN2", 10)):
        rn_prefix = "RN1" if group_name == "HN1" else "RN2"
        for index in range(1, maximum + 1):
            treatment_id = f"{group_name}-{index:02d}"
            if treatment_id not in frontiers_120:
                continue
            controls = {
                run_id: frontiers_120[run_id]
                for run_id in (f"{rn_prefix}{letter}-{index:02d}" for letter in "ABC")
                if run_id in frontiers_120
            }
            if len(controls) < 2:
                continue
            pairs, gate = build_120_control_gates(
                treatment_id,
                frontiers_120[treatment_id],
                controls,
                baseline_fn=baseline_fn["test"],
            )
            pair_frames_120.append(pairs)
            gate_rows_120.append(gate)
    pairs_120 = pd.concat(pair_frames_120, ignore_index=True)
    gates_120 = pd.DataFrame(gate_rows_120)

    all_runs = pd.DataFrame(all_summaries)
    designed_gates = pd.concat([gates_240, gates_40, gates_120], ignore_index=True)
    designed = designed_gates.merge(
        all_runs[
            [
                "experiment_family",
                "run_id",
                "performance_class",
                "absolute_baseline_pass",
                "delta_TN_at_baseline_fn",
                "safe_min_delta_TN",
                "safe_max_delta_TN",
                "safe_positive_budget_share",
                "safe_negative_budget_share",
                "safe_frontier_dominant",
                "full_frontier_dominant",
            ]
        ],
        on=["experiment_family", "run_id"],
        validate="one_to_one",
    )
    designed = classify_outcome_cohorts(designed)
    repeatability = build_method_repeatability(
        designed[
            [
                "experiment_family",
                "condition_id",
                "training_seed",
                "absolute_baseline_pass",
                "paired_control_pass",
                "safe_frontier_dominant",
                "paired_safe_frontier_pass",
            ]
        ]
    )
    pairs = pd.concat([pairs_240, pairs_40, pairs_120], ignore_index=True)
    mechanisms = _mechanism_tables(
        v3_path,
        designed.loc[designed["experiment_family"] == "240", ["triad_id", "outcome_cohort"]],
    )
    hypotheses = _build_hypothesis_registry(
        all_runs,
        designed,
        repeatability,
        mechanisms,
    )

    missing_registry = pd.DataFrame(
        [
            {
                "experiment_family": "120",
                "run_id": run_id,
                "status": "MISSING_RAW_AND_FRONTIER",
                "included": False,
            }
            for run_id in missing_120
        ]
    )
    tables: dict[str, pd.DataFrame] = {
        **baseline_tables,
        "all_run_baseline_dominance": all_runs,
        "paired_control_frontier_deltas": pairs,
        "designed_method_double_gates": designed,
        "method_repeatability_ranking": repeatability,
        "strong_secondary_harmful_cohorts": designed,
        "raw_calibrated_frontier_audit": pd.DataFrame(calibration_rows),
        "missing_and_confounded_registry": missing_registry,
        "hypothesis_registry": hypotheses,
        **mechanisms,
    }
    metadata = {
        "analysis_id": "gapvalue_performance_frontier_analysis_20260801_v5",
        "candidate_runs": len(all_runs),
        "run_counts": all_runs.groupby("experiment_family")["run_id"].nunique().to_dict(),
        "canonical_240_runs": len(frontiers_240),
        "complete_240_triads": gates_240["triad_id"].nunique(),
        "complete_40_pairs": len(gates_40),
        "valid_120_runs": len(frontiers_120),
        "missing_120_runs": missing_120,
        "baseline_fn": baseline_fn,
        "comparison_rule": "same-FN tie-safe frontier only",
        "double_gate_rule": "absolute zero-replay baseline plus paired random controls",
        "scientific_boundaries": [
            "240 absolute baseline comparison is seed-confounded descriptive evidence",
            "240 method attribution requires separate R1 and R2 success",
            "40 HN/RN and zero-replay baseline share seed 20260606",
            "120 frontiers are limited to FN70..120 in increments of five",
            "development/val_op discovery only; no blind or external claim",
        ],
        "source_paths": {
            "extracted_root": str(extracted),
            "inventory": str(inventory_file),
            "matrix": str(matrix_file),
            "baseline_root": str(baseline_path),
            "run40_root": str(run40_path),
            "run120_root": str(run120_path),
            "v3_report_dir": str(v3_path),
        },
    }
    from .performance_reporting import build_performance_report

    report_path = build_performance_report(output, tables=tables, metadata=metadata)
    return {
        "status": "PASS",
        "output_dir": str(report_path),
        **{key: metadata[key] for key in ("candidate_runs", "run_counts", "complete_240_triads", "complete_40_pairs", "valid_120_runs")},
    }
