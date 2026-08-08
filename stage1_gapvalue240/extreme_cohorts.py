"""Focused extreme-cohort analysis for the frozen GapValue 240-run study.

This module is intentionally read-only.  It consumes canonical v2 tables,
frozen selection manifests, and the audited 200 x 120,000 OOF probability
matrix.  It never changes the experiment matrix, selections, or run outputs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .deep_analysis import CanonicalInputError


COHORT_LABELS: dict[str, str] = {
    "S": "exceptional_safe_high_yield",
    "A": "safe_positive_below_300",
    "B": "high_yield_near_safe",
    "M": "mixed_or_inconsistent",
    "H": "jointly_harmful",
}

_TRIAD_METADATA = (
    "phase",
    "condition_slot",
    "condition_id",
    "method",
    "budget",
    "guard_ratio",
    "training_seed",
    "discovery_or_confirmation",
)

DEFAULT_WINDOWS: tuple[tuple[int, int, str], ...] = (
    (1, 40, "early_001_040"),
    (41, 120, "middle_041_120"),
    (121, 160, "late_121_160"),
    (161, 200, "late_161_200"),
)


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], *, context: str
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise CanonicalInputError(f"{context} missing columns: {missing}")


def _one_value(group: pd.DataFrame, column: str, *, context: str) -> Any:
    values = group[column].drop_duplicates()
    if len(values) != 1:
        raise CanonicalInputError(
            f"{context} has inconsistent {column}: {values.astype(str).tolist()}"
        )
    return values.iloc[0]


def classify_triad_cohorts(triad_deltas: pd.DataFrame) -> pd.DataFrame:
    """Assign every triad to one of five mutually-exclusive result cohorts.

    The two original overlapping expert definitions are retained as boolean
    evidence columns.  ``cohort_code`` makes them mutually exclusive:

    - S: strong-positive and high-value;
    - A: strong-positive only (safe, but <300 TN on at least one control);
    - B: high-value only (up to +2 FN allowed);
    - H: harmful against both controls;
    - M: every remaining mixed or inconsistent result.
    """

    required = {
        "triad_id",
        "control",
        "delta_TN",
        "delta_FN",
        *_TRIAD_METADATA,
    }
    _require_columns(triad_deltas, required, context="triad deltas")
    if triad_deltas.empty:
        raise CanonicalInputError("triad deltas are empty")

    duplicated = triad_deltas.duplicated(["triad_id", "control"], keep=False)
    control_sets = triad_deltas.groupby("triad_id", sort=False)["control"].agg(
        lambda values: tuple(sorted(str(v) for v in values))
    )
    if duplicated.any() or not control_sets.map(lambda x: x == ("R1", "R2")).all():
        raise CanonicalInputError("Each triad must contain exactly one R1 and one R2 row")

    rows: list[dict[str, Any]] = []
    for triad_id, group in triad_deltas.groupby("triad_id", sort=True):
        metadata = {
            column: _one_value(group, column, context=str(triad_id))
            for column in _TRIAD_METADATA
        }
        by_control = group.set_index("control")
        r1_tn = float(by_control.loc["R1", "delta_TN"])
        r1_fn = float(by_control.loc["R1", "delta_FN"])
        r2_tn = float(by_control.loc["R2", "delta_TN"])
        r2_fn = float(by_control.loc["R2", "delta_FN"])

        strong = r1_tn > 0 and r1_fn <= 0 and r2_tn > 0 and r2_fn <= 0
        high_value = (
            r1_tn >= 300 and r2_tn >= 300 and r1_fn <= 2 and r2_fn <= 2
        )
        harmful = r1_tn < 0 and r1_fn > 0 and r2_tn < 0 and r2_fn > 0
        if strong and high_value:
            code = "S"
            reason = "both controls: delta_TN>=300 and delta_FN<=0"
        elif strong:
            code = "A"
            reason = "both controls improve safely; at least one delta_TN<300"
        elif high_value:
            code = "B"
            reason = "both controls delta_TN>=300; delta_FN allowed up to +2"
        elif harmful:
            code = "H"
            reason = "both controls: delta_TN<0 and delta_FN>0"
        else:
            code = "M"
            reason = "control directions or TN/FN objectives are mixed"

        rows.append(
            {
                "triad_id": triad_id,
                **metadata,
                "delta_TN_R1": r1_tn,
                "delta_FN_R1": r1_fn,
                "delta_TN_R2": r2_tn,
                "delta_FN_R2": r2_fn,
                "strong_positive": bool(strong),
                "high_value": bool(high_value),
                "harmful": bool(harmful),
                "cohort_code": code,
                "cohort_label": COHORT_LABELS[code],
                "cohort_reason": reason,
            }
        )
        record = rows[-1]
        for column in (
            "machine_pair",
            "same_machine",
            "any_resumed",
            "control_machine_id",
            "control_resume_count",
            "control_input_snapshot_id",
        ):
            if column in group.columns:
                record[f"{column}_R1"] = by_control.loc["R1", column]
                record[f"{column}_R2"] = by_control.loc["R2", column]
        for column in (
            "t_run_slot",
            "t_machine_id",
            "t_resume_count",
            "t_input_snapshot_id",
        ):
            if column in group.columns:
                record[column] = _one_value(group, column, context=str(triad_id))
        if "same_machine" in group.columns:
            record["all_same_machine"] = bool(
                by_control.loc["R1", "same_machine"]
                and by_control.loc["R2", "same_machine"]
            )
        if "any_resumed" in group.columns:
            record["any_arm_resumed"] = bool(
                by_control.loc["R1", "any_resumed"]
                or by_control.loc["R2", "any_resumed"]
            )
    result = pd.DataFrame(rows)
    if result["cohort_code"].isna().any() or result["triad_id"].duplicated().any():
        raise AssertionError("Cohort assignment must be complete and unique")
    return result.sort_values("triad_id", kind="stable").reset_index(drop=True)


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or not np.isfinite(y).all():
        return float("nan")
    return float(np.polyfit(x.astype(float), y.astype(float), 1)[0])


def build_training_window_features(
    paired_epoch_differences: pd.DataFrame,
    cohorts: pd.DataFrame,
) -> pd.DataFrame:
    """Reduce each 200-epoch T-control curve to preregistered window features."""

    required = {
        "triad_id",
        "control",
        "epoch",
        "delta_train_loss",
        "delta_val_loss",
        "delta_top1",
        "condition_slot",
        "condition_id",
        "phase",
        "training_seed",
        "machine_pair",
        "any_resumed",
    }
    _require_columns(
        paired_epoch_differences, required, context="paired epoch differences"
    )
    _require_columns(
        cohorts, {"triad_id", "cohort_code", "cohort_label"}, context="cohorts"
    )

    expected_epochs = tuple(range(1, 201))
    rows: list[dict[str, Any]] = []
    for (triad_id, control), group in paired_epoch_differences.groupby(
        ["triad_id", "control"], sort=True
    ):
        group = group.sort_values("epoch", kind="stable")
        epochs = tuple(pd.to_numeric(group["epoch"], errors="raise").astype(int))
        if epochs != expected_epochs:
            raise CanonicalInputError(
                f"{triad_id}/{control} must contain epochs 1..200 exactly"
            )
        record: dict[str, Any] = {
            "triad_id": triad_id,
            "control": control,
        }
        for column in (
            "condition_slot",
            "condition_id",
            "phase",
            "training_seed",
            "machine_pair",
            "any_resumed",
        ):
            record[column] = _one_value(
                group, column, context=f"{triad_id}/{control}"
            )

        epoch_values = group["epoch"].to_numpy(dtype=float)
        for start, end, label in DEFAULT_WINDOWS:
            window = group.loc[group["epoch"].between(start, end)]
            for metric in ("delta_train_loss", "delta_val_loss", "delta_top1"):
                values = pd.to_numeric(window[metric], errors="coerce")
                if values.isna().any():
                    raise CanonicalInputError(
                        f"{triad_id}/{control}/{metric} contains non-numeric values"
                    )
                short = metric.removeprefix("delta_")
                record[f"mean_delta_{short}_e{start:03d}_{end:03d}"] = float(
                    values.mean()
                )

        late = group.loc[group["epoch"].between(121, 200)]
        for metric in ("delta_train_loss", "delta_val_loss", "delta_top1"):
            short = metric.removeprefix("delta_")
            record[f"{short}_slope_121_200"] = _linear_slope(
                late["epoch"].to_numpy(dtype=float),
                late[metric].to_numpy(dtype=float),
            )

        indexed = group.set_index("epoch")
        record["train_loss_extra_drop_epoch121_to_200"] = float(
            indexed.loc[121, "delta_train_loss"]
            - indexed.loc[200, "delta_train_loss"]
        )
        record["train_loss_robust_drop_121_130_to_191_200"] = float(
            indexed.loc[121:130, "delta_train_loss"].mean()
            - indexed.loc[191:200, "delta_train_loss"].mean()
        )
        late_val = indexed.loc[121:200, "delta_val_loss"]
        record["val_loss_late_rebound"] = float(
            indexed.loc[191:200, "delta_val_loss"].mean() - late_val.min()
        )
        rows.append(record)

    result = pd.DataFrame(rows).merge(
        cohorts[
            [
                column
                for column in (
                    "triad_id",
                    "cohort_code",
                    "cohort_label",
                    "method",
                    "budget",
                    "guard_ratio",
                    "discovery_or_confirmation",
                )
                if column in cohorts.columns
            ]
        ],
        on="triad_id",
        how="left",
        validate="many_to_one",
    )
    if result["cohort_code"].isna().any():
        raise CanonicalInputError("Paired epoch rows contain triads absent from cohorts")
    return result.sort_values(["triad_id", "control"], kind="stable").reset_index(
        drop=True
    )


def _normalized_fold(values: pd.Series) -> np.ndarray:
    normalized = values.astype(str).str.strip().str.replace(r"^0+(?=\d)", "", regex=True)
    return normalized.to_numpy(dtype=str)


def _trajectory_type(early: np.ndarray, late: np.ndarray) -> np.ndarray:
    result = np.full(len(early), "transitional", dtype=object)
    result[(early >= 0.5) & (late >= 0.5)] = "persistent_wrong"
    result[(early >= 0.5) & (late <= 0.1)] = "corrected"
    result[(early <= 0.1) & (late >= 0.5)] = "deteriorating"
    result[(early <= 0.1) & (late <= 0.1)] = "stable_correct"
    return result


def compute_operational_sample_dynamics(
    probabilities: np.ndarray,
    sample_index: pd.DataFrame,
    *,
    fn_limit: int = 285,
    repaired_fold: int | str = 1,
    repaired_epoch: int = 178,
    windows: Sequence[tuple[int, int, str]] = DEFAULT_WINDOWS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute per-sample dynamics at each epoch's tie-safe FN work point.

    The repaired fold/epoch cell is excluded only for affected fold samples.
    Other folds retain the valid epoch, avoiding a global epoch deletion.
    """

    _require_columns(
        sample_index, {"sample_id", "y_true", "oof_fold"}, context="OOF sample index"
    )
    if probabilities.ndim != 2:
        raise CanonicalInputError("OOF probabilities must be a 2-D epoch x sample array")
    epoch_count, sample_count = probabilities.shape
    if sample_count != len(sample_index):
        raise CanonicalInputError(
            f"OOF sample count mismatch: matrix={sample_count}, index={len(sample_index)}"
        )
    if sample_index["sample_id"].duplicated().any():
        raise CanonicalInputError("OOF sample IDs must be unique")
    y_true = pd.to_numeric(sample_index["y_true"], errors="raise").to_numpy(
        dtype=np.int8
    )
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise CanonicalInputError("OOF labels must be binary 0/1")
    defect_mask = y_true == 1
    normal_mask = ~defect_mask
    defect_count = int(defect_mask.sum())
    if fn_limit < 0 or fn_limit >= defect_count:
        raise CanonicalInputError(
            f"fn_limit must satisfy 0 <= fn_limit < {defect_count}"
        )
    if repaired_epoch < 1 or repaired_epoch > epoch_count:
        raise CanonicalInputError("repaired_epoch is outside the probability matrix")
    if not windows:
        raise CanonicalInputError("At least one dynamics window is required")
    names = [name for _, _, name in windows]
    if len(names) != len(set(names)):
        raise CanonicalInputError("Dynamics window names must be unique")
    for start, end, name in windows:
        if start < 1 or end > epoch_count or start > end or not name:
            raise CanonicalInputError(f"Invalid dynamics window: {(start, end, name)}")

    folds = _normalized_fold(sample_index["oof_fold"])
    repaired_fold_text = str(repaired_fold).lstrip("0") or "0"
    repaired_sample_mask = folds == repaired_fold_text

    valid_count = np.zeros(sample_count, dtype=np.int16)
    error_count = np.zeros(sample_count, dtype=np.int16)
    window_valid = {name: np.zeros(sample_count, dtype=np.int16) for _, _, name in windows}
    window_error = {name: np.zeros(sample_count, dtype=np.int16) for _, _, name in windows}
    forgetting = np.zeros(sample_count, dtype=np.int16)
    recovery = np.zeros(sample_count, dtype=np.int16)
    direction_changes = np.zeros(sample_count, dtype=np.int16)
    last_wrong = np.zeros(sample_count, dtype=np.int16)
    first_wrong = np.zeros(sample_count, dtype=np.int16)
    score_sum = np.zeros(sample_count, dtype=np.float64)
    score_sum_sq = np.zeros(sample_count, dtype=np.float64)
    epoch_sum = np.zeros(sample_count, dtype=np.float64)
    epoch_sum_sq = np.zeros(sample_count, dtype=np.float64)
    epoch_score_sum = np.zeros(sample_count, dtype=np.float64)
    has_previous = np.zeros(sample_count, dtype=bool)
    previous_correct = np.zeros(sample_count, dtype=bool)
    has_previous_score = np.zeros(sample_count, dtype=bool)
    previous_score = np.zeros(sample_count, dtype=np.float64)
    previous_direction = np.zeros(sample_count, dtype=np.int8)

    epoch_rows: list[dict[str, Any]] = []
    excluded_cell_count = 0
    for epoch_index in range(epoch_count):
        epoch = epoch_index + 1
        scores = np.asarray(probabilities[epoch_index], dtype=np.float64)
        if (
            scores.shape != (sample_count,)
            or not np.isfinite(scores).all()
            or float(scores.min()) < 0.0
            or float(scores.max()) > 1.0
        ):
            raise CanonicalInputError(f"OOF epoch {epoch} contains invalid probabilities")
        normal_scores = scores[normal_mask]
        defect_scores = scores[defect_mask]
        ordered_defect = np.sort(defect_scores)
        threshold = float(ordered_defect[fn_limit])
        predicted_defect = scores >= threshold
        actual_fn = int(np.count_nonzero(~predicted_defect[defect_mask]))
        actual_tn = int(np.count_nonzero(~predicted_defect[normal_mask]))
        if actual_fn > fn_limit:
            raise AssertionError("Tie-safe work point violated the FN constraint")
        normal_q68 = float(np.quantile(normal_scores, 0.68, method="nearest"))
        normal_q90 = float(np.quantile(normal_scores, 0.90, method="nearest"))
        defect_q50 = float(np.quantile(defect_scores, 0.50, method="nearest"))
        defect_q05 = float(np.quantile(defect_scores, 0.05, method="nearest"))
        epoch_rows.append(
            {
                "epoch": epoch,
                "threshold": threshold,
                "actual_FN": actual_fn,
                "TN_at_FN_limit": actual_tn,
                "fn_limit": fn_limit,
                "normal_q68_nearest": normal_q68,
                "normal_q90_nearest": normal_q90,
                "defect_q50_nearest": defect_q50,
                "defect_q05_nearest": defect_q05,
                "gap_q68_q050_nearest": defect_q50 - normal_q68,
                "tail_gap_q90_q05_nearest": defect_q05 - normal_q90,
            }
        )

        valid = np.ones(sample_count, dtype=bool)
        if epoch == repaired_epoch:
            valid[repaired_sample_mask] = False
            excluded_cell_count += int(repaired_sample_mask.sum())
        error = predicted_defect != defect_mask
        correct = ~error

        transition = valid & has_previous
        forgetting[transition & previous_correct & error] += 1
        recovery[transition & ~previous_correct & correct] += 1
        previous_correct[valid] = correct[valid]
        has_previous[valid] = True

        score_transition = valid & has_previous_score
        direction = np.zeros(sample_count, dtype=np.int8)
        difference = scores - previous_score
        direction[difference > 0] = 1
        direction[difference < 0] = -1
        nonzero = score_transition & (direction != 0)
        direction_changes[
            nonzero & (previous_direction != 0) & (direction != previous_direction)
        ] += 1
        previous_direction[nonzero] = direction[nonzero]
        previous_score[valid] = scores[valid]
        has_previous_score[valid] = True

        valid_count[valid] += 1
        error_count[valid & error] += 1
        first_wrong[valid & error & (first_wrong == 0)] = epoch
        last_wrong[valid & error] = epoch
        score_sum[valid] += scores[valid]
        score_sum_sq[valid] += scores[valid] ** 2
        epoch_sum[valid] += epoch
        epoch_sum_sq[valid] += epoch**2
        epoch_score_sum[valid] += epoch * scores[valid]
        for start, end, name in windows:
            if start <= epoch <= end:
                window_valid[name][valid] += 1
                window_error[name][valid & error] += 1

    if np.any(valid_count == 0):
        raise CanonicalInputError("At least one sample has no valid OOF epochs")
    means = score_sum / valid_count
    variances = np.maximum(score_sum_sq / valid_count - means**2, 0.0)
    denominator = valid_count * epoch_sum_sq - epoch_sum**2
    slopes = np.divide(
        valid_count * epoch_score_sum - epoch_sum * score_sum,
        denominator,
        out=np.full(sample_count, np.nan, dtype=np.float64),
        where=denominator != 0,
    )

    result = sample_index[["sample_id", "y_true", "oof_fold"]].copy()
    result["valid_epoch_count"] = valid_count
    result["operational_error_rate"] = error_count / valid_count
    result["operational_forgetting_count"] = forgetting
    result["operational_recovery_count"] = recovery
    result["score_direction_changes"] = direction_changes
    result["first_wrong_epoch"] = np.where(first_wrong > 0, first_wrong, np.nan)
    result["last_wrong_epoch"] = np.where(last_wrong > 0, last_wrong, np.nan)
    result["first_stably_correct_epoch"] = np.where(
        last_wrong == 0, 1.0, np.where(last_wrong < epoch_count, last_wrong + 1.0, np.nan)
    )
    result["mean_p_defect_operational"] = means
    result["std_p_defect_operational"] = np.sqrt(variances)
    result["p_defect_linear_slope"] = slopes
    for _, _, name in windows:
        rate = np.divide(
            window_error[name],
            window_valid[name],
            out=np.full(sample_count, np.nan, dtype=np.float64),
            where=window_valid[name] != 0,
        )
        result[f"error_rate_{name}"] = rate

    if "early" in window_error and "late" in window_error:
        early_name, late_name = "early", "late"
    else:
        early_name, late_name = names[0], names[-1]
    early_rate = result[f"error_rate_{early_name}"].to_numpy(dtype=float)
    late_rate = result[f"error_rate_{late_name}"].to_numpy(dtype=float)
    result["operational_correction"] = early_rate - late_rate
    result["trajectory_type"] = _trajectory_type(early_rate, late_rate)
    result["contains_repaired_epoch"] = repaired_sample_mask

    audit = {
        "epoch_count": int(epoch_count),
        "sample_count": int(sample_count),
        "normal_count": int(normal_mask.sum()),
        "defect_count": defect_count,
        "fn_limit": int(fn_limit),
        "threshold_rule": "predict_defect_when_score_gte_threshold",
        "fn_rule": "defect_score_lt_threshold",
        "repaired_fold": repaired_fold_text,
        "repaired_epoch": int(repaired_epoch),
        "affected_sample_count": int(repaired_sample_mask.sum()),
        "excluded_cell_count": int(excluded_cell_count),
    }
    return result, pd.DataFrame(epoch_rows), audit


def sample_set_digest(sample_ids: Iterable[str]) -> str:
    values = sorted(str(value) for value in sample_ids)
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest().upper()


def summarize_selection_set_outcomes(selection_rows: pd.DataFrame) -> pd.DataFrame:
    """Show when an identical Treatment sample set spans opposite cohorts."""

    required = {
        "triad_id",
        "sample_set_digest",
        "cohort_code",
        "condition_slot",
        "training_seed",
    }
    _require_columns(selection_rows, required, context="selection-set outcomes")
    rows: list[dict[str, Any]] = []
    for digest, group in selection_rows.groupby("sample_set_digest", sort=True):
        codes = sorted(set(group["cohort_code"].astype(str)))
        rows.append(
            {
                "sample_set_digest": digest,
                "triad_count": int(len(group)),
                "condition_slots": "|".join(
                    sorted(set(group["condition_slot"].astype(str)))
                ),
                "training_seeds": "|".join(
                    str(value) for value in sorted(set(group["training_seed"]))
                ),
                "cohort_codes": "|".join(codes),
                "exceptional_count": int((group["cohort_code"] == "S").sum()),
                "harmful_count": int((group["cohort_code"] == "H").sum()),
                "spans_exceptional_and_harmful": bool(
                    "S" in codes and "H" in codes
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["spans_exceptional_and_harmful", "triad_count", "sample_set_digest"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


_SELECTION_DYNAMIC_COLUMNS: tuple[str, ...] = (
    "operational_error_rate",
    "operational_forgetting_count",
    "operational_recovery_count",
    "score_direction_changes",
    "operational_correction",
    "error_rate_early_001_040",
    "error_rate_middle_041_120",
    "error_rate_late_121_160",
    "error_rate_late_161_200",
    "p_defect_linear_slope",
)


def summarize_selection_operational_features(
    canonical_runs: pd.DataFrame,
    selection_root: str | Path,
    operational_dynamics: pd.DataFrame,
    cohorts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate operational OOF dynamics for every frozen run selection."""

    run_columns = {
        "run_slot",
        "triad_id",
        "condition_slot",
        "condition_id",
        "phase",
        "method",
        "budget",
        "guard_ratio",
        "arm",
        "training_seed",
    }
    _require_columns(canonical_runs, run_columns, context="canonical runs")
    dynamic_columns = {"sample_id", "y_true", "trajectory_type"}
    available_dynamic = [
        column for column in _SELECTION_DYNAMIC_COLUMNS if column in operational_dynamics
    ]
    dynamic_columns.update(available_dynamic)
    _require_columns(
        operational_dynamics, dynamic_columns, context="operational sample dynamics"
    )
    if operational_dynamics["sample_id"].duplicated().any():
        raise CanonicalInputError("Operational dynamics sample IDs must be unique")
    _require_columns(
        cohorts, {"triad_id", "cohort_code", "cohort_label"}, context="cohorts"
    )
    root = Path(selection_root).resolve()
    if not root.is_dir():
        raise CanonicalInputError(f"Missing selection root: {root}")
    lookup = operational_dynamics.set_index("sample_id")
    if not lookup.index.is_unique:
        raise CanonicalInputError("Operational dynamics sample IDs must be unique")
    cohort_lookup = cohorts.set_index("triad_id")

    summaries: list[dict[str, Any]] = []
    treatment_sets: list[dict[str, Any]] = []
    for run in canonical_runs.sort_values("run_slot", kind="stable").itertuples(
        index=False
    ):
        path = root / str(run.run_slot) / "selection_manifest.csv"
        if not path.is_file():
            raise CanonicalInputError(f"Missing selection manifest: {path}")
        selection = pd.read_csv(path, usecols=lambda c: c in {"sample_id", "y_true"})
        _require_columns(selection, {"sample_id", "y_true"}, context=str(path))
        if selection["sample_id"].duplicated().any():
            raise CanonicalInputError(f"Duplicate sample IDs in {path}")
        if len(selection) != int(run.budget):
            raise CanonicalInputError(
                f"{run.run_slot} selection count {len(selection)} != budget {run.budget}"
            )
        missing = selection.loc[~selection["sample_id"].isin(lookup.index), "sample_id"]
        if len(missing):
            raise CanonicalInputError(
                f"{run.run_slot} contains {len(missing)} unknown OOF sample IDs"
            )
        joined = selection[["sample_id", "y_true"]].merge(
            operational_dynamics,
            on="sample_id",
            suffixes=("_selection", "_oof"),
            validate="one_to_one",
        )
        if not (
            pd.to_numeric(joined["y_true_selection"]).to_numpy()
            == pd.to_numeric(joined["y_true_oof"]).to_numpy()
        ).all():
            raise CanonicalInputError(f"Label mismatch in {run.run_slot}")
        joined["y_true"] = pd.to_numeric(joined["y_true_oof"]).astype(int)
        digest = sample_set_digest(joined["sample_id"])
        metadata = {
            column: getattr(run, column)
            for column in sorted(run_columns)
        }
        metadata["sample_set_digest"] = digest

        scopes = [("all", joined)]
        if (joined["y_true"] == 0).any():
            scopes.append(("normal", joined.loc[joined["y_true"] == 0]))
        if (joined["y_true"] == 1).any():
            scopes.append(("defect", joined.loc[joined["y_true"] == 1]))
        for scope, subset in scopes:
            record: dict[str, Any] = {
                **metadata,
                "scope": scope,
                "selected_count": int(len(subset)),
            }
            for column in available_dynamic:
                values = pd.to_numeric(subset[column], errors="coerce")
                record[f"mean_{column}"] = float(values.mean())
                record[f"median_{column}"] = float(values.median())
            trajectory = subset["trajectory_type"].astype(str)
            for label in (
                "corrected",
                "persistent_wrong",
                "deteriorating",
                "stable_correct",
                "transitional",
            ):
                record[f"share_{label}"] = float((trajectory == label).mean())
            summaries.append(record)

        if str(run.arm) == "T":
            if str(run.triad_id) not in cohort_lookup.index:
                raise CanonicalInputError(
                    f"Treatment triad absent from cohort table: {run.triad_id}"
                )
            cohort = cohort_lookup.loc[str(run.triad_id)]
            treatment_sets.append(
                {
                    "run_slot": run.run_slot,
                    "triad_id": run.triad_id,
                    "condition_slot": run.condition_slot,
                    "phase": run.phase,
                    "training_seed": run.training_seed,
                    "sample_set_digest": digest,
                    "cohort_code": cohort["cohort_code"],
                    "cohort_label": cohort["cohort_label"],
                    "selected_count": int(len(joined)),
                }
            )
    summary = pd.DataFrame(summaries)
    if summary["run_slot"].nunique() != canonical_runs["run_slot"].nunique():
        raise AssertionError("Selection summaries do not cover every canonical run")
    return summary, pd.DataFrame(treatment_sets)


def pair_selection_feature_deltas(
    selection_summary: pd.DataFrame, cohorts: pd.DataFrame
) -> pd.DataFrame:
    """Pair Treatment and control selection-dynamics features by scope."""

    _require_columns(
        selection_summary,
        {"run_slot", "triad_id", "arm", "scope"},
        context="selection operational summary",
    )
    feature_columns = [
        column
        for column in selection_summary.columns
        if column.startswith("mean_") or column.startswith("share_")
    ]
    if not feature_columns:
        raise CanonicalInputError("Selection summary has no pairable feature columns")
    _require_columns(
        cohorts, {"triad_id", "cohort_code", "cohort_label"}, context="cohorts"
    )
    rows: list[dict[str, Any]] = []
    for triad_id, triad in selection_summary.groupby("triad_id", sort=True):
        treatment = triad.loc[triad["arm"] == "T"]
        if treatment.empty:
            raise CanonicalInputError(f"{triad_id} has no Treatment selection summary")
        for control in ("R1", "R2"):
            control_rows = triad.loc[triad["arm"] == control]
            if control_rows.empty:
                raise CanonicalInputError(f"{triad_id} has no {control} summary")
            common_scopes = sorted(set(treatment["scope"]) & set(control_rows["scope"]))
            for scope in common_scopes:
                t_row = treatment.loc[treatment["scope"] == scope]
                c_row = control_rows.loc[control_rows["scope"] == scope]
                if len(t_row) != 1 or len(c_row) != 1:
                    raise CanonicalInputError(
                        f"{triad_id}/{control}/{scope} must be one-to-one"
                    )
                t = t_row.iloc[0]
                c = c_row.iloc[0]
                record: dict[str, Any] = {
                    "triad_id": triad_id,
                    "control": control,
                    "scope": scope,
                    "condition_slot": t.get("condition_slot"),
                    "condition_id": t.get("condition_id"),
                    "phase": t.get("phase"),
                    "budget": t.get("budget"),
                    "training_seed": t.get("training_seed"),
                    "t_run_slot": t["run_slot"],
                    "control_run_slot": c["run_slot"],
                }
                for column in feature_columns:
                    record[f"t_{column}"] = t[column]
                    record[f"control_{column}"] = c[column]
                    record[f"delta_{column}"] = float(t[column] - c[column])
                rows.append(record)
    result = pd.DataFrame(rows).merge(
        cohorts[["triad_id", "cohort_code", "cohort_label"]],
        on="triad_id",
        how="left",
        validate="many_to_one",
    )
    if result["cohort_code"].isna().any():
        raise CanonicalInputError("Selection pairs contain unknown cohort triads")
    return result.sort_values(["triad_id", "control", "scope"]).reset_index(
        drop=True
    )


def build_outcome_mechanism_pairs(
    triad_deltas: pd.DataFrame,
    canonical_runs: pd.DataFrame,
    calibration_diagnostics: pd.DataFrame,
    cohorts: pd.DataFrame,
) -> pd.DataFrame:
    """Attach threshold and AUROC changes as post-training diagnostics."""

    _require_columns(
        triad_deltas,
        {
            "triad_id",
            "control",
            "t_run_slot",
            "control_run_slot",
            "delta_TN",
            "delta_FN",
        },
        context="triad deltas",
    )
    _require_columns(
        canonical_runs,
        {"run_slot", "threshold_at_FN95", "raw_threshold_at_FN95"},
        context="canonical runs",
    )
    _require_columns(
        calibration_diagnostics,
        {"run_slot", "split", "auroc", "auroc_raw"},
        context="calibration diagnostics",
    )
    runs = canonical_runs[
        ["run_slot", "threshold_at_FN95", "raw_threshold_at_FN95"]
    ].copy()
    val_op = calibration_diagnostics.loc[
        calibration_diagnostics["split"] == "val_op",
        ["run_slot", "auroc", "auroc_raw"],
    ].copy()
    if val_op["run_slot"].duplicated().any():
        raise CanonicalInputError("val_op calibration rows must be unique per run")
    metrics = runs.merge(val_op, on="run_slot", validate="one_to_one")
    t = metrics.rename(
        columns={"run_slot": "t_run_slot", **{c: f"t_{c}" for c in metrics.columns if c != "run_slot"}}
    )
    c = metrics.rename(
        columns={
            "run_slot": "control_run_slot",
            **{col: f"control_{col}" for col in metrics.columns if col != "run_slot"},
        }
    )
    result = triad_deltas.merge(t, on="t_run_slot", validate="many_to_one").merge(
        c, on="control_run_slot", validate="many_to_one"
    )
    for column, output in (
        ("threshold_at_FN95", "delta_threshold"),
        ("raw_threshold_at_FN95", "delta_raw_threshold"),
        ("auroc", "delta_auroc"),
        ("auroc_raw", "delta_auroc_raw"),
    ):
        result[output] = result[f"t_{column}"] - result[f"control_{column}"]
    result = result.merge(
        cohorts[["triad_id", "cohort_code", "cohort_label"]],
        on="triad_id",
        how="left",
        validate="many_to_one",
    )
    result["evidence_role"] = "post_training_diagnostic_only"
    return result


def _cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    if not len(left) or not len(right):
        return float("nan")
    comparisons = np.sign(left[:, None] - right[None, :])
    return float(comparisons.mean())


def _bootstrap_mean_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float]:
    if not len(left) or not len(right) or samples < 1:
        return float("nan"), float("nan")
    values = np.empty(samples, dtype=np.float64)
    # Generate bootstrap replicates in bounded vectorized chunks.  The original
    # scalar loop became the dominant cost once dozens of stratified and tail
    # contrasts were requested; chunking preserves the estimator while keeping
    # peak memory independent of the total number of replicates.
    chunk_size = min(samples, 2048)
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        count = stop - start
        left_indices = rng.integers(0, len(left), size=(count, len(left)))
        right_indices = rng.integers(0, len(right), size=(count, len(right)))
        values[start:stop] = left[left_indices].mean(axis=1) - right[
            right_indices
        ].mean(axis=1)
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def summarize_extreme_feature_contrasts(
    features: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    random_seed: int = 20260801,
    bootstrap_samples: int = 2000,
) -> pd.DataFrame:
    """Compare S and H at the triad level under predefined scopes."""

    required = {
        "triad_id",
        "control",
        "cohort_code",
        "phase",
        "budget",
        "training_seed",
        "machine_pair",
        "any_resumed",
        *feature_columns,
    }
    _require_columns(features, required, context="extreme feature contrasts")
    scopes: list[tuple[str, pd.Series]] = [
        ("all", pd.Series(True, index=features.index)),
        ("phase_A_same_machine", (features["phase"] == "A") & (features["machine_pair"] == "same_machine")),
        ("phase_B_machine_confounded", features["phase"] == "B"),
        ("no_resume", ~features["any_resumed"].astype(bool)),
    ]
    for budget in sorted(pd.to_numeric(features["budget"], errors="coerce").dropna().unique()):
        scopes.append((f"budget_{int(budget)}", pd.to_numeric(features["budget"]) == budget))
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []
    for scope_name, mask in scopes:
        scoped = features.loc[mask & features["cohort_code"].isin(["S", "H"])]
        for control in ("R1", "R2"):
            subset = scoped.loc[scoped["control"] == control]
            exceptional = subset.loc[subset["cohort_code"] == "S"]
            harmful = subset.loc[subset["cohort_code"] == "H"]
            if exceptional.empty or harmful.empty:
                continue
            for feature in feature_columns:
                left = pd.to_numeric(exceptional[feature], errors="coerce").dropna().to_numpy(dtype=float)
                right = pd.to_numeric(harmful[feature], errors="coerce").dropna().to_numpy(dtype=float)
                if not len(left) or not len(right):
                    continue
                ci_low, ci_high = _bootstrap_mean_difference(
                    left, right, rng=rng, samples=bootstrap_samples
                )
                rows.append(
                    {
                        "analysis_scope": scope_name,
                        "control": control,
                        "feature": feature,
                        "n_exceptional": int(len(left)),
                        "n_harmful": int(len(right)),
                        "exceptional_mean": float(left.mean()),
                        "harmful_mean": float(right.mean()),
                        "exceptional_median": float(np.median(left)),
                        "harmful_median": float(np.median(right)),
                        "mean_difference_S_minus_H": float(left.mean() - right.mean()),
                        "median_difference_S_minus_H": float(np.median(left) - np.median(right)),
                        "cliffs_delta_S_vs_H": _cliffs_delta(left, right),
                        "bootstrap_95_low": ci_low,
                        "bootstrap_95_high": ci_high,
                        "statistical_unit": "triad",
                    }
                )
    return pd.DataFrame(rows)


def build_stratified_extreme_contrasts(
    features: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    block_columns: Sequence[str] = ("phase", "budget", "training_seed"),
) -> pd.DataFrame:
    """Compare S/H only inside blocks where both cohorts are observed."""

    required = {"triad_id", "control", "cohort_code", *block_columns, *feature_columns}
    _require_columns(features, required, context="stratified extreme contrasts")
    rows: list[dict[str, Any]] = []
    grouping = [*block_columns, "control"]
    for keys, group in features.loc[
        features["cohort_code"].isin(["S", "H"])
    ].groupby(grouping, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metadata = dict(zip(grouping, keys, strict=True))
        exceptional = group.loc[group["cohort_code"] == "S"]
        harmful = group.loc[group["cohort_code"] == "H"]
        if exceptional.empty or harmful.empty:
            continue
        for feature in feature_columns:
            left = pd.to_numeric(exceptional[feature], errors="coerce").dropna().to_numpy(dtype=float)
            right = pd.to_numeric(harmful[feature], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(left) or not len(right):
                continue
            rows.append(
                {
                    **metadata,
                    "feature": feature,
                    "n_exceptional": int(len(left)),
                    "n_harmful": int(len(right)),
                    "exceptional_mean": float(left.mean()),
                    "harmful_mean": float(right.mean()),
                    "mean_difference_S_minus_H": float(left.mean() - right.mean()),
                    "median_difference_S_minus_H": float(
                        np.median(left) - np.median(right)
                    ),
                    "cliffs_delta_S_vs_H": _cliffs_delta(left, right),
                    "statistical_unit": "triad",
                    "eligibility_rule": "block contains at least one S and one H",
                }
            )
    return pd.DataFrame(rows)


def summarize_leave_one_group_out(
    features: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    group_column: str,
) -> pd.DataFrame:
    """Recompute the overall S-H mean difference after omitting each group."""

    required = {
        "triad_id",
        "control",
        "cohort_code",
        group_column,
        *feature_columns,
    }
    _require_columns(features, required, context="leave-one-group-out contrasts")
    rows: list[dict[str, Any]] = []
    for omitted in sorted(features[group_column].drop_duplicates(), key=str):
        retained = features.loc[
            (features[group_column] != omitted)
            & features["cohort_code"].isin(["S", "H"])
        ]
        for control in ("R1", "R2"):
            subset = retained.loc[retained["control"] == control]
            exceptional = subset.loc[subset["cohort_code"] == "S"]
            harmful = subset.loc[subset["cohort_code"] == "H"]
            if exceptional.empty or harmful.empty:
                continue
            for feature in feature_columns:
                left = pd.to_numeric(exceptional[feature], errors="coerce").dropna().to_numpy(dtype=float)
                right = pd.to_numeric(harmful[feature], errors="coerce").dropna().to_numpy(dtype=float)
                if not len(left) or not len(right):
                    continue
                rows.append(
                    {
                        f"omitted_{group_column}": omitted,
                        "control": control,
                        "feature": feature,
                        "n_exceptional": int(len(left)),
                        "n_harmful": int(len(right)),
                        "mean_difference_S_minus_H": float(
                            left.mean() - right.mean()
                        ),
                        "median_difference_S_minus_H": float(
                            np.median(left) - np.median(right)
                        ),
                        "cliffs_delta_S_vs_H": _cliffs_delta(left, right),
                        "statistical_unit": "triad",
                    }
                )
    return pd.DataFrame(rows)
