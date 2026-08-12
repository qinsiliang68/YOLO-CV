"""Canonical raw-prediction and Neyman-Pearson frontier analysis.

This module is deliberately read-only with respect to the 240 run artifacts.
The transfer inventory selects the one canonical attempt for every run.  Run
predictions are loaded one at a time; no directory scan is used to guess a
preferred attempt.

Performance comparisons use the complete tie-safe frontier under the rule
``predict defect when score >= threshold``.  Candidate and control models are
therefore compared at the same false-negative *budget*, not at independently
chosen confidence thresholds.  Fixed sample tails are defined once from the
median raw score of R1/R2 controls only, preventing treatment-outcome leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .calibration import PlattModel
from .deep_analysis import CanonicalInputError
from .errors import ValidationError
from .performance_frontier import compare_frontiers, frontier_from_predictions


class RawFrontierError(RuntimeError):
    """Raised when canonical prediction evidence is incomplete or ambiguous."""


PREDICTION_COLUMNS = ("sample_id", "y_true", "score", "score_raw")
CANONICAL_INVENTORY_COLUMNS = (
    "run_slot",
    "triad_id",
    "phase",
    "condition_id",
    "method",
    "budget",
    "guard_ratio",
    "arm",
    "training_seed",
    "selection_seed",
    "package",
    "machine_id",
    "attempt_id",
    "resume_count",
    "selection_sha256",
    "release_ref",
    "release_commit",
    "input_snapshot_id",
)


@dataclass(frozen=True)
class RunPredictionAnalysis:
    """One streamed run result; callers may discard it before loading the next."""

    run_slot: str
    diagnostics: pd.DataFrame
    raw_frontier: pd.DataFrame
    calibrated_frontier: pd.DataFrame
    raw_calibrated_frontier_exact: bool


def _path_within(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def load_canonical_prediction_index(
    inventory_path: str | Path,
    extracted_root: str | Path,
    *,
    expected_runs: int = 240,
    expected_triads: int = 80,
    require_val_cal: bool = False,
) -> pd.DataFrame:
    """Resolve exactly one inventory-authorized val-op prediction per run."""

    inventory_path = Path(inventory_path).resolve()
    extracted_root = Path(extracted_root).resolve()
    if not inventory_path.is_file():
        raise RawFrontierError(f"Canonical inventory does not exist: {inventory_path}")
    if not extracted_root.is_dir():
        raise RawFrontierError(f"Extracted root does not exist: {extracted_root}")
    inventory = pd.read_csv(inventory_path, dtype={"run_slot": "string"})
    missing = sorted(set(CANONICAL_INVENTORY_COLUMNS).difference(inventory.columns))
    if missing:
        raise RawFrontierError(f"Canonical inventory missing columns: {missing}")
    if len(inventory) != expected_runs:
        raise RawFrontierError(
            f"Expected {expected_runs} canonical runs, found {len(inventory)}"
        )
    if inventory["run_slot"].isna().any() or inventory["run_slot"].duplicated().any():
        raise RawFrontierError("Canonical inventory run_slot must be non-null and unique")
    if inventory["attempt_id"].isna().any():
        raise RawFrontierError("Canonical inventory attempt_id must be non-null")
    if inventory["triad_id"].nunique() != expected_triads:
        raise RawFrontierError(
            f"Expected {expected_triads} canonical triads, "
            f"found {inventory['triad_id'].nunique()}"
        )
    expected_arms = {"T", "R1", "R2"}
    for triad_id, group in inventory.groupby("triad_id", sort=True):
        arms = group["arm"].astype(str)
        if len(group) != 3 or set(arms) != expected_arms or arms.duplicated().any():
            raise RawFrontierError(
                f"{triad_id} must contain exactly one T, R1 and R2 canonical run"
            )

    resolved: list[str] = []
    val_cal_resolved: list[str] = []
    platt_resolved: list[str] = []
    for row in inventory.itertuples(index=False):
        attempt = (
            extracted_root
            / f"stage1_gapvalue240_{row.package}_upload"
            / "runs"
            / str(row.run_slot)
            / str(row.attempt_id)
        ).resolve()
        if not _path_within(extracted_root, attempt):
            raise RawFrontierError(
                f"Canonical attempt escapes extracted root for {row.run_slot}"
            )
        prediction = attempt / "04_predictions" / "val_op_predictions.csv"
        if not prediction.is_file():
            raise RawFrontierError(
                f"Canonical val_op prediction missing for {row.run_slot}: {prediction}"
            )
        resolved.append(str(prediction))
        if require_val_cal:
            val_cal = attempt / "04_predictions" / "val_cal_predictions.csv"
            platt = attempt / "05_metrics" / "platt_calibration.json"
            if not val_cal.is_file():
                raise RawFrontierError(
                    f"Canonical val_cal prediction missing for {row.run_slot}: {val_cal}"
                )
            if not platt.is_file():
                raise RawFrontierError(
                    f"Canonical Platt calibration missing for {row.run_slot}: {platt}"
                )
            val_cal_resolved.append(str(val_cal))
            platt_resolved.append(str(platt))
    result = inventory.copy()
    result["run_slot"] = result["run_slot"].astype(str)
    result["prediction_path"] = resolved
    if require_val_cal:
        result["val_cal_prediction_path"] = val_cal_resolved
        result["platt_calibration_path"] = platt_resolved
    return result.sort_values("run_slot", kind="stable", ignore_index=True)


def read_val_op_predictions(
    path: str | Path,
    *,
    expected_rows: int | None = 120_000,
    expected_normal: int | None = 100_000,
    expected_defect: int | None = 20_000,
) -> pd.DataFrame:
    """Read and strictly validate one canonical prediction file."""

    path = Path(path)
    if not path.is_file():
        raise RawFrontierError(f"Prediction file does not exist: {path}")
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if header != list(PREDICTION_COLUMNS):
            raise RawFrontierError(
                f"{path} schema must be exactly {list(PREDICTION_COLUMNS)}, "
                f"found {header}"
            )
        frame = pd.read_csv(
            path,
            usecols=list(PREDICTION_COLUMNS),
            dtype={"sample_id": "string"},
        )
    except (ValueError, OSError) as exc:
        raise RawFrontierError(f"Cannot read canonical prediction {path}: {exc}") from exc
    if expected_rows is not None and len(frame) != expected_rows:
        raise RawFrontierError(
            f"{path} expected {expected_rows} rows, found {len(frame)}"
        )
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise RawFrontierError(f"{path} sample_id must be non-null and unique")
    try:
        labels = pd.to_numeric(frame["y_true"], errors="raise").to_numpy(dtype=np.int8)
        calibrated = pd.to_numeric(frame["score"], errors="raise").to_numpy(
            dtype=np.float64
        )
        raw = pd.to_numeric(frame["score_raw"], errors="raise").to_numpy(
            dtype=np.float64
        )
    except (TypeError, ValueError) as exc:
        raise RawFrontierError(f"{path} contains non-numeric prediction values") from exc
    if not set(np.unique(labels)).issubset({0, 1}) or len(np.unique(labels)) != 2:
        raise RawFrontierError(f"{path} y_true must contain both binary classes")
    if not np.isfinite(calibrated).all() or not np.isfinite(raw).all():
        raise RawFrontierError(f"{path} contains NaN/Inf prediction scores")
    if ((calibrated < 0) | (calibrated > 1)).any() or (
        (raw < 0) | (raw > 1)
    ).any():
        raise RawFrontierError(f"{path} prediction probabilities must be in [0, 1]")
    normal_count = int((labels == 0).sum())
    defect_count = int((labels == 1).sum())
    if expected_normal is not None and normal_count != expected_normal:
        raise RawFrontierError(
            f"{path} expected {expected_normal} normal rows, found {normal_count}"
        )
    if expected_defect is not None and defect_count != expected_defect:
        raise RawFrontierError(
            f"{path} expected {expected_defect} defect rows, found {defect_count}"
        )
    return pd.DataFrame(
        {
            "sample_id": frame["sample_id"].astype(str),
            "y_true": labels,
            "score": calibrated,
            "score_raw": raw,
        }
    )


def _ece(y_true: np.ndarray, probabilities: np.ndarray, bins: int) -> float:
    if bins <= 0:
        raise ValueError("ece_bins must be positive")
    edges = np.linspace(0.0, 1.0, bins + 1)
    # [0, edge_1), ..., [edge_(n-1), 1]; probability 1 belongs to last bin.
    bin_ids = np.minimum(
        np.digitize(probabilities, edges[1:-1], right=False), bins - 1
    )
    result = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if mask.any():
            result += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(y_true[mask].mean())
            )
    return float(result)


def probability_diagnostics(
    predictions: pd.DataFrame,
    *,
    ece_bins: int = 15,
) -> pd.DataFrame:
    """Recompute discrimination and calibration metrics in both score spaces."""

    required = set(PREDICTION_COLUMNS)
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise RawFrontierError(f"Prediction table missing columns: {missing}")
    if predictions["sample_id"].duplicated().any():
        raise RawFrontierError("Prediction sample IDs must be unique")
    y = pd.to_numeric(predictions["y_true"], errors="raise").to_numpy(dtype=np.int8)
    if set(np.unique(y)) != {0, 1}:
        raise RawFrontierError("Probability diagnostics require both binary classes")
    rows: list[dict[str, float | int | str]] = []
    for score_type, column in (("raw", "score_raw"), ("calibrated", "score")):
        values = pd.to_numeric(predictions[column], errors="raise").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise RawFrontierError(f"{column} contains NaN/Inf")
        if ((values < 0) | (values > 1)).any():
            raise RawFrontierError(f"{column} probabilities must be in [0, 1]")
        clipped = np.clip(values, 1e-12, 1 - 1e-12)
        rows.append(
            {
                "score_type": score_type,
                "score_column": column,
                "row_count": int(len(y)),
                "normal_count": int((y == 0).sum()),
                "defect_count": int((y == 1).sum()),
                "auroc": float(roc_auc_score(y, values)),
                "auprc": float(average_precision_score(y, values)),
                "brier": float(brier_score_loss(y, values)),
                "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
                "ece": _ece(y, values, ece_bins),
                "ece_bins": int(ece_bins),
            }
        )
    return pd.DataFrame(rows)


def raw_calibrated_ranking_audit(
    predictions: pd.DataFrame,
) -> dict[str, float | int | bool]:
    """Audit whether calibration preserves raw ordering and the exact NP frontier."""

    required = set(PREDICTION_COLUMNS)
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise RawFrontierError(f"Prediction table missing columns: {missing}")
    raw = pd.to_numeric(predictions["score_raw"], errors="raise").to_numpy(
        dtype=np.float64
    )
    calibrated = pd.to_numeric(predictions["score"], errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(raw).all() or not np.isfinite(calibrated).all():
        raise RawFrontierError("Ranking audit scores contain NaN/Inf")
    order = np.argsort(raw, kind="mergesort")
    ordered_raw = raw[order]
    ordered_calibrated = calibrated[order]
    starts = np.r_[0, np.flatnonzero(ordered_raw[1:] != ordered_raw[:-1]) + 1]
    group_min = np.minimum.reduceat(ordered_calibrated, starts)
    group_max = np.maximum.reduceat(ordered_calibrated, starts)
    violations = group_min[1:] < group_max[:-1]
    within_group_ranges = group_max - group_min
    raw_ranks = pd.Series(raw).rank(method="average").to_numpy(dtype=np.float64)
    calibrated_ranks = (
        pd.Series(calibrated).rank(method="average").to_numpy(dtype=np.float64)
    )
    spearman = float(np.corrcoef(raw_ranks, calibrated_ranks)[0, 1])
    raw_frontier = exact_np_frontier(predictions, score_column="score_raw")
    calibrated_frontier = exact_np_frontier(predictions, score_column="score")
    equality_columns = ["fn_budget", "actual_fn", "TN", "FP"]
    frontier_exact = raw_frontier[equality_columns].equals(
        calibrated_frontier[equality_columns]
    )
    return {
        "raw_unique_score_count": int(np.unique(raw).size),
        "calibrated_unique_score_count": int(np.unique(calibrated).size),
        "unique_score_count_delta": int(np.unique(calibrated).size - np.unique(raw).size),
        "monotonic_violation_count": int(violations.sum()),
        "monotonic_nondecreasing": bool(not violations.any()),
        "raw_tie_group_with_calibrated_spread_count": int(
            (within_group_ranges > 0).sum()
        ),
        "max_within_raw_tie_calibrated_range": float(within_group_ranges.max()),
        "spearman_rank_correlation": spearman,
        "raw_calibrated_frontier_exact": bool(frontier_exact),
        "raw_frontier_breakpoint_count": int(len(compact_frontier(raw_frontier))),
        "calibrated_frontier_breakpoint_count": int(
            len(compact_frontier(calibrated_frontier))
        ),
    }


def platt_calibration_audit(
    predictions: pd.DataFrame,
    parameters: dict[str, object],
) -> dict[str, float | int | bool]:
    """Validate and exactly replay one saved Platt transform."""

    required = {
        "coefficient",
        "intercept",
        "source_prevalence",
        "deployment_prevalence",
        "clip_low",
        "clip_high",
    }
    missing = sorted(required.difference(parameters))
    unexpected = sorted(set(parameters).difference(required))
    if missing or unexpected:
        raise RawFrontierError(
            f"Platt JSON schema mismatch: missing={missing}, unexpected={unexpected}"
        )
    try:
        values = {key: float(parameters[key]) for key in required}
    except (TypeError, ValueError) as exc:
        raise RawFrontierError("Platt JSON contains non-numeric values") from exc
    if not np.isfinite(list(values.values())).all():
        raise RawFrontierError("Platt JSON contains NaN/Inf")
    if not 0 < values["clip_low"] < values["clip_high"] < 1:
        raise RawFrontierError("Platt clipping bounds must satisfy 0 < low < high < 1")
    for name in ("source_prevalence", "deployment_prevalence"):
        if not 0 < values[name] < 1:
            raise RawFrontierError(f"Platt {name} must be in (0, 1)")
    model = PlattModel(
        coefficient=values["coefficient"],
        intercept=values["intercept"],
        source_prevalence=values["source_prevalence"],
        deployment_prevalence=values["deployment_prevalence"],
        clip_low=values["clip_low"],
        clip_high=values["clip_high"],
    )
    raw = pd.to_numeric(predictions["score_raw"], errors="raise").to_numpy(
        dtype=np.float64
    )
    saved = pd.to_numeric(predictions["score"], errors="raise").to_numpy(
        dtype=np.float64
    )
    labels = pd.to_numeric(predictions["y_true"], errors="raise").to_numpy(
        dtype=np.int8
    )
    recomputed = model.transform(raw)
    residual = np.abs(saved - recomputed)
    observed_prevalence = float(labels.mean())
    return {
        **values,
        "coefficient_positive": bool(values["coefficient"] > 0),
        "coefficient_zero": bool(values["coefficient"] == 0),
        "coefficient_negative": bool(values["coefficient"] < 0),
        "observed_source_prevalence": observed_prevalence,
        "source_prevalence_abs_error": abs(
            values["source_prevalence"] - observed_prevalence
        ),
        "raw_clipped_low_count": int((raw < values["clip_low"]).sum()),
        "raw_clipped_high_count": int((raw > values["clip_high"]).sum()),
        "max_abs_transform_residual": float(residual.max()),
        "mean_abs_transform_residual": float(residual.mean()),
        "transform_recomputed_exact_1e12": bool(residual.max() <= 1e-12),
    }


def exact_np_frontier(
    predictions: pd.DataFrame,
    *,
    score_column: str,
) -> pd.DataFrame:
    """Return the exact maximum-TN result for every integer ``FN <= k`` budget."""

    try:
        frontier = frontier_from_predictions(
            predictions,
            score_column=score_column,
        )
    except (CanonicalInputError, ValidationError, ValueError, KeyError) as exc:
        raise RawFrontierError(str(exc)) from exc
    result = frontier.copy()
    result.insert(0, "score_column", score_column)
    result["threshold_rule"] = "score >= threshold"
    result["whole_tie_groups"] = True
    return result


def compact_frontier(frontier: pd.DataFrame) -> pd.DataFrame:
    """Losslessly run-length encode adjacent FN budgets with one NP solution."""

    required = {
        "fn_budget",
        "actual_fn",
        "TN",
        "FP",
        "threshold",
        "tie_group_size",
    }
    missing = sorted(required.difference(frontier.columns))
    if missing:
        raise RawFrontierError(f"Frontier missing columns: {missing}")
    ordered = frontier.sort_values("fn_budget", kind="stable").reset_index(drop=True)
    budgets = ordered["fn_budget"].to_numpy(dtype=np.int64)
    if not np.array_equal(budgets, np.arange(len(ordered), dtype=np.int64)):
        raise RawFrontierError("Frontier fn_budget must be exactly 0..N")
    state_columns = ["actual_fn", "TN", "FP", "threshold", "tie_group_size"]
    state = ordered[state_columns]
    starts = np.r_[True, state.ne(state.shift()).any(axis=1).to_numpy()[1:]]
    start_indices = np.flatnonzero(starts)
    compact = ordered.iloc[start_indices].copy().reset_index(drop=True)
    next_indices = np.r_[start_indices[1:], len(ordered)]
    compact["next_fn_budget_exclusive"] = budgets[next_indices - 1] + 1
    compact["represented_budget_count"] = (
        compact["next_fn_budget_exclusive"].to_numpy(dtype=np.int64)
        - compact["fn_budget"].to_numpy(dtype=np.int64)
    )
    return compact


def iter_canonical_run_analyses(
    canonical_index: pd.DataFrame,
    *,
    expected_rows: int = 120_000,
    expected_normal: int = 100_000,
    expected_defect: int = 20_000,
    ece_bins: int = 15,
) -> Iterator[RunPredictionAnalysis]:
    """Load, analyze and yield one canonical run at a time."""

    required = {"run_slot", "prediction_path"}
    missing = sorted(required.difference(canonical_index.columns))
    if missing:
        raise RawFrontierError(f"Canonical prediction index missing columns: {missing}")
    if canonical_index["run_slot"].duplicated().any():
        raise RawFrontierError("Canonical prediction index has duplicate run_slot")
    for row in canonical_index.sort_values("run_slot", kind="stable").itertuples(
        index=False
    ):
        predictions = read_val_op_predictions(
            row.prediction_path,
            expected_rows=expected_rows,
            expected_normal=expected_normal,
            expected_defect=expected_defect,
        )
        diagnostics = probability_diagnostics(predictions, ece_bins=ece_bins)
        raw = exact_np_frontier(predictions, score_column="score_raw")
        calibrated = exact_np_frontier(predictions, score_column="score")
        comparison_columns = ["fn_budget", "actual_fn", "TN", "FP"]
        exact = raw[comparison_columns].equals(calibrated[comparison_columns])
        yield RunPredictionAnalysis(
            run_slot=str(row.run_slot),
            diagnostics=diagnostics,
            raw_frontier=raw,
            calibrated_frontier=calibrated,
            raw_calibrated_frontier_exact=bool(exact),
        )


def _load_reference_prediction(
    path: Path,
    *,
    expected_rows: int | None,
    expected_normal: int | None,
    expected_defect: int | None,
) -> pd.DataFrame:
    return read_val_op_predictions(
        path,
        expected_rows=expected_rows,
        expected_normal=expected_normal,
        expected_defect=expected_defect,
    ).sort_values("sample_id", kind="stable", ignore_index=True)


def _stable_tail_ids(
    frame: pd.DataFrame,
    *,
    count: int,
    highest: bool,
) -> pd.Index:
    if count <= 0:
        return pd.Index([], dtype=frame.index.dtype)
    ordered = frame.sort_values(
        ["control_median_score_raw", "sample_id"],
        ascending=[not highest, True],
        kind="mergesort",
    )
    return ordered.index[: min(count, len(ordered))]


def build_control_reference(
    control_prediction_paths: Sequence[str | Path],
    *,
    tn_target: int = 68_253,
    fn_limit: int = 95,
    normal_tail_fraction: float = 0.10,
    defect_tail_fraction: float = 0.05,
    expected_rows: int | None = None,
    expected_normal: int | None = None,
    expected_defect: int | None = None,
) -> pd.DataFrame:
    """Build an exact, treatment-independent median-risk sample reference.

    The caller must pass only canonical R1/R2 files.  Values remain float64;
    unlike the older report helper, this analysis does not downcast prediction
    probabilities before taking the median.
    """

    paths = [Path(path).resolve() for path in control_prediction_paths]
    if not paths:
        raise RawFrontierError("At least one canonical control prediction is required")
    if len(set(paths)) != len(paths):
        raise RawFrontierError("Canonical control prediction paths must be unique")
    if not (0 < normal_tail_fraction <= 1) or not (0 < defect_tail_fraction <= 1):
        raise ValueError("Tail fractions must be in (0, 1]")
    first = _load_reference_prediction(
        paths[0],
        expected_rows=expected_rows,
        expected_normal=expected_normal,
        expected_defect=expected_defect,
    )
    sample_ids = first["sample_id"].to_numpy(dtype=str)
    labels = first["y_true"].to_numpy(dtype=np.int8)
    score_matrix = np.empty((len(paths), len(first)), dtype=np.float64)
    for row_index, path in enumerate(paths):
        frame = (
            first
            if row_index == 0
            else _load_reference_prediction(
                path,
                expected_rows=expected_rows,
                expected_normal=expected_normal,
                expected_defect=expected_defect,
            )
        )
        if not np.array_equal(frame["sample_id"].to_numpy(dtype=str), sample_ids):
            raise RawFrontierError(f"Prediction sample ID set differs in {path}")
        if not np.array_equal(frame["y_true"].to_numpy(dtype=np.int8), labels):
            raise RawFrontierError(f"Prediction labels differ in {path}")
        score_matrix[row_index] = frame["score_raw"].to_numpy(dtype=np.float64)
    median = np.median(score_matrix, axis=0)
    del score_matrix
    result = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "y_true": labels,
            "control_median_score_raw": median,
            "control_run_count": len(paths),
            "reference_source_arms": "R1,R2",
        }
    )
    normal = result[result.y_true == 0]
    defect = result[result.y_true == 1]
    normal_operational_count = len(normal) - int(tn_target)
    if normal_operational_count < 0:
        raise RawFrontierError(
            f"TN target {tn_target} exceeds normal count {len(normal)}"
        )
    if fn_limit < 0 or fn_limit > len(defect):
        raise RawFrontierError(
            f"FN limit {fn_limit} is outside 0..{len(defect)}"
        )
    result["operational_tail"] = False
    result["distribution_tail"] = False
    result.loc[
        _stable_tail_ids(normal, count=normal_operational_count, highest=True),
        "operational_tail",
    ] = True
    result.loc[
        _stable_tail_ids(defect, count=fn_limit, highest=False),
        "operational_tail",
    ] = True
    result.loc[
        _stable_tail_ids(
            normal,
            count=max(1, int(round(len(normal) * normal_tail_fraction))),
            highest=True,
        ),
        "distribution_tail",
    ] = True
    result.loc[
        _stable_tail_ids(
            defect,
            count=max(1, int(round(len(defect) * defect_tail_fraction))),
            highest=False,
        ),
        "distribution_tail",
    ] = True
    result["reference_risk_rank_within_class"] = 0
    normal_rank = normal.sort_values(
        ["control_median_score_raw", "sample_id"],
        ascending=[False, True],
        kind="mergesort",
    ).index
    defect_rank = defect.sort_values(
        ["control_median_score_raw", "sample_id"],
        ascending=[True, True],
        kind="mergesort",
    ).index
    result.loc[normal_rank, "reference_risk_rank_within_class"] = np.arange(
        1, len(normal_rank) + 1
    )
    result.loc[defect_rank, "reference_risk_rank_within_class"] = np.arange(
        1, len(defect_rank) + 1
    )
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
    result["control_path_set_sha256"] = digest.hexdigest().upper()
    return result


def build_control_reference_from_index(
    canonical_index: pd.DataFrame,
    *,
    expected_controls: int = 160,
    tn_target: int = 68_253,
    fn_limit: int = 95,
    normal_tail_fraction: float = 0.10,
    defect_tail_fraction: float = 0.05,
    expected_rows: int | None = 120_000,
    expected_normal: int | None = 100_000,
    expected_defect: int | None = 20_000,
) -> pd.DataFrame:
    """Select only canonical R1/R2 rows and construct the fixed reference."""

    required = {"run_slot", "triad_id", "arm", "prediction_path"}
    missing = sorted(required.difference(canonical_index.columns))
    if missing:
        raise RawFrontierError(f"Canonical prediction index missing columns: {missing}")
    if canonical_index["run_slot"].duplicated().any():
        raise RawFrontierError("Canonical prediction index has duplicate run_slot")
    controls = canonical_index[canonical_index.arm.astype(str).isin({"R1", "R2"})]
    if len(controls) != expected_controls:
        raise RawFrontierError(
            f"Expected {expected_controls} canonical controls, found {len(controls)}"
        )
    for triad_id, group in controls.groupby("triad_id", sort=True):
        if len(group) != 2 or set(group.arm.astype(str)) != {"R1", "R2"}:
            raise RawFrontierError(
                f"{triad_id} control reference requires exactly one R1 and one R2"
            )
    reference = build_control_reference(
        controls.sort_values("run_slot", kind="stable")["prediction_path"].tolist(),
        tn_target=tn_target,
        fn_limit=fn_limit,
        normal_tail_fraction=normal_tail_fraction,
        defect_tail_fraction=defect_tail_fraction,
        expected_rows=expected_rows,
        expected_normal=expected_normal,
        expected_defect=expected_defect,
    )
    reference["reference_source_control_count"] = int(len(controls))
    reference["reference_source_treatment_count"] = 0
    return reference


def _align_pair_with_reference(
    treatment: pd.DataFrame,
    control: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    columns = list(PREDICTION_COLUMNS)
    for name, frame in (("treatment", treatment), ("control", control)):
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise RawFrontierError(f"{name} prediction table missing columns: {missing}")
        if frame["sample_id"].duplicated().any():
            raise RawFrontierError(f"{name} prediction sample IDs must be unique")
    required_reference = {
        "sample_id",
        "y_true",
        "operational_tail",
        "distribution_tail",
    }
    missing_reference = sorted(required_reference.difference(reference.columns))
    if missing_reference:
        raise RawFrontierError(f"Reference table missing columns: {missing_reference}")
    if reference["sample_id"].duplicated().any():
        raise RawFrontierError("Reference sample IDs must be unique")
    merged = treatment[columns].rename(
        columns={
            "y_true": "y_true_t",
            "score": "score_t",
            "score_raw": "score_raw_t",
        }
    ).merge(
        control[columns].rename(
            columns={
                "y_true": "y_true_c",
                "score": "score_c",
                "score_raw": "score_raw_c",
            }
        ),
        on="sample_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise RawFrontierError("Treatment/control prediction sample ID sets differ")
    if not (merged["y_true_t"].astype(int) == merged["y_true_c"].astype(int)).all():
        raise RawFrontierError("Treatment/control prediction labels differ")
    merged = merged.drop(columns="_merge").merge(
        reference[list(required_reference)],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if merged["operational_tail"].isna().any():
        raise RawFrontierError("Reference sample ID set differs from predictions")
    if not (merged["y_true_t"].astype(int) == merged["y_true"].astype(int)).all():
        raise RawFrontierError("Reference labels differ from predictions")
    return merged


def paired_tail_shifts(
    treatment: pd.DataFrame,
    control: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize treatment-control shifts on fixed control-defined tails."""

    merged = _align_pair_with_reference(treatment, control, reference)
    merged["raw_shift"] = merged["score_raw_t"].astype(float) - merged[
        "score_raw_c"
    ].astype(float)
    merged["calibrated_shift"] = merged["score_t"].astype(float) - merged[
        "score_c"
    ].astype(float)
    sample_shifts = merged[
        [
            "sample_id",
            "y_true",
            "operational_tail",
            "distribution_tail",
            "raw_shift",
            "calibrated_shift",
        ]
    ].copy()
    rows: list[dict[str, float | int | str]] = []
    for label_value, label_name in ((0, "normal"), (1, "defect")):
        class_rows = sample_shifts[sample_shifts.y_true.astype(int) == label_value]
        for scope, mask in (
            ("all", np.ones(len(class_rows), dtype=bool)),
            ("operational", class_rows.operational_tail.to_numpy(dtype=bool)),
            ("tail_gap", class_rows.distribution_tail.to_numpy(dtype=bool)),
        ):
            subset = class_rows.loc[mask]
            for score_type, column in (
                ("raw", "raw_shift"),
                ("calibrated", "calibrated_shift"),
            ):
                values = subset[column].to_numpy(dtype=np.float64)
                beneficial = values < 0 if label_value == 0 else values > 0
                harmful = values > 0 if label_value == 0 else values < 0
                rows.append(
                    {
                        "label": label_name,
                        "scope": scope,
                        "score_type": score_type,
                        "n": int(len(values)),
                        "mean_shift": float(values.mean()) if len(values) else np.nan,
                        "median_shift": (
                            float(np.median(values)) if len(values) else np.nan
                        ),
                        "std_shift": (
                            float(values.std(ddof=1)) if len(values) > 1 else 0.0
                        ),
                        "beneficial_rate": (
                            float(beneficial.mean()) if len(values) else np.nan
                        ),
                        "harmful_rate": float(harmful.mean()) if len(values) else np.nan,
                        "neutral_rate": (
                            float((values == 0).mean()) if len(values) else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows), sample_shifts


def paired_frontier_summary(
    treatment_frontier: pd.DataFrame,
    control_frontier: pd.DataFrame,
    *,
    safe_fn_limit: int = 95,
) -> dict[str, float | int | bool]:
    """Compare exact NP frontiers at identical integer false-negative budgets."""

    try:
        compared = compare_frontiers(treatment_frontier, control_frontier)
    except (CanonicalInputError, ValidationError, ValueError, KeyError) as exc:
        raise RawFrontierError(str(exc)) from exc
    maximum = int(compared["fn_budget"].max())
    if safe_fn_limit < 0 or safe_fn_limit > maximum:
        raise RawFrontierError(
            f"safe_fn_limit={safe_fn_limit} is outside 0..{maximum}"
        )
    safe = compared[compared.fn_budget <= safe_fn_limit]
    safe_delta = safe["delta_TN"].to_numpy(dtype=np.int64)
    full_delta = compared["delta_TN"].to_numpy(dtype=np.int64)
    at_zero = int(compared.loc[compared.fn_budget == 0, "delta_TN"].iloc[0])
    at_limit = int(
        compared.loc[compared.fn_budget == safe_fn_limit, "delta_TN"].iloc[0]
    )
    result: dict[str, float | int | bool] = {
        "same_fn_budget_comparison": True,
        "safe_fn_limit": int(safe_fn_limit),
        "delta_TN_at_FN0": at_zero,
        "delta_TN_at_safe_FN": at_limit,
        "safe_frontier_dominant": bool(
            np.all(safe_delta >= 0) and np.any(safe_delta > 0)
        ),
        "full_frontier_dominant": bool(
            np.all(full_delta >= 0) and np.any(full_delta > 0)
        ),
        "safe_positive_budget_share": float((safe_delta > 0).mean()),
        "safe_negative_budget_share": float((safe_delta < 0).mean()),
        "safe_min_delta_TN": int(safe_delta.min()),
        "safe_max_delta_TN": int(safe_delta.max()),
        "safe_mean_delta_TN": float(safe_delta.mean()),
        "full_positive_budget_share": float((full_delta > 0).mean()),
        "full_negative_budget_share": float((full_delta < 0).mean()),
        "full_min_delta_TN": int(full_delta.min()),
        "full_max_delta_TN": int(full_delta.max()),
        "full_mean_delta_TN": float(full_delta.mean()),
    }
    result[f"delta_TN_at_FN{safe_fn_limit}"] = at_limit
    return result


def analyze_canonical_pairs(
    canonical_index: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    safe_fn_limit: int = 95,
    expected_rows: int = 120_000,
    expected_normal: int = 100_000,
    expected_defect: int = 20_000,
    ece_bins: int = 15,
    frontier_sink: Callable[[str, str, pd.DataFrame], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """Stream all canonical triads and emit run/pair mechanism tables.

    At most one triad's three prediction frames and six frontiers are retained
    at once.  ``frontier_sink`` receives a losslessly compacted exact frontier
    for durable storage; when omitted, full frontiers are consumed and then
    released after each triad.
    """

    required = {
        "run_slot",
        "triad_id",
        "arm",
        "prediction_path",
        "phase",
        "condition_id",
        "method",
        "budget",
        "guard_ratio",
        "training_seed",
        "selection_seed",
        "machine_id",
        "resume_count",
        "input_snapshot_id",
        "selection_sha256",
        "release_ref",
        "release_commit",
    }
    missing = sorted(required.difference(canonical_index.columns))
    if missing:
        raise RawFrontierError(f"Canonical prediction index missing columns: {missing}")
    if canonical_index["run_slot"].duplicated().any():
        raise RawFrontierError("Canonical prediction index has duplicate run_slot")
    reference_required = {
        "sample_id",
        "y_true",
        "operational_tail",
        "distribution_tail",
        "control_run_count",
        "reference_source_arms",
        "reference_source_control_count",
        "reference_source_treatment_count",
    }
    missing_reference = sorted(reference_required.difference(reference.columns))
    if missing_reference:
        raise RawFrontierError(f"Reference table missing columns: {missing_reference}")
    treatment_counts = set(
        pd.to_numeric(
            reference["reference_source_treatment_count"], errors="raise"
        ).astype(int)
    )
    if treatment_counts != {0}:
        raise RawFrontierError("Fixed control reference must contain zero treatment runs")
    expected_control_count = int(
        canonical_index.arm.astype(str).isin({"R1", "R2"}).sum()
    )
    source_control_counts = set(
        pd.to_numeric(reference["reference_source_control_count"], errors="raise")
        .astype(int)
        .tolist()
    )
    control_run_counts = set(
        pd.to_numeric(reference["control_run_count"], errors="raise")
        .astype(int)
        .tolist()
    )
    if source_control_counts != {expected_control_count} or control_run_counts != {
        expected_control_count
    }:
        raise RawFrontierError(
            "Fixed reference control count differs from canonical R1/R2 index"
        )
    if set(reference["reference_source_arms"].astype(str)) != {"R1,R2"}:
        raise RawFrontierError("Fixed reference source arms must be exactly R1,R2")

    run_metric_rows: list[pd.DataFrame] = []
    equivalence_rows: list[dict[str, object]] = []
    frontier_rows: list[dict[str, object]] = []
    tail_rows: list[pd.DataFrame] = []
    for triad_id, triad in canonical_index.groupby("triad_id", sort=True):
        arms = triad.set_index(triad.arm.astype(str), drop=False)
        if len(triad) != 3 or set(arms.index) != {"T", "R1", "R2"}:
            raise RawFrontierError(
                f"{triad_id} must contain exactly one T, R1 and R2 canonical run"
            )
        predictions: dict[str, pd.DataFrame] = {}
        frontiers: dict[tuple[str, str], pd.DataFrame] = {}
        for arm in ("T", "R1", "R2"):
            row = arms.loc[arm]
            frame = read_val_op_predictions(
                row.prediction_path,
                expected_rows=expected_rows,
                expected_normal=expected_normal,
                expected_defect=expected_defect,
            )
            predictions[arm] = frame
            diagnostics = probability_diagnostics(frame, ece_bins=ece_bins)
            for column, value in (
                ("run_slot", str(row.run_slot)),
                ("triad_id", str(triad_id)),
                ("arm", arm),
                ("phase", str(row.phase)),
                ("condition_id", str(row.condition_id)),
                ("method", str(row.method)),
                ("budget", int(row.budget)),
                ("guard_ratio", float(row.guard_ratio)),
                ("training_seed", int(row.training_seed)),
                ("selection_seed", int(row.selection_seed)),
                ("machine_id", str(row.machine_id)),
                ("resume_count", int(row.resume_count)),
                ("input_snapshot_id", str(row.input_snapshot_id)),
                ("selection_sha256", str(row.selection_sha256)),
                ("release_ref", str(row.release_ref)),
                ("release_commit", str(row.release_commit)),
            ):
                diagnostics[column] = value
            run_metric_rows.append(diagnostics)
            raw = exact_np_frontier(frame, score_column="score_raw")
            calibrated = exact_np_frontier(frame, score_column="score")
            frontiers[(arm, "raw")] = raw
            frontiers[(arm, "calibrated")] = calibrated
            equality_columns = ["fn_budget", "actual_fn", "TN", "FP"]
            exact = raw[equality_columns].equals(calibrated[equality_columns])
            equivalence_rows.append(
                {
                    "run_slot": str(row.run_slot),
                    "triad_id": str(triad_id),
                    "arm": arm,
                    "raw_calibrated_frontier_exact": bool(exact),
                    "machine_id": str(row.machine_id),
                    "resume_count": int(row.resume_count),
                    "input_snapshot_id": str(row.input_snapshot_id),
                    "raw_breakpoint_count": int(len(compact_frontier(raw))),
                    "calibrated_breakpoint_count": int(
                        len(compact_frontier(calibrated))
                    ),
                }
            )
            if frontier_sink is not None:
                frontier_sink(str(row.run_slot), "raw", compact_frontier(raw))
                frontier_sink(
                    str(row.run_slot), "calibrated", compact_frontier(calibrated)
                )

        treatment_row = arms.loc["T"]
        for control_arm in ("R1", "R2"):
            control_row = arms.loc[control_arm]
            common = {
                "triad_id": str(triad_id),
                "condition_id": str(treatment_row.condition_id),
                "method": str(treatment_row.method),
                "budget": int(treatment_row.budget),
                "guard_ratio": float(treatment_row.guard_ratio),
                "training_seed": int(treatment_row.training_seed),
                "treatment_selection_seed": int(treatment_row.selection_seed),
                "control_selection_seed": int(control_row.selection_seed),
                "treatment_run_slot": str(treatment_row.run_slot),
                "control_run_slot": str(control_row.run_slot),
                "control_arm": control_arm,
                "treatment_machine_id": str(treatment_row.machine_id),
                "control_machine_id": str(control_row.machine_id),
                "same_machine": bool(
                    str(treatment_row.machine_id) == str(control_row.machine_id)
                ),
                "treatment_resume_count": int(treatment_row.resume_count),
                "control_resume_count": int(control_row.resume_count),
                "treatment_input_snapshot_id": str(
                    treatment_row.input_snapshot_id
                ),
                "control_input_snapshot_id": str(control_row.input_snapshot_id),
                "same_input_snapshot": bool(
                    str(treatment_row.input_snapshot_id)
                    == str(control_row.input_snapshot_id)
                ),
            }
            for score_type in ("raw", "calibrated"):
                summary = paired_frontier_summary(
                    frontiers[("T", score_type)],
                    frontiers[(control_arm, score_type)],
                    safe_fn_limit=safe_fn_limit,
                )
                frontier_rows.append({**common, "score_type": score_type, **summary})
            tail_summary, _ = paired_tail_shifts(
                predictions["T"],
                predictions[control_arm],
                reference,
            )
            for column, value in common.items():
                tail_summary[column] = value
            tail_rows.append(tail_summary)

    run_metrics = pd.concat(run_metric_rows, ignore_index=True)
    tails = pd.concat(tail_rows, ignore_index=True)
    return {
        "run_probability_metrics": run_metrics,
        "frontier_equivalence_audit": pd.DataFrame(equivalence_rows),
        "paired_frontier_dominance": pd.DataFrame(frontier_rows),
        "paired_tail_shift_summary": tails,
    }


def _prediction_identity_sha256(predictions: pd.DataFrame) -> str:
    ordered = predictions[["sample_id", "y_true"]].sort_values(
        "sample_id", kind="mergesort"
    )
    digest = hashlib.sha256()
    for row in ordered.itertuples(index=False):
        digest.update(str(row.sample_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(row.y_true)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def analyze_canonical_val_cal(
    canonical_index: pd.DataFrame,
    *,
    expected_rows: int = 120_000,
    expected_normal: int = 100_000,
    expected_defect: int = 20_000,
    ece_bins: int = 15,
) -> dict[str, pd.DataFrame]:
    """Stream and audit all canonical val-cal predictions and Platt models."""

    required = {
        "run_slot",
        "triad_id",
        "arm",
        "phase",
        "condition_id",
        "method",
        "budget",
        "guard_ratio",
        "training_seed",
        "selection_seed",
        "machine_id",
        "resume_count",
        "input_snapshot_id",
        "val_cal_prediction_path",
        "platt_calibration_path",
    }
    missing = sorted(required.difference(canonical_index.columns))
    if missing:
        raise RawFrontierError(f"Canonical val_cal index missing columns: {missing}")
    if canonical_index["run_slot"].duplicated().any():
        raise RawFrontierError("Canonical val_cal index has duplicate run_slot")
    metric_rows: list[pd.DataFrame] = []
    ranking_rows: list[dict[str, object]] = []
    platt_rows: list[dict[str, object]] = []
    canonical_identity_digest: str | None = None
    for row in canonical_index.sort_values("run_slot", kind="stable").itertuples(
        index=False
    ):
        predictions = read_val_op_predictions(
            row.val_cal_prediction_path,
            expected_rows=expected_rows,
            expected_normal=expected_normal,
            expected_defect=expected_defect,
        )
        identity_digest = _prediction_identity_sha256(predictions)
        if canonical_identity_digest is None:
            canonical_identity_digest = identity_digest
        elif identity_digest != canonical_identity_digest:
            raise RawFrontierError(
                f"val_cal sample identity/label set differs for {row.run_slot}"
            )
        metadata: dict[str, object] = {
            "run_slot": str(row.run_slot),
            "triad_id": str(row.triad_id),
            "arm": str(row.arm),
            "phase": str(row.phase),
            "condition_id": str(row.condition_id),
            "method": str(row.method),
            "budget": int(row.budget),
            "guard_ratio": float(row.guard_ratio),
            "training_seed": int(row.training_seed),
            "selection_seed": int(row.selection_seed),
            "machine_id": str(row.machine_id),
            "resume_count": int(row.resume_count),
            "input_snapshot_id": str(row.input_snapshot_id),
            "val_cal_identity_sha256": identity_digest,
        }
        diagnostics = probability_diagnostics(predictions, ece_bins=ece_bins)
        for column, value in metadata.items():
            diagnostics[column] = value
        metric_rows.append(diagnostics)
        ranking_rows.append(
            {**metadata, **raw_calibrated_ranking_audit(predictions)}
        )
        platt_path = Path(str(row.platt_calibration_path))
        try:
            parameters = json.loads(platt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RawFrontierError(
                f"Cannot read Platt calibration for {row.run_slot}: {platt_path}"
            ) from exc
        if not isinstance(parameters, dict):
            raise RawFrontierError(
                f"Platt calibration must be a JSON object for {row.run_slot}"
            )
        platt_rows.append({**metadata, **platt_calibration_audit(predictions, parameters)})
    return {
        "val_cal_probability_metrics": pd.concat(metric_rows, ignore_index=True),
        "val_cal_ranking_frontier_audit": pd.DataFrame(ranking_rows),
        "val_cal_platt_audit": pd.DataFrame(platt_rows),
    }
