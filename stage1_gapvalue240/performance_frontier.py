"""Tie-safe, threshold-invariant performance-frontier analysis.

The functions in this module compare models only at the same false-negative
budget.  They are intentionally independent of the historical fixed FN95 and
TN68253 summaries so a confidence-threshold slide cannot be mistaken for a
model improvement.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .deep_analysis import CanonicalInputError
from .metrics import validate_predictions


REAL_GAIN_CLASSES = frozenset({"DUAL_GAIN", "FN_SAFE_GAIN"})
CONTROLLED_GAIN_CLASSES = frozenset(
    {"CONTROLLED_GAIN_1", "CONTROLLED_GAIN_2", "CONTROLLED_GAIN_5"}
)


def frontier_from_predictions(
    predictions: pd.DataFrame,
    *,
    score_column: str,
) -> pd.DataFrame:
    """Return maximum attainable TN for every integer FN budget.

    Equal scores always move as a whole group.  If several thresholds have the
    same actual FN, only the threshold with maximum TN is retained.
    """

    required = {"sample_id", "y_true", score_column}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise CanonicalInputError(f"Prediction table missing columns: {missing}")
    validated = validate_predictions(
        predictions[["sample_id", "y_true", score_column]].rename(
            columns={score_column: "score"}
        )
    )
    y_true = validated["y_true"].to_numpy(dtype=np.int8)
    scores = validated["score"].to_numpy(dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    ordered_y = y_true[order]
    ordered_scores = scores[order]
    defect_count = int((validated["y_true"] == 1).sum())
    normal_count = int((validated["y_true"] == 0).sum())
    # End indices preserve whole equal-score groups while cumulative sums avoid
    # a Python loop over up to 120,000 unique values per run.
    group_end = np.r_[ordered_scores[1:] != ordered_scores[:-1], True]
    ends = np.flatnonzero(group_end)
    cumulative_tp = np.cumsum(ordered_y == 1, dtype=np.int64)
    cumulative_fp = np.cumsum(ordered_y == 0, dtype=np.int64)
    tp = np.r_[0, cumulative_tp[ends]]
    fp = np.r_[0, cumulative_fp[ends]]
    sweep = pd.DataFrame(
        {
            "threshold": np.r_[np.inf, ordered_scores[ends]],
            "TP": tp,
            "FP": fp,
            "TN": normal_count - fp,
            "FN": defect_count - tp,
            "tie_group_size": np.r_[0, np.diff(np.r_[-1, ends])],
        }
    )
    attainable = (
        sweep.sort_values(
            ["FN", "TN", "threshold"],
            ascending=[True, False, False],
            kind="stable",
        )
        .drop_duplicates("FN", keep="first")
        .sort_values("FN", kind="stable")
        .reset_index(drop=True)
    )
    actual_fn = attainable["FN"].to_numpy(dtype=np.int64)
    budgets = np.arange(defect_count + 1, dtype=np.int64)
    positions = np.searchsorted(actual_fn, budgets, side="right") - 1
    if (positions < 0).any():
        raise AssertionError("Tie-safe sweep did not provide an attainable FN=0 row")
    selected = attainable.iloc[positions].reset_index(drop=True)
    result = pd.DataFrame(
        {
            "fn_budget": budgets,
            "actual_fn": selected["FN"].to_numpy(dtype=np.int64),
            "TN": selected["TN"].to_numpy(dtype=np.int64),
            "FP": selected["FP"].to_numpy(dtype=np.int64),
            "threshold": selected["threshold"].to_numpy(dtype=np.float64),
            "tie_group_size": selected["tie_group_size"].to_numpy(dtype=np.int64),
        }
    )
    if not (result["TN"] + result["FP"] == normal_count).all():
        raise AssertionError("Frontier TN/FP counts do not match normal count")
    return result


def compare_frontiers(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Compare two frontiers at identical integer FN budgets."""

    required = {"fn_budget", "actual_fn", "TN", "FP", "threshold"}
    for name, frame in (("candidate", candidate), ("reference", reference)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise CanonicalInputError(f"{name} frontier missing columns: {missing}")
        if frame["fn_budget"].duplicated().any():
            raise CanonicalInputError(f"{name} frontier has duplicate FN budgets")
    left = candidate.rename(
        columns={column: f"candidate_{column}" for column in candidate.columns}
    )
    right = reference.rename(
        columns={column: f"reference_{column}" for column in reference.columns}
    )
    compared = left.merge(
        right,
        left_on="candidate_fn_budget",
        right_on="reference_fn_budget",
        how="inner",
        validate="one_to_one",
    )
    if len(compared) != len(candidate) or len(compared) != len(reference):
        raise CanonicalInputError("Candidate and reference FN budget grids differ")
    compared.insert(0, "fn_budget", compared["candidate_fn_budget"].astype(int))
    compared["delta_TN"] = compared["candidate_TN"] - compared["reference_TN"]
    compared["delta_FP"] = compared["candidate_FP"] - compared["reference_FP"]
    return compared


def classify_frontier_against_reference(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    baseline_fn: int,
    controlled_margins: Sequence[int] = (1, 2, 5),
) -> dict[str, Any]:
    """Classify candidate performance without comparing unequal FN budgets."""

    compared = compare_frontiers(candidate, reference)
    maximum_budget = int(compared["fn_budget"].max())
    k0 = int(baseline_fn)
    if k0 < 0 or k0 > maximum_budget:
        raise ValueError(f"baseline_fn={k0} is outside 0..{maximum_budget}")
    safe = compared.loc[compared["fn_budget"] <= k0].copy()
    reference_tn0 = int(
        compared.loc[compared["fn_budget"] == k0, "reference_TN"].iloc[0]
    )
    candidate_tn0 = int(
        compared.loc[compared["fn_budget"] == k0, "candidate_TN"].iloc[0]
    )
    dual_gain = bool(
        (
            (compared["fn_budget"] < k0)
            & (compared["candidate_TN"] > reference_tn0)
        ).any()
    )
    fn_safe_gain = candidate_tn0 > reference_tn0
    controlled_margin: int | None = None
    if not dual_gain and not fn_safe_gain:
        for margin in sorted({int(value) for value in controlled_margins}):
            budget = min(k0 + margin, maximum_budget)
            row = compared.loc[compared["fn_budget"] == budget].iloc[0]
            if int(row["candidate_TN"]) > int(row["reference_TN"]):
                controlled_margin = margin
                break

    safe_delta = safe["delta_TN"].to_numpy(dtype=np.int64)
    full_delta = compared["delta_TN"].to_numpy(dtype=np.int64)
    if dual_gain:
        performance_class = "DUAL_GAIN"
    elif fn_safe_gain:
        performance_class = "FN_SAFE_GAIN"
    elif controlled_margin is not None:
        performance_class = f"CONTROLLED_GAIN_{controlled_margin}"
    elif np.all(safe_delta <= 0) and np.any(safe_delta < 0):
        performance_class = "DOMINATED"
    elif np.all(safe_delta == 0):
        performance_class = "EQUIVALENT"
    else:
        performance_class = "CROSSOVER"

    positive_safe = safe_delta > 0
    negative_safe = safe_delta < 0
    return {
        "performance_class": performance_class,
        "baseline_fn": k0,
        "baseline_frontier_TN": reference_tn0,
        "candidate_frontier_TN": candidate_tn0,
        "delta_TN_at_baseline_fn": candidate_tn0 - reference_tn0,
        "dual_gain": dual_gain,
        "fn_safe_gain": fn_safe_gain,
        "controlled_margin": controlled_margin,
        "absolute_baseline_pass": performance_class in REAL_GAIN_CLASSES,
        "safe_frontier_dominant": bool(np.all(safe_delta >= 0) and np.any(positive_safe)),
        "full_frontier_dominant": bool(np.all(full_delta >= 0) and np.any(full_delta > 0)),
        "safe_positive_budget_share": float(positive_safe.mean()),
        "safe_negative_budget_share": float(negative_safe.mean()),
        "safe_min_delta_TN": int(safe_delta.min()),
        "safe_max_delta_TN": int(safe_delta.max()),
        "safe_mean_delta_TN": float(safe_delta.mean()),
        "full_positive_budget_share": float((full_delta > 0).mean()),
        "full_negative_budget_share": float((full_delta < 0).mean()),
        "full_min_delta_TN": int(full_delta.min()),
        "full_max_delta_TN": int(full_delta.max()),
        "full_mean_delta_TN": float(full_delta.mean()),
    }


def build_method_repeatability(run_gates: pd.DataFrame) -> pd.DataFrame:
    """Summarize the baseline-plus-control double gate by condition."""

    required = {
        "experiment_family",
        "condition_id",
        "training_seed",
        "absolute_baseline_pass",
        "paired_control_pass",
    }
    missing = sorted(required.difference(run_gates.columns))
    if missing:
        raise CanonicalInputError(f"Run gate table missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    for (family, condition), group in run_gates.groupby(
        ["experiment_family", "condition_id"], sort=True, dropna=False
    ):
        if group["training_seed"].duplicated().any():
            raise CanonicalInputError(
                f"{family}/{condition} contains duplicate training seeds"
            )
        absolute = group["absolute_baseline_pass"].astype(bool)
        paired = group["paired_control_pass"].astype(bool)
        double = absolute & paired
        if {"safe_frontier_dominant", "paired_safe_frontier_pass"}.issubset(
            group.columns
        ):
            robust = (
                group["safe_frontier_dominant"].astype(bool)
                & group["paired_safe_frontier_pass"].astype(bool)
            )
        else:
            robust = pd.Series(False, index=group.index)
        seeds = int(len(group))
        passes = int(double.sum())
        majority_required = seeds // 2 + 1
        if seeds < 2:
            repeatability = (
                "SINGLE_SEED_PROMISING" if passes == 1 else "SINGLE_SEED_NOT_STRONG"
            )
        elif passes >= majority_required and passes >= 2:
            repeatability = "REPEATABLE_STRONG"
        elif passes > 0:
            repeatability = "PROMISING_BUT_UNSTABLE"
        else:
            repeatability = "UNSTABLE_OR_HARMFUL"
        rows.append(
            {
                "experiment_family": str(family),
                "condition_id": str(condition),
                "seed_count": seeds,
                "majority_required": majority_required,
                "absolute_baseline_passes": int(absolute.sum()),
                "paired_control_passes": int(paired.sum()),
                "double_gate_passes": passes,
                "double_gate_rate": passes / seeds,
                "repeatability_class": repeatability,
                "robust_double_gate_passes": int(robust.sum()),
                "robust_double_gate_rate": float(robust.mean()),
                "robust_repeatability_class": (
                    "REPEATABLE_ROBUST"
                    if seeds >= 2 and int(robust.sum()) >= majority_required and int(robust.sum()) >= 2
                    else "SINGLE_SEED_ROBUST"
                    if seeds == 1 and bool(robust.iloc[0])
                    else "NOT_REPEATABLE_ROBUST"
                ),
            }
        )
    return pd.DataFrame(rows)
