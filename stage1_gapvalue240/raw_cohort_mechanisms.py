"""Outcome-stratified raw/calibrated mechanism analysis for all 80 triads.

The four scientific cohorts are deliberately not forced into one exclusive
partition.  ``HIGH_VALUE`` overlaps ``DUAL_IMPROVEMENT`` for twelve triads,
while ``MIXED`` means neither strict dual improvement nor strict dual harm.

Every feature in this module is measured after training.  The tables explain
where a successful model changed its ordering; they are not eligible as
pre-training sample-selection predictors.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


class RawCohortMechanismError(RuntimeError):
    """Raised when the canonical 80-triad mechanism evidence is incomplete."""


COHORTS = ("DUAL_IMPROVEMENT", "HIGH_VALUE", "DUAL_HARM", "MIXED")
CONTRAST_COHORTS = ("DUAL_IMPROVEMENT", "HIGH_VALUE", "MIXED")
SCORE_TYPES = ("raw", "calibrated")
CONTROL_ARMS = ("R1", "R2")


def _feature_specs() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, scope, fixed_set, expected_n in (
        ("normal", "operational", "fixed_operational_normal_tail", 31_747),
        ("defect", "operational", "fixed_95_weak_defect_tail", 95),
        ("normal", "tail_gap", "fixed_normal_top_10pct", 10_000),
        ("defect", "tail_gap", "fixed_defect_bottom_5pct", 1_000),
    ):
        for statistic in ("mean_shift", "median_shift", "beneficial_rate"):
            rows.append(
                {
                    "feature": f"raw_tail__{label}_{scope}__{statistic}",
                    "feature_family": (
                        "NORMAL_FIXED_TAIL" if label == "normal" else "DEFECT_FIXED_TAIL"
                    ),
                    "source_table": "raw_frontier_paired_tail_shift_summary.csv",
                    "fixed_set": fixed_set,
                    "expected_set_size": expected_n,
                    "beneficial_direction": (
                        1 if statistic == "beneficial_rate" or label == "defect" else -1
                    ),
                    "interpretation": (
                        "lower normal score is beneficial"
                        if label == "normal" and statistic != "beneficial_rate"
                        else "higher value is beneficial"
                    ),
                }
            )
    for metric, direction in (
        ("auroc", 1),
        ("auprc", 1),
        ("brier", -1),
        ("log_loss", -1),
        ("ece", -1),
    ):
        rows.append(
            {
                "feature": f"probability__delta_{metric}",
                "feature_family": "PROBABILITY_QUALITY",
                "source_table": "raw_frontier_run_probability_metrics.csv",
                "fixed_set": "all_val_op_samples",
                "expected_set_size": 120_000,
                "beneficial_direction": direction,
                "interpretation": "higher is beneficial" if direction > 0 else "lower is beneficial",
            }
        )
    for metric in (
        "safe_positive_budget_share",
        "safe_min_delta_TN",
        "safe_mean_delta_TN",
        "delta_TN_at_FN95",
        "safe_frontier_dominant",
    ):
        rows.append(
            {
                "feature": f"frontier__{metric}",
                "feature_family": "SAFE_FRONTIER",
                "source_table": "raw_frontier_paired_dominance.csv",
                "fixed_set": "same_FN_budget_0_to_95",
                "expected_set_size": 96,
                "beneficial_direction": 1,
                "interpretation": "higher is beneficial",
            }
        )
    result = pd.DataFrame(rows)
    result["analysis_role"] = "POSTTRAINING_OUTCOME_MECHANISM"
    result["allowed_as_pretraining_predictor"] = False
    return result


FEATURE_DICTIONARY = _feature_specs()
FEATURES = tuple(FEATURE_DICTIONARY.feature.astype(str))


def _as_bool(series: pd.Series, *, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise RawCohortMechanismError(f"{column} contains non-boolean values")
    return normalized.isin({"true", "1"})


def attach_cohort_memberships(
    outcomes: pd.DataFrame,
    *,
    expected_triads: int = 80,
) -> pd.DataFrame:
    """Expand triads into the four preregistered, intentionally overlapping cohorts."""

    required = {
        "triad_id",
        "condition_id",
        "training_seed",
        "dual_improvement",
        "high_value",
        "dual_harm",
    }
    missing = sorted(required.difference(outcomes.columns))
    if missing:
        raise RawCohortMechanismError(f"Triad outcomes missing columns: {missing}")
    if len(outcomes) != expected_triads or outcomes.triad_id.nunique() != expected_triads:
        raise RawCohortMechanismError(
            f"Expected {expected_triads} unique triads, found {len(outcomes)} rows / "
            f"{outcomes.triad_id.nunique()} IDs"
        )
    frame = outcomes.copy()
    for column in ("dual_improvement", "high_value", "dual_harm"):
        frame[column] = _as_bool(frame[column], column=column)
    if (frame.dual_improvement & frame.dual_harm).any():
        raise RawCohortMechanismError("A triad cannot be both dual improvement and dual harm")
    if (frame.high_value & frame.dual_harm).any():
        raise RawCohortMechanismError("A triad cannot be both high value and dual harm")

    rows: list[pd.DataFrame] = []
    membership_masks = {
        "DUAL_IMPROVEMENT": frame.dual_improvement,
        "HIGH_VALUE": frame.high_value,
        "DUAL_HARM": frame.dual_harm,
        "MIXED": ~(frame.dual_improvement | frame.dual_harm),
    }
    identity_columns = [
        column
        for column in (
            "triad_id",
            "phase",
            "condition_id",
            "method",
            "budget",
            "guard_ratio",
            "training_seed",
            "treatment_selection_seed",
            "dual_improvement",
            "high_value",
            "dual_harm",
        )
        if column in frame.columns
    ]
    for cohort, mask in membership_masks.items():
        selected = frame.loc[mask, identity_columns].copy()
        selected["cohort"] = cohort
        rows.append(selected)
    result = pd.concat(rows, ignore_index=True)
    if result.empty or result.duplicated(["triad_id", "cohort"]).any():
        raise RawCohortMechanismError("Cohort membership is empty or duplicated")
    return result.sort_values(["cohort", "triad_id"], kind="stable", ignore_index=True)


def _require_unique(frame: pd.DataFrame, keys: list[str], *, name: str) -> None:
    if frame.duplicated(keys).any():
        examples = frame.loc[frame.duplicated(keys, keep=False), keys].head().to_dict("records")
        raise RawCohortMechanismError(f"{name} contains duplicate keys {keys}: {examples}")


def build_pair_mechanism_features(
    outcomes: pd.DataFrame,
    tails: pd.DataFrame,
    dominance: pd.DataFrame,
    probability_metrics: pd.DataFrame,
    *,
    expected_triads: int = 80,
) -> pd.DataFrame:
    """Build one wide mechanism row per triad/control/score-type comparison."""

    membership = attach_cohort_memberships(outcomes, expected_triads=expected_triads)
    del membership  # validates cohort logic; wide rows retain triad-level booleans.
    tail_required = {
        "triad_id",
        "control_arm",
        "score_type",
        "label",
        "scope",
        "n",
        "mean_shift",
        "median_shift",
        "beneficial_rate",
    }
    dominance_required = {
        "triad_id",
        "control_arm",
        "score_type",
        "safe_positive_budget_share",
        "safe_min_delta_TN",
        "safe_mean_delta_TN",
        "delta_TN_at_FN95",
        "safe_frontier_dominant",
    }
    probability_required = {
        "triad_id",
        "run_slot",
        "arm",
        "score_type",
        "auroc",
        "auprc",
        "brier",
        "log_loss",
        "ece",
    }
    for name, frame, required in (
        ("tail shifts", tails, tail_required),
        ("frontier dominance", dominance, dominance_required),
        ("probability metrics", probability_metrics, probability_required),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise RawCohortMechanismError(f"{name} missing columns: {missing}")

    pair_keys = ["triad_id", "control_arm", "score_type"]
    base = dominance.copy()
    base = base[base.control_arm.astype(str).isin(CONTROL_ARMS)].copy()
    base = base[base.score_type.astype(str).isin(SCORE_TYPES)].copy()
    _require_unique(base, pair_keys, name="frontier dominance")
    expected_pairs = expected_triads * len(CONTROL_ARMS) * len(SCORE_TYPES)
    if len(base) != expected_pairs:
        raise RawCohortMechanismError(
            f"Expected {expected_pairs} frontier pairs, found {len(base)}"
        )
    frontier_columns = [
        "safe_positive_budget_share",
        "safe_min_delta_TN",
        "safe_mean_delta_TN",
        "delta_TN_at_FN95",
        "safe_frontier_dominant",
    ]
    base = base[pair_keys + frontier_columns].rename(
        columns={column: f"frontier__{column}" for column in frontier_columns}
    )
    base["frontier__safe_frontier_dominant"] = base[
        "frontier__safe_frontier_dominant"
    ].astype(float)

    selected_tails = tails[
        tails.label.astype(str).isin({"normal", "defect"})
        & tails.scope.astype(str).isin({"operational", "tail_gap"})
        & tails.control_arm.astype(str).isin(CONTROL_ARMS)
        & tails.score_type.astype(str).isin(SCORE_TYPES)
    ].copy()
    tail_keys = pair_keys + ["label", "scope"]
    _require_unique(selected_tails, tail_keys, name="selected tail shifts")
    if len(selected_tails) != expected_pairs * 4:
        raise RawCohortMechanismError(
            f"Expected {expected_pairs * 4} selected tail rows, found {len(selected_tails)}"
        )
    expected_sizes = {
        ("normal", "operational"): 31_747,
        ("defect", "operational"): 95,
        ("normal", "tail_gap"): 10_000,
        ("defect", "tail_gap"): 1_000,
    }
    # Synthetic unit tests may use smaller non-canonical tables.  The real 80-triad
    # run is strict about every fixed-tail size.
    if expected_triads == 80:
        for key, expected_n in expected_sizes.items():
            observed = selected_tails.loc[
                (selected_tails.label.astype(str) == key[0])
                & (selected_tails.scope.astype(str) == key[1]),
                "n",
            ]
            if not observed.astype(int).eq(expected_n).all():
                raise RawCohortMechanismError(
                    f"Fixed tail {key} must contain {expected_n} samples in every pair"
                )
    tail_wide_parts: list[pd.DataFrame] = []
    for (label, scope), group in selected_tails.groupby(["label", "scope"], sort=True):
        renamed = group[pair_keys + ["mean_shift", "median_shift", "beneficial_rate"]].copy()
        renamed = renamed.rename(
            columns={
                metric: f"raw_tail__{label}_{scope}__{metric}"
                for metric in ("mean_shift", "median_shift", "beneficial_rate")
            }
        )
        tail_wide_parts.append(renamed)
    for part in tail_wide_parts:
        base = base.merge(part, on=pair_keys, how="left", validate="one_to_one")

    metrics = probability_metrics.copy()
    metrics = metrics[
        metrics.arm.astype(str).isin({"T", *CONTROL_ARMS})
        & metrics.score_type.astype(str).isin(SCORE_TYPES)
    ].copy()
    _require_unique(metrics, ["triad_id", "arm", "score_type"], name="probability metrics")
    expected_metric_rows = expected_triads * 3 * len(SCORE_TYPES)
    if len(metrics) != expected_metric_rows:
        raise RawCohortMechanismError(
            f"Expected {expected_metric_rows} probability rows, found {len(metrics)}"
        )
    probability_columns = ["auroc", "auprc", "brier", "log_loss", "ece"]
    treatment = metrics.loc[metrics.arm.astype(str) == "T", ["triad_id", "score_type", *probability_columns]]
    probability_pair_parts: list[pd.DataFrame] = []
    for control_arm in CONTROL_ARMS:
        control = metrics.loc[
            metrics.arm.astype(str) == control_arm,
            ["triad_id", "score_type", *probability_columns],
        ]
        merged = treatment.merge(
            control,
            on=["triad_id", "score_type"],
            suffixes=("__T", "__control"),
            validate="one_to_one",
        )
        merged["control_arm"] = control_arm
        for column in probability_columns:
            merged[f"probability__delta_{column}"] = (
                merged[f"{column}__T"] - merged[f"{column}__control"]
            )
        probability_pair_parts.append(
            merged[[*pair_keys, *[f"probability__delta_{column}" for column in probability_columns]]]
        )
    probability_pairs = pd.concat(probability_pair_parts, ignore_index=True)
    _require_unique(probability_pairs, pair_keys, name="paired probability differences")
    base = base.merge(
        probability_pairs,
        on=pair_keys,
        how="left",
        validate="one_to_one",
    )

    identity_columns = [
        column
        for column in (
            "triad_id",
            "phase",
            "condition_id",
            "method",
            "budget",
            "guard_ratio",
            "training_seed",
            "treatment_selection_seed",
            "dual_improvement",
            "high_value",
            "dual_harm",
        )
        if column in outcomes.columns
    ]
    identity = outcomes[identity_columns].copy()
    _require_unique(identity, ["triad_id"], name="triad outcomes")
    base = base.merge(identity, on="triad_id", how="left", validate="many_to_one")
    missing_values = base[list(FEATURES)].isna().sum()
    if int(missing_values.sum()) != 0:
        raise RawCohortMechanismError(
            f"Pair mechanism feature join left missing values: {missing_values[missing_values > 0].to_dict()}"
        )
    return base.sort_values(pair_keys, kind="stable", ignore_index=True)


def _cluster_bootstrap_ci(
    frame: pd.DataFrame,
    *,
    cohort_a: str,
    cohort_b: str,
    value_column: str,
    cluster_column: str,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float, int, int]:
    clusters = sorted(frame[cluster_column].dropna().astype(str).unique())
    if len(clusters) < 2:
        return np.nan, np.nan, 0, len(clusters)
    sums_a: list[float] = []
    counts_a: list[int] = []
    sums_b: list[float] = []
    counts_b: list[int] = []
    for cluster in clusters:
        block = frame[frame[cluster_column].astype(str) == cluster]
        values_a = block.loc[block.cohort == cohort_a, value_column].to_numpy(dtype=float)
        values_b = block.loc[block.cohort == cohort_b, value_column].to_numpy(dtype=float)
        sums_a.append(float(values_a.sum()))
        counts_a.append(int(len(values_a)))
        sums_b.append(float(values_b.sum()))
        counts_b.append(int(len(values_b)))
    indices = rng.integers(0, len(clusters), size=(resamples, len(clusters)))
    sum_a = np.asarray(sums_a)[indices].sum(axis=1)
    count_a = np.asarray(counts_a)[indices].sum(axis=1)
    sum_b = np.asarray(sums_b)[indices].sum(axis=1)
    count_b = np.asarray(counts_b)[indices].sum(axis=1)
    valid = (count_a > 0) & (count_b > 0)
    sampled = sum_a[valid] / count_a[valid] - sum_b[valid] / count_b[valid]
    if len(sampled) == 0:
        return np.nan, np.nan, 0, len(clusters)
    return (
        float(np.quantile(sampled, 0.025)),
        float(np.quantile(sampled, 0.975)),
        int(len(sampled)),
        len(clusters),
    )


def seed_stratified_permutation(
    frame: pd.DataFrame,
    *,
    cohort_a: str,
    cohort_b: str,
    value_column: str,
    permutations: int = 20_000,
    bootstrap_resamples: int = 5_000,
    random_seed: int = 20260806,
) -> dict[str, float | int]:
    """Contrast cohorts using within-seed permutations and two cluster bootstraps."""

    required = {"training_seed", "condition_id", "cohort", value_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RawCohortMechanismError(f"Contrast frame missing columns: {missing}")
    if permutations <= 0 or bootstrap_resamples <= 0:
        raise ValueError("permutations and bootstrap_resamples must be positive")
    working = frame[frame.cohort.astype(str).isin({cohort_a, cohort_b})].copy()
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
    working = working[np.isfinite(working[value_column].to_numpy(dtype=float))]
    if working.empty:
        raise RawCohortMechanismError("No finite observations for cohort contrast")
    if "triad_id" in working and working.duplicated(["triad_id", "cohort"]).any():
        # A triad can appear in multiple scientific cohorts, but never twice in
        # the same requested contrast cohort.
        raise RawCohortMechanismError("Contrast contains duplicate cohort observations")

    blocks: list[tuple[np.ndarray, int]] = []
    observed_seed_differences: list[float] = []
    for _, block in working.groupby("training_seed", sort=True):
        a = block.loc[block.cohort == cohort_a, value_column].to_numpy(dtype=float)
        b = block.loc[block.cohort == cohort_b, value_column].to_numpy(dtype=float)
        if len(a) == 0 or len(b) == 0:
            continue
        values = np.concatenate([a, b])
        blocks.append((values, len(a)))
        observed_seed_differences.append(float(a.mean() - b.mean()))
    if not blocks:
        raise RawCohortMechanismError(
            f"No training seed contains both {cohort_a} and {cohort_b}"
        )
    observed = float(np.mean(observed_seed_differences))
    rng = np.random.default_rng(random_seed)
    permuted = np.zeros(permutations, dtype=float)
    for values, a_count in blocks:
        random_order = np.argpartition(
            rng.random((permutations, len(values))), a_count - 1, axis=1
        )[:, :a_count]
        a_sum = values[random_order].sum(axis=1)
        total = float(values.sum())
        b_count = len(values) - a_count
        permuted += a_sum / a_count - (total - a_sum) / b_count
    permuted /= len(blocks)
    p_value = float(
        (np.count_nonzero(np.abs(permuted) >= abs(observed) - 1e-15) + 1)
        / (permutations + 1)
    )
    seed_low, seed_high, seed_valid, seed_clusters = _cluster_bootstrap_ci(
        working,
        cohort_a=cohort_a,
        cohort_b=cohort_b,
        value_column=value_column,
        cluster_column="training_seed",
        resamples=bootstrap_resamples,
        rng=np.random.default_rng(random_seed + 1),
    )
    condition_low, condition_high, condition_valid, condition_clusters = _cluster_bootstrap_ci(
        working,
        cohort_a=cohort_a,
        cohort_b=cohort_b,
        value_column=value_column,
        cluster_column="condition_id",
        resamples=bootstrap_resamples,
        rng=np.random.default_rng(random_seed + 2),
    )
    values_a = working.loc[working.cohort == cohort_a, value_column]
    values_b = working.loc[working.cohort == cohort_b, value_column]
    return {
        "cohort_a_n": int(len(values_a)),
        "cohort_b_n": int(len(values_b)),
        "cohort_a_mean": float(values_a.mean()),
        "cohort_b_mean": float(values_b.mean()),
        "unblocked_mean_difference": float(values_a.mean() - values_b.mean()),
        "seed_blocked_mean_difference": observed,
        "eligible_seed_count": int(len(blocks)),
        "permutation_count": int(permutations),
        "permutation_p_two_sided": p_value,
        "seed_cluster_count": int(seed_clusters),
        "seed_cluster_bootstrap_valid": int(seed_valid),
        "seed_cluster_bootstrap_ci_low": seed_low,
        "seed_cluster_bootstrap_ci_high": seed_high,
        "condition_cluster_count": int(condition_clusters),
        "condition_cluster_bootstrap_valid": int(condition_valid),
        "condition_cluster_bootstrap_ci_low": condition_low,
        "condition_cluster_bootstrap_ci_high": condition_high,
    }


def _bh_adjust(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().astype(float)
    if valid.empty:
        return result
    ordered_index = valid.sort_values(kind="stable").index
    ordered = valid.loc[ordered_index].to_numpy(dtype=float)
    adjusted = ordered * len(ordered) / np.arange(1, len(ordered) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[ordered_index] = np.minimum(adjusted, 1.0)
    return result


def expand_pair_cohorts(pair_features: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    identity = membership[["triad_id", "cohort"]].copy()
    return pair_features.merge(identity, on="triad_id", how="inner", validate="many_to_many")


def summarize_cohort_mechanisms(expanded: pd.DataFrame) -> pd.DataFrame:
    specs = FEATURE_DICTIONARY.set_index("feature")
    rows: list[dict[str, Any]] = []
    for (cohort, control_arm, score_type), group in expanded.groupby(
        ["cohort", "control_arm", "score_type"], sort=True
    ):
        for feature in FEATURES:
            values = pd.to_numeric(group[feature], errors="coerce").dropna()
            spec = specs.loc[feature]
            rows.append(
                {
                    "cohort": cohort,
                    "control_arm": control_arm,
                    "score_type": score_type,
                    "feature": feature,
                    "feature_family": spec.feature_family,
                    "beneficial_direction": int(spec.beneficial_direction),
                    "analysis_role": spec.analysis_role,
                    "allowed_as_pretraining_predictor": False,
                    "triad_count": int(group.triad_id.nunique()),
                    "seed_count": int(group.training_seed.nunique()),
                    "condition_count": int(group.condition_id.nunique()),
                    "valid_n": int(len(values)),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "oriented_mean_higher_is_better": float(
                        values.mean() * int(spec.beneficial_direction)
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_cohort_contrasts(
    expanded: pd.DataFrame,
    *,
    permutations: int = 20_000,
    bootstrap_resamples: int = 5_000,
    random_seed: int = 20260806,
) -> pd.DataFrame:
    specs = FEATURE_DICTIONARY.set_index("feature")
    rows: list[dict[str, Any]] = []
    for cohort_a in CONTRAST_COHORTS:
        for control_arm in CONTROL_ARMS:
            for score_type in SCORE_TYPES:
                subset = expanded[
                    (expanded.control_arm.astype(str) == control_arm)
                    & (expanded.score_type.astype(str) == score_type)
                    & expanded.cohort.astype(str).isin({cohort_a, "DUAL_HARM"})
                ].copy()
                for feature_index, feature in enumerate(FEATURES):
                    seed_material = f"{cohort_a}|{control_arm}|{score_type}|{feature}|{random_seed}"
                    derived_seed = int.from_bytes(
                        hashlib.sha256(seed_material.encode("utf-8")).digest()[:4], "big"
                    )
                    inference = seed_stratified_permutation(
                        subset,
                        cohort_a=cohort_a,
                        cohort_b="DUAL_HARM",
                        value_column=feature,
                        permutations=permutations,
                        bootstrap_resamples=bootstrap_resamples,
                        random_seed=derived_seed + feature_index,
                    )
                    spec = specs.loc[feature]
                    direction = int(spec.beneficial_direction)
                    seed_low = float(inference["seed_cluster_bootstrap_ci_low"])
                    seed_high = float(inference["seed_cluster_bootstrap_ci_high"])
                    condition_low = float(inference["condition_cluster_bootstrap_ci_low"])
                    condition_high = float(inference["condition_cluster_bootstrap_ci_high"])
                    rows.append(
                        {
                            "contrast": f"{cohort_a}_MINUS_DUAL_HARM",
                            "cohort_a": cohort_a,
                            "cohort_b": "DUAL_HARM",
                            "control_arm": control_arm,
                            "score_type": score_type,
                            "feature": feature,
                            "feature_family": spec.feature_family,
                            "beneficial_direction": direction,
                            "analysis_role": spec.analysis_role,
                            "allowed_as_pretraining_predictor": False,
                            **inference,
                            "oriented_seed_blocked_difference_higher_is_better": (
                                float(inference["seed_blocked_mean_difference"]) * direction
                            ),
                            "oriented_seed_cluster_ci_low": (
                                seed_low if direction > 0 else -seed_high
                            ),
                            "oriented_seed_cluster_ci_high": (
                                seed_high if direction > 0 else -seed_low
                            ),
                            "oriented_condition_cluster_ci_low": (
                                condition_low if direction > 0 else -condition_high
                            ),
                            "oriented_condition_cluster_ci_high": (
                                condition_high if direction > 0 else -condition_low
                            ),
                        }
                    )
    result = pd.DataFrame(rows)
    result["fdr_q_global"] = _bh_adjust(result.permutation_p_two_sided)
    result["fdr_q_within_family"] = np.nan
    for _, indices in result.groupby("feature_family", sort=True).groups.items():
        result.loc[indices, "fdr_q_within_family"] = _bh_adjust(
            result.loc[indices, "permutation_p_two_sided"]
        )
    return result


def build_scoretype_differences(expanded: pd.DataFrame) -> pd.DataFrame:
    keys = ["triad_id", "cohort", "control_arm"]
    rows: list[dict[str, Any]] = []
    specs = FEATURE_DICTIONARY.set_index("feature")
    for feature in FEATURES:
        pivot = expanded.pivot(index=keys, columns="score_type", values=feature).reset_index()
        if not set(SCORE_TYPES).issubset(pivot.columns):
            raise RawCohortMechanismError(f"Raw/calibrated rows missing for {feature}")
        pivot["raw_minus_calibrated"] = pivot["raw"] - pivot["calibrated"]
        for (cohort, control_arm), group in pivot.groupby(["cohort", "control_arm"], sort=True):
            values = group.raw_minus_calibrated.astype(float)
            spec = specs.loc[feature]
            rows.append(
                {
                    "cohort": cohort,
                    "control_arm": control_arm,
                    "feature": feature,
                    "feature_family": spec.feature_family,
                    "beneficial_direction": int(spec.beneficial_direction),
                    "triad_count": int(len(group)),
                    "raw_mean": float(group.raw.mean()),
                    "calibrated_mean": float(group.calibrated.mean()),
                    "raw_minus_calibrated_mean": float(values.mean()),
                    "raw_minus_calibrated_median": float(values.median()),
                    "raw_minus_calibrated_std": (
                        float(values.std(ddof=1)) if len(values) > 1 else np.nan
                    ),
                    "analysis_role": "RAW_CALIBRATED_OUTCOME_DIAGNOSTIC",
                    "allowed_as_pretraining_predictor": False,
                }
            )
    return pd.DataFrame(rows)


def load_required_csv(path: str | Path, required: Iterable[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise RawCohortMechanismError(f"Required table missing: {path}")
    frame = pd.read_csv(path, low_memory=False)
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise RawCohortMechanismError(f"{path.name} missing columns: {missing}")
    return frame


def run_raw_cohort_mechanism_analysis(
    tables_dir: str | Path,
    *,
    permutations: int = 20_000,
    bootstrap_resamples: int = 5_000,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build compact all-80-triad cohort mechanism tables from frozen evidence."""

    root = Path(tables_dir)
    outcomes = load_required_csv(
        root / "triad_outcomes_80.csv",
        {"triad_id", "condition_id", "training_seed", "dual_improvement", "high_value", "dual_harm"},
    )
    tails = load_required_csv(
        root / "raw_frontier_paired_tail_shift_summary.csv",
        {"triad_id", "control_arm", "score_type", "label", "scope", "n", "mean_shift", "median_shift", "beneficial_rate"},
    )
    dominance = load_required_csv(
        root / "raw_frontier_paired_dominance.csv",
        {"triad_id", "control_arm", "score_type", "safe_positive_budget_share", "safe_min_delta_TN", "safe_mean_delta_TN", "delta_TN_at_FN95", "safe_frontier_dominant"},
    )
    probability = load_required_csv(
        root / "raw_frontier_run_probability_metrics.csv",
        {"triad_id", "run_slot", "arm", "score_type", "auroc", "auprc", "brier", "log_loss", "ece"},
    )
    membership = attach_cohort_memberships(outcomes)
    pairs = build_pair_mechanism_features(outcomes, tails, dominance, probability)
    expanded = expand_pair_cohorts(pairs, membership)
    cohort_summaries = summarize_cohort_mechanisms(expanded)
    contrasts = build_cohort_contrasts(
        expanded,
        permutations=permutations,
        bootstrap_resamples=bootstrap_resamples,
    )
    scoretype = build_scoretype_differences(expanded)

    membership_counts = {
        cohort: int((membership.cohort == cohort).sum()) for cohort in COHORTS
    }
    raw_good_harm = contrasts[
        (contrasts.cohort_a == "DUAL_IMPROVEMENT")
        & (contrasts.score_type == "raw")
    ].copy()

    def evidence(feature: str, control_arm: str) -> dict[str, Any]:
        row = raw_good_harm[
            (raw_good_harm.feature == feature)
            & (raw_good_harm.control_arm == control_arm)
        ]
        if len(row) != 1:
            raise RawCohortMechanismError(
                f"Missing unique raw good-vs-harm evidence for {feature}/{control_arm}"
            )
        record = row.iloc[0]
        return {
            "seed_blocked_good_minus_harm": float(record.seed_blocked_mean_difference),
            "oriented_higher_is_better": float(
                record.oriented_seed_blocked_difference_higher_is_better
            ),
            "permutation_p_two_sided": float(record.permutation_p_two_sided),
            "fdr_q_global": float(record.fdr_q_global),
            "fdr_q_within_family": float(record.fdr_q_within_family),
            "seed_cluster_ci": [
                float(record.seed_cluster_bootstrap_ci_low),
                float(record.seed_cluster_bootstrap_ci_high),
            ],
            "condition_cluster_ci": [
                float(record.condition_cluster_bootstrap_ci_low),
                float(record.condition_cluster_bootstrap_ci_high),
            ],
        }

    evidence_features = {
        "normal_operational_mean_shift": "raw_tail__normal_operational__mean_shift",
        "defect_operational_mean_shift": "raw_tail__defect_operational__mean_shift",
        "defect_operational_beneficial_rate": "raw_tail__defect_operational__beneficial_rate",
        "normal_top10pct_mean_shift": "raw_tail__normal_tail_gap__mean_shift",
        "defect_bottom5pct_mean_shift": "raw_tail__defect_tail_gap__mean_shift",
        "auroc": "probability__delta_auroc",
        "safe_positive_budget_share": "frontier__safe_positive_budget_share",
        "safe_min_delta_TN": "frontier__safe_min_delta_TN",
        "safe_mean_delta_TN": "frontier__safe_mean_delta_TN",
        "TN_at_FN95": "frontier__delta_TN_at_FN95",
    }
    key_evidence = {
        label: {arm: evidence(feature, arm) for arm in CONTROL_ARMS}
        for label, feature in evidence_features.items()
    }
    normal_oriented = [
        key_evidence["normal_operational_mean_shift"][arm]["oriented_higher_is_better"]
        for arm in CONTROL_ARMS
    ]
    defect_oriented = [
        key_evidence["defect_operational_mean_shift"][arm]["oriented_higher_is_better"]
        for arm in CONTROL_ARMS
    ]
    defect_rate_oriented = [
        key_evidence["defect_operational_beneficial_rate"][arm]["oriented_higher_is_better"]
        for arm in CONTROL_ARMS
    ]
    summary = {
        "status": "COMPLETE",
        "canonical_scope": "80 triads / 240 canonical runs / 160 T-control pairs",
        "cohort_membership_is_nonexclusive": True,
        "cohort_counts": membership_counts,
        "pair_feature_rows": int(len(pairs)),
        "expanded_pair_cohort_rows": int(len(expanded)),
        "feature_count": int(len(FEATURES)),
        "contrast_test_count": int(len(contrasts)),
        "permutations_per_test": int(permutations),
        "bootstrap_resamples_per_cluster_type": int(bootstrap_resamples),
        "global_fdr_significant_count": int((contrasts.fdr_q_global < 0.05).sum()),
        "within_family_fdr_significant_count": int(
            (contrasts.fdr_q_within_family < 0.05).sum()
        ),
        "key_raw_dual_improvement_vs_dual_harm": key_evidence,
        "defect_protection_direction_consistent_across_R1_R2": bool(
            all(value > 0 for value in defect_oriented)
            and all(value > 0 for value in defect_rate_oriented)
        ),
        "blanket_normal_suppression_explains_success": bool(
            all(value > 0 for value in normal_oriented)
        ),
        "mechanism_conclusion": (
            "Cohort differences are post-training outcome mechanisms. Defect-tail "
            "protection is assessed jointly with normal-tail movement and the full "
            "same-FN safe frontier; none of these fields may be reused as a pretraining predictor."
        ),
        "raw_calibrated_boundary": (
            "Raw and calibrated rows are reported separately. Platt mapping and clipping "
            "can alter ties and probability-quality metrics; calibrated evidence is not "
            "silently substituted for raw model ordering."
        ),
        "statistical_boundary": (
            "Cohorts are defined by observed outcomes, HIGH_VALUE overlaps DUAL_IMPROVEMENT, "
            "and tests are explanatory rather than prospective validation. Seed-stratified "
            "permutations and seed/condition cluster bootstraps reduce but cannot remove "
            "machine, seed and condition confounding."
        ),
        "source_tables_read_only": True,
    }
    tables = {
        "raw_cohort_mechanism_membership.csv": membership,
        "raw_cohort_mechanism_feature_dictionary.csv": FEATURE_DICTIONARY.copy(),
        "raw_cohort_mechanism_pair_features.csv": pairs,
        "raw_cohort_mechanism_cohort_summaries.csv": cohort_summaries,
        "raw_cohort_mechanism_contrasts.csv": contrasts,
        "raw_cohort_mechanism_scoretype_differences.csv": scoretype,
    }
    return tables, summary
