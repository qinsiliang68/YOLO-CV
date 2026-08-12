"""Grouped inference and leakage-safe validation for the frozen 240-run study.

The independent scientific unit is one T/R1/R2 triad.  This module never
pretends that the two control comparisons are independent observations, and it
never uses Phase-C outcomes to train or tune a candidate rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CONTROL_ORDER: tuple[str, str] = ("R1", "R2")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def build_joint_outcomes(paired_effects: pd.DataFrame) -> pd.DataFrame:
    """Collapse 160 paired rows to 80 joint two-control outcome rows.

    Deltas are T-control.  Positive TN and non-positive FN are favourable.
    ``dual_harm`` requires harm against *both* controls on *both* axes.
    """

    required = ("triad_id", "control", "delta_TN", "delta_FN")
    _require_columns(paired_effects, required, name="paired_effects")
    if paired_effects.duplicated(["triad_id", "control"]).any():
        raise ValueError("paired_effects contains duplicate triad/control rows")

    records: list[dict[str, object]] = []
    for triad_id, group in paired_effects.groupby("triad_id", sort=True):
        if len(group) != 2 or set(group["control"].astype(str)) != set(CONTROL_ORDER):
            raise ValueError(f"Triad {triad_id} must have exactly one R1 and one R2")
        indexed = group.set_index(group["control"].astype(str))
        record: dict[str, object] = {"triad_id": str(triad_id)}
        excluded = set(required)
        for column in group.columns:
            if column in excluded:
                continue
            values = group[column].drop_duplicates()
            if len(values) == 1:
                record[column] = values.iloc[0]
        for control in CONTROL_ORDER:
            record[f"delta_TN_{control}"] = float(indexed.loc[control, "delta_TN"])
            record[f"delta_FN_{control}"] = float(indexed.loc[control, "delta_FN"])
        record["G_TN"] = min(record["delta_TN_R1"], record["delta_TN_R2"])
        record["G_FN"] = max(record["delta_FN_R1"], record["delta_FN_R2"])
        record["HARM_TN"] = max(record["delta_TN_R1"], record["delta_TN_R2"])
        record["HARM_FN"] = min(record["delta_FN_R1"], record["delta_FN_R2"])
        record["dual_improvement"] = bool(record["G_TN"] > 0 and record["G_FN"] <= 0)
        record["high_value"] = bool(record["G_TN"] >= 300 and record["G_FN"] <= 2)
        record["dual_harm"] = bool(record["HARM_TN"] < 0 and record["HARM_FN"] > 0)
        if record["dual_improvement"]:
            cohort = "DUAL_IMPROVEMENT"
        elif record["high_value"]:
            cohort = "HIGH_VALUE"
        elif record["dual_harm"]:
            cohort = "DUAL_HARM"
        else:
            cohort = "MIXED_OR_REVERSAL"
        record["exclusive_cohort"] = cohort
        records.append(record)
    return pd.DataFrame(records).sort_values("triad_id", ignore_index=True)


def baseline_counts(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Return explicit overlapping label counts plus the exclusive mixed count."""

    _require_columns(
        outcomes,
        ("triad_id", "dual_improvement", "high_value", "dual_harm", "exclusive_cohort"),
        name="outcomes",
    )
    if outcomes["triad_id"].duplicated().any():
        raise ValueError("outcomes must contain one row per triad")
    n = len(outcomes)
    values = (
        ("ALL_TRIADS", n),
        ("DUAL_IMPROVEMENT", int(outcomes["dual_improvement"].astype(bool).sum())),
        ("HIGH_VALUE", int(outcomes["high_value"].astype(bool).sum())),
        ("DUAL_HARM", int(outcomes["dual_harm"].astype(bool).sum())),
        (
            "MIXED_OR_REVERSAL_EXCLUSIVE",
            int(outcomes["exclusive_cohort"].eq("MIXED_OR_REVERSAL").sum()),
        ),
    )
    return pd.DataFrame(
        [
            {"label": label, "count": count, "rate": count / n if n else np.nan}
            for label, count in values
        ]
    )


def assert_frozen_baseline(
    outcomes: pd.DataFrame,
    *,
    expected_triads: int = 80,
    expected_dual_improvement: int = 15,
    expected_high_value: int = 13,
    expected_dual_harm: int = 23,
) -> None:
    """Fail if the canonical 240-run labels drift from the audited baseline."""

    observed = baseline_counts(outcomes).set_index("label")["count"].to_dict()
    expected = {
        "ALL_TRIADS": expected_triads,
        "DUAL_IMPROVEMENT": expected_dual_improvement,
        "HIGH_VALUE": expected_high_value,
        "DUAL_HARM": expected_dual_harm,
    }
    errors = []
    for label, count in expected.items():
        if int(observed[label]) != int(count):
            errors.append(f"{label.lower()}={observed[label]} expected {count}")
    if errors:
        raise ValueError("Frozen baseline mismatch: " + "; ".join(errors))


def paired_sign_flip_test(
    values: Sequence[float],
    *,
    clusters: Sequence[object] | None = None,
    alternative: str = "two-sided",
    max_exact_clusters: int = 20,
    resamples: int = 100_000,
    random_state: int = 0,
) -> dict[str, object]:
    """Test a paired mean delta by flipping signs at the cluster level."""

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be two-sided, greater, or less")
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("values must be a non-empty finite one-dimensional sequence")
    if clusters is None:
        cluster_array = np.arange(len(array), dtype=object)
    else:
        cluster_array = np.asarray(clusters, dtype=object)
        if len(cluster_array) != len(array):
            raise ValueError("clusters and values must have equal length")
    unique_clusters = pd.unique(cluster_array)
    observed = float(array.mean())

    def is_extreme(statistic: float) -> bool:
        if alternative == "greater":
            return statistic >= observed - 1e-15
        if alternative == "less":
            return statistic <= observed + 1e-15
        return abs(statistic) >= abs(observed) - 1e-15

    if len(unique_clusters) <= max_exact_clusters:
        extreme = 0
        total = 0
        for signs in product((-1.0, 1.0), repeat=len(unique_clusters)):
            sign_map = dict(zip(unique_clusters, signs))
            statistic = float(
                np.mean(array * np.asarray([sign_map[value] for value in cluster_array]))
            )
            extreme += int(is_extreme(statistic))
            total += 1
        p_value = extreme / total
        method = "exact_cluster_sign_flip"
    else:
        rng = np.random.default_rng(random_state)
        extreme = 0
        for _ in range(int(resamples)):
            signs = rng.choice((-1.0, 1.0), size=len(unique_clusters))
            sign_map = dict(zip(unique_clusters, signs))
            statistic = float(
                np.mean(array * np.asarray([sign_map[value] for value in cluster_array]))
            )
            extreme += int(is_extreme(statistic))
        p_value = (extreme + 1) / (int(resamples) + 1)
        method = "monte_carlo_cluster_sign_flip"
    return {
        "n": len(array),
        "n_clusters": len(unique_clusters),
        "mean": observed,
        "alternative": alternative,
        "p_value": float(p_value),
        "method": method,
    }


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving NaN positions."""

    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid_positions = np.flatnonzero(np.isfinite(values))
    if len(valid_positions) == 0:
        return adjusted
    valid = values[valid_positions]
    order = np.argsort(valid)
    ranked = valid[order]
    m = len(ranked)
    raw = ranked * m / np.arange(1, m + 1)
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    restored = np.empty(m, dtype=float)
    restored[order] = np.clip(monotone, 0.0, 1.0)
    adjusted[valid_positions] = restored
    return adjusted


def _cluster_bootstrap_statistic(
    frame: pd.DataFrame,
    *,
    cluster_column: str,
    statistic,
    resamples: int,
    random_state: int,
) -> tuple[float, float, int]:
    clusters = pd.unique(frame[cluster_column])
    if len(clusters) == 0:
        return np.nan, np.nan, 0
    grouped = {cluster: frame.loc[frame[cluster_column] == cluster] for cluster in clusters}
    rng = np.random.default_rng(random_state)
    samples: list[float] = []
    attempts = 0
    maximum_attempts = max(int(resamples) * 20, 100)
    while len(samples) < int(resamples) and attempts < maximum_attempts:
        attempts += 1
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        replicate = pd.concat([grouped[value] for value in chosen], ignore_index=True)
        value = float(statistic(replicate))
        if np.isfinite(value):
            samples.append(value)
    if not samples:
        return np.nan, np.nan, 0
    low, high = np.quantile(np.asarray(samples), [0.025, 0.975])
    return float(low), float(high), len(samples)


def _cluster_bootstrap_mean_difference(
    frame: pd.DataFrame,
    *,
    cluster_column: str,
    value_column: str,
    label_column: str,
    positive_label: str,
    negative_label: str,
    resamples: int,
    random_state: int,
) -> tuple[float, float, int]:
    """Vectorized cluster bootstrap for a two-cohort mean difference.

    This is mathematically equivalent to rebuilding each row-level cluster
    sample, but avoids hundreds of thousands of pandas concatenations when the
    same contrast is repeated across a wide feature matrix.
    """

    clusters = pd.unique(frame[cluster_column])
    if len(clusters) == 0:
        return np.nan, np.nan, 0
    aggregates = []
    for cluster in clusters:
        group = frame.loc[frame[cluster_column] == cluster]
        positive = pd.to_numeric(
            group.loc[group[label_column] == positive_label, value_column],
            errors="coerce",
        ).dropna()
        negative = pd.to_numeric(
            group.loc[group[label_column] == negative_label, value_column],
            errors="coerce",
        ).dropna()
        aggregates.append(
            (
                float(positive.sum()),
                int(len(positive)),
                float(negative.sum()),
                int(len(negative)),
            )
        )
    aggregate = np.asarray(aggregates, dtype=float)
    rng = np.random.default_rng(random_state)
    accepted: list[np.ndarray] = []
    accepted_count = 0
    attempts = 0
    target = int(resamples)
    maximum_attempts = max(target * 20, 100)
    while accepted_count < target and attempts < maximum_attempts:
        batch_size = min(max(target - accepted_count, 256), 16_384)
        draws = rng.integers(0, len(clusters), size=(batch_size, len(clusters)))
        totals = aggregate[draws].sum(axis=1)
        valid = (totals[:, 1] > 0) & (totals[:, 3] > 0)
        values = totals[valid, 0] / totals[valid, 1] - totals[valid, 2] / totals[valid, 3]
        if len(values):
            take = min(target - accepted_count, len(values))
            accepted.append(values[:take])
            accepted_count += take
        attempts += batch_size
    if not accepted:
        return np.nan, np.nan, 0
    samples = np.concatenate(accepted)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high), int(len(samples))


def build_paired_inference(
    paired_features: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    bootstrap_cluster: str = "training_seed",
    bootstrap_resamples: int = 10_000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Infer paired T-control feature deltas, with R1 and R2 always separate."""

    _require_columns(
        paired_features,
        ("triad_id", "control", bootstrap_cluster, *feature_columns),
        name="paired_features",
    )
    if paired_features.duplicated(["triad_id", "control"]).any():
        raise ValueError("paired_features contains duplicate triad/control rows")
    records: list[dict[str, object]] = []
    for control in CONTROL_ORDER:
        control_rows = paired_features.loc[paired_features["control"].astype(str) == control]
        for feature_index, feature in enumerate(feature_columns):
            values = pd.to_numeric(control_rows[feature], errors="coerce")
            valid = control_rows.loc[values.notna()].copy()
            valid[feature] = values.loc[values.notna()].astype(float)
            if valid.empty:
                continue
            low, high, completed = _cluster_bootstrap_statistic(
                valid,
                cluster_column=bootstrap_cluster,
                statistic=lambda data, column=feature: data[column].mean(),
                resamples=bootstrap_resamples,
                random_state=random_state + feature_index + (0 if control == "R1" else 10_000),
            )
            sign_flip = paired_sign_flip_test(
                valid[feature].to_numpy(dtype=float),
                clusters=valid[bootstrap_cluster].to_numpy(),
                alternative="two-sided",
                random_state=random_state + feature_index,
            )
            records.append(
                {
                    "control": control,
                    "feature": feature,
                    "n_triads": len(valid),
                    "n_clusters": valid[bootstrap_cluster].nunique(),
                    "mean": float(valid[feature].mean()),
                    "median": float(valid[feature].median()),
                    "bootstrap_cluster": bootstrap_cluster,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "bootstrap_resamples_completed": completed,
                    "sign_flip_method": sign_flip["method"],
                    "p_value": sign_flip["p_value"],
                }
            )
    result = pd.DataFrame(records)
    if result.empty:
        return result
    result["q_value_bh"] = np.nan
    for control, indices in result.groupby("control").groups.items():
        result.loc[indices, "q_value_bh"] = benjamini_hochberg(
            result.loc[indices, "p_value"]
        )
    return result.sort_values(["control", "feature"], ignore_index=True)


def build_descriptive_cohort_contrasts(
    triads: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    positive_cohort: str = "DUAL_IMPROVEMENT",
    negative_cohort: str = "DUAL_HARM",
    bootstrap_resamples: int = 10_000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Describe good-versus-harmful cohorts without calling it prediction."""

    _require_columns(
        triads,
        (
            "triad_id",
            "training_seed",
            "condition_slot",
            "exclusive_cohort",
            *feature_columns,
        ),
        name="triads",
    )
    subset = triads.loc[
        triads["exclusive_cohort"].isin([positive_cohort, negative_cohort])
    ].copy()
    records: list[dict[str, object]] = []
    for index, feature in enumerate(feature_columns):
        subset[feature] = pd.to_numeric(subset[feature], errors="coerce")
        valid = subset.loc[subset[feature].notna()].copy()
        positive = valid.loc[valid["exclusive_cohort"] == positive_cohort, feature]
        negative = valid.loc[valid["exclusive_cohort"] == negative_cohort, feature]
        if positive.empty or negative.empty:
            continue

        seed_low, seed_high, seed_n = _cluster_bootstrap_mean_difference(
            valid,
            cluster_column="training_seed",
            value_column=feature,
            label_column="exclusive_cohort",
            positive_label=positive_cohort,
            negative_label=negative_cohort,
            resamples=bootstrap_resamples,
            random_state=random_state + index,
        )
        condition_low, condition_high, condition_n = _cluster_bootstrap_mean_difference(
            valid,
            cluster_column="condition_slot",
            value_column=feature,
            label_column="exclusive_cohort",
            positive_label=positive_cohort,
            negative_label=negative_cohort,
            resamples=bootstrap_resamples,
            random_state=random_state + 100_000 + index,
        )
        records.append(
            {
                "feature": feature,
                "positive_cohort": positive_cohort,
                "negative_cohort": negative_cohort,
                "positive_n": len(positive),
                "negative_n": len(negative),
                "positive_mean": float(positive.mean()),
                "negative_mean": float(negative.mean()),
                "mean_difference": float(positive.mean() - negative.mean()),
                "seed_bootstrap_ci_low": seed_low,
                "seed_bootstrap_ci_high": seed_high,
                "seed_bootstrap_resamples_completed": seed_n,
                "condition_bootstrap_ci_low": condition_low,
                "condition_bootstrap_ci_high": condition_high,
                "condition_bootstrap_resamples_completed": condition_n,
                "interpretation": "DESCRIPTIVE_NOT_PREDICTIVE",
            }
        )
    return pd.DataFrame(records)


def stratified_permutation_cohort_contrasts(
    triads: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    positive_cohort: str = "DUAL_IMPROVEMENT",
    negative_cohort: str = "DUAL_HARM",
    stratify_by: str = "training_seed",
    resamples: int = 10_000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Permutation-test good-versus-harm contrasts within frozen seed strata."""

    _require_columns(
        triads,
        ("triad_id", "exclusive_cohort", stratify_by, *feature_columns),
        name="triads",
    )
    subset = triads.loc[
        triads["exclusive_cohort"].isin([positive_cohort, negative_cohort])
    ].copy()
    if subset.empty:
        return pd.DataFrame()
    labels = subset["exclusive_cohort"].eq(positive_cohort).to_numpy(dtype=bool)
    if labels.all() or (~labels).all():
        raise ValueError("Permutation contrast requires both cohorts")
    values = subset[list(feature_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)

    def differences(label_matrix: np.ndarray) -> np.ndarray:
        positive_weights = label_matrix.astype(float)
        negative_weights = (~label_matrix).astype(float)
        positive_sum = positive_weights @ filled
        negative_sum = negative_weights @ filled
        positive_count = positive_weights @ valid.astype(float)
        negative_count = negative_weights @ valid.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return positive_sum / positive_count - negative_sum / negative_count

    observed = differences(labels.reshape(1, -1))[0]
    rng = np.random.default_rng(random_state)
    permutations = np.broadcast_to(labels, (int(resamples), len(labels))).copy()
    strata = subset[stratify_by].astype(str).to_numpy()
    for stratum in pd.unique(strata):
        positions = np.flatnonzero(strata == stratum)
        base = labels[positions]
        random_keys = rng.random((int(resamples), len(positions)))
        orders = np.argsort(random_keys, axis=1)
        permutations[:, positions] = base[orders]
    permuted = differences(permutations)
    records: list[dict[str, object]] = []
    for index, feature in enumerate(feature_columns):
        null = permuted[:, index]
        finite = np.isfinite(null)
        if not np.isfinite(observed[index]) or not finite.any():
            p_value = np.nan
            completed = int(finite.sum())
        else:
            extreme = np.abs(null[finite]) >= abs(observed[index]) - 1e-15
            p_value = float((int(extreme.sum()) + 1) / (int(finite.sum()) + 1))
            completed = int(finite.sum())
        feature_valid = valid[:, index]
        records.append(
            {
                "feature": str(feature),
                "positive_cohort": positive_cohort,
                "negative_cohort": negative_cohort,
                "positive_n": int((feature_valid & labels).sum()),
                "negative_n": int((feature_valid & ~labels).sum()),
                "positive_minus_negative": float(observed[index]),
                "stratification": stratify_by,
                "strata_count": int(pd.Series(strata).nunique()),
                "permutation_resamples_completed": completed,
                "p_value": p_value,
            }
        )
    result = pd.DataFrame(records)
    result["q_value_bh"] = benjamini_hochberg(result["p_value"].to_numpy())
    return result.sort_values("feature", ignore_index=True)


def _base_feature_name(column: str) -> str:
    value = str(column)
    for prefix in ("R1__", "R2__"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    if value.startswith("delta__"):
        value = value[len("delta__") :]
    return value


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def eligible_feature_columns(
    columns: Iterable[str],
    feature_time_registry: pd.DataFrame,
    *,
    cutoff: int,
) -> list[str]:
    """Apply the registry as an explicit allowlist and time gate."""

    _require_columns(
        feature_time_registry,
        ("feature", "available_epoch", "allowed_as_predictor"),
        name="feature_time_registry",
    )
    allowed: set[str] = set()
    for row in feature_time_registry.itertuples(index=False):
        epoch = pd.to_numeric(pd.Series([row.available_epoch]), errors="coerce").iloc[0]
        if _as_bool(row.allowed_as_predictor) and pd.notna(epoch) and float(epoch) <= cutoff:
            allowed.add(str(row.feature))
    return [str(column) for column in columns if _base_feature_name(str(column)) in allowed]


def one_sided_binomial_lower_bound(
    successes: int,
    total: int,
    *,
    alpha: float = 0.05,
) -> float:
    """Exact one-sided Clopper-Pearson lower confidence bound."""

    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Require 0 <= successes <= total")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, total - successes + 1))


def minimum_confirmations(
    *,
    target_rate: float = 0.8,
    alpha: float = 0.05,
    max_failures: int = 2,
    maximum_total: int = 100_000,
) -> pd.DataFrame:
    """Minimum confirmation counts whose one-sided lower bound exceeds target."""

    if not 0 < target_rate < 1:
        raise ValueError("target_rate must lie in (0, 1)")
    records = []
    for failures in range(max_failures + 1):
        found = False
        for total in range(failures + 1, maximum_total + 1):
            successes = total - failures
            lower = one_sided_binomial_lower_bound(successes, total, alpha=alpha)
            if lower > target_rate:
                records.append(
                    {
                        "failures": failures,
                        "successes": successes,
                        "total": total,
                        "lower_bound": lower,
                    }
                )
                found = True
                break
        if not found:
            raise ValueError(
                f"No confirmation count found for {failures} failures up to {maximum_total}"
            )
    return pd.DataFrame(records)


def evaluate_candidate_predictions(
    predictions: pd.DataFrame,
    *,
    target_column: str = "y_true",
    harmful_column: str = "dual_harm",
    probability_column: str = "probability",
    selected_column: str = "selected",
    target_rate: float = 0.8,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Summarize candidate screening; AUC is never a success probability."""

    _require_columns(
        predictions,
        (target_column, harmful_column, probability_column, selected_column),
        name="predictions",
    )
    y = predictions[target_column].astype(bool).to_numpy(dtype=bool)
    harm = predictions[harmful_column].astype(bool).to_numpy(dtype=bool)
    probability = pd.to_numeric(predictions[probability_column], errors="raise").to_numpy()
    selected = predictions[selected_column].astype(bool).to_numpy(dtype=bool)
    selected_n = int(selected.sum())
    successes = int(np.logical_and(selected, y).sum())
    harmful = int(np.logical_and(selected, harm).sum())
    false_positives = int(np.logical_and(selected, ~y).sum())
    false_negatives = int(np.logical_and(~selected, y).sum())
    true_negatives = int(np.logical_and(~selected, ~y).sum())
    precision = successes / selected_n if selected_n else np.nan
    harmful_rate = harmful / selected_n if selected_n else np.nan
    success_recall = successes / int(y.sum()) if y.any() else np.nan
    negative_n = int((~y).sum())
    specificity = true_negatives / negative_n if negative_n else np.nan
    balanced_accuracy = (
        (success_recall + specificity) / 2
        if np.isfinite(success_recall) and np.isfinite(specificity)
        else np.nan
    )
    f1 = (
        2 * precision * success_recall / (precision + success_recall)
        if np.isfinite(precision)
        and np.isfinite(success_recall)
        and precision + success_recall > 0
        else np.nan
    )
    lower = one_sided_binomial_lower_bound(successes, selected_n, alpha=alpha)
    roc_auc = float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else np.nan
    average_precision = (
        float(average_precision_score(y, probability)) if y.any() else np.nan
    )
    return {
        "n": len(predictions),
        "positive_n": int(y.sum()),
        "baseline_positive_rate": float(y.mean()) if len(y) else np.nan,
        "selected_n": selected_n,
        "selected_successes": successes,
        "selected_harmful": harmful,
        "true_positives": successes,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "precision": float(precision),
        "coverage": selected_n / len(predictions) if len(predictions) else np.nan,
        "harmful_rate": float(harmful_rate),
        "success_recall": float(success_recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "f1": float(f1),
        "precision_lower_bound_one_sided": lower,
        "target_rate": target_rate,
        "confirmed_above_target": bool(selected_n > 0 and lower > target_rate),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "auc_is_not_success_probability": True,
    }


def validate_frozen_split(frame: pd.DataFrame) -> dict[str, int]:
    """Enforce 75 discovery triads and five Phase-C A02 falsification seeds."""

    _require_columns(
        frame,
        (
            "triad_id",
            "phase",
            "condition_slot",
            "discovery_or_confirmation",
            "training_seed",
        ),
        name="candidate frame",
    )
    if frame["triad_id"].duplicated().any():
        raise ValueError("candidate frame must contain one row per triad")
    discovery = frame.loc[frame["discovery_or_confirmation"].eq("discovery")]
    external = frame.loc[frame["discovery_or_confirmation"].eq("confirmation")]
    if len(frame) != 80 or len(discovery) != 75 or len(external) != 5:
        raise ValueError("Frozen split requires 80=75 discovery+5 confirmation triads")
    if not discovery["phase"].isin(["A", "B"]).all():
        raise ValueError("Discovery rows must be Phase A/B only")
    seed_sizes = discovery.groupby("training_seed").size()
    if len(seed_sizes) != 3 or set(seed_sizes.astype(int)) != {25}:
        raise ValueError("Discovery requires three training seeds with 25 conditions each")
    if not external["phase"].eq("C").all() or not external["condition_slot"].eq("A02").all():
        raise ValueError("External falsification must contain only five Phase-C A02 triads")
    if external["training_seed"].nunique() != 5:
        raise ValueError("Phase C requires five distinct confirmation seeds")
    if set(discovery["training_seed"]) & set(external["training_seed"]):
        raise ValueError("Discovery and external training seeds must be disjoint")
    return {
        "all_triads": len(frame),
        "discovery_triads": len(discovery),
        "discovery_seeds": discovery["training_seed"].nunique(),
        "conditions_per_discovery_seed": int(seed_sizes.iloc[0]),
        "external_triads": len(external),
        "external_seeds": external["training_seed"].nunique(),
    }


@dataclass(frozen=True)
class CandidateValidationResult:
    """Cross-fitted predictions and honest screening summaries."""

    feature_columns: tuple[str, ...]
    predictions: pd.DataFrame
    summaries: pd.DataFrame
    fold_details: pd.DataFrame


def _new_pipeline(feature_count: int, max_features: int, random_state: int) -> Pipeline:
    if feature_count <= 0:
        raise ValueError("At least one eligible feature is required")
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=min(max_features, feature_count))),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _fit_pipeline(
    train: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    target_column: str,
    max_features: int,
    random_state: int,
) -> Pipeline:
    target = train[target_column].astype(int)
    if target.nunique() != 2:
        raise ValueError("Every model-training fold must contain both outcome classes")
    pipeline = _new_pipeline(len(feature_columns), max_features, random_state)
    # A perfectly separating synthetic/real feature has zero within-class
    # variance, for which sklearn's ANOVA score is correctly infinite.  The
    # resulting RuntimeWarning is numerical noise rather than a failed fit.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"sklearn\.feature_selection\._univariate_selection",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Features .* are constant\.",
            category=UserWarning,
            module=r"sklearn\.feature_selection\._univariate_selection",
        )
        pipeline.fit(train[list(feature_columns)], target)
    return pipeline


def _selected_feature_names(model: Pipeline, feature_columns: Sequence[str]) -> list[str]:
    support = model.named_steps["select"].get_support()
    return [feature for feature, keep in zip(feature_columns, support) if keep]


def _inner_oof_probabilities(
    train: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    target_column: str,
    group_column: str,
    max_features: int,
    random_state: int,
) -> np.ndarray:
    groups = pd.unique(train[group_column])
    if len(groups) < 2:
        raise ValueError("Fold-local threshold selection needs at least two inner groups")
    probabilities = pd.Series(np.nan, index=train.index, dtype=float)
    for fold_index, group in enumerate(groups):
        inner_test = train[group_column].eq(group)
        inner_train = train.loc[~inner_test]
        model = _fit_pipeline(
            inner_train,
            feature_columns,
            target_column=target_column,
            max_features=max_features,
            random_state=random_state + fold_index,
        )
        probabilities.loc[inner_test] = model.predict_proba(
            train.loc[inner_test, list(feature_columns)]
        )[:, 1]
    if probabilities.isna().any():
        raise ValueError("Inner cross-fitting failed to score every training triad")
    return probabilities.to_numpy(dtype=float)


def _select_screening_threshold(
    probabilities: np.ndarray,
    target: np.ndarray,
    harmful: np.ndarray,
    *,
    target_precision: float,
    max_harmful_rate: float,
) -> tuple[float, dict[str, float]]:
    candidates = np.unique(probabilities)[::-1]
    feasible: list[tuple[int, float, float, float]] = []
    for threshold in candidates:
        selected = probabilities >= threshold
        count = int(selected.sum())
        precision = float(target[selected].mean())
        harmful_rate = float(harmful[selected].mean())
        if precision >= target_precision and harmful_rate <= max_harmful_rate:
            feasible.append((count, precision, -harmful_rate, float(threshold)))
    if not feasible:
        return float("inf"), {"precision": np.nan, "coverage": 0.0, "harmful_rate": np.nan}
    count, precision, negative_harm, threshold = max(feasible)
    return threshold, {
        "precision": precision,
        "coverage": count / len(probabilities),
        "harmful_rate": -negative_harm,
    }


def _prediction_records(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    threshold: float,
    scheme: str,
    outer_group: str,
    target_column: str,
    harmful_column: str,
) -> list[dict[str, object]]:
    records = []
    for position, (_, row) in enumerate(test.iterrows()):
        records.append(
            {
                "triad_id": str(row["triad_id"]),
                "phase": str(row["phase"]),
                "condition_slot": str(row["condition_slot"]),
                "training_seed": row["training_seed"],
                "selection_digest": str(row["selection_digest"]),
                "validation_scheme": scheme,
                "outer_group": outer_group,
                "y_true": bool(row[target_column]),
                "dual_harm": bool(row[harmful_column]),
                "probability": float(probabilities[position]),
                "threshold": float(threshold),
                "selected": bool(probabilities[position] >= threshold),
            }
        )
    return records


def _fit_score_outer_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    target_column: str,
    harmful_column: str,
    inner_group_column: str,
    max_features: int,
    target_precision: float,
    max_harmful_rate: float,
    random_state: int,
) -> tuple[np.ndarray, float, dict[str, object]]:
    inner_probability = _inner_oof_probabilities(
        train,
        feature_columns,
        target_column=target_column,
        group_column=inner_group_column,
        max_features=max_features,
        random_state=random_state,
    )
    threshold, threshold_metrics = _select_screening_threshold(
        inner_probability,
        train[target_column].astype(bool).to_numpy(),
        train[harmful_column].astype(bool).to_numpy(),
        target_precision=target_precision,
        max_harmful_rate=max_harmful_rate,
    )
    model = _fit_pipeline(
        train,
        feature_columns,
        target_column=target_column,
        max_features=max_features,
        random_state=random_state,
    )
    probability = model.predict_proba(test[list(feature_columns)])[:, 1]
    details: dict[str, object] = {
        "train_n": len(train),
        "test_n": len(test),
        "threshold": threshold,
        "threshold_training_precision": threshold_metrics["precision"],
        "threshold_training_coverage": threshold_metrics["coverage"],
        "threshold_training_harmful_rate": threshold_metrics["harmful_rate"],
        "selected_feature_count": len(_selected_feature_names(model, feature_columns)),
        "selected_features": ";".join(_selected_feature_names(model, feature_columns)),
    }
    return probability, threshold, details


def _grouped_predictions(
    discovery: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    outer_group_column: str,
    scheme: str,
    target_column: str,
    harmful_column: str,
    max_features: int,
    target_precision: float,
    max_harmful_rate: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for fold_index, outer_group in enumerate(pd.unique(discovery[outer_group_column])):
        test_mask = discovery[outer_group_column].eq(outer_group)
        train = discovery.loc[~test_mask]
        test = discovery.loc[test_mask]
        probability, threshold, detail = _fit_score_outer_fold(
            train,
            test,
            feature_columns,
            target_column=target_column,
            harmful_column=harmful_column,
            inner_group_column="training_seed",
            max_features=max_features,
            target_precision=target_precision,
            max_harmful_rate=max_harmful_rate,
            random_state=random_state + fold_index * 101,
        )
        records.extend(
            _prediction_records(
                test,
                probability,
                threshold=threshold,
                scheme=scheme,
                outer_group=str(outer_group),
                target_column=target_column,
                harmful_column=harmful_column,
            )
        )
        details.append(
            {
                "validation_scheme": scheme,
                "outer_group": str(outer_group),
                **detail,
            }
        )
    return pd.DataFrame(records), pd.DataFrame(details)


def _double_exclusion_predictions(
    discovery: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    target_column: str,
    harmful_column: str,
    max_features: int,
    target_precision: float,
    max_harmful_rate: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scheme = "DISCOVERY_DOUBLE_EXCLUSION_SEED_DIGEST"
    records: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for fold_index, (_, test_row) in enumerate(discovery.iterrows()):
        test = test_row.to_frame().T
        train = discovery.loc[
            discovery["training_seed"].ne(test_row["training_seed"])
            & discovery["selection_digest"].ne(test_row["selection_digest"])
        ]
        probability, threshold, detail = _fit_score_outer_fold(
            train,
            test,
            feature_columns,
            target_column=target_column,
            harmful_column=harmful_column,
            inner_group_column="training_seed",
            max_features=max_features,
            target_precision=target_precision,
            max_harmful_rate=max_harmful_rate,
            random_state=random_state + fold_index * 103,
        )
        outer_group = f"{test_row['training_seed']}|{test_row['selection_digest']}"
        records.extend(
            _prediction_records(
                test,
                probability,
                threshold=threshold,
                scheme=scheme,
                outer_group=outer_group,
                target_column=target_column,
                harmful_column=harmful_column,
            )
        )
        details.append(
            {"validation_scheme": scheme, "outer_group": outer_group, **detail}
        )
    return pd.DataFrame(records), pd.DataFrame(details)


def run_candidate_validation(
    frame: pd.DataFrame,
    feature_time_registry: pd.DataFrame,
    *,
    cutoff: int,
    target_column: str = "dual_improvement",
    harmful_column: str = "dual_harm",
    max_features: int = 16,
    target_precision: float = 0.8,
    max_harmful_rate: float = 0.2,
    include_digest_diagnostics: bool = True,
    random_state: int = 0,
) -> CandidateValidationResult:
    """Run nested, group-safe screening and external Phase-C falsification.

    Main discovery validation is leave-one-training-seed-out over the 75 Phase
    A/B triads.  Phase C is scored once by a model and threshold learned solely
    from discovery.  Digest diagnostics are sensitivity analyses, never extra
    independent evidence.
    """

    validate_frozen_split(frame)
    _require_columns(
        frame,
        ("selection_digest", target_column, harmful_column),
        name="candidate frame",
    )
    feature_columns = eligible_feature_columns(frame.columns, feature_time_registry, cutoff=cutoff)
    if not feature_columns:
        raise ValueError(f"No registered predictor is available by epoch {cutoff}")
    discovery = frame.loc[frame["discovery_or_confirmation"].eq("discovery")].copy()
    external = frame.loc[frame["discovery_or_confirmation"].eq("confirmation")].copy()

    prediction_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []
    loso, loso_details = _grouped_predictions(
        discovery,
        feature_columns,
        outer_group_column="training_seed",
        scheme="DISCOVERY_LOSO_SEED",
        target_column=target_column,
        harmful_column=harmful_column,
        max_features=max_features,
        target_precision=target_precision,
        max_harmful_rate=max_harmful_rate,
        random_state=random_state,
    )
    prediction_frames.append(loso)
    detail_frames.append(loso_details)

    if include_digest_diagnostics:
        digest, digest_details = _grouped_predictions(
            discovery,
            feature_columns,
            outer_group_column="selection_digest",
            scheme="DISCOVERY_LEAVE_SELECTION_DIGEST_OUT",
            target_column=target_column,
            harmful_column=harmful_column,
            max_features=max_features,
            target_precision=target_precision,
            max_harmful_rate=max_harmful_rate,
            random_state=random_state + 1_000_000,
        )
        double, double_details = _double_exclusion_predictions(
            discovery,
            feature_columns,
            target_column=target_column,
            harmful_column=harmful_column,
            max_features=max_features,
            target_precision=target_precision,
            max_harmful_rate=max_harmful_rate,
            random_state=random_state + 2_000_000,
        )
        prediction_frames.extend([digest, double])
        detail_frames.extend([digest_details, double_details])

    external_threshold, threshold_metrics = _select_screening_threshold(
        loso["probability"].to_numpy(dtype=float),
        loso["y_true"].to_numpy(dtype=bool),
        loso["dual_harm"].to_numpy(dtype=bool),
        target_precision=target_precision,
        max_harmful_rate=max_harmful_rate,
    )
    final_model = _fit_pipeline(
        discovery,
        feature_columns,
        target_column=target_column,
        max_features=max_features,
        random_state=random_state + 3_000_000,
    )
    external_probability = final_model.predict_proba(external[feature_columns])[:, 1]
    external_scheme = "PHASE_C_EXTERNAL_FALSIFICATION"
    prediction_frames.append(
        pd.DataFrame(
            _prediction_records(
                external,
                external_probability,
                threshold=external_threshold,
                scheme=external_scheme,
                outer_group="PHASE_C_A02_FIVE_UNSEEN_SEEDS",
                target_column=target_column,
                harmful_column=harmful_column,
            )
        )
    )
    detail_frames.append(
        pd.DataFrame(
            [
                {
                    "validation_scheme": external_scheme,
                    "outer_group": "PHASE_C_A02_FIVE_UNSEEN_SEEDS",
                    "train_n": len(discovery),
                    "test_n": len(external),
                    "threshold": external_threshold,
                    "threshold_training_precision": threshold_metrics["precision"],
                    "threshold_training_coverage": threshold_metrics["coverage"],
                    "threshold_training_harmful_rate": threshold_metrics["harmful_rate"],
                    "selected_feature_count": len(
                        _selected_feature_names(final_model, feature_columns)
                    ),
                    "selected_features": ";".join(
                        _selected_feature_names(final_model, feature_columns)
                    ),
                }
            ]
        )
    )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    summaries = []
    for scheme, group in predictions.groupby("validation_scheme", sort=False):
        summary = {"validation_scheme": scheme}
        summary.update(
            evaluate_candidate_predictions(
                group,
                target_rate=target_precision,
            )
        )
        summary["interpretation"] = (
            "EXTERNAL_FALSIFICATION_ONLY"
            if scheme == external_scheme
            else "DISCOVERY_CROSS_VALIDATION_DIAGNOSTIC"
        )
        summaries.append(summary)
    return CandidateValidationResult(
        feature_columns=tuple(feature_columns),
        predictions=predictions.sort_values(
            ["validation_scheme", "triad_id"], ignore_index=True
        ),
        summaries=pd.DataFrame(summaries),
        fold_details=pd.concat(detail_frames, ignore_index=True),
    )
