"""Descriptive confound sensitivity for the frozen 80 GapValue triads.

The unit of analysis is always a complete T/R1/R2 triad.  Machine, input
snapshot, and resume strata are observational execution attributes.  They are
reported as sensitivity slices and never used as a causal adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .grouped_statistics import benjamini_hochberg


class ConfoundSensitivityError(RuntimeError):
    """Raised when one of the four frozen inputs is incomplete or inconsistent."""


@dataclass(frozen=True)
class ConfoundSensitivityResult:
    triads: pd.DataFrame
    strata_summary: pd.DataFrame
    binary_contrasts: pd.DataFrame
    within_condition_seed: pd.DataFrame
    within_pair_summary: pd.DataFrame
    summary: dict[str, Any]


_ARMS = ("T", "R1", "R2")
_CONTROLS = ("R1", "R2")
_EXCLUSIVE_LEVELS = (
    "DUAL_IMPROVEMENT",
    "HIGH_VALUE",
    "DUAL_HARM",
    "MIXED_OR_REVERSAL",
)
_OUTCOME_COLUMNS = {
    "triad_id",
    "phase",
    "condition_id",
    "method",
    "budget",
    "training_seed",
    "discovery_or_confirmation",
    "delta_TN_R1",
    "delta_FN_R1",
    "delta_TN_R2",
    "delta_FN_R2",
    "G_TN",
    "G_FN",
    "HARM_TN",
    "HARM_FN",
    "dual_improvement",
    "high_value",
    "dual_harm",
    "exclusive_cohort",
}
_RESOURCE_COLUMNS = {
    "triad_id",
    "all_arms_same_machine",
    "all_arms_same_snapshot",
    "resumed_arm_count",
}
_RAW_COLUMNS = {
    "triad_id",
    "control_arm",
    "score_type",
    "safe_frontier_dominant",
}
_CANONICAL_COLUMNS = {
    "run_slot",
    "triad_id",
    "arm",
    "phase",
    "condition_id",
    "method",
    "budget",
    "training_seed",
    "discovery_or_confirmation",
    "machine_id",
    "input_snapshot_id",
    "resume_count",
    "TN_at_FN95",
    "FN_at_TN68253",
}


def _require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ConfoundSensitivityError(f"{name} missing columns: {missing}")


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def _bool_series(values: pd.Series) -> pd.Series:
    return values.map(_as_bool).astype(bool)


def _assert_close(actual: float, expected: float, message: str) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-9):
        raise ConfoundSensitivityError(
            f"{message}: observed={actual!r}, expected={expected!r}"
        )


def _canonical_pair_table(canonical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(canonical, _CANONICAL_COLUMNS, "canonical metrics")
    if canonical["run_slot"].astype(str).duplicated().any():
        raise ConfoundSensitivityError("canonical metrics contains duplicate run_slot")

    triad_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for triad_id, group in canonical.groupby("triad_id", sort=True):
        if len(group) != 3 or set(group["arm"].astype(str)) != set(_ARMS):
            raise ConfoundSensitivityError(
                f"{triad_id} must have exactly one canonical T, R1, and R2"
            )
        if group["arm"].astype(str).duplicated().any():
            raise ConfoundSensitivityError(f"{triad_id} has duplicate canonical arms")
        indexed = group.set_index(group["arm"].astype(str))
        treatment = indexed.loc["T"]
        common_columns = (
            "phase",
            "condition_id",
            "method",
            "budget",
            "training_seed",
            "discovery_or_confirmation",
        )
        for column in common_columns:
            if group[column].astype(str).nunique(dropna=False) != 1:
                raise ConfoundSensitivityError(
                    f"{triad_id} canonical arms disagree on {column}"
                )
        machines = group["machine_id"].astype(str)
        snapshots = group["input_snapshot_id"].astype(str)
        resumes = pd.to_numeric(group["resume_count"], errors="raise").astype(int)
        triad_rows.append(
            {
                "triad_id": str(triad_id),
                "canonical_all_arms_same_machine": bool(machines.nunique() == 1),
                "canonical_all_arms_same_snapshot": bool(snapshots.nunique() == 1),
                "canonical_resumed_arm_count": int((resumes > 0).sum()),
                "canonical_resume_event_count": int(resumes.sum()),
            }
        )
        for control in _CONTROLS:
            reference = indexed.loc[control]
            delta_tn = float(treatment["TN_at_FN95"] - reference["TN_at_FN95"])
            delta_fn = float(
                treatment["FN_at_TN68253"] - reference["FN_at_TN68253"]
            )
            pair_rows.append(
                {
                    "triad_id": str(triad_id),
                    "phase": str(treatment["phase"]),
                    "condition_id": str(treatment["condition_id"]),
                    "method": str(treatment["method"]),
                    "budget": int(treatment["budget"]),
                    "training_seed": int(treatment["training_seed"]),
                    "discovery_or_confirmation": str(
                        treatment["discovery_or_confirmation"]
                    ),
                    "control_arm": control,
                    "treatment_run_slot": str(treatment["run_slot"]),
                    "control_run_slot": str(reference["run_slot"]),
                    "treatment_machine_id": str(treatment["machine_id"]),
                    "control_machine_id": str(reference["machine_id"]),
                    "same_machine": bool(
                        str(treatment["machine_id"]) == str(reference["machine_id"])
                    ),
                    "treatment_input_snapshot_id": str(
                        treatment["input_snapshot_id"]
                    ),
                    "control_input_snapshot_id": str(reference["input_snapshot_id"]),
                    "same_input_snapshot": bool(
                        str(treatment["input_snapshot_id"])
                        == str(reference["input_snapshot_id"])
                    ),
                    "treatment_resume_count": int(treatment["resume_count"]),
                    "control_resume_count": int(reference["resume_count"]),
                    "any_pair_resume": bool(
                        int(treatment["resume_count"]) > 0
                        or int(reference["resume_count"]) > 0
                    ),
                    "delta_TN": delta_tn,
                    "delta_FN": delta_fn,
                }
            )
    return (
        pd.DataFrame(triad_rows).sort_values("triad_id", ignore_index=True),
        pd.DataFrame(pair_rows).sort_values(
            ["triad_id", "control_arm"], ignore_index=True
        ),
    )


def _validate_and_join_inputs(
    outcomes: pd.DataFrame,
    resources: pd.DataFrame,
    raw_frontier: pd.DataFrame,
    canonical: pd.DataFrame,
    *,
    expected_triads: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(outcomes, _OUTCOME_COLUMNS, "triad outcomes")
    _require_columns(resources, _RESOURCE_COLUMNS, "resource triads")
    _require_columns(raw_frontier, _RAW_COLUMNS, "raw frontier")
    if outcomes["triad_id"].astype(str).duplicated().any():
        raise ConfoundSensitivityError("triad outcomes contains duplicate triad_id")
    if resources["triad_id"].astype(str).duplicated().any():
        raise ConfoundSensitivityError("resource triads contains duplicate triad_id")
    if len(outcomes) != expected_triads:
        raise ConfoundSensitivityError(
            f"triad outcomes has {len(outcomes)} rows; expected {expected_triads}"
        )
    if set(outcomes["triad_id"].astype(str)) != set(resources["triad_id"].astype(str)):
        raise ConfoundSensitivityError("resource triad IDs do not match outcomes")

    canonical_triads, pairs = _canonical_pair_table(canonical)
    if len(canonical_triads) != expected_triads:
        raise ConfoundSensitivityError(
            f"canonical metrics has {len(canonical_triads)} triads; expected {expected_triads}"
        )
    if set(outcomes["triad_id"].astype(str)) != set(canonical_triads["triad_id"]):
        raise ConfoundSensitivityError("canonical triad IDs do not match outcomes")

    resource = resources.copy()
    resource["triad_id"] = resource["triad_id"].astype(str)
    merged_resource = resource.merge(
        canonical_triads, on="triad_id", how="inner", validate="one_to_one"
    )
    for row in merged_resource.itertuples(index=False):
        if _as_bool(row.all_arms_same_machine) != bool(
            row.canonical_all_arms_same_machine
        ):
            raise ConfoundSensitivityError(f"{row.triad_id} same-machine drift")
        if _as_bool(row.all_arms_same_snapshot) != bool(
            row.canonical_all_arms_same_snapshot
        ):
            raise ConfoundSensitivityError(f"{row.triad_id} snapshot drift")
        if int(row.resumed_arm_count) != int(row.canonical_resumed_arm_count):
            raise ConfoundSensitivityError(f"{row.triad_id} resume-count drift")

    raw = raw_frontier.loc[
        raw_frontier["score_type"].astype(str).str.lower().eq("raw")
    ].copy()
    raw["triad_id"] = raw["triad_id"].astype(str)
    raw["control_arm"] = raw["control_arm"].astype(str)
    if raw.duplicated(["triad_id", "control_arm"]).any():
        raise ConfoundSensitivityError("raw frontier contains duplicate triad/control rows")
    counts = raw.groupby("triad_id")["control_arm"].agg(list)
    invalid = [
        triad_id
        for triad_id, controls in counts.items()
        if len(controls) != 2 or set(controls) != set(_CONTROLS)
    ]
    if invalid or len(counts) != expected_triads:
        raise ConfoundSensitivityError(
            "raw frontier must have exactly one raw R1 and R2 row per triad"
        )
    raw["safe_frontier_dominant"] = _bool_series(raw["safe_frontier_dominant"])
    raw_wide = raw.pivot(
        index="triad_id", columns="control_arm", values="safe_frontier_dominant"
    )
    raw_dual = (
        raw_wide["R1"].astype(bool) & raw_wide["R2"].astype(bool)
    ).rename("raw_dual_safe")

    joined = outcomes.copy()
    joined["triad_id"] = joined["triad_id"].astype(str)
    joined = joined.merge(
        merged_resource[
            [
                "triad_id",
                "all_arms_same_machine",
                "all_arms_same_snapshot",
                "resumed_arm_count",
                "canonical_resume_event_count",
            ]
        ],
        on="triad_id",
        how="inner",
        validate="one_to_one",
    ).merge(raw_dual, on="triad_id", how="inner", validate="one_to_one")
    joined["all_arms_same_machine"] = _bool_series(joined["all_arms_same_machine"])
    joined["all_arms_same_snapshot"] = _bool_series(joined["all_arms_same_snapshot"])
    joined["raw_dual_safe"] = _bool_series(joined["raw_dual_safe"])
    for column in ("dual_improvement", "high_value", "dual_harm"):
        joined[column] = _bool_series(joined[column])

    expected_exclusive: list[str] = []
    for row in joined.itertuples(index=False):
        _assert_close(min(row.delta_TN_R1, row.delta_TN_R2), row.G_TN, f"{row.triad_id} G_TN drift")
        _assert_close(max(row.delta_FN_R1, row.delta_FN_R2), row.G_FN, f"{row.triad_id} G_FN drift")
        _assert_close(max(row.delta_TN_R1, row.delta_TN_R2), row.HARM_TN, f"{row.triad_id} HARM_TN drift")
        _assert_close(min(row.delta_FN_R1, row.delta_FN_R2), row.HARM_FN, f"{row.triad_id} HARM_FN drift")
        derived_dual = bool(row.G_TN > 0 and row.G_FN <= 0)
        derived_high = bool(row.G_TN >= 300 and row.G_FN <= 2)
        derived_harm = bool(row.HARM_TN < 0 and row.HARM_FN > 0)
        if derived_dual != bool(row.dual_improvement):
            raise ConfoundSensitivityError(f"{row.triad_id} dual-improvement drift")
        if derived_high != bool(row.high_value):
            raise ConfoundSensitivityError(f"{row.triad_id} high-value drift")
        if derived_harm != bool(row.dual_harm):
            raise ConfoundSensitivityError(f"{row.triad_id} dual-harm drift")
        expected_exclusive.append(
            "DUAL_IMPROVEMENT"
            if derived_dual
            else "HIGH_VALUE"
            if derived_high
            else "DUAL_HARM"
            if derived_harm
            else "MIXED_OR_REVERSAL"
        )
    if expected_exclusive != joined["exclusive_cohort"].astype(str).tolist():
        raise ConfoundSensitivityError("exclusive cohort drift")

    pair_lookup = pairs.set_index(["triad_id", "control_arm"])
    outcome_lookup = joined.set_index("triad_id")
    for (triad_id, control), pair in pair_lookup.iterrows():
        expected = outcome_lookup.loc[triad_id]
        _assert_close(
            pair["delta_TN"], expected[f"delta_TN_{control}"], f"{triad_id}/{control} delta TN drift"
        )
        _assert_close(
            pair["delta_FN"], expected[f"delta_FN_{control}"], f"{triad_id}/{control} delta FN drift"
        )
    raw_pair = raw[["triad_id", "control_arm", "safe_frontier_dominant"]].rename(
        columns={"safe_frontier_dominant": "raw_safe_frontier_dominant"}
    )
    pairs = pairs.merge(
        raw_pair,
        on=["triad_id", "control_arm"],
        how="inner",
        validate="one_to_one",
    )
    pairs["paired_improvement"] = (pairs["delta_TN"] > 0) & (
        pairs["delta_FN"] <= 0
    )
    pairs["paired_high_value"] = (pairs["delta_TN"] >= 300) & (
        pairs["delta_FN"] <= 2
    )
    pairs["paired_harm"] = (pairs["delta_TN"] < 0) & (pairs["delta_FN"] > 0)

    joined["machine_stratum"] = np.where(
        joined["all_arms_same_machine"], "SAME_MACHINE", "CROSS_MACHINE"
    )
    joined["snapshot_stratum"] = np.where(
        joined["all_arms_same_snapshot"], "SAME_SNAPSHOT", "CROSS_SNAPSHOT"
    )
    joined["resume_stratum"] = np.where(
        pd.to_numeric(joined["resumed_arm_count"]) == 0,
        "NO_RESUME",
        "ANY_ARM_RESUME",
    )
    joined["budget_stratum"] = "B" + joined["budget"].astype(int).astype(str)
    joined["discovery_confirmation_stratum"] = (
        joined["discovery_or_confirmation"].astype(str).str.upper()
    )
    return joined.sort_values("triad_id", ignore_index=True), pairs


_STRATA: dict[str, tuple[str, tuple[str, ...]]] = {
    "machine": ("machine_stratum", ("SAME_MACHINE", "CROSS_MACHINE")),
    "snapshot": ("snapshot_stratum", ("SAME_SNAPSHOT", "CROSS_SNAPSHOT")),
    "resume": ("resume_stratum", ("NO_RESUME", "ANY_ARM_RESUME")),
    "budget": ("budget_stratum", ("B600", "B3000", "B6000")),
    "discovery_confirmation": (
        "discovery_confirmation_stratum",
        ("DISCOVERY", "CONFIRMATION"),
    ),
}


def _strata_summary(triads: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family, (column, levels) in _STRATA.items():
        for level in levels:
            group = triads.loc[triads[column].astype(str).eq(level)]
            if group.empty:
                continue
            row: dict[str, object] = {
                "stratum_family": family,
                "stratum_level": level,
                "n_triads": len(group),
                "n_training_seeds": group["training_seed"].nunique(),
                "n_conditions": group["condition_id"].nunique(),
                "G_TN_mean": float(group["G_TN"].mean()),
                "G_TN_median": float(group["G_TN"].median()),
                "G_FN_mean": float(group["G_FN"].mean()),
                "G_FN_median": float(group["G_FN"].median()),
                "dual_improvement_count": int(group["dual_improvement"].sum()),
                "dual_improvement_rate": float(group["dual_improvement"].mean()),
                "high_value_count": int(group["high_value"].sum()),
                "high_value_rate": float(group["high_value"].mean()),
                "dual_harm_count": int(group["dual_harm"].sum()),
                "dual_harm_rate": float(group["dual_harm"].mean()),
                "raw_dual_safe_count": int(group["raw_dual_safe"].sum()),
                "raw_dual_safe_rate": float(group["raw_dual_safe"].mean()),
            }
            for cohort in _EXCLUSIVE_LEVELS:
                key = cohort.lower()
                mask = group["exclusive_cohort"].astype(str).eq(cohort)
                row[f"exclusive_{key}_count"] = int(mask.sum())
                row[f"exclusive_{key}_rate"] = float(mask.mean())
            rows.append(row)
    return pd.DataFrame(rows)


def _cluster_bootstrap_difference(
    frame: pd.DataFrame,
    *,
    group_column: str,
    level_a: str,
    level_b: str,
    value_column: str,
    resamples: int,
    random_state: int,
) -> tuple[float, float, int]:
    seeds = pd.unique(frame["training_seed"])
    if len(seeds) == 0:
        return np.nan, np.nan, 0
    aggregates = []
    for seed in seeds:
        group = frame.loc[frame["training_seed"].eq(seed)]
        values = pd.to_numeric(group[value_column], errors="coerce")
        a = values.loc[group[group_column].astype(str).eq(level_a)]
        b = values.loc[group[group_column].astype(str).eq(level_b)]
        aggregates.append(
            (
                float(a.sum(skipna=True)),
                int(a.notna().sum()),
                float(b.sum(skipna=True)),
                int(b.notna().sum()),
            )
        )
    values = np.asarray(aggregates, dtype=float)
    rng = np.random.default_rng(random_state)
    samples: list[float] = []
    for _ in range(int(resamples)):
        chosen = rng.integers(0, len(seeds), size=len(seeds))
        totals = values[chosen].sum(axis=0)
        if totals[1] > 0 and totals[3] > 0:
            samples.append(float(totals[0] / totals[1] - totals[2] / totals[3]))
    if not samples:
        return np.nan, np.nan, 0
    low, high = np.quantile(np.asarray(samples), [0.025, 0.975])
    return float(low), float(high), len(samples)


def _seed_stratified_permutation(
    frame: pd.DataFrame,
    *,
    group_column: str,
    level_a: str,
    level_b: str,
    value_column: str,
    resamples: int,
    random_state: int,
) -> tuple[float, int, int]:
    subset = frame.loc[frame[group_column].astype(str).isin([level_a, level_b])].copy()
    subset[value_column] = pd.to_numeric(subset[value_column], errors="coerce")
    subset = subset.dropna(subset=[value_column])
    observed = float(
        subset.loc[subset[group_column].astype(str).eq(level_a), value_column].mean()
        - subset.loc[subset[group_column].astype(str).eq(level_b), value_column].mean()
    )
    swappable = [
        group.index.to_numpy()
        for _, group in subset.groupby("training_seed", sort=False)
        if group[group_column].astype(str).nunique() == 2
    ]
    if not swappable:
        return np.nan, 0, 0
    labels = subset[group_column].astype(str).to_numpy(copy=True)
    values = subset[value_column].to_numpy(dtype=float)
    index_to_position = {index: position for position, index in enumerate(subset.index)}
    positions = [
        np.asarray([index_to_position[index] for index in indices], dtype=int)
        for indices in swappable
    ]
    rng = np.random.default_rng(random_state)
    extreme = 0
    for _ in range(int(resamples)):
        permuted = labels.copy()
        for position in positions:
            permuted[position] = rng.permutation(permuted[position])
        a = values[permuted == level_a]
        b = values[permuted == level_b]
        statistic = float(a.mean() - b.mean())
        extreme += int(abs(statistic) >= abs(observed) - 1e-15)
    return (
        float((extreme + 1) / (int(resamples) + 1)),
        len(swappable),
        int(sum(len(position) for position in positions)),
    )


def _binary_contrasts(
    triads: pd.DataFrame,
    *,
    min_group_n: int,
    bootstrap_resamples: int,
    permutation_resamples: int,
    random_state: int,
) -> pd.DataFrame:
    frame = triads.copy()
    for cohort in _EXCLUSIVE_LEVELS:
        frame[f"exclusive__{cohort}"] = frame["exclusive_cohort"].astype(str).eq(
            cohort
        )
    metric_columns = (
        "G_TN",
        "G_FN",
        "dual_improvement",
        "high_value",
        "dual_harm",
        "raw_dual_safe",
        *tuple(f"exclusive__{cohort}" for cohort in _EXCLUSIVE_LEVELS),
    )
    comparisons = []
    for family, (column, levels) in _STRATA.items():
        for left_index, level_a in enumerate(levels):
            for level_b in levels[left_index + 1 :]:
                comparisons.append((family, column, level_a, level_b))
    rows: list[dict[str, object]] = []
    for comparison_index, (family, column, level_a, level_b) in enumerate(comparisons):
        subset = frame.loc[frame[column].astype(str).isin([level_a, level_b])].copy()
        n_a = int(subset[column].astype(str).eq(level_a).sum())
        n_b = int(subset[column].astype(str).eq(level_b).sum())
        enough = n_a >= min_group_n and n_b >= min_group_n
        for metric_index, metric in enumerate(metric_columns):
            values = pd.to_numeric(subset[metric], errors="coerce")
            a_values = values.loc[subset[column].astype(str).eq(level_a)].dropna()
            b_values = values.loc[subset[column].astype(str).eq(level_b)].dropna()
            difference = float(a_values.mean() - b_values.mean())
            ci_low = ci_high = np.nan
            bootstrap_valid = 0
            permutation_p = np.nan
            swappable_seed_count = 0
            swappable_rows = 0
            if enough:
                ci_low, ci_high, bootstrap_valid = _cluster_bootstrap_difference(
                    subset,
                    group_column=column,
                    level_a=level_a,
                    level_b=level_b,
                    value_column=metric,
                    resamples=bootstrap_resamples,
                    random_state=random_state + comparison_index * 101 + metric_index,
                )
                (
                    permutation_p,
                    swappable_seed_count,
                    swappable_rows,
                ) = _seed_stratified_permutation(
                    subset,
                    group_column=column,
                    level_a=level_a,
                    level_b=level_b,
                    value_column=metric,
                    resamples=permutation_resamples,
                    random_state=random_state + comparison_index * 211 + metric_index,
                )
            if not enough:
                status = "INSUFFICIENT_GROUP_SIZE"
            elif swappable_seed_count == 0:
                status = "NO_WITHIN_SEED_EXCHANGEABILITY"
            else:
                status = "ESTIMATED"
            rows.append(
                {
                    "contrast_id": f"{family}:{level_a}-vs-{level_b}",
                    "stratum_family": family,
                    "level_a": level_a,
                    "level_b": level_b,
                    "metric": metric.removeprefix("exclusive__"),
                    "metric_kind": "mean" if metric in {"G_TN", "G_FN"} else "rate",
                    "n_level_a": n_a,
                    "n_level_b": n_b,
                    "level_a_mean_or_rate": float(a_values.mean()),
                    "level_b_mean_or_rate": float(b_values.mean()),
                    "level_a_minus_level_b": difference,
                    "seed_cluster_bootstrap_ci_low": ci_low,
                    "seed_cluster_bootstrap_ci_high": ci_high,
                    "bootstrap_valid_resamples": bootstrap_valid,
                    "bootstrap_cluster": "training_seed",
                    "permutation_p_value": permutation_p,
                    "permutation_stratification": "training_seed",
                    "permutation_swappable_seed_count": swappable_seed_count,
                    "permutation_swappable_row_count": swappable_rows,
                    "analysis_status": status,
                    "causal_interpretation": "NOT_CAUSAL_ADJUSTMENT",
                }
            )
    result = pd.DataFrame(rows)
    result["q_value_bh"] = benjamini_hochberg(result["permutation_p_value"])
    return result


def _within_pair_outputs(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = pairs.copy()
    detail["machine_stratum"] = np.where(
        detail["same_machine"], "SAME_MACHINE", "CROSS_MACHINE"
    )
    detail["snapshot_stratum"] = np.where(
        detail["same_input_snapshot"], "SAME_SNAPSHOT", "CROSS_SNAPSHOT"
    )
    detail["resume_stratum"] = np.where(
        detail["any_pair_resume"], "ANY_ARM_RESUME", "NO_RESUME"
    )
    detail["budget_stratum"] = "B" + detail["budget"].astype(int).astype(str)
    detail["discovery_confirmation_stratum"] = (
        detail["discovery_or_confirmation"].astype(str).str.upper()
    )
    rows: list[dict[str, object]] = []
    for family, (column, levels) in _STRATA.items():
        for level in levels:
            for control in _CONTROLS:
                group = detail.loc[
                    detail[column].astype(str).eq(level)
                    & detail["control_arm"].astype(str).eq(control)
                ]
                if group.empty:
                    continue
                rows.append(
                    {
                        "stratum_family": family,
                        "stratum_level": level,
                        "control_arm": control,
                        "n_condition_seed_pairs": len(group),
                        "n_training_seeds": group["training_seed"].nunique(),
                        "n_conditions": group["condition_id"].nunique(),
                        "delta_TN_mean": float(group["delta_TN"].mean()),
                        "delta_TN_median": float(group["delta_TN"].median()),
                        "delta_FN_mean": float(group["delta_FN"].mean()),
                        "delta_FN_median": float(group["delta_FN"].median()),
                        "paired_improvement_rate": float(group["paired_improvement"].mean()),
                        "paired_high_value_rate": float(group["paired_high_value"].mean()),
                        "paired_harm_rate": float(group["paired_harm"].mean()),
                        "raw_safe_frontier_dominant_rate": float(
                            group["raw_safe_frontier_dominant"].mean()
                        ),
                    }
                )
    ordered = [
        "triad_id",
        "phase",
        "condition_id",
        "method",
        "budget",
        "training_seed",
        "discovery_or_confirmation",
        "control_arm",
        "treatment_run_slot",
        "control_run_slot",
        "treatment_machine_id",
        "control_machine_id",
        "same_machine",
        "treatment_input_snapshot_id",
        "control_input_snapshot_id",
        "same_input_snapshot",
        "treatment_resume_count",
        "control_resume_count",
        "any_pair_resume",
        "delta_TN",
        "delta_FN",
        "paired_improvement",
        "paired_high_value",
        "paired_harm",
        "raw_safe_frontier_dominant",
        "machine_stratum",
        "snapshot_stratum",
        "resume_stratum",
        "budget_stratum",
        "discovery_confirmation_stratum",
    ]
    return detail[ordered].sort_values(
        ["triad_id", "control_arm"], ignore_index=True
    ), pd.DataFrame(rows)


def analyze_confound_sensitivity(
    outcomes: pd.DataFrame,
    resources: pd.DataFrame,
    raw_frontier: pd.DataFrame,
    canonical: pd.DataFrame,
    *,
    expected_triads: int = 80,
    min_group_n: int = 5,
    bootstrap_resamples: int = 5_000,
    permutation_resamples: int = 20_000,
    random_state: int = 20260806,
) -> ConfoundSensitivityResult:
    """Build descriptive confound slices and seed-aware two-level contrasts."""

    if min_group_n < 2:
        raise ValueError("min_group_n must be at least 2")
    triads, pairs = _validate_and_join_inputs(
        outcomes,
        resources,
        raw_frontier,
        canonical,
        expected_triads=expected_triads,
    )
    strata_summary = _strata_summary(triads)
    contrasts = _binary_contrasts(
        triads,
        min_group_n=min_group_n,
        bootstrap_resamples=bootstrap_resamples,
        permutation_resamples=permutation_resamples,
        random_state=random_state,
    )
    within_detail, within_summary = _within_pair_outputs(pairs)
    baseline = {
        "triads": len(triads),
        "dual_improvement": int(triads["dual_improvement"].sum()),
        "high_value": int(triads["high_value"].sum()),
        "dual_harm": int(triads["dual_harm"].sum()),
        "mixed_or_reversal_exclusive": int(
            triads["exclusive_cohort"].eq("MIXED_OR_REVERSAL").sum()
        ),
        "raw_dual_safe": int(triads["raw_dual_safe"].sum()),
    }
    summary: dict[str, Any] = {
        "schema_version": "stage1-gapvalue240-confound-sensitivity-v1",
        "analysis_unit": "T_R1_R2_TRIAD",
        "machine_adjustment_interpretation": (
            "DESCRIPTIVE_SENSITIVITY_NOT_CAUSAL_CORRECTION"
        ),
        "raw_dual_safe_definition": (
            "raw score safe_frontier_dominant is true against both R1 and R2 "
            "over integer FN budgets 0..95"
        ),
        "four_class_definition": (
            "exclusive priority: DUAL_IMPROVEMENT, HIGH_VALUE, DUAL_HARM, "
            "MIXED_OR_REVERSAL"
        ),
        "baseline": baseline,
        "strata_counts": {
            family: {
                str(row.stratum_level): int(row.n_triads)
                for row in strata_summary.loc[
                    strata_summary["stratum_family"].eq(family)
                ].itertuples(index=False)
            }
            for family in _STRATA
        },
        "inference": {
            "minimum_group_n": int(min_group_n),
            "bootstrap_cluster": "training_seed",
            "bootstrap_resamples": int(bootstrap_resamples),
            "permutation_stratification": "training_seed",
            "permutation_resamples": int(permutation_resamples),
            "estimated_contrast_rows": int(
                contrasts["analysis_status"].eq("ESTIMATED").sum()
            ),
            "no_within_seed_exchangeability_rows": int(
                contrasts["analysis_status"]
                .eq("NO_WITHIN_SEED_EXCHANGEABILITY")
                .sum()
            ),
            "insufficient_group_size_rows": int(
                contrasts["analysis_status"].eq("INSUFFICIENT_GROUP_SIZE").sum()
            ),
        },
        "limitations": [
            "Machine, resume, snapshot, budget, and discovery strata were not randomized.",
            "Stratified differences are sensitivity descriptions, not causal corrections.",
            "Phase C condition and training seeds do not overlap discovery seeds, so a "
            "seed-stratified permutation is not exchangeable for that contrast.",
            "Raw dual-safe frontier dominance is stricter than a gain at one threshold.",
        ],
    }
    return ConfoundSensitivityResult(
        triads=triads,
        strata_summary=strata_summary,
        binary_contrasts=contrasts,
        within_condition_seed=within_detail,
        within_pair_summary=within_summary,
        summary=summary,
    )


def _stage_bytes(payload: bytes, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def publish_confound_sensitivity(
    result: ConfoundSensitivityResult,
    output_dir: str | Path,
    *,
    source_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, int]:
    """Atomically publish confound tables without touching analysis state."""

    output = Path(output_dir).resolve()
    if not output.name.endswith(".inprogress"):
        raise ValueError("Confound sensitivity output must remain .inprogress")
    tables = output / "tables"
    frames = {
        "confound_sensitivity_triad_strata.csv": result.triads,
        "confound_sensitivity_strata_summary.csv": result.strata_summary,
        "confound_sensitivity_binary_contrasts.csv": result.binary_contrasts,
        "confound_sensitivity_within_condition_seed.csv": result.within_condition_seed,
        "confound_sensitivity_within_pair_summary.csv": result.within_pair_summary,
    }
    summary_name = "confound_sensitivity_summary.json"
    manifest_name = "confound_sensitivity_output_manifest.csv"
    targets = [tables / name for name in (*frames, summary_name, manifest_name)]
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite confound outputs: {existing}")

    source_hashes = {
        str(label): sha256(Path(path).read_bytes()).hexdigest().upper()
        for label, path in (source_paths or {}).items()
    }
    summary = {**result.summary, "source_sha256": source_hashes}
    payloads: dict[str, tuple[bytes, int]] = {
        name: (frame.to_csv(index=False, lineterminator="\n").encode("utf-8"), len(frame))
        for name, frame in frames.items()
    }
    payloads[summary_name] = (
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
        1,
    )
    manifest = pd.DataFrame(
        [
            {
                "filename": name,
                "sha256": sha256(payload).hexdigest().upper(),
                "size_bytes": len(payload),
                "row_count": rows,
            }
            for name, (payload, rows) in payloads.items()
        ]
    ).sort_values("filename", ignore_index=True)
    payloads[manifest_name] = (
        manifest.to_csv(index=False, lineterminator="\n").encode("utf-8"),
        len(manifest),
    )

    staged: dict[Path, Path] = {}
    try:
        for name, (payload, _) in payloads.items():
            target = tables / name
            staged[target] = _stage_bytes(payload, target)
        for target, temporary in staged.items():
            os.replace(temporary, target)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
    return {
        "output_files": len(payloads),
        "triad_rows": len(result.triads),
        "strata_summary_rows": len(result.strata_summary),
        "binary_contrast_rows": len(result.binary_contrasts),
        "within_condition_seed_rows": len(result.within_condition_seed),
        "within_pair_summary_rows": len(result.within_pair_summary),
        "manifest_rows": len(manifest),
    }
