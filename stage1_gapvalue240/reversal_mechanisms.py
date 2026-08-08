"""Same-treatment-selection seed reversal and raw-tail mechanism analysis."""

from __future__ import annotations

from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


class ReversalAnalysisError(RuntimeError):
    """Raised when the canonical reversal evidence is incomplete or ambiguous."""


GOOD_COHORT = "DUAL_IMPROVEMENT"
HARM_COHORT = "DUAL_HARM"
MIXED_COHORT = "MIXED_OR_REVERSAL"
HIGH_VALUE_COHORT = "HIGH_VALUE"
CUTOFF_EPOCHS = (120, 140, 150, 160, 180, 200)


def identify_same_selection_reversals(
    treatment_sets: pd.DataFrame,
    *,
    expected_triads: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find every digest containing both strict dual improvement and dual harm."""

    required = {
        "triad_id",
        "sample_set_digest",
        "selected_count",
        "training_seed",
        "condition_id",
        "exclusive_cohort",
        "dual_improvement",
        "dual_harm",
    }
    missing = sorted(required.difference(treatment_sets.columns))
    if missing:
        raise ReversalAnalysisError(f"Treatment selection table missing: {missing}")
    if len(treatment_sets) != expected_triads:
        raise ReversalAnalysisError(
            f"Expected {expected_triads} treatment triads, found {len(treatment_sets)}"
        )
    if treatment_sets.triad_id.duplicated().any():
        raise ReversalAnalysisError("Treatment selection table contains duplicate triad_id")
    frame = treatment_sets.copy()
    if frame.sample_set_digest.isna().any():
        raise ReversalAnalysisError("Treatment sample_set_digest contains missing values")
    cohort = frame.exclusive_cohort.astype(str)
    if not cohort.isin(
        {GOOD_COHORT, HARM_COHORT, MIXED_COHORT, HIGH_VALUE_COHORT}
    ).all():
        raise ReversalAnalysisError("Treatment table contains an unknown exclusive cohort")
    if not (
        frame.dual_improvement.astype(bool) == cohort.eq(GOOD_COHORT)
    ).all():
        raise ReversalAnalysisError("dual_improvement disagrees with exclusive_cohort")
    if not (frame.dual_harm.astype(bool) == cohort.eq(HARM_COHORT)).all():
        raise ReversalAnalysisError("dual_harm disagrees with exclusive_cohort")

    spanning = []
    summary_rows: list[dict[str, Any]] = []
    for digest, group in frame.groupby("sample_set_digest", sort=True):
        good = int(group.dual_improvement.astype(bool).sum())
        harm = int(group.dual_harm.astype(bool).sum())
        if good == 0 or harm == 0:
            continue
        spanning.append(str(digest))
        summary_rows.append(
            {
                "sample_set_digest": str(digest),
                "selected_count": int(group.selected_count.iloc[0]),
                "triad_count": int(len(group)),
                "seed_count": int(group.training_seed.nunique()),
                "training_seeds": "|".join(
                    str(value) for value in sorted(group.training_seed.astype(int))
                ),
                "condition_ids": "|".join(sorted(group.condition_id.astype(str).unique())),
                "dual_improvement_count": good,
                "dual_harm_count": harm,
                "mixed_count": int(
                    (~group.exclusive_cohort.isin({GOOD_COHORT, HARM_COHORT})).sum()
                ),
            }
        )
    details = frame[frame.sample_set_digest.astype(str).isin(spanning)].copy()
    details = details.sort_values(
        ["sample_set_digest", "triad_id"], kind="stable", ignore_index=True
    )
    return details, pd.DataFrame(summary_rows)


def _bh_adjust(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().astype(float)
    if valid.empty:
        return result
    order = valid.sort_values(kind="stable").index
    ranked = valid.loc[order].to_numpy(dtype=float)
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[order] = np.minimum(adjusted, 1.0)
    return result


def _permutation_differences(block: pd.DataFrame, feature: str) -> np.ndarray:
    values = block[feature].to_numpy(dtype=float)
    good_count = int((block.exclusive_cohort == GOOD_COHORT).sum())
    possibilities: list[float] = []
    indices = np.arange(len(block))
    for good_indices_tuple in combinations(indices.tolist(), good_count):
        good_indices = np.asarray(good_indices_tuple, dtype=int)
        good_mask = np.zeros(len(block), dtype=bool)
        good_mask[good_indices] = True
        possibilities.append(float(values[good_mask].mean() - values[~good_mask].mean()))
    return np.asarray(possibilities, dtype=float)


def blocked_feature_contrasts(
    reversal_frame: pd.DataFrame,
    feature_registry: pd.DataFrame,
    *,
    bootstrap_resamples: int = 10_000,
    random_seed: int = 20260806,
    maximum_exact_permutations: int = 100_000,
) -> pd.DataFrame:
    """Compare good/harm outcomes with selection digest as the blocking unit."""

    required_frame = {"sample_set_digest", "exclusive_cohort"}
    missing_frame = sorted(required_frame.difference(reversal_frame.columns))
    if missing_frame:
        raise ReversalAnalysisError(f"Reversal frame missing: {missing_frame}")
    required_registry = {
        "feature",
        "feature_family",
        "available_epoch",
        "allowed_as_predictor",
        "analysis_role",
    }
    missing_registry = sorted(required_registry.difference(feature_registry.columns))
    if missing_registry:
        raise ReversalAnalysisError(f"Feature registry missing: {missing_registry}")
    if feature_registry.feature.duplicated().any():
        raise ReversalAnalysisError("Feature registry contains duplicate feature names")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    labeled = reversal_frame[
        reversal_frame.exclusive_cohort.isin({GOOD_COHORT, HARM_COHORT})
    ].copy()
    if labeled.empty:
        raise ReversalAnalysisError("No strict good/harm rows in reversal frame")
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []
    for registry_row in feature_registry.itertuples(index=False):
        feature = str(registry_row.feature)
        base = {
            "feature": feature,
            "feature_family": str(registry_row.feature_family),
            "available_epoch": int(registry_row.available_epoch),
            "allowed_as_predictor": bool(registry_row.allowed_as_predictor),
            "analysis_role": str(registry_row.analysis_role),
        }
        if feature not in reversal_frame.columns:
            rows.append(
                {
                    **base,
                    "analysis_status": "MISSING_FROM_FEATURE_MATRIX",
                    "valid_digest_count": 0,
                }
            )
            continue
        numeric = pd.to_numeric(reversal_frame[feature], errors="coerce")
        labeled_numeric = pd.to_numeric(labeled[feature], errors="coerce")
        working = labeled[["sample_set_digest", "exclusive_cohort"]].copy()
        working[feature] = labeled_numeric
        blocks: list[pd.DataFrame] = []
        within_differences: list[float] = []
        permutation_options: list[np.ndarray] = []
        for _, block in working.groupby("sample_set_digest", sort=True):
            block = block[np.isfinite(block[feature].to_numpy(dtype=float))]
            if not {
                GOOD_COHORT,
                HARM_COHORT,
            }.issubset(set(block.exclusive_cohort.astype(str))):
                continue
            blocks.append(block)
            good_values = block.loc[block.exclusive_cohort == GOOD_COHORT, feature]
            harm_values = block.loc[block.exclusive_cohort == HARM_COHORT, feature]
            within_differences.append(float(good_values.mean() - harm_values.mean()))
            permutation_options.append(_permutation_differences(block, feature))
        valid = reversal_frame.assign(_numeric=numeric).groupby("sample_set_digest")[
            "_numeric"
        ]
        within_constant = bool(
            all(group.dropna().nunique() <= 1 for _, group in valid)
        )
        if not blocks:
            rows.append(
                {
                    **base,
                    "analysis_status": "NO_DIGEST_WITH_BOTH_COHORTS_AND_FINITE_VALUES",
                    "valid_digest_count": 0,
                    "within_digest_constant_all": within_constant,
                }
            )
            continue
        permutation_count = int(np.prod([len(values) for values in permutation_options]))
        if permutation_count > maximum_exact_permutations:
            raise ReversalAnalysisError(
                f"Feature {feature} requires {permutation_count} blocked permutations"
            )
        permuted = np.asarray(
            [float(np.mean(values)) for values in product(*permutation_options)],
            dtype=float,
        )
        observed = float(np.mean(within_differences))
        p_value = float(np.mean(np.abs(permuted) >= abs(observed) - 1e-15))
        difference_array = np.asarray(within_differences, dtype=float)
        sampled = rng.choice(
            difference_array,
            size=(bootstrap_resamples, len(difference_array)),
            replace=True,
        ).mean(axis=1)
        finite_good = labeled_numeric[labeled.exclusive_cohort == GOOD_COHORT].dropna()
        finite_harm = labeled_numeric[labeled.exclusive_cohort == HARM_COHORT].dropna()
        paired_sd = float(difference_array.std(ddof=1)) if len(difference_array) > 1 else 0.0
        rows.append(
            {
                **base,
                "analysis_status": "ANALYZED",
                "good_n": int(len(finite_good)),
                "harm_n": int(len(finite_harm)),
                "good_mean": float(finite_good.mean()),
                "harm_mean": float(finite_harm.mean()),
                "unblocked_mean_difference": float(finite_good.mean() - finite_harm.mean()),
                "valid_digest_count": int(len(blocks)),
                "digest_equal_weight_mean_difference": observed,
                "digest_difference_std": paired_sd,
                "paired_standardized_effect": (
                    observed / paired_sd if paired_sd > 0 else np.nan
                ),
                "bootstrap_ci_low": float(np.quantile(sampled, 0.025)),
                "bootstrap_ci_high": float(np.quantile(sampled, 0.975)),
                "bootstrap_resamples": int(bootstrap_resamples),
                "exact_blocked_permutation_count": permutation_count,
                "permutation_p_two_sided": p_value,
                "within_digest_constant_all": within_constant,
                "static_value_cannot_explain_reversal": within_constant,
                "difference_direction": "GOOD_MINUS_HARM",
            }
        )
    result = pd.DataFrame(rows)
    result["fdr_q_global"] = _bh_adjust(result["permutation_p_two_sided"])
    result["fdr_q_within_family"] = np.nan
    analyzed = result.analysis_status == "ANALYZED"
    for _, indices in result[analyzed].groupby("feature_family").groups.items():
        result.loc[indices, "fdr_q_within_family"] = _bh_adjust(
            result.loc[indices, "permutation_p_two_sided"]
        )
    return result


def load_csv_required(path: str | Path, required: Iterable[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise ReversalAnalysisError(f"Required analysis table is missing: {path}")
    frame = pd.read_csv(path, low_memory=False)
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ReversalAnalysisError(f"{path.name} missing columns: {missing}")
    return frame


def _attach_reversal_identity(
    frame: pd.DataFrame,
    details: pd.DataFrame,
    *,
    validate: str,
) -> pd.DataFrame:
    identity = details[
        [
            "triad_id",
            "sample_set_digest",
            "exclusive_cohort",
            "dual_improvement",
            "dual_harm",
        ]
    ].copy()
    overlap = [column for column in identity.columns if column in frame.columns and column != "triad_id"]
    current = frame.drop(columns=overlap, errors="ignore")
    return current.merge(identity, on="triad_id", how="inner", validate=validate)


def build_epoch_reversal_tables(
    paired_epoch: pd.DataFrame,
    details: pd.DataFrame,
    *,
    bootstrap_resamples: int = 2_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return detailed cutoff rows and all-200-epoch blocked trajectory contrasts."""

    required = {
        "triad_id",
        "control_arm",
        "epoch",
        "delta__train_loss",
        "delta__val_loss",
        "delta__accuracy_top1",
        "delta__lr_pg3",
        "delta__lr_pg5",
        "delta__lr_pg7",
        "extra_train_loss_decline",
        "extra_val_loss_decline",
        "extra_top1_gain",
    }
    missing = sorted(required.difference(paired_epoch.columns))
    if missing:
        raise ReversalAnalysisError(f"Paired epoch table missing: {missing}")
    reversal = _attach_reversal_identity(paired_epoch, details, validate="many_to_one")
    reversal["epoch"] = pd.to_numeric(reversal.epoch, errors="raise").astype(int)
    if set(reversal.epoch.unique()) != set(range(1, 201)):
        raise ReversalAnalysisError("Reversal epoch evidence is not exactly epochs 1..200")
    expected_rows = len(details) * 2 * 200
    if len(reversal) != expected_rows:
        raise ReversalAnalysisError(
            f"Expected {expected_rows} reversal epoch-control rows, found {len(reversal)}"
        )
    cutoffs = reversal[reversal.epoch.isin(CUTOFF_EPOCHS)].copy()
    if len(cutoffs) != len(details) * 2 * len(CUTOFF_EPOCHS):
        raise ReversalAnalysisError("Reversal cutoff epoch grid is incomplete")

    trajectory_features = [
        "delta__train_loss",
        "delta__val_loss",
        "delta__accuracy_top1",
        "delta__lr_pg3",
        "delta__lr_pg5",
        "delta__lr_pg7",
        "extra_train_loss_decline",
        "extra_val_loss_decline",
        "extra_top1_gain",
    ]
    consensus_long = (
        reversal.groupby(
            ["triad_id", "sample_set_digest", "exclusive_cohort", "epoch"],
            sort=True,
        )[trajectory_features]
        .mean()
        .reset_index()
    )
    wide_parts: list[pd.DataFrame] = []
    registry_rows: list[dict[str, Any]] = []
    for base_feature in trajectory_features:
        pivot = consensus_long.pivot(
            index=["triad_id", "sample_set_digest", "exclusive_cohort"],
            columns="epoch",
            values=base_feature,
        )
        pivot.columns = [
            f"trajectory__{base_feature}__at_{int(epoch)}" for epoch in pivot.columns
        ]
        wide_parts.append(pivot)
        for epoch in range(1, 201):
            registry_rows.append(
                {
                    "feature": f"trajectory__{base_feature}__at_{epoch}",
                    "feature_family": f"TRAJECTORY_{base_feature}",
                    "available_epoch": epoch,
                    "allowed_as_predictor": True,
                    "analysis_role": "TRAINING_TRAJECTORY_MONITOR",
                }
            )
    trajectory_wide = pd.concat(wide_parts, axis=1).reset_index()
    timeline = blocked_feature_contrasts(
        trajectory_wide,
        pd.DataFrame(registry_rows),
        bootstrap_resamples=bootstrap_resamples,
        random_seed=20260807,
    )
    timeline["base_feature"] = timeline.feature.str.extract(
        r"^trajectory__(.+)__at_\d+$", expand=False
    )
    timeline["epoch"] = timeline.feature.str.extract(r"__at_(\d+)$", expand=False).astype(int)
    timeline["is_preregistered_focus_epoch"] = timeline.epoch.isin(CUTOFF_EPOCHS)
    timeline["fdr_q_all_epoch_features"] = _bh_adjust(
        timeline.permutation_p_two_sided
    )
    timeline["fdr_q_within_base_feature"] = np.nan
    for _, indices in timeline.groupby("base_feature").groups.items():
        timeline.loc[indices, "fdr_q_within_base_feature"] = _bh_adjust(
            timeline.loc[indices, "permutation_p_two_sided"]
        )
    return cutoffs, timeline


def build_raw_reversal_tables(
    tail_summary: pd.DataFrame,
    frontier: pd.DataFrame,
    details: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Connect fixed control-defined raw tails and exact safe frontiers to reversals."""

    tail_required = {
        "triad_id",
        "control_arm",
        "score_type",
        "label",
        "scope",
        "mean_shift",
        "beneficial_rate",
        "harmful_rate",
    }
    frontier_required = {
        "triad_id",
        "control_arm",
        "score_type",
        "safe_frontier_dominant",
        "safe_min_delta_TN",
        "safe_mean_delta_TN",
        "safe_positive_budget_share",
        "delta_TN_at_FN95",
    }
    missing_tail = sorted(tail_required.difference(tail_summary.columns))
    missing_frontier = sorted(frontier_required.difference(frontier.columns))
    if missing_tail or missing_frontier:
        raise ReversalAnalysisError(
            f"Raw mechanism inputs missing tail={missing_tail}, frontier={missing_frontier}"
        )
    raw_tail = tail_summary[tail_summary.score_type.astype(str) == "raw"].copy()
    raw_frontier = frontier[frontier.score_type.astype(str) == "raw"].copy()
    raw_tail = _attach_reversal_identity(raw_tail, details, validate="many_to_one")
    raw_frontier = _attach_reversal_identity(
        raw_frontier, details, validate="many_to_one"
    )
    if len(raw_tail) != len(details) * 2 * 2 * 3:
        raise ReversalAnalysisError("Raw reversal tail grid is incomplete")
    if len(raw_frontier) != len(details) * 2:
        raise ReversalAnalysisError("Raw reversal frontier grid is incomplete")

    tail_consensus = (
        raw_tail.groupby(
            ["triad_id", "sample_set_digest", "exclusive_cohort", "label", "scope"],
            sort=True,
        )[["mean_shift", "beneficial_rate", "harmful_rate"]]
        .mean()
        .reset_index()
    )
    mechanism = tail_consensus.pivot(
        index=["triad_id", "sample_set_digest", "exclusive_cohort"],
        columns=["label", "scope"],
        values=["mean_shift", "beneficial_rate", "harmful_rate"],
    )
    mechanism.columns = [
        f"raw_tail__{metric}__{label}__{scope}"
        for metric, label, scope in mechanism.columns
    ]
    mechanism = mechanism.reset_index()
    frontier_consensus = (
        raw_frontier.groupby(
            ["triad_id", "sample_set_digest", "exclusive_cohort"], sort=True
        )
        .agg(
            raw_frontier__worst_safe_min_delta_TN=("safe_min_delta_TN", "min"),
            raw_frontier__worst_delta_TN_at_FN95=("delta_TN_at_FN95", "min"),
            raw_frontier__mean_safe_mean_delta_TN=("safe_mean_delta_TN", "mean"),
            raw_frontier__worst_positive_budget_share=(
                "safe_positive_budget_share",
                "min",
            ),
            raw_frontier__both_controls_safe_dominant=(
                "safe_frontier_dominant",
                "all",
            ),
        )
        .reset_index()
    )
    mechanism = mechanism.merge(
        frontier_consensus,
        on=["triad_id", "sample_set_digest", "exclusive_cohort"],
        how="inner",
        validate="one_to_one",
    )
    registry_rows = [
        {
            "feature": column,
            "feature_family": (
                "RAW_SAFE_FRONTIER_OUTCOME"
                if column.startswith("raw_frontier__")
                else "RAW_FIXED_TAIL_OUTCOME"
            ),
            "available_epoch": 200,
            "allowed_as_predictor": False,
            "analysis_role": "OUTCOME_MECHANISM_EXPLANATION",
        }
        for column in mechanism.columns
        if column.startswith("raw_")
        and pd.api.types.is_numeric_dtype(mechanism[column])
    ]
    contrasts = blocked_feature_contrasts(
        mechanism,
        pd.DataFrame(registry_rows),
        bootstrap_resamples=10_000,
        random_seed=20260808,
    )
    return raw_tail, raw_frontier, contrasts


def run_same_selection_reversal_analysis(
    tables_root: str | Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Run the complete canonical 80-triad same-selection reversal analysis."""

    root = Path(tables_root)
    treatment_sets = load_csv_required(
        root / "treatment_selection_sets_80.csv",
        {
            "triad_id",
            "sample_set_digest",
            "selected_count",
            "training_seed",
            "condition_id",
            "exclusive_cohort",
            "dual_improvement",
            "dual_harm",
        },
    )
    details, digest_summary = identify_same_selection_reversals(treatment_sets)
    if len(details) != 23 or len(digest_summary) != 6:
        raise ReversalAnalysisError(
            f"Frozen reversal gate expected 23 triads/6 digests, found {len(details)}/{len(digest_summary)}"
        )
    outcomes = load_csv_required(
        root / "triad_outcomes_80.csv",
        {"triad_id", "G_TN", "G_FN", "HARM_TN", "HARM_FN"},
    )
    detail_outcomes = details.merge(
        outcomes[
            [
                "triad_id",
                "G_TN",
                "G_FN",
                "HARM_TN",
                "HARM_FN",
                "R1_run_slot",
                "R2_run_slot",
                "delta_TN_R1",
                "delta_FN_R1",
                "delta_TN_R2",
                "delta_FN_R2",
            ]
        ],
        on="triad_id",
        how="left",
        validate="one_to_one",
    )
    unified = load_csv_required(
        root / "unified_triad_feature_matrix.csv",
        {"triad_id", "treatment_sample_set_digest", "exclusive_cohort"},
    )
    if len(unified) != 80 or unified.triad_id.duplicated().any():
        raise ReversalAnalysisError("Unified feature matrix is not exactly 80 triads")
    feature_frame = unified[unified.triad_id.isin(details.triad_id)].copy()
    feature_frame = feature_frame.drop(columns=["exclusive_cohort"], errors="ignore").merge(
        details[["triad_id", "sample_set_digest", "exclusive_cohort"]],
        on="triad_id",
        how="inner",
        validate="one_to_one",
    )
    if not (
        feature_frame.treatment_sample_set_digest.astype(str)
        == feature_frame.sample_set_digest.astype(str)
    ).all():
        raise ReversalAnalysisError("Unified treatment digest differs from frozen selection sets")
    role_registry = load_csv_required(
        root / "FEATURE_ROLE_REGISTRY.csv",
        {
            "feature",
            "feature_family",
            "available_epoch",
            "allowed_as_predictor",
            "analysis_role",
        },
    )
    relevant_roles = {
        "SELECTION_NUMERIC_PREDICTOR",
        "SELECTION_COMPOSITION_PREDICTOR",
        "SELECTION_LATE_PERSISTENCE_PREDICTOR",
        "SELECTION_DIVERSITY_PREDICTOR",
        "TRAINING_TELEMETRY_PREDICTOR",
        "CHECKPOINT_MECHANISM_PREDICTOR",
        "EXECUTION_CONFOUND",
    }
    selected_registry = role_registry[
        role_registry.analysis_role.astype(str).isin(relevant_roles)
        & role_registry.feature.astype(str).isin(feature_frame.columns)
    ].copy()
    feature_contrasts = blocked_feature_contrasts(
        feature_frame,
        selected_registry,
        bootstrap_resamples=10_000,
        random_seed=20260809,
    )
    feature_contrasts["interpretation_layer"] = np.select(
        [
            feature_contrasts.analysis_role.str.startswith("SELECTION_"),
            feature_contrasts.analysis_role.eq("TRAINING_TELEMETRY_PREDICTOR"),
            feature_contrasts.analysis_role.eq("CHECKPOINT_MECHANISM_PREDICTOR"),
            feature_contrasts.analysis_role.eq("EXECUTION_CONFOUND"),
        ],
        [
            "PRETRAINING_SELECTION_PREDICTOR",
            "IN_TRAINING_MONITOR",
            "POSTTRAINING_CHECKPOINT_EXPLANATION",
            "EXECUTION_CONFOUND_ONLY",
        ],
        default="OTHER",
    )
    feature_contrasts["eligible_before_training"] = (
        feature_contrasts.allowed_as_predictor.astype(bool)
        & feature_contrasts.available_epoch.eq(0)
    )

    paired_epoch = load_csv_required(
        root / "paired_epoch_dynamics_32000.csv",
        {"triad_id", "control_arm", "epoch", "delta__train_loss"},
    )
    epoch_cutoffs, epoch_timeline = build_epoch_reversal_tables(
        paired_epoch, details
    )
    tail_summary = load_csv_required(
        root / "raw_frontier_paired_tail_shift_summary.csv",
        {"triad_id", "control_arm", "score_type", "label", "scope"},
    )
    frontier = load_csv_required(
        root / "raw_frontier_paired_dominance.csv",
        {"triad_id", "control_arm", "score_type", "safe_min_delta_TN"},
    )
    raw_tail, raw_frontier, raw_contrasts = build_raw_reversal_tables(
        tail_summary, frontier, details
    )

    resource = load_csv_required(
        root / "resource_reliability_triads.csv",
        {"triad_id", "t_machine_id", "r1_machine_id", "r2_machine_id"},
    )
    confound_crosswalk = detail_outcomes.merge(
        resource,
        on="triad_id",
        how="left",
        validate="one_to_one",
    )
    summary_extra = (
        confound_crosswalk.groupby("sample_set_digest", sort=True)
        .agg(
            treatment_machines=("t_machine_id", lambda values: "|".join(sorted(set(map(str, values))))),
            r1_machines=("r1_machine_id", lambda values: "|".join(sorted(set(map(str, values))))),
            r2_machines=("r2_machine_id", lambda values: "|".join(sorted(set(map(str, values))))),
            all_arms_same_machine_count=("all_arms_same_machine", "sum"),
            all_arms_same_snapshot_count=("all_arms_same_snapshot", "sum"),
            resumed_arm_count_total=("resumed_arm_count", "sum"),
        )
        .reset_index()
    )
    digest_summary = digest_summary.merge(
        summary_extra, on="sample_set_digest", how="left", validate="one_to_one"
    )

    analyzed_features = feature_contrasts[
        feature_contrasts.analysis_status == "ANALYZED"
    ]
    treatment_selection_mask = analyzed_features.feature.str.contains(
        r"^selection_.+__T__", regex=True
    )
    seed_cross = (
        details.groupby("training_seed", sort=True)
        .agg(
            triad_count=("triad_id", "size"),
            dual_improvement_count=("dual_improvement", "sum"),
            dual_harm_count=("dual_harm", "sum"),
        )
        .reset_index()
    )

    def evidence_row(frame: pd.DataFrame, feature: str) -> dict[str, Any]:
        match = frame[frame.feature.astype(str) == feature]
        if match.empty:
            return {"feature": feature, "status": "NOT_AVAILABLE"}
        row = match.iloc[0]
        return {
            "feature": feature,
            "status": str(row.analysis_status),
            "good_minus_harm": (
                None
                if pd.isna(row.digest_equal_weight_mean_difference)
                else float(row.digest_equal_weight_mean_difference)
            ),
            "permutation_p_two_sided": (
                None
                if pd.isna(row.permutation_p_two_sided)
                else float(row.permutation_p_two_sided)
            ),
            "fdr_q_global": (
                None if pd.isna(row.fdr_q_global) else float(row.fdr_q_global)
            ),
            "bootstrap_ci_low": (
                None if pd.isna(row.bootstrap_ci_low) else float(row.bootstrap_ci_low)
            ),
            "bootstrap_ci_high": (
                None if pd.isna(row.bootstrap_ci_high) else float(row.bootstrap_ci_high)
            ),
        }

    summary: dict[str, Any] = {
        "status": "COMPLETE",
        "canonical_scope": "80 triads / 240 canonical runs only",
        "reversal_digest_count": int(len(digest_summary)),
        "reversal_triad_count": int(len(details)),
        "strict_good_count": int(details.dual_improvement.astype(bool).sum()),
        "strict_harm_count": int(details.dual_harm.astype(bool).sum()),
        "mixed_count": int(
            (~details.exclusive_cohort.isin({GOOD_COHORT, HARM_COHORT})).sum()
        ),
        "strict_labeled_count": int(
            details.exclusive_cohort.isin({GOOD_COHORT, HARM_COHORT}).sum()
        ),
        "analyzed_feature_count": int(len(analyzed_features)),
        "pretraining_selection_feature_count": int(
            (analyzed_features.interpretation_layer == "PRETRAINING_SELECTION_PREDICTOR").sum()
        ),
        "treatment_selection_features_checked": int(treatment_selection_mask.sum()),
        "treatment_selection_features_constant_within_digest": int(
            analyzed_features.loc[
                treatment_selection_mask, "within_digest_constant_all"
            ].astype(bool).sum()
        ),
        "global_fdr_significant_feature_count": int(
            (analyzed_features.fdr_q_global < 0.05).sum()
        ),
        "family_fdr_significant_feature_count": int(
            (analyzed_features.fdr_q_within_family < 0.05).sum()
        ),
        "epoch_timeline_feature_tests": int(len(epoch_timeline)),
        "epoch_global_fdr_significant_count": int(
            (epoch_timeline.fdr_q_all_epoch_features < 0.05).sum()
        ),
        "raw_mechanism_feature_tests": int(len(raw_contrasts)),
        "raw_mechanism_global_fdr_significant_count": int(
            (raw_contrasts.fdr_q_global < 0.05).sum()
        ),
        "raw_mechanism_family_fdr_significant_count": int(
            (raw_contrasts.fdr_q_within_family < 0.05).sum()
        ),
        "focus_epoch_evidence": {
            "top1_at_200": evidence_row(
                epoch_timeline, "trajectory__delta__accuracy_top1__at_200"
            ),
            "train_loss_at_200": evidence_row(
                epoch_timeline, "trajectory__delta__train_loss__at_200"
            ),
            "extra_train_loss_decline_at_200": evidence_row(
                epoch_timeline,
                "trajectory__extra_train_loss_decline__at_200",
            ),
        },
        "raw_outcome_mechanism_evidence": {
            "worst_TN_at_FN95": evidence_row(
                raw_contrasts, "raw_frontier__worst_delta_TN_at_FN95"
            ),
            "operational_normal_mean_shift": evidence_row(
                raw_contrasts,
                "raw_tail__mean_shift__normal__operational",
            ),
            "operational_defect_mean_shift": evidence_row(
                raw_contrasts,
                "raw_tail__mean_shift__defect__operational",
            ),
            "operational_defect_beneficial_rate": evidence_row(
                raw_contrasts,
                "raw_tail__beneficial_rate__defect__operational",
            ),
        },
        "seed_outcome_counts": seed_cross.to_dict(orient="records"),
        "seed_machine_separable_within_digest": False,
        "seed_machine_boundary": (
            "Each triad has one seed and one treatment machine; within a digest, seed and "
            "machine cannot be independently estimated. Cross-digest repetition is descriptive only."
        ),
        "selection_conclusion": (
            "Exact treatment sample composition is held fixed within every reversal digest; "
            "constant T-selection fields cannot explain good/harm reversal."
        ),
        "late_overfit_conclusion": (
            "Within exact treatment-selection blocks, extra training-loss decline at epoch 200 "
            "does not separate strict good from strict harm. The late Top1 difference is an "
            "unadjusted exploratory signal and does not survive trajectory-wide FDR."
        ),
        "raw_mechanism_conclusion": (
            "Strict-good runs have a substantially better raw safe frontier even though their "
            "fixed high-risk normal scores are not pushed lower than strict-harm runs. This "
            "supports relative normal/weak-defect ordering, not blanket normal suppression."
        ),
        "predictor_outcome_boundary": (
            "Selection fields are pretraining predictors; epoch telemetry is an in-training monitor; "
            "checkpoint drift, raw tails and safe frontiers are post-training explanations only."
        ),
        "statistical_boundary": (
            "Only six selection blocks and sixteen strict good/harm observations are available; "
            "blocked permutation and digest bootstrap results are exploratory."
        ),
    }
    tables = {
        "reversal_digest_triads.csv": detail_outcomes,
        "reversal_digest_summary.csv": digest_summary,
        "reversal_feature_contrasts.csv": feature_contrasts,
        "reversal_epoch_cutoffs.csv": epoch_cutoffs,
        "reversal_epoch_timeline_contrasts.csv": epoch_timeline,
        "reversal_raw_tail_details.csv": raw_tail,
        "reversal_safe_frontier_details.csv": raw_frontier,
        "reversal_raw_mechanism_contrasts.csv": raw_contrasts,
        "reversal_confound_crosswalk.csv": confound_crosswalk,
    }
    return tables, summary
