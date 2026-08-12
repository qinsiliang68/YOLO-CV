"""Paired epoch trajectories, late-fit divergence, and effective LR-group audit."""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


_METRICS = {
    "train/loss": "train_loss",
    "val/loss": "val_loss",
    "metrics/accuracy_top1": "accuracy_top1",
    "metrics/accuracy_top5": "accuracy_top5",
    **{f"lr/pg{index}": f"lr_pg{index}" for index in range(8)},
}


def _cohort_name(value: object) -> str:
    return str(value).strip().upper()


def build_paired_epoch_deltas(
    curves: pd.DataFrame,
    triad_outcomes: pd.DataFrame,
    *,
    baseline_epoch: int = 121,
) -> pd.DataFrame:
    """Pair every Treatment epoch with R1/R2 and anchor late fitting at one epoch."""

    required = {"run_slot", "triad_id", "arm", "epoch", *_METRICS}
    missing = required - set(curves.columns)
    if missing:
        raise ValueError(f"Epoch curves missing columns: {sorted(missing)}")
    if curves.duplicated(["run_slot", "epoch"]).any():
        raise ValueError("Epoch curves contain duplicate run/epoch rows")
    if triad_outcomes["triad_id"].duplicated().any():
        raise ValueError("Triad outcomes must contain one row per triad")
    records: list[pd.DataFrame] = []
    metric_columns = list(_METRICS)
    for triad_id, group in curves.groupby("triad_id", sort=True):
        if sorted(group["arm"].drop_duplicates().astype(str).tolist()) != ["R1", "R2", "T"]:
            raise ValueError(f"Triad {triad_id} does not contain T/R1/R2")
        treatment = group.loc[group["arm"].astype(str) == "T", ["epoch", "run_slot", *metric_columns]].rename(
            columns={
                "run_slot": "treatment_run_slot",
                **{column: f"treatment__{_METRICS[column]}" for column in metric_columns},
            }
        )
        for control_arm in ("R1", "R2"):
            control = group.loc[
                group["arm"].astype(str) == control_arm,
                ["epoch", "run_slot", *metric_columns],
            ].rename(
                columns={
                    "run_slot": "control_run_slot",
                    **{column: f"control__{_METRICS[column]}" for column in metric_columns},
                }
            )
            paired = treatment.merge(control, on="epoch", validate="one_to_one")
            paired.insert(0, "triad_id", str(triad_id))
            paired.insert(1, "control_arm", control_arm)
            for normalized in _METRICS.values():
                paired[f"delta__{normalized}"] = (
                    pd.to_numeric(paired[f"treatment__{normalized}"], errors="raise")
                    - pd.to_numeric(paired[f"control__{normalized}"], errors="raise")
                )
            baseline = paired.loc[pd.to_numeric(paired["epoch"]) == baseline_epoch]
            if len(baseline) != 1:
                raise ValueError(
                    f"Triad {triad_id}/{control_arm} needs exactly one baseline epoch {baseline_epoch}"
                )
            base = baseline.iloc[0]
            paired["extra_train_loss_decline"] = (
                float(base["delta__train_loss"]) - paired["delta__train_loss"]
            )
            paired["extra_val_loss_decline"] = (
                float(base["delta__val_loss"]) - paired["delta__val_loss"]
            )
            paired["extra_top1_gain"] = (
                paired["delta__accuracy_top1"] - float(base["delta__accuracy_top1"])
            )
            before_baseline = pd.to_numeric(paired["epoch"]) < baseline_epoch
            paired.loc[
                before_baseline,
                ["extra_train_loss_decline", "extra_val_loss_decline", "extra_top1_gain"],
            ] = np.nan
            records.append(paired)
    result = pd.concat(records, ignore_index=True).merge(
        triad_outcomes,
        on="triad_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_outcome"),
    )
    if result["exclusive_cohort"].isna().any():
        raise ValueError("Epoch pairs are missing frozen outcome cohorts")
    result["exclusive_cohort_normalized"] = result["exclusive_cohort"].map(_cohort_name)
    return result.sort_values(
        ["triad_id", "control_arm", "epoch"], ignore_index=True
    )


def build_divergence_timeline(
    paired_epochs: pd.DataFrame,
    *,
    features: Iterable[str] = (
        "extra_train_loss_decline",
        "extra_val_loss_decline",
        "extra_top1_gain",
        "delta__train_loss",
        "delta__val_loss",
        "delta__accuracy_top1",
    ),
) -> pd.DataFrame:
    """Contrast strict dual-improvement and dual-harm trajectories by epoch."""

    feature_columns = tuple(features)
    required = {"triad_id", "epoch", "control_arm", "exclusive_cohort_normalized", *feature_columns}
    missing = required - set(paired_epochs.columns)
    if missing:
        raise ValueError(f"Paired epochs missing columns: {sorted(missing)}")
    consensus = (
        paired_epochs.groupby(
            ["triad_id", "epoch", "exclusive_cohort_normalized"], sort=True
        )[list(feature_columns)]
        .mean()
        .reset_index()
    )
    subset = consensus.loc[
        consensus["exclusive_cohort_normalized"].isin(
            ["DUAL_IMPROVEMENT", "DUAL_HARM"]
        )
    ]
    records: list[dict[str, object]] = []
    for epoch, group in subset.groupby("epoch", sort=True):
        labels = group["exclusive_cohort_normalized"].eq("DUAL_HARM").astype(int)
        for feature in feature_columns:
            values = pd.to_numeric(group[feature], errors="coerce")
            valid = values.notna()
            feature_labels = labels.loc[valid]
            values = values.loc[valid]
            if values.empty:
                continue
            positive = values.loc[feature_labels.eq(0)]
            negative = values.loc[feature_labels.eq(1)]
            pooled_variance = (
                ((len(positive) - 1) * positive.var(ddof=1) if len(positive) > 1 else 0.0)
                + ((len(negative) - 1) * negative.var(ddof=1) if len(negative) > 1 else 0.0)
            ) / max(len(positive) + len(negative) - 2, 1)
            pooled_std = float(np.sqrt(max(pooled_variance, 0.0)))
            if feature_labels.nunique() == 2 and values.nunique() > 1:
                auc = float(roc_auc_score(feature_labels, values))
            else:
                auc = 0.5
            records.append(
                {
                    "epoch": int(epoch),
                    "feature": feature,
                    "positive_n": int(len(positive)),
                    "negative_n": int(len(negative)),
                    "dual_improvement_mean": float(positive.mean()),
                    "dual_harm_mean": float(negative.mean()),
                    "harm_minus_improvement": float(negative.mean() - positive.mean()),
                    "cohens_d_harm_minus_improvement": (
                        float((negative.mean() - positive.mean()) / pooled_std)
                        if pooled_std > 0
                        else 0.0
                    ),
                    "harm_prediction_auc": auc,
                }
            )
    return pd.DataFrame(records)


def audit_learning_rate_groups(
    curves: pd.DataFrame,
    optimizer_groups: pd.DataFrame,
    run_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Separate active optimizer groups from empty scheduler scaffolding."""

    required_optimizer = {"run_slot", "group_index", "active"}
    missing = required_optimizer - set(optimizer_groups.columns)
    if missing:
        raise ValueError(f"Optimizer groups missing columns: {sorted(missing)}")
    if set(run_metadata["run_slot"].astype(str)) != set(curves["run_slot"].astype(str)):
        raise ValueError("LR metadata run slots do not match curves")
    metadata = run_metadata[["run_slot", "budget"]].drop_duplicates("run_slot")
    if "budget" in curves.columns:
        joined = curves.copy()
        check = joined[["run_slot", "budget"]].drop_duplicates("run_slot").merge(
            metadata,
            on="run_slot",
            how="left",
            validate="one_to_one",
            suffixes=("_curves", "_metadata"),
        )
        if not pd.to_numeric(check["budget_curves"], errors="raise").equals(
            pd.to_numeric(check["budget_metadata"], errors="raise")
        ):
            raise ValueError("LR curve budget differs from run metadata")
    else:
        joined = curves.merge(metadata, on="run_slot", validate="many_to_one")
    records: list[dict[str, object]] = []
    for budget, budget_curves in joined.groupby("budget", sort=True):
        budget_runs = set(budget_curves["run_slot"].astype(str))
        for group_index in range(8):
            group_status = optimizer_groups.loc[
                (optimizer_groups["group_index"] == group_index)
                & optimizer_groups["run_slot"].astype(str).isin(budget_runs)
            ]
            if group_status["run_slot"].nunique() != len(budget_runs):
                raise ValueError(
                    f"Budget {budget}/pg{group_index} optimizer status is incomplete"
                )
            column = f"lr/pg{group_index}"
            schedules = budget_curves.pivot(
                index="run_slot", columns="epoch", values=column
            ).sort_index(axis=1)
            unique_signatures = {
                hashlib.sha256(np.asarray(row, dtype=np.float64).tobytes()).hexdigest()
                for row in schedules.to_numpy()
            }
            epoch_values = schedules.mean(axis=0)
            epoch_axis = epoch_values.index.to_numpy(dtype=float)
            lr_axis = epoch_values.to_numpy(dtype=float)
            slopes = np.diff(lr_axis) / np.diff(epoch_axis)
            second_difference = np.diff(slopes)
            late = epoch_values.loc[epoch_values.index >= 120]
            late_epoch_axis = late.index.to_numpy(dtype=float)
            late_lr_axis = late.to_numpy(dtype=float)
            late_slopes = np.diff(late_lr_axis) / np.diff(late_epoch_axis)
            late_second_difference = np.diff(late_slopes)
            record = {
                "budget": int(budget),
                "group_index": group_index,
                "run_count": len(budget_runs),
                "active_run_count": int(group_status["active"].astype(bool).sum()),
                "active_all_runs": bool(group_status["active"].astype(bool).all()),
                "scheduler_only_all_runs": bool((~group_status["active"].astype(bool)).all()),
                "schedule_signature_count": len(unique_signatures),
                "first_epoch": int(epoch_values.index.min()),
                "last_epoch": int(epoch_values.index.max()),
                "lr_first": float(epoch_values.iloc[0]),
                "lr_last": float(epoch_values.iloc[-1]),
                "max_abs_second_difference": (
                    float(np.max(np.abs(second_difference)))
                    if len(second_difference)
                    else 0.0
                ),
                "max_abs_second_difference_after_120": (
                    float(np.max(np.abs(late_second_difference)))
                    if len(late_second_difference)
                    else 0.0
                ),
            }
            for cutoff in (120, 140, 150, 160, 180, 200):
                record[f"lr_at_{cutoff}"] = (
                    float(epoch_values.loc[cutoff]) if cutoff in epoch_values.index else np.nan
                )
            records.append(record)
    return pd.DataFrame(records).sort_values(
        ["budget", "group_index"], ignore_index=True
    )
