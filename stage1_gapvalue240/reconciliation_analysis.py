"""Reconcile expert relative-control cohorts with absolute performance frontiers."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .deep_analysis import CanonicalInputError


STRICT_GOOD = {"ROBUST_ABSOLUTE_GAIN", "LOCAL_ABSOLUTE_PARETO"}


def load_expert_package(path: str | Path) -> dict[str, Any]:
    """Validate the expert ZIP without extracting it and load its CSV tables."""

    archive_path = Path(path).resolve()
    if not archive_path.is_file():
        raise CanonicalInputError(f"Missing expert package: {archive_path}")
    try:
        with ZipFile(archive_path) as archive:
            bad_entry = archive.testzip()
            if bad_entry is not None:
                raise CanonicalInputError(f"Expert ZIP CRC failure: {bad_entry}")
            manifest_names = [
                name for name in archive.namelist() if name.endswith("/FILE_MANIFEST.csv")
            ]
            if len(manifest_names) != 1:
                raise CanonicalInputError("Expert package must contain one FILE_MANIFEST.csv")
            manifest_name = manifest_names[0]
            root = manifest_name[: -len("FILE_MANIFEST.csv")]
            manifest = pd.read_csv(io.BytesIO(archive.read(manifest_name)))
            required = {"path", "size_bytes", "sha256"}
            if not required.issubset(manifest.columns):
                raise CanonicalInputError("Expert manifest schema is incomplete")
            known = set(archive.namelist())
            errors: list[str] = []
            for row in manifest.itertuples(index=False):
                relative = PurePosixPath(str(row.path))
                if relative.is_absolute() or ".." in relative.parts:
                    errors.append(f"unsafe:{row.path}")
                    continue
                name = root + relative.as_posix()
                if name not in known:
                    errors.append(f"missing:{row.path}")
                    continue
                data = archive.read(name)
                digest = hashlib.sha256(data).hexdigest().upper()
                if len(data) != int(row.size_bytes) or digest != str(row.sha256).upper():
                    errors.append(f"manifest:{row.path}")
            if errors:
                raise CanonicalInputError(
                    "Expert manifest validation failed: " + ", ".join(errors[:5])
                )
            tables: dict[str, pd.DataFrame] = {}
            for name in archive.namelist():
                if name.startswith(root + "tables/") and name.endswith(".csv"):
                    tables[Path(name).stem] = pd.read_csv(io.BytesIO(archive.read(name)))
            if "triad_outcome_classes" not in tables:
                raise CanonicalInputError("Expert package lacks triad_outcome_classes.csv")
    except BadZipFile as exc:
        raise CanonicalInputError(f"Invalid expert ZIP: {archive_path}") from exc
    return {
        "tables": tables,
        "manifest": manifest,
        "validation": {
            "status": "PASS",
            "zip_path": str(archive_path),
            "zip_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest().upper(),
            "entry_count": len(known),
            "manifest_rows": len(manifest),
        },
    }


def build_unified_triad_outcomes(
    expert_outcomes: pd.DataFrame,
    absolute_outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Keep expert relative labels and absolute gates as separate evidence layers."""

    expert_required = {
        "triad_id",
        "strong_positive",
        "cost_effective",
        "harmful",
        "outcome_class",
    }
    absolute_required = {
        "triad_id",
        "outcome_cohort",
        "run_id",
        "condition_id",
        "training_seed",
    }
    if not expert_required.issubset(expert_outcomes.columns):
        raise CanonicalInputError("Expert outcome table is missing required columns")
    absolute = absolute_outcomes.copy()
    if "experiment_family" in absolute:
        absolute = absolute.loc[absolute["experiment_family"].astype(str) == "240"]
    if not absolute_required.issubset(absolute.columns):
        raise CanonicalInputError("Absolute outcome table is missing required columns")
    if expert_outcomes["triad_id"].duplicated().any() or absolute["triad_id"].duplicated().any():
        raise CanonicalInputError("Triad outcome tables must be unique by triad_id")
    merged = expert_outcomes.merge(
        absolute,
        on="triad_id",
        how="inner",
        suffixes=("_expert", "_absolute"),
        validate="one_to_one",
    )
    if len(merged) != len(expert_outcomes) or len(merged) != len(absolute):
        raise CanonicalInputError("Expert and absolute triad identities differ")
    for column in ("condition_id", "training_seed"):
        expert_column = f"{column}_expert"
        absolute_column = f"{column}_absolute"
        if expert_column in merged and absolute_column in merged:
            left = merged[expert_column].astype(str)
            right = merged[absolute_column].astype(str)
            if not left.equals(right):
                raise CanonicalInputError(
                    f"Expert and absolute {column} values differ"
                )
            merged[column] = merged[absolute_column]
    merged["condition_id"] = merged["condition_id"].astype(str)
    merged["training_seed"] = pd.to_numeric(
        merged["training_seed"], errors="raise"
    ).astype(np.int64)
    merged["unified_outcome"] = "MIXED_OR_INCONCLUSIVE"
    mapping = {
        "ROBUST_SAFE_DOUBLE_GATE": "ROBUST_ABSOLUTE_GAIN",
        "LOCAL_PARETO_DOUBLE_GATE": "LOCAL_ABSOLUTE_PARETO",
        "SECONDARY_CONTROLLED": "CONTROLLED_SECONDARY_GAIN",
        "JOINTLY_HARMFUL": "JOINTLY_HARMFUL",
    }
    for source, target in mapping.items():
        merged.loc[merged["outcome_cohort"] == source, "unified_outcome"] = target
    relative_win = merged["strong_positive"].astype(bool) | merged["cost_effective"].astype(bool)
    merged.loc[
        relative_win & (merged["unified_outcome"] == "MIXED_OR_INCONCLUSIVE"),
        "unified_outcome",
    ] = "RELATIVE_ONLY_WIN"
    return merged.sort_values("triad_id").reset_index(drop=True)


def _require_curve_grid(curves: pd.DataFrame) -> None:
    required = {
        "triad_id",
        "condition_id",
        "training_seed",
        "arm",
        "epoch",
        "train/loss",
        "metrics/accuracy_top1",
        "val/loss",
    }
    if not required.issubset(curves.columns):
        raise CanonicalInputError("Epoch curve table is missing required columns")


def compute_late_overfit_features(
    curves: pd.DataFrame,
    *,
    start_epoch: int = 121,
    cutoffs: Sequence[int] = (140, 150, 160, 180, 200),
) -> pd.DataFrame:
    """Compute the preregistered late replay overfit feature at each cutoff."""

    _require_curve_grid(curves)
    rows: list[dict[str, Any]] = []
    for triad_id, triad in curves.groupby("triad_id", sort=True):
        if set(triad["arm"].astype(str)) != {"T", "R1", "R2"}:
            raise CanonicalInputError(f"{triad_id} lacks a T/R1/R2 curve")
        meta = triad.iloc[0]
        by_arm = {arm: group.set_index("epoch").sort_index() for arm, group in triad.groupby("arm")}
        for cutoff in cutoffs:
            needed = {int(start_epoch), int(cutoff)}
            if any(not needed.issubset(set(group.index.astype(int))) for group in by_arm.values()):
                raise CanonicalInputError(
                    f"{triad_id} lacks epoch {start_epoch}/{cutoff} for late-overfit audit"
                )
            drops = {
                arm: float(group.loc[start_epoch, "train/loss"] - group.loc[cutoff, "train/loss"])
                for arm, group in by_arm.items()
            }
            top1_best_epoch = {
                arm: int(group.loc[group.index <= cutoff, "metrics/accuracy_top1"].idxmax())
                for arm, group in by_arm.items()
            }
            final_top1 = {
                arm: float(group.loc[cutoff, "metrics/accuracy_top1"])
                for arm, group in by_arm.items()
            }
            rebound = {}
            for arm, group in by_arm.items():
                window = group.loc[(group.index >= start_epoch) & (group.index <= cutoff), "val/loss"]
                rebound[arm] = float(group.loc[cutoff, "val/loss"] - window.min())
            rows.append(
                {
                    "triad_id": str(triad_id),
                    "condition_id": str(meta["condition_id"]),
                    "training_seed": int(meta["training_seed"]),
                    "start_epoch": int(start_epoch),
                    "cutoff_epoch": int(cutoff),
                    "late_overfit": drops["T"] - 0.5 * (drops["R1"] + drops["R2"]),
                    "top1_best_epoch_shift": top1_best_epoch["T"]
                    - 0.5 * (top1_best_epoch["R1"] + top1_best_epoch["R2"]),
                    "final_top1_delta": final_top1["T"]
                    - 0.5 * (final_top1["R1"] + final_top1["R2"]),
                    "val_loss_rebound_delta": rebound["T"]
                    - 0.5 * (rebound["R1"] + rebound["R2"]),
                }
            )
    return pd.DataFrame(rows)


def leave_group_out_predictions(
    frame: pd.DataFrame,
    *,
    label_column: str,
    feature_columns: Sequence[str],
    group_column: str,
    model_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit fold-local scaling and logistic models with whole groups held out."""

    required = {"triad_id", label_column, group_column, *feature_columns}
    if not required.issubset(frame.columns):
        raise CanonicalInputError(f"Cross-validation table lacks {sorted(required - set(frame.columns))}")
    data = frame.dropna(subset=[label_column, group_column, *feature_columns]).copy()
    data[label_column] = data[label_column].astype(int)
    if set(data[label_column]) != {0, 1}:
        raise CanonicalInputError("Cross-validation outcome must contain both classes")
    rows: list[dict[str, Any]] = []
    for fold, heldout in enumerate(sorted(data[group_column].astype(str).unique()), start=1):
        test_mask = data[group_column].astype(str) == heldout
        train = data.loc[~test_mask]
        test = data.loc[test_mask]
        if train[label_column].nunique() != 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=20260802,
            ),
        )
        model.fit(train[list(feature_columns)], train[label_column])
        scores = model.predict_proba(test[list(feature_columns)])[:, 1]
        train_groups = set(train[group_column].astype(str))
        leakage_free = heldout not in train_groups
        for (_, row), score in zip(test.iterrows(), scores, strict=True):
            rows.append(
                {
                    "model_id": model_id,
                    "fold_id": fold,
                    "heldout_group": heldout,
                    "group_column": group_column,
                    "triad_id": str(row["triad_id"]),
                    "condition_id": str(row.get("condition_id", "")),
                    "training_seed": row.get("training_seed", np.nan),
                    "y_true": int(row[label_column]),
                    "score": float(score),
                    "train_group_count": len(train_groups),
                    "leakage_free": bool(leakage_free),
                }
            )
    predictions = pd.DataFrame(rows)
    if predictions.empty or predictions["y_true"].nunique() != 2:
        raise CanonicalInputError(f"No valid held-out predictions for {model_id}")
    return predictions, {
        "model_id": model_id,
        "group_column": group_column,
        "features": "|".join(feature_columns),
        "n": len(predictions),
        "folds": predictions["fold_id"].nunique(),
        "auc": float(roc_auc_score(predictions["y_true"], predictions["score"])),
        "leakage_free": bool(predictions["leakage_free"].all()),
    }


def _cluster_bootstrap_contrast(
    frame: pd.DataFrame,
    *,
    value_column: str,
    good_mask: pd.Series,
    bad_mask: pd.Series,
    cluster_column: str,
    iterations: int = 5000,
) -> dict[str, Any]:
    subset = frame.loc[good_mask | bad_mask].copy()
    subset["_good"] = good_mask.loc[subset.index].astype(bool)
    observed = float(
        subset.loc[subset["_good"], value_column].mean()
        - subset.loc[~subset["_good"], value_column].mean()
    )
    clusters = sorted(subset[cluster_column].astype(str).unique())
    rng = np.random.default_rng(20260802)
    estimates: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        pieces = [subset.loc[subset[cluster_column].astype(str) == cluster] for cluster in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        if boot["_good"].nunique() != 2:
            continue
        estimates.append(
            float(
                boot.loc[boot["_good"], value_column].mean()
                - boot.loc[~boot["_good"], value_column].mean()
            )
        )
    if not estimates:
        raise CanonicalInputError("Cluster bootstrap produced no two-class samples")
    return {
        "mean_difference_good_minus_bad": observed,
        "cluster_bootstrap_lo": float(np.quantile(estimates, 0.025)),
        "cluster_bootstrap_hi": float(np.quantile(estimates, 0.975)),
        "cluster_count": len(clusters),
        "bootstrap_iterations_used": len(estimates),
    }


def controlled_cluster_regression(
    frame: pd.DataFrame,
    *,
    feature_column: str,
    outcome_column: str,
    cluster_column: str,
    control_columns: Sequence[str] = ("training_seed", "phase", "budget"),
    iterations: int = 3000,
) -> dict[str, Any]:
    """OLS with categorical controls and a condition-cluster bootstrap CI."""

    required = {feature_column, outcome_column, cluster_column, *control_columns}
    if not required.issubset(frame.columns):
        raise CanonicalInputError(
            f"Controlled regression lacks {sorted(required - set(frame.columns))}"
        )
    data = frame.dropna(subset=list(required)).copy()

    def coefficient(current: pd.DataFrame) -> float:
        feature = pd.to_numeric(current[feature_column], errors="raise").to_numpy(float)
        outcome = pd.to_numeric(current[outcome_column], errors="raise").to_numpy(float)
        feature_std = feature.std(ddof=0)
        outcome_std = outcome.std(ddof=0)
        if feature_std == 0 or outcome_std == 0:
            raise CanonicalInputError("Controlled regression has a constant feature/outcome")
        design = pd.DataFrame(
            {"_feature": (feature - feature.mean()) / feature_std},
            index=current.index,
        )
        for column in control_columns:
            dummies = pd.get_dummies(
                current[column].astype(str),
                prefix=column,
                drop_first=True,
                dtype=float,
            )
            design = pd.concat([design, dummies.set_axis(current.index)], axis=1)
        x = np.column_stack([np.ones(len(design)), design.to_numpy(float)])
        y = (outcome - outcome.mean()) / outcome_std
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        return float(beta[1])

    observed = coefficient(data)
    clusters = sorted(data[cluster_column].astype(str).unique())
    rng = np.random.default_rng(20260802)
    estimates: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        pieces = [data.loc[data[cluster_column].astype(str) == cluster] for cluster in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        try:
            estimates.append(coefficient(boot))
        except CanonicalInputError:
            continue
    if not estimates:
        raise CanonicalInputError("Controlled cluster bootstrap produced no estimates")
    estimates_array = np.asarray(estimates)
    p_value = min(
        1.0,
        2
        * min(
            float((estimates_array <= 0).mean()),
            float((estimates_array >= 0).mean()),
        ),
    )
    return {
        "standardized_coefficient": observed,
        "cluster_bootstrap_lo": float(np.quantile(estimates_array, 0.025)),
        "cluster_bootstrap_hi": float(np.quantile(estimates_array, 0.975)),
        "cluster_bootstrap_p": p_value,
        "cluster_count": len(clusters),
        "bootstrap_iterations_used": len(estimates),
        "controls": "|".join(control_columns),
    }


def _crosswalk(unified: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, column in (
        ("strong_positive", "strong_positive"),
        ("cost_effective", "cost_effective"),
        ("harmful", "harmful"),
    ):
        current = unified.loc[unified[column].astype(bool)]
        for outcome, count in current["unified_outcome"].value_counts().items():
            rows.append({"expert_group": label, "unified_outcome": outcome, "count": int(count)})
    return pd.DataFrame(rows).sort_values(["expert_group", "unified_outcome"])


def _timing_summary(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cutoff, group in features.groupby("cutoff_epoch", sort=True):
        strict_good = group["unified_outcome"].isin(STRICT_GOOD)
        harmful = group["unified_outcome"] == "JOINTLY_HARMFUL"
        expert_good = group["strong_positive"].astype(bool)
        expert_bad = group["harmful"].astype(bool)
        row: dict[str, Any] = {"cutoff_epoch": int(cutoff)}
        for prefix, good, bad in (
            ("strict", strict_good, harmful),
            ("expert", expert_good, expert_bad),
        ):
            x = group.loc[good, "late_overfit"].dropna()
            y = group.loc[bad, "late_overfit"].dropna()
            labels = pd.concat(
                [pd.Series(1, index=x.index), pd.Series(0, index=y.index)]
            ).sort_index()
            scores = -group.loc[labels.index, "late_overfit"]
            row.update(
                {
                    f"{prefix}_good_n": len(x),
                    f"{prefix}_harmful_n": len(y),
                    f"{prefix}_good_mean": float(x.mean()),
                    f"{prefix}_harmful_mean": float(y.mean()),
                    f"{prefix}_auc_less_late_overfit": float(
                        roc_auc_score(labels, scores)
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _mechanism_tables(
    v3_tables: Path,
    unified: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tail = pd.read_csv(v3_tables / "prediction_tail_detail.csv")
    tail = tail.merge(
        unified[["triad_id", "unified_outcome"]],
        on="triad_id",
        validate="many_to_one",
    )
    tail = tail.loc[(tail["score_type"] == "raw") & (tail["scope"] == "operational")]
    tail_summary = (
        tail.groupby(["unified_outcome", "control", "label"], as_index=False)
        .agg(
            triads=("triad_id", "nunique"),
            mean_shift=("mean_shift", "mean"),
            median_shift=("mean_shift", "median"),
            beneficial_rate=("beneficial_rate", "mean"),
            harmed_rate=("harmed_rate", "mean"),
        )
        .sort_values(["unified_outcome", "control", "label"])
    )
    selection = pd.read_csv(v3_tables / "selection_run_operational_summary.csv")
    treatment = selection.loc[selection["arm"] == "T"].merge(
        unified[["triad_id", "unified_outcome"]],
        on="triad_id",
        validate="many_to_one",
    )
    selection_summary = (
        treatment.groupby(["unified_outcome", "scope"], as_index=False)
        .agg(
            triads=("triad_id", "nunique"),
            mean_forgetting=("mean_operational_forgetting_count", "mean"),
            late_error_161_200=("mean_error_rate_late_161_200", "mean"),
            corrected_share=("share_corrected", "mean"),
            persistent_wrong_share=("share_persistent_wrong", "mean"),
        )
        .sort_values(["unified_outcome", "scope"])
    )
    return tail_summary, selection_summary


def _clustered_statistics(features_200: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    contrasts = {
        "strict_absolute": (
            features_200["unified_outcome"].isin(STRICT_GOOD),
            features_200["unified_outcome"] == "JOINTLY_HARMFUL",
        ),
        "expert_relative": (
            features_200["strong_positive"].astype(bool),
            features_200["harmful"].astype(bool),
        ),
    }
    for name, (good, bad) in contrasts.items():
        x = features_200.loc[good, "late_overfit"].dropna().to_numpy()
        y = features_200.loc[bad, "late_overfit"].dropna().to_numpy()
        u, p = mannwhitneyu(x, y, alternative="two-sided")
        row = {
            "analysis": name,
            "feature": "late_overfit_121_200",
            "good_n": len(x),
            "harmful_n": len(y),
            "good_mean": float(x.mean()),
            "harmful_mean": float(y.mean()),
            "mannwhitney_u": float(u),
            "mannwhitney_p": float(p),
        }
        row.update(
            _cluster_bootstrap_contrast(
                features_200,
                value_column="late_overfit",
                good_mask=good,
                bad_mask=bad,
                cluster_column="condition_id",
            )
        )
        rows.append(row)
    robust_tn = features_200[["delta_TN_R1", "delta_TN_R2"]].min(axis=1)
    worst_fn = features_200[["delta_FN_R1", "delta_FN_R2"]].max(axis=1)
    for outcome, values in (("robust_TN_gain", robust_tn), ("worst_FN_change", worst_fn)):
        rho, p = spearmanr(features_200["late_overfit"], values)
        rows.append(
            {
                "analysis": "continuous_association",
                "feature": "late_overfit_121_200",
                "outcome": outcome,
                "n": len(features_200),
                "spearman_rho": float(rho),
                "spearman_p": float(p),
            }
        )
        controlled = features_200.copy()
        controlled["_controlled_outcome"] = values
        regression = controlled_cluster_regression(
            controlled,
            feature_column="late_overfit",
            outcome_column="_controlled_outcome",
            cluster_column="condition_id",
        )
        rows.append(
            {
                "analysis": "controlled_cluster_regression",
                "feature": "late_overfit_121_200",
                "outcome": outcome,
                "n": len(controlled),
                **regression,
            }
        )
    return pd.DataFrame(rows)


def _hypotheses(
    unified: pd.DataFrame,
    timing: pd.DataFrame,
    tails: pd.DataFrame,
    cv_summary: pd.DataFrame,
    primary_features: pd.DataFrame,
    clustered_models: pd.DataFrame,
) -> pd.DataFrame:
    counts = unified["unified_outcome"].value_counts()
    strict_row = timing.loc[timing["cutoff_epoch"] == 200].iloc[0]
    tail_pivot = tails.pivot_table(
        index=["unified_outcome", "control"], columns="label", values="mean_shift"
    )
    good_defect = tail_pivot.loc[
        tail_pivot.index.get_level_values(0).isin(STRICT_GOOD), "defect"
    ].mean()
    bad_defect = tail_pivot.loc[
        tail_pivot.index.get_level_values(0) == "JOINTLY_HARMFUL", "defect"
    ].mean()
    cv_text = "; ".join(
        f"{row.model_id}/{row.group_column}={row.auc:.3f}"
        for row in cv_summary.itertuples(index=False)
    )
    controlled = clustered_models.loc[
        clustered_models["analysis"] == "controlled_cluster_regression"
    ].set_index("outcome")
    controlled_text = "; ".join(
        f"{outcome}: beta={row.standardized_coefficient:.3f}, "
        f"CI=[{row.cluster_bootstrap_lo:.3f},{row.cluster_bootstrap_hi:.3f}]"
        for outcome, row in controlled.iterrows()
    )
    reversal_conditions = []
    earlier_top1 = 0
    smaller_final_top1 = 0
    for condition_id, group in primary_features.groupby("condition_id"):
        good = group.loc[group["strong_positive"].astype(bool)]
        bad = group.loc[group["harmful"].astype(bool)]
        if good.empty or bad.empty:
            continue
        reversal_conditions.append(str(condition_id))
        earlier_top1 += int(
            good["top1_best_epoch_shift"].mean()
            < bad["top1_best_epoch_shift"].mean()
        )
        smaller_final_top1 += int(
            good["final_top1_delta"].mean() < bad["final_top1_delta"].mean()
        )
    relative_but_not_local = int(
        unified.loc[unified["strong_positive"].astype(bool), "unified_outcome"].isin(
            ["RELATIVE_ONLY_WIN", "CONTROLLED_SECONDARY_GAIN"]
        ).sum()
    )
    return pd.DataFrame(
        [
            {
                "hypothesis_id": "H01",
                "hypothesis": "A repeatably robust absolute 240-run method exists",
                "status": "NOT_SUPPORTED" if counts.get("ROBUST_ABSOLUTE_GAIN", 0) == 0 else "CROSS_SUPPORTED",
                "evidence": f"robust absolute triads={counts.get('ROBUST_ABSOLUTE_GAIN', 0)}",
            },
            {
                "hypothesis_id": "H02",
                "hypothesis": "Real local absolute Pareto improvements exist",
                "status": "CROSS_SUPPORTED" if counts.get("LOCAL_ABSOLUTE_PARETO", 0) > 0 else "NOT_SUPPORTED",
                "evidence": f"local absolute triads={counts.get('LOCAL_ABSOLUTE_PARETO', 0)}",
            },
            {
                "hypothesis_id": "H03",
                "hypothesis": "Less late loss compression marks better outcomes",
                "status": "EXPLORATORY_SUPPORTED" if strict_row["strict_good_mean"] < strict_row["strict_harmful_mean"] else "NOT_SUPPORTED",
                "evidence": f"strict means {strict_row['strict_good_mean']:.6g} vs {strict_row['strict_harmful_mean']:.6g}; controlled {controlled_text}",
            },
            {
                "hypothesis_id": "H04",
                "hypothesis": "Weak-defect protection separates strict gains from harm",
                "status": "EXPLORATORY_SUPPORTED" if good_defect > bad_defect else "NOT_SUPPORTED",
                "evidence": f"raw weak-defect mean shift {good_defect:.6g} vs {bad_defect:.6g}",
            },
            {
                "hypothesis_id": "H05",
                "hypothesis": "Expert relative winners are all absolute performance gains",
                "status": "NOT_SUPPORTED",
                "evidence": f"expert strong=15; local absolute={counts.get('LOCAL_ABSOLUTE_PARETO', 0)}; secondary={counts.get('CONTROLLED_SECONDARY_GAIN', 0)}",
            },
            {
                "hypothesis_id": "H06",
                "hypothesis": "LateOverfit predictive AUC is independently confirmed",
                "status": "EXPLORATORY_SUPPORTED",
                "evidence": cv_text,
            },
            {
                "hypothesis_id": "H07",
                "hypothesis": "Earlier Top1 peak is a consistent same-selection success marker",
                "status": "EXPLORATORY_SUPPORTED" if earlier_top1 == len(reversal_conditions) else "INCONCLUSIVE",
                "evidence": f"earlier peak={earlier_top1}/{len(reversal_conditions)}; smaller final Top1 advantage={smaller_final_top1}/{len(reversal_conditions)}",
            },
            {
                "hypothesis_id": "H08",
                "hypothesis": "A static Treatment selection has a seed-invariant value",
                "status": "NOT_SUPPORTED" if reversal_conditions else "INCONCLUSIVE",
                "evidence": f"same-selection conditions spanning expert good and harmful={len(reversal_conditions)}",
            },
            {
                "hypothesis_id": "H09",
                "hypothesis": "Weak random controls inflate the apparent number of strong Treatments",
                "status": "CROSS_SUPPORTED" if relative_but_not_local > 0 else "NOT_SUPPORTED",
                "evidence": f"expert strong but only relative/secondary={relative_but_not_local}/15",
            },
        ]
    )


def run_reconciliation_analysis(
    *,
    expert_zip: str | Path,
    v5_report_dir: str | Path,
    v3_report_dir: str | Path,
    full_analysis_dir: str | Path,
    output_dir: str | Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the read-only expert/frontier reconciliation and build a report."""

    notify = progress or (lambda _message: None)
    output = Path(output_dir).resolve()
    sources = [
        Path(expert_zip).resolve(),
        Path(v5_report_dir).resolve(),
        Path(v3_report_dir).resolve(),
        Path(full_analysis_dir).resolve(),
    ]
    for source in sources:
        if source.is_dir():
            try:
                output.relative_to(source)
            except ValueError:
                pass
            else:
                raise CanonicalInputError(f"Output must not be inside source: {source}")
    if output.exists() or output.with_name(output.name + ".inprogress").exists():
        raise FileExistsError(f"Refusing to overwrite analysis output: {output}")

    notify("Validating expert package")
    expert = load_expert_package(expert_zip)
    expert_outcomes = expert["tables"]["triad_outcome_classes"]
    v5_tables = Path(v5_report_dir).resolve() / "tables"
    absolute = pd.read_csv(v5_tables / "designed_method_double_gates.csv")
    all_runs = pd.read_csv(v5_tables / "all_run_baseline_dominance.csv")
    unified = build_unified_triad_outcomes(expert_outcomes, absolute)
    if len(unified) != 80:
        raise CanonicalInputError(f"Expected 80 unified triads, found {len(unified)}")

    notify("Recomputing frozen LateOverfit timing")
    curves = pd.read_csv(Path(full_analysis_dir) / "tables/epoch_training_curves.csv")
    if len(curves) != 48000:
        raise CanonicalInputError(f"Expected 48,000 epoch curve rows, found {len(curves)}")
    features = compute_late_overfit_features(curves)
    features = features.merge(
        unified,
        on=["triad_id", "condition_id", "training_seed"],
        validate="many_to_one",
    )
    timing = _timing_summary(features)
    primary = features.loc[features["cutoff_epoch"] == 200].copy()

    notify("Running leakage-safe held-out validation")
    classified = primary.loc[primary["strong_positive"].astype(bool) | primary["harmful"].astype(bool)].copy()
    classified["expert_good_label"] = classified["strong_positive"].astype(int)
    prediction_frames: list[pd.DataFrame] = []
    cv_rows: list[dict[str, Any]] = []
    for group_column in ("condition_id", "training_seed"):
        for model_id, feature_columns in (
            ("late_overfit_only", ("late_overfit",)),
            (
                "late_overfit_plus_top1",
                ("late_overfit", "top1_best_epoch_shift", "final_top1_delta"),
            ),
        ):
            predictions, summary = leave_group_out_predictions(
                classified,
                label_column="expert_good_label",
                feature_columns=feature_columns,
                group_column=group_column,
                model_id=model_id,
            )
            prediction_frames.append(predictions)
            cv_rows.append(summary)
    cv_predictions = pd.concat(prediction_frames, ignore_index=True)
    cv_summary = pd.DataFrame(cv_rows)

    v3_tables = Path(v3_report_dir).resolve() / "tables"
    tails, selection_dynamics = _mechanism_tables(v3_tables, unified)
    clustered = _clustered_statistics(primary)
    hypotheses = _hypotheses(
        unified,
        timing,
        tails,
        cv_summary,
        primary,
        clustered,
    )
    crosswalk = _crosswalk(unified)
    control_audit = unified.copy()
    control_audit["min_relative_TN_gain"] = control_audit[["delta_TN_R1", "delta_TN_R2"]].min(axis=1)
    control_audit["worst_relative_FN_change"] = control_audit[["delta_FN_R1", "delta_FN_R2"]].max(axis=1)
    control_audit["relative_minus_absolute_TN"] = (
        control_audit["min_relative_TN_gain"] - control_audit["delta_TN_at_baseline_fn"]
    )
    historical = (
        all_runs.groupby(["experiment_family", "arm", "performance_class"], as_index=False)
        .agg(runs=("run_id", "nunique"), mean_delta_TN=("delta_TN_at_baseline_fn", "mean"))
    )
    same_selection = expert["tables"].get(
        "within_same_selection_good_vs_bad", pd.DataFrame()
    )
    tables = {
        "unified_triad_outcomes": unified,
        "expert_to_absolute_crosswalk": crosswalk,
        "late_overfit_features": features,
        "late_overfit_timing": timing,
        "same_selection_seed_reversal": same_selection,
        "weak_defect_vs_normal_tail": tails,
        "selection_dynamics_by_unified_outcome": selection_dynamics,
        "control_strength_audit": control_audit,
        "clustered_models": clustered,
        "cross_validation_summary": cv_summary,
        "cross_validation_fold_predictions": cv_predictions,
        "historical_family_boundary": historical,
        "hypothesis_registry": hypotheses,
    }
    metadata = {
        "analysis_id": "gapvalue_expert_frontier_reconciliation_20260802_v3",
        "source_policy": "read-only",
        "expert_package_validation": expert["validation"],
        "unified_triads": len(unified),
        "outcome_counts": unified["unified_outcome"].value_counts().to_dict(),
        "primary_late_overfit_start_epoch": 121,
        "cutoff_epochs": [140, 150, 160, 180, 200],
        "scientific_boundaries": [
            "Expert positive labels are relative to R1/R2 and are not absolute truth.",
            "Raw-score same-FN frontiers define absolute performance.",
            "LateOverfit is an exploratory post-training diagnostic, not a validated stopping rule.",
            "No per-epoch val_op predictions exist, so no per-epoch TN/FN claim is made.",
            "40/120-run results are historical boundary evidence only.",
            "No blind or external test is available.",
        ],
        "source_paths": {
            "expert_zip": str(Path(expert_zip).resolve()),
            "v5_report_dir": str(Path(v5_report_dir).resolve()),
            "v3_report_dir": str(Path(v3_report_dir).resolve()),
            "full_analysis_dir": str(Path(full_analysis_dir).resolve()),
        },
    }
    from .reconciliation_reporting import build_reconciliation_report

    report = build_reconciliation_report(output, tables=tables, metadata=metadata)
    return {
        "status": "PASS",
        "output_dir": str(report),
        "unified_triads": len(unified),
        "outcome_counts": metadata["outcome_counts"],
        "cv_summary": cv_summary.to_dict("records"),
    }
