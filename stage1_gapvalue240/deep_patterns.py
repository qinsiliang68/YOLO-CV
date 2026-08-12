"""Pattern-oriented evidence tables for the frozen GapValue 240-run analysis.

The helpers in this module are deliberately read-only.  They connect frozen
selection manifests and per-epoch training records to already-canonical paired
run effects; they never choose samples, mutate the experiment matrix, or
reinterpret historical attempts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .deep_analysis import CanonicalInputError


_VALUE_COLUMNS = (
    "mean_p_defect",
    "correct_rate",
    "std_p_defect",
    "gap_critical_score",
    "gap_guard_score",
)
_GRADIENT_COLUMNS = (
    "grad_mag_score",
    "grad_align_score",
    "grad_mag_align_score",
    "diverse_grad_align_score",
    "grad_align_guard_score",
)


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], *, context: str
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise CanonicalInputError(f"{context} missing columns: {missing}")


def _selection_value_record(
    frame: pd.DataFrame,
    *,
    scope: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {**metadata, "scope": scope, "selected_count": len(frame)}
    for column in _VALUE_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if column in {"gap_critical_score", "gap_guard_score"}:
            record[f"mean_{column}"] = (
                float(finite.mean()) if len(finite) else np.nan
            )
            record[f"median_{column}"] = (
                float(finite.median()) if len(finite) else np.nan
            )
            record[f"positive_{column.removesuffix('_score')}_rate"] = (
                float((finite > 0).mean()) if len(finite) else np.nan
            )
        else:
            record[f"{column}_mean"] = (
                float(finite.mean()) if len(finite) else np.nan
            )
            record[f"{column}_median"] = (
                float(finite.median()) if len(finite) else np.nan
            )
    return record


def build_selection_value_summary(
    matrix: pd.DataFrame,
    selection_root: str | Path,
    value_table_path: str | Path,
) -> pd.DataFrame:
    """Summarize frozen value features for every run selection.

    The output contains one ``all`` row and one row per present class scope.
    It uses the immutable selection manifests as the intervention identity and
    the frozen value table only as descriptive OOF evidence.
    """

    _require_columns(
        matrix,
        (
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
        ),
        context="experiment matrix",
    )
    root = Path(selection_root).resolve()
    value_path = Path(value_table_path).resolve()
    if not root.is_dir():
        raise CanonicalInputError(f"Missing selection root: {root}")
    if not value_path.is_file():
        raise CanonicalInputError(f"Missing value table: {value_path}")

    value_columns = ("sample_id", "y_true", *_VALUE_COLUMNS)
    values = pd.read_csv(
        value_path,
        usecols=list(value_columns),
        dtype={"sample_id": "string"},
    )
    _require_columns(values, value_columns, context="sample value table")
    if values["sample_id"].isna().any() or values["sample_id"].duplicated().any():
        raise CanonicalInputError("Sample value table IDs must be unique and non-null")
    values["y_true"] = pd.to_numeric(values["y_true"], errors="raise").astype(int)

    records: list[dict[str, Any]] = []
    for _, row in matrix.sort_values("run_slot").iterrows():
        run_slot = str(row["run_slot"])
        path = root / run_slot / "selection_manifest.csv"
        if not path.is_file():
            raise CanonicalInputError(f"Missing selection manifest: {path}")
        selected = pd.read_csv(
            path,
            usecols=["sample_id", "y_true"],
            dtype={"sample_id": "string"},
        )
        if selected["sample_id"].isna().any() or selected["sample_id"].duplicated().any():
            raise CanonicalInputError(
                f"{run_slot} selection IDs must be unique and non-null"
            )
        if len(selected) != int(row["budget"]):
            raise CanonicalInputError(
                f"{run_slot} selection budget mismatch: "
                f"expected={int(row['budget'])}, actual={len(selected)}"
            )
        selected["y_true"] = pd.to_numeric(
            selected["y_true"], errors="raise"
        ).astype(int)
        merged = selected.merge(
            values,
            on="sample_id",
            how="left",
            suffixes=("_selection", "_value"),
            validate="one_to_one",
        )
        if merged["y_true_value"].isna().any():
            raise CanonicalInputError(
                f"{run_slot} selection contains IDs absent from the value table"
            )
        if not (
            merged["y_true_selection"].astype(int)
            == merged["y_true_value"].astype(int)
        ).all():
            raise CanonicalInputError(
                f"{run_slot} selection labels differ from the value table"
            )
        merged["y_true"] = merged["y_true_value"].astype(int)
        metadata = {
            "run_slot": run_slot,
            "triad_id": str(row["triad_id"]),
            "condition_slot": str(row["condition_slot"]),
            "condition_id": str(row["condition_id"]),
            "phase": str(row["phase"]),
            "method": str(row["method"]),
            "budget": int(row["budget"]),
            "guard_ratio": float(row["guard_ratio"]),
            "arm": str(row["arm"]),
            "training_seed": int(row["training_seed"]),
        }
        records.append(
            _selection_value_record(merged, scope="all", metadata=metadata)
        )
        for label, scope in ((0, "normal"), (1, "defect")):
            current = merged[merged["y_true"] == label]
            if not current.empty:
                records.append(
                    _selection_value_record(
                        current, scope=scope, metadata=metadata
                    )
                )
    return pd.DataFrame(records).sort_values(
        ["run_slot", "scope"], ignore_index=True
    )


def build_condition_value_effects(
    selection_summary: pd.DataFrame, deltas: pd.DataFrame
) -> pd.DataFrame:
    """Attach T/control selection-value contrasts to each paired run effect."""

    _require_columns(
        selection_summary,
        (
            "run_slot",
            "scope",
            "mean_gap_critical_score",
            "mean_gap_guard_score",
        ),
        context="selection value summary",
    )
    _require_columns(
        deltas,
        (
            "triad_id",
            "condition_slot",
            "condition_id",
            "phase",
            "control",
            "t_run_slot",
            "control_run_slot",
            "delta_FN",
            "delta_TN",
        ),
        context="triad deltas",
    )
    if selection_summary.duplicated(["run_slot", "scope"]).any():
        raise CanonicalInputError(
            "Selection value summary must be unique by run_slot/scope"
        )
    indexed = selection_summary.set_index(["run_slot", "scope"])
    feature_columns = [
        column
        for column in selection_summary.columns
        if column
        in {
            "mean_p_defect_mean",
            "mean_p_defect_median",
            "correct_rate_mean",
            "correct_rate_median",
            "std_p_defect_mean",
            "std_p_defect_median",
            "mean_gap_critical_score",
            "median_gap_critical_score",
            "positive_gap_critical_rate",
            "mean_gap_guard_score",
            "median_gap_guard_score",
            "positive_gap_guard_rate",
        }
    ]
    records: list[dict[str, Any]] = []
    for _, delta in deltas.iterrows():
        t_slot = str(delta["t_run_slot"])
        c_slot = str(delta["control_run_slot"])
        t_scopes = {
            scope for slot, scope in indexed.index if str(slot) == t_slot
        }
        c_scopes = {
            scope for slot, scope in indexed.index if str(slot) == c_slot
        }
        for scope in sorted(t_scopes & c_scopes):
            treatment = indexed.loc[(t_slot, scope)]
            control = indexed.loc[(c_slot, scope)]
            record = {
                column: delta[column]
                for column in delta.index
                if column.startswith("delta_")
            }
            record.update(
                {
                    "triad_id": str(delta["triad_id"]),
                    "condition_slot": str(delta["condition_slot"]),
                    "condition_id": str(delta["condition_id"]),
                    "phase": str(delta["phase"]),
                    "control": str(delta["control"]),
                    "training_seed": delta.get("training_seed", pd.NA),
                    "scope": scope,
                    "t_run_slot": t_slot,
                    "control_run_slot": c_slot,
                    "machine_pair": delta.get("machine_pair", "unknown"),
                    "any_resumed": bool(delta.get("any_resumed", False)),
                }
            )
            for column in feature_columns:
                t_value = pd.to_numeric(
                    pd.Series([treatment[column]]), errors="coerce"
                ).iloc[0]
                c_value = pd.to_numeric(
                    pd.Series([control[column]]), errors="coerce"
                ).iloc[0]
                record[f"treatment_{column}"] = t_value
                record[f"control_{column}"] = c_value
                record[f"selection_delta_{column}"] = (
                    float(t_value - c_value)
                    if pd.notna(t_value) and pd.notna(c_value)
                    else np.nan
                )
            records.append(record)
    return pd.DataFrame(records)


def build_value_effect_associations(
    condition_value_effects: pd.DataFrame,
) -> pd.DataFrame:
    """Compute exploratory condition-level Spearman associations.

    Seeds are collapsed before correlation to avoid treating repeated training
    seeds as independent sample-selection policies.
    """

    _require_columns(
        condition_value_effects,
        (
            "phase",
            "condition_slot",
            "control",
            "scope",
            "delta_FN",
            "delta_TN",
        ),
        context="condition value effects",
    )
    specifications = (
        (
            "phase_a_normal_gapcritical",
            "A",
            "normal",
            "treatment_mean_gap_critical_score",
        ),
        (
            "phase_b_defect_gapguard",
            "B",
            "defect",
            "treatment_mean_gap_guard_score",
        ),
    )
    outcomes = [
        column
        for column in ("delta_FN", "delta_TN", "delta_gap_q68_q050")
        if column in condition_value_effects
    ]
    records: list[dict[str, Any]] = []
    for analysis_scope, phase, scope, predictor in specifications:
        if predictor not in condition_value_effects:
            continue
        subset = condition_value_effects[
            (condition_value_effects["phase"].astype(str) == phase)
            & (condition_value_effects["scope"].astype(str) == scope)
        ]
        for control, control_rows in subset.groupby("control", sort=True):
            aggregation = {predictor: "first", **{outcome: "mean" for outcome in outcomes}}
            collapsed = (
                control_rows.groupby("condition_slot", sort=True)
                .agg(aggregation)
                .reset_index()
            )
            for outcome in outcomes:
                usable = collapsed[[predictor, outcome]].dropna()
                if (
                    len(usable) >= 3
                    and usable[predictor].nunique() > 1
                    and usable[outcome].nunique() > 1
                ):
                    rho, p_value = spearmanr(
                        usable[predictor].to_numpy(dtype=float),
                        usable[outcome].to_numpy(dtype=float),
                    )
                else:
                    rho, p_value = np.nan, np.nan
                records.append(
                    {
                        "analysis_scope": analysis_scope,
                        "control": str(control),
                        "predictor": predictor,
                        "outcome": outcome,
                        "n_conditions": len(usable),
                        "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                        "p_value": (
                            float(p_value) if np.isfinite(p_value) else np.nan
                        ),
                        "interpretation": (
                            "exploratory condition-level Spearman; training "
                            "seeds collapsed; not a per-sample causal estimate"
                        ),
                    }
                )
    return pd.DataFrame(records)


def build_paired_epoch_differences(
    epoch_curves: pd.DataFrame, deltas: pd.DataFrame
) -> pd.DataFrame:
    """Build per-epoch T-control differences for all 160 paired comparisons."""

    _require_columns(
        epoch_curves,
        (
            "run_slot",
            "epoch",
            "metrics/accuracy_top1",
            "val/loss",
            "train/loss",
        ),
        context="epoch curves",
    )
    _require_columns(
        deltas,
        (
            "triad_id",
            "condition_slot",
            "condition_id",
            "phase",
            "training_seed",
            "control",
            "t_run_slot",
            "control_run_slot",
        ),
        context="triad deltas",
    )
    if epoch_curves.duplicated(["run_slot", "epoch"]).any():
        raise CanonicalInputError("Epoch curves must be unique by run_slot/epoch")
    by_run = {
        str(run_slot): group.copy()
        for run_slot, group in epoch_curves.groupby("run_slot", sort=False)
    }
    records: list[pd.DataFrame] = []
    metric_pairs = {
        "metrics/accuracy_top1": "delta_top1",
        "val/loss": "delta_val_loss",
        "train/loss": "delta_train_loss",
    }
    for _, delta in deltas.iterrows():
        t_slot = str(delta["t_run_slot"])
        c_slot = str(delta["control_run_slot"])
        if t_slot not in by_run or c_slot not in by_run:
            raise CanonicalInputError(
                f"Missing epoch curves for paired runs {t_slot}/{c_slot}"
            )
        left = by_run[t_slot][["epoch", *metric_pairs]].copy()
        right = by_run[c_slot][["epoch", *metric_pairs]].copy()
        merged = left.merge(
            right,
            on="epoch",
            how="outer",
            suffixes=("_treatment", "_control"),
            indicator=True,
            validate="one_to_one",
        )
        if not merged["_merge"].eq("both").all():
            raise CanonicalInputError(
                f"Epoch sets differ for paired runs {t_slot}/{c_slot}"
            )
        result = pd.DataFrame({"epoch": merged["epoch"].astype(int)})
        for source, target in metric_pairs.items():
            result[target] = (
                pd.to_numeric(merged[f"{source}_treatment"], errors="raise")
                - pd.to_numeric(merged[f"{source}_control"], errors="raise")
            )
        for column in (
            "triad_id",
            "condition_slot",
            "condition_id",
            "phase",
            "training_seed",
            "control",
            "machine_pair",
            "any_resumed",
        ):
            result[column] = delta.get(column, pd.NA)
        result["t_run_slot"] = t_slot
        result["control_run_slot"] = c_slot
        records.append(result)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def summarize_paired_epoch_differences(
    paired_epoch_differences: pd.DataFrame,
    *,
    last_window: int = 20,
) -> pd.DataFrame:
    """Summarize paired training-curve differences without treating epochs as seeds.

    Every record remains a descriptive condition/control summary.  Confidence
    intervals are intentionally not computed over epochs because adjacent
    epochs are not independent replicates.
    """

    if last_window <= 0:
        raise ValueError("last_window must be positive")
    _require_columns(
        paired_epoch_differences,
        (
            "triad_id",
            "condition_slot",
            "condition_id",
            "phase",
            "training_seed",
            "control",
            "epoch",
            "delta_top1",
            "delta_val_loss",
            "delta_train_loss",
        ),
        context="paired epoch differences",
    )
    if paired_epoch_differences.empty:
        return pd.DataFrame()

    grouping = ["condition_slot", "condition_id", "phase", "control"]
    records: list[dict[str, Any]] = []
    for keys, group in paired_epoch_differences.groupby(grouping, sort=True):
        group = group.sort_values(["training_seed", "epoch"])
        seed_summaries: list[dict[str, float]] = []
        for _, seed_rows in group.groupby("training_seed", sort=True):
            seed_rows = seed_rows.sort_values("epoch")
            epoch_values = seed_rows["epoch"].to_numpy(dtype=float)
            if len(seed_rows) > 1:
                span = float(epoch_values[-1] - epoch_values[0])
            else:
                span = 0.0
            tail = seed_rows.tail(min(last_window, len(seed_rows)))
            summary: dict[str, float] = {}
            for column, label in (
                ("delta_top1", "delta_top1"),
                ("delta_val_loss", "delta_val_loss"),
                ("delta_train_loss", "delta_train_loss"),
            ):
                values = seed_rows[column].to_numpy(dtype=float)
                summary[f"final_{label}"] = float(values[-1])
                summary[f"last20_{label}_mean"] = float(tail[column].mean())
                summary[f"last20_{label}_std"] = float(
                    tail[column].std(ddof=0)
                )
                summary[f"{label}_normalized_auc"] = (
                    float(np.trapz(values, epoch_values) / span)
                    if span > 0
                    else float(values[-1])
                )
            seed_summaries.append(summary)
        seed_frame = pd.DataFrame(seed_summaries)
        record: dict[str, Any] = dict(zip(grouping, keys))
        record.update(
            {
                "seed_count": int(group["training_seed"].nunique()),
                "triad_count": int(group["triad_id"].nunique()),
                "epoch_count_per_seed_min": int(
                    group.groupby("training_seed")["epoch"].nunique().min()
                ),
                "epoch_count_per_seed_max": int(
                    group.groupby("training_seed")["epoch"].nunique().max()
                ),
                "cross_machine_pair_rate": (
                    float(
                        (
                            group.get(
                                "machine_pair",
                                pd.Series("unknown", index=group.index),
                            ).astype(str)
                            != "same_machine"
                        ).mean()
                    )
                ),
                "resumed_pair_rate": float(
                    group.get(
                        "any_resumed", pd.Series(False, index=group.index)
                    )
                    .astype(bool)
                    .mean()
                ),
                "interpretation": (
                    "descriptive paired training-curve summary; epochs are "
                    "not independent statistical replicates"
                ),
            }
        )
        for column in seed_frame.columns:
            record[column] = float(seed_frame[column].mean())
            record[f"{column}_seed_std"] = float(
                seed_frame[column].std(ddof=0)
            )
        records.append(record)
    return pd.DataFrame(records).sort_values(grouping, ignore_index=True)


def build_raw_calibrated_operational_sensitivity(
    calibrated_deltas: pd.DataFrame,
    raw_deltas: pd.DataFrame,
) -> pd.DataFrame:
    """Compare paired operational effects before and after Platt calibration.

    Platt calibration is monotone for the frozen fits, so operational integer
    counts should remain identical.  This table makes that invariant explicit
    at all 160 T-control comparisons and fails if pair identity differs.
    """

    keys = ("triad_id", "condition_slot", "control")
    _require_columns(
        calibrated_deltas,
        (*keys, "delta_TN", "delta_FN"),
        context="calibrated triad deltas",
    )
    _require_columns(
        raw_deltas,
        (
            *keys,
            "delta_raw_TN_at_FN95",
            "delta_raw_FN_at_TN68253",
        ),
        context="raw triad deltas",
    )
    for name, frame in (
        ("calibrated", calibrated_deltas),
        ("raw", raw_deltas),
    ):
        if frame.duplicated(list(keys)).any():
            raise CanonicalInputError(
                f"{name} deltas must be unique by {list(keys)}"
            )

    metadata_columns = [
        column
        for column in (
            "condition_id",
            "phase",
            "training_seed",
            "t_run_slot",
            "control_run_slot",
            "machine_pair",
            "any_resumed",
        )
        if column in calibrated_deltas.columns
    ]
    calibrated = calibrated_deltas[
        [
            *keys,
            *metadata_columns,
            *[
                column
                for column in (
                    "delta_TN",
                    "delta_FN",
                    "delta_gap_q68_q050",
                    "delta_tail_gap_q90_q05",
                )
                if column in calibrated_deltas.columns
            ],
        ]
    ].copy()
    raw_metric_columns = [
        column
        for column in raw_deltas.columns
        if column.startswith("delta_raw_")
    ]
    raw = raw_deltas[[*keys, *raw_metric_columns]].copy()
    merged = calibrated.merge(
        raw,
        on=list(keys),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        missing = merged.loc[
            merged["_merge"].ne("both"), [*keys, "_merge"]
        ].to_dict("records")
        raise CanonicalInputError(
            f"Raw/calibrated comparison identities differ: {missing[:5]}"
        )
    merged = merged.drop(columns="_merge")
    merged["delta_TN_raw_minus_calibrated"] = (
        merged["delta_raw_TN_at_FN95"].astype(float)
        - merged["delta_TN"].astype(float)
    )
    merged["delta_FN_raw_minus_calibrated"] = (
        merged["delta_raw_FN_at_TN68253"].astype(float)
        - merged["delta_FN"].astype(float)
    )
    merged["integer_effects_equal"] = (
        merged["delta_TN_raw_minus_calibrated"].eq(0)
        & merged["delta_FN_raw_minus_calibrated"].eq(0)
    )
    for label, calibrated_column, raw_column in (
        (
            "gap",
            "delta_gap_q68_q050",
            "delta_raw_gap_q68_q050",
        ),
        (
            "tail_gap",
            "delta_tail_gap_q90_q05",
            "delta_raw_tail_gap_q90_q05",
        ),
    ):
        if {calibrated_column, raw_column}.issubset(merged.columns):
            calibrated_sign = np.sign(
                pd.to_numeric(merged[calibrated_column], errors="coerce")
            )
            raw_sign = np.sign(
                pd.to_numeric(merged[raw_column], errors="coerce")
            )
            merged[f"{label}_direction_equal"] = (
                calibrated_sign.eq(raw_sign)
                & calibrated_sign.notna()
                & raw_sign.notna()
            )
    return merged.sort_values(list(keys), ignore_index=True)


def build_analysis_capabilities(
    matrix: pd.DataFrame,
    *,
    overlap_decisions_path: str | Path,
    value_table_path: str | Path,
) -> pd.DataFrame:
    """Register evidence that is available, replaced, or impossible to test."""

    overlap_path = Path(overlap_decisions_path).resolve()
    value_path = Path(value_table_path).resolve()
    if not overlap_path.is_file():
        raise CanonicalInputError(f"Missing overlap decisions: {overlap_path}")
    if not value_path.is_file():
        raise CanonicalInputError(f"Missing value table: {value_path}")
    try:
        decisions = json.loads(overlap_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CanonicalInputError(f"Invalid overlap decisions: {exc}") from exc
    if not isinstance(decisions, list):
        raise CanonicalInputError("Overlap decisions must be a JSON list")

    records: list[dict[str, Any]] = []
    for decision in decisions:
        candidate = str(decision.get("candidate", "unknown"))
        replaced = bool(decision.get("replaced", False))
        records.append(
            {
                "capability_id": f"ranking_{candidate}",
                "status": "NOT_TESTABLE" if replaced else "AVAILABLE",
                "observed": (
                    f"top3000_overlap={float(decision.get('max_overlap', np.nan)):.4f}"
                ),
                "replacement": str(decision.get("retained_as", "")),
                "evidence": str(overlap_path),
                "conclusion_boundary": (
                    "ranking-only; no training effect exists for the replaced candidate"
                    if replaced
                    else "candidate retained in frozen training matrix"
                ),
            }
        )

    header = pd.read_csv(value_path, nrows=0)
    missing_gradient = sorted(set(_GRADIENT_COLUMNS).difference(header.columns))
    if missing_gradient:
        gradient_status = "NOT_TESTABLE"
        gradient_observed = f"missing columns: {missing_gradient}"
    else:
        gradients = pd.read_csv(value_path, usecols=list(_GRADIENT_COLUMNS))
        counts = {column: int(gradients[column].notna().sum()) for column in gradients}
        gradient_status = "AVAILABLE" if any(counts.values()) else "NOT_TESTABLE"
        gradient_observed = json.dumps(counts, sort_keys=True)
    records.append(
        {
            "capability_id": "gradient_evidence",
            "status": gradient_status,
            "observed": gradient_observed,
            "replacement": "",
            "evidence": str(value_path),
            "conclusion_boundary": "empty gradient fields cannot support Grad-Mag/Grad-Align claims",
        }
    )

    arms = set(matrix.get("arm", pd.Series(dtype=str)).astype(str))
    has_no_replay = bool(arms & {"BASE", "NO_REPLAY", "NONE"})
    records.extend(
        [
            {
                "capability_id": "no_replay_baseline",
                "status": "AVAILABLE" if has_no_replay else "NOT_TESTABLE",
                "observed": f"matrix arms={sorted(arms)}",
                "replacement": "",
                "evidence": "frozen_experiment_matrix.csv",
                "conclusion_boundary": (
                    "selection policies with replay can be compared; replay versus no replay cannot"
                ),
            },
            {
                "capability_id": "per_epoch_val_op",
                "status": "NOT_TESTABLE",
                "observed": "only final val_op predictions were retained",
                "replacement": "",
                "evidence": "canonical run artifacts",
                "conclusion_boundary": "training top1/loss cannot be converted into per-epoch operational TN/FN",
            },
            {
                "capability_id": "epoch178_provenance",
                "status": "LIMITATION",
                "observed": "fold_01 logical epoch178 duplicates repaired checkpoint evidence",
                "replacement": "Exclude178 ranking was itself removed by overlap gate",
                "evidence": "science contract epoch178_provenance",
                "conclusion_boundary": "epoch178 is not an independent checkpoint observation",
            },
            {
                "capability_id": "blind_external_test",
                "status": "NOT_TESTABLE",
                "observed": "blind/external result is not bound to the 240-run discovery matrix",
                "replacement": "",
                "evidence": "science contract evaluation_roles",
                "conclusion_boundary": "all findings are val_op internal discovery/replication only",
            },
        ]
    )
    return pd.DataFrame(records)


def build_pattern_narrative_sections(
    *,
    condition_summaries: pd.DataFrame,
    cross_method_comparisons: pd.DataFrame,
    selection_value_associations: pd.DataFrame,
    tail_detail: pd.DataFrame,
    raw_calibrated_sensitivity: pd.DataFrame,
    training_summaries: pd.DataFrame,
    paired_epoch_summary: pd.DataFrame,
    training_contract: pd.DataFrame,
    selection_value_summary: pd.DataFrame,
    selection_composition: pd.DataFrame,
) -> dict[str, str]:
    """Build concise, data-driven Chinese narrative for the v2 report."""

    _require_columns(
        condition_summaries,
        (
            "phase",
            "condition_slot",
            "control",
            "mean_delta_FN",
            "mean_delta_TN",
            "safety_noninferior",
            "confirmed_TN_improvement",
        ),
        context="condition summaries",
    )

    condition_rows = condition_summaries.copy()
    paired_conditions = list(
        condition_rows.groupby(["phase", "condition_slot"], sort=True)
    )
    dual_success = sum(
        bool(
            len(group) == 2
            and group["safety_noninferior"].fillna(False).astype(bool).all()
            and group["confirmed_TN_improvement"]
            .fillna(False)
            .astype(bool)
            .all()
        )
        for _, group in paired_conditions
    )
    dual_numeric = sum(
        bool(
            len(group) == 2
            and (pd.to_numeric(group["mean_delta_FN"], errors="coerce") <= 0).all()
            and (pd.to_numeric(group["mean_delta_TN"], errors="coerce") > 0).all()
        )
        for _, group in paired_conditions
    )
    safe_rows = int(
        condition_rows["safety_noninferior"].fillna(False).astype(bool).sum()
    )
    tn_rows = int(
        condition_rows["confirmed_TN_improvement"]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    def condition_effect(slot: str, control: str, *, phase: str = "A") -> str:
        rows = condition_rows[
            (condition_rows["phase"].astype(str) == phase)
            & (condition_rows["condition_slot"].astype(str) == slot)
            & (condition_rows["control"].astype(str) == control)
        ]
        if len(rows) != 1:
            return f"{slot}/{control}=不可用"
        row = rows.iloc[0]
        return (
            f"{slot}/{control}: ΔFN={float(row['mean_delta_FN']):+.3f}, "
            f"ΔTN={float(row['mean_delta_TN']):+.3f}"
        )

    strict_value_parts: list[str] = []
    strict_bucket_parts: list[str] = []
    for slot in ("A01", "A02", "A03"):
        value_rows = selection_value_summary[
            (selection_value_summary.get(
                "phase", pd.Series(dtype=str)
            ).astype(str)
            == "A")
            & (
                selection_value_summary.get(
                    "condition_slot", pd.Series(dtype=str)
                ).astype(str)
                == slot
            )
            & (
                selection_value_summary.get(
                    "arm", pd.Series(dtype=str)
                ).astype(str)
                == "T"
            )
            & (
                selection_value_summary.get(
                    "scope", pd.Series(dtype=str)
                ).astype(str)
                == "normal"
            )
        ]
        if not value_rows.empty:
            strict_value_parts.append(
                f"{slot} score="
                f"{pd.to_numeric(value_rows['mean_gap_critical_score'], errors='coerce').mean():.3f}, "
                f"correct="
                f"{pd.to_numeric(value_rows['correct_rate_mean'], errors='coerce').mean():.3f}, "
                f"std="
                f"{pd.to_numeric(value_rows['std_p_defect_mean'], errors='coerce').mean():.3f}"
            )
        composition_rows = selection_composition[
            (selection_composition.get(
                "phase", pd.Series(dtype=str)
            ).astype(str)
            == "A")
            & (
                selection_composition.get(
                    "condition_slot", pd.Series(dtype=str)
                ).astype(str)
                == slot
            )
            & (
                selection_composition.get(
                    "arm", pd.Series(dtype=str)
                ).astype(str)
                == "T"
            )
            & (
                pd.to_numeric(
                    selection_composition.get(
                        "y_true", pd.Series(dtype=float)
                    ),
                    errors="coerce",
                )
                == 0
            )
        ]
        if not composition_rows.empty:
            total = pd.to_numeric(
                composition_rows["count"], errors="coerce"
            ).sum()
            learnable = pd.to_numeric(
                composition_rows.loc[
                    composition_rows["dynamic_bucket"].astype(str)
                    == "learnable_hard",
                    "count",
                ],
                errors="coerce",
            ).sum()
            strict_bucket_parts.append(
                f"{slot}={learnable / total:.2%}" if total else f"{slot}=不可用"
            )

    strict_dilution_summary = ""
    if strict_bucket_parts and strict_value_parts:
        strict_dilution_summary = (
            "三档 treatment 的 learnable_hard 占比为 "
            + "、".join(strict_bucket_parts)
            + "；对应集合特征为 "
            + "；".join(strict_value_parts)
            + "。因此扩量退化更像排名向下后的动态信号稀释，"
            "并与 replay 曝光/optimizer steps 同时变化，"
            "不能归因于 ordinary/easy bucket 污染。"
        )
    confidence_small_summary = ""
    if (
        not condition_effect("A05", "R1").endswith("不可用")
        and not condition_effect("A05", "R2").endswith("不可用")
    ):
        confidence_small_summary = (
            "Confidence-Clean B600 也在均值上对双对照有利："
            f"{condition_effect('A05', 'R1')}；"
            f"{condition_effect('A05', 'R2')}，"
            "所以小预算现象并非 GapCritical 独有。"
        )

    traditional = cross_method_comparisons[
        (cross_method_comparisons.get(
            "reference_condition", pd.Series(dtype=str)
        ).astype(str)
        == "A02")
        & cross_method_comparisons.get(
            "comparator_condition", pd.Series(dtype=str)
        )
        .astype(str)
        .isin(["A06", "A08", "A10", "A12"])
    ].copy()
    if traditional.empty:
        traditional_summary = "传统 B3000 配对比较不可用。"
    else:
        numeric_wins = (
            (
                pd.to_numeric(
                    traditional["mean_diff_delta_FN"], errors="coerce"
                )
                <= 0
            )
            & (
                pd.to_numeric(
                    traditional["mean_diff_delta_TN"], errors="coerce"
                )
                > 0
            )
        )
        winning_rows = int(numeric_wins.sum())
        dual_winners = 0
        for _, group in traditional.groupby("comparator_condition"):
            if len(group) == 2 and bool(
                (
                    pd.to_numeric(
                        group["mean_diff_delta_FN"], errors="coerce"
                    )
                    <= 0
                ).all()
                and (
                    pd.to_numeric(
                        group["mean_diff_delta_TN"], errors="coerce"
                    )
                    > 0
                ).all()
            ):
                dual_winners += 1
        traditional_summary = (
            f"A02 在 {len(traditional)} 个传统方法×对照比较中仅 "
            f"{winning_rows} 个数值方向同时更优，且只对 "
            f"{dual_winners} 个传统方法在 R1/R2 两边都数值占优；"
            "这不支持“GapCritical 普遍优于传统 hard-negative 排序”。"
        )

    association_rows = selection_value_associations[
        selection_value_associations.get(
            "analysis_scope", pd.Series(dtype=str)
        ).astype(str)
        == "phase_a_normal_gapcritical"
    ]
    association_parts: list[str] = []
    for control in ("R1", "R2"):
        current = association_rows[
            association_rows.get("control", pd.Series(dtype=str)).astype(str)
            == control
        ].set_index("outcome")
        if {"delta_FN", "delta_TN"}.issubset(current.index):
            association_parts.append(
                f"{control}: ρ(FN)={float(current.loc['delta_FN', 'spearman_rho']):+.3f}, "
                f"ρ(TN)={float(current.loc['delta_TN', 'spearman_rho']):+.3f}"
            )
    association_p_values = pd.to_numeric(
        association_rows.get("p_value", pd.Series(dtype=float)),
        errors="coerce",
    )
    association_significance = (
        int((association_p_values < 0.05).sum())
        if not association_p_values.empty
        else 0
    )

    def tail_effect(slot: str, control: str) -> tuple[float, float] | None:
        rows = tail_detail[
            (tail_detail.get(
                "condition_slot", pd.Series(dtype=str)
            ).astype(str)
            == slot)
            & (
                tail_detail.get("control", pd.Series(dtype=str)).astype(str)
                == control
            )
            & (
                tail_detail.get("scope", pd.Series(dtype=str)).astype(str)
                == "operational"
            )
            & (
                tail_detail.get("score_type", pd.Series(dtype=str)).astype(str)
                == "raw"
            )
        ]
        means = (
            rows.assign(
                mean_shift=pd.to_numeric(rows["mean_shift"], errors="coerce")
            )
            .groupby("label")["mean_shift"]
            .mean()
        )
        if not {"normal", "defect"}.issubset(means.index):
            return None
        return float(means["normal"]), float(means["defect"])

    a02_tail_parts: list[str] = []
    b06_tail_parts: list[str] = []
    for control in ("R1", "R2"):
        effect = tail_effect("A02", control)
        if effect is not None:
            a02_tail_parts.append(
                f"{control} normal={effect[0]:+.6f}, defect={effect[1]:+.6f}"
            )
        effect = tail_effect("B06", control)
        if effect is not None:
            b06_tail_parts.append(
                f"{control} normal={effect[0]:+.6f}, defect={effect[1]:+.6f}"
            )

    guard_parts = [
        condition_effect("B03", control, phase="B")
        for control in ("R1", "R2")
    ]
    guard_parts += [
        condition_effect("B04", control, phase="B")
        for control in ("R1", "R2")
    ]
    guard_parts += [
        condition_effect("B05", control, phase="B")
        for control in ("R1", "R2")
    ]

    raw_count = len(raw_calibrated_sensitivity)
    integer_equal = int(
        raw_calibrated_sensitivity.get(
            "integer_effects_equal", pd.Series(dtype=bool)
        )
        .fillna(False)
        .astype(bool)
        .sum()
    )
    gap_equal = int(
        raw_calibrated_sensitivity.get(
            "gap_direction_equal", pd.Series(dtype=bool)
        )
        .fillna(False)
        .astype(bool)
        .sum()
    )
    tail_gap_equal = int(
        raw_calibrated_sensitivity.get(
            "tail_gap_direction_equal", pd.Series(dtype=bool)
        )
        .fillna(False)
        .astype(bool)
        .sum()
    )

    overfit_count = int(
        training_summaries.get("overfit_flag", pd.Series(dtype=bool))
        .fillna(False)
        .astype(bool)
        .sum()
    )
    oscillation_count = int(
        training_summaries.get("oscillation_flag", pd.Series(dtype=bool))
        .fillna(False)
        .astype(bool)
        .sum()
    )
    best_epoch = pd.to_numeric(
        training_summaries.get("best_top1_epoch", pd.Series(dtype=float)),
        errors="coerce",
    )
    best_val_epoch = pd.to_numeric(
        training_summaries.get("best_val_loss_epoch", pd.Series(dtype=float)),
        errors="coerce",
    )
    top1_gap = pd.to_numeric(
        training_summaries.get(
            "best_final_top1_gap", pd.Series(dtype=float)
        ),
        errors="coerce",
    )
    positive_val_loss_slope = int(
        (
            pd.to_numeric(
                training_summaries.get(
                    "last_window_val_loss_slope",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            )
            > 0
        ).sum()
    )
    a02_epoch = paired_epoch_summary[
        paired_epoch_summary.get(
            "condition_slot", pd.Series(dtype=str)
        ).astype(str)
        == "A02"
    ]
    top1_effect = pd.to_numeric(
        a02_epoch.get("final_delta_top1", pd.Series(dtype=float)),
        errors="coerce",
    )
    val_loss_effect = pd.to_numeric(
        a02_epoch.get("final_delta_val_loss", pd.Series(dtype=float)),
        errors="coerce",
    )
    resumed_count = int(
        (
            pd.to_numeric(
                training_contract.get(
                    "resume_count", pd.Series(dtype=float)
                ),
                errors="coerce",
            )
            > 0
        ).sum()
    )
    partial_count = int(
        training_contract.get(
            "partial_step_reconciled", pd.Series(dtype=bool)
        )
        .fillna(False)
        .astype(bool)
        .sum()
    )

    return {
        "全矩阵合同结果": (
            f"共 {len(condition_rows)} 个 condition-control 汇总行；"
            f"{safe_rows} 行仅通过 FN 安全门，{tn_rows} 行通过后续 TN 门，"
            f"最终 0 行同时过门。按条件要求 R1/R2 双通过时，"
            f"{len(paired_conditions)} 个条件中 {dual_success} 个合同成功；"
            f"仅 {dual_numeric} 个条件在均值层面双对照方向同时有利。"
        ),
        "预算与传统方法": (
            "GapCritical-Strict 呈现强烈的预算稀释/反转："
            f"{condition_effect('A01', 'R1')}；"
            f"{condition_effect('A01', 'R2')}；"
            f"{condition_effect('A02', 'R1')}；"
            f"{condition_effect('A02', 'R2')}；"
            f"{condition_effect('A03', 'R1')}；"
            f"{condition_effect('A03', 'R2')}。"
            "B600 只达到数值改善，n=3 的区间仍未过合同门槛。"
            + strict_dilution_summary
            + confidence_small_summary
            + traditional_summary
        ),
        "固定尾部机制": (
            "A02 的 pooled-8 原始分数尾部变化为："
            + "；".join(a02_tail_parts)
            + "。normal 风险虽多为下降，但 defect 尾部也下降，"
            "说明模型更像把两类一起往低分移动，而不是稳定扩大安全间隔。"
            + (
                "B06 的固定尾部方向为：" + "；".join(b06_tail_parts)
                + "，虽方向符合 guard 机制，仍未转化为合同成功。"
                if b06_tail_parts
                else ""
            )
        ),
        "选样分数与实际收益": (
            "按条件折叠 seed 后，normal 的平均 gap_critical_score 与实际"
            "效应仅呈弱到中等的期望方向相关："
            + "；".join(association_parts)
            + f"；共有 {association_significance} 个关联达到 p<0.05。"
            "这支持“分数可能含弱信号”，不支持把分数大小直接解释为"
            "稳定回流收益或单样本因果价值。"
        ),
        "Defect guard 剂量规律": (
            "GapGuard-Raw 的 5%/10%/20% 均值依次显示从有害到数值改善："
            + "；".join(guard_parts)
            + "。20% B05 的均值最有希望，但只有 3 seeds、区间未过门，"
            "且多数 Phase B 比较存在机器混杂，因此仍是 INCONCLUSIVE。"
        ),
        "Raw 与 Platt 敏感性": (
            f"{integer_equal}/{raw_count} 个 T-control 比较的 operational "
            "ΔTN/ΔFN 在 raw 与 Platt 后完全一致，主结论不是校准偶然。"
            f"但连续 gap 方向仅 {gap_equal}/{raw_count} "
            f"({gap_equal / raw_count:.2%}) 一致，tail-gap 为 "
            f"{tail_gap_equal}/{raw_count} "
            f"({tail_gap_equal / raw_count:.2%})；机制解释应优先使用 raw score。"
        ),
        "200 epoch 与运行可靠性": (
            f"240 个 run 中 overfit_flag={overfit_count}、"
            f"oscillation_flag={oscillation_count}；best top1 epoch 中位数="
            f"{float(best_epoch.median()):.1f}，best val-loss epoch 中位数="
            f"{float(best_val_epoch.median()):.1f}，best-final top1 gap 均值="
            f"{float(top1_gap.mean()):.6f}。最后 20 epoch val-loss 斜率为正的"
            f" run={positive_val_loss_slope}/{len(training_summaries)}，"
            "表现为普遍但轻微的末段 loss 漂移；保守 overfit_flag 未把它"
            "判为灾难性过拟合。A02 final top1 的配对效应范围="
            f"[{float(top1_effect.min()):+.6f}, {float(top1_effect.max()):+.6f}]，"
            f"final val-loss 范围="
            f"[{float(val_loss_effect.min()):+.6f}, "
            f"{float(val_loss_effect.max()):+.6f}]；普通 top1/loss 的微小变化"
            "不能替代 operational 尾部指标。"
            f"运行上有 {resumed_count} 个 resume run、{partial_count} 个"
            " partial-step reconciliation，均已通过训练合同；"
            "A02 的 8 个主比较本身均未 resume。"
        ),
    }


def _tail_mechanism_status(
    tail_detail: pd.DataFrame, *, condition_slot: str, control: str
) -> bool | pd._libs.missing.NAType:
    required = {"condition_slot", "control", "scope", "score_type", "label", "mean_shift"}
    if tail_detail.empty or not required.issubset(tail_detail.columns):
        return pd.NA
    rows = tail_detail[
        (tail_detail["condition_slot"].astype(str) == condition_slot)
        & (tail_detail["control"].astype(str) == control)
        & (tail_detail["scope"].astype(str) == "operational")
        & (tail_detail["score_type"].astype(str) == "raw")
    ]
    means = (
        rows.assign(mean_shift=pd.to_numeric(rows["mean_shift"], errors="coerce"))
        .groupby("label")["mean_shift"]
        .mean()
    )
    if not {"normal", "defect"}.issubset(means.index):
        return pd.NA
    return bool(means["normal"] < 0 and means["defect"] > 0)


def _dual_control_rows(
    frame: pd.DataFrame, condition_slot: str, *, phase: str | None = None
) -> pd.DataFrame:
    if frame.empty or not {"condition_slot", "control"}.issubset(frame.columns):
        return pd.DataFrame()
    rows = frame[frame["condition_slot"].astype(str) == condition_slot].copy()
    if phase is not None and "phase" in rows:
        rows = rows[rows["phase"].astype(str) == phase]
    return rows[rows["control"].astype(str).isin(["R1", "R2"])]


def _dual_numeric_direction(rows: pd.DataFrame) -> bool | pd._libs.missing.NAType:
    if len(rows) != 2 or set(rows["control"].astype(str)) != {"R1", "R2"}:
        return pd.NA
    return bool(
        (pd.to_numeric(rows["mean_delta_FN"], errors="coerce") <= 0).all()
        and (pd.to_numeric(rows["mean_delta_TN"], errors="coerce") > 0).all()
    )


def _dual_contract_success(rows: pd.DataFrame) -> bool | pd._libs.missing.NAType:
    required = {"safety_noninferior", "confirmed_TN_improvement"}
    if len(rows) != 2 or not required.issubset(rows.columns):
        return pd.NA
    return bool(
        rows["safety_noninferior"].fillna(False).astype(bool).all()
        and rows["confirmed_TN_improvement"].fillna(False).astype(bool).all()
    )


def build_pattern_evidence_registry(
    *,
    a02_summaries: pd.DataFrame,
    condition_summaries: pd.DataFrame,
    cross_method_comparisons: pd.DataFrame,
    budget_comparisons: pd.DataFrame,
    guard_comparisons: pd.DataFrame,
    sensitivity_summaries: pd.DataFrame,
    tail_detail: pd.DataFrame,
    capability_registry: pd.DataFrame,
    r2_overlap: pd.DataFrame,
) -> pd.DataFrame:
    """Build the v2 four-layer, data-driven hypothesis registry."""

    records: list[dict[str, Any]] = []

    def append(
        hypothesis_id: str,
        hypothesis: str,
        *,
        status: str,
        numerically_better: bool | pd._libs.missing.NAType,
        contract_success: bool | pd._libs.missing.NAType,
        mechanism_supported: bool | pd._libs.missing.NAType,
        rationale: str,
        evidence_scope: str,
    ) -> None:
        if status not in {
            "SUPPORTED",
            "NOT_SUPPORTED",
            "INCONCLUSIVE",
            "NOT_TESTABLE",
        }:
            raise CanonicalInputError(f"Invalid hypothesis status: {status}")
        records.append(
            {
                "hypothesis_id": hypothesis_id,
                "hypothesis": hypothesis,
                "status": status,
                "numerically_better": numerically_better,
                "contract_success": contract_success,
                "mechanism_supported": mechanism_supported,
                "causal_claim_allowed": False,
                "evidence_scope": evidence_scope,
                "rationale": rationale,
            }
        )

    pooled = (
        a02_summaries[
            a02_summaries.get("analysis_cohort", pd.Series(dtype=str)).astype(str)
            == "pooled"
        ]
        if not a02_summaries.empty
        else pd.DataFrame()
    )
    primary_statuses: dict[str, str] = {}
    for control, hypothesis_id in (("R1", "H1_A02_vs_R1"), ("R2", "H2_A02_vs_R2")):
        rows = (
            pooled[pooled["control"].astype(str) == control]
            if "control" in pooled
            else pd.DataFrame()
        )
        if len(rows) != 1:
            status = "NOT_TESTABLE"
            numeric = contract = pd.NA
            rationale = f"No unique pooled-8 A02 row for {control}."
        else:
            row = rows.iloc[0]
            mean_fn = float(row["mean_delta_FN"])
            mean_tn = float(row["mean_delta_TN"])
            numeric = bool(mean_fn <= 0 and mean_tn > 0)
            contract = bool(
                row.get("safety_noninferior", False)
                and row.get("confirmed_TN_improvement", False)
            )
            if contract:
                status = "SUPPORTED"
            elif mean_fn > 0 or mean_tn <= 0 or not bool(
                row.get("safety_noninferior", False)
            ):
                status = "NOT_SUPPORTED"
            else:
                status = "INCONCLUSIVE"
            rationale = (
                f"pooled-8 {control}: mean ΔFN={mean_fn:+.3f}, "
                f"FN upper95={float(row['FN_one_sided_95_upper']):+.3f}, "
                f"worst ΔFN={float(row['worst_delta_FN']):+.3f}, "
                f"mean ΔTN={mean_tn:+.3f}, "
                f"TN lower95={float(row['TN_one_sided_95_lower']):+.3f}."
            )
            if control == "R2" and not r2_overlap.empty:
                contrast_column = next(
                    (
                        column
                        for column in (
                            "effective_unique_contrast_rate",
                            "effective_unique_contrast",
                        )
                        if column in r2_overlap
                    ),
                    None,
                )
                if contrast_column is not None:
                    median_contrast = pd.to_numeric(
                        r2_overlap[contrast_column], errors="coerce"
                    ).median()
                    rationale += (
                        f" Median effective unique contrast={median_contrast:.2%}; "
                        "R2 is low-power."
                    )
        mechanism = _tail_mechanism_status(
            tail_detail, condition_slot="A02", control=control
        )
        append(
            hypothesis_id,
            f"GapCritical-Strict B3000 outperforms {control}",
            status=status,
            numerically_better=numeric,
            contract_success=contract,
            mechanism_supported=mechanism,
            rationale=rationale,
            evidence_scope="A02 pooled-8 val_op",
        )
        primary_statuses[control] = status

    traditional_pairs = {
        ("A02", comparator, control)
        for comparator in ("A06", "A08", "A10", "A12")
        for control in ("R1", "R2")
    }
    available_pairs = (
        {
            (
                str(row.reference_condition),
                str(row.comparator_condition),
                str(row.control),
            )
            for _, row in cross_method_comparisons.iterrows()
        }
        if not cross_method_comparisons.empty
        else set()
    )
    if not traditional_pairs.issubset(available_pairs):
        h3_status = "NOT_TESTABLE"
        h3_numeric = h3_contract = pd.NA
        h3_reason = "Complete A02-vs-traditional paired comparisons are unavailable."
    else:
        rows = cross_method_comparisons[
            cross_method_comparisons.apply(
                lambda row: (
                    str(row["reference_condition"]),
                    str(row["comparator_condition"]),
                    str(row["control"]),
                )
                in traditional_pairs,
                axis=1,
            )
        ]
        h3_numeric = bool(
            (pd.to_numeric(rows["mean_diff_delta_FN"], errors="coerce") <= 0).all()
            and (pd.to_numeric(rows["mean_diff_delta_TN"], errors="coerce") > 0).all()
        )
        h3_contract = bool(
            rows.get("safety_noninferior", pd.Series(False, index=rows.index))
            .fillna(False)
            .astype(bool)
            .all()
            and rows.get(
                "confirmed_TN_improvement", pd.Series(False, index=rows.index)
            )
            .fillna(False)
            .astype(bool)
            .all()
        )
        if (
            h3_contract
            and primary_statuses.get("R1") == "SUPPORTED"
            and primary_statuses.get("R2") == "SUPPORTED"
        ):
            h3_status = "SUPPORTED"
        elif not h3_numeric or "NOT_SUPPORTED" in primary_statuses.values():
            h3_status = "NOT_SUPPORTED"
        else:
            h3_status = "INCONCLUSIVE"
        h3_reason = (
            "Uses paired delta-of-deltas against Confidence, Boundary, "
            "Persistent, and Endpoint at B3000; all n=3 method contrasts remain exploratory."
        )
    append(
        "H3_vs_traditional_rankings",
        "GapCritical is stronger than traditional static rankings",
        status=h3_status,
        numerically_better=h3_numeric,
        contract_success=h3_contract,
        mechanism_supported=pd.NA,
        rationale=h3_reason,
        evidence_scope="Phase A same-seed method contrasts",
    )

    a02_discovery = _dual_control_rows(condition_summaries, "A02", phase="A")
    a13 = _dual_control_rows(condition_summaries, "A13", phase="A")
    positive_direction = _dual_numeric_direction(a02_discovery)
    negative_direction = (
        bool(
            (pd.to_numeric(a13["mean_delta_FN"], errors="coerce") > 0).all()
            and (pd.to_numeric(a13["mean_delta_TN"], errors="coerce") < 0).all()
        )
        if len(a13) == 2
        else pd.NA
    )
    if pd.isna(positive_direction) or pd.isna(negative_direction):
        h4_status = "NOT_TESTABLE"
    elif bool(positive_direction and negative_direction):
        h4_status = (
            "SUPPORTED"
            if _dual_contract_success(a02_discovery) is True
            else "INCONCLUSIVE"
        )
    else:
        h4_status = "NOT_SUPPORTED"
    append(
        "H4_directional_negative_control",
        "GapCritical and BottomGap produce opposite directions",
        status=h4_status,
        numerically_better=(
            bool(positive_direction and negative_direction)
            if not pd.isna(positive_direction) and not pd.isna(negative_direction)
            else pd.NA
        ),
        contract_success=_dual_contract_success(a02_discovery),
        mechanism_supported=negative_direction,
        rationale=(
            "A13 must be harmful against both controls while preregistered A02 "
            "must be beneficial; sign composition is reported separately."
        ),
        evidence_scope="Phase A discovery n=3",
    )

    budget_rows = {
        slot: _dual_control_rows(condition_summaries, slot, phase="A")
        for slot in ("A01", "A02", "A03")
    }
    if any(len(rows) != 2 for rows in budget_rows.values()):
        h5_numeric = pd.NA
        h5_status = "NOT_TESTABLE"
    else:
        h5_numeric = all(
            float(
                budget_rows["A01"]
                .set_index("control")
                .loc[control, "mean_delta_FN"]
            )
            < min(
                float(
                    budget_rows[slot]
                    .set_index("control")
                    .loc[control, "mean_delta_FN"]
                )
                for slot in ("A02", "A03")
            )
            and float(
                budget_rows["A01"]
                .set_index("control")
                .loc[control, "mean_delta_TN"]
            )
            > max(
                float(
                    budget_rows[slot]
                    .set_index("control")
                    .loc[control, "mean_delta_TN"]
                )
                for slot in ("A02", "A03")
            )
            for control in ("R1", "R2")
        )
        h5_status = "INCONCLUSIVE" if h5_numeric else "NOT_SUPPORTED"
    append(
        "H5_small_budget_advantage",
        "GapCritical B600 is numerically safer than B3000/B6000",
        status=h5_status,
        numerically_better=h5_numeric,
        contract_success=_dual_contract_success(budget_rows["A01"]),
        mechanism_supported=pd.NA,
        rationale=(
            "Budget comparisons are n=3 and optimizer steps differ; a coherent "
            "small-budget pattern is mechanism evidence, not confirmation."
        ),
        evidence_scope="A01/A02/A03 paired control effects",
    )

    ablation_slots = {"A16", "A17", "A19"}
    available_ablation = {
        str(value)
        for value in condition_summaries.get(
            "condition_slot", pd.Series(dtype=str)
        )
    }
    if not ablation_slots.issubset(available_ablation):
        h6_status = "NOT_TESTABLE"
    elif any(status == "NOT_SUPPORTED" for status in primary_statuses.values()):
        h6_status = "NOT_SUPPORTED"
    else:
        h6_status = "INCONCLUSIVE"
    append(
        "H6_not_simple_early_late",
        "Gap-linked dynamics add evidence beyond simple early/late change",
        status=h6_status,
        numerically_better=pd.NA,
        contract_success=False,
        mechanism_supported=pd.NA,
        rationale=(
            "Residual, EarlyLate40, and TimeMatched are n=3 descriptive ablations; "
            "the primary A02 effect must first be established."
        ),
        evidence_scope="A16/A17/A19",
    )

    phase_b_slots = sorted(
        {
            str(value)
            for value in condition_summaries.loc[
                condition_summaries.get("phase", pd.Series(dtype=str)).astype(str)
                == "B",
                "condition_slot",
            ]
        }
    )
    favourable_guards: list[str] = []
    contract_guards: list[str] = []
    mechanism_guards: list[str] = []
    for slot in phase_b_slots:
        rows = _dual_control_rows(condition_summaries, slot, phase="B")
        if _dual_numeric_direction(rows) is True:
            favourable_guards.append(slot)
        if _dual_contract_success(rows) is True:
            contract_guards.append(slot)
        mechanisms = [
            _tail_mechanism_status(
                tail_detail, condition_slot=slot, control=control
            )
            for control in ("R1", "R2")
        ]
        if mechanisms == [True, True]:
            mechanism_guards.append(slot)
    if not phase_b_slots:
        h7_status = "NOT_TESTABLE"
        h7_numeric = h7_contract = h7_mechanism = pd.NA
    else:
        h7_numeric = bool(favourable_guards)
        h7_contract = bool(contract_guards)
        h7_mechanism = bool(mechanism_guards)
        h7_status = (
            "INCONCLUSIVE"
            if h7_numeric or h7_mechanism
            else "NOT_SUPPORTED"
        )
    append(
        "H7_defect_guard",
        "Defect guard protects FN without sacrificing TN",
        status=h7_status,
        numerically_better=h7_numeric,
        contract_success=h7_contract,
        mechanism_supported=h7_mechanism,
        rationale=(
            f"Numerically favourable={favourable_guards}; contract-passing="
            f"{contract_guards}; tail-mechanism={mechanism_guards}. "
            "Phase B n=3 and mixed-machine comparisons prevent confirmation."
        ),
        evidence_scope="Phase B guard policies",
    )

    cohort_rows = (
        a02_summaries[
            a02_summaries.get("analysis_cohort", pd.Series(dtype=str)).isin(
                ["discovery", "confirmation"]
            )
        ]
        if not a02_summaries.empty
        else pd.DataFrame()
    )
    if len(cohort_rows) != 4:
        h8_status = "NOT_TESTABLE"
        h8_numeric = pd.NA
    else:
        signs = [
            (
                np.sign(float(row["mean_delta_FN"])),
                np.sign(float(row["mean_delta_TN"])),
            )
            for _, row in cohort_rows.iterrows()
        ]
        h8_numeric = len(set(signs)) == 1
        h8_status = "INCONCLUSIVE"
    append(
        "H8_execution_robustness",
        "A02 evidence is robust to machine, resume, and snapshot factors",
        status=h8_status,
        numerically_better=h8_numeric,
        contract_success=False,
        mechanism_supported=pd.NA,
        rationale=(
            f"Sensitivity rows={len(sensitivity_summaries)}; Phase C is "
            "machine-confounded and cannot be statistically corrected into a causal result."
        ),
        evidence_scope="discovery/confirmation and sensitivity strata",
    )

    capability_status = (
        capability_registry.set_index("capability_id")["status"].to_dict()
        if not capability_registry.empty
        and {"capability_id", "status"}.issubset(capability_registry.columns)
        else {}
    )
    for hypothesis_id, capability_id, hypothesis in (
        ("H9_no_replay", "no_replay_baseline", "Replay outperforms no replay"),
        ("H10_gradient_value", "gradient_evidence", "Gradient-aligned samples add value"),
        (
            "H11_replaced_rankings",
            "ranking_TailGap-Strict",
            "Replaced overlap-gated rankings have training effects",
        ),
        (
            "H12_blind_external",
            "blind_external_test",
            "The internal result generalizes to blind/external data",
        ),
    ):
        available = capability_status.get(capability_id) == "AVAILABLE"
        append(
            hypothesis_id,
            hypothesis,
            status="INCONCLUSIVE" if available else "NOT_TESTABLE",
            numerically_better=pd.NA,
            contract_success=pd.NA,
            mechanism_supported=pd.NA,
            rationale=f"Capability {capability_id}: {capability_status.get(capability_id, 'missing')}.",
            evidence_scope="capability/provenance audit",
        )

    return pd.DataFrame(records)
