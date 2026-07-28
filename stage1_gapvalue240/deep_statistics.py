"""Paired scientific statistics for the frozen GapValue 240-run experiment.

This module deliberately consumes an already-canonical run-level table.  It
does not discover attempts, select runs, or read predictions.  R1 and R2 remain
separate throughout because they answer different control questions.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .statistics import paired_summary


METRIC_COLUMNS: tuple[str, ...] = (
    "TN_at_FN95",
    "FN_at_TN68253",
    "gap_q68_q050",
    "tail_gap_q90_q05",
    "normal_q68",
    "normal_q90",
    "defect_q50",
    "defect_q05",
)

DELTA_NAMES: dict[str, str] = {
    "TN_at_FN95": "delta_TN",
    "FN_at_TN68253": "delta_FN",
    **{metric: f"delta_{metric}" for metric in METRIC_COLUMNS[2:]},
}

TRIAD_METADATA: tuple[str, ...] = (
    "phase",
    "condition_slot",
    "condition_id",
    "method",
    "budget",
    "guard_ratio",
    "training_seed",
    "discovery_or_confirmation",
)

_METHOD_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("A01", "A05"),
    ("A01", "A07"),
    ("A01", "A09"),
    ("A01", "A11"),
    ("A02", "A04"),
    ("A02", "A06"),
    ("A02", "A08"),
    ("A02", "A10"),
    ("A02", "A12"),
    ("A02", "A13"),
    ("A02", "A14"),
    ("A02", "A15"),
    ("A02", "A16"),
    ("A02", "A17"),
    ("A02", "A19"),
    ("A03", "A18"),
)

_BUDGET_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("A02", "A01"),
    ("A03", "A02"),
    ("A03", "A01"),
    ("A06", "A05"),
    ("A08", "A07"),
    ("A10", "A09"),
    ("A12", "A11"),
)

_GUARD_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("B04", "B03"),
    ("B05", "B04"),
    ("B04", "B01"),
    ("B04", "B02"),
    ("B04", "B06"),
)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required run-level columns: {missing}")


def _single_value(group: pd.DataFrame, column: str, triad_id: str):
    values = group[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"Triad {triad_id} has inconsistent {column}: {values.tolist()}")
    return values.iloc[0]


def _optional_value(row: pd.Series, column: str, default):
    if column not in row.index or pd.isna(row[column]):
        return default
    return row[column]


def build_triad_deltas(
    run_results: pd.DataFrame,
    *,
    metric_columns: Sequence[str] = METRIC_COLUMNS,
) -> pd.DataFrame:
    """Build one T-control record for each R1 and R2 comparison.

    Every input triad must contain exactly one T, one R1, and one R2 row.
    Metric deltas are always ``T - control``.  Consequently, lower delta_FN and
    normal quantiles are favourable, while higher delta_TN, gap, and defect
    quantiles are favourable.
    """

    required = ("run_slot", "triad_id", "arm", *TRIAD_METADATA, *metric_columns)
    _require_columns(run_results, required)
    duplicated = run_results.duplicated(["triad_id", "arm"], keep=False)
    if duplicated.any():
        pairs = (
            run_results.loc[duplicated, ["triad_id", "arm"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(f"Found duplicate triad/arm rows: {pairs}")

    records: list[dict] = []
    for triad_id, group in run_results.groupby("triad_id", sort=True, dropna=False):
        arms = set(group["arm"].astype(str))
        if arms != {"T", "R1", "R2"} or len(group) != 3:
            raise ValueError(
                f"Triad {triad_id} must contain exactly T, R1, and R2; got {sorted(arms)}"
            )
        metadata = {
            column: _single_value(group, column, str(triad_id))
            for column in TRIAD_METADATA
        }
        indexed = group.set_index("arm")
        treatment = indexed.loc["T"]
        for control_name in ("R1", "R2"):
            control = indexed.loc[control_name]
            t_machine = str(_optional_value(treatment, "machine_id", "unknown"))
            c_machine = str(_optional_value(control, "machine_id", "unknown"))
            machines_known = t_machine != "unknown" and c_machine != "unknown"
            same_machine = bool(t_machine == c_machine) if machines_known else pd.NA
            t_resume = int(_optional_value(treatment, "resume_count", 0))
            c_resume = int(_optional_value(control, "resume_count", 0))
            t_snapshot = str(
                _optional_value(treatment, "input_snapshot_id", "unknown")
            )
            c_snapshot = str(_optional_value(control, "input_snapshot_id", "unknown"))
            snapshots_known = t_snapshot != "unknown" and c_snapshot != "unknown"

            record = {
                "triad_id": triad_id,
                **metadata,
                "control": control_name,
                "t_run_slot": treatment["run_slot"],
                "control_run_slot": control["run_slot"],
                "t_selection_seed": _optional_value(
                    treatment, "selection_seed", pd.NA
                ),
                "control_selection_seed": _optional_value(
                    control, "selection_seed", pd.NA
                ),
                "t_machine_id": t_machine,
                "control_machine_id": c_machine,
                "machine_pair": (
                    "same_machine"
                    if same_machine is True
                    else "cross_machine"
                    if same_machine is False
                    else "unknown"
                ),
                "same_machine": same_machine,
                "t_resume_count": t_resume,
                "control_resume_count": c_resume,
                "any_resumed": bool(t_resume > 0 or c_resume > 0),
                "t_input_snapshot_id": t_snapshot,
                "control_input_snapshot_id": c_snapshot,
                "same_input_snapshot": (
                    bool(t_snapshot == c_snapshot) if snapshots_known else pd.NA
                ),
            }
            for metric in metric_columns:
                record[DELTA_NAMES.get(metric, f"delta_{metric}")] = float(
                    treatment[metric] - control[metric]
                )
            records.append(record)

    return pd.DataFrame(records).sort_values(
        ["triad_id", "control"], key=lambda x: x.map({"R1": 0, "R2": 1}).fillna(x)
        if x.name == "control"
        else x,
        ignore_index=True,
    )


def _delta_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("delta_")]


def _summarize_delta_group(group: pd.DataFrame) -> dict:
    if group.empty:
        raise ValueError("Cannot summarize an empty paired group")
    result = paired_summary(group["delta_FN"], group["delta_TN"])
    for column in _delta_columns(group):
        values = group[column].astype(float)
        suffix = column.removeprefix("delta_")
        result.setdefault(f"mean_delta_{suffix}", float(values.mean()))
        result.setdefault(
            f"std_delta_{suffix}",
            float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        )
        result[f"median_delta_{suffix}"] = float(values.median())
    result["exploratory"] = bool(len(group) <= 3)
    return result


def build_condition_summaries(deltas: pd.DataFrame) -> pd.DataFrame:
    """Summarize each frozen condition and control without pooling R1/R2."""

    _require_columns(
        deltas,
        (
            "condition_id",
            "condition_slot",
            "phase",
            "method",
            "budget",
            "guard_ratio",
            "control",
            "delta_FN",
            "delta_TN",
        ),
    )
    keys = (
        "phase",
        "condition_slot",
        "condition_id",
        "method",
        "budget",
        "guard_ratio",
        "control",
    )
    records = []
    for values, group in deltas.groupby(list(keys), sort=True, dropna=False):
        record = dict(zip(keys, values))
        record.update(_summarize_delta_group(group))
        records.append(record)
    return pd.DataFrame(records)


def build_a02_summaries(deltas: pd.DataFrame) -> pd.DataFrame:
    """Return distinct A02 discovery, confirmation, and pooled summaries."""

    _require_columns(
        deltas,
        (
            "condition_slot",
            "discovery_or_confirmation",
            "control",
            "same_machine",
            "delta_FN",
            "delta_TN",
        ),
    )
    primary = deltas[deltas["condition_slot"] == "A02"].copy()
    records = []
    for control in ("R1", "R2"):
        control_rows = primary[primary["control"] == control]
        for cohort in ("discovery", "confirmation"):
            group = control_rows[
                control_rows["discovery_or_confirmation"].astype(str) == cohort
            ]
            if group.empty:
                continue
            record = {
                "condition_slot": "A02",
                "analysis_cohort": cohort,
                "control": control,
                "machine_confounded": not bool(
                    group["same_machine"].fillna(False).all()
                ),
            }
            record.update(_summarize_delta_group(group))
            records.append(record)
        if not control_rows.empty:
            record = {
                "condition_slot": "A02",
                "analysis_cohort": "pooled",
                "control": control,
                "machine_confounded": not bool(
                    control_rows["same_machine"].fillna(False).all()
                ),
            }
            record.update(_summarize_delta_group(control_rows))
            records.append(record)
    return pd.DataFrame(records)


def build_sensitivity_summaries(deltas: pd.DataFrame) -> pd.DataFrame:
    """Summarize all, same/cross-machine, and no-resume paired subsets."""

    _require_columns(
        deltas,
        (
            "condition_id",
            "condition_slot",
            "control",
            "same_machine",
            "any_resumed",
            "delta_FN",
            "delta_TN",
        ),
    )
    selectors = (
        ("all", lambda frame: pd.Series(True, index=frame.index)),
        ("same_machine", lambda frame: frame["same_machine"].fillna(False)),
        ("cross_machine", lambda frame: frame["same_machine"].eq(False)),
        ("no_resume", lambda frame: ~frame["any_resumed"].astype(bool)),
    )
    keys = ("condition_slot", "condition_id", "control")
    records = []
    for values, condition in deltas.groupby(list(keys), sort=True, dropna=False):
        for analysis_set, selector in selectors:
            group = condition.loc[selector(condition)]
            if group.empty:
                continue
            record = dict(zip(keys, values))
            record["analysis_set"] = analysis_set
            record.update(_summarize_delta_group(group))
            records.append(record)
    return pd.DataFrame(records)


def _comparison_summaries(
    deltas: pd.DataFrame,
    specs: Sequence[tuple[str, str]],
    comparison_type: str,
) -> pd.DataFrame:
    _require_columns(
        deltas,
        ("condition_slot", "training_seed", "control", "delta_FN", "delta_TN"),
    )
    delta_columns = _delta_columns(deltas)
    records = []
    for reference, comparator in specs:
        reference_rows = deltas[deltas["condition_slot"] == reference]
        comparator_rows = deltas[deltas["condition_slot"] == comparator]
        if reference_rows.empty or comparator_rows.empty:
            continue
        join_columns = ["training_seed", "control"]
        merged = reference_rows[join_columns + delta_columns].merge(
            comparator_rows[join_columns + delta_columns],
            on=join_columns,
            how="inner",
            suffixes=("_reference", "_comparator"),
            validate="one_to_one",
        )
        for control, group in merged.groupby("control", sort=True):
            if group.empty:
                continue
            differences = pd.DataFrame(
                {
                    column: (
                        group[f"{column}_reference"]
                        - group[f"{column}_comparator"]
                    )
                    for column in delta_columns
                }
            )
            record = {
                "comparison_type": comparison_type,
                "reference_condition": reference,
                "comparator_condition": comparator,
                "control": control,
            }
            summary = _summarize_delta_group(differences)
            record.update(
                {
                    key.replace("delta_", "diff_delta_"): value
                    if key.startswith(("mean_delta_", "std_delta_", "median_delta_"))
                    else value
                    for key, value in summary.items()
                }
            )
            # Primary paired_summary fields do not all begin with mean/std/median.
            record["mean_diff_delta_FN"] = summary["mean_delta_FN"]
            record["mean_diff_delta_TN"] = summary["mean_delta_TN"]
            record["n"] = summary["n"]
            records.append(record)
    return pd.DataFrame(records)


def build_cross_method_comparisons(deltas: pd.DataFrame) -> pd.DataFrame:
    """Compare preregistered same-budget methods by paired treatment effects."""

    return _comparison_summaries(deltas, _METHOD_COMPARISONS, "cross_method")


def build_direct_treatment_comparisons(
    run_results: pd.DataFrame,
    *,
    specs: Sequence[tuple[str, str]] = _METHOD_COMPARISONS,
    metric_columns: Sequence[str] = METRIC_COLUMNS,
) -> pd.DataFrame:
    """Compare treatment arms directly at the same training seed.

    These rows are easier to read than a delta-of-deltas, but unlike a
    within-triad effect they do not automatically cancel machine differences.
    The machine pairing flag is therefore mandatory output.
    """

    _require_columns(
        run_results,
        (
            "condition_slot",
            "training_seed",
            "arm",
            "run_slot",
            "machine_id",
            "resume_count",
            *metric_columns,
        ),
    )
    treatments = run_results[run_results.arm.astype(str) == "T"].copy()
    records: list[dict] = []
    for reference, comparator in specs:
        left = treatments[treatments.condition_slot.astype(str) == reference]
        right = treatments[treatments.condition_slot.astype(str) == comparator]
        if left.empty or right.empty:
            continue
        columns = [
            "training_seed",
            "run_slot",
            "machine_id",
            "resume_count",
            *metric_columns,
        ]
        merged = left[columns].merge(
            right[columns],
            on="training_seed",
            how="inner",
            suffixes=("_reference", "_comparator"),
            validate="one_to_one",
        )
        for _, row in merged.iterrows():
            record = {
                "reference_condition": reference,
                "comparator_condition": comparator,
                "training_seed": int(row.training_seed),
                "reference_run_slot": row.run_slot_reference,
                "comparator_run_slot": row.run_slot_comparator,
                "reference_machine_id": row.machine_id_reference,
                "comparator_machine_id": row.machine_id_comparator,
                "machine_pair": (
                    "same_machine"
                    if row.machine_id_reference == row.machine_id_comparator
                    else "cross_machine"
                ),
                "any_resumed": bool(
                    int(row.resume_count_reference) > 0
                    or int(row.resume_count_comparator) > 0
                ),
            }
            for metric in metric_columns:
                name = DELTA_NAMES.get(metric, f"delta_{metric}").removeprefix(
                    "delta_"
                )
                record[f"direct_delta_{name}"] = float(
                    row[f"{metric}_reference"] - row[f"{metric}_comparator"]
                )
            records.append(record)
    return pd.DataFrame(records)


def build_budget_comparisons(deltas: pd.DataFrame) -> pd.DataFrame:
    """Compare preregistered budgets by paired treatment effects."""

    return _comparison_summaries(deltas, _BUDGET_COMPARISONS, "budget")


def build_guard_comparisons(deltas: pd.DataFrame) -> pd.DataFrame:
    """Compare preregistered guard methods and ratios."""

    return _comparison_summaries(deltas, _GUARD_COMPARISONS, "guard")
