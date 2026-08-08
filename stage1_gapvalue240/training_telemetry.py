"""Full saved-telemetry and paired-outcome analysis for the 240-run Goal."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .comprehensive_audit import DataCoverageError, EPOCH_TELEMETRY_COLUMNS


METRIC_SLUGS = {
    "time": "time",
    "train/loss": "train_loss",
    "metrics/accuracy_top1": "accuracy_top1",
    "metrics/accuracy_top5": "accuracy_top5",
    "val/loss": "val_loss",
    **{f"lr/pg{index}": f"lr_pg{index}" for index in range(8)},
}


def build_triad_outcomes(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Reduce 240 run rows to the Goal's 80 independent triad outcomes."""

    required = {
        "run_slot",
        "triad_id",
        "arm",
        "TN_at_FN95",
        "FN_at_TN68253",
    }
    missing = required - set(run_metrics.columns)
    if missing:
        raise DataCoverageError(f"Run metrics missing columns: {sorted(missing)}")
    records: list[dict[str, object]] = []
    for triad_id, group in run_metrics.groupby("triad_id", sort=True):
        if len(group) != 3 or sorted(group["arm"].astype(str)) != ["R1", "R2", "T"]:
            raise DataCoverageError(
                f"Triad {triad_id} must contain exactly T/R1/R2"
            )
        indexed = group.set_index(group["arm"].astype(str))
        treatment = indexed.loc["T"]
        record: dict[str, object] = {
            "triad_id": str(triad_id),
            "treatment_run_slot": str(treatment["run_slot"]),
        }
        for column in group.columns:
            if column not in required and column != "arm":
                values = group[column].drop_duplicates()
                if len(values) == 1:
                    record[column] = values.iloc[0]
                elif column in {
                    "machine_id",
                    "resume_count",
                    "input_snapshot_id",
                    "selection_seed",
                    "selection_sha256",
                }:
                    record[f"treatment_{column}"] = treatment[column]
        for control in ("R1", "R2"):
            control_row = indexed.loc[control]
            record[f"{control}_run_slot"] = str(control_row["run_slot"])
            record[f"delta_TN_{control}"] = float(treatment["TN_at_FN95"]) - float(
                control_row["TN_at_FN95"]
            )
            record[f"delta_FN_{control}"] = float(
                treatment["FN_at_TN68253"]
            ) - float(control_row["FN_at_TN68253"])
        record["G_TN"] = min(
            float(record["delta_TN_R1"]), float(record["delta_TN_R2"])
        )
        record["G_FN"] = max(
            float(record["delta_FN_R1"]), float(record["delta_FN_R2"])
        )
        record["HARM_TN"] = max(
            float(record["delta_TN_R1"]), float(record["delta_TN_R2"])
        )
        record["HARM_FN"] = min(
            float(record["delta_FN_R1"]), float(record["delta_FN_R2"])
        )
        record["dual_improvement"] = bool(
            float(record["G_TN"]) > 0 and float(record["G_FN"]) <= 0
        )
        record["high_value"] = bool(
            float(record["G_TN"]) >= 300 and float(record["G_FN"]) <= 2
        )
        record["dual_harm"] = bool(
            float(record["HARM_TN"]) < 0 and float(record["HARM_FN"]) > 0
        )
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


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.all(x == x[0]):
        return 0.0
    centered = x - x.mean()
    return float(np.dot(centered, y - y.mean()) / np.dot(centered, centered))


def _segment_cumulative_time(values: np.ndarray) -> tuple[int, float]:
    if not len(values):
        return 0, 0.0
    reset_indices = np.where(np.diff(values) < 0)[0] + 1
    segments = np.split(values, reset_indices)
    return len(reset_indices), float(sum(float(segment[-1]) for segment in segments if len(segment)))


def build_run_telemetry_features(
    curves: pd.DataFrame,
    *,
    cutoffs: Iterable[int] = (120, 140, 150, 160, 180, 200),
    slope_window: int = 20,
) -> pd.DataFrame:
    """Create cutoff-local slopes, curvature, AUC and constants for all metrics."""

    required = {"run_slot", *EPOCH_TELEMETRY_COLUMNS}
    missing = required - set(curves.columns)
    if missing:
        raise DataCoverageError(f"Epoch curves missing columns: {sorted(missing)}")
    cutoffs = tuple(sorted(set(int(value) for value in cutoffs)))
    if not cutoffs or cutoffs[0] < 1 or slope_window < 2:
        raise DataCoverageError("Invalid cutoffs or slope_window")
    metrics = list(METRIC_SLUGS)
    metadata_columns = [
        column
        for column in curves.columns
        if column not in {"epoch", *metrics}
    ]
    records: list[dict[str, object]] = []
    for run_slot, group in curves.groupby("run_slot", sort=True):
        group = group.sort_values("epoch").copy()
        epoch_values = pd.to_numeric(group["epoch"], errors="raise").astype(int)
        maximum = cutoffs[-1]
        if epoch_values.tolist() != list(range(1, maximum + 1)):
            raise DataCoverageError(
                f"{run_slot} epoch grid must be exactly 1..{maximum}"
            )
        record: dict[str, object] = {"run_slot": str(run_slot)}
        for column in metadata_columns:
            values = group[column].drop_duplicates()
            if len(values) != 1:
                raise DataCoverageError(
                    f"{run_slot} metadata column varies within run: {column}"
                )
            record[column] = values.iloc[0]
        time = pd.to_numeric(group["time"], errors="raise").to_numpy(dtype=float)
        resets, segment_time = _segment_cumulative_time(time)
        record["time_reset_count"] = resets
        record["time_segment_cumulative_seconds"] = segment_time

        for cutoff in cutoffs:
            upto = group.loc[epoch_values <= cutoff]
            start = max(1, cutoff - slope_window + 1)
            window = upto.loc[pd.to_numeric(upto["epoch"]) >= start]
            x = pd.to_numeric(window["epoch"], errors="raise").to_numpy(dtype=float)
            for metric in metrics:
                slug = METRIC_SLUGS[metric]
                all_y = pd.to_numeric(upto[metric], errors="raise").to_numpy(dtype=float)
                y = pd.to_numeric(window[metric], errors="raise").to_numpy(dtype=float)
                if not np.isfinite(all_y).all():
                    raise DataCoverageError(f"{run_slot}/{metric} contains NaN/Inf")
                record[f"{slug}__at_{cutoff}"] = float(all_y[-1])
                record[f"{slug}__mean_{start}_to_{cutoff}"] = float(y.mean())
                record[f"{slug}__slope_{start}_to_{cutoff}"] = _linear_slope(x, y)
                split = max(2, len(y) // 2)
                early_x, early_y = x[:split], y[:split]
                late_x, late_y = x[-split:], y[-split:]
                record[f"{slug}__curvature_{start}_to_{cutoff}"] = (
                    _linear_slope(late_x, late_y) - _linear_slope(early_x, early_y)
                )
                epochs_upto = pd.to_numeric(
                    upto["epoch"], errors="raise"
                ).to_numpy(dtype=float)
                record[f"{slug}__auc_1_to_{cutoff}"] = (
                    float(np.trapz(all_y, epochs_upto) / (epochs_upto[-1] - epochs_upto[0]))
                    if len(all_y) > 1
                    else float(all_y[0])
                )
                record[f"{slug}__unique_to_{cutoff}"] = int(
                    pd.Series(all_y).nunique(dropna=False)
                )
        records.append(record)
    return pd.DataFrame(records).sort_values("run_slot", ignore_index=True)


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_OPTIMIZER = re.compile(
    r"optimizer:\s*(?P<name>[A-Za-z0-9_]+)\(lr=(?P<lr>[-+0-9.eE]+),\s*"
    r"momentum=(?P<momentum>[-+0-9.eE]+)\)\s+with parameter groups\s+"
    r"(?P<groups>.+)$"
)
_GROUP = re.compile(
    r"(?P<count>\d+)\s+(?P<label>[A-Za-z_]+)\(decay=(?P<decay>[-+0-9.eE]+)\)"
)


def parse_effective_optimizers(file_inventory: pd.DataFrame) -> pd.DataFrame:
    """Parse the actual optimizer selected by Ultralytics, not args.yaml."""

    required = {"run_slot", "attempt_dir", "relative_path", "normalized_path"}
    missing = required - set(file_inventory.columns)
    if missing:
        raise DataCoverageError(f"File inventory missing columns: {sorted(missing)}")
    logs = file_inventory.loc[
        file_inventory["normalized_path"].astype(str)
        == "02_logs/train_{TIMESTAMP}_{TOKEN}.log"
    ]
    records: list[dict[str, object]] = []
    for run_slot, group in logs.groupby("run_slot", sort=True):
        matches: list[tuple[str, float, float, tuple[tuple[int, str, float], ...], str]] = []
        ignored = False
        evidence = []
        for row in group.itertuples(index=False):
            path = Path(str(row.attempt_dir)) / str(row.relative_path)
            text = _ANSI.sub("", path.read_text(encoding="utf-8", errors="replace"))
            ignored = ignored or "ignoring 'lr0=0.01' and 'momentum=0.937'" in text
            evidence.append(str(row.relative_path))
            for line in text.splitlines():
                match = _OPTIMIZER.search(line)
                if not match:
                    continue
                parsed_groups = tuple(
                    (
                        int(item.group("count")),
                        str(item.group("label")),
                        float(item.group("decay")),
                    )
                    for item in _GROUP.finditer(match.group("groups"))
                )
                matches.append(
                    (
                        match.group("name"),
                        float(match.group("lr")),
                        float(match.group("momentum")),
                        parsed_groups,
                        str(row.relative_path),
                    )
                )
        signatures = {(name, lr, momentum, groups) for name, lr, momentum, groups, _ in matches}
        if len(signatures) != 1:
            raise DataCoverageError(
                f"{run_slot} effective optimizer evidence is missing or inconsistent: {signatures}"
            )
        name, lr, momentum, groups = next(iter(signatures))
        records.append(
            {
                "run_slot": str(run_slot),
                "effective_optimizer": name,
                "effective_lr": lr,
                "effective_momentum": momentum,
                "configured_lr_ignored": bool(ignored),
                "parameter_group_count": len(groups),
                "parameter_group_parameter_counts": ",".join(str(item[0]) for item in groups),
                "parameter_group_labels": ",".join(item[1] for item in groups),
                "parameter_group_weight_decay": ",".join(f"{item[2]:g}" for item in groups),
                "evidence_files": ";".join(sorted(evidence)),
            }
        )
    return pd.DataFrame(records).sort_values("run_slot", ignore_index=True)


def build_execution_exposure_audit(
    file_inventory: pd.DataFrame,
    curves: pd.DataFrame,
    *,
    cutoffs: Iterable[int] = (120, 140, 150, 160, 180, 200),
) -> pd.DataFrame:
    """Join actual manifest dose, optimizer steps and LR update opportunity."""

    required_files = {"run_slot", "attempt_dir", "relative_path", "normalized_path"}
    missing = required_files - set(file_inventory.columns)
    if missing:
        raise DataCoverageError(f"File inventory missing columns: {sorted(missing)}")
    curve_required = {"run_slot", "epoch", *{f"lr/pg{i}" for i in range(8)}}
    missing_curves = curve_required - set(curves.columns)
    if missing_curves:
        raise DataCoverageError(
            f"Curves missing execution/LR columns: {sorted(missing_curves)}"
        )
    cutoffs = tuple(sorted(set(int(value) for value in cutoffs)))
    records: list[dict[str, object]] = []
    relevant = file_inventory.loc[
        file_inventory["normalized_path"].isin(
            {
                "01_manifests/manifest_summary.json",
                "02_logs/training_execution_audit.json",
            }
        )
    ]
    for run_slot, group in relevant.groupby("run_slot", sort=True):
        indexed = group.set_index("normalized_path")
        if set(indexed.index) != {
            "01_manifests/manifest_summary.json",
            "02_logs/training_execution_audit.json",
        }:
            raise DataCoverageError(
                f"{run_slot} needs exactly one manifest summary and training audit"
            )

        def load(relative_type: str) -> dict[str, object]:
            row = indexed.loc[relative_type]
            if isinstance(row, pd.DataFrame):
                raise DataCoverageError(f"{run_slot} duplicate {relative_type}")
            path = Path(str(row["attempt_dir"])) / str(row["relative_path"])
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise DataCoverageError(f"{run_slot} invalid {relative_type}")
            return value

        manifest = load("01_manifests/manifest_summary.json")
        audit = load("02_logs/training_execution_audit.json")
        base_normal = int(manifest["base_normal_rows"])
        base_defect = int(manifest["base_train_rows"])
        replay_normal = int(manifest["replay_normal_rows"])
        replay_defect = int(manifest["replay_defect_rows"])
        epoch_samples = int(manifest["epoch_samples"])
        base_total = base_normal + base_defect
        replay_total = replay_normal + replay_defect
        if base_total + replay_total != epoch_samples:
            raise DataCoverageError(
                f"{run_slot} base+replay does not equal epoch_samples"
            )
        completed_epochs = int(audit["completed_epochs"])
        steps_per_epoch = int(audit["expected_steps_per_epoch"])
        optimizer_steps_total = int(audit["optimizer_steps_total"])
        if steps_per_epoch * completed_epochs != optimizer_steps_total:
            raise DataCoverageError(
                f"{run_slot} effective optimizer steps violate the final audit"
            )
        run_curves = curves.loc[curves["run_slot"].astype(str) == str(run_slot)].copy()
        run_curves = run_curves.sort_values("epoch")
        if len(run_curves) != completed_epochs:
            raise DataCoverageError(
                f"{run_slot} curve/audit epoch mismatch: {len(run_curves)} vs {completed_epochs}"
            )
        record: dict[str, object] = {
            "run_slot": str(run_slot),
            "base_normal_rows": base_normal,
            "base_defect_rows": base_defect,
            "base_total_rows": base_total,
            "replay_normal_rows": replay_normal,
            "replay_defect_rows": replay_defect,
            "replay_total_rows": replay_total,
            "epoch_samples": epoch_samples,
            "replay_fraction": replay_total / epoch_samples,
            "completed_epochs": completed_epochs,
            "effective_batch_size": int(audit["effective_batch_size"]),
            "steps_per_epoch": steps_per_epoch,
            "optimizer_steps_total": optimizer_steps_total,
            "total_replay_exposures": replay_total * completed_epochs,
            "resume_count": int(audit.get("resume_count", 0)),
            "resume_mode": str(audit.get("resume_mode", "")),
            "resume_segment_count": len(audit.get("resume_segments", [])),
        }
        for cutoff in cutoffs:
            if cutoff > completed_epochs:
                raise DataCoverageError(
                    f"{run_slot} cutoff {cutoff} exceeds completed epochs {completed_epochs}"
                )
            upto = run_curves.loc[pd.to_numeric(run_curves["epoch"]) <= cutoff]
            record[f"optimizer_steps_to_{cutoff}"] = steps_per_epoch * cutoff
            record[f"optimizer_steps_after_{cutoff}"] = steps_per_epoch * (
                completed_epochs - cutoff
            )
            record[f"replay_exposures_to_{cutoff}"] = replay_total * cutoff
            record[f"replay_exposures_after_{cutoff}"] = replay_total * (
                completed_epochs - cutoff
            )
            for index in range(8):
                lr = pd.to_numeric(upto[f"lr/pg{index}"], errors="raise").to_numpy(
                    dtype=float
                )
                record[f"lr_step_integral_pg{index}_to_{cutoff}"] = float(
                    lr.sum() * steps_per_epoch
                )
                record[f"lr_replay_integral_pg{index}_to_{cutoff}"] = float(
                    lr.sum() * replay_total
                )
        records.append(record)
    result = pd.DataFrame(records).sort_values("run_slot", ignore_index=True)
    curve_runs = set(curves["run_slot"].astype(str))
    if set(result["run_slot"].astype(str)) != curve_runs:
        raise DataCoverageError("Execution/exposure audit does not cover every curve run")
    return result


def build_paired_telemetry_deltas(
    run_features: pd.DataFrame,
    *,
    feature_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute T-control telemetry differences without merging R1 and R2."""

    required = {"run_slot", "triad_id", "arm"}
    missing = required - set(run_features.columns)
    if missing:
        raise DataCoverageError(f"Run features missing columns: {sorted(missing)}")
    if feature_columns is None:
        prefixes = (
            "time_",
            "optimizer_steps_",
            "replay_",
            "lr_step_integral_",
            "lr_replay_integral_",
            "effective_",
            "parameter_group_",
        )
        candidates = [
            column
            for column in run_features.columns
            if "__" in column or column.startswith(prefixes)
        ]
        feature_columns = [
            column
            for column in candidates
            if pd.to_numeric(run_features[column], errors="coerce").notna().all()
        ]
    feature_columns = tuple(dict.fromkeys(str(column) for column in feature_columns))
    missing_features = set(feature_columns) - set(run_features.columns)
    if missing_features:
        raise DataCoverageError(
            f"Requested telemetry features missing: {sorted(missing_features)}"
        )
    records: list[dict[str, object]] = []
    for triad_id, group in run_features.groupby("triad_id", sort=True):
        if len(group) != 3 or sorted(group["arm"].astype(str)) != ["R1", "R2", "T"]:
            raise DataCoverageError(
                f"Triad {triad_id} must contain exactly T/R1/R2 telemetry rows"
            )
        indexed = group.set_index(group["arm"].astype(str))
        treatment = indexed.loc["T"]
        for control in ("R1", "R2"):
            control_row = indexed.loc[control]
            record: dict[str, object] = {
                "triad_id": str(triad_id),
                "control": control,
                "treatment_run_slot": str(treatment["run_slot"]),
                "control_run_slot": str(control_row["run_slot"]),
            }
            for column in run_features.columns:
                if column in required or column in feature_columns:
                    continue
                values = group[column].drop_duplicates()
                if len(values) == 1:
                    record[column] = values.iloc[0]
                elif column in {
                    "machine_id",
                    "resume_count",
                    "input_snapshot_id",
                    "selection_seed",
                }:
                    record[f"treatment_{column}"] = treatment[column]
                    record[f"control_{column}"] = control_row[column]
            for column in feature_columns:
                left = pd.to_numeric(pd.Series([treatment[column]]), errors="raise").iloc[0]
                right = pd.to_numeric(pd.Series([control_row[column]]), errors="raise").iloc[0]
                record[f"delta__{column}"] = float(left) - float(right)
            records.append(record)
    return pd.DataFrame(records).sort_values(
        ["triad_id", "control"], ignore_index=True
    )
