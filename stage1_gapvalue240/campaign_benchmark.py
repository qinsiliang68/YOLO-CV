from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def summarize_crossed_telemetry_benchmarks(
    baseline_first: dict[str, Any],
    telemetry_first: dict[str, Any],
) -> dict[str, Any]:
    """Estimate telemetry overhead by matching first/second execution positions."""

    reports = (baseline_first, telemetry_first)
    if any(report.get("status") != "PASS" for report in reports):
        raise ValueError("crossed telemetry benchmarks must both have PASS status")
    settings = {(int(report.get("epochs", -1)), int(report.get("batch", -1))) for report in reports}
    if len(settings) != 1:
        raise ValueError("crossed telemetry benchmark dimensions differ")
    expected_orders = (
        ["LOCAL_BASELINE", "LOCAL_TELEMETRY"],
        ["LOCAL_TELEMETRY", "LOCAL_BASELINE"],
    )
    for report, expected in zip(reports, expected_orders, strict=True):
        if list(map(str, report.get("execution_order", []))) != expected:
            raise ValueError("crossed telemetry benchmark execution order is invalid")
    by_run = []
    for report in reports:
        rows = {str(row["run_id"]): row for row in report.get("runs", [])}
        if set(rows) != {"LOCAL_BASELINE", "LOCAL_TELEMETRY"}:
            raise ValueError("crossed telemetry benchmark run identities are incomplete")
        by_run.append(rows)
    first_ratio = float(by_run[1]["LOCAL_TELEMETRY"]["duration_seconds"]) / float(
        by_run[0]["LOCAL_BASELINE"]["duration_seconds"]
    )
    second_ratio = float(by_run[0]["LOCAL_TELEMETRY"]["duration_seconds"]) / float(
        by_run[1]["LOCAL_BASELINE"]["duration_seconds"]
    )
    telemetry_bytes = {
        int(rows["LOCAL_TELEMETRY"]["telemetry_bytes"])
        for rows in by_run
    }
    if len(telemetry_bytes) != 1:
        raise ValueError("crossed telemetry byte counts differ")
    telemetry_total = telemetry_bytes.pop()
    epochs = settings.pop()[0]
    parity = all(
        bool(report.get("numerical_parity", {}).get("numerical_parity_passed"))
        for report in reports
    )
    return {
        "schema_version": "stage1.crossed_telemetry_benchmark.v1",
        "status": "PASS" if parity else "FAILED_NUMERICAL_PARITY",
        "numerical_parity_both_orders": parity,
        "position_matched_runtime_ratios": {
            "first": round(first_ratio, 6),
            "second": round(second_ratio, 6),
        },
        "order_balanced_runtime_ratio": round(math.sqrt(first_ratio * second_ratio), 6),
        "order_balanced_overhead_fraction": round(math.sqrt(first_ratio * second_ratio) - 1.0, 6),
        "telemetry_bytes": telemetry_total,
        "telemetry_bytes_per_epoch": telemetry_total // epochs,
        "interpretation": (
            "Ratios compare telemetry and baseline at the same execution position; "
            "the geometric mean reduces one-time process, CUDA, and cache warmup bias."
        ),
    }


def _difference_report(left: Any, right: Any) -> dict[str, Any]:
    differences: list[tuple[str, float | None]] = []
    compared_tensors = 0
    compared_values = 0

    def compare(a: Any, b: Any, path: str) -> None:
        nonlocal compared_tensors, compared_values
        if torch.is_tensor(a) and torch.is_tensor(b):
            compared_tensors += 1
            compared_values += int(a.numel())
            if a.shape != b.shape or a.dtype != b.dtype:
                differences.append((path, None))
                return
            if not torch.equal(a, b):
                max_abs = float((a.to(torch.float64) - b.to(torch.float64)).abs().max())
                differences.append((path, max_abs))
            return
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                differences.append((f"{path}.__keys__", None))
            for key in sorted(set(a) & set(b), key=str):
                compare(a[key], b[key], f"{path}.{key}")
            return
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                differences.append((f"{path}.__length__", None))
            for index, (left_item, right_item) in enumerate(zip(a, b)):
                compare(left_item, right_item, f"{path}[{index}]")
            return
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            compared_values += int(a.size)
            if not np.array_equal(a, b, equal_nan=True):
                differences.append((path, None))
            return
        compared_values += 1
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return
        if a != b:
            differences.append((path, None))

    compare(left, right, "root")
    finite_differences = [value for _, value in differences if value is not None]
    return {
        "compared_tensors": compared_tensors,
        "compared_values": compared_values,
        "different_values": len(differences),
        "max_abs_difference": max(finite_differences, default=0.0),
        "difference_examples": [path for path, _ in differences[:10]],
    }


def _module_state(value: Any) -> Any:
    return value.state_dict() if hasattr(value, "state_dict") else value


def _compare_results(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = pd.read_csv(left_path)
    right = pd.read_csv(right_path)
    excluded_columns = [column for column in ("time",) if column in left.columns or column in right.columns]
    same_columns = list(left.columns) == list(right.columns)
    same_rows = len(left) == len(right)
    different_values = 0
    difference_examples: list[str] = []
    compared_values = 0
    if same_columns and same_rows:
        for column in left.columns:
            if column in excluded_columns:
                continue
            left_values = left[column]
            right_values = right[column]
            equal = left_values.eq(right_values) | (left_values.isna() & right_values.isna())
            compared_values += len(equal)
            count = int((~equal).sum())
            different_values += count
            if count and len(difference_examples) < 10:
                indices = np.flatnonzero((~equal).to_numpy())
                difference_examples.extend(f"{column}[{index}]" for index in indices[: 10 - len(difference_examples)])
    else:
        different_values = 1
        difference_examples = ["dataframe_shape_or_columns"]
    return {
        "same_columns": same_columns,
        "same_rows": same_rows,
        "excluded_columns": excluded_columns,
        "compared_values": compared_values,
        "different_values": different_values,
        "difference_examples": difference_examples,
    }


def compare_training_numerical_parity(
    *,
    baseline_checkpoint: str | Path,
    telemetry_checkpoint: str | Path,
    baseline_results: str | Path,
    telemetry_results: str | Path,
    yolo_root: str | Path | None = None,
) -> dict[str, Any]:
    if yolo_root is not None:
        from stage1_gapvalue240.campaign_dynamic_training import _activate_local_ultralytics

        _activate_local_ultralytics(Path(yolo_root).resolve())

    baseline = torch.load(Path(baseline_checkpoint), map_location="cpu", weights_only=False)
    telemetry = torch.load(Path(telemetry_checkpoint), map_location="cpu", weights_only=False)
    model = _difference_report(_module_state(baseline.get("model")), _module_state(telemetry.get("model")))
    ema = _difference_report(_module_state(baseline.get("ema")), _module_state(telemetry.get("ema")))
    optimizer = _difference_report(baseline.get("optimizer"), telemetry.get("optimizer"))
    controls = _difference_report(
        {key: baseline.get(key) for key in ("epoch", "updates")},
        {key: telemetry.get(key) for key in ("epoch", "updates")},
    )
    results = _compare_results(Path(baseline_results), Path(telemetry_results))
    passed = all(
        component["different_values"] == 0
        for component in (model, ema, optimizer, controls, results)
    )
    return {
        "schema_version": "stage1.training_numerical_parity.v1",
        "numerical_parity_passed": passed,
        "model": model,
        "ema": ema,
        "optimizer": optimizer,
        "checkpoint_controls": controls,
        "results": results,
    }
