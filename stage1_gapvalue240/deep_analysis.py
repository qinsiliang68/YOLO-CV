"""Strict, read-only ingestion for the GapValue 240-run deep analysis.

This module intentionally does not discover attempts.  The transfer audit's
canonical inventory is the sole authority for selecting an attempt directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .metrics import operational_metrics


class CanonicalInputError(RuntimeError):
    """Raised when frozen analysis inputs or canonical run evidence disagree."""


@dataclass(frozen=True)
class CanonicalIngestionResult:
    """Canonical run-level evidence and its two audit tables."""

    runs: pd.DataFrame
    metric_audit: pd.DataFrame
    training_summaries: pd.DataFrame


MATRIX_FIELDS = (
    "run_slot",
    "triad_id",
    "phase",
    "condition_id",
    "method",
    "budget",
    "guard_ratio",
    "arm",
    "training_seed",
    "selection_seed",
)
IDENTITY_FIELDS = (
    "run_slot",
    "machine_id",
    "input_snapshot_id",
    "release_ref",
    "release_commit",
    "resume_count",
    "selection_sha256",
)
REQUIRED_ARTIFACTS = (
    "00_identity/run_identity.json",
    "01_manifests/selection_manifest.csv",
    "02_logs/epoch_training_metrics.csv",
    "02_logs/training_execution_audit.json",
    "05_metrics/operational_metrics.json",
    "07_validation/postflight_report.json",
)
INTEGER_METRIC_LEAVES = {
    "row_count",
    "normal_count",
    "defect_count",
    "TN_at_FN95.actual_FN",
    "TN_at_FN95.actual_FP",
    "TN_at_FN95.actual_TN",
    "TN_at_FN95.actual_TP",
    "TN_at_FN95.tie_group_size",
    "FN_at_TN68253.actual_FN",
    "FN_at_TN68253.actual_FP",
    "FN_at_TN68253.actual_TN",
    "FN_at_TN68253.actual_TP",
    "FN_at_TN68253.tie_group_size",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CanonicalInputError(f"Missing required JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CanonicalInputError(f"Invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CanonicalInputError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _normalized_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if pd.isna(value):
            return None
        return float(value)
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number.is_integer():
        return int(number)
    return number


def _assert_equal(actual: Any, expected: Any, context: str) -> None:
    left = _normalized_scalar(actual)
    right = _normalized_scalar(expected)
    if isinstance(left, float) and isinstance(right, float):
        equal = math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
    else:
        equal = left == right
    if not equal:
        raise CanonicalInputError(
            f"{context} mismatch: expected {expected!r}, found {actual!r}"
        )


def _attempt_path(extracted_root: Path, inventory_row: pd.Series) -> Path:
    """Resolve the one attempt explicitly named by inventory, without scanning."""

    package = str(inventory_row["package"])
    return (
        extracted_root
        / f"stage1_gapvalue240_{package}_upload"
        / "runs"
        / str(inventory_row["run_slot"])
        / str(inventory_row["attempt_id"])
    )


def _validate_matrix_shape(
    inventory: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    expected_runs: int,
    expected_triads: int,
    expected_comparisons: int,
) -> None:
    if len(inventory) != expected_runs:
        raise CanonicalInputError(
            f"Expected {expected_runs} canonical runs, found {len(inventory)}"
        )
    if len(matrix) != expected_runs:
        raise CanonicalInputError(
            f"Expected {expected_runs} matrix rows, found {len(matrix)}"
        )
    for name, frame in (("inventory", inventory), ("matrix", matrix)):
        if frame["run_slot"].duplicated().any():
            duplicates = frame.loc[frame["run_slot"].duplicated(), "run_slot"].tolist()
            raise CanonicalInputError(f"Duplicate {name} run slots: {duplicates}")
    if set(inventory["run_slot"]) != set(matrix["run_slot"]):
        raise CanonicalInputError("Inventory and matrix run-slot sets differ")

    triad_count = int(matrix["triad_id"].nunique())
    if triad_count != expected_triads:
        raise CanonicalInputError(
            f"Expected {expected_triads} triads, found {triad_count}"
        )
    comparisons = 0
    for triad_id, group in matrix.groupby("triad_id", sort=False):
        arms = group["arm"].astype(str).tolist()
        if len(group) != 3 or sorted(arms) != ["R1", "R2", "T"]:
            raise CanonicalInputError(
                f"Triad {triad_id} must contain exactly T/R1/R2, found {arms}"
            )
        comparisons += 2
    if comparisons != expected_comparisons:
        raise CanonicalInputError(
            f"Expected {expected_comparisons} comparisons, found {comparisons}"
        )


def _validate_artifact_manifest(
    attempt: Path,
    *,
    include_prediction: bool,
) -> None:
    manifest_path = attempt / "07_validation/artifact_manifest.csv"
    if not manifest_path.is_file():
        raise CanonicalInputError(f"Missing artifact manifest: {manifest_path}")
    manifest = pd.read_csv(
        manifest_path,
        dtype={"relative_path": "string", "sha256": "string"},
        keep_default_na=False,
    )
    required_columns = {"relative_path", "size_bytes", "sha256"}
    if not required_columns.issubset(manifest.columns):
        raise CanonicalInputError(
            f"Artifact manifest missing columns: {sorted(required_columns-set(manifest.columns))}"
        )
    if manifest["relative_path"].duplicated().any():
        raise CanonicalInputError(f"Duplicate artifact manifest paths: {manifest_path}")
    indexed = manifest.set_index("relative_path")
    required = list(REQUIRED_ARTIFACTS)
    if include_prediction:
        required.append("04_predictions/val_op_predictions.csv")
    for relative in required:
        if relative not in indexed.index:
            raise CanonicalInputError(
                f"Artifact manifest does not list required file {relative}: {attempt}"
            )
        path = attempt / relative
        if not path.is_file():
            raise CanonicalInputError(f"Missing required artifact: {path}")
        row = indexed.loc[relative]
        if int(row["size_bytes"]) != path.stat().st_size:
            raise CanonicalInputError(f"Artifact size mismatch: {path}")
        if str(row["sha256"]).upper() != _sha256(path):
            raise CanonicalInputError(f"Artifact SHA mismatch: {path}")


def _flatten_mapping(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flattened.update(_flatten_mapping(item, name))
        else:
            flattened[name] = item
    return flattened


def _metric_comparison(
    saved: dict[str, Any], recomputed: dict[str, Any]
) -> dict[str, Any]:
    left = _flatten_mapping(saved)
    right = _flatten_mapping(recomputed)
    if set(left) != set(right):
        missing = sorted(set(left) ^ set(right))
        return {
            "exact_match": False,
            "integer_fields_exact": False,
            "max_float_abs_error": math.inf,
            "metric_issue": f"metric leaf mismatch: {missing}",
        }

    integer_exact = True
    nonnumeric_exact = True
    max_float_error = 0.0
    for key in sorted(left):
        saved_value = left[key]
        recomputed_value = right[key]
        if key in INTEGER_METRIC_LEAVES:
            if int(saved_value) != int(recomputed_value):
                integer_exact = False
            continue
        try:
            a = float(saved_value)
            b = float(recomputed_value)
        except (TypeError, ValueError):
            if saved_value != recomputed_value:
                nonnumeric_exact = False
            continue
        if math.isnan(a) and math.isnan(b):
            error = 0.0
        elif math.isinf(a) or math.isinf(b):
            error = 0.0 if a == b else math.inf
        else:
            error = abs(a - b)
        max_float_error = max(max_float_error, error)
    exact = integer_exact and nonnumeric_exact and max_float_error <= 1e-12
    return {
        "exact_match": exact,
        "integer_fields_exact": integer_exact,
        "max_float_abs_error": max_float_error,
        "metric_issue": "" if exact else "saved and recomputed metrics differ",
    }


def _training_summary(
    attempt: Path,
    *,
    run_slot: str,
    expected_epochs: int,
    last_window: int = 20,
) -> dict[str, Any]:
    epoch_path = attempt / "02_logs/epoch_training_metrics.csv"
    epochs = pd.read_csv(epoch_path)
    required = {
        "epoch",
        "time",
        "train/loss",
        "metrics/accuracy_top1",
        "val/loss",
    }
    if not required.issubset(epochs.columns):
        raise CanonicalInputError(
            f"{run_slot} epoch metrics missing columns: {sorted(required-set(epochs.columns))}"
        )
    if len(epochs) != expected_epochs:
        raise CanonicalInputError(
            f"{run_slot} expected {expected_epochs} epoch rows, found {len(epochs)}"
        )
    epoch_numbers = epochs["epoch"].astype(int).tolist()
    if epoch_numbers != list(range(1, expected_epochs + 1)):
        raise CanonicalInputError(f"{run_slot} epoch sequence is not 1..{expected_epochs}")
    numeric = epochs[list(required - {"epoch"})].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise CanonicalInputError(f"{run_slot} epoch metrics contain NaN/Inf")

    audit = _read_json(attempt / "02_logs/training_execution_audit.json")
    _assert_equal(audit.get("completed_epochs"), expected_epochs, f"{run_slot} audit")
    if audit.get("loss_finite") is not True:
        raise CanonicalInputError(f"{run_slot} training audit reports non-finite loss")

    top1 = epochs["metrics/accuracy_top1"].astype(float)
    train_loss = epochs["train/loss"].astype(float)
    val_loss = epochs["val/loss"].astype(float)
    best_index = int(top1.idxmax())
    window_size = min(last_window, expected_epochs)
    tail = epochs.tail(window_size)
    x = np.arange(window_size, dtype=float)
    val_tail = tail["val/loss"].to_numpy(dtype=float)
    val_slope = float(np.polyfit(x, val_tail, 1)[0]) if window_size > 1 else 0.0
    if expected_epochs > 1:
        top1_auc = float(
            np.trapz(top1.to_numpy(dtype=float), epochs["epoch"].to_numpy(dtype=float))
            / (expected_epochs - 1)
        )
    else:
        top1_auc = float(top1.iloc[0])
    return {
        "run_slot": run_slot,
        "completed_epochs": expected_epochs,
        "expected_steps_per_epoch": int(audit["expected_steps_per_epoch"]),
        "optimizer_steps_total": int(audit["optimizer_steps_total"]),
        "loss_finite": bool(audit["loss_finite"]),
        "best_top1_epoch": int(epochs.loc[best_index, "epoch"]),
        "best_top1": float(top1.loc[best_index]),
        "final_top1": float(top1.iloc[-1]),
        "final_train_loss": float(train_loss.iloc[-1]),
        "final_val_loss": float(val_loss.iloc[-1]),
        "last_window_epochs": window_size,
        "last_window_top1_mean": float(tail["metrics/accuracy_top1"].mean()),
        "last_window_top1_std": float(
            tail["metrics/accuracy_top1"].std(ddof=0)
        ),
        "last_window_val_loss_mean": float(tail["val/loss"].mean()),
        "last_window_val_loss_std": float(tail["val/loss"].std(ddof=0)),
        "last_window_val_loss_slope": val_slope,
        "top1_normalized_auc": top1_auc,
    }


def _metric_run_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    try:
        tn_point = metrics["TN_at_FN95"]
        fn_point = metrics["FN_at_TN68253"]
        return {
            "TN_at_FN95": int(tn_point["actual_TN"]),
            "actual_FN_at_FN95": int(tn_point["actual_FN"]),
            "threshold_at_FN95": float(tn_point["threshold"]),
            "tie_group_at_FN95": int(tn_point["tie_group_size"]),
            "FN_at_TN68253": int(fn_point["actual_FN"]),
            "actual_TN_at_TN68253": int(fn_point["actual_TN"]),
            "threshold_at_TN68253": float(fn_point["threshold"]),
            "tie_group_at_TN68253": int(fn_point["tie_group_size"]),
            "normal_q68": float(metrics["normal_q68"]),
            "normal_q90": float(metrics["normal_q90"]),
            "defect_q50": float(metrics["defect_q50"]),
            "defect_q05": float(metrics["defect_q05"]),
            "gap_q68_q050": float(metrics["gap_q68_q050"]),
            "tail_gap_q90_q05": float(metrics["tail_gap_q90_q05"]),
            "prediction_rows": int(metrics["row_count"]),
            "normal_count": int(metrics["normal_count"]),
            "defect_count": int(metrics["defect_count"]),
            "metric_version": str(metrics["metric_version"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise CanonicalInputError(f"Malformed operational metrics: {exc}") from exc


def ingest_canonical_runs(
    extracted_root: str | Path,
    inventory_path: str | Path,
    matrix_path: str | Path,
    *,
    expected_runs: int = 240,
    expected_triads: int = 80,
    expected_comparisons: int = 160,
    expected_epochs: int = 200,
    expected_prediction_rows: int | None = 120_000,
    expected_normal_count: int | None = 100_000,
    expected_defect_count: int | None = 20_000,
    recompute_predictions: bool = False,
    fn_limit: int = 95,
    tn_target: int = 68_253,
) -> CanonicalIngestionResult:
    """Validate and ingest exactly the attempts named by the transfer inventory.

    The function is read-only.  It never walks run directories, chooses a newer
    attempt, or rewrites an input artifact.
    """

    root = Path(extracted_root).resolve()
    inventory_file = Path(inventory_path).resolve()
    matrix_file = Path(matrix_path).resolve()
    if not root.is_dir():
        raise CanonicalInputError(f"Extracted root does not exist: {root}")
    for path, label in ((inventory_file, "inventory"), (matrix_file, "matrix")):
        if not path.is_file():
            raise CanonicalInputError(f"Missing {label}: {path}")

    inventory = pd.read_csv(inventory_file, keep_default_na=False)
    matrix = pd.read_csv(matrix_file, keep_default_na=False)
    required_inventory = set(MATRIX_FIELDS) | {
        "package",
        "attempt_id",
        *IDENTITY_FIELDS,
    }
    required_matrix = set(MATRIX_FIELDS) | {
        "condition_slot",
        "discovery_or_confirmation",
    }
    if not required_inventory.issubset(inventory.columns):
        raise CanonicalInputError(
            f"Inventory missing columns: {sorted(required_inventory-set(inventory.columns))}"
        )
    if not required_matrix.issubset(matrix.columns):
        raise CanonicalInputError(
            f"Matrix missing columns: {sorted(required_matrix-set(matrix.columns))}"
        )
    _validate_matrix_shape(
        inventory,
        matrix,
        expected_runs=expected_runs,
        expected_triads=expected_triads,
        expected_comparisons=expected_comparisons,
    )
    matrix_sha = _sha256(matrix_file)
    matrix_by_slot = matrix.set_index("run_slot", drop=False)

    run_rows: list[dict[str, Any]] = []
    metric_audit_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for _, inventory_row in inventory.sort_values("run_slot").iterrows():
        slot = str(inventory_row["run_slot"])
        matrix_row = matrix_by_slot.loc[slot]
        for field in MATRIX_FIELDS:
            _assert_equal(
                inventory_row[field],
                matrix_row[field],
                f"{slot} inventory/matrix {field}",
            )

        attempt = _attempt_path(root, inventory_row)
        if not attempt.is_dir():
            raise CanonicalInputError(
                f"Canonical inventory attempt does not exist: {attempt}"
            )
        status = _read_json(attempt / "08_status/status.json")
        if status.get("state") != "VALIDATED":
            raise CanonicalInputError(
                f"{slot} canonical attempt is not VALIDATED: {status.get('state')}"
            )
        _assert_equal(status.get("run_slot"), slot, f"{slot} status run_slot")
        postflight = _read_json(attempt / "07_validation/postflight_report.json")
        if postflight.get("status") != "PASS" or postflight.get("issues", []):
            raise CanonicalInputError(f"{slot} postflight did not pass cleanly")

        identity = _read_json(attempt / "00_identity/run_identity.json")
        if identity.get("dry_run") is not False:
            raise CanonicalInputError(f"{slot} canonical attempt is a dry run")
        expected_attempt_id = str(inventory_row["attempt_id"]).removeprefix("attempt_")
        _assert_equal(
            identity.get("attempt_id"),
            expected_attempt_id,
            f"{slot} identity attempt_id",
        )
        for field in IDENTITY_FIELDS:
            _assert_equal(
                identity.get(field),
                inventory_row[field],
                f"{slot} identity {field}",
            )
        _assert_equal(
            identity.get("matrix_sha256"),
            matrix_sha,
            f"{slot} identity matrix SHA",
        )
        identity_run_row = identity.get("run_row")
        if not isinstance(identity_run_row, dict):
            raise CanonicalInputError(f"{slot} identity lacks run_row")
        for field in required_matrix:
            _assert_equal(
                identity_run_row.get(field),
                matrix_row[field],
                f"{slot} identity/matrix {field}",
            )

        selection_path = attempt / "01_manifests/selection_manifest.csv"
        if not selection_path.is_file():
            raise CanonicalInputError(f"Missing selection manifest: {selection_path}")
        actual_selection_sha = _sha256(selection_path)
        if actual_selection_sha != str(inventory_row["selection_sha256"]).upper():
            raise CanonicalInputError(
                f"{slot} selection SHA mismatch: expected "
                f"{inventory_row['selection_sha256']}, found {actual_selection_sha}"
            )
        _validate_artifact_manifest(
            attempt, include_prediction=recompute_predictions
        )

        saved_metrics = _read_json(attempt / "05_metrics/operational_metrics.json")
        metric_fields = _metric_run_fields(saved_metrics)
        if (
            expected_prediction_rows is not None
            and metric_fields["prediction_rows"] != expected_prediction_rows
        ):
            raise CanonicalInputError(
                f"{slot} expected {expected_prediction_rows} saved prediction rows, "
                f"found {metric_fields['prediction_rows']}"
            )
        if (
            expected_normal_count is not None
            and metric_fields["normal_count"] != expected_normal_count
        ):
            raise CanonicalInputError(
                f"{slot} expected {expected_normal_count} normal predictions, "
                f"found {metric_fields['normal_count']}"
            )
        if (
            expected_defect_count is not None
            and metric_fields["defect_count"] != expected_defect_count
        ):
            raise CanonicalInputError(
                f"{slot} expected {expected_defect_count} defect predictions, "
                f"found {metric_fields['defect_count']}"
            )

        audit_row: dict[str, Any] = {
            "run_slot": slot,
            "recomputed": False,
            "exact_match": pd.NA,
            "integer_fields_exact": pd.NA,
            "max_float_abs_error": pd.NA,
            "metric_issue": "",
        }
        if recompute_predictions:
            prediction_path = attempt / "04_predictions/val_op_predictions.csv"
            predictions = pd.read_csv(
                prediction_path,
                usecols=["sample_id", "y_true", "score"],
                dtype={"sample_id": "string"},
            )
            if (
                expected_prediction_rows is not None
                and len(predictions) != expected_prediction_rows
            ):
                raise CanonicalInputError(
                    f"{slot} expected {expected_prediction_rows} prediction rows, "
                    f"found {len(predictions)}"
                )
            if (
                expected_normal_count is not None
                and int((predictions["y_true"] == 0).sum()) != expected_normal_count
            ):
                raise CanonicalInputError(f"{slot} prediction normal count mismatch")
            if (
                expected_defect_count is not None
                and int((predictions["y_true"] == 1).sum()) != expected_defect_count
            ):
                raise CanonicalInputError(f"{slot} prediction defect count mismatch")
            recomputed, _ = operational_metrics(
                predictions, fn_limit=fn_limit, tn_target=tn_target
            )
            comparison = _metric_comparison(saved_metrics, recomputed)
            audit_row.update({"recomputed": True, **comparison})
            if not comparison["exact_match"]:
                raise CanonicalInputError(
                    f"{slot} operational metric recompute mismatch: "
                    f"{comparison['metric_issue']}"
                )
        metric_audit_rows.append(audit_row)

        training = _training_summary(
            attempt, run_slot=slot, expected_epochs=expected_epochs
        )
        training_rows.append(training)
        run_row = {field: matrix_row[field] for field in matrix.columns}
        run_row.update(
            {
                "attempt_dir": str(attempt),
                "attempt_id": str(inventory_row["attempt_id"]),
                "package": str(inventory_row["package"]),
                "machine_id": str(inventory_row["machine_id"]),
                "input_snapshot_id": str(inventory_row["input_snapshot_id"]),
                "release_ref": str(inventory_row["release_ref"]),
                "release_commit": str(inventory_row["release_commit"]),
                "selection_sha256": actual_selection_sha,
                "resume_count": int(inventory_row["resume_count"]),
                "resume_mode": str(identity.get("resume_mode", "none")),
                **metric_fields,
            }
        )
        run_rows.append(run_row)

    runs = pd.DataFrame(run_rows).sort_values("run_slot").reset_index(drop=True)
    metric_audit = (
        pd.DataFrame(metric_audit_rows).sort_values("run_slot").reset_index(drop=True)
    )
    training_summaries = (
        pd.DataFrame(training_rows).sort_values("run_slot").reset_index(drop=True)
    )
    return CanonicalIngestionResult(
        runs=runs,
        metric_audit=metric_audit,
        training_summaries=training_summaries,
    )

