# -*- coding: utf-8 -*-
"""Validate Stage-1 OOF prediction artifacts before using machine results.

This is intentionally strict.  The old raw-only OOF table is invalid for
confidence/difficulty conclusions.  A valid table must contain calibrated and
operational confidence columns, non-empty cal/op scores, and operational
predictions consistent with the selected threshold.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.predict_stage1_oof_folds_20260621 import difficulty_bucket  # noqa: E402


DEFAULT_PREDICTION_ROOT = Path("artifacts") / "stage1_oof_predictions_calop_20260621"

REQUIRED_COLUMNS = {
    "human_fold",
    "y_true",
    "p_defect_cal",
    "p_defect_operational",
    "true_confidence_cal",
    "wrong_confidence_cal",
    "difficulty_bucket_cal",
    "true_confidence_operational",
    "wrong_confidence_operational",
    "difficulty_bucket_operational",
    "operational_threshold",
    "y_pred_operational",
    "operational_correct",
}

REQUIRED_NON_EMPTY_COLUMNS = {
    "p_defect_cal",
    "p_defect_operational",
    "true_confidence_cal",
    "wrong_confidence_cal",
    "difficulty_bucket_cal",
    "true_confidence_operational",
    "wrong_confidence_operational",
    "difficulty_bucket_operational",
    "operational_threshold",
    "y_pred_operational",
    "operational_correct",
}

LEGACY_RAW_CONFIDENCE_COLUMNS = {
    "true_confidence_raw",
    "wrong_confidence_raw",
    "difficulty_bucket_raw",
}


def probability(value: str, *, column: str, row_number: int) -> float:
    if value == "":
        raise ValueError(f"row {row_number}: empty required cal/op column {column}")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"row {row_number}: probability out of range in {column}: {value}")
    return parsed


def close_enough(left: float, right: float, *, tolerance: float = 5e-7) -> bool:
    return abs(left - right) <= tolerance


def validate_rows(fieldnames: Iterable[str], rows: list[dict[str, str]]) -> list[str]:
    columns = set(fieldnames)
    errors: list[str] = []

    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")

    legacy = sorted(LEGACY_RAW_CONFIDENCE_COLUMNS & columns)
    if legacy:
        errors.append(f"Found legacy raw confidence columns; regenerate cal/op artifacts: {', '.join(legacy)}")

    if not rows:
        errors.append("Prediction table has zero rows.")
        return errors

    max_row_errors = 30
    suppressed = 0
    for index, row in enumerate(rows, start=2):
        row_errors: list[str] = []
        for column in REQUIRED_NON_EMPTY_COLUMNS:
            if row.get(column, "") == "":
                row_errors.append(f"row {index}: empty required cal/op column {column}")

        if not row_errors and not missing:
            try:
                y_true = int(row["y_true"])
                if y_true not in (0, 1):
                    row_errors.append(f"row {index}: y_true must be 0 or 1")

                p_cal = probability(row["p_defect_cal"], column="p_defect_cal", row_number=index)
                p_op = probability(row["p_defect_operational"], column="p_defect_operational", row_number=index)
                threshold = probability(row["operational_threshold"], column="operational_threshold", row_number=index)

                expected_true_cal = p_cal if y_true == 1 else 1.0 - p_cal
                expected_wrong_cal = 1.0 - expected_true_cal
                actual_true_cal = probability(row["true_confidence_cal"], column="true_confidence_cal", row_number=index)
                actual_wrong_cal = probability(row["wrong_confidence_cal"], column="wrong_confidence_cal", row_number=index)
                if not close_enough(actual_true_cal, expected_true_cal):
                    row_errors.append(f"row {index}: true_confidence_cal mismatch")
                if not close_enough(actual_wrong_cal, expected_wrong_cal):
                    row_errors.append(f"row {index}: wrong_confidence_cal mismatch")
                if row["difficulty_bucket_cal"] != difficulty_bucket(actual_wrong_cal):
                    row_errors.append(f"row {index}: difficulty_bucket_cal mismatch")

                expected_true_op = p_op if y_true == 1 else 1.0 - p_op
                expected_wrong_op = 1.0 - expected_true_op
                actual_true_op = probability(
                    row["true_confidence_operational"],
                    column="true_confidence_operational",
                    row_number=index,
                )
                actual_wrong_op = probability(
                    row["wrong_confidence_operational"],
                    column="wrong_confidence_operational",
                    row_number=index,
                )
                if not close_enough(actual_true_op, expected_true_op):
                    row_errors.append(f"row {index}: true_confidence_operational mismatch")
                if not close_enough(actual_wrong_op, expected_wrong_op):
                    row_errors.append(f"row {index}: wrong_confidence_operational mismatch")
                if row["difficulty_bucket_operational"] != difficulty_bucket(actual_wrong_op):
                    row_errors.append(f"row {index}: difficulty_bucket_operational mismatch")

                expected_pred = 1 if p_op >= threshold else 0
                actual_pred = int(row["y_pred_operational"])
                if actual_pred != expected_pred:
                    row_errors.append(f"row {index}: y_pred_operational mismatch")
                expected_correct = int(expected_pred == y_true)
                actual_correct = int(row["operational_correct"])
                if actual_correct != expected_correct:
                    row_errors.append(f"row {index}: operational_correct mismatch")
            except ValueError as exc:
                row_errors.append(str(exc))

        if row_errors:
            if len(errors) < max_row_errors:
                errors.extend(row_errors)
            else:
                suppressed += len(row_errors)

    if suppressed:
        errors.append(f"Suppressed {suppressed} additional row-level errors.")
    return errors


def read_prediction_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def validate_prediction_file(path: Path) -> list[str]:
    fieldnames, rows = read_prediction_csv(path)
    return validate_rows(fieldnames, rows)


def resolve_prediction_csv(prediction_root: Path, merged_csv: Path | None) -> Path:
    if merged_csv is not None:
        return merged_csv
    candidates = [
        prediction_root / "oof_predictions_merged.csv",
        prediction_root / "oof_predictions_merged_10fold.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--merged-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = resolve_prediction_csv(args.prediction_root, args.merged_csv)
    errors = validate_prediction_file(csv_path)
    if errors:
        print(f"INVALID: {csv_path}", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    fieldnames, rows = read_prediction_csv(csv_path)
    folds = sorted({row.get("human_fold", "") for row in rows})
    print(f"VALID_CAL_OP_OOF_PREDICTIONS={csv_path}")
    print(f"rows={len(rows)}")
    print(f"folds={','.join(folds)}")
    print(f"columns={len(fieldnames)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
