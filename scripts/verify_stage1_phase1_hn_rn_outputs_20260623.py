# -*- coding: utf-8 -*-
"""Verify completed phase-1 HN/RN training and calibrated evaluation outputs.

This is intentionally stricter than the pipeline's inline checks. A run is not
"done" for deployment bookkeeping unless this script can verify weights,
post-train validation, calibrated prediction CSVs, threshold/calibration JSON,
metrics consistency, row counts, and artifact inventory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


FORMAL_MODEL = "l"
FORMAL_EPOCHS = 200
FORMAL_BATCH = 128
FORMAL_EVAL_BATCH = 64
FORMAL_IMGSZ = 224
FORMAL_WORKERS = 4
FORMAL_SEED = 20260606
FORMAL_SAVE_PERIOD = -1
FORMAL_EVAL_SPLITS = ("val_cal", "val_op", "test")
TARGET_RECALL = 0.995
SCORE_COLUMN = "p_defect_operational"
UNKNOWN_TOKENS = {"", "u", "unknown", "unk", "nan", "inf", "-inf", "none", "null", "n/a"}

REQUIRED_EVAL_FILES = (
    "artifact_manifest.csv",
    "artifact_manifest.json",
    "calibration.json",
    "metrics_at_selected_threshold.csv",
    "threshold.json",
    *(f"predictions_{split}.csv" for split in FORMAL_EVAL_SPLITS),
)

REQUIRED_PREDICTION_COLUMNS = (
    "eval_split",
    "y_true",
    "p_defect_raw",
    "p_normal_raw",
    "p_defect_cal",
    "p_defect_operational",
    "Filename",
    "canonical_image_relpath",
    "image_path",
)

PROBABILITY_COLUMNS = (
    "p_defect_raw",
    "p_normal_raw",
    "p_defect_cal",
    "p_defect_operational",
)

METRIC_RATE_COLUMNS = (
    "recall",
    "specificity",
    "precision",
    "accuracy",
    "f1",
    "fpr",
    "predicted_positive_rate",
    "weighted_precision",
    "weighted_accuracy",
    "weighted_f1",
    "weighted_pass_through_rate",
)

METRIC_COUNT_COLUMNS = ("n", "positive_n", "negative_n", "tp", "fp", "tn", "fn")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing CSV: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing JSON: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_float(value: str | int | float, label: str) -> float:
    try:
        parsed = float(value)
    except Exception as exc:  # pragma: no cover - defensive error path
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} is not finite: {value!r}")
    return parsed


def parse_int(value: str | int, label: str) -> int:
    try:
        parsed = int(value)
    except Exception as exc:  # pragma: no cover - defensive error path
        raise ValueError(f"{label} is not an int: {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{label} must be non-negative: {parsed}")
    return parsed


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def reject_unknown(value: str | None, label: str) -> None:
    text = "" if value is None else str(value).strip()
    if text.lower() in UNKNOWN_TOKENS:
        raise ValueError(f"{label} is uncovered/unknown: {value!r}")


def expected_split_counts(dataset_root: Path | None, eval_limit_per_class: int | None) -> dict[str, int] | None:
    if dataset_root is None:
        return None
    manifest_root = dataset_root / "manifests"
    split_files = {
        "val_cal": ("val_cal_manifest.csv", "normal_val_cal_manifest.csv"),
        "val_op": ("val_op_manifest.csv", "normal_val_op_manifest.csv"),
        "test": ("test_manifest.csv", "normal_test_manifest.csv"),
    }
    counts: dict[str, int] = {}
    for split, filenames in split_files.items():
        total = 0
        for filename in filenames:
            rows = read_csv_rows(manifest_root / filename)
            total += min(len(rows), eval_limit_per_class) if eval_limit_per_class is not None else len(rows)
        counts[split] = total
    return counts


def verify_summary(summary: dict, run_id: str, formal: bool) -> tuple[Path, Path]:
    require(summary.get("run_id") == run_id, f"summary run_id mismatch: {summary.get('run_id')} != {run_id}")
    require(summary.get("status") == "ok", f"summary status is not ok: {summary.get('status')}")
    for key in ("train_exit", "eval_exit", "preflight_exit", "post_train_validation_exit"):
        require(summary.get(key) == 0, f"{key} is not 0: {summary.get(key)}")

    if formal:
        expected = {
            "model": FORMAL_MODEL,
            "epochs": FORMAL_EPOCHS,
            "seed": FORMAL_SEED,
            "batch": FORMAL_BATCH,
            "eval_batch": FORMAL_EVAL_BATCH,
            "imgsz": FORMAL_IMGSZ,
            "workers": FORMAL_WORKERS,
            "save_period": FORMAL_SAVE_PERIOD,
            "eval_splits": ",".join(FORMAL_EVAL_SPLITS),
            "eval_limit_per_class": None,
        }
        for key, value in expected.items():
            require(summary.get(key) == value, f"formal hyperparameter mismatch {key}: {summary.get(key)} != {value}")

    best_weight = Path(str(summary.get("best_weight", "")))
    reject_unknown(str(best_weight), "best_weight")
    require(best_weight.exists(), f"missing best_weight: {best_weight}")
    require(best_weight.stat().st_size > 1024 * 1024, f"best_weight too small: {best_weight}")
    last_weight = best_weight.parent / "last.pt"
    require(last_weight.exists(), f"missing last.pt next to best.pt: {last_weight}")
    require(last_weight.stat().st_size > 1024 * 1024, f"last.pt too small: {last_weight}")

    post_csv = Path(str(summary.get("post_train_validation_csv", "")))
    post_json = Path(str(summary.get("post_train_validation_json", "")))
    require(post_csv.exists(), f"missing post-train validation CSV: {post_csv}")
    require(post_json.exists(), f"missing post-train validation JSON: {post_json}")

    eval_dir = Path(str(summary.get("eval_verification", {}).get("eval_run_dir", "")))
    reject_unknown(str(eval_dir), "eval_run_dir")
    require(eval_dir.exists(), f"missing eval run dir: {eval_dir}")
    return best_weight, eval_dir


def verify_artifact_manifest(eval_dir: Path) -> None:
    manifest_rows = read_csv_rows(eval_dir / "artifact_manifest.csv")
    manifest_json = read_json(eval_dir / "artifact_manifest.json")
    require(manifest_json.get("artifact_count") == len(manifest_rows), "artifact manifest count mismatch")
    by_relative = {row.get("relative_path", ""): row for row in manifest_rows}
    for filename in REQUIRED_EVAL_FILES:
        if filename in {"artifact_manifest.csv", "artifact_manifest.json"}:
            continue
        require(filename in by_relative, f"artifact manifest missing {filename}")
        path = eval_dir / filename
        require(path.exists(), f"artifact file missing on disk: {path}")
        expected_size = parse_int(by_relative[filename].get("size_bytes", "0"), f"{filename}.size_bytes")
        require(path.stat().st_size == expected_size, f"artifact size mismatch for {filename}")
        sha = by_relative[filename].get("sha256", "")
        require(len(sha) == 64 and all(ch in "0123456789abcdef" for ch in sha.lower()), f"bad sha256 for {filename}")


def verify_predictions(eval_dir: Path, expected_counts: dict[str, int] | None) -> dict[str, int]:
    row_counts: dict[str, int] = {}
    for split in FORMAL_EVAL_SPLITS:
        path = eval_dir / f"predictions_{split}.csv"
        rows = read_csv_rows(path)
        require(rows, f"{path} has no rows")
        header = set(rows[0])
        missing = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in header]
        require(not missing, f"{path} missing columns: {missing}")
        if expected_counts is not None:
            require(len(rows) == expected_counts[split], f"{split} row count mismatch: {len(rows)} != {expected_counts[split]}")

        seen_filenames: set[str] = set()
        positives = 0
        negatives = 0
        for index, row in enumerate(rows, start=2):
            label = f"{path.name}:line{index}"
            require(row.get("eval_split") == split, f"{label} eval_split mismatch: {row.get('eval_split')}")
            reject_unknown(row.get("Filename"), f"{label}.Filename")
            reject_unknown(row.get("canonical_image_relpath"), f"{label}.canonical_image_relpath")
            reject_unknown(row.get("image_path"), f"{label}.image_path")
            filename = str(row["Filename"])
            require(filename not in seen_filenames, f"{label} duplicate Filename in split: {filename}")
            seen_filenames.add(filename)

            y_true = parse_int(row.get("y_true", ""), f"{label}.y_true")
            require(y_true in {0, 1}, f"{label}.y_true must be 0/1: {y_true}")
            positives += int(y_true == 1)
            negatives += int(y_true == 0)
            for column in PROBABILITY_COLUMNS:
                reject_unknown(row.get(column), f"{label}.{column}")
                value = parse_float(row[column], f"{label}.{column}")
                require(0.0 <= value <= 1.0, f"{label}.{column} outside [0,1]: {value}")

        require(positives > 0 and negatives > 0, f"{split} must include both classes")
        row_counts[split] = len(rows)
    return row_counts


def verify_metrics(eval_dir: Path, row_counts: dict[str, int]) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(eval_dir / "metrics_at_selected_threshold.csv")
    require(len(rows) == len(FORMAL_EVAL_SPLITS), f"metrics row count mismatch: {len(rows)}")
    by_split = {row.get("split", ""): row for row in rows}
    require(set(by_split) == set(FORMAL_EVAL_SPLITS), f"metrics split set mismatch: {set(by_split)}")

    threshold_values: set[str] = set()
    for split, row in by_split.items():
        require(row.get("score_column") == SCORE_COLUMN, f"{split} score_column mismatch: {row.get('score_column')}")
        threshold_values.add(row.get("threshold", ""))
        for column in METRIC_COUNT_COLUMNS:
            reject_unknown(row.get(column), f"{split}.{column}")
        n = parse_int(row["n"], f"{split}.n")
        positive_n = parse_int(row["positive_n"], f"{split}.positive_n")
        negative_n = parse_int(row["negative_n"], f"{split}.negative_n")
        tp = parse_int(row["tp"], f"{split}.tp")
        fp = parse_int(row["fp"], f"{split}.fp")
        tn = parse_int(row["tn"], f"{split}.tn")
        fn = parse_int(row["fn"], f"{split}.fn")
        require(n == row_counts[split], f"{split} metrics n mismatch: {n} != {row_counts[split]}")
        require(n == positive_n + negative_n, f"{split} positive+negative != n")
        require(positive_n == tp + fn, f"{split} tp+fn != positive_n")
        require(negative_n == tn + fp, f"{split} tn+fp != negative_n")
        require(n == tp + fp + tn + fn, f"{split} confusion total != n")
        for column in METRIC_RATE_COLUMNS:
            reject_unknown(row.get(column), f"{split}.{column}")
            value = parse_float(row[column], f"{split}.{column}")
            require(0.0 <= value <= 1.0, f"{split}.{column} outside [0,1]: {value}")
        threshold = parse_float(row.get("threshold", ""), f"{split}.threshold")
        require(0.0 <= threshold <= 1.0, f"{split}.threshold outside [0,1]: {threshold}")

    require(len(threshold_values) == 1, f"metrics thresholds differ: {threshold_values}")
    return by_split


def verify_threshold_and_calibration(eval_dir: Path, metrics: dict[str, dict[str, str]]) -> None:
    threshold = read_json(eval_dir / "threshold.json")
    calibration = read_json(eval_dir / "calibration.json")
    require(threshold.get("selection_split") == "val_op", f"threshold selection_split mismatch: {threshold.get('selection_split')}")
    require(threshold.get("score_column") == SCORE_COLUMN, f"threshold score_column mismatch: {threshold.get('score_column')}")
    selected = parse_float(threshold.get("selected_threshold", ""), "selected_threshold")
    metric_threshold = parse_float(metrics["val_op"].get("threshold", ""), "metrics.val_op.threshold")
    require(abs(selected - metric_threshold) <= 1e-9, f"threshold mismatch: {selected} != {metric_threshold}")
    target_recall = parse_float(threshold.get("target_recall", ""), "target_recall")
    require(abs(target_recall - TARGET_RECALL) <= 1e-12, f"target_recall mismatch: {target_recall}")
    val_op_recall = parse_float(metrics["val_op"]["recall"], "metrics.val_op.recall")
    require(val_op_recall + 1e-12 >= TARGET_RECALL, f"val_op recall below target: {val_op_recall} < {TARGET_RECALL}")

    require(calibration.get("fit_split") == "val_cal", f"calibration fit_split mismatch: {calibration.get('fit_split')}")
    for key in ("coef", "intercept", "source_prevalence", "deployment_defect_prevalence", "prior_adjustment"):
        reject_unknown(calibration.get(key), f"calibration.{key}")
        value = parse_float(calibration[key], f"calibration.{key}")
        if "prevalence" in key:
            require(0.0 < value < 1.0, f"calibration.{key} outside (0,1): {value}")


def verify_run(args: argparse.Namespace, run_id: str) -> dict[str, object]:
    phase_root = Path(args.phase_root).resolve()
    summary_path = phase_root / "pipeline_summaries" / f"{run_id}.json"
    summary = read_json(summary_path)
    best_weight, eval_dir = verify_summary(summary, run_id, formal=not args.allow_nonformal)
    for filename in REQUIRED_EVAL_FILES:
        require((eval_dir / filename).exists(), f"missing eval file: {eval_dir / filename}")
    verify_artifact_manifest(eval_dir)

    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else None
    expected_counts = expected_split_counts(dataset_root, summary.get("eval_limit_per_class"))
    row_counts = verify_predictions(eval_dir, expected_counts)
    metrics = verify_metrics(eval_dir, row_counts)
    verify_threshold_and_calibration(eval_dir, metrics)

    return {
        "run_id": run_id,
        "status": "verified",
        "summary_path": str(summary_path),
        "best_weight": str(best_weight),
        "last_weight": str(best_weight.parent / "last.pt"),
        "eval_dir": str(eval_dir),
        "prediction_rows": row_counts,
        "selected_threshold": read_json(eval_dir / "threshold.json").get("selected_threshold"),
        "val_op_recall": metrics["val_op"].get("recall"),
        "test_recall": metrics["test"].get("recall"),
        "test_specificity": metrics["test"].get("specificity"),
        "test_accuracy": metrics["test"].get("accuracy"),
    }


def parse_run_ids(values: list[str] | None) -> list[str]:
    if not values:
        raise ValueError("Provide at least one --run-id.")
    run_ids: list[str] = []
    for value in values:
        run_ids.extend(part.strip() for part in value.split(",") if part.strip())
    return run_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify completed phase-1 HN/RN outputs.")
    parser.add_argument("--phase-root", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--allow-nonformal", action="store_true")
    args = parser.parse_args()

    reports = []
    failures = []
    for run_id in parse_run_ids(args.run_id):
        try:
            reports.append(verify_run(args, run_id))
            print(f"verified_run={run_id}")
        except Exception as exc:
            failures.append({"run_id": run_id, "error": str(exc)})
            print(f"failed_run={run_id} error={exc}", file=sys.stderr)

    payload = {"verified_runs": reports, "failed_runs": failures, "failed_count": len(failures)}
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if failures:
        return 1
    print(f"verified_count={len(reports)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
