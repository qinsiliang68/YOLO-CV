# -*- coding: utf-8 -*-
"""Stage-1 binary classification gate evaluation.

This script is deliberately separate from training:
1. Training produces one or more YOLO11-cls weights.
2. This script runs a chosen weight on val_model, val_cal, val_op, and test.
3. val_cal fits a simple probability calibration model.
4. val_op selects the highest threshold that satisfies a recall target.
5. test reports the final thresholded metrics.

The most important output is the per-image prediction CSV. It is the reusable
material for later sample-value analysis because every row keeps the original
manifest identifiers and the model score.

Path guide for humans and future agents:
- --weights is the trained model checkpoint to evaluate, usually best.pt or last.pt.
- --dataset-root is the final sampled dataset root that contains Det/ and manifests/.
- --yolo-root is the local Ultralytics YOLOv11 source tree used for import.
- --output-root is where this script writes evaluation runs and manifests.
- On another Windows machine, keep the same dataset structure but override paths
  with CLI arguments or STAGE1_* environment variables instead of editing code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression


SEED = 20260606
TARGET_CLASS_NAME = "target_defect"
NORMAL_CLASS_NAME = "no_target"
TARGET_LABEL_COLUMNS = ("PF", "DE", "FS", "RB", "AF", "OB")

# =========================
# Path defaults
# =========================
# These defaults are repository-relative. They are intentionally not hard-coded
# to C:\GitHub\YOLO-CV because the training/evaluation machines may use
# different drive letters or parent folders.
#
# Override examples:
#   --weights D:\ssh\AI\runs\stage1_cls_sweep\...\weights\best.pt
#   --dataset-root D:\ssh\AI\data\final_sewerml_dataset
#   --yolo-root D:\ssh\AI\YOLOv11
#   --output-root D:\ssh\AI\runs\stage1_cls_eval
DEFAULT_YOLO_ROOT = Path("YOLOv11")
DEFAULT_DATASET_ROOT = Path("data") / "final_sewerml_dataset"
DEFAULT_OUTPUT_ROOT = DEFAULT_YOLO_ROOT / "runs" / "stage1_cls_eval"

ARTIFACT_MANIFEST_CSV_FILENAME = "artifact_manifest.csv"
ARTIFACT_MANIFEST_JSON_FILENAME = "artifact_manifest.json"

SPLIT_MANIFESTS = {
    "val_model": ("val_model_manifest.csv", "normal_val_model_manifest.csv"),
    "val_cal": ("val_cal_manifest.csv", "normal_val_cal_manifest.csv"),
    "val_op": ("val_op_manifest.csv", "normal_val_op_manifest.csv"),
    "test": ("test_manifest.csv", "normal_test_manifest.csv"),
}

PREDICTION_COLUMNS = (
    "eval_split",
    "source_split",
    "y_true",
    "y_pred_raw",
    "raw_correct",
    "p_defect_raw",
    "p_normal_raw",
    "raw_logit",
    "p_defect_cal",
    "p_defect_operational",
    "raw_cross_entropy",
    "cal_cross_entropy",
    "operational_cross_entropy",
    "raw_uncertainty",
    "cal_uncertainty",
    "operational_uncertainty",
    "sample_version",
    "sample_seed",
    "sample_order",
    "source_csv_row_number",
    "source_csv_line_number",
    "Filename",
    "canonical_image_relpath",
    "image_path",
    "target_labels",
    "target_label_count",
    "Defect",
    *TARGET_LABEL_COLUMNS,
)


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    yolo_root: Path
    dataset_root: Path
    manifest_dir: Path
    output_root: Path


@dataclass(frozen=True)
class EvalConfig:
    weights: Path
    run_name: str
    splits: tuple[str, ...]
    seed: int
    imgsz: int
    batch: int
    device: str
    limit_per_class: int | None
    target_recall: float
    deployment_defect_prevalence: float
    dry_run: bool
    exist_ok: bool


@dataclass(frozen=True)
class Calibrator:
    coef: float
    intercept: float
    source_prevalence: float
    deployment_prevalence: float


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def build_paths(args: argparse.Namespace) -> Paths:
    repo_root = repo_root_from_script()
    yolo_root = Path(args.yolo_root).resolve() if args.yolo_root else repo_root / DEFAULT_YOLO_ROOT
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else repo_root / DEFAULT_DATASET_ROOT
    output_root = Path(args.output_root).resolve() if args.output_root else repo_root / DEFAULT_OUTPUT_ROOT
    return Paths(
        repo_root=repo_root,
        yolo_root=yolo_root,
        dataset_root=dataset_root,
        manifest_dir=dataset_root / "manifests",
        output_root=output_root,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def choose_rows(rows: list[dict[str, str]], count: int | None, seed: int, salt: str) -> list[dict[str, str]]:
    if count is None or count >= len(rows):
        return list(rows)
    rng = random.Random(f"{seed}:{salt}")
    return rng.sample(rows, count)


def safe_float(value: str | int | float | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def canonical_image_path(row: dict[str, str], dataset_root: Path) -> Path:
    rel = row.get("canonical_image_relpath", "")
    if rel:
        candidate = dataset_root / Path(rel)
        if candidate.exists():
            return candidate
    source = row.get("source_image_path", "")
    if source:
        candidate = Path(source)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found for row: {row.get('Filename', '<missing Filename>')}")


def manifest_row_to_record(row: dict[str, str], eval_split: str, y_true: int, dataset_root: Path) -> dict[str, str]:
    image_path = canonical_image_path(row, dataset_root)
    record: dict[str, str] = {
        "eval_split": eval_split,
        "source_split": row.get("split", ""),
        "y_true": str(y_true),
        "sample_version": row.get("sample_version", ""),
        "sample_seed": row.get("sample_seed", ""),
        "sample_order": row.get("sample_order", ""),
        "source_csv_row_number": row.get("source_csv_row_number", ""),
        "source_csv_line_number": row.get("source_csv_line_number", ""),
        "Filename": row.get("Filename", ""),
        "canonical_image_relpath": row.get("canonical_image_relpath", ""),
        "image_path": str(image_path),
        "target_labels": row.get("target_labels", ""),
        "target_label_count": row.get("target_label_count", ""),
        "Defect": row.get("Defect", ""),
    }
    for col in TARGET_LABEL_COLUMNS:
        record[col] = row.get(col, "")
    return record


def load_split_records(paths: Paths, split: str, cfg: EvalConfig) -> list[dict[str, str]]:
    defect_manifest, normal_manifest = SPLIT_MANIFESTS[split]
    defect_rows = read_manifest(paths.manifest_dir / defect_manifest)
    normal_rows = read_manifest(paths.manifest_dir / normal_manifest)

    defect_rows = choose_rows(defect_rows, cfg.limit_per_class, cfg.seed, f"{split}:defect")
    normal_rows = choose_rows(normal_rows, cfg.limit_per_class, cfg.seed, f"{split}:normal")

    records = [manifest_row_to_record(row, split, 1, paths.dataset_root) for row in defect_rows]
    records.extend(manifest_row_to_record(row, split, 0, paths.dataset_root) for row in normal_rows)
    return records


def ensure_yolo_import(yolo_root: Path):
    if not yolo_root.exists():
        raise FileNotFoundError(f"YOLO root not found: {yolo_root}")
    sys.path.insert(0, str(yolo_root))
    from ultralytics import YOLO

    return YOLO


def model_names(model) -> dict[int, str]:
    names = getattr(model, "names", None) or getattr(getattr(model, "model", None), "names", None)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    raise ValueError(f"Unable to read class names from model: {names!r}")


def find_class_index(names: dict[int, str], class_name: str) -> int:
    for idx, name in names.items():
        if name == class_name:
            return idx
    raise ValueError(f"Class {class_name!r} not found in model names: {names}")


def clip_probability(prob: float) -> float:
    return min(max(float(prob), 1e-7), 1.0 - 1e-7)


def logit(prob: float) -> float:
    p = clip_probability(prob)
    return math.log(p / (1.0 - p))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def cross_entropy(y_true: int, prob_positive: float) -> float:
    p = clip_probability(prob_positive)
    return -math.log(p if y_true == 1 else 1.0 - p)


def uncertainty(prob_positive: float) -> float:
    return 1.0 - abs(2.0 * clip_probability(prob_positive) - 1.0)


def predict_records(model, records: list[dict[str, str]], cfg: EvalConfig) -> list[dict[str, str]]:
    names = model_names(model)
    defect_index = find_class_index(names, TARGET_CLASS_NAME)
    normal_index = next((idx for idx, name in names.items() if name == NORMAL_CLASS_NAME), None)

    image_paths = [record["image_path"] for record in records]
    results = model.predict(
        source=image_paths,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        verbose=False,
        stream=True,
    )

    output: list[dict[str, str]] = []
    for record, result in zip(records, results, strict=True):
        probs_tensor = result.probs.data.detach().cpu()
        probs = [float(x) for x in probs_tensor.tolist()]
        p_defect = clip_probability(probs[defect_index])
        p_normal = clip_probability(probs[normal_index]) if normal_index is not None else clip_probability(1.0 - p_defect)
        y_true = int(record["y_true"])
        y_pred = 1 if p_defect >= p_normal else 0

        enriched = dict(record)
        enriched.update(
            {
                "y_pred_raw": str(y_pred),
                "raw_correct": str(int(y_pred == y_true)),
                "p_defect_raw": f"{p_defect:.10f}",
                "p_normal_raw": f"{p_normal:.10f}",
                "raw_logit": f"{logit(p_defect):.10f}",
                "p_defect_cal": "",
                "p_defect_operational": "",
                "raw_cross_entropy": f"{cross_entropy(y_true, p_defect):.10f}",
                "cal_cross_entropy": "",
                "operational_cross_entropy": "",
                "raw_uncertainty": f"{uncertainty(p_defect):.10f}",
                "cal_uncertainty": "",
                "operational_uncertainty": "",
            }
        )
        output.append(enriched)
    return output


def fit_calibrator(predictions: list[dict[str, str]]) -> Calibrator:
    y = np.array([int(row["y_true"]) for row in predictions], dtype=np.int64)
    if len(set(y.tolist())) != 2:
        raise ValueError("val_cal must contain both positive and negative samples.")
    x = np.array([[logit(safe_float(row["p_defect_raw"]))] for row in predictions], dtype=np.float64)
    model = LogisticRegression(C=1_000_000.0, solver="lbfgs", max_iter=1000)
    model.fit(x, y)
    return Calibrator(
        coef=float(model.coef_[0][0]),
        intercept=float(model.intercept_[0]),
        source_prevalence=float(y.mean()),
        deployment_prevalence=0.0,
    )


def apply_calibration(predictions: list[dict[str, str]], calibrator: Calibrator, deployment_prevalence: float) -> Calibrator:
    if not 0.0 < deployment_prevalence < 1.0:
        raise ValueError("--deployment-defect-prevalence must be between 0 and 1.")
    adjusted = Calibrator(
        coef=calibrator.coef,
        intercept=calibrator.intercept,
        source_prevalence=calibrator.source_prevalence,
        deployment_prevalence=deployment_prevalence,
    )
    prior_shift = logit(deployment_prevalence) - logit(adjusted.source_prevalence)
    for row in predictions:
        y_true = int(row["y_true"])
        raw_logit = logit(safe_float(row["p_defect_raw"]))
        p_cal = clip_probability(sigmoid(adjusted.coef * raw_logit + adjusted.intercept))
        p_operational = clip_probability(sigmoid(logit(p_cal) + prior_shift))
        row["p_defect_cal"] = f"{p_cal:.10f}"
        row["p_defect_operational"] = f"{p_operational:.10f}"
        row["cal_cross_entropy"] = f"{cross_entropy(y_true, p_cal):.10f}"
        row["operational_cross_entropy"] = f"{cross_entropy(y_true, p_operational):.10f}"
        row["cal_uncertainty"] = f"{uncertainty(p_cal):.10f}"
        row["operational_uncertainty"] = f"{uncertainty(p_operational):.10f}"
    return adjusted


def confusion_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (scores >= threshold).astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": int(len(y_true)),
        "positive_n": int((y_true == 1).sum()),
        "negative_n": int((y_true == 0).sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "accuracy": accuracy,
        "f1": f1,
        "fpr": 1.0 - specificity,
        "predicted_positive_rate": float(y_pred.mean()) if len(y_pred) else 0.0,
    }


def prevalence_adjusted_metrics(recall: float, specificity: float, prevalence: float) -> dict[str, float]:
    fpr = 1.0 - specificity
    precision_denominator = prevalence * recall + (1.0 - prevalence) * fpr
    precision = prevalence * recall / precision_denominator if precision_denominator else 0.0
    accuracy = prevalence * recall + (1.0 - prevalence) * specificity
    pass_through_rate = prevalence * recall + (1.0 - prevalence) * fpr
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "weighted_precision": precision,
        "weighted_accuracy": accuracy,
        "weighted_f1": f1,
        "weighted_pass_through_rate": pass_through_rate,
    }


def select_threshold_for_recall(
    predictions: list[dict[str, str]], score_column: str, target_recall: float
) -> tuple[float, dict[str, float | int]]:
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("--target-recall must be in (0, 1].")
    y_true = np.array([int(row["y_true"]) for row in predictions], dtype=np.int64)
    scores = np.array([safe_float(row[score_column]) for row in predictions], dtype=np.float64)
    positive_scores = np.sort(scores[y_true == 1])[::-1]
    if len(positive_scores) == 0:
        raise ValueError("val_op has no positive samples.")
    required_tp = max(1, int(math.ceil(target_recall * len(positive_scores))))
    threshold = float(positive_scores[required_tp - 1])
    return threshold, confusion_metrics(y_true, scores, threshold)


def metrics_for_split(
    split: str,
    predictions: list[dict[str, str]],
    score_column: str,
    threshold: float,
    deployment_prevalence: float,
) -> dict[str, str]:
    y_true = np.array([int(row["y_true"]) for row in predictions], dtype=np.int64)
    scores = np.array([safe_float(row[score_column]) for row in predictions], dtype=np.float64)
    metrics = confusion_metrics(y_true, scores, threshold)
    weighted = prevalence_adjusted_metrics(
        recall=float(metrics["recall"]),
        specificity=float(metrics["specificity"]),
        prevalence=deployment_prevalence,
    )
    row = {
        "split": split,
        "score_column": score_column,
        "threshold": f"{threshold:.10f}",
        "deployment_defect_prevalence": f"{deployment_prevalence:.6f}",
    }
    for key, value in {**metrics, **weighted}.items():
        row[key] = str(value) if isinstance(value, int) else f"{value:.10f}"
    return row


def write_csv(path: Path, rows: list[dict[str, str]], columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def collect_artifact_rows(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    ignored = {ARTIFACT_MANIFEST_CSV_FILENAME, ARTIFACT_MANIFEST_JSON_FILENAME}
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        relative_path = path.relative_to(run_dir).as_posix()
        if relative_path in ignored:
            continue
        stat = path.stat()
        rows.append(
            {
                "relative_path": relative_path,
                "size_bytes": str(stat.st_size),
                "sha256": file_sha256(path),
                "modified_at_local": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
        )
    return rows


def write_artifact_manifest(run_dir: Path) -> tuple[Path, Path]:
    rows = collect_artifact_rows(run_dir)
    csv_path = run_dir / ARTIFACT_MANIFEST_CSV_FILENAME
    json_path = run_dir / ARTIFACT_MANIFEST_JSON_FILENAME
    write_csv(csv_path, rows, ("relative_path", "size_bytes", "sha256", "modified_at_local"))
    write_json(
        json_path,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(run_dir),
            "artifact_count": len(rows),
            "artifacts": rows,
        },
    )
    return csv_path, json_path


def create_run_dir(paths: Paths, cfg: EvalConfig) -> Path:
    run_dir = paths.output_root / cfg.run_name
    if run_dir.exists() and not cfg.exist_ok:
        raise FileExistsError(f"Run directory already exists: {run_dir}. Use --exist-ok to reuse it.")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_readme(run_dir: Path, cfg: EvalConfig, paths: Paths, threshold: float) -> None:
    text = f"""# Stage-1 CLS Gate Evaluation

weights: `{cfg.weights}`
dataset_root: `{paths.dataset_root}`
splits: `{','.join(cfg.splits)}`
seed: `{cfg.seed}`
imgsz: `{cfg.imgsz}`
batch: `{cfg.batch}`
device: `{cfg.device}`
target_recall: `{cfg.target_recall}`
deployment_defect_prevalence: `{cfg.deployment_defect_prevalence}`
selected_threshold_column: `p_defect_operational`
selected_threshold: `{threshold:.10f}`

Files:

- `predictions_*.csv`: per-image predictions with manifest identifiers.
- `calibration.json`: Platt calibration parameters fitted on val_cal.
- `threshold.json`: threshold selected on val_op.
- `metrics_at_selected_threshold.csv`: metrics for each split at the selected threshold.
- `run_config.json`: reproducibility snapshot for this evaluation run.
- `artifact_manifest.csv/json`: output file inventory with size and SHA-256.
"""
    (run_dir / "README.md").write_text(text, encoding="utf-8")


def parse_splits(raw: str) -> tuple[str, ...]:
    splits = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = [split for split in splits if split not in SPLIT_MANIFESTS]
    if unknown:
        raise ValueError(f"Unknown split(s): {unknown}. Valid splits: {tuple(SPLIT_MANIFESTS)}")
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a stage-1 YOLO11-cls binary gate.")
    parser.add_argument(
        "--weights",
        required=True,
        help="Input checkpoint path, usually a trained weights/best.pt or weights/last.pt.",
    )
    parser.add_argument("--run-name", default=None, help="Name of the output run directory under --output-root.")
    parser.add_argument("--splits", default="val_model,val_cal,val_op,test")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("STAGE1_EVAL_SEED", SEED)))
    parser.add_argument("--imgsz", type=int, default=int(os.environ.get("STAGE1_EVAL_IMGSZ", 224)))
    parser.add_argument("--batch", type=int, default=int(os.environ.get("STAGE1_EVAL_BATCH", 64)))
    parser.add_argument("--device", default=os.environ.get("STAGE1_EVAL_DEVICE", "cpu"))
    parser.add_argument("--limit-per-class", type=int, default=None, help="Debug mode: sample N defect and N normal rows per split.")
    parser.add_argument("--target-recall", type=float, default=float(os.environ.get("STAGE1_EVAL_TARGET_RECALL", 0.995)))
    parser.add_argument(
        "--deployment-defect-prevalence",
        type=float,
        default=float(os.environ.get("STAGE1_EVAL_DEPLOYMENT_PREVALENCE", 0.10)),
        help="Target deployment prior used for operational probability and weighted metrics.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only verify manifests and write run_config.json.")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("STAGE1_DATASET_ROOT"),
        help="Final sampled dataset root containing Det/ and manifests/. Defaults to data/final_sewerml_dataset.",
    )
    parser.add_argument(
        "--yolo-root",
        default=os.environ.get("STAGE1_YOLO_ROOT"),
        help="Local YOLOv11 source directory used for importing ultralytics. Defaults to YOLOv11.",
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("STAGE1_EVAL_OUTPUT_ROOT"),
        help="Directory where evaluation runs are written. Defaults to YOLOv11/runs/stage1_cls_eval.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EvalConfig:
    weights = Path(args.weights).resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_run_name = f"eval_{weights.stem}_{timestamp}"
    return EvalConfig(
        weights=weights,
        run_name=args.run_name or default_run_name,
        splits=parse_splits(args.splits),
        seed=args.seed,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        limit_per_class=args.limit_per_class,
        target_recall=args.target_recall,
        deployment_defect_prevalence=args.deployment_defect_prevalence,
        dry_run=args.dry_run,
        exist_ok=args.exist_ok,
    )


def main() -> int:
    args = parse_args()
    paths = build_paths(args)
    cfg = build_config(args)
    run_dir = create_run_dir(paths, cfg)
    started = time.time()

    print(f"run_dir={run_dir}")
    print(f"weights={cfg.weights}")
    print(f"splits={','.join(cfg.splits)}")

    records_by_split = {split: load_split_records(paths, split, cfg) for split in cfg.splits}
    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(paths.repo_root),
        "yolo_root": str(paths.yolo_root),
        "dataset_root": str(paths.dataset_root),
        "manifest_dir": str(paths.manifest_dir),
        "output_root": str(paths.output_root),
        "weights": str(cfg.weights),
        "weights_sha256": file_sha256(cfg.weights),
        "run_name": cfg.run_name,
        "splits": cfg.splits,
        "seed": cfg.seed,
        "imgsz": cfg.imgsz,
        "batch": cfg.batch,
        "device": cfg.device,
        "limit_per_class": cfg.limit_per_class,
        "target_recall": cfg.target_recall,
        "deployment_defect_prevalence": cfg.deployment_defect_prevalence,
        "split_counts": {split: len(rows) for split, rows in records_by_split.items()},
    }
    write_json(run_dir / "run_config.json", run_config)

    if cfg.dry_run:
        write_artifact_manifest(run_dir)
        print("dry_run=true; prediction skipped")
        return 0

    YOLO = ensure_yolo_import(paths.yolo_root)
    model = YOLO(str(cfg.weights))

    predictions_by_split: dict[str, list[dict[str, str]]] = {}
    for split, records in records_by_split.items():
        print(f"predict split={split} images={len(records)}")
        predictions_by_split[split] = predict_records(model, records, cfg)

    missing_required = [split for split in ("val_cal", "val_op", "test") if split not in predictions_by_split]
    if missing_required:
        raise ValueError(f"Calibration/threshold/test require these missing splits: {missing_required}")

    calibrator = fit_calibrator(predictions_by_split["val_cal"])
    adjusted_calibrator = apply_calibration(
        [row for rows in predictions_by_split.values() for row in rows],
        calibrator,
        cfg.deployment_defect_prevalence,
    )

    score_column = "p_defect_operational"
    threshold, val_op_metrics = select_threshold_for_recall(
        predictions_by_split["val_op"],
        score_column,
        cfg.target_recall,
    )

    for split, rows in predictions_by_split.items():
        write_csv(run_dir / f"predictions_{split}.csv", rows, PREDICTION_COLUMNS)

    metrics_rows = [
        metrics_for_split(split, rows, score_column, threshold, cfg.deployment_defect_prevalence)
        for split, rows in predictions_by_split.items()
    ]
    metric_columns = list(metrics_rows[0].keys()) if metrics_rows else []
    write_csv(run_dir / "metrics_at_selected_threshold.csv", metrics_rows, metric_columns)

    write_json(
        run_dir / "calibration.json",
        {
            "method": "Platt scaling on logit(p_defect_raw)",
            "fit_split": "val_cal",
            "coef": adjusted_calibrator.coef,
            "intercept": adjusted_calibrator.intercept,
            "source_prevalence": adjusted_calibrator.source_prevalence,
            "deployment_defect_prevalence": adjusted_calibrator.deployment_prevalence,
            "prior_adjustment": logit(adjusted_calibrator.deployment_prevalence)
            - logit(adjusted_calibrator.source_prevalence),
        },
    )
    write_json(
        run_dir / "threshold.json",
        {
            "selection_split": "val_op",
            "score_column": score_column,
            "target_recall": cfg.target_recall,
            "selected_threshold": threshold,
            "val_op_metrics": val_op_metrics,
        },
    )
    write_readme(run_dir, cfg, paths, threshold)
    artifact_manifest_csv, artifact_manifest_json = write_artifact_manifest(run_dir)
    print(f"artifact_manifest_csv={artifact_manifest_csv}")
    print(f"artifact_manifest_json={artifact_manifest_json}")
    print(f"metrics={run_dir / 'metrics_at_selected_threshold.csv'}")
    print(f"duration_sec={time.time() - started:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
