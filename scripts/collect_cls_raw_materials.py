from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import yaml
from PIL import Image

from pipeline_common import REPO_ROOT, YOLOV11_ROOT, ensure_yolov11_importable, load_json_config, resolve_relative_path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_SOURCE_CONFIG = "YOLOv11/configs/runtime/cls_source_cls6.json"
DEFAULT_THRESHOLDS = [round(0.01 * idx, 2) for idx in range(1, 100)]
DATASET_FIELDNAMES = [
    "split",
    "label",
    "image_id",
    "relative_path",
    "absolute_path",
    "file_size_bytes",
    "width",
    "height",
    "aspect_ratio",
    "sha1",
]
PREDICTION_FIELDNAMES = [
    "row_id",
    "img_id",
    "split",
    "gt_label",
    "gt_index",
    "img_rel_path",
    "img_path",
    "pred_label",
    "pred_index",
    "correct",
    "top5_correct",
    "top1_prob",
    "top2_label",
    "top2_prob",
    "top3_label",
    "top3_prob",
    "margin",
    "entropy",
    "entropy_norm",
    "p_normal",
    "p_abnormal",
    "abnormal_conf",
    "width",
    "height",
    "aspect_ratio",
    "embedding_index",
    "top5_labels_json",
    "top5_probs_json",
    "probs_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect raw classification experiment materials after training so later analysis does not require reruns."
    )
    parser.add_argument("--config", default=DEFAULT_SOURCE_CONFIG, help="Classification config JSON.")
    parser.add_argument("--weights", default="", help="Override best.pt weights path.")
    parser.add_argument("--run-dir", default="", help="Override run directory. Defaults to weights parent run.")
    parser.add_argument("--data", default="", help="Override dataset root.")
    parser.add_argument("--output-dir", default="", help="Override raw-material output directory.")
    parser.add_argument("--normal-class", default="Normal", help="Class treated as normal for binary gate analysis.")
    parser.add_argument("--device", default="0", help="Inference device for raw prediction export.")
    parser.add_argument("--batch", type=int, default=16, help="Prediction batch size. Must be between 1 and 32.")
    parser.add_argument("--imgsz", type=int, default=-1, help="Override inference image size.")
    parser.add_argument("--skip-file-hash", action="store_true", help="Skip SHA1 file hashing for dataset manifests.")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip validation embedding export.")
    return parser.parse_args()


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def require_batch_limit(batch: int) -> None:
    if batch <= 0 or batch > 32:
        raise SystemExit(f"Batch must be between 1 and 32, got {batch}.")


def resolve_config_path(path: str) -> Path:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    return config_path.resolve()


def load_run_args(run_dir: Path) -> dict[str, Any]:
    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        return {}
    with args_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def infer_run_dir(cfg: dict[str, Any], run_args: dict[str, Any], override: str, weights_override: str) -> Path:
    if override:
        return Path(override).resolve()
    if weights_override:
        return Path(weights_override).resolve().parents[1]
    save_dir = str(run_args.get("save_dir") or "").strip()
    if save_dir:
        return Path(save_dir).resolve()
    project = resolve_relative_path(cfg.get("project"), YOLOV11_ROOT)
    name = str(cfg.get("name") or "").strip()
    if not project or not name:
        raise SystemExit("Could not infer run directory; pass --run-dir or --weights explicitly.")
    return (Path(project) / name).resolve()


def infer_weights_path(cfg: dict[str, Any], run_args: dict[str, Any], run_dir: Path, override: str) -> Path:
    if override:
        return Path(override).resolve()
    candidate = run_dir / "weights" / "best.pt"
    if candidate.exists():
        return candidate
    model_path = str(run_args.get("model") or "").strip()
    if model_path.lower().endswith(".pt") and Path(model_path).exists():
        return Path(model_path).resolve()
    project = resolve_relative_path(cfg.get("project"), YOLOV11_ROOT)
    name = str(cfg.get("name") or "").strip()
    if project and name:
        return (Path(project) / name / "weights" / "best.pt").resolve()
    raise SystemExit("Could not infer weights path.")


def infer_data_root(cfg: dict[str, Any], run_args: dict[str, Any], override: str) -> Path:
    if override:
        return Path(override).resolve()
    data_value = run_args.get("data") or cfg.get("data")
    resolved = resolve_relative_path(data_value, YOLOV11_ROOT)
    if not resolved:
        raise SystemExit("Could not infer dataset root from args.yaml or config.")
    return Path(resolved).resolve()


def infer_imgsz(cfg: dict[str, Any], run_args: dict[str, Any], override: int) -> int:
    if override > 0:
        return override
    value = run_args.get("imgsz", cfg.get("imgsz", 640))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 640


def infer_output_dir(run_dir: Path, override: str) -> Path:
    if override:
        return Path(override).resolve()
    return (REPO_ROOT / "research" / "materials" / run_dir.name).resolve()


def class_names_from_model_names(names: dict[int, str] | list[str]) -> list[str]:
    if isinstance(names, dict):
        return [str(names[idx]) for idx in sorted(int(key) for key in names)]
    return [str(name) for name in names]


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def image_stats(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def collect_dataset_rows(data_root: Path, skip_hash: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        split_root = data_root / split
        if not split_root.exists():
            continue
        images = sorted(path for path in split_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            continue
        print_step("dataset", f"scanning {split}: {len(images)} images")
        for image_path in images:
            width, height = image_stats(image_path)
            rows.append(
                {
                    "split": split,
                    "label": image_path.parent.name,
                    "image_id": image_path.stem,
                    "relative_path": image_path.relative_to(data_root).as_posix(),
                    "absolute_path": str(image_path.resolve()),
                    "file_size_bytes": image_path.stat().st_size,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / height, 6) if height else None,
                    "sha1": "" if skip_hash else sha1_file(image_path),
                }
            )
    if not rows:
        raise SystemExit(f"No dataset images found under {data_root}")
    return rows


def summary_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "p90": None}
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.9) - 1))
    return {
        "count": len(ordered),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
        "mean": round(statistics.fmean(ordered), 6),
        "median": round(statistics.median(ordered), 6),
        "p90": round(ordered[p90_index], 6),
    }


def build_dataset_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts = Counter(str(row["split"]) for row in rows)
    class_counts = Counter(str(row["label"]) for row in rows)
    split_class_counts: dict[str, dict[str, int]] = defaultdict(dict)
    per_split_values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        split = str(row["split"])
        label = str(row["label"])
        split_class_counts[split][label] = split_class_counts[split].get(label, 0) + 1
        per_split_values[split].append(row)

    width_values = [float(row["width"]) for row in rows]
    height_values = [float(row["height"]) for row in rows]
    aspect_values = [float(row["aspect_ratio"]) for row in rows if row["aspect_ratio"] is not None]
    extreme_ratio_count = sum(1 for value in aspect_values if value > 2.0 or value < 0.5)

    split_stats: dict[str, Any] = {}
    for split, split_rows in per_split_values.items():
        split_stats[split] = {
            "count": len(split_rows),
            "class_counts": dict(sorted(split_class_counts[split].items())),
            "width": summary_stats([float(row["width"]) for row in split_rows]),
            "height": summary_stats([float(row["height"]) for row in split_rows]),
            "aspect_ratio": summary_stats([float(row["aspect_ratio"]) for row in split_rows if row["aspect_ratio"] is not None]),
        }

    return {
        "total_images": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "split_class_counts": {key: dict(sorted(value.items())) for key, value in sorted(split_class_counts.items())},
        "width": summary_stats(width_values),
        "height": summary_stats(height_values),
        "aspect_ratio": summary_stats(aspect_values),
        "extreme_aspect_ratio_count": extreme_ratio_count,
        "extreme_aspect_ratio_rule": "aspect_ratio > 2.0 or aspect_ratio < 0.5",
        "per_split": split_stats,
    }


def build_dataset_stats_with_gate(rows: list[dict[str, Any]], normal_class: str) -> dict[str, Any]:
    stats = build_dataset_stats(rows)
    total_normal = sum(1 for row in rows if str(row["label"]) == normal_class)
    total_abnormal = len(rows) - total_normal
    binary_per_split: dict[str, dict[str, int]] = {}
    for split in sorted({str(row["split"]) for row in rows}):
        split_rows = [row for row in rows if str(row["split"]) == split]
        split_normal = sum(1 for row in split_rows if str(row["label"]) == normal_class)
        binary_per_split[split] = {
            "normal": split_normal,
            "abnormal": len(split_rows) - split_normal,
        }
    stats["binary_gate_counts"] = {
        "normal_class": normal_class,
        "normal": total_normal,
        "abnormal": total_abnormal,
    }
    stats["binary_gate_per_split"] = binary_per_split
    return stats


def normalize_existing_epoch_metrics(run_dir: Path) -> list[dict[str, Any]]:
    epoch_metrics_path = run_dir / "epoch_metrics.csv"
    if epoch_metrics_path.exists():
        with epoch_metrics_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    results_path = run_dir / "results.csv"
    if not results_path.exists():
        return []

    with results_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_rows = [{key.strip(): value.strip() for key, value in row.items()} for row in reader]

    normalized: list[dict[str, Any]] = []
    best_top1 = -1.0
    best_epoch = 0
    for row in raw_rows:
        epoch = int(float(row.get("epoch", "0") or 0))
        top1 = safe_float(row.get("metrics/accuracy_top1"))
        if top1 is not None and top1 >= best_top1:
            best_top1 = top1
            best_epoch = epoch
        normalized.append(
            {
                "epoch": epoch,
                "train_loss": safe_float(row.get("train/loss")),
                "val_loss": safe_float(row.get("val/loss")),
                "top1": top1,
                "top5": safe_float(row.get("metrics/accuracy_top5")),
                "lr_pg0": safe_float(row.get("lr/pg0")),
                "lr_pg1": safe_float(row.get("lr/pg1")),
                "lr_pg2": safe_float(row.get("lr/pg2")),
                "epoch_time_seconds": None,
                "elapsed_seconds": None,
                "batch_size": None,
                "accumulate": None,
                "effective_batch": None,
                "gpu_mem_reserved_gb": None,
                "gpu_mem_peak_reserved_gb": None,
                "gpu_mem_peak_allocated_gb": None,
                "fitness": None,
                "best_fitness_so_far": None,
                "best_epoch_so_far": best_epoch,
                "possible_early_stop": None,
                "stop_triggered": None,
                "is_best_epoch": int(best_epoch == epoch),
            }
        )
    return normalized


def chunked(items: list[Path], size: int) -> list[list[Path]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def compute_entropy(probs: list[float]) -> tuple[float, float]:
    eps = 1e-12
    entropy = -sum(prob * math.log(max(prob, eps)) for prob in probs)
    max_entropy = math.log(len(probs)) if probs else 1.0
    entropy_norm = entropy / max_entropy if max_entropy else 0.0
    return entropy, entropy_norm


def collect_validation_predictions(
    weights_path: Path,
    data_root: Path,
    normal_class: str,
    device: str,
    batch: int,
    imgsz: int,
    collect_embeddings: bool,
) -> tuple[list[str], list[dict[str, Any]], np.ndarray | None, dict[str, float]]:
    ensure_yolov11_importable()
    from ultralytics import YOLO

    val_root = data_root / "val"
    image_paths = sorted(path for path in val_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise SystemExit(f"No validation images found under {val_root}")

    predict_model = YOLO(str(weights_path), task="classify")
    embed_model = YOLO(str(weights_path), task="classify") if collect_embeddings else None
    class_names = class_names_from_model_names(predict_model.names)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    normal_index = class_to_index.get(normal_class)

    rows: list[dict[str, Any]] = []
    embedding_vectors: list[np.ndarray] = []
    predict_seconds = 0.0
    embed_seconds = 0.0
    embedding_failed = False

    print_step("predict", f"exporting validation predictions: {len(image_paths)} images")
    for batch_paths in chunked(image_paths, batch):
        predict_start = time.time()
        batch_results = predict_model.predict(
            source=[str(path) for path in batch_paths],
            verbose=False,
            stream=False,
            batch=len(batch_paths),
            device=device,
            imgsz=imgsz,
        )
        predict_seconds += time.time() - predict_start

        batch_embeddings: list[np.ndarray] | None = None
        if collect_embeddings and not embedding_failed:
            try:
                embed_start = time.time()
                batch_embed_results = embed_model.embed(
                    source=[str(path) for path in batch_paths],
                    stream=False,
                    batch=len(batch_paths),
                    device=device,
                    imgsz=imgsz,
                )
                embed_seconds += time.time() - embed_start
                batch_embeddings = [tensor.detach().cpu().numpy() for tensor in batch_embed_results]
            except Exception as exc:
                embedding_failed = True
                batch_embeddings = None
                print_step("warn", f"embedding export disabled after failure: {exc}")

        for local_index, (path, result) in enumerate(zip(batch_paths, batch_results, strict=True)):
            probs = [float(value) for value in result.probs.data.detach().cpu().tolist()]
            top_indices = sorted(range(len(probs)), key=lambda idx: probs[idx], reverse=True)
            top5_indices = top_indices[: min(5, len(top_indices))]
            entropy, entropy_norm = compute_entropy(probs)
            gt_label = path.parent.name
            gt_index = class_to_index.get(gt_label, -1)
            top1_index = top_indices[0]
            top2_index = top_indices[1] if len(top_indices) > 1 else top_indices[0]
            top3_index = top_indices[2] if len(top_indices) > 2 else top_indices[-1]
            p_normal = probs[normal_index] if normal_index is not None else None
            abnormal_conf = (1.0 - p_normal) if p_normal is not None else None

            embedding_index = None
            if batch_embeddings is not None:
                embedding_vectors.append(batch_embeddings[local_index])
                embedding_index = len(embedding_vectors) - 1

            height, width = result.orig_shape
            rows.append(
                {
                    "row_id": len(rows),
                    "img_id": path.stem,
                    "split": "val",
                    "gt_label": gt_label,
                    "gt_index": gt_index,
                    "img_rel_path": path.relative_to(data_root).as_posix(),
                    "img_path": str(path.resolve()),
                    "pred_label": class_names[top1_index],
                    "pred_index": top1_index,
                    "correct": int(class_names[top1_index] == gt_label),
                    "top5_correct": int(gt_index in top5_indices),
                    "top1_prob": probs[top1_index],
                    "top2_label": class_names[top2_index],
                    "top2_prob": probs[top2_index],
                    "top3_label": class_names[top3_index],
                    "top3_prob": probs[top3_index],
                    "margin": probs[top1_index] - probs[top2_index],
                    "entropy": entropy,
                    "entropy_norm": entropy_norm,
                    "p_normal": p_normal,
                    "p_abnormal": abnormal_conf,
                    "abnormal_conf": abnormal_conf,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / height, 6) if height else None,
                    "embedding_index": embedding_index,
                    "top5_labels_json": json.dumps([class_names[idx] for idx in top5_indices], ensure_ascii=False),
                    "top5_probs_json": json.dumps([probs[idx] for idx in top5_indices], ensure_ascii=False),
                    "probs_json": json.dumps({class_names[idx]: probs[idx] for idx in range(len(class_names))}, ensure_ascii=False),
                }
            )

    embeddings_array = np.stack(embedding_vectors) if embedding_vectors else None
    runtime = {
        "predict_seconds": round(predict_seconds, 6),
        "predict_images_per_second": round(len(image_paths) / predict_seconds, 6) if predict_seconds else None,
        "embed_seconds": round(embed_seconds, 6) if embedding_vectors else None,
        "embed_images_per_second": round(len(image_paths) / embed_seconds, 6) if embed_seconds else None,
    }
    return class_names, rows, embeddings_array, runtime


def build_confusion_matrix(class_names: list[str], prediction_rows: list[dict[str, Any]]) -> np.ndarray:
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for row in prediction_rows:
        gt_index = int(row["gt_index"])
        pred_index = int(row["pred_index"])
        if gt_index >= 0 and pred_index >= 0:
            matrix[gt_index, pred_index] += 1
    return matrix


def per_class_metrics(class_names: list[str], matrix: np.ndarray) -> dict[str, dict[str, Any]]:
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    metrics: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(class_names):
        tp = int(matrix[index, index])
        fp = int(predicted[index] - tp)
        fn = int(support[index] - tp)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        metrics[name] = {
            "support": int(support[index]),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    return metrics


def compute_macro_weighted_summary(class_names: list[str], metrics_by_class: dict[str, dict[str, Any]]) -> dict[str, float]:
    supports = np.array([metrics_by_class[name]["support"] for name in class_names], dtype=float)
    weights = supports / supports.sum() if supports.sum() else np.zeros_like(supports)
    precision_values = np.array([metrics_by_class[name]["precision"] for name in class_names], dtype=float)
    recall_values = np.array([metrics_by_class[name]["recall"] for name in class_names], dtype=float)
    f1_values = np.array([metrics_by_class[name]["f1"] for name in class_names], dtype=float)
    return {
        "macro_precision": round(float(precision_values.mean()) if len(precision_values) else 0.0, 6),
        "macro_recall": round(float(recall_values.mean()) if len(recall_values) else 0.0, 6),
        "macro_f1": round(float(f1_values.mean()) if len(f1_values) else 0.0, 6),
        "weighted_precision": round(float((precision_values * weights).sum()) if len(weights) else 0.0, 6),
        "weighted_recall": round(float((recall_values * weights).sum()) if len(weights) else 0.0, 6),
        "weighted_f1": round(float((f1_values * weights).sum()) if len(weights) else 0.0, 6),
        "balanced_accuracy": round(float(recall_values.mean()) if len(recall_values) else 0.0, 6),
    }


def compute_threshold_rows(prediction_rows: list[dict[str, Any]], normal_class: str, thresholds: list[float]) -> list[dict[str, Any]]:
    scored = [
        {
            "abnormal_conf": float(row["abnormal_conf"]) if row["abnormal_conf"] is not None else None,
            "is_abnormal": row["gt_label"] != normal_class,
        }
        for row in prediction_rows
        if row["abnormal_conf"] is not None
    ]
    return compute_binary_threshold_rows(scored, thresholds)


def compute_binary_threshold_rows(scored: list[dict[str, Any]], thresholds: list[float]) -> list[dict[str, Any]]:
    actual_abnormal = sum(1 for row in scored if row["is_abnormal"])
    actual_normal = len(scored) - actual_abnormal
    output: list[dict[str, Any]] = []
    for threshold in thresholds:
        tp = fn = fp = tn = 0
        for row in scored:
            pred_abnormal = float(row["abnormal_conf"]) >= threshold
            if row["is_abnormal"] and pred_abnormal:
                tp += 1
            elif row["is_abnormal"] and not pred_abnormal:
                fn += 1
            elif not row["is_abnormal"] and pred_abnormal:
                fp += 1
            else:
                tn += 1

        recall = safe_div(tp, actual_abnormal)
        precision = safe_div(tp, tp + fp)
        specificity = safe_div(tn, actual_normal)
        accuracy = safe_div(tp + tn, len(scored))
        f1 = safe_div(2 * precision * recall, precision + recall)
        fpr = safe_div(fp, actual_normal)
        fnr = safe_div(fn, actual_abnormal)
        npv = safe_div(tn, tn + fn)
        ptr = safe_div(tp + fp, len(scored))
        output.append(
            {
                "threshold": round(threshold, 6),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "recall": round(recall, 6),
                "precision": round(precision, 6),
                "specificity": round(specificity, 6),
                "fpr": round(fpr, 6),
                "fnr": round(fnr, 6),
                "npv": round(npv, 6),
                "accuracy": round(accuracy, 6),
                "f1": round(f1, 6),
                "ptr": round(ptr, 6),
                "normal_filtered": tn,
                "normal_left": fp,
                "abnormal_kept": tp,
                "abnormal_missed": fn,
            }
        )
    return output


def exact_curve_thresholds(scores: list[float]) -> list[float]:
    if not scores:
        return []
    unique_scores = sorted({round(float(score), 12) for score in scores}, reverse=True)
    return [1.01, *unique_scores, -0.01]


def build_curve_rows(prediction_rows: list[dict[str, Any]], normal_class: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scored = [
        {
            "abnormal_conf": float(row["abnormal_conf"]) if row["abnormal_conf"] is not None else None,
            "is_abnormal": row["gt_label"] != normal_class,
        }
        for row in prediction_rows
        if row["abnormal_conf"] is not None
    ]
    scores = [float(row["abnormal_conf"]) for row in scored if row["abnormal_conf"] is not None]
    curve_rows = compute_binary_threshold_rows(scored, exact_curve_thresholds(scores))
    roc_rows = [
        {
            "threshold": row["threshold"],
            "tpr": row["recall"],
            "recall": row["recall"],
            "fpr": row["fpr"],
            "specificity": row["specificity"],
            "tp": row["tp"],
            "fn": row["fn"],
            "fp": row["fp"],
            "tn": row["tn"],
        }
        for row in curve_rows
    ]
    pr_rows = [
        {
            "threshold": row["threshold"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
            "tp": row["tp"],
            "fn": row["fn"],
            "fp": row["fp"],
            "tn": row["tn"],
        }
        for row in curve_rows
    ]
    return roc_rows, pr_rows


def build_calibration_rows(prediction_rows: list[dict[str, Any]], normal_class: str, bins: int = 10) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored = [
        (float(row["abnormal_conf"]), 0 if row["gt_label"] == normal_class else 1)
        for row in prediction_rows
        if row["abnormal_conf"] is not None
    ]
    if not scored:
        return [], {"bins": bins, "ece": 0.0, "mce": 0.0, "brier_score": 0.0}

    total = len(scored)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    for idx in range(bins):
        lower = idx / bins
        upper = (idx + 1) / bins
        members = [
            (score, label)
            for score, label in scored
            if (lower <= score < upper) or (idx == bins - 1 and lower <= score <= upper)
        ]
        if members:
            avg_conf = statistics.fmean(score for score, _ in members)
            positive_rate = statistics.fmean(label for _, label in members)
            gap = abs(avg_conf - positive_rate)
            ece += (len(members) / total) * gap
            mce = max(mce, gap)
        else:
            avg_conf = None
            positive_rate = None
            gap = None
        rows.append(
            {
                "bin_index": idx,
                "lower": round(lower, 6),
                "upper": round(upper, 6),
                "count": len(members),
                "avg_confidence": round(avg_conf, 6) if avg_conf is not None else None,
                "empirical_positive_rate": round(positive_rate, 6) if positive_rate is not None else None,
                "gap": round(gap, 6) if gap is not None else None,
            }
        )

    brier_score = statistics.fmean((score - label) ** 2 for score, label in scored)
    summary = {
        "bins": bins,
        "ece": round(ece, 6),
        "mce": round(mce, 6),
        "brier_score": round(brier_score, 6),
        "positive_class": "abnormal",
        "normal_class": normal_class,
    }
    return rows, summary


def select_threshold_row(rows: list[dict[str, Any]], target: float) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: abs(float(row["threshold"]) - target))


def choose_best_row(rows: list[dict[str, Any]], field: str, minimum: float) -> dict[str, Any] | None:
    candidates = [row for row in rows if float(row[field]) >= minimum]
    if not candidates:
        return None
    if field == "recall":
        return max(candidates, key=lambda row: (float(row["specificity"]), float(row["precision"]), -float(row["threshold"])))
    return max(candidates, key=lambda row: (float(row["recall"]), float(row["precision"]), -float(row["threshold"])))


def auc_from_xy(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    points = sorted(zip(xs, ys), key=lambda item: item[0])
    area = 0.0
    for idx in range(1, len(points)):
        x0, y0 = points[idx - 1]
        x1, y1 = points[idx]
        area += (x1 - x0) * (y0 + y1) * 0.5
    return area


def binary_scores_and_labels(prediction_rows: list[dict[str, Any]], normal_class: str) -> tuple[list[float], list[int]]:
    pairs = [
        (float(row["abnormal_conf"]), 0 if row["gt_label"] == normal_class else 1)
        for row in prediction_rows
        if row["abnormal_conf"] is not None
    ]
    return [score for score, _ in pairs], [label for _, label in pairs]


def binary_auroc(scores: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0

    indexed = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(indexed):
        end = start
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = (start + 1 + end) * 0.5
        for idx in range(start, end):
            ranks[indexed[idx][0]] = average_rank
        start = end
    positive_rank_sum = sum(ranks[idx] for idx, label in enumerate(labels) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) * 0.5) / (positives * negatives)


def binary_average_precision(scores: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    last_recall = 0.0
    area = 0.0
    for _, label in ordered:
        if label == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        area += (recall - last_recall) * precision
        last_recall = recall
    return area


def build_threshold_summary(rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]], normal_class: str) -> dict[str, Any]:
    if not rows:
        return {}
    best_f1 = max(rows, key=lambda row: float(row["f1"]))
    best_recall = max(rows, key=lambda row: float(row["recall"]))
    best_accuracy = max(rows, key=lambda row: float(row["accuracy"]))
    scores, labels = binary_scores_and_labels(prediction_rows, normal_class)
    roc_auc = binary_auroc(scores, labels)
    pr_auc = binary_average_precision(scores, labels)
    operating_points = {
        "recall_ge_99_5": choose_best_row(rows, "recall", 0.995),
        "recall_ge_99_0": choose_best_row(rows, "recall", 0.990),
        "recall_ge_98_0": choose_best_row(rows, "recall", 0.980),
        "specificity_ge_60": choose_best_row(rows, "specificity", 0.60),
        "specificity_ge_70": choose_best_row(rows, "specificity", 0.70),
        "specificity_ge_75": choose_best_row(rows, "specificity", 0.75),
    }
    return {
        "best_f1": best_f1,
        "best_recall": best_recall,
        "best_accuracy": best_accuracy,
        "auroc_exact": round(roc_auc, 6),
        "average_precision_exact": round(pr_auc, 6),
        "operating_points": operating_points,
    }


def write_confusion_matrix_csv(path: Path, class_names: list[str], matrix: np.ndarray) -> None:
    fieldnames = ["actual_label", *class_names]
    rows = []
    for index, name in enumerate(class_names):
        row = {"actual_label": name}
        for pred_index, pred_name in enumerate(class_names):
            value = matrix[index, pred_index]
            row[pred_name] = round(float(value), 6) if matrix.dtype.kind == "f" else int(value)
        rows.append(row)
    write_csv(path, fieldnames, rows)


def build_confused_pairs(class_names: list[str], matrix: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    support = matrix.sum(axis=1)
    for gt_index, gt_name in enumerate(class_names):
        for pred_index, pred_name in enumerate(class_names):
            if gt_index == pred_index:
                continue
            count = int(matrix[gt_index, pred_index])
            if count <= 0:
                continue
            rows.append(
                {
                    "gt_label": gt_name,
                    "pred_label": pred_name,
                    "count": count,
                    "rate_within_gt": round(safe_div(count, int(support[gt_index])), 6),
                }
            )
    rows.sort(key=lambda row: (int(row["count"]), float(row["rate_within_gt"])), reverse=True)
    return rows


def write_embeddings(output_dir: Path, prediction_rows: list[dict[str, Any]], embeddings: np.ndarray | None) -> None:
    if embeddings is None:
        return
    np.save(output_dir / "val_embeddings.npy", embeddings)
    index_rows = [
        {
            "embedding_index": row["embedding_index"],
            "row_id": row["row_id"],
            "img_id": row["img_id"],
            "img_rel_path": row["img_rel_path"],
            "gt_label": row["gt_label"],
        }
        for row in prediction_rows
        if row["embedding_index"] is not None
    ]
    write_csv(output_dir / "val_embeddings_index.csv", ["embedding_index", "row_id", "img_id", "img_rel_path", "gt_label"], index_rows)


def build_env_info() -> tuple[dict[str, Any], str]:
    ensure_yolov11_importable()
    import torch
    from ultralytics import __version__ as ultralytics_version

    gpu_info: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpu_info.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 6),
                    "multi_processor_count": props.multi_processor_count,
                }
            )

    driver_version = None
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        driver_version = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else None
    except Exception:
        driver_version = None

    git_commit = None
    try:
        git_commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        git_commit = None

    try:
        pip_freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except Exception as exc:
        pip_freeze = f"# pip freeze unavailable: {exc}\n"

    env_info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "ultralytics_version": ultralytics_version,
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 6),
        "gpu": gpu_info,
        "nvidia_driver_version": driver_version,
        "git_commit": git_commit,
    }
    return env_info, pip_freeze


def build_model_profile(weights_path: Path, run_dir: Path, imgsz: int) -> dict[str, Any]:
    ensure_yolov11_importable()
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_gradients, get_num_params

    model = YOLO(str(weights_path), task="classify")
    raw_model = model.model
    last_path = run_dir / "weights" / "last.pt"
    return {
        "weights_path": str(weights_path),
        "last_weights_path": str(last_path),
        "model_name": str(getattr(raw_model, "yaml", {}).get("yaml_file") or getattr(model, "model_name", "")),
        "layers": len(list(raw_model.modules())),
        "parameters": int(get_num_params(raw_model)),
        "trainable_parameters": int(get_num_gradients(raw_model)),
        "gflops": round(float(get_flops(raw_model, imgsz=imgsz)), 6),
        "best_weights_size_mb": round(weights_path.stat().st_size / 1e6, 6) if weights_path.exists() else None,
        "last_weights_size_mb": round(last_path.stat().st_size / 1e6, 6) if last_path.exists() else None,
        "fused": bool(getattr(raw_model, "is_fused", lambda: False)()),
    }


def build_visual_sample_candidates(prediction_rows: list[dict[str, Any]], normal_class: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_class_correct: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_class_wrong: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        if int(row["correct"]) == 1:
            per_class_correct[str(row["gt_label"])].append(row)
        else:
            per_class_wrong[str(row["gt_label"])].append(row)

    for class_name, candidates in per_class_correct.items():
        high_conf = sorted(candidates, key=lambda row: float(row["top1_prob"]), reverse=True)[:20]
        low_conf = sorted(candidates, key=lambda row: float(row["top1_prob"]))[:20]
        for row in high_conf:
            rows.append({"group": "correct_high_conf", "class_name": class_name, "img_rel_path": row["img_rel_path"], "score": row["top1_prob"]})
        for row in low_conf:
            rows.append({"group": "correct_low_conf", "class_name": class_name, "img_rel_path": row["img_rel_path"], "score": row["top1_prob"]})

    for class_name, candidates in per_class_wrong.items():
        high_conf = sorted(candidates, key=lambda row: float(row["top1_prob"]), reverse=True)[:20]
        for row in high_conf:
            rows.append({"group": "wrong_high_conf", "class_name": class_name, "img_rel_path": row["img_rel_path"], "score": row["top1_prob"]})

    normal_candidates = [row for row in prediction_rows if row["gt_label"] == normal_class]
    abnormal_candidates = [row for row in prediction_rows if row["gt_label"] != normal_class]
    for row in sorted(normal_candidates, key=lambda item: float(item["abnormal_conf"] or 0.0), reverse=True)[:20]:
        rows.append({"group": "normal_looks_abnormal", "class_name": normal_class, "img_rel_path": row["img_rel_path"], "score": row["abnormal_conf"]})
    for row in sorted(abnormal_candidates, key=lambda item: float(item["abnormal_conf"] or 0.0))[:20]:
        rows.append({"group": "abnormal_looks_normal", "class_name": str(row["gt_label"]), "img_rel_path": row["img_rel_path"], "score": row["abnormal_conf"]})
    return rows


def copy_run_supporting_files(run_dir: Path, output_dir: Path) -> dict[str, str]:
    artifacts_dir = output_dir / "raw_run_artifacts"
    copied: dict[str, str] = {}
    candidates = {
        "args_yaml": run_dir / "args.yaml",
        "results_csv": run_dir / "results.csv",
        "training_runtime_json": run_dir / "training_runtime.json",
        "results_png": run_dir / "results.png",
        "confusion_matrix_png": run_dir / "confusion_matrix.png",
        "confusion_matrix_normalized_png": run_dir / "confusion_matrix_normalized.png",
    }
    for name, src in candidates.items():
        copied_path = copy_if_exists(src, artifacts_dir / src.name)
        if copied_path is not None:
            copied[name] = copied_path
    return copied


def upsert_run_master(path: Path, row: dict[str, Any]) -> None:
    fieldnames = list(row.keys())
    existing: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
        existing = [item for item in existing if item.get("run_name") != row["run_name"]]
    existing.append({key: row.get(key, "") for key in fieldnames})
    write_csv(path, fieldnames, existing)


def main() -> None:
    args = parse_args()
    require_batch_limit(args.batch)

    config_path = resolve_config_path(args.config)
    cfg = load_json_config(config_path)
    temp_run_dir = infer_run_dir(cfg, {}, args.run_dir, args.weights)
    run_args = load_run_args(temp_run_dir)
    run_dir = infer_run_dir(cfg, run_args, args.run_dir, args.weights)
    weights_path = infer_weights_path(cfg, run_args, run_dir, args.weights)
    data_root = infer_data_root(cfg, run_args, args.data)
    imgsz = infer_imgsz(cfg, run_args, args.imgsz)
    output_dir = infer_output_dir(run_dir, args.output_dir)

    print_step("run", f"run_dir={run_dir}")
    print_step("run", f"weights={weights_path}")
    print_step("run", f"data={data_root}")
    print_step("run", f"output={output_dir}")

    dataset_rows = collect_dataset_rows(data_root, skip_hash=args.skip_file_hash)
    write_csv(output_dir / "dataset_manifest.csv", DATASET_FIELDNAMES, dataset_rows)
    for split in ("train", "val", "test"):
        split_rows = [row for row in dataset_rows if row["split"] == split]
        if split_rows:
            write_csv(output_dir / f"split_{split}.csv", DATASET_FIELDNAMES, split_rows)
    write_json(output_dir / "dataset_stats.json", build_dataset_stats_with_gate(dataset_rows, args.normal_class))

    epoch_rows = normalize_existing_epoch_metrics(run_dir)
    if epoch_rows:
        write_csv(output_dir / "epoch_metrics.csv", list(epoch_rows[0].keys()), epoch_rows)

    class_names, prediction_rows, embeddings, collect_runtime = collect_validation_predictions(
        weights_path=weights_path,
        data_root=data_root,
        normal_class=args.normal_class,
        device=args.device,
        batch=args.batch,
        imgsz=imgsz,
        collect_embeddings=not args.no_embeddings,
    )
    write_csv(output_dir / "val_predictions.csv", PREDICTION_FIELDNAMES, prediction_rows)
    write_embeddings(output_dir, prediction_rows, embeddings)

    matrix = build_confusion_matrix(class_names, prediction_rows)
    normalized_matrix = matrix.astype(float)
    row_sums = normalized_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(normalized_matrix, row_sums, out=np.zeros_like(normalized_matrix), where=row_sums != 0)
    write_confusion_matrix_csv(output_dir / "confusion_matrix.csv", class_names, matrix)
    write_confusion_matrix_csv(output_dir / "confusion_matrix_normalized.csv", class_names, normalized_matrix)

    metrics_by_class = per_class_metrics(class_names, matrix)
    summary_by_class = compute_macro_weighted_summary(class_names, metrics_by_class)
    top1 = safe_div(sum(int(row["correct"]) for row in prediction_rows), len(prediction_rows))
    top5 = safe_div(sum(int(row["top5_correct"]) for row in prediction_rows), len(prediction_rows))

    threshold_rows = compute_threshold_rows(prediction_rows, args.normal_class, DEFAULT_THRESHOLDS)
    threshold_summary = build_threshold_summary(threshold_rows, prediction_rows, args.normal_class)
    if threshold_rows:
        write_csv(output_dir / "threshold_sweep.csv", list(threshold_rows[0].keys()), threshold_rows)
        write_json(output_dir / "threshold_operating_points.json", threshold_summary)
    roc_rows, pr_rows = build_curve_rows(prediction_rows, args.normal_class)
    if roc_rows:
        write_csv(output_dir / "roc_curve.csv", list(roc_rows[0].keys()), roc_rows)
    if pr_rows:
        write_csv(output_dir / "pr_curve.csv", list(pr_rows[0].keys()), pr_rows)
    calibration_rows, calibration_summary = build_calibration_rows(prediction_rows, args.normal_class)
    if calibration_rows:
        write_csv(output_dir / "calibration_curve.csv", list(calibration_rows[0].keys()), calibration_rows)
    write_json(output_dir / "calibration_summary.json", calibration_summary)

    default_gate_row = select_threshold_row(threshold_rows, 0.50)
    fp_rows = [row for row in prediction_rows if row["gt_label"] == args.normal_class and float(row["abnormal_conf"] or 0.0) >= 0.5]
    fn_rows = [row for row in prediction_rows if row["gt_label"] != args.normal_class and float(row["abnormal_conf"] or 0.0) < 0.5]
    misclassified_rows = sorted([row for row in prediction_rows if int(row["correct"]) == 0], key=lambda row: float(row["top1_prob"]), reverse=True)
    hard_examples_topk = misclassified_rows[: min(200, len(misclassified_rows))]
    confused_pairs = build_confused_pairs(class_names, matrix)
    visual_candidates = build_visual_sample_candidates(prediction_rows, args.normal_class)
    write_csv(output_dir / "fp_normal.csv", PREDICTION_FIELDNAMES, fp_rows)
    write_csv(output_dir / "fn_abnormal.csv", PREDICTION_FIELDNAMES, fn_rows)
    write_csv(output_dir / "misclassified_samples.csv", PREDICTION_FIELDNAMES, misclassified_rows)
    write_csv(output_dir / "hard_examples_topk.csv", PREDICTION_FIELDNAMES, hard_examples_topk)
    if confused_pairs:
        write_csv(output_dir / "confused_pairs.csv", list(confused_pairs[0].keys()), confused_pairs)
    if visual_candidates:
        write_csv(output_dir / "visual_sample_candidates.csv", list(visual_candidates[0].keys()), visual_candidates)

    env_info, pip_freeze = build_env_info()
    save_text(output_dir / "pip_freeze.txt", pip_freeze)
    write_json(output_dir / "env_info.json", env_info)
    write_json(output_dir / "model_profile.json", build_model_profile(weights_path, run_dir, imgsz))
    copied_run_artifacts = copy_run_supporting_files(run_dir, output_dir)
    train_runtime_json = load_json_if_exists(run_dir / "training_runtime.json")

    run_manifest = {
        "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_dir.name,
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "weights_path": str(weights_path),
        "data_root": str(data_root),
        "normal_class": args.normal_class,
        "device": args.device,
        "batch": args.batch,
        "imgsz": imgsz,
        "class_names": class_names,
        "config": cfg,
        "run_args": run_args,
        "raw_run_files": {
            "args_yaml": str(run_dir / "args.yaml"),
            "results_csv": str(run_dir / "results.csv"),
            "epoch_metrics_csv": str(run_dir / "epoch_metrics.csv"),
            "training_runtime_json": str(run_dir / "training_runtime.json"),
            "best_weights": str(run_dir / "weights" / "best.pt"),
            "last_weights": str(run_dir / "weights" / "last.pt"),
        },
        "copied_supporting_files": copied_run_artifacts,
    }
    write_json(output_dir / "run_manifest.json", run_manifest)

    write_json(
        output_dir / "val_summary.json",
        {
            "run_name": run_dir.name,
            "num_classes": len(class_names),
            "class_names": class_names,
            "top1_accuracy": round(top1, 6),
            "top5_accuracy": round(top5, 6),
            "accuracy": round(top1, 6),
            "per_class": metrics_by_class,
            "summary": summary_by_class,
            "confusion_matrix_files": {
                "raw": "confusion_matrix.csv",
                "normalized": "confusion_matrix_normalized.csv",
            },
            "binary_gate_default_threshold_0_50": default_gate_row,
            "threshold_summary": threshold_summary,
            "calibration": calibration_summary,
            "curve_files": {
                "roc_curve": "roc_curve.csv",
                "pr_curve": "pr_curve.csv",
                "calibration_curve": "calibration_curve.csv",
            },
        },
    )

    write_json(
        output_dir / "runtime_profile.json",
        {
            "collection_runtime": collect_runtime,
            "train_runtime_path": str(run_dir / "training_runtime.json"),
            "train_runtime": train_runtime_json,
            "epoch_metrics_path": str(output_dir / "epoch_metrics.csv"),
            "val_images": len(prediction_rows),
            "embeddings_exported": embeddings is not None,
        },
    )

    best_epoch = None
    best_top1 = None
    if epoch_rows:
        best_row = max(epoch_rows, key=lambda row: safe_float(row.get("top1")) or -1.0)
        best_epoch = int(best_row["epoch"])
        best_top1 = safe_float(best_row["top1"])
    upsert_run_master(
        output_dir.parent / "run_master.csv",
        {
            "run_name": run_dir.name,
            "collected_at": run_manifest["collected_at"],
            "model": str(run_args.get("model") or cfg.get("model") or ""),
            "data_root": str(data_root),
            "top1_accuracy": round(top1, 6),
            "top5_accuracy": round(top5, 6),
            "best_epoch": best_epoch if best_epoch is not None else "",
            "best_top1": round(best_top1, 6) if best_top1 is not None else "",
            "gate_recall_at_0_50": round(float(default_gate_row["recall"]), 6) if default_gate_row else "",
            "gate_specificity_at_0_50": round(float(default_gate_row["specificity"]), 6) if default_gate_row else "",
            "gate_precision_at_0_50": round(float(default_gate_row["precision"]), 6) if default_gate_row else "",
            "gate_f1_at_0_50": round(float(default_gate_row["f1"]), 6) if default_gate_row else "",
            "auroc_exact": round(float(threshold_summary.get("auroc_exact", 0.0)), 6) if threshold_summary else "",
            "average_precision_exact": round(float(threshold_summary.get("average_precision_exact", 0.0)), 6)
            if threshold_summary
            else "",
            "materials_dir": str(output_dir),
            "run_dir": str(run_dir),
            "weights_path": str(weights_path),
        },
    )

    print_step("done", f"raw materials written to {output_dir}")


if __name__ == "__main__":
    main()
