from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from pipeline_common import ensure_yolov11_importable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score train-side normals for RCD-Lite with calibrated TTA consistency.")
    parser.add_argument("--weights", required=True, help="Path to trained stage-1 gate weights.")
    parser.add_argument("--data-root", required=True, help="Binary gate dataset root containing train/Normal.")
    parser.add_argument("--output-dir", required=True, help="Directory for exported scores and optional gallery.")
    parser.add_argument("--device", default="0", help="Inference device.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--batch", type=int, default=1, help="Inference batch size.")
    parser.add_argument("--chunk-size", type=int, default=16, help="Reserved compatibility flag.")
    parser.add_argument("--top-k", type=int, default=250, help="Top-K train-side risky normals to keep.")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature fitted on val-cal for the anchor checkpoint.")
    parser.add_argument("--normal-class", default="Normal", help="Class treated as normal.")
    parser.add_argument("--gallery-top-n", type=int, default=0, help="Optional number of top risky normals to copy into a gallery. Default 0 keeps only CSV/JSON paths.")
    return parser.parse_args()


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def safe_prob_to_logit(prob: float, eps: float = 1e-6) -> float:
    clipped = min(max(prob, eps), 1.0 - eps)
    return math.log(clipped / (1.0 - clipped))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def build_tta_variants(image: Image.Image) -> list[tuple[str, Image.Image]]:
    rgb = image.convert("RGB")
    return [
        ("orig", rgb),
        ("bright_down", ImageEnhance.Brightness(rgb).enhance(0.92)),
        ("bright_up", ImageEnhance.Brightness(rgb).enhance(1.08)),
        ("contrast_down", ImageEnhance.Contrast(rgb).enhance(0.92)),
        ("blur_light", rgb.filter(ImageFilter.GaussianBlur(radius=0.8))),
    ]


def heuristic_group(image_path: Path) -> tuple[str, str]:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        hsv = rgb.convert("HSV")
        rgb_np = np.asarray(rgb, dtype=np.float32)
        hsv_np = np.asarray(hsv, dtype=np.float32)
        gray = np.asarray(rgb.convert("L"), dtype=np.float32)

    mean_v = float(hsv_np[..., 2].mean())
    std_v = float(hsv_np[..., 2].std())
    mean_s = float(hsv_np[..., 1].mean())
    bright_ratio = float((hsv_np[..., 2] > 245).mean())
    dark_ratio = float((hsv_np[..., 2] < 45).mean())
    grad_y, grad_x = np.gradient(gray / 255.0)
    edge_ratio = float((np.hypot(grad_x, grad_y) > 0.18).mean())

    if bright_ratio > 0.06 and std_v > 35:
        return "highlight", "high-intensity reflective region"
    if dark_ratio > 0.42:
        return "shadow", "large dark-area ratio"
    if mean_s < 35 and std_v < 28:
        return "low_contrast", "low saturation with flat brightness"
    if edge_ratio > 0.18 and 1.20 <= (rgb_np.shape[1] / max(rgb_np.shape[0], 1)) <= 1.30:
        return "boundary_texture", "dense edge response near pipe boundary aspect"
    if std_v > 45 and mean_s > 40:
        return "dirty_texture", "high brightness and color variation"
    if edge_ratio > 0.12:
        return "texture_like", "dense texture edges trigger false alarm risk"
    return "other", "no dominant heuristic pattern"


def predict_variant_probs(
    model,
    variants: list[tuple[str, Image.Image]],
    imgsz: int,
    batch: int,
    device: str,
    normal_class: str,
) -> list[tuple[str, float]]:
    use_half = device.lower() != "cpu"
    sources = [np.asarray(image, dtype=np.uint8) for _, image in variants]
    results = model.predict(
        source=sources,
        stream=False,
        verbose=False,
        imgsz=imgsz,
        batch=min(batch, len(sources)),
        device=device,
        half=use_half,
    )
    rows: list[tuple[str, float]] = []
    for (name, _image), result in zip(variants, results, strict=True):
        probs = result.probs.data.detach().cpu().numpy().astype(float)
        class_names = result.names
        if isinstance(class_names, dict):
            normal_index = next((int(idx) for idx, class_name in class_names.items() if class_name == normal_class), None)
        else:
            normal_index = next((idx for idx, class_name in enumerate(class_names) if class_name == normal_class), None)
        if normal_index is None:
            raise SystemExit(f"Normal class '{normal_class}' not found in model class names: {class_names}")
        p_normal = float(probs[normal_index])
        p_abnormal_raw = float(1.0 - p_normal)
        rows.append((name, p_abnormal_raw))
    if use_half:
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    return rows


def copy_gallery(rows: list[dict[str, Any]], gallery_dir: Path) -> None:
    gallery_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, start=1):
        source = Path(str(row["img_path"]))
        if not source.exists():
            continue
        score = float(row["p_abnormal_cal_mean"])
        target = gallery_dir / f"{index:03d}_{score:.4f}_{source.name}"
        shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    ensure_yolov11_importable()
    from ultralytics import YOLO

    weights = Path(args.weights).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    normal_root = data_root / "train" / "Normal"
    image_paths = sorted(path for path in normal_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise SystemExit(f"No train-side normal images found under {normal_root}")

    model = YOLO(str(weights), task="classify")
    rows: list[dict[str, Any]] = []
    tta_rows: list[dict[str, Any]] = []
    print_step("data", f"scoring {len(image_paths)} train-side normal images for RCD-Lite")
    for image_path in image_paths:
        with Image.open(image_path) as image:
            variants = build_tta_variants(image)
        variant_probs = predict_variant_probs(model, variants, args.imgsz, args.batch, str(args.device), args.normal_class)
        raw_probs = [prob for _name, prob in variant_probs]
        raw_logits = [safe_prob_to_logit(prob) for prob in raw_probs]
        cal_probs = [sigmoid(logit / float(args.temperature)) for logit in raw_logits]
        cal_logits = [safe_prob_to_logit(prob) for prob in cal_probs]
        group, reason = heuristic_group(image_path)
        row_id = len(rows)
        for aug_name, raw_prob in variant_probs:
            raw_logit = safe_prob_to_logit(raw_prob)
            cal_prob = sigmoid(raw_logit / float(args.temperature))
            cal_logit = safe_prob_to_logit(cal_prob)
            tta_rows.append(
                {
                    "row_id": row_id,
                    "img_rel_path": str(image_path.relative_to(data_root)).replace("\\", "/"),
                    "aug_name": aug_name,
                    "p_abnormal_raw": round(raw_prob, 6),
                    "logit_abnormal_raw": round(raw_logit, 6),
                    "p_abnormal_cal": round(cal_prob, 6),
                    "logit_abnormal_cal": round(cal_logit, 6),
                }
            )
        rows.append(
            {
                "img_path": str(image_path),
                "img_rel_path": str(image_path.relative_to(data_root)).replace("\\", "/"),
                "gt": "Normal",
                "heuristic_group": group,
                "heuristic_reason": reason,
                "tta_count": len(variant_probs),
                "p_abnormal_raw_mean": round(float(np.mean(raw_probs)), 6),
                "logit_abnormal_raw_mean": round(float(np.mean(raw_logits)), 6),
                "p_abnormal_cal_mean": round(float(np.mean(cal_probs)), 6),
                "logit_abnormal_cal_mean": round(float(np.mean(cal_logits)), 6),
                "logit_abnormal_cal_var": round(float(np.var(np.asarray(cal_logits, dtype=np.float32))), 6),
                "p_abnormal_cal_max": round(float(np.max(cal_probs)), 6),
                "p_abnormal_cal_min": round(float(np.min(cal_probs)), 6),
            }
        )

    rows_sorted = sorted(rows, key=lambda item: float(item["p_abnormal_cal_mean"]), reverse=True)
    top_rows = rows_sorted[: args.top_k]
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "train_normal_rcd_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (output_dir / "train_normal_rcd_tta.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tta_rows[0].keys()) if tta_rows else [])
        writer.writeheader()
        writer.writerows(tta_rows)
    with (output_dir / "top_false_positive_normals.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(top_rows[0].keys()))
        writer.writeheader()
        writer.writerows(top_rows)
    gallery_top_n = max(int(args.gallery_top_n), 0)
    if gallery_top_n > 0:
        copy_gallery(top_rows[: min(len(top_rows), gallery_top_n)], output_dir / "hardest_normal_gallery")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "weights": str(weights),
                "data_root": str(data_root),
                "score_device": str(args.device),
                "temperature": float(args.temperature),
                "total_train_normals": len(rows_sorted),
                "top_k": args.top_k,
                "tta_count": 5,
                "gallery_top_n": gallery_top_n,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print_step("done", f"wrote {output_dir}")


if __name__ == "__main__":
    main()
