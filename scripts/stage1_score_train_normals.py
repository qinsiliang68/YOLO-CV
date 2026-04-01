from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline_common import ensure_yolov11_importable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score train-side normal images using an existing stage-1 gate model.")
    parser.add_argument("--weights", required=True, help="Path to trained stage-1 gate weights.")
    parser.add_argument("--data-root", required=True, help="Binary gate dataset root containing train/Normal.")
    parser.add_argument("--output-dir", required=True, help="Directory for exported scores and gallery.")
    parser.add_argument("--device", default="0", help="Inference device.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--batch", type=int, default=1, help="Inference batch size.")
    parser.add_argument("--chunk-size", type=int, default=32, help="How many image paths to process per predict call.")
    parser.add_argument("--top-k", type=int, default=200, help="Top-K train-side hard negatives to keep.")
    return parser.parse_args()


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def iter_predict_pairs(model, image_paths: list[Path], imgsz: int, batch: int, chunk_size: int, device: str):
    use_half = device.lower() != "cpu"
    for start in range(0, len(image_paths), chunk_size):
        chunk = image_paths[start : start + chunk_size]
        chunk_batch = min(batch, len(chunk))
        results = model.predict(
            source=[str(path) for path in chunk],
            stream=True,
            verbose=False,
            imgsz=imgsz,
            batch=chunk_batch,
            device=device,
            half=use_half,
        )
        for image_path, result in zip(chunk, results, strict=True):
            yield image_path, result
        if use_half:
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass


def collect_rows(model, image_paths: list[Path], data_root: Path, imgsz: int, batch: int, chunk_size: int, device: str) -> list[dict]:
    rows: list[dict] = []
    for image_path, result in iter_predict_pairs(model, image_paths, imgsz, batch, chunk_size, device):
        probs = result.probs.data.detach().cpu().numpy().astype(float)
        class_names = result.names
        p_abnormal = float(probs[0])
        p_normal = float(probs[1])
        pred_index = int(np.argmax(probs))
        pred_label = class_names[pred_index]
        group, reason = heuristic_group(image_path)
        top_indices = list(np.argsort(-probs)[:3])
        top1_idx = top_indices[0]
        top2_idx = top_indices[1] if len(top_indices) > 1 else top_indices[0]
        top3_idx = top_indices[2] if len(top_indices) > 2 else None
        rows.append(
            {
                "img_path": str(image_path),
                "img_rel_path": str(image_path.relative_to(data_root)),
                "gt": "Normal",
                "pred": pred_label,
                "p_abnormal": round(p_abnormal, 6),
                "p_normal": round(p_normal, 6),
                "top1_label": class_names[top1_idx],
                "top1_prob": round(float(probs[top1_idx]), 6),
                "top2_label": class_names[top2_idx],
                "top2_prob": round(float(probs[top2_idx]), 6),
                "top3_label": class_names[top3_idx] if top3_idx is not None else "",
                "top3_prob": round(float(probs[top3_idx]), 6) if top3_idx is not None else 0.0,
                "heuristic_group": group,
                "heuristic_reason": reason,
            }
        )
    return rows


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
        return "反光", "高亮区域比例偏高且亮度波动明显。"
    if dark_ratio > 0.42:
        return "阴影", "低亮度区域占比较高。"
    if mean_s < 35 and std_v < 28:
        return "水膜", "饱和度较低且整体亮度变化平缓。"
    if edge_ratio > 0.18 and 1.20 <= (rgb_np.shape[1] / max(rgb_np.shape[0], 1)) <= 1.30:
        return "接缝", "边缘密度较高且呈现典型管道边界结构。"
    if std_v > 45 and mean_s > 40:
        return "污渍", "亮度与颜色波动同时较强。"
    if edge_ratio > 0.12:
        return "纹理伪异常", "纹理边缘密集，疑似结构纹理触发误报。"
    return "其他", "未命中显著启发式规则。"


def copy_gallery(rows: list[dict], gallery_dir: Path) -> None:
    gallery_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, start=1):
        source = Path(row["img_path"])
        if not source.exists():
            continue
        score = float(row["p_abnormal"])
        group = str(row["heuristic_group"])
        target = gallery_dir / f"{index:03d}_{score:.4f}_{group}_{source.name}"
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

    print_step("data", f"scoring {len(image_paths)} train-side normal images")
    model = YOLO(str(weights), task="classify")
    score_device = str(args.device)
    try:
        rows = collect_rows(model, image_paths, data_root, args.imgsz, args.batch, args.chunk_size, score_device)
    except Exception as exc:
        if score_device.lower() != "cpu" and "out of memory" in str(exc).lower():
            print_step("warn", f"CUDA OOM during scoring on device={score_device}; retry on CPU")
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
            score_device = "cpu"
            rows = collect_rows(model, image_paths, data_root, args.imgsz, 1, 8, score_device)
        else:
            raise

    rows_sorted = sorted(rows, key=lambda item: item["p_abnormal"], reverse=True)
    top_rows = rows_sorted[: args.top_k]
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows_sorted[0].keys())
    with (output_dir / "train_normal_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    with (output_dir / "top_false_positive_normals.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(top_rows)
    copy_gallery(top_rows[: min(len(top_rows), 24)], output_dir / "hardest_normal_gallery")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "weights": str(weights),
                "data_root": str(data_root),
                "score_device": score_device,
                "total_train_normals": len(rows_sorted),
                "top_k": args.top_k,
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
