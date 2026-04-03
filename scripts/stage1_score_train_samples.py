from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pipeline_common import ensure_yolov11_importable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score all train-side gate samples for hard mining.")
    parser.add_argument("--weights", required=True, help="Path to trained stage-1 gate weights.")
    parser.add_argument("--data-root", required=True, help="Binary gate dataset root containing train folders.")
    parser.add_argument("--output-dir", required=True, help="Directory for exported scores.")
    parser.add_argument("--device", default="0", help="Inference device.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--batch", type=int, default=1, help="Inference batch size.")
    parser.add_argument("--chunk-size", type=int, default=32, help="How many image paths to process per predict call.")
    parser.add_argument("--normal-class", default="Normal", help="Class treated as normal.")
    return parser.parse_args()


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def chunked(items: list[Path], size: int) -> list[list[Path]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def iter_predict_pairs(model, image_paths: list[Path], imgsz: int, batch: int, chunk_size: int, device: str):
    use_half = device.lower() != "cpu"
    for chunk in chunked(image_paths, chunk_size):
        results = model.predict(
            source=[str(path) for path in chunk],
            stream=True,
            verbose=False,
            imgsz=imgsz,
            batch=min(batch, len(chunk)),
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


def collect_train_images(data_root: Path) -> list[Path]:
    train_root = data_root / "train"
    if not train_root.exists():
        raise SystemExit(f"Missing train split under {data_root}")
    images = sorted(path for path in train_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"No train images found under {train_root}")
    return images


def collect_rows(model, image_paths: list[Path], data_root: Path, imgsz: int, batch: int, chunk_size: int, device: str, normal_class: str) -> list[dict]:
    rows: list[dict] = []
    class_names = None
    normal_index = None
    for image_path, result in iter_predict_pairs(model, image_paths, imgsz, batch, chunk_size, device):
        probs = result.probs.data.detach().cpu().numpy().astype(float)
        if class_names is None:
            class_names = result.names
            if normal_class not in class_names.values():
                raise SystemExit(f"Normal class '{normal_class}' not found in model names: {class_names}")
            normal_index = [idx for idx, name in class_names.items() if name == normal_class][0]
        pred_index = int(np.argmax(probs))
        pred_label = class_names[pred_index]
        p_normal = float(probs[normal_index])
        p_abnormal = float(1.0 - p_normal)
        gt_label = image_path.parent.name
        is_normal = gt_label == normal_class
        hardness_score = p_abnormal if is_normal else (1.0 - p_abnormal)
        rows.append(
            {
                "img_path": str(image_path),
                "img_rel_path": str(image_path.relative_to(data_root)).replace("\\", "/"),
                "split": "train",
                "gt_label": gt_label,
                "pred_label": pred_label,
                "correct": int(pred_label == gt_label),
                "p_abnormal": round(p_abnormal, 6),
                "p_normal": round(p_normal, 6),
                "hardness_score": round(float(hardness_score), 6),
                "hardness_type": "hard_negative" if is_normal else "hard_positive",
            }
        )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    ensure_yolov11_importable()
    from ultralytics import YOLO

    weights = Path(args.weights).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    image_paths = collect_train_images(data_root)
    print_step("data", f"scoring {len(image_paths)} train-side images")
    model = YOLO(str(weights), task="classify")

    score_device = str(args.device)
    try:
        rows = collect_rows(model, image_paths, data_root, args.imgsz, args.batch, args.chunk_size, score_device, args.normal_class)
    except Exception as exc:
        if score_device.lower() != "cpu" and "out of memory" in str(exc).lower():
            print_step("warn", f"CUDA OOM during train scoring on device={score_device}; retry on CPU")
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
            score_device = "cpu"
            rows = collect_rows(model, image_paths, data_root, args.imgsz, 1, 8, score_device, args.normal_class)
        else:
            raise

    rows_sorted = sorted(rows, key=lambda item: (item["hardness_type"], item["hardness_score"]), reverse=True)
    hard_negatives = sorted((row for row in rows if row["gt_label"] == args.normal_class), key=lambda item: item["p_abnormal"], reverse=True)
    hard_positives = sorted((row for row in rows if row["gt_label"] != args.normal_class), key=lambda item: item["p_abnormal"])

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    write_csv(output_dir / "train_sample_scores.csv", fieldnames, rows_sorted)
    write_csv(output_dir / "hard_negative_candidates.csv", fieldnames, hard_negatives)
    write_csv(output_dir / "hard_positive_candidates.csv", fieldnames, hard_positives)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "weights": str(weights),
                "data_root": str(data_root),
                "score_device": score_device,
                "total_train_images": len(rows),
                "normal_class": args.normal_class,
                "normal_count": sum(1 for row in rows if row["gt_label"] == args.normal_class),
                "abnormal_count": sum(1 for row in rows if row["gt_label"] != args.normal_class),
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
