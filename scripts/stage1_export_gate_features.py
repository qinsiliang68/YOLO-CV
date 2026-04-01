from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from pipeline_common import ensure_yolov11_importable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export train/val gate features for post-hoc PTSG evaluation.")
    parser.add_argument("--weights", required=True, help="Classification model weights.")
    parser.add_argument("--data-root", required=True, help="Dataset root containing train/val folders.")
    parser.add_argument("--output-dir", required=True, help="Output directory for exported features.")
    parser.add_argument("--device", default="0", help="Inference device.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--batch", type=int, default=4, help="Per-call prediction batch size.")
    parser.add_argument("--chunk-size", type=int, default=32, help="Number of paths per chunk.")
    parser.add_argument("--normal-class", default="Normal", help="Class treated as normal.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to export.")
    return parser.parse_args()


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def chunked(items: list[Path], size: int) -> list[list[Path]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def class_names_from_model_names(names: dict[int, str] | list[str]) -> list[str]:
    if isinstance(names, dict):
        return [str(names[idx]) for idx in sorted(int(key) for key in names)]
    return [str(name) for name in names]


def safe_prob_to_logit(prob: float, eps: float = 1e-6) -> float:
    clipped = min(max(prob, eps), 1.0 - eps)
    return math.log(clipped / (1.0 - clipped))


def flatten_embedding(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(1)
    return array.reshape(-1)


def collect_image_paths(data_root: Path, split: str) -> list[Path]:
    split_root = data_root / split
    if not split_root.exists():
        return []
    return sorted(path for path in split_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def export_split(
    *,
    weights_path: Path,
    data_root: Path,
    split: str,
    output_dir: Path,
    device: str,
    imgsz: int,
    batch: int,
    chunk_size: int,
    normal_class: str,
) -> dict[str, Any]:
    ensure_yolov11_importable()
    import torch
    from ultralytics import YOLO

    image_paths = collect_image_paths(data_root, split)
    if not image_paths:
        raise SystemExit(f"No images found for split '{split}' under {data_root}")

    model = YOLO(str(weights_path), task="classify")
    use_half = str(device).lower() != "cpu"

    class_names = class_names_from_model_names(model.names)
    if normal_class not in class_names:
        raise SystemExit(f"Normal class '{normal_class}' not found in model classes: {class_names}")
    normal_index = class_names.index(normal_class)

    rows: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []
    predict_seconds = 0.0
    embed_seconds = 0.0

    print_step("export", f"{split}: {len(image_paths)} images")
    for batch_paths in chunked(image_paths, chunk_size):
        source = [str(path) for path in batch_paths]

        predict_start = time.time()
        batch_results = model.predict(
            source=source,
            verbose=False,
            stream=False,
            batch=min(batch, len(batch_paths)),
            device=device,
            imgsz=imgsz,
            half=use_half,
        )
        predict_seconds += time.time() - predict_start

        if torch.cuda.is_available() and use_half:
            torch.cuda.empty_cache()

        embed_start = time.time()
        batch_embed_results = model.embed(
            source=source,
            verbose=False,
            stream=False,
            batch=min(batch, len(batch_paths)),
            device=device,
            imgsz=imgsz,
            half=use_half,
        )
        embed_seconds += time.time() - embed_start

        if torch.cuda.is_available() and use_half:
            torch.cuda.empty_cache()

        for path, result, embedding in zip(batch_paths, batch_results, batch_embed_results, strict=True):
            probs = [float(value) for value in result.probs.data.detach().cpu().tolist()]
            top_indices = sorted(range(len(probs)), key=lambda idx: probs[idx], reverse=True)
            pred_index = top_indices[0]
            p_normal = probs[normal_index]
            p_abnormal = 1.0 - p_normal
            embedding_vector = flatten_embedding(embedding.detach().cpu().numpy())
            embedding_index = len(embeddings)
            embeddings.append(embedding_vector)
            gt_label = path.parent.name
            row = {
                "row_id": len(rows),
                "img_id": path.stem,
                "split": split,
                "gt_label": gt_label,
                "y_true": 0 if gt_label == normal_class else 1,
                "img_rel_path": str(path.relative_to(data_root)).replace("\\", "/"),
                "img_path": str(path),
                "pred_label": class_names[pred_index],
                "pred_index": pred_index,
                "correct": int(class_names[pred_index] == gt_label),
                "top1_prob": round(probs[pred_index], 12),
                "p_normal_raw": round(p_normal, 12),
                "p_abnormal_raw": round(p_abnormal, 12),
                "logit_abnormal": round(safe_prob_to_logit(p_abnormal), 12),
                "embedding_index": embedding_index,
                "embedding_dim": int(embedding_vector.shape[0]),
                "probs_json": json.dumps({class_names[idx]: probs[idx] for idx in range(len(class_names))}, ensure_ascii=False),
            }
            rows.append(row)

    embeddings_array = np.stack(embeddings) if embeddings else np.empty((0, 0), dtype=np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{split}_features.csv"
    npy_path = output_dir / f"{split}_embeddings.npy"
    meta_path = output_dir / f"{split}_features_meta.json"

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    np.save(npy_path, embeddings_array)
    meta_path.write_text(
        json.dumps(
            {
                "weights": str(weights_path),
                "data_root": str(data_root),
                "split": split,
                "rows": len(rows),
                "embedding_dim": 0 if embeddings_array.size == 0 else int(embeddings_array.shape[1]),
                "class_names": class_names,
                "normal_class": normal_class,
                "device": device,
                "imgsz": imgsz,
                "batch": batch,
                "chunk_size": chunk_size,
                "predict_seconds": round(predict_seconds, 6),
                "embed_seconds": round(embed_seconds, 6),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "split": split,
        "csv_path": str(csv_path),
        "npy_path": str(npy_path),
        "meta_path": str(meta_path),
        "rows": len(rows),
        "embedding_dim": 0 if embeddings_array.size == 0 else int(embeddings_array.shape[1]),
    }


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    split_summaries: list[dict[str, Any]] = []
    for split in args.splits:
        summary = export_split(
            weights_path=weights_path,
            data_root=data_root,
            split=split,
            output_dir=output_dir,
            device=args.device,
            imgsz=args.imgsz,
            batch=args.batch,
            chunk_size=args.chunk_size,
            normal_class=args.normal_class,
        )
        split_summaries.append(summary)

    summary_path = output_dir / "export_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "weights": str(weights_path),
                "data_root": str(data_root),
                "output_dir": str(output_dir),
                "splits": split_summaries,
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
