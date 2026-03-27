from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from pipeline_common import STRUCT6_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CAM heatmaps into first-pass pseudo boxes.")
    parser.add_argument("--cam-manifest", required=True, help="Manifest produced by export_cam.py.")
    parser.add_argument("--output", required=True, help="Output detector-style dataset root.")
    parser.add_argument("--thresholds", default="", help="Optional JSON mapping class name -> threshold in [0, 1].")
    parser.add_argument("--default-threshold", type=float, default=0.45, help="Fallback threshold in [0, 1].")
    parser.add_argument("--min-area-ratio", type=float, default=0.001, help="Minimum box area ratio to keep.")
    parser.add_argument("--max-area-ratio", type=float, default=0.85, help="Maximum box area ratio to keep.")
    parser.add_argument("--max-boxes", type=int, default=1, help="How many connected components to keep per image.")
    parser.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink", help="How to place images.")
    parser.add_argument("--keep-normal", action="store_true", help="Copy normal images with empty label files.")
    return parser.parse_args()


def safe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def load_thresholds(path: str, default: float) -> dict[str, float]:
    thresholds = {name: default for name in STRUCT6_CLASSES}
    if not path:
        return thresholds
    with Path(path).open("r", encoding="utf-8") as handle:
        thresholds.update(json.load(handle))
    return thresholds


def connected_component_boxes(mask: np.ndarray, max_boxes: int) -> list[tuple[int, int, int, int, int]]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: list[tuple[int, int, int, int, int]] = []
    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        boxes.append((x, y, w, h, area))
    boxes.sort(key=lambda item: item[4], reverse=True)
    return boxes[:max_boxes]


def xywhn(box: tuple[int, int, int, int], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = box
    xc = (x + w / 2) / width
    yc = (y + h / 2) / height
    return xc, yc, w / width, h / height


def main() -> None:
    args = parse_args()
    output_root = Path(args.output).resolve()
    images_root = output_root / "images"
    labels_root = output_root / "labels"
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)

    class_to_index = {name: idx for idx, name in enumerate(STRUCT6_CLASSES)}
    thresholds = load_thresholds(args.thresholds, args.default_threshold)

    summary_path = output_root / "manifest.csv"
    with (
        Path(args.cam_manifest).open("r", encoding="utf-8", newline="") as source,
        summary_path.open("w", encoding="utf-8", newline="") as target,
    ):
        reader = csv.DictReader(source)
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "filename",
                "relative_path",
                "target_class",
                "threshold",
                "boxes_written",
                "label_path",
                "image_path",
            ],
        )
        writer.writeheader()

        for row in reader:
            target_class = row.get("target_class", "")
            if target_class not in class_to_index:
                if not args.keep_normal:
                    continue
                boxes: list[tuple[int, int, int, int, int]] = []
            else:
                boxes = None

            image_path = Path(row["source_path"])
            heatmap_path = Path(row["heatmap_path"])
            relative_path = Path(row["relative_path"])

            image = cv2.imread(str(image_path))
            heatmap = cv2.imread(str(heatmap_path), cv2.IMREAD_GRAYSCALE)
            if image is None or heatmap is None:
                continue

            if boxes is None:
                threshold = thresholds.get(target_class, args.default_threshold)
                _, mask = cv2.threshold(heatmap, int(threshold * 255), 255, cv2.THRESH_BINARY)
                boxes = connected_component_boxes(mask.astype(np.uint8), args.max_boxes)

            valid_boxes = []
            image_area = float(image.shape[0] * image.shape[1])
            for x, y, w, h, area in boxes:
                area_ratio = area / image_area
                if area_ratio < args.min_area_ratio or area_ratio > args.max_area_ratio:
                    continue
                valid_boxes.append((x, y, w, h))

            output_image = images_root / relative_path
            output_label = (labels_root / relative_path).with_suffix(".txt")
            safe_link_or_copy(image_path, output_image, args.mode)
            output_label.parent.mkdir(parents=True, exist_ok=True)
            with output_label.open("w", encoding="utf-8") as handle:
                for box in valid_boxes:
                    if target_class not in class_to_index:
                        continue
                    xc, yc, wn, hn = xywhn(box, image.shape[1], image.shape[0])
                    handle.write(
                        f"{class_to_index[target_class]} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n"
                    )

            writer.writerow(
                {
                    "filename": row["filename"],
                    "relative_path": row["relative_path"],
                    "target_class": target_class,
                    "threshold": thresholds.get(target_class, args.default_threshold),
                    "boxes_written": len(valid_boxes),
                    "label_path": str(output_label),
                    "image_path": str(output_image),
                }
            )


if __name__ == "__main__":
    main()
