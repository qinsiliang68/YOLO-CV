from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_common import REPO_ROOT


EXPECTED_DIRS = [
    "data/sewerml/annotations",
    "data/sewerml/images_all",
    "YOLOv11/datasets/sewerml_cls6_train3000/train",
    "YOLOv11/datasets/sewerml_cls6_train3000/val",
    "YOLOv11/datasets/struct6_cls_target/train",
    "YOLOv11/datasets/struct6_cls_target/val",
    "YOLOv11/datasets/struct6_cls_target/test",
    "YOLOv11/datasets/struct6_det_pseudo/images/train",
    "YOLOv11/datasets/struct6_det_pseudo/images/val",
    "YOLOv11/datasets/struct6_det_pseudo/images/test",
    "YOLOv11/datasets/struct6_det_pseudo/labels/train",
    "YOLOv11/datasets/struct6_det_pseudo/labels/val",
    "YOLOv11/datasets/struct6_det_pseudo/labels/test",
    "YOLOv11/datasets/struct6_det_reviewed/images/train",
    "YOLOv11/datasets/struct6_det_reviewed/images/val",
    "YOLOv11/datasets/struct6_det_reviewed/images/test",
    "YOLOv11/datasets/struct6_det_reviewed/labels/train",
    "YOLOv11/datasets/struct6_det_reviewed/labels/val",
    "YOLOv11/datasets/struct6_det_reviewed/labels/test",
    "data/foshan/images",
    "data/foshan/labels_cls",
    "data/foshan/cam_outputs",
    "data/foshan/pseudo_boxes",
    "data/foshan/reviewed_boxes",
    "data/local/images",
    "data/local/labels_cls",
    "data/local/cam_outputs",
    "data/local/pseudo_boxes",
    "data/local/reviewed_boxes",
    "data/local/inference_samples",
    "data/normal/images",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or verify the local-only dataset layout.")
    parser.add_argument("--create-dirs", action="store_true", help="Create the expected local-only directories if missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing: list[str] = []
    for relative in EXPECTED_DIRS:
        target = REPO_ROOT / relative
        if args.create_dirs:
            target.mkdir(parents=True, exist_ok=True)
        if target.exists():
            print(f"[OK] {relative}")
        else:
            print(f"[MISSING] {relative}")
            missing.append(relative)

    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
