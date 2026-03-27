from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_common import YOLOV11_ROOT, compact_dict, ensure_yolov11_importable, load_json_config, resolve_model_value, resolve_relative_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a source-domain classification model for the SewerML pipeline.")
    parser.add_argument(
        "--config",
        default="YOLOv11/configs/runtime/cls_source_cls3.json",
        help="JSON runtime config. Use cls_source_cls3.json or cls_source_cls6.json.",
    )
    parser.add_argument("--data", default="", help="Override the classification dataset directory.")
    parser.add_argument("--model", default="", help="Override the classification model or weights.")
    parser.add_argument("--epochs", type=int, default=-1, help="Override epochs.")
    parser.add_argument("--batch", type=int, default=-1, help="Override batch size.")
    parser.add_argument("--imgsz", type=int, default=-1, help="Override image size.")
    parser.add_argument("--device", default="", help="Override device, e.g. 0 or cpu.")
    parser.add_argument("--project", default="", help="Override run project directory.")
    parser.add_argument("--name", default="", help="Override run name.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved arguments without training.")
    return parser.parse_args()


def build_train_kwargs(args: argparse.Namespace) -> tuple[str, dict]:
    cfg = load_json_config(args.config)
    model_value = args.model or cfg.get("model")
    data_value = args.data or cfg.get("data")
    project_value = args.project or cfg.get("project")

    model = resolve_model_value(model_value, YOLOV11_ROOT)
    train_kwargs = compact_dict(
        {
            "data": resolve_relative_path(data_value, YOLOV11_ROOT),
            "epochs": args.epochs if args.epochs > 0 else cfg.get("epochs"),
            "imgsz": args.imgsz if args.imgsz > 0 else cfg.get("imgsz"),
            "batch": args.batch if args.batch > 0 else cfg.get("batch"),
            "device": args.device or cfg.get("device"),
            "workers": cfg.get("workers"),
            "project": resolve_relative_path(project_value, YOLOV11_ROOT),
            "name": args.name or cfg.get("name"),
            "pretrained": cfg.get("pretrained"),
            "patience": cfg.get("patience"),
            "optimizer": cfg.get("optimizer"),
            "cache": cfg.get("cache"),
            "resume": cfg.get("resume"),
        }
    )
    return model, train_kwargs


def main() -> None:
    args = parse_args()
    model, train_kwargs = build_train_kwargs(args)

    if args.dry_run:
        print(json.dumps({"model": model, "train_kwargs": train_kwargs}, indent=2, ensure_ascii=False))
        return

    ensure_yolov11_importable()
    from ultralytics import YOLO

    trainer = YOLO(model, task="classify")
    trainer.train(**train_kwargs)


if __name__ == "__main__":
    main()
