from __future__ import annotations

import argparse
import json
from typing import Any

from pipeline_common import YOLOV11_ROOT, compact_dict, ensure_yolov11_importable, load_json_config, resolve_model_value, resolve_relative_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv11 detect train/val/predict with local source code.")
    parser.add_argument("--action", choices=("train", "val", "predict"), required=True, help="YOLO action to execute.")
    parser.add_argument("--config", required=True, help="Runtime JSON config relative to the repo root.")
    parser.add_argument("--data", default="", help="Override dataset YAML path.")
    parser.add_argument("--source", default="", help="Override prediction source.")
    parser.add_argument("--model", default="", help="Override model path or weights name.")
    parser.add_argument("--device", default="", help="Override device, e.g. 0 or cpu.")
    parser.add_argument("--epochs", type=int, default=-1, help="Override epochs for training.")
    parser.add_argument("--batch", type=int, default=-1, help="Override batch size.")
    parser.add_argument("--imgsz", type=int, default=-1, help="Override image size.")
    parser.add_argument("--split", default="", help="Override validation/test split.")
    parser.add_argument("--project", default="", help="Override run project directory.")
    parser.add_argument("--name", default="", help="Override run name.")
    parser.add_argument("--conf", type=float, default=-1.0, help="Override prediction confidence threshold.")
    parser.add_argument("--iou", type=float, default=-1.0, help="Override prediction IoU threshold.")
    parser.add_argument("--save-json", action="store_true", help="Force save_json=true for validation.")
    parser.add_argument("--save-txt", action="store_true", help="Force save_txt=true for prediction.")
    parser.add_argument("--save-conf", action="store_true", help="Force save_conf=true for prediction.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Append extra YOLO keyword arguments. May be provided multiple times.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved call without executing it.")
    return parser.parse_args()


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_extra_kwargs(items: list[str]) -> dict[str, Any]:
    extra_kwargs: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set value: {item!r}. Expected KEY=VALUE.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --set value: {item!r}. Empty key is not allowed.")
        extra_kwargs[key] = parse_scalar(raw_value.strip())
    return extra_kwargs


def build_kwargs(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    cfg = load_json_config(args.config)
    task = cfg.get("task", "detect")
    model = resolve_model_value(args.model or cfg.get("model"), YOLOV11_ROOT)
    project = resolve_relative_path(args.project or cfg.get("project"), YOLOV11_ROOT)
    common_kwargs = compact_dict(
        {
            "imgsz": args.imgsz if args.imgsz > 0 else cfg.get("imgsz"),
            "device": args.device or cfg.get("device"),
            "project": project,
            "name": args.name or cfg.get("name"),
        }
    )

    if args.action == "train":
        data = resolve_relative_path(args.data or cfg.get("data"), YOLOV11_ROOT)
        if not data:
            raise ValueError("Training requires a dataset YAML. Set it in config or pass --data.")
        kwargs = compact_dict(
            {
                **common_kwargs,
                "data": data,
                "epochs": args.epochs if args.epochs > 0 else cfg.get("epochs"),
                "batch": args.batch if args.batch > 0 else cfg.get("batch"),
                "workers": cfg.get("workers"),
                "pretrained": cfg.get("pretrained"),
                "patience": cfg.get("patience"),
                "optimizer": cfg.get("optimizer"),
                "cache": cfg.get("cache"),
                "resume": cfg.get("resume"),
            }
        )
    elif args.action == "val":
        data = resolve_relative_path(args.data or cfg.get("data"), YOLOV11_ROOT)
        if not data:
            raise ValueError("Validation requires a dataset YAML. Set it in config or pass --data.")
        kwargs = compact_dict(
            {
                **common_kwargs,
                "data": data,
                "split": args.split or cfg.get("split"),
                "batch": args.batch if args.batch > 0 else cfg.get("batch"),
                "save_json": True if args.save_json else cfg.get("save_json"),
            }
        )
    else:
        source = resolve_relative_path(args.source or cfg.get("source"), YOLOV11_ROOT)
        if not source:
            raise ValueError("Prediction requires a source path. Set it in config or pass --source.")
        kwargs = compact_dict(
            {
                **common_kwargs,
                "source": source,
                "conf": args.conf if args.conf >= 0 else cfg.get("conf"),
                "iou": args.iou if args.iou >= 0 else cfg.get("iou"),
                "save": cfg.get("save"),
                "save_txt": True if args.save_txt else cfg.get("save_txt"),
                "save_conf": True if args.save_conf else cfg.get("save_conf"),
            }
        )

    kwargs.update(parse_extra_kwargs(args.set))
    return task, model, kwargs


def main() -> None:
    args = parse_args()
    task, model, kwargs = build_kwargs(args)

    if args.dry_run:
        print(json.dumps({"task": task, "action": args.action, "model": model, "kwargs": kwargs}, indent=2, ensure_ascii=False))
        return

    ensure_yolov11_importable()
    from ultralytics import YOLO

    runner = YOLO(model, task=task)
    if args.action == "train":
        runner.train(**kwargs)
    elif args.action == "val":
        runner.val(**kwargs)
    else:
        runner.predict(**kwargs)


if __name__ == "__main__":
    main()
