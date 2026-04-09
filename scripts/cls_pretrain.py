from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from classify_train_callbacks import register_classification_material_callbacks
from pipeline_common import YOLOV11_ROOT, compact_dict, ensure_yolov11_importable, load_json_config, resolve_model_value, resolve_relative_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a source-domain classification model for the SewerML pipeline.")
    parser.add_argument(
        "--config",
        required=True,
        help="JSON runtime config for source-domain classification.",
    )
    parser.add_argument("--data", default="", help="Override the classification dataset directory.")
    parser.add_argument("--model", default="", help="Override the classification model or weights.")
    parser.add_argument("--epochs", type=int, default=-1, help="Override epochs.")
    parser.add_argument("--batch", type=int, default=-1, help="Override batch size.")
    parser.add_argument("--imgsz", type=int, default=-1, help="Override image size.")
    parser.add_argument("--device", default="", help="Override device, e.g. 0 or cpu.")
    parser.add_argument("--project", default="", help="Override run project directory.")
    parser.add_argument("--name", default="", help="Override run name.")
    parser.add_argument("--stdout-log", default="", help="Optional path to tee stdout log.")
    parser.add_argument("--stderr-log", default="", help="Optional path to tee stderr log.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved arguments without training.")
    return parser.parse_args()


class TeeStream:
    def __init__(self, stream, log_path: str) -> None:
        self._stream = stream
        self._handle = None
        if log_path:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("a", encoding="utf-8")

    def write(self, data: str) -> int:
        written = self._stream.write(data)
        if self._handle is not None:
            self._handle.write(data)
        return written

    def flush(self) -> None:
        self._stream.flush()
        if self._handle is not None:
            self._handle.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._stream, "isatty", lambda: False)())

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()


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
            "exist_ok": cfg.get("exist_ok"),
            "pretrained": cfg.get("pretrained"),
            "patience": cfg.get("patience"),
            "optimizer": cfg.get("optimizer"),
            "cache": cfg.get("cache"),
            "resume": cfg.get("resume"),
            "save_period": cfg.get("save_period"),
            "seed": cfg.get("seed"),
        }
    )
    return model, train_kwargs


def main() -> None:
    args = parse_args()
    model, train_kwargs = build_train_kwargs(args)

    if args.dry_run:
        print(json.dumps({"model": model, "train_kwargs": train_kwargs}, indent=2, ensure_ascii=False))
        return

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout_tee = TeeStream(sys.stdout, args.stdout_log)
    stderr_tee = TeeStream(sys.stderr, args.stderr_log)
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee
    ensure_yolov11_importable()
    from ultralytics import YOLO

    trainer = YOLO(model, task="classify")
    register_classification_material_callbacks(trainer)
    try:
        trainer.train(**train_kwargs)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        stdout_tee.flush()
        stderr_tee.flush()
        stdout_tee.close()
        stderr_tee.close()


if __name__ == "__main__":
    main()
