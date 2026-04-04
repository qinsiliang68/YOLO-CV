from __future__ import annotations

import argparse
import json
import sys

from classify_train_callbacks import register_classification_material_callbacks
from pipeline_common import YOLOV11_ROOT, compact_dict, ensure_yolov11_importable, load_json_config, resolve_model_value, resolve_relative_path


CUSTOM_LOSS_KEYS = (
    "cls_loss_type",
    "cls_pos_weight",
    "cls_focal_gamma",
    "cls_focal_alpha",
    "cls_recall_margin",
    "cls_recall_penalty",
    "contrastive_enable",
    "contrastive_method",
    "contrastive_weight",
    "contrastive_temperature",
    "contrastive_proj_dim",
    "contrastive_proj_hidden",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stage-1 gate experiments with optional HN and custom binary loss.")
    parser.add_argument(
        "--config",
        default="YOLOv11/configs/runtime/stage1_gate_l_hn.json",
        help="JSON runtime config for stage-1 gate training.",
    )
    parser.add_argument("--stdout-log", default="", help="Optional path to tee stdout log.")
    parser.add_argument("--stderr-log", default="", help="Optional path to tee stderr log.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved overrides without training.")
    return parser.parse_args()


class TeeStream:
    def __init__(self, stream, log_path: str) -> None:
        self._stream = stream
        self._handle = None
        if log_path:
            from pathlib import Path

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


def load_stage1_config(path: str) -> tuple[dict, dict]:
    cfg = load_json_config(path)
    custom = {key: cfg.pop(key) for key in list(cfg.keys()) if key in CUSTOM_LOSS_KEYS}
    return cfg, custom


def build_overrides(cfg: dict) -> tuple[str, dict]:
    model = resolve_model_value(cfg.get("model"), YOLOV11_ROOT)
    overrides = compact_dict(
        {
            "task": "classify",
            "data": resolve_relative_path(cfg.get("data"), YOLOV11_ROOT),
            "epochs": cfg.get("epochs"),
            "imgsz": cfg.get("imgsz"),
            "batch": cfg.get("batch"),
            "device": cfg.get("device"),
            "workers": cfg.get("workers"),
            "project": resolve_relative_path(cfg.get("project"), YOLOV11_ROOT),
            "name": cfg.get("name"),
            "pretrained": cfg.get("pretrained"),
            "patience": cfg.get("patience"),
            "optimizer": cfg.get("optimizer"),
            "cache": cfg.get("cache"),
            "resume": cfg.get("resume"),
            "save_period": cfg.get("save_period"),
            "dropout": cfg.get("dropout"),
            "seed": cfg.get("seed"),
        }
    )
    return model, overrides


def main() -> None:
    args = parse_args()
    cfg, custom_loss_cfg = load_stage1_config(args.config)
    model, overrides = build_overrides(cfg)

    if args.dry_run:
        print(json.dumps({"model": model, "overrides": overrides, "custom_loss_cfg": custom_loss_cfg}, indent=2, ensure_ascii=False))
        return

    stdout_tee = TeeStream(sys.stdout, args.stdout_log)
    stderr_tee = TeeStream(sys.stderr, args.stderr_log)
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee
    ensure_yolov11_importable()
    from ultralytics.models.yolo.classify import ClassificationTrainer

    try:
        trainer = ClassificationTrainer(overrides={**overrides, "model": model})
        for key, value in custom_loss_cfg.items():
            setattr(trainer.args, key, value)
        register_classification_material_callbacks(trainer)
        trainer.train()
    finally:
        stdout_tee.flush()
        stderr_tee.flush()
        stdout_tee.close()
        stderr_tee.close()


if __name__ == "__main__":
    main()
