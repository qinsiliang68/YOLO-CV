from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
YOLOV11_ROOT = REPO_ROOT / "YOLOv11"
STRUCT6_CLASSES = [
    "CrackBreak",
    "SurfaceDamage",
    "Deformation",
    "JointDislocation",
    "Intrusion",
    "Infiltration",
]


def ensure_yolov11_importable() -> None:
    """Make the local YOLOv11 checkout importable without requiring repo-wide installation."""
    root = str(YOLOV11_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_json_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_relative_path(value: str | None, base_dir: Path) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def resolve_model_value(value: str | None, base_dir: Path) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return str(path)
    candidate = (base_dir / path).resolve()
    if candidate.exists() or "/" in text or "\\" in text:
        return str(candidate)
    return text


def compact_dict(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None and v != ""}
