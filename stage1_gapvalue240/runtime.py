from __future__ import annotations

import os
import shutil
from pathlib import Path

from .util import sha256_file


def _font_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        from matplotlib.ft2font import FT2Font

        FT2Font(str(path))
        return True
    except Exception:
        return False


def _image_is_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def ensure_ultralytics_runtime(config_root: str | Path) -> Path:
    """Create a writable Ultralytics config with a loadable plotting font."""

    config_root = Path(config_root).resolve()
    config_dir = config_root / "Ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    destination = config_dir / "Arial.ttf"
    if _font_is_valid(destination):
        return config_dir

    candidates = [
        Path.home() / "AppData/Roaming/Ultralytics/Arial.ttf",
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/arial.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]
    try:
        from matplotlib import font_manager

        candidates.append(Path(font_manager.findfont("DejaVu Sans")))
    except Exception:
        pass
    source = next((path for path in candidates if _font_is_valid(path)), None)
    if source is None:
        raise RuntimeError("No loadable runtime font found for Ultralytics plotting")
    shutil.copy2(source, destination)
    if not _font_is_valid(destination):
        raise RuntimeError(f"Copied Ultralytics font is invalid: {destination}")
    return config_dir


def ensure_ultralytics_assets(yolo_root: str | Path) -> dict[str, dict[str, str]]:
    """Provide the two local images required by the Ultralytics AMP self-check.

    The repository intentionally ignores upstream sample assets. Training nodes
    retain them in the pinned venv or known-good historical runtime, so copy from
    those local sources and never make formal startup depend on the network.
    """

    root = Path(yolo_root).resolve()
    target_root = root / "ultralytics/assets"
    ai_root = root.parents[2] if len(root.parents) >= 3 else None
    source_roots = []
    if ai_root is not None:
        source_roots.extend(
            [
                ai_root / "venvs/yolo-cv/Lib/site-packages/ultralytics/assets",
                ai_root / "projects/YOLO-CV/YOLOv11/ultralytics/assets",
            ]
        )
    source_roots.append(Path.home() / "Desktop/ssh/AI/projects/YOLO-CV/YOLOv11/ultralytics/assets")

    report: dict[str, dict[str, str]] = {}
    for filename in ("bus.jpg", "zidane.jpg"):
        destination = target_root / filename
        if not _image_is_valid(destination):
            source = next(
                (candidate / filename for candidate in source_roots if _image_is_valid(candidate / filename)),
                None,
            )
            if source is None:
                raise RuntimeError(
                    f"Missing valid Ultralytics runtime asset {filename}; checked: "
                    + ", ".join(str(candidate / filename) for candidate in source_roots)
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        if not _image_is_valid(destination):
            raise RuntimeError(f"Ultralytics runtime asset is invalid after copy: {destination}")
        report[filename] = {"path": str(destination), "sha256": sha256_file(destination)}
    return report
