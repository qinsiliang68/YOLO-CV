from __future__ import annotations

import os
import shutil
from pathlib import Path


def _font_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        from matplotlib.ft2font import FT2Font

        FT2Font(str(path))
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
