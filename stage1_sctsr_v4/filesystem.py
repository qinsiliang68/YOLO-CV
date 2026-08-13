from __future__ import annotations

import os
from pathlib import Path


def windows_safe_resolved_path(value: str | Path) -> Path:
    """Return an absolute path with Win32 extended-length semantics.

    SCTSR's registered experiment root is intentionally descriptive, and the
    required run/epoch partition hierarchy can exceed legacy MAX_PATH. PyArrow
    demonstrably fails at that boundary unless the native path carries the
    ``\\\\?\\`` prefix. Keep one prefixed root so every child operation uses the
    same identity instead of conditionally rewriting individual artifacts.
    """

    raw = str(value)
    if os.name != "nt":
        return Path(raw).expanduser().resolve()
    if raw.startswith("\\\\?\\"):
        return Path(raw)
    resolved = Path(raw).expanduser().resolve()
    if str(resolved).startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + str(resolved)[2:])
    return Path("\\\\?\\" + str(resolved))
