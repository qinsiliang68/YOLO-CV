from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


def _normalize(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _normalize(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bytes):
        return value.hex().upper()
    if isinstance(value, Mapping):
        return {str(k): _normalize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_normalize(v) for v in value]
    if isinstance(value, set):
        return sorted((_normalize(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Canonical JSON forbids NaN and Infinity")
        return value
    # Handle NumPy scalar values without importing NumPy as a hard dependency.
    if hasattr(value, "item") and type(value).__module__.startswith("numpy"):
        return _normalize(value.item())
    return value


def canonical_json_bytes(value: Any) -> bytes:
    normalized = _normalize(value)
    return (json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def stable_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest().upper()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest().upper()


def _fsync_directory(path: str | Path) -> None:
    directory = Path(path)
    if os.name == "nt":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Registered Windows run roots can legitimately be close to MAX_PATH;
    # repeating a long destination leaf in the tempfile prefix needlessly
    # makes an otherwise safe atomic write fail.
    fd, tmp_name = tempfile.mkstemp(prefix=".aw.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def atomic_write_text(path: str | Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, value: Any) -> Path:
    return atomic_write_bytes(path, canonical_json_bytes(value))


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_record(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:
    p = Path(path)
    relative = p.relative_to(Path(root)).as_posix() if root is not None else p.as_posix()
    return {"path": relative, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
