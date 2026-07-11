from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Iterable

import yaml


def _json_default(value: Any):
    try:
        import numpy as np
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, np.ndarray): return value.tolist()
    except Exception:
        pass
    if isinstance(value, Path): return str(value)
    if isinstance(value, set): return sorted(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default).encode("utf-8")


def stable_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def atomic_write_bytes(path: str | Path, data: bytes, overwrite: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {path}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


def atomic_write_text(path: str | Path, text: str, overwrite: bool = False) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"), overwrite=overwrite)


def atomic_write_json(path: str | Path, value: Any, overwrite: bool = False) -> Path:
    return atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n", overwrite)


def atomic_write_yaml(path: str | Path, value: Any, overwrite: bool = False) -> Path:
    return atomic_write_text(path, yaml.safe_dump(value, sort_keys=False, allow_unicode=True, width=120), overwrite)


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def ensure_relative_safe(path: str) -> None:
    p = Path(path)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Expected safe relative path, got: {path}")


def environment_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "pid": os.getpid(),
    }
    for name in ["numpy", "pandas", "sklearn", "torch", "ultralytics", "polars"]:
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # environment audit, never hide the error
            out[name] = {"status": "UNAVAILABLE", "error": str(exc)}
    return out


def write_sha256_manifest(root: str | Path, output: str | Path, exclude: Iterable[str] = ()) -> Path:
    root = Path(root).resolve()
    output = Path(output).resolve()
    excluded = {str(Path(x).as_posix()) for x in exclude}
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel in excluded or path.resolve() == output:
            continue
        rows.append((rel, path.stat().st_size, sha256_file(path)))
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["relative_path", "size_bytes", "sha256"])
            w.writerows(rows)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp_name, output)
    finally:
        if os.path.exists(tmp_name): os.unlink(tmp_name)
    return output
