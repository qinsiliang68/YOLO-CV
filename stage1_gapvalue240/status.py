from __future__ import annotations
from pathlib import Path
from .util import atomic_write_json

VALID_STATUSES = {
    "PLANNED", "STAGED", "RUNNING", "TRAIN_COMPLETED", "EVALUATED", "VALIDATED",
    "FAILED_INPUT", "FAILED_TRAIN", "FAILED_EVAL", "INVALID_ARTIFACT", "SUPERSEDED"
}
ALLOWED_TRANSITIONS = {
    "PLANNED": {"STAGED", "FAILED_INPUT"},
    "STAGED": {"RUNNING", "FAILED_INPUT"},
    "RUNNING": {"TRAIN_COMPLETED", "FAILED_TRAIN"},
    "TRAIN_COMPLETED": {"EVALUATED", "FAILED_EVAL"},
    "EVALUATED": {"VALIDATED", "INVALID_ARTIFACT"},
    "VALIDATED": {"SUPERSEDED"},
    "FAILED_INPUT": set(), "FAILED_TRAIN": set(), "FAILED_EVAL": set(),
    "INVALID_ARTIFACT": set(), "SUPERSEDED": set(),
}

def set_status(attempt_dir: Path, status: str, payload: dict | None = None) -> None:
    if status not in VALID_STATUSES: raise ValueError(status)
    status_dir = attempt_dir / "08_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    current_files = [p for p in status_dir.iterdir() if p.name in VALID_STATUSES]
    if current_files:
        current = current_files[0].name
        if status not in ALLOWED_TRANSITIONS[current]:
            raise RuntimeError(f"Invalid status transition {current} -> {status}")
        current_files[0].unlink()
    atomic_write_json(status_dir / status, payload or {"status": status})
