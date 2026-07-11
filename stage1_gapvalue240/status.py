from __future__ import annotations
from pathlib import Path
from .util import atomic_write_json

VALID_STATUSES = {
    "PLANNED", "STAGED", "RUNNING", "TRAIN_COMPLETED", "EVALUATED", "VALIDATED",
    "RECOVERING", "VALIDATING", "DRY_RUN_VALIDATED",
    "FAILED_INPUT", "FAILED_TRAIN", "FAILED_EVAL", "FAILED_TRAIN_RETRYABLE",
    "FAILED_EVAL_RETRYABLE", "INVALID_ARTIFACT", "SUPERSEDED"
}
ALLOWED_TRANSITIONS = {
    "PLANNED": {"STAGED", "FAILED_INPUT"},
    "STAGED": {"RUNNING", "FAILED_INPUT"},
    "RUNNING": {"RECOVERING", "TRAIN_COMPLETED", "FAILED_TRAIN", "FAILED_TRAIN_RETRYABLE"},
    "RECOVERING": {"RUNNING", "FAILED_TRAIN", "FAILED_TRAIN_RETRYABLE"},
    "TRAIN_COMPLETED": {"EVALUATED", "FAILED_EVAL", "FAILED_EVAL_RETRYABLE"},
    "EVALUATED": {"VALIDATING", "DRY_RUN_VALIDATED", "INVALID_ARTIFACT"},
    "VALIDATING": {"VALIDATED", "DRY_RUN_VALIDATED", "INVALID_ARTIFACT"},
    "VALIDATED": {"SUPERSEDED"},
    "DRY_RUN_VALIDATED": {"SUPERSEDED"},
    "FAILED_INPUT": set(), "FAILED_TRAIN": set(), "FAILED_EVAL": set(),
    "FAILED_TRAIN_RETRYABLE": {"RECOVERING", "RUNNING", "FAILED_TRAIN"},
    "FAILED_EVAL_RETRYABLE": {"TRAIN_COMPLETED", "EVALUATED"},
    "INVALID_ARTIFACT": {"VALIDATING"}, "SUPERSEDED": set(),
}


def read_status(attempt_dir: Path) -> dict:
    status_dir = Path(attempt_dir) / "08_status"
    canonical = status_dir / "status.json"
    if canonical.exists():
        import json
        value = json.loads(canonical.read_text(encoding="utf-8"))
        if value.get("state") not in VALID_STATUSES:
            raise RuntimeError(f"Invalid canonical state in {canonical}: {value}")
        return value
    markers = [p for p in status_dir.iterdir() if p.name in VALID_STATUSES] if status_dir.exists() else []
    if len(markers) != 1:
        raise RuntimeError(f"Expected one status for {attempt_dir}, found {[p.name for p in markers]}")
    return {"state": markers[0].name, "legacy_marker_only": True}

def set_status(attempt_dir: Path, status: str, payload: dict | None = None) -> None:
    if status not in VALID_STATUSES: raise ValueError(status)
    status_dir = attempt_dir / "08_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    current_files = [p for p in status_dir.iterdir() if p.name in VALID_STATUSES]
    canonical = status_dir / "status.json"
    if canonical.exists() or current_files:
        current = read_status(attempt_dir)["state"]
        if status not in ALLOWED_TRANSITIONS[current]:
            raise RuntimeError(f"Invalid status transition {current} -> {status}")
    record = {"state": status, **(payload or {})}
    # Canonical state is replaced first, so a crash can never leave a zero-state window.
    atomic_write_json(canonical, record, overwrite=True)
    for marker in current_files:
        marker.unlink(missing_ok=True)
    atomic_write_json(status_dir / status, record)
