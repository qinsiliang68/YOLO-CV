from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


CheckpointValidator = Callable[[Path], bool]


@dataclass(frozen=True)
class RunDecision:
    action: str
    attempt_dir: Path | None = None
    checkpoint: Path | None = None
    superseded_attempt: Path | None = None
    reason: str = ""


def _identity_matches(attempt: Path, expected: dict) -> bool:
    path = attempt / "00_identity/run_identity.json"
    if not path.is_file():
        return False
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return all(actual.get(key) == value for key, value in expected.items())


def _state(attempt: Path) -> str | None:
    path = attempt / "08_status/status.json"
    if path.is_file():
        try:
            return str(json.loads(path.read_text(encoding="utf-8"))["state"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
    status_dir = attempt / "08_status"
    known = [
        marker.name
        for marker in status_dir.iterdir()
        if marker.is_file() and marker.name.isupper()
    ] if status_dir.is_dir() else []
    return known[0] if len(known) == 1 else None


def _valid_checkpoint(path: Path, validator: CheckpointValidator) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        return bool(validator(path))
    except Exception:
        return False


def discover_run_action(
    run_parent: str | Path,
    expected_identity: dict,
    *,
    checkpoint_validator: CheckpointValidator,
) -> RunDecision:
    """Choose the only safe idempotent action for a frozen run slot.

    This function never edits an old attempt.  In particular, a corrupt
    checkpoint remains in place as failure evidence and the caller creates a
    new UUID attempt beside it.
    """

    parent = Path(run_parent)
    attempts = sorted(
        (path for path in parent.glob("attempt_*") if path.is_dir()),
        key=lambda value: (value.stat().st_mtime_ns, value.name),
    ) if parent.is_dir() else []
    matching = [attempt for attempt in attempts if _identity_matches(attempt, expected_identity)]

    validated = [attempt for attempt in matching if _state(attempt) == "VALIDATED"]
    if len(validated) > 1:
        raise RuntimeError(f"Multiple matching validated attempts: {validated}")
    if validated:
        return RunDecision("SKIP_VALIDATED", attempt_dir=validated[0], reason="already validated")

    dry_validated = [attempt for attempt in matching if _state(attempt) == "DRY_RUN_VALIDATED"]
    if len(dry_validated) > 1:
        raise RuntimeError(f"Multiple matching dry-run attempts: {dry_validated}")
    if dry_validated:
        return RunDecision("SKIP_DRY_RUN", attempt_dir=dry_validated[0], reason="dry run already validated")

    active = [
        attempt
        for attempt in matching
        if _state(attempt) not in {"SUPERSEDED", "FAILED_INPUT", "FAILED_TRAIN", "FAILED_EVAL"}
    ]
    if len(active) > 1:
        raise RuntimeError(f"Multiple matching active attempts: {active}")
    if not active:
        previous = attempts[-1] if attempts else None
        reason = "no matching runtime identity" if previous is not None else "no prior attempt"
        return RunDecision("NEW_ATTEMPT", superseded_attempt=previous, reason=reason)

    attempt = active[0]
    state = _state(attempt)
    if state in {"PLANNED"}:
        return RunDecision(
            "NEW_ATTEMPT",
            superseded_attempt=attempt,
            reason="preparation was interrupted before the frozen inputs were staged",
        )
    if state in {"STAGED"}:
        return RunDecision("TRAIN", attempt_dir=attempt, reason="training not started")
    if state in {"TRAIN_COMPLETED", "FAILED_EVAL_RETRYABLE"}:
        return RunDecision("EVALUATE", attempt_dir=attempt, reason="evaluation incomplete")
    if state in {"EVALUATED", "INVALID_ARTIFACT", "VALIDATING"}:
        return RunDecision("VALIDATE", attempt_dir=attempt, reason="validation incomplete")
    if state in {"RUNNING", "RECOVERING", "FAILED_TRAIN_RETRYABLE"}:
        checkpoint = attempt / "training_state/last.pt"
        if _valid_checkpoint(checkpoint, checkpoint_validator):
            return RunDecision(
                "RESUME_TRAIN",
                attempt_dir=attempt,
                checkpoint=checkpoint,
                reason="loadable native-resume checkpoint",
            )
        return RunDecision(
            "NEW_ATTEMPT",
            superseded_attempt=attempt,
            reason="missing or corrupt native-resume checkpoint",
        )
    return RunDecision(
        "NEW_ATTEMPT",
        superseded_attempt=attempt,
        reason=f"unrecoverable state: {state}",
    )
