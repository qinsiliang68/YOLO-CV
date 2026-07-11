from __future__ import annotations

import os

from .errors import ConfigurationError, ContractError, ExternalCommandError, LockHeldError, ValidationError


EXIT_SUCCESS = 0
EXIT_RETRYABLE = 20
EXIT_TERMINAL = 30


def exit_code_for_exception(error: BaseException) -> int:
    if isinstance(error, LockHeldError):
        return EXIT_RETRYABLE
    if isinstance(error, (ContractError, ConfigurationError, ValidationError, FileNotFoundError, ValueError)):
        return EXIT_TERMINAL
    if isinstance(error, (ExternalCommandError, TimeoutError, OSError)):
        return EXIT_RETRYABLE
    return EXIT_RETRYABLE


def aiops_status_payload(
    *,
    run_slot: str,
    phase: str,
    last_epoch: int | None,
    resume_count: int,
    retryable: bool,
    error_code: str | None,
    attempt_id: str,
) -> dict:
    return {
        "run_slot": str(run_slot),
        "phase": str(phase),
        "pid": os.getpid(),
        "last_epoch": None if last_epoch is None else int(last_epoch),
        "resume_count": int(resume_count),
        "retryable": bool(retryable),
        "error_code": error_code,
        "attempt_id": str(attempt_id),
    }
