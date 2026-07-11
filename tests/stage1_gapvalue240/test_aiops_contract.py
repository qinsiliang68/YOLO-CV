from __future__ import annotations

import os

from stage1_gapvalue240.aiops import (
    EXIT_RETRYABLE,
    EXIT_SUCCESS,
    EXIT_TERMINAL,
    aiops_status_payload,
    exit_code_for_exception,
)
from stage1_gapvalue240.errors import ContractError, ExternalCommandError, LockHeldError, ValidationError


def test_aiops_exit_code_contract_is_stable():
    assert EXIT_SUCCESS == 0
    assert EXIT_RETRYABLE == 20
    assert EXIT_TERMINAL == 30
    assert exit_code_for_exception(ExternalCommandError("worker failed")) == 20
    assert exit_code_for_exception(TimeoutError("worker timed out")) == 20
    assert exit_code_for_exception(LockHeldError("busy")) == 20
    assert exit_code_for_exception(ContractError("wrong contract")) == 30
    assert exit_code_for_exception(ValidationError("bad frozen input")) == 30
    assert exit_code_for_exception(FileNotFoundError("missing input")) == 30


def test_aiops_status_has_required_fields():
    payload = aiops_status_payload(
        run_slot="RUN_001",
        phase="train",
        last_epoch=17,
        resume_count=2,
        retryable=True,
        error_code="TRAIN_WORKER_FAILED",
        attempt_id="attempt-a",
    )
    assert payload == {
        "run_slot": "RUN_001",
        "phase": "train",
        "pid": os.getpid(),
        "last_epoch": 17,
        "resume_count": 2,
        "retryable": True,
        "error_code": "TRAIN_WORKER_FAILED",
        "attempt_id": "attempt-a",
    }
