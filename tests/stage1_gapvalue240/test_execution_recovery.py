from __future__ import annotations

import json
from pathlib import Path

import pytest

from stage1_gapvalue240.execution_recovery import discover_run_action
from stage1_gapvalue240.util import atomic_write_json


EXPECTED = {
    "runtime_contract_sha256": "runtime",
    "science_contract_sha256": "science",
    "matrix_sha256": "matrix",
    "selection_index_sha256": "index",
    "selection_sha256": "selection",
    "input_snapshot_id": "snapshot",
}


def _attempt(parent: Path, name: str, state: str, *, final: bool = False,
             identity: dict | None = None, last: bytes | None = None) -> Path:
    suffix = "" if final else ".inprogress"
    attempt = parent / f"attempt_{name}{suffix}"
    (attempt / "00_identity").mkdir(parents=True)
    (attempt / "08_status").mkdir(parents=True)
    (attempt / "training_state").mkdir(parents=True)
    atomic_write_json(
        attempt / "00_identity/run_identity.json",
        {"attempt_id": name, **EXPECTED, **(identity or {})},
    )
    atomic_write_json(attempt / "08_status/status.json", {"state": state})
    if last is not None:
        (attempt / "training_state/last.pt").write_bytes(last)
    return attempt


def test_validated_matching_attempt_is_skipped(tmp_path):
    parent = tmp_path / "RUN_001"
    attempt = _attempt(parent, "done", "VALIDATED", final=True)
    decision = discover_run_action(parent, EXPECTED, checkpoint_validator=lambda _: True)
    assert decision.action == "SKIP_VALIDATED"
    assert decision.attempt_dir == attempt


def test_running_attempt_resumes_only_from_loadable_last_checkpoint(tmp_path):
    parent = tmp_path / "RUN_001"
    attempt = _attempt(parent, "running", "RUNNING", last=b"valid")
    decision = discover_run_action(parent, EXPECTED, checkpoint_validator=lambda p: p.read_bytes() == b"valid")
    assert decision.action == "RESUME_TRAIN"
    assert decision.attempt_dir == attempt
    assert decision.checkpoint == attempt / "training_state/last.pt"


def test_corrupt_last_checkpoint_requires_new_attempt_and_preserves_evidence(tmp_path):
    parent = tmp_path / "RUN_001"
    attempt = _attempt(parent, "broken", "RUNNING", last=b"broken")
    decision = discover_run_action(parent, EXPECTED, checkpoint_validator=lambda _: False)
    assert decision.action == "NEW_ATTEMPT"
    assert decision.superseded_attempt == attempt
    assert "checkpoint" in decision.reason.lower()
    assert (attempt / "training_state/last.pt").read_bytes() == b"broken"


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("TRAIN_COMPLETED", "EVALUATE"),
        ("FAILED_EVAL_RETRYABLE", "EVALUATE"),
        ("EVALUATED", "VALIDATE"),
        ("INVALID_ARTIFACT", "VALIDATE"),
    ],
)
def test_restart_continues_from_first_incomplete_phase(tmp_path, state, expected):
    parent = tmp_path / "RUN_001"
    attempt = _attempt(parent, "partial", state)
    decision = discover_run_action(parent, EXPECTED, checkpoint_validator=lambda _: True)
    assert decision.action == expected
    assert decision.attempt_dir == attempt


def test_identity_mismatch_is_never_resumed(tmp_path):
    parent = tmp_path / "RUN_001"
    mismatch = _attempt(parent, "old", "RUNNING", identity={"matrix_sha256": "old"}, last=b"valid")
    decision = discover_run_action(parent, EXPECTED, checkpoint_validator=lambda _: True)
    assert decision.action == "NEW_ATTEMPT"
    assert decision.superseded_attempt == mismatch
    assert decision.reason == "no matching runtime identity"


def test_multiple_matching_active_attempts_are_rejected(tmp_path):
    parent = tmp_path / "RUN_001"
    _attempt(parent, "one", "RUNNING", last=b"valid")
    _attempt(parent, "two", "RUNNING", last=b"valid")
    with pytest.raises(RuntimeError, match="Multiple matching active attempts"):
        discover_run_action(parent, EXPECTED, checkpoint_validator=lambda _: True)
