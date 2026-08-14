from __future__ import annotations

import pytest

from stage1_sctsr_v4.evidence_runtime import EpochEvidenceRecorder
from stage1_sctsr_v4.formal_training import _abort_failed_epoch


class _RecorderWhoseAbortFails:
    def __init__(self) -> None:
        self.calls = 0

    def abort(self) -> None:
        self.calls += 1
        raise RuntimeError("telemetry stop failed")


class _Transaction:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.fail = fail

    def abort(self, reason: str) -> None:
        self.calls.append(reason)
        if self.fail:
            raise OSError("quarantine rename failed")


def test_cleanup_error_does_not_skip_epoch_transaction_abort_or_replace_primary_failure() -> None:
    recorder = _RecorderWhoseAbortFails()
    transaction = _Transaction()
    primary = ValueError("original training failure")

    _abort_failed_epoch(recorder=recorder, transaction=transaction, primary_error=primary)

    assert recorder.calls == 1
    assert transaction.calls == ["EPOCH_RUNTIME_OR_EVIDENCE_FAILURE"]
    assert str(primary) == "original training failure"
    assert any("telemetry stop failed" in note for note in primary.__notes__)


def test_both_cleanup_failures_are_attached_to_primary_failure() -> None:
    recorder = _RecorderWhoseAbortFails()
    transaction = _Transaction(fail=True)
    primary = ValueError("original training failure")

    _abort_failed_epoch(recorder=recorder, transaction=transaction, primary_error=primary)

    assert transaction.calls == ["EPOCH_RUNTIME_OR_EVIDENCE_FAILURE"]
    assert any("telemetry stop failed" in note for note in primary.__notes__)
    assert any("quarantine rename failed" in note for note in primary.__notes__)


def test_transaction_is_aborted_when_recorder_construction_failed() -> None:
    transaction = _Transaction()
    primary = RuntimeError("recorder construction failed")

    _abort_failed_epoch(recorder=None, transaction=transaction, primary_error=primary)

    assert transaction.calls == ["EPOCH_RUNTIME_OR_EVIDENCE_FAILURE"]
    assert not getattr(primary, "__notes__", [])


class _CleanupComponent:
    def __init__(self, message: str | None = None) -> None:
        self.calls = 0
        self.message = message

    def stop(self) -> None:
        self.calls += 1
        if self.message:
            raise RuntimeError(self.message)

    def abort(self) -> None:
        self.calls += 1
        if self.message:
            raise RuntimeError(self.message)


def test_recorder_abort_attempts_both_writers_when_telemetry_stop_fails() -> None:
    recorder = object.__new__(EpochEvidenceRecorder)
    recorder._closed = False
    recorder.telemetry = _CleanupComponent("telemetry stop failed")
    recorder.occurrence_writer = _CleanupComponent()
    recorder.step_writer = _CleanupComponent()

    with pytest.raises(RuntimeError, match="telemetry stop failed"):
        recorder.abort()

    assert recorder.telemetry.calls == 1
    assert recorder.occurrence_writer.calls == 1
    assert recorder.step_writer.calls == 1
    assert recorder._closed is True
