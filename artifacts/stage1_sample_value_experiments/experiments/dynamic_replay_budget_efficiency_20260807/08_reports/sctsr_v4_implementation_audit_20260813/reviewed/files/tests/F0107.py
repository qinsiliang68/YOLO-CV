from __future__ import annotations

import pytest

from stage1_sctsr_v4.completion import CompletionAudit, blocker_reason_template, completion_check_template
from stage1_sctsr_v4.errors import ErrorCode, SctsrError


def test_completion_rejects_incomplete_arbitrary_check_map():
    audit = CompletionAudit(
        implementation_status="IMPLEMENTED_NOT_FORMALLY_RUN",
        source_tree_digest="A" * 64,
        checks={"core": "PASS"},
    )
    with pytest.raises(SctsrError) as caught:
        audit.validate()
    assert caught.value.code is ErrorCode.CLOSEOUT_NOT_VALIDATED


def test_implementation_check_may_not_be_hidden_as_blocked():
    audit = CompletionAudit(
        implementation_status="IMPLEMENTED_NOT_FORMALLY_RUN",
        source_tree_digest="A" * 64,
        checks={"prediction_identity": "BLOCKED_WITH_REASON"},
        blocker_reasons={"prediction_identity": "not implemented"},
    )
    with pytest.raises(SctsrError) as caught:
        audit.validate()
    assert caught.value.code is ErrorCode.CLOSEOUT_NOT_VALIDATED


def test_completion_requires_canonical_source_digest():
    audit = CompletionAudit(
        implementation_status="IMPLEMENTED_NOT_FORMALLY_RUN",
        source_tree_digest="not-a-sha",
        checks={},
    )
    with pytest.raises(SctsrError) as caught:
        audit.validate()
    assert caught.value.code is ErrorCode.SOURCE_TREE_MISMATCH


def test_strict_completion_requires_per_check_command_and_log_evidence():
    audit = CompletionAudit(
        implementation_status="IMPLEMENTED_NOT_FORMALLY_RUN",
        source_tree_digest="A" * 64,
        checks=completion_check_template(),
        blocker_reasons=blocker_reason_template(),
    )
    with pytest.raises(SctsrError) as caught:
        audit.validate(require_evidence=True)
    assert caught.value.code is ErrorCode.CLOSEOUT_NOT_VALIDATED


def test_legacy_gate_flags_do_not_masquerade_as_active_v4_side_effects():
    audit = CompletionAudit(
        implementation_status="IMPLEMENTED_NOT_FORMALLY_RUN",
        source_tree_digest="A" * 64,
        checks=completion_check_template(),
        blocker_reasons=blocker_reason_template(),
        legacy_engineering_gate_detected=True,
        legacy_pilot_release_detected=True,
        legacy_assignments_detected=True,
    )
    audit.validate()
    assert not audit.active_v4_engineering_gate_generated
