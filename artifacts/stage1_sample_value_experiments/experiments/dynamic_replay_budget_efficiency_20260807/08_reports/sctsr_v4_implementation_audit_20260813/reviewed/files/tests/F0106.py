import pytest

from stage1_sctsr_v4.completion import CompletionAudit, blocker_reason_template, completion_check_template
from stage1_sctsr_v4.errors import SctsrError


def _audit(**overrides):
    values = {
        "implementation_status": "IMPLEMENTED_NOT_FORMALLY_RUN",
        "source_tree_digest": "A" * 64,
        "checks": completion_check_template(),
        "blocker_reasons": blocker_reason_template(),
    }
    values.update(overrides)
    return CompletionAudit(**values)


def test_completion_requires_reason_for_every_blocker():
    _audit().validate()


def test_completion_rejects_prohibited_side_effect():
    with pytest.raises(SctsrError):
        _audit(formal_training_started=True).validate()


def test_completion_state_is_implementation_only():
    audit = _audit()
    audit.validate()
    assert audit.completion_state == "IMPLEMENTATION_ACCEPTANCE_PASS_NOT_TRAINING_AUTHORIZATION"
