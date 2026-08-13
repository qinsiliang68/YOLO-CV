from __future__ import annotations

from datetime import datetime, timezone

import pytest

from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.implementation_self_audit import (
    LEGACY_FIELDS,
    SIDE_EFFECT_FIELDS,
    build_implementation_self_audit,
    parse_taskbook_self_audit,
    validate_implementation_self_audit,
)
from stage1_sctsr_v4.serialization import atomic_write_json, sha256_file


def _taskbook(repository_root):
    return repository_root / "artifacts/stage1_sample_value_experiments/experiments/dynamic_replay_budget_efficiency_20260807/03_preregistration_v4_sctsr/SCTSR_EXPERT_IMPLEMENTATION_TASKBOOK.md"


def _audit(repository_root, tmp_path, *, failed_id=None):
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    evidence = tmp_path / "evidence.json"
    source = tmp_path / "source.py"
    test = tmp_path / "test_source.py"
    stdout.write_text("all registered checks executed\n", encoding="utf-8")
    stderr.write_bytes(b"")
    evidence.write_text('{"status":"PASS"}\n', encoding="utf-8")
    source.write_text("VALUE = 1\n", encoding="utf-8")
    test.write_text("def test_value(): assert True\n", encoding="utf-8")
    checks = []
    for requirement in parse_taskbook_self_audit(_taskbook(repository_root)):
        status = "FAIL" if requirement["check_id"] == failed_id else "PASS"
        checks.append(
            {
                **requirement,
                "status": status,
                "evidence_paths": [evidence.name],
                "reproduction_command": "uv run pytest registered-test-id -q",
                "exit_code": 0,
                "stdout_log_path": stdout.name,
                "stdout_log_bytes": stdout.stat().st_size,
                "stdout_log_sha256": sha256_file(stdout),
                "stderr_log_path": stderr.name,
                "stderr_log_bytes": stderr.stat().st_size,
                "stderr_log_sha256": sha256_file(stderr),
                "observed_result": "The registered evidence was evaluated; requirement mismatch recorded." if status == "FAIL" else "The registered evidence satisfies this exact requirement.",
                "expected_result": "The taskbook requirement is satisfied by reproducible evidence.",
                "reviewed_source_files": [source.name],
                "reviewed_test_files": [test.name],
                "remaining_risk": "The scientific method remains untested; this row assesses implementation only.",
                "required_action_if_not_pass": "Restore the taskbook-required evidence and rerun the exact command." if status == "FAIL" else "NOT_APPLICABLE_REQUIREMENT_PASSED",
            }
        )
    return build_implementation_self_audit(
        taskbook_path=_taskbook(repository_root),
        implementation_source_commit="a" * 40,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        generated_by="Codex primary agent self-audit",
        checks=checks,
        side_effects={field: False for field in SIDE_EFFECT_FIELDS},
        legacy_detected={field: True for field in LEGACY_FIELDS},
    )


def test_taskbook_parser_extracts_every_stable_sa_requirement(repository_root):
    requirements = parse_taskbook_self_audit(_taskbook(repository_root))
    assert len(requirements) == 206
    assert requirements[0]["check_id"] == "SA-001"
    assert requirements[-1]["check_id"] == "SA-309"
    assert len({row["check_id"] for row in requirements}) == 206


def test_self_audit_validates_exact_206_row_pass_ledger(repository_root, tmp_path):
    audit = _audit(repository_root, tmp_path)
    result = validate_implementation_self_audit(
        audit,
        taskbook_path=_taskbook(repository_root),
        evidence_root=tmp_path,
    )
    assert result["status"] == "PASS"
    assert result["check_count"] == 206


def test_self_audit_round_trips_canonical_sorted_json(repository_root, tmp_path):
    audit = _audit(repository_root, tmp_path)
    path = tmp_path / "audit.json"
    atomic_write_json(path, audit)
    result = validate_implementation_self_audit(
        path,
        taskbook_path=_taskbook(repository_root),
        evidence_root=tmp_path,
    )
    assert result["status"] == "PASS"


def test_self_audit_preserves_v3_baseline_failure_instead_of_claiming_pass(repository_root, tmp_path):
    audit = _audit(repository_root, tmp_path, failed_id="SA-266")
    result = validate_implementation_self_audit(
        audit,
        taskbook_path=_taskbook(repository_root),
        evidence_root=tmp_path,
    )
    assert result["status"] == "VALID_AUDIT_WITH_FAILURES"
    assert result["overall_status"] == "SELF_AUDIT_FAIL"
    assert result["failed_check_ids"] == ["SA-266"]


def test_self_audit_rejects_log_drift(repository_root, tmp_path):
    audit = _audit(repository_root, tmp_path)
    (tmp_path / "stdout.log").write_text("replaced\n", encoding="utf-8")
    with pytest.raises(SctsrError) as caught:
        validate_implementation_self_audit(
            audit,
            taskbook_path=_taskbook(repository_root),
            evidence_root=tmp_path,
        )
    assert caught.value.code is ErrorCode.CLOSEOUT_NOT_VALIDATED
