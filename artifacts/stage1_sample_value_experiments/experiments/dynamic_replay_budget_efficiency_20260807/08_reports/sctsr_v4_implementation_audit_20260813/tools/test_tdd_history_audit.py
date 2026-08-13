from __future__ import annotations

import json
import sys
from hashlib import sha256
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tdd_history_audit import TddHistoryAuditError, validate_tdd_history_audit


def _sha_bytes(data: bytes) -> str:
    return sha256(data).hexdigest().upper()


def _write_text(path: Path, text: str) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    path.write_bytes(data)
    return {
        "path": path.name,
        "bytes": len(data),
        "sha256": _sha_bytes(data),
    }


def _event(
    root: Path,
    *,
    event_id: str,
    kind: str,
    timestamp: str,
    exit_code: int | None = None,
    test_ids: list[str] | None = None,
    output: str = "",
    failure_phase: str | None = None,
) -> dict[str, object]:
    raw_record = json.dumps(
        {
            "event_id": event_id,
            "kind": kind,
            "timestamp": timestamp,
            "exit_code": exit_code,
            "test_ids": test_ids or [],
            "output": output,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    raw = raw_record.encode("utf-8")
    event: dict[str, object] = {
        "event_id": event_id,
        "kind": kind,
        "timestamp_utc": timestamp,
        "source_line": int(event_id.removeprefix("E")),
        "call_id": f"call_{event_id}",
        "raw_record": raw_record,
        "raw_record_sha256": _sha_bytes(raw),
    }
    if kind == "TEST":
        event.update(
            {
                "exit_code": exit_code,
                "test_ids": test_ids or [],
                "failure_phase": failure_phase,
                "output": _write_text(root / f"{event_id}.log", output),
            }
        )
    elif kind == "COMMIT":
        event.update({"exit_code": exit_code, "commit": "b" * 40})
    return event


def _valid_audit(tmp_path: Path) -> dict[str, object]:
    events = [
        _event(
            tmp_path,
            event_id="E1",
            kind="TEST",
            timestamp="2026-08-13T00:00:01+00:00",
            exit_code=1,
            test_ids=["tests/test_guard.py::test_rejects_forgery"],
            output="FAILED tests/test_guard.py::test_rejects_forgery - Expected DID_NOT_RAISE\n",
            failure_phase="EXPECTED_BEHAVIOR_ASSERTION",
        ),
        _event(tmp_path, event_id="E2", kind="PATCH", timestamp="2026-08-13T00:00:02+00:00"),
        _event(
            tmp_path,
            event_id="E3",
            kind="TEST",
            timestamp="2026-08-13T00:00:03+00:00",
            exit_code=0,
            test_ids=["tests/test_guard.py::test_rejects_forgery"],
            output="1 passed in 0.01s\n",
            failure_phase=None,
        ),
        _event(tmp_path, event_id="E4", kind="COMMIT", timestamp="2026-08-13T00:00:04+00:00", exit_code=0),
        _event(
            tmp_path,
            event_id="E5",
            kind="TEST",
            timestamp="2026-08-13T00:00:05+00:00",
            exit_code=0,
            test_ids=["tests/test_guard.py::test_rejects_forgery"],
            output="1 passed in 0.01s\n",
            failure_phase=None,
        ),
    ]
    bundle_data = json.dumps(
        {"schema_version": "stage1.sctsr.tdd_event_bundle.v1", "events": events},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    bundle = tmp_path / "events.json"
    bundle.write_bytes(bundle_data)
    return {
        "schema_version": "stage1.sctsr.tdd_history_audit.v1",
        "baseline_commit": "a" * 40,
        "implementation_source_commit": "b" * 40,
        "rollout_provenance": {
            "rollout_id": "019fea7a-400b-7322-b14d-252e736f7d39",
            "reviewer_independence_claim": "PRIMARY_AGENT_HISTORY_NOT_INDEPENDENT_REVIEW",
            "source_path_claim": "C:/private/rollout.jsonl",
            "prefix_line_count": 4,
            "prefix_sha256": "C" * 64,
            "event_bundle_path": bundle.name,
            "event_bundle_bytes": len(bundle_data),
            "event_bundle_sha256": _sha_bytes(bundle_data),
        },
        "commit_units": [
            {
                "commit": "b" * 40,
                "subject": "feat: guarded behavior",
                "classification": "BEHAVIOR_CHANGE",
                "classification_reason": "Runtime guard behavior changed.",
                "changed_paths": ["stage1_sctsr_v4/guard.py", "tests/test_guard.py"],
                "behavior_claims": [
                    {
                        "claim_id": "CLAIM-001",
                        "description": "Reject a forged runtime guard input.",
                        "changed_paths": ["stage1_sctsr_v4/guard.py"],
                        "pair_ids": ["PAIR-001"],
                    }
                ],
                "commit_event_id": "E4",
                "pairs": [
                    {
                        "pair_id": "PAIR-001",
                        "test_id": "tests/test_guard.py::test_rejects_forgery",
                        "red_event_id": "E1",
                        "patch_event_ids": ["E2"],
                        "historical_green_event_id": "E3",
                        "exact_green_event_id": "E5",
                        "expected_failure_pattern": "DID_NOT_RAISE",
                    }
                ],
            }
        ],
    }


def _validate(audit: dict[str, object], root: Path) -> dict[str, object]:
    return validate_tdd_history_audit(
        audit,
        evidence_root=root,
        expected_commit_order=("b" * 40,),
    )


def test_accepts_assertion_level_red_then_patch_then_same_test_id_green(tmp_path):
    result = _validate(_valid_audit(tmp_path), tmp_path)
    assert result["status"] == "PASS"
    assert result["behavior_commit_count"] == 1
    assert result["pair_count"] == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("exit_code", 0, "red test must fail"),
        ("failure_phase", "IMPORT_OR_COLLECTION_FAILURE", "intended behavior"),
    ],
)
def test_rejects_non_failing_or_accidental_red(tmp_path, field, value, message):
    audit = _valid_audit(tmp_path)
    bundle = json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))
    bundle["events"][0][field] = value
    data = json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    (tmp_path / "events.json").write_bytes(data)
    audit["rollout_provenance"]["event_bundle_bytes"] = len(data)
    audit["rollout_provenance"]["event_bundle_sha256"] = _sha_bytes(data)
    with pytest.raises(TddHistoryAuditError, match=message):
        _validate(audit, tmp_path)


def test_rejects_import_collection_or_syntax_failure_even_if_labeled_assertion(tmp_path):
    audit = _valid_audit(tmp_path)
    red_log = tmp_path / "E1.log"
    data = b"ERROR collecting tests/test_guard.py\nImportError: missing symbol\n"
    red_log.write_bytes(data)
    bundle = json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))
    bundle["events"][0]["output"].update(bytes=len(data), sha256=_sha_bytes(data))
    bundle_data = json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    (tmp_path / "events.json").write_bytes(bundle_data)
    audit["rollout_provenance"]["event_bundle_bytes"] = len(bundle_data)
    audit["rollout_provenance"]["event_bundle_sha256"] = _sha_bytes(bundle_data)
    with pytest.raises(TddHistoryAuditError, match="import, collection, or syntax"):
        _validate(audit, tmp_path)


def test_accepts_explicit_public_api_absence_red_when_missing_symbol_is_bound(tmp_path):
    audit = _valid_audit(tmp_path)
    red_log = tmp_path / "E1.log"
    data = (
        b"ERROR collecting tests/test_guard.py\n"
        b"ImportError: cannot import name 'validate_formal_endpoint_evidence' "
        b"from 'stage1_sctsr_v4.evaluation'\n"
    )
    red_log.write_bytes(data)
    bundle = json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))
    bundle["events"][0]["failure_phase"] = "EXPECTED_PUBLIC_API_ABSENCE"
    bundle["events"][0]["output"].update(bytes=len(data), sha256=_sha_bytes(data))
    bundle_data = json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    (tmp_path / "events.json").write_bytes(bundle_data)
    audit["rollout_provenance"]["event_bundle_bytes"] = len(bundle_data)
    audit["rollout_provenance"]["event_bundle_sha256"] = _sha_bytes(bundle_data)
    audit["commit_units"][0]["pairs"][0]["expected_failure_pattern"] = (
        "cannot import name 'validate_formal_endpoint_evidence'"
    )
    result = _validate(audit, tmp_path)
    assert result["status"] == "PASS"


def test_rejects_unbound_or_mislabeled_public_api_import_failure(tmp_path):
    audit = _valid_audit(tmp_path)
    red_log = tmp_path / "E1.log"
    data = b"ModuleNotFoundError: No module named 'unrelated_optional_dependency'\n"
    red_log.write_bytes(data)
    bundle = json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))
    bundle["events"][0]["failure_phase"] = "EXPECTED_PUBLIC_API_ABSENCE"
    bundle["events"][0]["output"].update(bytes=len(data), sha256=_sha_bytes(data))
    bundle_data = json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    (tmp_path / "events.json").write_bytes(bundle_data)
    audit["rollout_provenance"]["event_bundle_bytes"] = len(bundle_data)
    audit["rollout_provenance"]["event_bundle_sha256"] = _sha_bytes(bundle_data)
    with pytest.raises(TddHistoryAuditError, match="explicitly bind the missing public API"):
        _validate(audit, tmp_path)

    audit = _valid_audit(tmp_path)
    red_log = tmp_path / "E1.log"
    data = b"ImportError: cannot import name 'guard_runtime' from 'stage1_sctsr_v4.guard'\n"
    red_log.write_bytes(data)
    bundle = json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))
    bundle["events"][0]["output"].update(bytes=len(data), sha256=_sha_bytes(data))
    bundle_data = json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    (tmp_path / "events.json").write_bytes(bundle_data)
    audit["rollout_provenance"]["event_bundle_bytes"] = len(bundle_data)
    audit["rollout_provenance"]["event_bundle_sha256"] = _sha_bytes(bundle_data)
    audit["commit_units"][0]["pairs"][0]["expected_failure_pattern"] = "cannot import name 'guard_runtime'"
    with pytest.raises(TddHistoryAuditError, match="import, collection, or syntax"):
        _validate(audit, tmp_path)


def test_rejects_red_green_test_id_mismatch(tmp_path):
    audit = _valid_audit(tmp_path)
    audit["commit_units"][0]["pairs"][0]["test_id"] = "tests/test_guard.py::test_other"
    with pytest.raises(TddHistoryAuditError, match="same exact test ID"):
        _validate(audit, tmp_path)


def test_accepts_different_historical_regression_scope_when_exact_green_matches_red(tmp_path):
    audit = _valid_audit(tmp_path)
    bundle = json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))
    bundle["events"][2]["test_ids"] = ["tests/test_guard_regression.py"]
    data = json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    (tmp_path / "events.json").write_bytes(data)
    audit["rollout_provenance"]["event_bundle_bytes"] = len(data)
    audit["rollout_provenance"]["event_bundle_sha256"] = _sha_bytes(data)
    result = _validate(audit, tmp_path)
    assert result["status"] == "PASS"


def test_rejects_patch_before_red_or_green_after_commit(tmp_path):
    audit = _valid_audit(tmp_path)
    bundle = json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))
    bundle["events"][1]["timestamp_utc"] = "2026-08-12T23:59:59+00:00"
    data = json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    (tmp_path / "events.json").write_bytes(data)
    audit["rollout_provenance"]["event_bundle_bytes"] = len(data)
    audit["rollout_provenance"]["event_bundle_sha256"] = _sha_bytes(data)
    with pytest.raises(TddHistoryAuditError, match="red-patch-green-commit chronology"):
        _validate(audit, tmp_path)


def test_rejects_behavior_commit_without_pair_and_unregistered_commit(tmp_path):
    audit = _valid_audit(tmp_path)
    audit["commit_units"][0]["pairs"] = []
    with pytest.raises(TddHistoryAuditError, match="behavior commit has no failing-first pair"):
        _validate(audit, tmp_path)
    audit = _valid_audit(tmp_path)
    with pytest.raises(TddHistoryAuditError, match="commit inventory"):
        validate_tdd_history_audit(
            audit,
            evidence_root=tmp_path,
            expected_commit_order=("d" * 40,),
        )


def test_rejects_behavior_source_path_or_claim_without_pair_coverage(tmp_path):
    audit = _valid_audit(tmp_path)
    audit["commit_units"][0]["changed_paths"].append("stage1_sctsr_v4/uncovered.py")
    with pytest.raises(TddHistoryAuditError, match="behavior source path is not covered"):
        _validate(audit, tmp_path)
    audit = _valid_audit(tmp_path)
    audit["commit_units"][0]["behavior_claims"][0]["pair_ids"] = []
    with pytest.raises(TddHistoryAuditError, match="behavior claim has no TDD pair"):
        _validate(audit, tmp_path)


def test_rejects_tampered_event_bundle_raw_record_or_log(tmp_path):
    audit = _valid_audit(tmp_path)
    (tmp_path / "E1.log").write_text("replaced\n", encoding="utf-8")
    with pytest.raises(TddHistoryAuditError, match="bytes or SHA"):
        _validate(audit, tmp_path)

    audit = _valid_audit(tmp_path)
    bundle = json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))
    bundle["events"][0]["raw_record"] += "tampered"
    data = json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    (tmp_path / "events.json").write_bytes(data)
    audit["rollout_provenance"]["event_bundle_bytes"] = len(data)
    audit["rollout_provenance"]["event_bundle_sha256"] = _sha_bytes(data)
    with pytest.raises(TddHistoryAuditError, match="raw record SHA"):
        _validate(audit, tmp_path)
