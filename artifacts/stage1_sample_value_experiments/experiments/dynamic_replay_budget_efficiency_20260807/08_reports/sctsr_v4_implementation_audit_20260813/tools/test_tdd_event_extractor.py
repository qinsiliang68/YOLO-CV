from __future__ import annotations

import json
import sys
from hashlib import sha256
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_tdd_events import TddEventExtractionError, extract_tdd_events


def _line(timestamp: str, payload: dict[str, object]) -> bytes:
    return (
        json.dumps(
            {"timestamp": timestamp, "type": "response_item", "payload": payload},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    commit = "b" * 40
    rows = [
        _line(
            "2026-08-13T00:00:01Z",
            {"type": "custom_tool_call", "call_id": "red", "name": "exec", "input": "pytest exact::id"},
        ),
        _line(
            "2026-08-13T00:00:02Z",
            {"type": "custom_tool_call_output", "call_id": "red", "output": "Exit code: 1\nFAILED exact::id - DID_NOT_RAISE\n"},
        ),
        _line(
            "2026-08-13T00:00:03Z",
            {"type": "custom_tool_call", "call_id": "patch", "name": "apply_patch", "input": "*** Begin Patch"},
        ),
        _line(
            "2026-08-13T00:00:04Z",
            {"type": "custom_tool_call_output", "call_id": "patch", "output": "{}"},
        ),
        _line(
            "2026-08-13T00:00:05Z",
            {"type": "custom_tool_call", "call_id": "commit", "name": "exec", "input": "git commit"},
        ),
        _line(
            "2026-08-13T00:00:06Z",
            {"type": "custom_tool_call_output", "call_id": "commit", "output": f"Exit code: 0\n[{commit[:7]}] saved\n"},
        ),
    ]
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_bytes(b"".join(rows))
    selection = {
        "schema_version": "stage1.sctsr.tdd_event_selection.v1",
        "rollout_id": "019fea7a-400b-7322-b14d-252e736f7d39",
        "selections": [
            {
                "event_id": "RED-1",
                "kind": "TEST",
                "call_line": 1,
                "output_line": 2,
                "test_ids": ["exact::id"],
                "failure_phase": "EXPECTED_BEHAVIOR_ASSERTION",
                "output_path": "logs/red.log",
            },
            {"event_id": "PATCH-1", "kind": "PATCH", "call_line": 3, "output_line": 4},
            {"event_id": "COMMIT-1", "kind": "COMMIT", "call_line": 5, "output_line": 6, "commit": commit},
        ],
    }
    return rollout, selection


def test_extracts_sha_bound_events_and_immutable_rollout_prefix(tmp_path):
    rollout, selection = _fixture(tmp_path)
    result = extract_tdd_events(
        rollout_path=rollout,
        selection=selection,
        evidence_root=tmp_path,
        bundle_path=tmp_path / "events.json",
    )
    bundle = json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))
    assert result["prefix_line_count"] == 6
    assert result["prefix_sha256"] == sha256(rollout.read_bytes()).hexdigest().upper()
    assert [row["kind"] for row in bundle["events"]] == ["TEST", "PATCH", "COMMIT"]
    assert bundle["events"][0]["exit_code"] == 1
    assert bundle["events"][2]["commit"] == "b" * 40
    assert (tmp_path / "logs" / "red.log").read_text(encoding="utf-8").endswith("DID_NOT_RAISE\n")


def test_rejects_mismatched_call_output_or_missing_exit_code(tmp_path):
    rollout, selection = _fixture(tmp_path)
    data = rollout.read_text(encoding="utf-8").replace('"call_id":"red"', '"call_id":"wrong"', 1)
    rollout.write_text(data, encoding="utf-8", newline="")
    with pytest.raises(TddEventExtractionError, match="call_id"):
        extract_tdd_events(rollout_path=rollout, selection=selection, evidence_root=tmp_path, bundle_path=tmp_path / "events.json")

    rollout, selection = _fixture(tmp_path)
    data = rollout.read_text(encoding="utf-8").replace("Exit code: 1", "no exit")
    rollout.write_text(data, encoding="utf-8", newline="")
    with pytest.raises(TddEventExtractionError, match="exit code"):
        extract_tdd_events(rollout_path=rollout, selection=selection, evidence_root=tmp_path, bundle_path=tmp_path / "events.json")


def test_rejects_commit_not_present_in_selected_raw_output(tmp_path):
    rollout, selection = _fixture(tmp_path)
    selection["selections"][2]["commit"] = "c" * 40
    with pytest.raises(TddEventExtractionError, match="commit identity"):
        extract_tdd_events(rollout_path=rollout, selection=selection, evidence_root=tmp_path, bundle_path=tmp_path / "events.json")
