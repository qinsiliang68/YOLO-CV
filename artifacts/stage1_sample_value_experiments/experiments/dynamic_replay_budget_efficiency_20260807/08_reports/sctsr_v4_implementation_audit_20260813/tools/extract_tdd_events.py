from __future__ import annotations

import argparse
import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping


SELECTION_SCHEMA = "stage1.sctsr.tdd_event_selection.v1"
BUNDLE_SCHEMA = "stage1.sctsr.tdd_event_bundle.v1"
COMMON_SELECTION_FIELDS = {"event_id", "kind", "call_line", "output_line"}
TEST_SELECTION_FIELDS = COMMON_SELECTION_FIELDS | {
    "test_ids",
    "failure_phase",
    "output_path",
}
COMMIT_SELECTION_FIELDS = COMMON_SELECTION_FIELDS | {"commit"}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
EXIT_CODE = re.compile(r"Exit code:\s*(-?\d+)")
HEX_TOKEN = re.compile(r"(?<![0-9a-f])[0-9a-f]{7,40}(?![0-9a-f])")


class TddEventExtractionError(ValueError):
    """Fail-closed error while extracting selected raw rollout events."""


def _fail(message: str) -> None:
    raise TddEventExtractionError(message)


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest().upper()


def _strict_mapping(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        _fail(f"{name} schema is invalid")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{name} must be nonempty text")
    return value


def _flatten_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
            else:
                _fail("selected output contains an unsupported content block")
        return "".join(parts)
    _fail("selected output is not textual")


def _safe_output_path(root: Path, relative_text: str) -> Path:
    relative = Path(_text(relative_text, "output_path"))
    if relative.is_absolute():
        _fail("output_path must be relative")
    result = (root / relative).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise TddEventExtractionError("output_path escapes the evidence root") from exc
    return result


def _file_ref(path: Path, root: Path, data: bytes) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(data),
        "sha256": _sha(data),
    }


def extract_tdd_events(
    *,
    rollout_path: str | Path,
    selection: Mapping[str, Any] | str | Path,
    evidence_root: str | Path,
    bundle_path: str | Path,
) -> dict[str, object]:
    """Extract only registered tool-call facts from an immutable rollout prefix.

    The private rollout itself is never copied. Each selected event embeds the
    exact call/output JSON records and SHA, while the prefix SHA binds every byte
    through the latest selected output line.
    """

    source = Path(rollout_path).resolve()
    if not source.is_file():
        _fail("rollout source is missing")
    if isinstance(selection, (str, Path)):
        try:
            selection_value = json.loads(Path(selection).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TddEventExtractionError("selection JSON is unavailable or invalid") from exc
    else:
        selection_value = selection
    selection_row = _strict_mapping(
        selection_value,
        {"schema_version", "rollout_id", "selections"},
        "selection",
    )
    if selection_row["schema_version"] != SELECTION_SCHEMA:
        _fail("selection schema version is unsupported")
    _text(selection_row["rollout_id"], "rollout_id")
    selections = selection_row["selections"]
    if not isinstance(selections, list) or not selections:
        _fail("selection has no events")

    raw_lines = source.read_bytes().splitlines(keepends=True)
    decoded: dict[int, tuple[str, Mapping[str, Any]]] = {}
    selected_lines: set[int] = set()
    event_ids: set[str] = set()
    max_line = 0
    normalized: list[dict[str, object]] = []

    for index, value in enumerate(selections):
        if not isinstance(value, Mapping):
            _fail(f"selection[{index}] is not an object")
        kind = value.get("kind")
        fields = TEST_SELECTION_FIELDS if kind == "TEST" else COMMIT_SELECTION_FIELDS if kind == "COMMIT" else COMMON_SELECTION_FIELDS
        row = _strict_mapping(value, fields, f"selection[{index}]")
        event_id = _text(row["event_id"], "event_id")
        if event_id in event_ids:
            _fail(f"duplicate event ID: {event_id}")
        event_ids.add(event_id)
        if kind not in {"TEST", "PATCH", "COMMIT"}:
            _fail(f"unsupported event kind: {kind}")
        call_line = row["call_line"]
        output_line = row["output_line"]
        if (
            type(call_line) is not int
            or type(output_line) is not int
            or call_line <= 0
            or output_line <= call_line
            or output_line > len(raw_lines)
        ):
            _fail(f"event {event_id} has invalid rollout line references")
        if call_line in selected_lines or output_line in selected_lines:
            _fail("a raw rollout line is selected more than once")
        selected_lines.update({call_line, output_line})
        max_line = max(max_line, output_line)

        for line_number in (call_line, output_line):
            if line_number not in decoded:
                try:
                    raw_text = raw_lines[line_number - 1].decode("utf-8").rstrip("\r\n")
                    parsed = json.loads(raw_text)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise TddEventExtractionError(f"rollout line {line_number} is not strict UTF-8 JSON") from exc
                if not isinstance(parsed, Mapping):
                    _fail(f"rollout line {line_number} is not a JSON object")
                decoded[line_number] = (raw_text, parsed)
        call_text, call_record = decoded[call_line]
        output_text, output_record = decoded[output_line]
        call_payload = call_record.get("payload")
        output_payload = output_record.get("payload")
        if not isinstance(call_payload, Mapping) or not isinstance(output_payload, Mapping):
            _fail(f"event {event_id} has no tool payload")
        call_id = call_payload.get("call_id")
        if not isinstance(call_id, str) or not call_id or output_payload.get("call_id") != call_id:
            _fail(f"event {event_id} call_id does not match its output")
        timestamp = call_record.get("timestamp")
        _text(timestamp, f"event {event_id} timestamp")
        raw_record = json.dumps(
            {
                "call_line": call_line,
                "call_record": call_text,
                "output_line": output_line,
                "output_record": output_text,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        event: dict[str, object] = {
            "event_id": event_id,
            "kind": kind,
            "timestamp_utc": timestamp,
            "source_line": call_line,
            "call_id": call_id,
            "raw_record": raw_record,
            "raw_record_sha256": _sha(raw_record.encode("utf-8")),
        }
        if kind == "TEST":
            test_ids = row["test_ids"]
            if not isinstance(test_ids, list) or not test_ids or any(not isinstance(item, str) or not item for item in test_ids):
                _fail(f"event {event_id} has no exact test IDs")
            flattened = _flatten_output(output_payload.get("output"))
            exit_matches = EXIT_CODE.findall(flattened)
            if not exit_matches:
                _fail(f"event {event_id} output has no exit code")
            output_file = _safe_output_path(Path(evidence_root).resolve(), str(row["output_path"]))
            output_bytes = flattened.encode("utf-8")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_bytes(output_bytes)
            event.update(
                {
                    "exit_code": int(exit_matches[-1]),
                    "test_ids": test_ids,
                    "failure_phase": row["failure_phase"],
                    "output": _file_ref(output_file, Path(evidence_root).resolve(), output_bytes),
                }
            )
        elif kind == "COMMIT":
            commit = str(row["commit"])
            if not HEX40.fullmatch(commit):
                _fail(f"event {event_id} commit identity is invalid")
            flattened = _flatten_output(output_payload.get("output"))
            exit_matches = EXIT_CODE.findall(flattened)
            if not exit_matches or int(exit_matches[-1]) != 0:
                _fail(f"event {event_id} commit command was not successful")
            tokens = HEX_TOKEN.findall(flattened.lower())
            if not any(commit.startswith(token) for token in tokens):
                _fail(f"event {event_id} commit identity is absent from selected raw output")
            event.update({"exit_code": 0, "commit": commit})
        normalized.append(event)

    root = Path(evidence_root).resolve()
    output_bundle = Path(bundle_path).resolve()
    try:
        output_bundle.relative_to(root)
    except ValueError as exc:
        raise TddEventExtractionError("bundle path escapes the evidence root") from exc
    bundle_bytes = (
        json.dumps(
            {"schema_version": BUNDLE_SCHEMA, "events": normalized},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    output_bundle.parent.mkdir(parents=True, exist_ok=True)
    output_bundle.write_bytes(bundle_bytes)
    prefix_bytes = b"".join(raw_lines[:max_line])
    return {
        "status": "PASS",
        "rollout_id": selection_row["rollout_id"],
        "source_path_claim": source.as_posix(),
        "prefix_line_count": max_line,
        "prefix_sha256": _sha(prefix_bytes),
        "event_count": len(normalized),
        "event_bundle_path": output_bundle.relative_to(root).as_posix(),
        "event_bundle_bytes": len(bundle_bytes),
        "event_bundle_sha256": _sha(bundle_bytes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract SHA-bound SCTSR TDD events from a raw rollout prefix.")
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    try:
        result = extract_tdd_events(
            rollout_path=args.rollout,
            selection=args.selection,
            evidence_root=args.evidence_root,
            bundle_path=args.bundle,
        )
    except TddEventExtractionError as exc:
        result = {"status": "FAIL", "error": str(exc)}
    receipt = Path(args.receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
