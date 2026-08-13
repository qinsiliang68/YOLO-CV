from __future__ import annotations

import json
import re
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence


AUDIT_SCHEMA = "stage1.sctsr.tdd_history_audit.v1"
EVENT_BUNDLE_SCHEMA = "stage1.sctsr.tdd_event_bundle.v1"
REVIEW_IDENTITY = "PRIMARY_AGENT_HISTORY_NOT_INDEPENDENT_REVIEW"
AUDIT_FIELDS = {
    "schema_version",
    "baseline_commit",
    "implementation_source_commit",
    "rollout_provenance",
    "commit_units",
}
PROVENANCE_FIELDS = {
    "rollout_id",
    "reviewer_independence_claim",
    "source_path_claim",
    "prefix_line_count",
    "prefix_sha256",
    "event_bundle_path",
    "event_bundle_bytes",
    "event_bundle_sha256",
}
UNIT_FIELDS = {
    "commit",
    "subject",
    "classification",
    "classification_reason",
    "changed_paths",
    "behavior_claims",
    "commit_event_id",
    "pairs",
}
CLAIM_FIELDS = {"claim_id", "description", "changed_paths", "pair_ids"}
PAIR_FIELDS = {
    "pair_id",
    "test_id",
    "red_event_id",
    "patch_event_ids",
    "historical_green_event_id",
    "exact_green_event_id",
    "expected_failure_pattern",
}
COMMON_EVENT_FIELDS = {
    "event_id",
    "kind",
    "timestamp_utc",
    "source_line",
    "call_id",
    "raw_record",
    "raw_record_sha256",
}
TEST_EVENT_FIELDS = COMMON_EVENT_FIELDS | {
    "exit_code",
    "test_ids",
    "failure_phase",
    "output",
}
COMMIT_EVENT_FIELDS = COMMON_EVENT_FIELDS | {"exit_code", "commit"}
FILE_REF_FIELDS = {"path", "bytes", "sha256"}
ALLOWED_CLASSIFICATIONS = {"BEHAVIOR_CHANGE", "NON_BEHAVIOR_CHANGE"}
ALLOWED_RED_PHASES = {
    "EXPECTED_BEHAVIOR_ASSERTION",
    "EXPECTED_COMMAND_GUARD",
    "EXPECTED_PUBLIC_API_ABSENCE",
}
FORBIDDEN_RED_PATTERNS = (
    "SyntaxError",
    "ModuleNotFoundError",
    "ImportError:",
    "ImportError while importing test module",
    "ERROR collecting",
    "collected 0 items",
)
PUBLIC_API_ABSENCE_PATTERNS = (
    re.compile(r"cannot import name '[A-Za-z_][A-Za-z0-9_]*'"),
    re.compile(r"No module named '(?:stage1_sctsr_v4|scripts\.stage1_sctsr_v4)(?:\.[A-Za-z_][A-Za-z0-9_]*)+'"),
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9A-F]{64}$")
ROLLOUT_ID = re.compile(r"^[0-9a-f-]{36}$")
BEHAVIOR_PATH_PREFIXES = (
    "stage1_sctsr_v4/",
    "scripts/stage1_sctsr_v4/",
    "configs/stage1_sctsr_v4/",
    "integrations/ultralytics/",
)


class TddHistoryAuditError(ValueError):
    """Fail-closed validation error for the historical TDD evidence bundle."""


def _fail(message: str) -> None:
    raise TddHistoryAuditError(message)


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest().upper()


def _mapping(value: Any, *, name: str, fields: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        _fail(f"{name} schema is invalid")
    return value


def _nonempty(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{name} must be nonempty text")
    return value


def _timestamp(value: Any, *, name: str) -> datetime:
    text = _nonempty(value, name=name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TddHistoryAuditError(f"{name} is not an ISO timestamp") from exc
    if parsed.tzinfo is None:
        _fail(f"{name} must include a timezone")
    return parsed


def _resolve_file_ref(root: Path, value: Any, *, name: str) -> Path:
    row = _mapping(value, name=name, fields=FILE_REF_FIELDS)
    relative = Path(_nonempty(row["path"], name=f"{name}.path"))
    if relative.is_absolute():
        _fail(f"{name}.path must be relative to the evidence root")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise TddHistoryAuditError(f"{name}.path escapes the evidence root") from exc
    if not path.is_file():
        _fail(f"{name} file is missing")
    data = path.read_bytes()
    if type(row["bytes"]) is not int or row["bytes"] != len(data) or row["sha256"] != _sha(data):
        _fail(f"{name} bytes or SHA do not match")
    return path


def _load_json(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    path = Path(value)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TddHistoryAuditError("TDD history audit JSON is unavailable or invalid") from exc
    if not isinstance(loaded, Mapping):
        _fail("TDD history audit must be a JSON object")
    return loaded


def _validate_event(event: Any, *, root: Path) -> tuple[Mapping[str, Any], datetime]:
    if not isinstance(event, Mapping):
        _fail("event bundle contains a non-object event")
    kind = event.get("kind")
    fields = TEST_EVENT_FIELDS if kind == "TEST" else COMMIT_EVENT_FIELDS if kind == "COMMIT" else COMMON_EVENT_FIELDS
    row = _mapping(event, name=f"event[{event.get('event_id', '?')}]", fields=fields)
    event_id = _nonempty(row["event_id"], name="event_id")
    if kind not in {"TEST", "PATCH", "COMMIT"}:
        _fail(f"event {event_id} has an unsupported kind")
    if type(row["source_line"]) is not int or row["source_line"] <= 0:
        _fail(f"event {event_id} has an invalid source line")
    _nonempty(row["call_id"], name=f"event {event_id} call_id")
    raw_record = _nonempty(row["raw_record"], name=f"event {event_id} raw_record")
    if not HEX64.fullmatch(str(row["raw_record_sha256"])) or row["raw_record_sha256"] != _sha(raw_record.encode("utf-8")):
        _fail(f"event {event_id} raw record SHA does not match")
    timestamp = _timestamp(row["timestamp_utc"], name=f"event {event_id} timestamp")
    if kind == "TEST":
        if type(row["exit_code"]) is not int:
            _fail(f"test event {event_id} has no integer exit code")
        test_ids = row["test_ids"]
        if not isinstance(test_ids, list) or not test_ids or any(not isinstance(item, str) or not item.strip() for item in test_ids):
            _fail(f"test event {event_id} has no exact test IDs")
        if len(set(test_ids)) != len(test_ids):
            _fail(f"test event {event_id} repeats a test ID")
        _resolve_file_ref(root, row["output"], name=f"test event {event_id} output")
    if kind == "COMMIT":
        if row["exit_code"] != 0 or not HEX40.fullmatch(str(row["commit"])):
            _fail(f"commit event {event_id} is not a successful exact commit")
    return row, timestamp


def validate_tdd_history_audit(
    audit: Mapping[str, Any] | str | Path,
    *,
    evidence_root: str | Path,
    expected_commit_order: Sequence[str],
) -> dict[str, Any]:
    """Validate exact red-patch-green-commit evidence for an implementation range.

    `expected_commit_order` must come from the audited repository's
    `git rev-list --reverse <baseline>..<source>` result. Keeping this argument
    explicit makes unit tests deterministic and prevents a bundle from hiding a
    commit by editing its own inventory.
    """

    raw = _mapping(_load_json(audit), name="audit", fields=AUDIT_FIELDS)
    if raw["schema_version"] != AUDIT_SCHEMA:
        _fail("audit schema version is unsupported")
    baseline = str(raw["baseline_commit"])
    source = str(raw["implementation_source_commit"])
    if not HEX40.fullmatch(baseline) or not HEX40.fullmatch(source):
        _fail("audit commit identity is invalid")
    expected = tuple(expected_commit_order)
    if not expected or any(not HEX40.fullmatch(item) for item in expected):
        _fail("expected commit inventory is invalid")

    root = Path(evidence_root).resolve()
    provenance = _mapping(raw["rollout_provenance"], name="rollout_provenance", fields=PROVENANCE_FIELDS)
    if not ROLLOUT_ID.fullmatch(str(provenance["rollout_id"])):
        _fail("rollout provenance ID is invalid")
    if provenance["reviewer_independence_claim"] != REVIEW_IDENTITY:
        _fail("rollout provenance improperly claims independent review")
    _nonempty(provenance["source_path_claim"], name="rollout source path claim")
    if type(provenance["prefix_line_count"]) is not int or provenance["prefix_line_count"] <= 0:
        _fail("rollout prefix line count is invalid")
    if not HEX64.fullmatch(str(provenance["prefix_sha256"])):
        _fail("rollout prefix SHA is invalid")
    bundle_ref = {
        "path": provenance["event_bundle_path"],
        "bytes": provenance["event_bundle_bytes"],
        "sha256": provenance["event_bundle_sha256"],
    }
    bundle_path = _resolve_file_ref(root, bundle_ref, name="event bundle")
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TddHistoryAuditError("event bundle is not strict UTF-8 JSON") from exc
    bundle = _mapping(bundle, name="event bundle", fields={"schema_version", "events"})
    if bundle["schema_version"] != EVENT_BUNDLE_SCHEMA or not isinstance(bundle["events"], list):
        _fail("event bundle schema is invalid")

    events: dict[str, Mapping[str, Any]] = {}
    times: dict[str, datetime] = {}
    lines: set[int] = set()
    for item in bundle["events"]:
        event, timestamp = _validate_event(item, root=root)
        event_id = str(event["event_id"])
        source_line = int(event["source_line"])
        if event_id in events or source_line in lines:
            _fail("event IDs and source lines must be unique")
        events[event_id] = event
        times[event_id] = timestamp
        lines.add(source_line)

    units = raw["commit_units"]
    if not isinstance(units, list) or not units:
        _fail("audit has no commit inventory")
    actual_order = tuple(str(unit.get("commit")) for unit in units if isinstance(unit, Mapping))
    if actual_order != expected:
        _fail("audit commit inventory differs from the repository commit inventory")

    referenced_events: set[str] = set()
    pair_ids: set[str] = set()
    behavior_count = 0
    pair_count = 0
    for index, value in enumerate(units):
        unit = _mapping(value, name=f"commit_units[{index}]", fields=UNIT_FIELDS)
        commit = str(unit["commit"])
        _nonempty(unit["subject"], name=f"commit {commit} subject")
        classification = str(unit["classification"])
        if classification not in ALLOWED_CLASSIFICATIONS:
            _fail(f"commit {commit} classification is invalid")
        _nonempty(unit["classification_reason"], name=f"commit {commit} classification reason")
        paths = unit["changed_paths"]
        if not isinstance(paths, list) or not paths or any(not isinstance(path, str) or not path.strip() for path in paths):
            _fail(f"commit {commit} changed path inventory is empty")
        commit_event_id = str(unit["commit_event_id"])
        commit_event = events.get(commit_event_id)
        if commit_event is None or commit_event["kind"] != "COMMIT" or commit_event["commit"] != commit:
            _fail(f"commit {commit} lacks its successful commit event")
        referenced_events.add(commit_event_id)
        pairs = unit["pairs"]
        claims = unit["behavior_claims"]
        if not isinstance(pairs, list):
            _fail(f"commit {commit} pairs are not a list")
        if not isinstance(claims, list):
            _fail(f"commit {commit} behavior claims are not a list")
        if classification == "BEHAVIOR_CHANGE" and not pairs:
            _fail(f"behavior commit has no failing-first pair: {commit}")
        if classification == "BEHAVIOR_CHANGE" and not claims:
            _fail(f"behavior commit has no behavior claims: {commit}")
        if classification == "NON_BEHAVIOR_CHANGE" and (pairs or claims):
            _fail(f"non-behavior commit unexpectedly claims behavior or a TDD pair: {commit}")
        if classification == "BEHAVIOR_CHANGE":
            behavior_count += 1
        unit_pair_ids: set[str] = set()
        for pair_value in pairs:
            pair = _mapping(pair_value, name=f"commit {commit} pair", fields=PAIR_FIELDS)
            pair_id = _nonempty(pair["pair_id"], name="pair_id")
            if pair_id in pair_ids:
                _fail(f"duplicate TDD pair ID: {pair_id}")
            pair_ids.add(pair_id)
            unit_pair_ids.add(pair_id)
            test_id = _nonempty(pair["test_id"], name=f"pair {pair_id} test_id")
            pattern = _nonempty(pair["expected_failure_pattern"], name=f"pair {pair_id} expected failure pattern")
            red_id = str(pair["red_event_id"])
            historical_green_id = str(pair["historical_green_event_id"])
            exact_green_id = str(pair["exact_green_event_id"])
            patch_ids = pair["patch_event_ids"]
            if not isinstance(patch_ids, list) or not patch_ids:
                _fail(f"pair {pair_id} has no implementation patch event")
            red = events.get(red_id)
            historical_green = events.get(historical_green_id)
            exact_green = events.get(exact_green_id)
            if (
                red is None
                or historical_green is None
                or exact_green is None
                or red["kind"] != "TEST"
                or historical_green["kind"] != "TEST"
                or exact_green["kind"] != "TEST"
            ):
                _fail(f"pair {pair_id} does not reference red, historical-green, and exact-green test events")
            if red["exit_code"] == 0:
                _fail(f"pair {pair_id} red test must fail")
            if red["failure_phase"] not in ALLOWED_RED_PHASES:
                _fail(f"pair {pair_id} red failure did not reach the intended behavior")
            if historical_green["exit_code"] != 0 or exact_green["exit_code"] != 0:
                _fail(f"pair {pair_id} green test did not pass")
            if test_id not in red["test_ids"] or test_id not in exact_green["test_ids"]:
                _fail(f"pair {pair_id} red and exact green do not use the same exact test ID")
            red_output = _resolve_file_ref(root, red["output"], name=f"pair {pair_id} red output").read_text(encoding="utf-8", errors="strict")
            if red["failure_phase"] == "EXPECTED_PUBLIC_API_ABSENCE":
                if "SyntaxError" in red_output or not any(
                    public_pattern.search(pattern) and public_pattern.search(red_output)
                    for public_pattern in PUBLIC_API_ABSENCE_PATTERNS
                ):
                    _fail(f"pair {pair_id} must explicitly bind the missing public API")
                if pattern not in red_output:
                    _fail(f"pair {pair_id} red output does not contain the expected failure pattern")
            else:
                if any(forbidden in red_output for forbidden in FORBIDDEN_RED_PATTERNS):
                    _fail(f"pair {pair_id} red output is an import, collection, or syntax failure")
                if pattern not in red_output:
                    _fail(f"pair {pair_id} red output does not contain the expected failure pattern")
            patch_times: list[datetime] = []
            for patch_id_value in patch_ids:
                patch_id = str(patch_id_value)
                patch = events.get(patch_id)
                if patch is None or patch["kind"] != "PATCH":
                    _fail(f"pair {pair_id} references a non-patch event")
                patch_times.append(times[patch_id])
                referenced_events.add(patch_id)
            if not (
                times[red_id]
                < min(patch_times)
                <= max(patch_times)
                < times[historical_green_id]
                < times[commit_event_id]
                and times[exact_green_id] >= times[historical_green_id]
            ):
                _fail(f"pair {pair_id} violates red-patch-green-commit chronology")
            referenced_events.update({red_id, historical_green_id, exact_green_id})
            pair_count += 1

        claim_ids: set[str] = set()
        claimed_paths: set[str] = set()
        claimed_pair_ids: set[str] = set()
        for claim_value in claims:
            claim = _mapping(claim_value, name=f"commit {commit} behavior claim", fields=CLAIM_FIELDS)
            claim_id = _nonempty(claim["claim_id"], name=f"commit {commit} behavior claim ID")
            if claim_id in claim_ids:
                _fail(f"commit {commit} repeats a behavior claim ID")
            claim_ids.add(claim_id)
            _nonempty(claim["description"], name=f"claim {claim_id} description")
            claim_paths = claim["changed_paths"]
            if not isinstance(claim_paths, list) or not claim_paths:
                _fail(f"behavior claim {claim_id} has no changed paths")
            if any(path not in paths for path in claim_paths):
                _fail(f"behavior claim {claim_id} cites a path outside its commit")
            claim_pairs = claim["pair_ids"]
            if not isinstance(claim_pairs, list) or not claim_pairs:
                _fail(f"behavior claim has no TDD pair: {claim_id}")
            if any(pair_id not in unit_pair_ids for pair_id in claim_pairs):
                _fail(f"behavior claim {claim_id} cites an unknown pair")
            claimed_paths.update(claim_paths)
            claimed_pair_ids.update(claim_pairs)
        behavior_paths = {path for path in paths if path.startswith(BEHAVIOR_PATH_PREFIXES)}
        if classification == "BEHAVIOR_CHANGE" and not behavior_paths.issubset(claimed_paths):
            _fail(f"behavior source path is not covered by a behavior claim: {commit}")
        if classification == "BEHAVIOR_CHANGE" and claimed_pair_ids != unit_pair_ids:
            _fail(f"commit {commit} has an unclaimed or missing TDD pair")

    if set(events) != referenced_events:
        _fail("event bundle contains unreferenced or omitted selected events")
    return {
        "status": "PASS",
        "commit_count": len(units),
        "behavior_commit_count": behavior_count,
        "non_behavior_commit_count": len(units) - behavior_count,
        "pair_count": pair_count,
        "event_count": len(events),
        "rollout_id": provenance["rollout_id"],
        "reviewer_independence_claim": REVIEW_IDENTITY,
    }
