#!/usr/bin/env python3
"""Build and execute the strict Stage1 tripartite crosswalk evidence.

The legacy matrices remain immutable inputs.  Every migrated row receives one
safe, exact ``rg -n`` source reproduction, an actual exit code, and a dedicated
deterministic JSON result whose SHA-256 is recorded in the v2 matrix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_dynamic_replay_v3.tripartite_crosswalk import (
    CROSSWALK_FIELDS,
    SOURCE_MISSING_REFERENCE,
)

EXPERIMENT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "stage1_sample_value_experiments"
    / "experiments"
    / "dynamic_replay_budget_efficiency_20260807"
)
REPRODUCTION_ROOT = EXPERIMENT_ROOT / "01_field_audit" / "expert_review_reproductions"
RESULT_ROOT = REPRODUCTION_ROOT / "tripartite_reproduction_v2"

_LINE_REFERENCE = re.compile(
    r"^(?P<path>[^:\r\n]+):(?P<start>[1-9]\d*)(?:-(?P<end>[1-9]\d*))?$"
)
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{4,}")
_INVALID_STATUS = "NOT_APPLICABLE_CAPABILITY_ABSENT"
_TOKEN_STOP = {
    "assert",
    "class",
    "False",
    "import",
    "None",
    "raise",
    "return",
    "True",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = (
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_atomic(path, data)


def _write_csv(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(CROSSWALK_FIELDS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _write_atomic(path, buffer.getvalue().encode("utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _normalise_refs(value: str) -> str:
    return ";".join(part.strip().replace("\\", "/") for part in value.split(";") if part.strip())


def _normalise_status(value: str) -> str:
    status = value.strip()
    return "NOT_APPLICABLE" if status == _INVALID_STATUS else status


def _first_reference(raw: str) -> tuple[str, int, int]:
    reference = raw.split(";", 1)[0].strip().replace("\\", "/")
    match = _LINE_REFERENCE.fullmatch(reference)
    if match is None:
        raise ValueError(f"invalid v3 line reference: {reference!r}")
    start = int(match.group("start"))
    end = int(match.group("end") or start)
    return match.group("path"), start, end


def _derive_rg_reproduction(repo_root: Path, refs: str) -> tuple[str, list[str], str]:
    relative, start, end = _first_reference(refs)
    source = (repo_root / relative).resolve()
    source.relative_to(repo_root.resolve())
    if not source.is_file():
        raise ValueError(
            "referenced v3 source is not present in active tree after runtime "
            f"retirement: {relative}"
        )
    lines = source.read_text(encoding="utf-8").splitlines()
    if not lines or start < 1 or end < start or end > len(lines):
        raise ValueError(f"line reference exceeds source: {relative}:{start}-{end}")
    lower = max(0, start - 3)
    upper = min(len(lines), end + 2)
    candidates: list[str] = []
    for line in lines[start - 1 : end] + lines[lower : start - 1] + lines[end:upper]:
        candidates.extend(_TOKEN.findall(line))
    candidates = [token for token in candidates if token not in _TOKEN_STOP]
    if not candidates:
        raise ValueError(f"no safe rg token near source reference: {relative}:{start}-{end}")
    pattern = max(
        dict.fromkeys(candidates),
        key=lambda token: ("_" in token, len(token), token),
    )
    if any(character.isspace() for character in relative) or any(
        character.isspace() for character in pattern
    ):
        raise ValueError("generated rg command would require unresolved shell quoting")
    command = f"rg -n {pattern} {relative}"
    return command, ["rg", "-n", pattern, relative], f"{relative}:{start}-{end}"


def _artifact_reference(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _run_row(
    *,
    repo_root: Path,
    result_dir: Path,
    requirement_id: str,
    v3_refs: str,
) -> tuple[str, int, str, Path, dict[str, Any]]:
    command, argv, source_reference = _derive_rg_reproduction(repo_root, v3_refs)
    result = subprocess.run(
        argv,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", requirement_id)
    artifact = result_dir / f"{safe_id}.json"
    payload = {
        "argv": argv,
        "command": command,
        "exit_code": result.returncode,
        "requirement_id": requirement_id,
        "schema_version": "stage1.tripartite_static_reproduction.v2",
        "scientific_result": False,
        "source_reference": source_reference,
        "stderr": result.stderr,
        "stdout": result.stdout,
        "validation_scope": "STATIC_SOURCE_REPRODUCTION_ONLY",
    }
    _write_json(artifact, payload)
    return command, result.returncode, _sha256(artifact), artifact, payload


def _require(value: str, field: str, requirement_id: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{requirement_id} has empty legacy field {field}")
    return cleaned


def _build_budgeted_row(
    legacy: Mapping[str, str],
    *,
    repo_root: Path,
    result_dir: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    finding_id = _require(legacy.get("finding_id", ""), "finding_id", "BUDGETED")
    requirement_id = f"BUDGETED:{finding_id}"
    source_json = _require(legacy.get("source_json", ""), "source_json", requirement_id)
    source_json = source_json.replace("\\", "/")
    source_line = _require(legacy.get("source_line", ""), "source_line", requirement_id)
    v3_refs = _normalise_refs(
        _require(legacy.get("v3_source_refs", ""), "v3_source_refs", requirement_id)
    )
    command, exit_code, result_sha, artifact, payload = _run_row(
        repo_root=repo_root,
        result_dir=result_dir,
        requirement_id=requirement_id,
        v3_refs=v3_refs,
    )
    observed = _require(legacy.get("observed_result", ""), "observed_result", requirement_id)
    observed += f" [STATIC_REPRODUCTION_ARTIFACT={_artifact_reference(artifact, repo_root)}]"
    row = {
        "requirement_id": requirement_id,
        "overall_status": "NOT_TESTABLE_SOURCE_MISSING",
        "expert_source_status": "NOT_TESTABLE_SOURCE_MISSING",
        "expert_claim_refs": f"{source_json}:{source_line}",
        "expert_source_refs": SOURCE_MISSING_REFERENCE,
        "v3_status": _normalise_status(
            _require(legacy.get("v3_status", ""), "v3_status", requirement_id)
        ),
        "v3_source_refs": v3_refs,
        "reproduction_command": command,
        "exit_code": str(exit_code),
        "result_artifact_sha": result_sha,
        "observed_result": observed,
        "remaining_risk": _require(
            legacy.get("remaining_risk", ""), "remaining_risk", requirement_id
        ),
        "required_action": _require(
            legacy.get("required_action", ""), "required_action", requirement_id
        ),
    }
    result_record = {
        "command": command,
        "exit_code": exit_code,
        "path": str(artifact.resolve()),
        "requirement_id": requirement_id,
        "sha256": result_sha,
        "stdout_line_count": len(str(payload["stdout"]).splitlines()),
    }
    return row, result_record


def _build_dynamic_row(
    legacy: Mapping[str, str],
    *,
    repo_root: Path,
    result_dir: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    finding_id = _require(legacy.get("finding_id", ""), "finding_id", "DYNAMIC")
    requirement_id = f"DYNAMIC:{finding_id}"
    v3_refs = _normalise_refs(
        _require(legacy.get("v3_source_refs", ""), "v3_source_refs", requirement_id)
    )
    command, exit_code, result_sha, artifact, payload = _run_row(
        repo_root=repo_root,
        result_dir=result_dir,
        requirement_id=requirement_id,
        v3_refs=v3_refs,
    )
    observed = _require(legacy.get("observed_result", ""), "observed_result", requirement_id)
    observed += f" [STATIC_REPRODUCTION_ARTIFACT={_artifact_reference(artifact, repo_root)}]"
    status = _normalise_status(
        _require(legacy.get("status", ""), "status", requirement_id)
    )
    row = {
        "requirement_id": requirement_id,
        "overall_status": status,
        "expert_source_status": "CONFIRMED_PRESENT",
        "expert_claim_refs": _normalise_refs(
            _require(legacy.get("expert_review_ref", ""), "expert_review_ref", requirement_id)
        ),
        "expert_source_refs": _normalise_refs(
            _require(legacy.get("expert_source_refs", ""), "expert_source_refs", requirement_id)
        ),
        "v3_status": status,
        "v3_source_refs": v3_refs,
        "reproduction_command": command,
        "exit_code": str(exit_code),
        "result_artifact_sha": result_sha,
        "observed_result": observed,
        "remaining_risk": _require(
            legacy.get("remaining_risk", ""), "remaining_risk", requirement_id
        ),
        "required_action": _require(
            legacy.get("required_action", ""), "required_action", requirement_id
        ),
    }
    result_record = {
        "command": command,
        "exit_code": exit_code,
        "path": str(artifact.resolve()),
        "requirement_id": requirement_id,
        "sha256": result_sha,
        "stdout_line_count": len(str(payload["stdout"]).splitlines()),
    }
    return row, result_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--budgeted-source",
        type=Path,
        default=REPRODUCTION_ROOT / "expert_vs_v3_full_matrix_v1.csv",
    )
    parser.add_argument(
        "--dynamic-source",
        type=Path,
        default=REPRODUCTION_ROOT / "dynamic_review_vs_v3_matrix_v1.csv",
    )
    parser.add_argument(
        "--budgeted-output",
        type=Path,
        default=REPRODUCTION_ROOT / "expert_vs_v3_tripartite_v2.csv",
    )
    parser.add_argument(
        "--dynamic-output",
        type=Path,
        default=REPRODUCTION_ROOT / "dynamic_review_vs_v3_tripartite_v2.csv",
    )
    parser.add_argument("--result-dir", type=Path, default=RESULT_ROOT / "results")
    parser.add_argument(
        "--receipt", type=Path, default=RESULT_ROOT / "EXECUTION_RECEIPT.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    budgeted_legacy = _read_csv(args.budgeted_source)
    dynamic_legacy = _read_csv(args.dynamic_source)
    if len(budgeted_legacy) != 31 or len(dynamic_legacy) != 15:
        raise ValueError(
            "legacy row-count contract failed: "
            f"budgeted={len(budgeted_legacy)} dynamic={len(dynamic_legacy)}"
        )

    budgeted_rows: list[dict[str, str]] = []
    dynamic_rows: list[dict[str, str]] = []
    results: list[dict[str, Any]] = []
    for legacy in budgeted_legacy:
        row, result = _build_budgeted_row(
            legacy, repo_root=repo_root, result_dir=args.result_dir
        )
        budgeted_rows.append(row)
        results.append(result)
    for legacy in dynamic_legacy:
        row, result = _build_dynamic_row(
            legacy, repo_root=repo_root, result_dir=args.result_dir
        )
        dynamic_rows.append(row)
        results.append(result)

    _write_csv(args.budgeted_output, budgeted_rows)
    _write_csv(args.dynamic_output, dynamic_rows)
    nonzero = sum(int(result["exit_code"]) != 0 for result in results)
    receipt = {
        "budgeted_output": {
            "path": str(args.budgeted_output.resolve()),
            "rows": len(budgeted_rows),
            "sha256": _sha256(args.budgeted_output),
        },
        "commands_executed": True,
        "dynamic_output": {
            "path": str(args.dynamic_output.resolve()),
            "rows": len(dynamic_rows),
            "sha256": _sha256(args.dynamic_output),
        },
        "input_sha256": {
            "budgeted": _sha256(args.budgeted_source),
            "dynamic": _sha256(args.dynamic_source),
        },
        "nonzero_exit_count": nonzero,
        "results": results,
        "row_count": len(results),
        "schema_version": "stage1.tripartite_execution_receipt.v2",
        "scientific_result": False,
        "status": "PASS" if nonzero == 0 else "FAIL",
        "validation_scope": "STATIC_SOURCE_REPRODUCTIONS_ONLY",
    }
    _write_json(args.receipt, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if nonzero == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
