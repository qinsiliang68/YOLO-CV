"""Auditable standalone-entry, reassignment, and source-immutability validators."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import time
from typing import Any, Iterable

import pandas as pd

from .campaign_assignment import load_campaign_assignment
from .campaign_controller import load_campaign_release
from .errors import ValidationError
from .util import atomic_write_json, sha256_file, stable_hash


STANDALONE_SCHEMA = "stage1.standalone_entry_validation.v1"
REASSIGNMENT_SCHEMA = "stage1.assignment_reassignment_validation.v1"
SOURCE_MANIFEST_SCHEMA = "stage1.source_tree_manifest.v1"
SOURCE_IMMUTABILITY_SCHEMA = "stage1.source_tree_immutability_validation.v1"

_SOURCE_SUFFIXES = {
    ".py",
    ".pyi",
    ".ps1",
    ".sh",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".md",
    ".csv",
}
_IGNORED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
_FORBIDDEN_JOB_OPTIONS = {
    "--job-ids",
    "--job-list",
    "--jobs",
    "--range",
    "--job-range",
    "--count",
    "--next-job",
    "--auto-next",
}


def _queue_identity(queue_dir: Path) -> dict[str, str]:
    validation_path = queue_dir / "RUN_QUEUE_VALIDATION.json"
    registry_path = queue_dir / "JOB_EXECUTION_REGISTRY.csv"
    if not validation_path.is_file() or not registry_path.is_file():
        raise ValidationError(f"queue is incomplete: {queue_dir}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    registry_sha = sha256_file(registry_path)
    if validation.get("schema_version") != "stage1.dynamic_campaign_run_queue.v2":
        raise ValidationError("standalone validation requires run queue v2")
    if validation.get("status") != "PASS":
        raise ValidationError("queue validation is not PASS")
    if str(validation.get("job_registry_sha256", "")).upper() != registry_sha:
        raise ValidationError("queue registry checksum mismatch")
    canonical_sha = str(validation.get("canonical_lock_file_sha256", "")).upper()
    if len(canonical_sha) != 64:
        raise ValidationError("queue has no canonical lock checksum")
    return {
        "queue_registry_sha256": registry_sha,
        "canonical_lock_file_sha256": canonical_sha,
    }


def _safe_command_tokens(command: str) -> list[str]:
    try:
        tokens = shlex.split(str(command), posix=True)
    except ValueError as exc:
        raise ValidationError(f"standalone command cannot be parsed: {command}") from exc
    if not tokens:
        raise ValidationError("standalone command is empty")
    return tokens


def _option_values(tokens: list[str], option: str) -> list[str]:
    values: list[str] = []
    for index, token in enumerate(tokens):
        if token == option:
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                raise ValidationError(f"standalone command option has no value: {option}")
            values.append(tokens[index + 1])
        elif token.startswith(option + "="):
            values.append(token.split("=", 1)[1])
    return values


def _source_rows(root: Path, suffixes: Iterable[str] | None = None) -> list[dict[str, Any]]:
    allowed = set(suffixes or _SOURCE_SUFFIXES)
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if any(part in _IGNORED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() not in allowed:
            continue
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def build_source_tree_manifest(
    source_root: str | Path,
    output_path: str | Path,
    *,
    suffixes: Iterable[str] | None = None,
) -> Path:
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = _source_rows(root, suffixes)
    payload = {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "status": "COMPLETE",
        "source_root_name": root.name,
        "file_count": len(rows),
        "files": rows,
        "root_digest": stable_hash(rows),
    }
    return atomic_write_json(output_path, payload, overwrite=True)


def validate_source_tree_immutability(
    source_root: str | Path,
    baseline_manifest: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(source_root).resolve()
    baseline_path = Path(baseline_manifest).resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("schema_version") != SOURCE_MANIFEST_SCHEMA:
        raise ValidationError("unsupported source tree manifest schema")
    observed = _source_rows(root)
    observed_digest = stable_hash(observed)
    expected_rows = baseline.get("files")
    issues: list[str] = []
    if not isinstance(expected_rows, list):
        issues.append("baseline manifest has no file list")
        expected_rows = []
    if observed != expected_rows:
        before = {str(row["relative_path"]): row for row in expected_rows}
        after = {str(row["relative_path"]): row for row in observed}
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        changed = sorted(
            path
            for path in set(before) & set(after)
            if before[path].get("sha256") != after[path].get("sha256")
            or int(before[path].get("size_bytes", -1)) != int(after[path].get("size_bytes", -2))
        )
        issues.append(
            f"source files changed: added={added[:10]}, removed={removed[:10]}, changed={changed[:10]}"
        )
    report = {
        "schema_version": SOURCE_IMMUTABILITY_SCHEMA,
        "status": "PASS" if not issues else "FAIL",
        "created_at_unix": time.time(),
        "issues": issues,
        "baseline_manifest": str(baseline_path),
        "baseline_manifest_sha256": sha256_file(baseline_path),
        "expected_source_tree_sha256": str(baseline.get("root_digest", "")).upper(),
        "observed_source_tree_sha256": observed_digest,
        "file_count": len(observed),
    }
    atomic_write_json(output_path, report, overwrite=True)
    if issues:
        raise ValidationError(f"source tree immutability validation failed; see {output_path}")
    return report


def validate_standalone_entry(
    queue_dir: str | Path,
    release_path: str | Path,
    assignment_manifest: str | Path,
    *,
    repo_root: str | Path,
    controller_offline_smoke_report: str | Path | None,
    output_path: str | Path,
) -> dict[str, Any]:
    queue = Path(queue_dir).resolve()
    repo = Path(repo_root).resolve()
    queue_identity = _queue_identity(queue)
    assignment = load_campaign_assignment(
        queue,
        release_path,
        assignment_manifest,
        expected_campaign_id=json.loads(Path(assignment_manifest).read_text(encoding="utf-8"))[
            "campaign_id"
        ],
        repo_root=repo,
    )
    release = load_campaign_release(queue, release_path, expected_campaign_id=assignment.campaign_id)
    manifest = json.loads(assignment.manifest_path.read_text(encoding="utf-8"))
    commands_path = assignment.manifest_path.parent / str(manifest["standalone_commands_relpath"])
    if sha256_file(commands_path) != str(manifest["standalone_commands_sha256"]).upper():
        raise ValidationError("standalone commands checksum mismatch")
    commands = pd.read_csv(commands_path, keep_default_na=False)
    required = {
        "job_id",
        "assigned_machine_id",
        "release_id",
        "canonical_lock_file_sha256",
        "command",
        "command_sha256",
    }
    missing = required - set(commands.columns)
    issues: list[str] = []
    if missing:
        issues.append(f"standalone command table missing columns: {sorted(missing)}")
    expected_jobs = list(release.job_ids)
    observed_jobs = commands.job_id.astype(str).tolist() if not missing else []
    if observed_jobs != expected_jobs or len(observed_jobs) != len(set(observed_jobs)):
        issues.append("command rows and released jobs are not one-to-one in release order")
    for row in commands.itertuples(index=False):
        command = str(row.command)
        expected_command_sha = hashlib.sha256(command.encode("utf-8")).hexdigest().upper()
        if expected_command_sha != str(row.command_sha256).upper():
            issues.append(f"command checksum mismatch: {row.job_id}")
            continue
        try:
            tokens = _safe_command_tokens(command)
            job_values = _option_values(tokens, "--job-id")
            if job_values != [str(row.job_id)]:
                issues.append(f"command must contain exactly one matching --job-id: {row.job_id}")
            for forbidden in _FORBIDDEN_JOB_OPTIONS:
                if _option_values(tokens, forbidden) or forbidden in tokens:
                    issues.append(f"forbidden batch option {forbidden}: {row.job_id}")
            worker_tokens = [token for token in tokens if token.replace("\\", "/").endswith(
                "scripts/stage1_gapvalue240/dynamic_campaign_train_worker.py"
            )]
            if len(worker_tokens) != 1:
                issues.append(f"command does not call exactly one single-job worker: {row.job_id}")
            if any("controller" in token.lower() for token in tokens):
                issues.append(f"command routes through controller: {row.job_id}")
            expected_flags = {
                "--machine-config": 1,
                "--campaign-root": 1,
                "--release": 1,
                "--assignment": 1,
                "--expected-release-id": 1,
                "--expected-canonical-lock-sha256": 1,
            }
            for flag, count in expected_flags.items():
                if len(_option_values(tokens, flag)) != count:
                    issues.append(f"command flag count mismatch {flag}: {row.job_id}")
            if _option_values(tokens, "--expected-release-id") != [release.release_id]:
                issues.append(f"release identity flag mismatch: {row.job_id}")
            if _option_values(tokens, "--expected-canonical-lock-sha256") != [
                queue_identity["canonical_lock_file_sha256"]
            ]:
                issues.append(f"canonical lock flag mismatch: {row.job_id}")
        except ValidationError as exc:
            issues.append(f"{row.job_id}: {exc}")
    smoke_status = "NOT_RUN"
    smoke_sha = None
    if controller_offline_smoke_report is None:
        issues.append("controller-offline standalone smoke report is required")
    else:
        smoke_path = Path(controller_offline_smoke_report).resolve()
        if not smoke_path.is_file():
            issues.append(f"missing controller-offline standalone smoke: {smoke_path}")
        else:
            smoke_sha = sha256_file(smoke_path)
            try:
                smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
            except Exception as exc:
                issues.append(f"unreadable standalone smoke report: {exc}")
            else:
                smoke_status = str(smoke.get("status", ""))
                expected_smoke = {
                    "schema_version": "stage1.standalone_job_smoke.v1",
                    "status": "PASS",
                    "controller_offline": True,
                    "single_process": True,
                    "exit_code": 0,
                }
                mismatches = {
                    key: {"expected": value, "observed": smoke.get(key)}
                    for key, value in expected_smoke.items()
                    if smoke.get(key) != value
                }
                if str(smoke.get("job_id", "")) not in set(expected_jobs):
                    mismatches["job_id"] = {
                        "expected": "one released job",
                        "observed": smoke.get("job_id"),
                    }
                if mismatches:
                    issues.append(f"standalone smoke mismatch: {mismatches}")
    source_rows = _source_rows(repo)
    report = {
        "schema_version": STANDALONE_SCHEMA,
        "status": "PASS" if not issues else "FAIL",
        "created_at_unix": time.time(),
        "issues": issues,
        "command_count": len(commands),
        "released_job_count": len(expected_jobs),
        "one_job_per_command": not any("--job-id" in issue for issue in issues),
        "controller_optional": bool(manifest.get("controller_optional")),
        "controller_offline_smoke": smoke_status,
        "controller_offline_smoke_sha256": smoke_sha,
        "assignment_id": assignment.assignment_id,
        "assignment_sha256": assignment.sha256,
        "release_id": release.release_id,
        "release_sha256": release.sha256,
        "identity": {
            **queue_identity,
            "source_tree_sha256": stable_hash(source_rows),
        },
        "commands_sha256": sha256_file(commands_path),
    }
    atomic_write_json(output_path, report, overwrite=True)
    if issues:
        raise ValidationError(f"standalone entry validation failed; see {output_path}")
    return report


_SCIENTIFIC_ASSIGNMENT_COLUMNS = [
    "campaign_id",
    "release_id",
    "release_sha256",
    "queue_registry_sha256",
    "job_id",
    "cycle_id",
    "seed_id",
    "block_id",
    "planned_machine_slot",
    "dependency_job_id",
]


def validate_assignment_reassignment(
    queue_dir: str | Path,
    release_path: str | Path,
    old_assignment_manifest: str | Path,
    new_assignment_manifest: str | Path,
    *,
    source_root: str | Path,
    source_manifest_before: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    queue = Path(queue_dir).resolve()
    old_manifest_path = Path(old_assignment_manifest).resolve()
    new_manifest_path = Path(new_assignment_manifest).resolve()
    old_payload = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    new_payload = json.loads(new_manifest_path.read_text(encoding="utf-8"))
    campaign_id = str(new_payload.get("campaign_id", ""))
    old = load_campaign_assignment(
        queue,
        release_path,
        old_manifest_path,
        expected_campaign_id=campaign_id,
    )
    new = load_campaign_assignment(
        queue,
        release_path,
        new_manifest_path,
        expected_campaign_id=campaign_id,
    )
    issues: list[str] = []
    old_rows = old.rows.sort_values("assignment_order", kind="stable").reset_index(drop=True)
    new_rows = new.rows.sort_values("assignment_order", kind="stable").reset_index(drop=True)
    scientific_equal = False
    try:
        pd.testing.assert_frame_equal(
            old_rows[_SCIENTIFIC_ASSIGNMENT_COLUMNS],
            new_rows[_SCIENTIFIC_ASSIGNMENT_COLUMNS],
            check_dtype=False,
        )
        scientific_equal = True
    except AssertionError as exc:
        issues.append(f"scientific identity changed: {exc}")
    expected_parent_sha = sha256_file(old_manifest_path)
    if str(new_payload.get("supersedes_assignment_sha256", "")).upper() != expected_parent_sha:
        issues.append("new assignment parent SHA does not bind the old assignment")
    if not str(new_payload.get("reassignment_reason", "")).strip():
        issues.append("reassignment reason is empty")
    placement_columns = ["assigned_machine_id", "machine_config_path", "machine_config_sha256"]
    placement_changed = not old_rows[placement_columns].equals(new_rows[placement_columns])
    if not placement_changed:
        issues.append("reassignment did not change any placement field")
    source_validation_path = Path(output_path).with_name("SOURCE_TREE_IMMUTABILITY_VALIDATION.json")
    try:
        source_report = validate_source_tree_immutability(
            source_root,
            source_manifest_before,
            source_validation_path,
        )
        source_unchanged = source_report["status"] == "PASS"
    except ValidationError as exc:
        source_unchanged = False
        issues.append(str(exc))
        source_report = json.loads(source_validation_path.read_text(encoding="utf-8"))
    queue_identity = _queue_identity(queue)
    report = {
        "schema_version": REASSIGNMENT_SCHEMA,
        "status": "PASS" if not issues else "FAIL",
        "created_at_unix": time.time(),
        "issues": issues,
        "old_assignment_id": old.assignment_id,
        "old_assignment_sha256": old.sha256,
        "new_assignment_id": new.assignment_id,
        "new_assignment_sha256": new.sha256,
        "parent_assignment_sha256": expected_parent_sha,
        "reassignment_reason": new_payload.get("reassignment_reason"),
        "scientific_identity_columns": _SCIENTIFIC_ASSIGNMENT_COLUMNS,
        "scientific_identity_equal": scientific_equal,
        "placement_changed": placement_changed,
        "old_assignment_preserved": old_manifest_path.is_file(),
        "source_tree_unchanged": source_unchanged,
        "source_tree_validation_sha256": sha256_file(source_validation_path),
        "identity": {
            **queue_identity,
            "source_tree_sha256": source_report.get("observed_source_tree_sha256"),
        },
    }
    atomic_write_json(output_path, report, overwrite=True)
    if issues:
        raise ValidationError(f"assignment reassignment validation failed; see {output_path}")
    return report


__all__ = [
    "REASSIGNMENT_SCHEMA",
    "SOURCE_IMMUTABILITY_SCHEMA",
    "SOURCE_MANIFEST_SCHEMA",
    "STANDALONE_SCHEMA",
    "build_source_tree_manifest",
    "validate_assignment_reassignment",
    "validate_source_tree_immutability",
    "validate_standalone_entry",
]
