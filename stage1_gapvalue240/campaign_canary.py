"""Shared-root and ten-machine canary generation/aggregation contracts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import socket
import time
from typing import Any, Iterable, Mapping
import uuid

import pandas as pd
import yaml

from .errors import ValidationError
from .util import atomic_write_bytes, atomic_write_json, sha256_file, stable_hash


COORDINATION_NODE_SCHEMA = "stage1.coordination_root_canary_node.v1"
COORDINATION_AGGREGATE_SCHEMA = "stage1.coordination_root_canary_aggregate.v1"
REAL_DATA_NODE_SCHEMA = "stage1.ten_machine_real_data_canary_node.v1"
REAL_DATA_AGGREGATE_SCHEMA = "stage1.ten_machine_real_data_canary_aggregate.v1"


def _exclusive_bytes(path: Path, data: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def _root_identity(root: Path) -> dict[str, Any]:
    marker = root / "COORDINATION_ROOT_IDENTITY.json"
    candidate = {
        "schema_version": "stage1.coordination_root_identity.v1",
        "root_uuid": uuid.uuid4().hex,
        "created_at_unix": time.time(),
    }
    encoded = (json.dumps(candidate, sort_keys=True) + "\n").encode("utf-8")
    if not _exclusive_bytes(marker, encoded):
        deadline = time.monotonic() + 5
        while True:
            try:
                candidate = json.loads(marker.read_text(encoding="utf-8"))
                break
            except (FileNotFoundError, PermissionError, json.JSONDecodeError):
                if time.monotonic() >= deadline:
                    raise ValidationError(f"coordination root identity is not readable: {marker}")
                time.sleep(0.02)
    if candidate.get("schema_version") != "stage1.coordination_root_identity.v1":
        raise ValidationError("coordination root identity schema mismatch")
    root_uuid = str(candidate.get("root_uuid", ""))
    if not root_uuid:
        raise ValidationError("coordination root identity has no root_uuid")
    return candidate


def _filesystem_info(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "resolved_path": str(path.resolve()),
    }
    try:
        stats = os.statvfs(path)
        result.update(
            {
                "block_size": int(stats.f_bsize),
                "fragment_size": int(stats.f_frsize),
                "total_blocks": int(stats.f_blocks),
                "free_blocks": int(stats.f_bavail),
            }
        )
    except (AttributeError, OSError) as exc:
        result["statvfs_error"] = f"{type(exc).__name__}: {exc}"
    if os.name != "nt" and Path("/proc/mounts").is_file():
        try:
            mounts = []
            for line in Path("/proc/mounts").read_text(encoding="utf-8", errors="replace").splitlines():
                fields = line.split()
                if len(fields) >= 3:
                    mounts.append({"device": fields[0], "mount": fields[1], "filesystem": fields[2]})
            best = None
            resolved = path.resolve()
            for item in mounts:
                mount = Path(item["mount"])
                try:
                    resolved.relative_to(mount)
                except ValueError:
                    continue
                if best is None or len(str(mount)) > len(str(best["mount"])):
                    best = item
            result["mount"] = best
        except OSError as exc:
            result["mount_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _timed(operation) -> tuple[Any, float]:
    started = time.perf_counter()
    value = operation()
    return value, time.perf_counter() - started


def run_coordination_root_canary(
    coordination_root: str | Path,
    *,
    machine_id: str,
    campaign_id: str,
    generation: str,
    expected_machine_ids: Iterable[str],
    output_dir: str | Path,
    visibility_timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Run one node's independent shared-filesystem canary command."""

    root = Path(coordination_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    machine = str(machine_id).strip()
    expected = tuple(map(str, expected_machine_ids))
    if not machine or machine not in expected or len(expected) != len(set(expected)):
        raise ValidationError("invalid coordination canary machine identity set")
    if visibility_timeout_seconds <= 0:
        raise ValidationError("visibility timeout must be positive")
    identity = _root_identity(root)
    root_id = str(identity["root_uuid"])
    canary_root = root / "canaries" / str(campaign_id) / str(generation)
    tokens = canary_root / "tokens"
    competition = canary_root / "competition.lock"
    operations: dict[str, Any] = {}
    payload = {
        "machine_id": machine,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "campaign_id": str(campaign_id),
        "generation": str(generation),
        "coordination_root_id": root_id,
        "created_at_unix": time.time(),
        "token": uuid.uuid4().hex,
    }
    token_bytes = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    scratch = canary_root / "scratch" / machine
    scratch.mkdir(parents=True, exist_ok=True)
    write_path = scratch / "write.tmp"
    renamed_path = tokens / f"{machine}.json"
    tokens.mkdir(parents=True, exist_ok=True)
    _, operations["write_seconds"] = _timed(lambda: write_path.write_bytes(token_bytes))
    observed, operations["read_seconds"] = _timed(lambda: write_path.read_bytes())
    if observed != token_bytes:
        raise ValidationError("coordination canary read differs from write")
    _, operations["rename_seconds"] = _timed(lambda: os.replace(write_path, renamed_path))
    if not renamed_path.is_file():
        raise ValidationError("coordination canary rename did not publish token")
    delete_probe = scratch / "delete.tmp"
    delete_probe.write_bytes(b"delete")
    _, operations["delete_seconds"] = _timed(lambda: delete_probe.unlink())
    competition_payload = f"{machine}\n".encode("utf-8")
    winner, operations["exclusive_create_seconds"] = _timed(
        lambda: _exclusive_bytes(competition, competition_payload)
    )
    deadline = time.monotonic() + visibility_timeout_seconds
    token_payloads: dict[str, dict[str, Any]] = {}
    while True:
        token_payloads = {}
        for expected_machine in expected:
            path = tokens / f"{expected_machine}.json"
            try:
                token_payloads[expected_machine] = json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, PermissionError, json.JSONDecodeError):
                continue
        if len(token_payloads) == len(expected):
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    visible = sorted(token_payloads)
    token_hashes = {
        item: sha256_file(tokens / f"{item}.json")
        for item in visible
    }
    issues = []
    if set(visible) != set(expected):
        issues.append(f"not all expected tokens became visible: {visible}")
    for expected_machine, token in token_payloads.items():
        if token.get("machine_id") != expected_machine:
            issues.append(f"token machine identity mismatch: {expected_machine}")
        if token.get("generation") != str(generation) or token.get("campaign_id") != str(campaign_id):
            issues.append(f"token generation/campaign mismatch: {expected_machine}")
        if token.get("coordination_root_id") != root_id:
            issues.append(f"token root identity mismatch: {expected_machine}")
    now = time.time()
    report = {
        "schema_version": COORDINATION_NODE_SCHEMA,
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "machine_id": machine,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "started_at_unix": payload["created_at_unix"],
        "ended_at_unix": now,
        "campaign_id": str(campaign_id),
        "generation": str(generation),
        "coordination_root_id": root_id,
        "coordination_root_path": str(root),
        "atomic_competition": "WINNER" if winner else "LOSER",
        "operations": operations,
        "visible_tokens": visible,
        "token_hashes": token_hashes,
        "clock_diagnostic": {
            "local_minus_root_marker_seconds": now - float(identity.get("created_at_unix", now)),
            "lease_depends_on_clock_sync": False,
        },
        "filesystem": _filesystem_info(root),
    }
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / f"{machine}.json", report, overwrite=True)
    if issues:
        raise ValidationError(f"coordination root canary failed for {machine}")
    return report


def aggregate_coordination_root_canary(
    reports_dir: str | Path,
    *,
    expected_machine_ids: Iterable[str],
    campaign_id: str,
    generation: str,
    output_path: str | Path,
) -> dict[str, Any]:
    reports_root = Path(reports_dir).resolve()
    expected = tuple(map(str, expected_machine_ids))
    issues: list[str] = []
    reports: list[dict[str, Any]] = []
    for path in sorted(reports_root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append(f"unreadable node report {path.name}: {exc}")
            continue
        if payload.get("schema_version") != COORDINATION_NODE_SCHEMA:
            issues.append(f"node report schema mismatch: {path.name}")
        reports.append(payload)
    machines = [str(report.get("machine_id", "")) for report in reports]
    if len(reports) != len(expected):
        issues.append(f"expected {len(expected)} reports, observed {len(reports)}")
    if len(machines) != len(set(machines)):
        issues.append("duplicate machine IDs in coordination canary reports")
    if set(machines) != set(expected):
        issues.append(f"machine set mismatch: {sorted(machines)}")
    if any(report.get("status") != "PASS" for report in reports):
        issues.append("one or more coordination node reports are not PASS")
    if any(report.get("campaign_id") != str(campaign_id) for report in reports):
        issues.append("campaign ID mismatch across node reports")
    if any(report.get("generation") != str(generation) for report in reports):
        issues.append("generation mismatch across node reports")
    root_ids = {str(report.get("coordination_root_id", "")) for report in reports}
    if len(root_ids) != 1 or "" in root_ids:
        issues.append("coordination root identity differs across node reports")
    winner_count = sum(report.get("atomic_competition") == "WINNER" for report in reports)
    if winner_count != 1:
        issues.append(f"atomic competition winner count is {winner_count}, expected 1")
    visible_complete = all(set(report.get("visible_tokens", [])) == set(expected) for report in reports)
    if not visible_complete:
        issues.append("node token visibility matrix is incomplete")
    # Every node must agree on every token hash.
    for machine in expected:
        hashes = {
            str(report.get("token_hashes", {}).get(machine, ""))
            for report in reports
        }
        if len(hashes) != 1 or "" in hashes:
            issues.append(f"token hash disagreement for {machine}")
    aggregate = {
        "schema_version": COORDINATION_AGGREGATE_SCHEMA,
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "campaign_id": str(campaign_id),
        "generation": str(generation),
        "expected_machine_ids": list(expected),
        "node_count": len(reports),
        "atomic_competition_winner_count": winner_count,
        "visible_token_matrix_complete": visible_complete,
        "coordination_root_id": next(iter(root_ids)) if len(root_ids) == 1 else None,
        "node_report_sha256": {
            path.name: sha256_file(path) for path in sorted(reports_root.glob("*.json"))
        },
    }
    atomic_write_json(output_path, aggregate, overwrite=True)
    if issues:
        raise ValidationError(f"coordination root canary aggregation failed; see {output_path}")
    return aggregate


def _machine_config_ids(machine_configs_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted((*machine_configs_dir.glob("*.yaml"), *machine_configs_dir.glob("*.yml"))):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        machine_id = str(payload.get("machine_id", "")) if isinstance(payload, dict) else ""
        if not machine_id or machine_id in result:
            raise ValidationError(f"invalid or duplicate machine config: {path}")
        result[machine_id] = path
    return result


def _quote(token: str) -> str:
    # shlex.quote is also accepted by PowerShell for the path/IDs used here.
    return shlex.quote(str(token))


def build_coordination_canary_commands(
    machine_configs_dir: str | Path,
    *,
    output_dir: str | Path,
    repo_root: str | Path,
    campaign_id: str,
    generation: str,
    expected_machine_ids: Iterable[str],
    coordination_root_placeholder: str = "<SET_SHARED_COORDINATION_ROOT>",
) -> dict[str, Path]:
    configs = _machine_config_ids(Path(machine_configs_dir).resolve())
    expected = tuple(map(str, expected_machine_ids))
    if set(configs) != set(expected):
        raise ValidationError("machine config identities do not match expected canary nodes")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"canary command output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    script = Path(repo_root).resolve() / "scripts/stage1_gapvalue240/run_coordination_root_canary.py"
    expected_csv = ",".join(expected)
    rows = []
    for machine in expected:
        command = " ".join(
            _quote(token)
            for token in (
                "uv",
                "run",
                "python",
                str(script),
                "--coordination-root",
                coordination_root_placeholder,
                "--machine-id",
                machine,
                "--campaign-id",
                campaign_id,
                "--generation",
                generation,
                "--expected-machine-ids",
                expected_csv,
                "--output-dir",
                f"{coordination_root_placeholder}/canaries/{campaign_id}/{generation}/results",
            )
        )
        rows.append(
            {
                "machine_id": machine,
                "machine_config_path": str(configs[machine]),
                "machine_config_sha256": sha256_file(configs[machine]),
                "command": command,
                "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest().upper(),
            }
        )
    commands_path = output / "TEN_MACHINE_COORDINATION_CANARY_COMMANDS.csv"
    atomic_write_bytes(commands_path, pd.DataFrame(rows).to_csv(index=False).encode("utf-8"))
    template_path = output / "TEN_MACHINE_COORDINATION_CANARY_TEMPLATE.json"
    atomic_write_json(
        template_path,
        {
            "schema_version": "stage1.coordination_root_canary_template.v1",
            "status": "OWNER_ACTION_REQUIRED",
            "campaign_id": campaign_id,
            "generation": generation,
            "expected_machine_ids": list(expected),
            "coordination_root_placeholder": coordination_root_placeholder,
            "commands_sha256": sha256_file(commands_path),
            "aggregate_command": (
                "uv run python scripts/stage1_gapvalue240/aggregate_coordination_root_canary.py "
                f"--reports-dir {coordination_root_placeholder}/canaries/{campaign_id}/{generation}/results "
                f"--expected-machine-ids {expected_csv} --campaign-id {campaign_id} "
                f"--generation {generation} --output {coordination_root_placeholder}/canaries/"
                f"{campaign_id}/{generation}/COORDINATION_ROOT_CANARY_AGGREGATE.json"
            ),
        },
    )
    return {"commands_csv": commands_path, "template_json": template_path}


def build_ten_machine_real_data_canary_commands(
    standalone_commands_path: str | Path,
    *,
    output_dir: str | Path,
    expected_machine_ids: Iterable[str],
    canary_job_ids: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    source = Path(standalone_commands_path).resolve()
    frame = pd.read_csv(source, keep_default_na=False)
    required = {"job_id", "assigned_machine_id", "command", "command_sha256"}
    missing = required - set(frame.columns)
    if missing:
        raise ValidationError(f"standalone command table missing columns: {sorted(missing)}")
    expected = tuple(map(str, expected_machine_ids))
    selected_rows = []
    for machine in expected:
        candidates = frame.loc[frame.assigned_machine_id.astype(str).eq(machine)].copy()
        if canary_job_ids is not None:
            job_id = str(canary_job_ids.get(machine, ""))
            candidates = candidates.loc[candidates.job_id.astype(str).eq(job_id)]
        if len(candidates) != 1:
            raise ValidationError(
                f"real-data canary requires exactly one selected job for {machine}, observed {len(candidates)}"
            )
        selected_rows.append(candidates.iloc[0].to_dict())
    selected = pd.DataFrame(selected_rows)
    if selected.job_id.astype(str).duplicated().any():
        raise ValidationError("real-data canary jobs must be unique")
    for row in selected.itertuples(index=False):
        command = str(row.command)
        if command.count("--job-id") != 1 or str(row.job_id) not in command:
            raise ValidationError(f"real-data canary command is not a single-job command: {row.job_id}")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"real-data canary output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    commands = output / "TEN_MACHINE_REAL_DATA_CANARY_COMMANDS.csv"
    atomic_write_bytes(commands, selected.to_csv(index=False).encode("utf-8"))
    template = output / "TEN_MACHINE_REAL_DATA_CANARY_TEMPLATE.json"
    atomic_write_json(
        template,
        {
            "schema_version": "stage1.ten_machine_real_data_canary_template.v1",
            "status": "OWNER_ACTION_REQUIRED",
            "expected_machine_ids": list(expected),
            "job_ids": selected.job_id.astype(str).tolist(),
            "commands_sha256": sha256_file(commands),
            "required_epochs": 1,
            "required_workers": 4,
            "formal_result_schema": REAL_DATA_NODE_SCHEMA,
        },
    )
    return {"commands_csv": commands, "template_json": template}


def aggregate_ten_machine_real_data_canary(
    reports_dir: str | Path,
    *,
    expected_machine_ids: Iterable[str],
    expected_commands_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    expected = tuple(map(str, expected_machine_ids))
    commands = pd.read_csv(expected_commands_path, keep_default_na=False)
    expected_by_machine = {
        str(row.assigned_machine_id): str(row.job_id) for row in commands.itertuples(index=False)
    }
    issues = []
    reports = []
    root = Path(reports_dir).resolve()
    for path in sorted(root.glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append(f"unreadable node report {path.name}: {exc}")
            continue
        reports.append(report)
    machines = [str(report.get("machine_id", "")) for report in reports]
    if len(reports) != 10 or len(expected) != 10:
        issues.append(f"ten-machine canary requires exactly 10 nodes, observed {len(reports)}")
    if len(machines) != len(set(machines)):
        issues.append("duplicate machine IDs in real-data canary reports")
    if set(machines) != set(expected):
        issues.append("real-data canary machine set mismatch")
    checks = {
        "schema_version": REAL_DATA_NODE_SCHEMA,
        "status": "PASS",
        "canonical_lock_validation": "PASS",
        "machine_config_validation": "PASS",
        "dataset_identity_validation": "PASS",
        "workers": 4,
        "lease_validation": "PASS",
        "completed_epochs": 1,
        "telemetry_validation": "PASS",
        "checkpoint_sidecar_validation": "PASS",
        "resource_log_validation": "PASS",
        "gpu_memory_released": True,
        "child_workers_released": True,
    }
    for report in reports:
        machine = str(report.get("machine_id", ""))
        mismatches = {
            key: {"expected": value, "observed": report.get(key)}
            for key, value in checks.items()
            if report.get(key) != value
        }
        if report.get("job_id") != expected_by_machine.get(machine):
            mismatches["job_id"] = {
                "expected": expected_by_machine.get(machine),
                "observed": report.get("job_id"),
            }
        if mismatches:
            issues.append(f"node {machine} validation mismatch: {mismatches}")
    aggregate = {
        "schema_version": REAL_DATA_AGGREGATE_SCHEMA,
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "node_count": len(reports),
        "expected_machine_ids": list(expected),
        "commands_sha256": sha256_file(expected_commands_path),
        "node_report_sha256": {
            path.name: sha256_file(path) for path in sorted(root.glob("*.json"))
        },
    }
    atomic_write_json(output_path, aggregate, overwrite=True)
    if issues:
        raise ValidationError(f"ten-machine real-data canary aggregation failed; see {output_path}")
    return aggregate


__all__ = [
    "COORDINATION_AGGREGATE_SCHEMA",
    "COORDINATION_NODE_SCHEMA",
    "REAL_DATA_AGGREGATE_SCHEMA",
    "REAL_DATA_NODE_SCHEMA",
    "aggregate_coordination_root_canary",
    "aggregate_ten_machine_real_data_canary",
    "build_coordination_canary_commands",
    "build_ten_machine_real_data_canary_commands",
    "run_coordination_root_canary",
]
