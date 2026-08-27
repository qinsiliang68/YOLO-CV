"""Read-only daily/cycle campaign reporting. Never mutates or creates scientific jobs."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Iterable

import pandas as pd

from .errors import ValidationError
from .util import atomic_write_json, atomic_write_bytes, sha256_file


DAILY_SCHEMA = "stage1.daily_campaign_status.v1"
CLOSEOUT_SCHEMA = "stage1.cycle_closeout_validation.v1"
VALID_STATES = {"PENDING", "CLAIMED", "RUNNING", "COMPLETE", "FAILED", "RETRY", "FENCED"}
FINAL_STATES = {"COMPLETE", "FAILED", "FENCED"}
FORBIDDEN_PATH_TOKENS = {"blind_holdout", "external_test", "blind/", "external/"}


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"unreadable {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{label} is not a JSON object: {path}")
    return value


def _read_events(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValidationError(f"invalid status JSONL line {number}: {exc}") from exc
        frame = pd.DataFrame(rows)
    else:
        frame = pd.read_csv(path, keep_default_na=False)
    required = {"job_id", "machine_id", "state", "attempt_id", "updated_at_unix", "retry_count", "bytes_written_24h", "disk_free_bytes"}
    missing = required - set(frame.columns)
    if missing:
        raise ValidationError(f"status events missing columns: {sorted(missing)}")
    invalid = sorted(set(frame.state.astype(str).str.upper()) - VALID_STATES)
    if invalid:
        raise ValidationError(f"status events contain unsupported states: {invalid}")
    if any(token in str(path).replace("\\", "/").lower() for token in FORBIDDEN_PATH_TOKENS):
        raise ValidationError("AIOps input may not read blind/external paths")
    return frame


def _latest(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["updated_at_unix"] = pd.to_numeric(work.updated_at_unix, errors="raise")
    work["state"] = work.state.astype(str).str.upper()
    work = work.sort_values(["job_id", "updated_at_unix"], kind="stable")
    return work.groupby("job_id", as_index=False, sort=True).tail(1).sort_values("job_id", kind="stable")


def _machine_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    metrics = [
        "gpu_train_seconds", "dataloader_wait_seconds", "eval_seconds", "write_seconds",
        "queue_idle_seconds", "gpu_util_pct", "gpu_memory_peak_bytes", "cpu_util_pct", "rss_peak_bytes",
    ]
    rows = []
    for machine, group in frame.groupby(frame.machine_id.astype(str), sort=True):
        row: dict[str, Any] = {
            "machine_id": machine,
            "job_count": int(group.job_id.nunique()),
            "failed_job_count": int(group.state.astype(str).str.upper().eq("FAILED").sum()),
            "retry_count": int(pd.to_numeric(group.retry_count, errors="coerce").fillna(0).sum()),
            "disk_free_bytes_min": int(pd.to_numeric(group.disk_free_bytes, errors="coerce").fillna(0).min()),
            "bytes_written_24h": int(pd.to_numeric(group.bytes_written_24h, errors="coerce").fillna(0).sum()),
        }
        for metric in metrics:
            if metric in group:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                row[f"{metric}_sum"] = float(values.sum()) if not values.empty else 0.0
                row[f"{metric}_mean"] = float(values.mean()) if not values.empty else None
        rows.append(row)
    return rows


def build_daily_campaign_status(
    status_events_path: str | Path,
    *,
    release_path: str | Path,
    assignment_manifest_path: str | Path,
    canonical_lock_path: str | Path,
    output_json: str | Path,
    output_markdown: str | Path,
    now_unix: float | None = None,
    preregistered_gate_path: str | Path | None = None,
) -> dict[str, Any]:
    events_path = Path(status_events_path).resolve()
    release_file = Path(release_path).resolve()
    assignment_file = Path(assignment_manifest_path).resolve()
    canonical_file = Path(canonical_lock_path).resolve()
    events = _read_events(events_path)
    latest = _latest(events)
    release = _load_json(release_file, "release")
    assignment = _load_json(assignment_file, "assignment")
    canonical = _load_json(canonical_file, "canonical lock")
    now = time.time() if now_unix is None else float(now_unix)
    counts = {state.lower(): int(latest.state.eq(state).sum()) for state in sorted(VALID_STATES)}
    recent = events.loc[pd.to_numeric(events.updated_at_unix, errors="raise") >= now - 86400]
    failure_reasons = {}
    if "failure_reason" in recent:
        failure_reasons = {
            str(key): int(value)
            for key, value in recent.loc[recent.state.astype(str).str.upper().eq("FAILED"), "failure_reason"]
            .astype(str).replace("", "UNSPECIFIED").value_counts().items()
        }
    completed = set(latest.loc[latest.state.eq("COMPLETE"), "job_id"].astype(str))
    assigned_jobs = list(map(str, assignment.get("job_ids", [])))
    next_ready = next((job for job in assigned_jobs if job not in set(latest.job_id.astype(str))), None)
    gate = None
    if preregistered_gate_path is not None:
        gate_path = Path(preregistered_gate_path).resolve()
        if any(token in str(gate_path).replace("\\", "/").lower() for token in FORBIDDEN_PATH_TOKENS):
            raise ValidationError("stop/scale gate may not come from blind/external paths")
        gate = _load_json(gate_path, "preregistered stop/scale gate")
    report = {
        "schema_version": DAILY_SCHEMA,
        "status": "PASS",
        "created_at_unix": now,
        "read_only": True,
        "scientific_jobs_created": 0,
        "blind_or_external_accessed": False,
        "counts": counts,
        "identities": {
            "release_id": release.get("release_id"),
            "release_sha256": sha256_file(release_file),
            "assignment_id": assignment.get("assignment_id"),
            "assignment_sha256": sha256_file(assignment_file),
            "canonical_lock_file_sha256": sha256_file(canonical_file),
            "queue_registry_sha256": release.get("queue_registry_sha256"),
        },
        "machines": _machine_rows(recent),
        "failure_reasons_24h": failure_reasons,
        "bytes_written_24h": int(pd.to_numeric(recent.bytes_written_24h, errors="coerce").fillna(0).sum()),
        "minimum_reviewable_results": {
            "completed_job_count": len(completed),
            "complete_seed_blocks": int(latest.loc[latest.state.eq("COMPLETE"), "block_id"].nunique()) if "block_id" in latest else 0,
        },
        "next_ready_block": next_ready,
        "preregistered_stop_scale_gate": gate,
        "source_sha256": {
            "status_events": sha256_file(events_path),
            "release": sha256_file(release_file),
            "assignment": sha256_file(assignment_file),
            "canonical_lock": sha256_file(canonical_file),
        },
    }
    atomic_write_json(output_json, report, overwrite=True)
    lines = [
        "# Stage1 Dynamic Replay 每日状态",
        "",
        f"- 生成时间：`{now:.3f}`",
        f"- release：`{report['identities']['release_id']}`",
        f"- assignment：`{report['identities']['assignment_id']}`",
        f"- 最近24小时新增字节：`{report['bytes_written_24h']}`",
        f"- 下一 ready block：`{next_ready or 'NONE'}`",
        "",
        "## 状态计数",
    ]
    lines.extend(f"- {state}: {count}" for state, count in counts.items())
    lines.extend(["", "## 每机资源与失败", ""])
    for machine in report["machines"]:
        lines.append(
            f"- {machine['machine_id']}: jobs={machine['job_count']}, failed={machine['failed_job_count']}, "
            f"retry={machine['retry_count']}, disk_free_min={machine['disk_free_bytes_min']}"
        )
    atomic_write_bytes(output_markdown, ("\n".join(lines) + "\n").encode("utf-8"), overwrite=True)
    return report


def validate_cycle_closeout(
    status_events_path: str | Path,
    *,
    expected_jobs: Iterable[str],
    release_path: str | Path,
    assignment_manifest_path: str | Path,
    canonical_lock_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    events = _read_events(Path(status_events_path).resolve())
    latest = _latest(events)
    expected = tuple(map(str, expected_jobs))
    issues: list[str] = []
    observed = set(latest.job_id.astype(str))
    if observed != set(expected):
        issues.append(f"job set mismatch: missing={sorted(set(expected)-observed)}, extra={sorted(observed-set(expected))}")
    completion_counts = events.loc[events.state.astype(str).str.upper().eq("COMPLETE")].groupby("job_id").attempt_id.nunique()
    duplicates = completion_counts.loc[completion_counts > 1]
    if not duplicates.empty:
        issues.append(f"multiple canonical completions: {duplicates.to_dict()}")
    nonfinal = latest.loc[~latest.state.isin(FINAL_STATES), ["job_id", "state"]]
    if not nonfinal.empty:
        issues.append(f"cycle contains non-final jobs: {nonfinal.to_dict('records')}")
    if "canonical_completion" in events:
        invalid_complete = events.loc[
            events.state.astype(str).str.upper().eq("COMPLETE")
            & ~events.canonical_completion.astype(str).str.lower().isin({"true", "1", "yes"})
        ]
        if not invalid_complete.empty:
            issues.append("COMPLETE attempt not marked canonical_completion")
    report = {
        "schema_version": CLOSEOUT_SCHEMA,
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "expected_job_count": len(expected),
        "observed_job_count": len(observed),
        "state_counts": latest.state.value_counts().to_dict(),
        "identities": {
            "release_sha256": sha256_file(release_path),
            "assignment_sha256": sha256_file(assignment_manifest_path),
            "canonical_lock_file_sha256": sha256_file(canonical_lock_path),
        },
        "blind_or_external_accessed": False,
        "read_only": True,
    }
    atomic_write_json(output_path, report, overwrite=True)
    if issues:
        raise ValidationError(f"cycle closeout validation failed; see {output_path}")
    return report
