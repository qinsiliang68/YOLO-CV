from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Sequence

import psutil

from .errors import ExternalCommandError
from .util import atomic_write_json


def _empty_termination() -> dict:
    return {"terminated_pids": [], "killed_pids": [], "errors": []}


def terminate_process_tree(pid: int, grace_seconds: float = 5.0) -> dict:
    """Terminate a subprocess and every descendant known before termination.

    PyTorch DataLoader workers are separate processes on both Windows and Linux.
    Capturing descendants before terminating the parent prevents them from being
    orphaned when a command times out or the controller is interrupted.
    """

    report = _empty_termination()
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return report

    try:
        descendants = parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
        descendants = []
        report["errors"].append(f"enumerate:{type(exc).__name__}:{exc}")

    targets = descendants + [parent]
    unique = {process.pid: process for process in targets}
    for process in unique.values():
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as exc:
            report["errors"].append(f"terminate:{process.pid}:{type(exc).__name__}:{exc}")

    gone, alive = psutil.wait_procs(list(unique.values()), timeout=grace_seconds)
    report["terminated_pids"] = sorted(process.pid for process in gone)
    for process in alive:
        try:
            process.kill()
            report["killed_pids"].append(process.pid)
        except psutil.NoSuchProcess:
            report["terminated_pids"].append(process.pid)
        except psutil.Error as exc:
            report["errors"].append(f"kill:{process.pid}:{type(exc).__name__}:{exc}")
    if alive:
        psutil.wait_procs(alive, timeout=grace_seconds)
    report["terminated_pids"] = sorted(set(report["terminated_pids"]))
    report["killed_pids"] = sorted(set(report["killed_pids"]))
    return report


def _popen_group_options() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def run_logged(
    command: Sequence[str],
    cwd: str | Path,
    log_path: str | Path,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> dict:
    """Run a command with durable logs and a machine-readable lifecycle result."""

    cwd = Path(cwd)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result_path = log_path.with_suffix(log_path.suffix + ".result.json")
    command_text = list(map(str, command))
    started = time.time()
    merged = os.environ.copy()
    merged.update(env or {})
    proc: subprocess.Popen | None = None
    termination = _empty_termination()

    with log_path.open("wb") as log:
        try:
            proc = subprocess.Popen(
                command_text,
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=merged,
                **_popen_group_options(),
            )
            code = proc.wait(timeout=timeout)
            status = "PASS" if code == 0 else "FAILED"
            timed_out = False
        except subprocess.TimeoutExpired:
            assert proc is not None
            termination = terminate_process_tree(proc.pid)
            try:
                code = proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
                code = proc.wait()
            status = "TIMEOUT"
            timed_out = True
        except BaseException as exc:
            if proc is not None and proc.poll() is None:
                termination = terminate_process_tree(proc.pid)
            code = proc.poll() if proc is not None else None
            result = {
                "schema_version": "stage1_gapvalue240_subprocess_v1",
                "status": "START_FAILED" if proc is None else "ABORTED",
                "command": command_text,
                "cwd": str(cwd),
                "pid": proc.pid if proc is not None else None,
                "returncode": code,
                "timed_out": False,
                "started_at_unix": started,
                "ended_at_unix": time.time(),
                "duration_seconds": time.time() - started,
                "log": str(log_path),
                "termination": termination,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            atomic_write_json(result_path, result, overwrite=True)
            if proc is None:
                raise ExternalCommandError(f"Command failed to start; see {result_path}") from exc
            raise

    ended = time.time()
    result = {
        "schema_version": "stage1_gapvalue240_subprocess_v1",
        "status": status,
        "command": command_text,
        "cwd": str(cwd),
        "pid": proc.pid if proc is not None else None,
        "returncode": code,
        "timed_out": timed_out,
        "started_at_unix": started,
        "ended_at_unix": ended,
        "duration_seconds": ended - started,
        "log": str(log_path),
        "termination": termination,
    }
    atomic_write_json(result_path, result, overwrite=True)
    if timed_out:
        raise ExternalCommandError(f"Command timed out; see {result_path}")
    if code != 0:
        raise ExternalCommandError(f"Command failed ({code}); see {log_path}")
    return result
