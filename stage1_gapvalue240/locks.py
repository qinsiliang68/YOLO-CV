from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import psutil

from .errors import LockHeldError


class RunLock:
    """Small cross-platform exclusive lock backed by O_EXCL.

    AIOps owns process restart and alerting.  A restarted local controller may
    reclaim a lock only when its recorded local PID is provably dead; remote,
    unreadable, and live ownership is never guessed away.
    """

    def __init__(
        self,
        path: str | Path,
        payload: dict | None = None,
        *,
        reclaim_dead_local: bool = False,
    ):
        self.path = Path(path)
        self.payload = dict(payload or {})
        self.reclaim_dead_local = bool(reclaim_dead_local)
        self._held = False

    def _reclaim_if_dead_local(self) -> bool:
        if not self.reclaim_dead_local or not self.path.is_file():
            return False
        try:
            owner = json.loads(self.path.read_text(encoding="utf-8"))
            owner_pid = int(owner["pid"])
            owner_host = str(owner["hostname"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return False
        if owner_host != socket.gethostname():
            return False
        if psutil.pid_exists(owner_pid):
            expected_create_time = owner.get("process_create_time")
            if expected_create_time is None:
                return False
            try:
                actual_create_time = psutil.Process(owner_pid).create_time()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                actual_create_time = None
            if actual_create_time is not None and abs(actual_create_time - float(expected_create_time)) < 1.0:
                return False
        stale = self.path.with_name(f"{self.path.name}.stale.{time.time_ns()}")
        try:
            os.replace(self.path, stale)
        except FileNotFoundError:
            return True
        return True

    def acquire(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            **self.payload,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "created_at_unix": time.time(),
            "process_create_time": psutil.Process(os.getpid()).create_time(),
        }
        for attempt in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                break
            except FileExistsError as exc:
                if attempt == 0 and self._reclaim_if_dead_local():
                    continue
                owner = self.path.read_text(encoding="utf-8", errors="replace") if self.path.exists() else ""
                raise LockHeldError(f"Run lock already exists: {self.path}; owner={owner}") from exc
        else:  # pragma: no cover - loop either opens or raises
            raise AssertionError("unreachable lock acquisition state")
        try:
            os.write(fd, (json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        self._held = True
        return self

    def release(self) -> None:
        if self._held:
            self.path.unlink(missing_ok=True)
            self._held = False

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
