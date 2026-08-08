"""Shared-filesystem assignment activation and fenced cross-machine job leases."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import socket
import tempfile
import threading
import time
from typing import Any
import uuid

from .errors import LockHeldError, ValidationError
from .util import atomic_write_json, stable_hash


class CampaignLeaseError(ValidationError):
    """Raised when shared coordination state is malformed."""


class AssignmentInactiveError(CampaignLeaseError):
    """Raised when a worker presents an assignment that is no longer active."""


class LeaseLostError(CampaignLeaseError):
    """Raised after fencing proves that this process no longer owns its job."""


@dataclass(frozen=True)
class ActiveAssignment:
    campaign_id: str
    release_id: str
    assignment_id: str
    assignment_sha256: str
    job_ids: tuple[str, ...]
    activated_at_unix: float
    path: Path


_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _safe_id(value: str, label: str) -> str:
    text = str(value).strip()
    if not text or not _SAFE_ID.fullmatch(text):
        raise CampaignLeaseError(f"unsafe {label}: {value!r}")
    return text


def _validate_sha(value: str, label: str) -> str:
    text = str(value).upper()
    if len(text) != 64 or any(character not in "0123456789ABCDEF" for character in text):
        raise CampaignLeaseError(f"invalid {label} SHA-256")
    return text


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CampaignLeaseError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CampaignLeaseError(f"unreadable {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise CampaignLeaseError(f"{label} is not a JSON object: {path}")
    return payload


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)


_WINDOWS_SHARING_WINERRORS = {32, 33}
_WINDOWS_ACCESS_DENIED = 5


def _parent_is_writable(parent: Path) -> bool:
    """Probe the parent without touching the contested lock path.

    Windows may report ``ERROR_ACCESS_DENIED`` for an O_EXCL create while another
    process still has the directory entry open.  That is a retryable sharing race
    only when the directory itself remains writable.  A real ACL/path failure must
    escape unchanged so operators do not mistake a configuration problem for
    ordinary contention.
    """

    try:
        fd, probe = tempfile.mkstemp(prefix=".coord-write-probe-", dir=parent)
    except OSError:
        return False
    try:
        os.close(fd)
        Path(probe).unlink(missing_ok=True)
    except OSError:
        return False
    return True


def _is_retryable_lock_contention(exc: PermissionError, path: Path) -> bool:
    """Return True only for bounded Windows sharing/contention failures."""

    winerror = getattr(exc, "winerror", None)
    if winerror in _WINDOWS_SHARING_WINERRORS:
        return True
    if winerror == _WINDOWS_ACCESS_DENIED and _parent_is_writable(path.parent):
        return True
    # Some Python/SMB combinations expose only errno=EACCES.  Restrict this
    # fallback to Windows and require proof that the parent remains writable.
    return os.name == "nt" and getattr(exc, "errno", None) == 13 and _parent_is_writable(
        path.parent
    )


class _ControlLock:
    def __init__(
        self,
        root: Path,
        *,
        timeout_seconds: float = 10.0,
        retry_interval_seconds: float = 0.01,
        stale_after_seconds: float = 60.0,
    ):
        self.root = root
        self.path = root / "control.lock"
        self.timeout_seconds = timeout_seconds
        self.retry_interval_seconds = retry_interval_seconds
        self.stale_after_seconds = stale_after_seconds
        self.token = uuid.uuid4().hex
        self.held = False

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            payload = {
                "schema_version": "stage1.coordination_control_lock.v1",
                "token": self.token,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "created_at_unix": time.time(),
            }
            try:
                _exclusive_json(self.path, payload)
                self.held = True
                return self
            except FileExistsError as exc:
                contention_error: OSError = exc
            except PermissionError as exc:
                if not _is_retryable_lock_contention(exc, self.path):
                    raise
                contention_error = exc
            try:
                try:
                    age = time.time() - self.path.stat().st_mtime
                except (FileNotFoundError, PermissionError):
                    if time.monotonic() >= deadline:
                        raise LockHeldError(
                            f"coordination control lock is busy: {self.path}"
                        ) from contention_error
                    time.sleep(self.retry_interval_seconds)
                    continue
                if age > self.stale_after_seconds:
                    stale = self.root / "history/control_locks" / (
                        f"stale_{int(time.time())}_{uuid.uuid4().hex}.json"
                    )
                    stale.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.replace(self.path, stale)
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise LockHeldError(
                        f"coordination control lock is busy: {self.path}"
                    ) from contention_error
                time.sleep(self.retry_interval_seconds)
            except LockHeldError:
                raise

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.held:
            return
        try:
            payload = _read_json(self.path, "coordination control lock")
            if payload.get("token") == self.token:
                self.path.unlink(missing_ok=True)
        except CampaignLeaseError:
            pass
        self.held = False


def _active_path(root: Path) -> Path:
    return root / "active_assignment.json"


def _load_active(root: Path) -> ActiveAssignment:
    path = _active_path(root)
    payload = _read_json(path, "active assignment")
    if payload.get("schema_version") != "stage1.active_campaign_assignment.v1":
        raise CampaignLeaseError("active assignment schema mismatch")
    jobs = tuple(map(str, payload.get("job_ids", [])))
    if not jobs or len(jobs) != len(set(jobs)):
        raise CampaignLeaseError("active assignment job identities are empty or duplicated")
    if payload.get("job_ids_sha256") != stable_hash(list(jobs)):
        raise CampaignLeaseError("active assignment job identity checksum mismatch")
    return ActiveAssignment(
        campaign_id=str(payload["campaign_id"]),
        release_id=str(payload["release_id"]),
        assignment_id=str(payload["assignment_id"]),
        assignment_sha256=_validate_sha(payload["assignment_sha256"], "assignment"),
        job_ids=jobs,
        activated_at_unix=float(payload["activated_at_unix"]),
        path=path,
    )


def _assert_active(
    root: Path,
    *,
    campaign_id: str,
    release_id: str,
    assignment_id: str,
    assignment_sha256: str,
) -> ActiveAssignment:
    active = _load_active(root)
    expected = (
        str(campaign_id),
        str(release_id),
        str(assignment_id),
        str(assignment_sha256).upper(),
    )
    observed = (
        active.campaign_id,
        active.release_id,
        active.assignment_id,
        active.assignment_sha256,
    )
    if observed != expected:
        raise AssignmentInactiveError(
            f"assignment is not active: observed={observed}, requested={expected}"
        )
    return active


def _claim_path(root: Path, job_id: str) -> Path:
    return root / "claims" / f"{_safe_id(job_id, 'job_id')}.json"


def _heartbeat_path(root: Path, job_id: str, token: str) -> Path:
    return root / "heartbeats" / _safe_id(job_id, "job_id") / f"{_safe_id(token, 'lease token')}.json"


def _claim_last_seen(root: Path, claim: dict[str, Any]) -> float:
    heartbeat = _heartbeat_path(root, str(claim["job_id"]), str(claim["lease_token"]))
    if heartbeat.is_file():
        payload = _read_json(heartbeat, "job heartbeat")
        if payload.get("lease_token") != claim.get("lease_token"):
            raise CampaignLeaseError("job heartbeat token mismatch")
        return float(payload.get("heartbeat_at_unix", 0.0))
    return float(claim.get("claimed_at_unix", 0.0))


def _claim_is_stale(root: Path, claim: dict[str, Any], now: float) -> bool:
    ttl = float(claim.get("ttl_seconds", 0.0))
    if ttl <= 0:
        raise CampaignLeaseError("job claim has invalid TTL")
    return now > _claim_last_seen(root, claim) + ttl


def _archive_claim(root: Path, claim_path: Path, claim: dict[str, Any], status: str) -> Path:
    destination = root / "history/claims" / (
        f"{_safe_id(str(claim['job_id']), 'job_id')}_"
        f"{_safe_id(str(claim['lease_token']), 'lease token')}_"
        f"{_safe_id(status, 'claim status')}_{time.time_ns()}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(claim_path, destination)
    return destination


def _reap_stale_claims(root: Path, *, now: float) -> list[Path]:
    reaped: list[Path] = []
    claims = root / "claims"
    if not claims.is_dir():
        return reaped
    for path in sorted(claims.glob("*.json")):
        claim = _read_json(path, "job claim")
        if _claim_is_stale(root, claim, now):
            reaped.append(_archive_claim(root, path, claim, "STALE_REAPED"))
    return reaped


def activate_assignment(
    coordination_root: str | Path,
    *,
    campaign_id: str,
    release_id: str,
    assignment_id: str,
    assignment_sha256: str,
    job_ids: tuple[str, ...],
    expected_previous_assignment_sha256: str | None = None,
) -> ActiveAssignment:
    """Atomically activate one assignment only when no live job claim exists."""

    root = Path(coordination_root).resolve()
    campaign = _safe_id(campaign_id, "campaign_id")
    release = _safe_id(release_id, "release_id")
    assignment = _safe_id(assignment_id, "assignment_id")
    assignment_sha = _validate_sha(assignment_sha256, "assignment")
    jobs = tuple(_safe_id(value, "job_id") for value in job_ids)
    if not jobs or len(jobs) != len(set(jobs)):
        raise CampaignLeaseError("activation job identities are empty or duplicated")
    previous_sha = (
        _validate_sha(expected_previous_assignment_sha256, "previous assignment")
        if expected_previous_assignment_sha256 is not None
        else None
    )
    with _ControlLock(root):
        current = _load_active(root) if _active_path(root).is_file() else None
        if current is not None and (
            current.assignment_id == assignment
            and current.assignment_sha256 == assignment_sha
            and current.job_ids == jobs
        ):
            return current
        if current is None and previous_sha is not None:
            raise CampaignLeaseError("activation expected a previous assignment, but none is active")
        if current is not None and previous_sha != current.assignment_sha256:
            raise CampaignLeaseError("active assignment does not match expected supersession parent")
        _reap_stale_claims(root, now=time.time())
        live_claims = sorted((root / "claims").glob("*.json")) if (root / "claims").is_dir() else []
        if live_claims:
            raise LockHeldError(f"assignment activation is blocked by active claims: {live_claims}")
        activated = time.time()
        atomic_write_json(
            _active_path(root),
            {
                "schema_version": "stage1.active_campaign_assignment.v1",
                "campaign_id": campaign,
                "release_id": release,
                "assignment_id": assignment,
                "assignment_sha256": assignment_sha,
                "job_ids": list(jobs),
                "job_ids_sha256": stable_hash(list(jobs)),
                "activated_at_unix": activated,
                "supersedes_assignment_sha256": current.assignment_sha256 if current else None,
            },
            overwrite=True,
        )
        return _load_active(root)


class JobLease:
    def __init__(
        self,
        *,
        root: Path,
        claim_path: Path,
        heartbeat_path: Path,
        payload: dict[str, Any],
        heartbeat_seconds: float,
    ) -> None:
        self.root = root
        self.claim_path = claim_path
        self.heartbeat_path = heartbeat_path
        self.payload = payload
        self.heartbeat_seconds = heartbeat_seconds
        self._stop = threading.Event()
        self._lost = threading.Event()
        self._lost_error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._released = False

    @property
    def token(self) -> str:
        return str(self.payload["lease_token"])

    def check_now(self) -> None:
        try:
            _assert_active(
                self.root,
                campaign_id=str(self.payload["campaign_id"]),
                release_id=str(self.payload["release_id"]),
                assignment_id=str(self.payload["assignment_id"]),
                assignment_sha256=str(self.payload["assignment_sha256"]),
            )
        except AssignmentInactiveError as exc:
            raise LeaseLostError(f"active assignment changed: {exc}") from exc
        claim = _read_json(self.claim_path, "active job claim")
        if claim.get("lease_token") != self.token:
            raise LeaseLostError("job lease fencing token changed")

    def heartbeat(self) -> None:
        self.check_now()
        atomic_write_json(
            self.heartbeat_path,
            {
                "schema_version": "stage1.job_lease_heartbeat.v1",
                "campaign_id": self.payload["campaign_id"],
                "release_id": self.payload["release_id"],
                "assignment_id": self.payload["assignment_id"],
                "assignment_sha256": self.payload["assignment_sha256"],
                "job_id": self.payload["job_id"],
                "machine_id": self.payload["machine_id"],
                "lease_token": self.token,
                "heartbeat_at_unix": time.time(),
                "status": "HELD",
            },
            overwrite=True,
        )

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            try:
                self.heartbeat()
            except BaseException as exc:
                self._lost_error = exc
                self._lost.set()
                return

    def start_heartbeat(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise CampaignLeaseError("job lease heartbeat is already running")
        self.heartbeat()
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"job-lease-{self.payload['job_id']}",
            daemon=True,
        )
        self._thread.start()

    def raise_if_lost(self) -> None:
        if self._lost.is_set():
            error = self._lost_error or LeaseLostError("job lease heartbeat was lost")
            if isinstance(error, LeaseLostError):
                raise error
            raise LeaseLostError(f"job lease heartbeat failed: {error}") from error

    def release(self, *, status: str) -> None:
        if self._released:
            return
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=self.heartbeat_seconds + 2.0)
        status_value = _safe_id(status, "lease release status")
        with _ControlLock(self.root):
            if self.claim_path.is_file():
                claim = _read_json(self.claim_path, "active job claim")
                if claim.get("lease_token") == self.token:
                    _archive_claim(self.root, self.claim_path, claim, status_value)
            atomic_write_json(
                self.heartbeat_path,
                {
                    "schema_version": "stage1.job_lease_heartbeat.v1",
                    "campaign_id": self.payload["campaign_id"],
                    "release_id": self.payload["release_id"],
                    "assignment_id": self.payload["assignment_id"],
                    "assignment_sha256": self.payload["assignment_sha256"],
                    "job_id": self.payload["job_id"],
                    "machine_id": self.payload["machine_id"],
                    "lease_token": self.token,
                    "heartbeat_at_unix": time.time(),
                    "status": status_value,
                },
                overwrite=True,
            )
        self._released = True

    def __enter__(self) -> "JobLease":
        self.start_heartbeat()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release(status="COMPLETE" if exc_type is None else "FAILED")


def claim_job_lease(
    coordination_root: str | Path,
    *,
    campaign_id: str,
    release_id: str,
    assignment_id: str,
    assignment_sha256: str,
    job_id: str,
    machine_id: str,
    ttl_seconds: float,
    heartbeat_seconds: float,
) -> JobLease:
    """Claim one active-assignment job with an immutable fencing token."""

    if ttl_seconds <= 0 or heartbeat_seconds <= 0 or heartbeat_seconds >= ttl_seconds:
        raise CampaignLeaseError("job lease requires 0 < heartbeat_seconds < ttl_seconds")
    root = Path(coordination_root).resolve()
    job = _safe_id(job_id, "job_id")
    machine = _safe_id(machine_id, "machine_id")
    token = uuid.uuid4().hex
    with _ControlLock(root):
        active = _assert_active(
            root,
            campaign_id=campaign_id,
            release_id=release_id,
            assignment_id=assignment_id,
            assignment_sha256=assignment_sha256,
        )
        if job not in set(active.job_ids):
            raise AssignmentInactiveError(f"job {job} is not part of the active assignment")
        claim_path = _claim_path(root, job)
        if claim_path.is_file():
            existing = _read_json(claim_path, "job claim")
            if _claim_is_stale(root, existing, time.time()):
                _archive_claim(root, claim_path, existing, "STALE_REAPED")
            else:
                raise LockHeldError(f"active job lease already exists: {claim_path}")
        payload = {
            "schema_version": "stage1.job_lease_claim.v1",
            "campaign_id": str(campaign_id),
            "release_id": str(release_id),
            "assignment_id": str(assignment_id),
            "assignment_sha256": str(assignment_sha256).upper(),
            "job_id": job,
            "machine_id": machine,
            "lease_token": token,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "claimed_at_unix": time.time(),
            "ttl_seconds": float(ttl_seconds),
        }
        _exclusive_json(claim_path, payload)
        heartbeat_path = _heartbeat_path(root, job, token)
        atomic_write_json(
            heartbeat_path,
            {
                "schema_version": "stage1.job_lease_heartbeat.v1",
                **{
                    key: payload[key]
                    for key in (
                        "campaign_id",
                        "release_id",
                        "assignment_id",
                        "assignment_sha256",
                        "job_id",
                        "machine_id",
                        "lease_token",
                    )
                },
                "heartbeat_at_unix": time.time(),
                "status": "HELD",
            },
            overwrite=False,
        )
    return JobLease(
        root=root,
        claim_path=claim_path,
        heartbeat_path=heartbeat_path,
        payload=payload,
        heartbeat_seconds=float(heartbeat_seconds),
    )


__all__ = [
    "ActiveAssignment",
    "AssignmentInactiveError",
    "CampaignLeaseError",
    "JobLease",
    "LeaseLostError",
    "activate_assignment",
    "claim_job_lease",
]
