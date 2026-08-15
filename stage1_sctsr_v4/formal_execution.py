from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import socket
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from .errors import ErrorCode, SctsrError
from .formal_release import (
    SIGNATURE_ALGORITHM,
    _load_mapping,
    _parse_utc,
    _resolve_secret,
    verify_formal_release,
)
from .serialization import _fsync_directory, atomic_write_bytes, atomic_write_json, canonical_json_bytes, load_json, sha256_file, stable_digest


EXECUTION_TOKEN_SCHEMA = "stage1.sctsr.formal_execution_token.v1"
CLAIM_REGISTRY_SCHEMA = "stage1.sctsr.execution_claim_registry.v1"
FORMAL_EXECUTION_CLAIM_SCHEMA = "stage1.sctsr.formal_execution_claim.v2"
LOGICAL_JOB_FENCE_SCHEMA = "stage1.sctsr.logical_job_fence.v1"
LOGICAL_JOB_HEARTBEAT_SCHEMA = "stage1.sctsr.logical_job_heartbeat.v1"
EXECUTION_ATTEMPT_SNAPSHOT_SCHEMA = "stage1.sctsr.execution_attempt_snapshot.v2"
LOGICAL_JOB_INVARIANT_FIELDS = (
    "run_role",
    "logical_run_id",
    "arm_id",
    "training_seed",
    "output_root_digest",
    "parent_checkpoint_sha256",
    "lineage_digest",
    "schedule_digest",
)
LOGICAL_JOB_LEASE_TIMEOUT_SECONDS = 6 * 60 * 60
LOGICAL_JOB_CONTROL_LOCK_TIMEOUT_SECONDS = 5.0
JOB_BINDING_FIELDS = (
    "action",
    "run_role",
    "logical_run_id",
    "arm_id",
    "training_seed",
    "output_root_digest",
    "parent_checkpoint_sha256",
    "resume_checkpoint_sha256",
    "lineage_digest",
    "schedule_digest",
    "resume_from_receipt_digest",
)
MAX_EXECUTION_TOKEN_LIFETIME = timedelta(hours=24)
REQUIRED_CLAIM_REGISTRY_FIELDS = frozenset(
    {"schema_version", "registry_id", "mode", "state", "experiment_id", "registry_root_digest"}
)
REQUIRED_EXECUTION_TOKEN_FIELDS = frozenset(
    {
        "schema_version",
        "authorization",
        "formal_execution_authorized",
        "execution_id",
        "key_id",
        "signature_algorithm",
        "signature",
        "issued_at_utc",
        "expires_at_utc",
        "nonce",
        "release_id",
        "release_nonce",
        "release_manifest_sha256",
        "claim_registry_digest",
        "job_binding_digest",
        *JOB_BINDING_FIELDS,
    }
)
REQUIRED_EXECUTION_CLAIM_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "execution_id",
        "execution_nonce_sha256",
        "execution_token_sha256",
        "release_id",
        "release_manifest_sha256",
        "claim_registry_digest",
        "job_binding_digest",
        "job_bindings",
        "logical_job_digest",
        "fence_generation",
        "previous_fence_digest",
        "claimed_at_utc",
        "claimant_host",
        "claimant_pid",
        "claim_digest",
    }
)
REQUIRED_EXECUTION_CLAIM_BINDING_FIELDS = frozenset(
    {
        "status",
        "execution_id",
        "claim_path",
        "claim_bytes",
        "claim_sha256",
        "claim_digest",
        "execution_token_sha256",
        "job_binding_digest",
        "logical_job_digest",
        "fence_generation",
        "fence_claim_path",
        "fence_claim_bytes",
        "fence_claim_sha256",
        "fence_digest",
        "lease_heartbeat_path",
        "lease_timeout_seconds",
        "execution_token_path",
        "execution_token_bytes",
        "claim_registry_path",
        "claim_registry_bytes",
        "claim_registry_sha256",
        "claim_registry_digest",
    }
)

REQUIRED_LOGICAL_JOB_FENCE_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "logical_job_digest",
        "fence_generation",
        "previous_fence_digest",
        "execution_id",
        "execution_nonce_sha256",
        "execution_claim_path",
        "execution_claim_sha256",
        "job_invariants",
        "claimed_at_utc",
        "fence_digest",
    }
)
REQUIRED_LOGICAL_JOB_HEARTBEAT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "logical_job_digest",
        "fence_generation",
        "execution_id",
        "fence_digest",
        "renewed_at_utc",
        "heartbeat_digest",
    }
)


def execution_signature_payload(token: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes({key: value for key, value in token.items() if key != "signature"})


def output_root_digest(path: str | Path) -> str:
    return stable_digest({"absolute_output_root": Path(path).resolve().as_posix()})


def build_execution_job_bindings(
    *,
    action: str,
    run_role: str,
    logical_run_id: str,
    arm_id: str,
    training_seed: int,
    output_root: str | Path,
    parent_checkpoint_sha256: str,
    resume_checkpoint_sha256: str,
    lineage_digest: str,
    schedule_digest: str,
    resume_from_receipt_digest: str,
) -> dict[str, Any]:
    return {
        "action": action,
        "run_role": run_role,
        "logical_run_id": logical_run_id,
        "arm_id": arm_id,
        "training_seed": training_seed,
        "output_root_digest": output_root_digest(output_root),
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "resume_checkpoint_sha256": resume_checkpoint_sha256,
        "lineage_digest": lineage_digest,
        "schedule_digest": schedule_digest,
        "resume_from_receipt_digest": resume_from_receipt_digest,
    }


def logical_job_invariants(job_bindings: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable identity shared by START and every RESUME attempt."""

    if set(job_bindings) != set(JOB_BINDING_FIELDS):
        _fail(
            "Logical job identity requires the complete execution job binding",
            field="job_bindings",
            observed=sorted(job_bindings),
            expected=sorted(JOB_BINDING_FIELDS),
        )
    return {field: job_bindings[field] for field in LOGICAL_JOB_INVARIANT_FIELDS}


def logical_job_digest(job_bindings: Mapping[str, Any]) -> str:
    return stable_digest(logical_job_invariants(job_bindings))


def _fence_path(claims_root: Path, job_digest: str, generation: int) -> Path:
    return claims_root / f"{job_digest}.fence_{generation:08d}.json"


def _heartbeat_path(claims_root: Path, job_digest: str) -> Path:
    return claims_root / f"{job_digest}.heartbeat.json"


def _exclusive_write(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("exclusive formal execution write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _logical_job_control_lock(claims_root: Path, job_digest: str) -> Iterator[None]:
    """Serialize fence changes and canonical epoch publication on shared storage.

    The directory lock is intentionally fail-closed.  A machine death inside
    this millisecond-scale critical section leaves evidence that requires an
    operator audit instead of guessing that the lock is stale.
    """

    locks_root = claims_root / ".logical_job_locks"
    locks_root.mkdir(exist_ok=True)
    if locks_root.is_symlink() or not locks_root.is_dir():
        _fail("Logical-job lock root is indirect or invalid", field="claim_registry_root")
    lock_path = locks_root / f"{job_digest}.lock"
    deadline = time.monotonic() + LOGICAL_JOB_CONTROL_LOCK_TIMEOUT_SECONDS
    while True:
        try:
            lock_path.mkdir()
            break
        except FileExistsError as exc:
            if time.monotonic() >= deadline:
                raise SctsrError(
                    ErrorCode.LOGICAL_JOB_LEASE_ACTIVE,
                    "Logical-job control lock is held by another process",
                    artifact_path=lock_path.as_posix(),
                    recoverable=True,
                    required_action="Wait for the active claim/publication critical section; audit a persistent lock before removal.",
                ) from exc
            time.sleep(0.01)
    owner_path = lock_path / "OWNER.json"
    atomic_write_json(
        owner_path,
        {
            "schema_version": "stage1.sctsr.logical_job_control_lock.v1",
            "logical_job_digest": job_digest,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "acquired_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    )
    try:
        yield
    finally:
        owner_path.unlink(missing_ok=True)
        os.rmdir(lock_path)
        _fsync_directory(locks_root)


def _parse_fence_generation(path: Path, job_digest: str) -> int | None:
    match = re.fullmatch(rf"{re.escape(job_digest)}\.fence_(\d{{8}})\.json", path.name)
    return None if match is None else int(match.group(1))


def _load_fence_chain(claims_root: Path, job_digest: str) -> list[tuple[Path, dict[str, Any]]]:
    found: list[tuple[int, Path]] = []
    for path in claims_root.glob(f"{job_digest}.fence_*.json"):
        generation = _parse_fence_generation(path, job_digest)
        if generation is not None:
            found.append((generation, path))
    found.sort(key=lambda item: item[0])
    if [generation for generation, _path in found] != list(range(1, len(found) + 1)):
        raise SctsrError(
            ErrorCode.LOGICAL_JOB_FENCE_CORRUPT,
            "Logical-job fence generations are not one contiguous append-only chain",
            observed=[generation for generation, _path in found],
        )
    previous = "0" * 64
    validated: list[tuple[Path, dict[str, Any]]] = []
    for generation, path in found:
        if path.is_symlink() or not path.is_file():
            raise SctsrError(ErrorCode.LOGICAL_JOB_FENCE_CORRUPT, "Logical-job fence is missing or indirect", artifact_path=path.as_posix())
        value = load_json(path)
        if not isinstance(value, Mapping) or set(value) != REQUIRED_LOGICAL_JOB_FENCE_FIELDS:
            raise SctsrError(ErrorCode.LOGICAL_JOB_FENCE_CORRUPT, "Logical-job fence schema is invalid", artifact_path=path.as_posix())
        core = {key: item for key, item in value.items() if key != "fence_digest"}
        if any(
            (
                value.get("schema_version") != LOGICAL_JOB_FENCE_SCHEMA,
                value.get("status") != "FENCE_CLAIMED",
                value.get("logical_job_digest") != job_digest,
                value.get("fence_generation") != generation,
                value.get("previous_fence_digest") != previous,
                value.get("fence_digest") != stable_digest(core),
                not isinstance(value.get("job_invariants"), Mapping),
                stable_digest(dict(value.get("job_invariants", {}))) != job_digest,
            )
        ):
            raise SctsrError(ErrorCode.LOGICAL_JOB_FENCE_CORRUPT, "Logical-job fence chain is invalid", artifact_path=path.as_posix())
        claim_path = Path(str(value["execution_claim_path"])).resolve()
        if (
            claim_path.parent != claims_root.resolve()
            or claim_path.is_symlink()
            or not claim_path.is_file()
            or sha256_file(claim_path) != value["execution_claim_sha256"]
        ):
            raise SctsrError(
                ErrorCode.LOGICAL_JOB_FENCE_CORRUPT,
                "Logical-job fence does not bind an intact claim in the shared registry",
                artifact_path=path.as_posix(),
            )
        previous = str(value["fence_digest"])
        validated.append((path, dict(value)))
    return validated


def _load_heartbeat(path: Path, *, latest_fence: Mapping[str, Any]) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SctsrError(ErrorCode.LOGICAL_JOB_FENCE_CORRUPT, "Logical-job heartbeat is missing or indirect", artifact_path=path.as_posix())
    value = load_json(path)
    if not isinstance(value, Mapping) or set(value) != REQUIRED_LOGICAL_JOB_HEARTBEAT_FIELDS:
        raise SctsrError(ErrorCode.LOGICAL_JOB_FENCE_CORRUPT, "Logical-job heartbeat schema is invalid", artifact_path=path.as_posix())
    core = {key: item for key, item in value.items() if key != "heartbeat_digest"}
    if any(
        (
            value.get("schema_version") != LOGICAL_JOB_HEARTBEAT_SCHEMA,
            value.get("status") not in {"ACTIVE", "FAILED", "COMPLETE"},
            value.get("logical_job_digest") != latest_fence.get("logical_job_digest"),
            value.get("fence_generation") != latest_fence.get("fence_generation"),
            value.get("execution_id") != latest_fence.get("execution_id"),
            value.get("fence_digest") != latest_fence.get("fence_digest"),
            value.get("heartbeat_digest") != stable_digest(core),
        )
    ):
        raise SctsrError(ErrorCode.LOGICAL_JOB_FENCE_CORRUPT, "Logical-job heartbeat does not bind the current fence", artifact_path=path.as_posix())
    _parse_utc(value.get("renewed_at_utc"), field="logical_job_heartbeat.renewed_at_utc")
    return dict(value)


def _heartbeat_payload(
    *,
    status: str,
    job_digest: str,
    fence: Mapping[str, Any],
    renewed_at: datetime,
) -> dict[str, Any]:
    core = {
        "schema_version": LOGICAL_JOB_HEARTBEAT_SCHEMA,
        "status": status,
        "logical_job_digest": job_digest,
        "fence_generation": fence["fence_generation"],
        "execution_id": fence["execution_id"],
        "fence_digest": fence["fence_digest"],
        "renewed_at_utc": renewed_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    return {**core, "heartbeat_digest": stable_digest(core)}


def _fail(message: str, *, field: str | None = None, observed: Any = None, expected: Any = None) -> None:
    raise SctsrError(
        ErrorCode.FORMAL_EXECUTION_TOKEN_INVALID,
        message,
        failing_field=field,
        observed=observed,
        expected=expected,
        required_action="Provide a new owner-signed one-attempt execution token bound to this exact job.",
    )


def _sha256(value: Any, *, field: str, allow_zero: bool = True) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.upper() != value:
        _fail("Execution token digest is not canonical SHA-256", field=field, observed=value)
    try:
        bytes.fromhex(value)
    except ValueError:
        _fail("Execution token digest is not hexadecimal", field=field, observed=value)
    if not allow_zero and value == "0" * 64:
        _fail("Execution token digest may not be the zero sentinel", field=field)
    return value


def _manifest_sha256(value: Mapping[str, Any] | str | Path) -> str:
    return stable_digest(value) if isinstance(value, Mapping) else sha256_file(value)


def _validate_claim_registry(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if set(value) != REQUIRED_CLAIM_REGISTRY_FIELDS:
        _fail(
            "Execution claim registry schema fields are invalid",
            field="claim_registry",
            observed={
                "missing": sorted(REQUIRED_CLAIM_REGISTRY_FIELDS - set(value)),
                "extra": sorted(set(value) - REQUIRED_CLAIM_REGISTRY_FIELDS),
            },
        )
    expected = {
        "schema_version": CLAIM_REGISTRY_SCHEMA,
        "mode": "SHARED_MULTI_MACHINE_EXCLUSIVE_CREATE",
        "state": "ACTIVE",
        "experiment_id": "dynamic_replay_budget_efficiency_20260807",
    }
    mismatch = {field: {"observed": value.get(field), "expected": target} for field, target in expected.items() if value.get(field) != target}
    if mismatch or not isinstance(value.get("registry_id"), str) or len(str(value["registry_id"])) < 16:
        _fail("Execution claim registry is not the active SCTSR shared registry", field="claim_registry", observed=mismatch or value.get("registry_id"))
    _sha256(value.get("registry_root_digest"), field="registry_root_digest", allow_zero=False)
    return value


def verify_formal_execution_token(
    token: Mapping[str, Any] | str | Path,
    *,
    release: Mapping[str, Any] | str | Path,
    release_trust_policy: Mapping[str, Any] | str | Path,
    expected_release_bindings: Mapping[str, str],
    release_manifest_sha256: str,
    claim_registry: Mapping[str, Any],
    expected_job_bindings: Mapping[str, Any],
    verification_secret: bytes | str | None = None,
    now_utc: datetime | None = None,
) -> Mapping[str, Any]:
    manifest = _load_mapping(token, role="Execution token")
    release_manifest = verify_formal_release(
        release,
        trust_policy=release_trust_policy,
        expected_bindings=expected_release_bindings,
        verification_secret=verification_secret,
        now_utc=now_utc,
    )
    trust = _load_mapping(release_trust_policy, role="Release trust policy")
    registry = _validate_claim_registry(claim_registry)

    if set(manifest) != REQUIRED_EXECUTION_TOKEN_FIELDS:
        _fail(
            "Execution token fields do not exactly match the registered schema",
            observed={
                "missing": sorted(REQUIRED_EXECUTION_TOKEN_FIELDS - set(manifest)),
                "extra": sorted(set(manifest) - REQUIRED_EXECUTION_TOKEN_FIELDS),
            },
        )
    if manifest.get("schema_version") != EXECUTION_TOKEN_SCHEMA:
        _fail("Execution token schema is not supported", field="schema_version", observed=manifest.get("schema_version"), expected=EXECUTION_TOKEN_SCHEMA)
    if manifest.get("authorization") != "SIGNED_SCTSR_V4_FORMAL_EXECUTION" or manifest.get("formal_execution_authorized") is not True:
        _fail("Execution token does not authorize one SCTSR v4 process attempt")
    if manifest.get("signature_algorithm") != SIGNATURE_ALGORITHM:
        _fail("Execution token signature algorithm is not authorized", field="signature_algorithm")

    if not isinstance(manifest.get("execution_id"), str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{7,127}", str(manifest["execution_id"])) is None:
        _fail("Execution ID is missing", field="execution_id")
    if not isinstance(manifest.get("nonce"), str) or len(str(manifest["nonce"])) < 32:
        _fail("Execution nonce must contain at least 32 characters", field="nonce")
    if manifest.get("release_id") != release_manifest.get("release_id"):
        _fail("Execution token references a different release", field="release_id", observed=manifest.get("release_id"), expected=release_manifest.get("release_id"))
    if manifest.get("release_nonce") != release_manifest.get("nonce"):
        _fail("Execution token release nonce differs from the signed matrix release", field="release_nonce")
    if manifest.get("key_id") != release_manifest.get("key_id"):
        _fail("Execution token and release use different signing keys", field="key_id")

    supplied_release_sha = _sha256(release_manifest_sha256, field="release_manifest_sha256", allow_zero=False)
    actual_release_sha = _manifest_sha256(release)
    if supplied_release_sha != actual_release_sha or manifest.get("release_manifest_sha256") != actual_release_sha:
        _fail(
            "Execution token does not bind the exact signed release bytes",
            field="release_manifest_sha256",
            observed={"token": manifest.get("release_manifest_sha256"), "supplied": supplied_release_sha},
            expected=actual_release_sha,
        )
    registry_digest = stable_digest(registry)
    if manifest.get("claim_registry_digest") != registry_digest:
        _fail("Execution token binds a different claim registry", field="claim_registry_digest", observed=manifest.get("claim_registry_digest"), expected=registry_digest)

    if set(expected_job_bindings) != set(JOB_BINDING_FIELDS):
        _fail(
            "Caller did not provide the complete execution job binding",
            field="expected_job_bindings",
            observed=sorted(expected_job_bindings),
            expected=sorted(JOB_BINDING_FIELDS),
        )
    job = {field: manifest.get(field) for field in JOB_BINDING_FIELDS}
    job_mismatch = {
        field: {"token": job[field], "expected": expected_job_bindings[field]}
        for field in JOB_BINDING_FIELDS
        if job[field] != expected_job_bindings[field]
    }
    expected_job_digest = stable_digest(dict(expected_job_bindings))
    if manifest.get("job_binding_digest") != stable_digest(job) or manifest.get("job_binding_digest") != expected_job_digest:
        job_mismatch["job_binding_digest"] = {
            "token": manifest.get("job_binding_digest"),
            "expected": expected_job_digest,
        }
    if job_mismatch:
        _fail("Execution token is not bound to the requested job", field="job_bindings", observed=job_mismatch)

    action = manifest.get("action")
    if action not in {"START", "RESUME"}:
        _fail("Execution action must be START or RESUME", field="action", observed=action)
    if manifest.get("run_role") not in {"COMMON_PARENT", "BRANCH"}:
        _fail("Execution run role is invalid", field="run_role", observed=manifest.get("run_role"))
    if manifest.get("run_role") == "COMMON_PARENT" and manifest.get("arm_id") != "COMMON_PARENT_NR":
        _fail("Common parent token must use COMMON_PARENT_NR", field="arm_id", observed=manifest.get("arm_id"))
    if type(manifest.get("training_seed")) is not int or int(manifest["training_seed"]) < 0:
        _fail("Execution training seed is invalid", field="training_seed", observed=manifest.get("training_seed"))
    if not isinstance(manifest.get("logical_run_id"), str) or not str(manifest["logical_run_id"]).strip():
        _fail("Execution logical run ID is missing", field="logical_run_id")
    for field in (
        "release_manifest_sha256",
        "claim_registry_digest",
        "job_binding_digest",
        "output_root_digest",
        "parent_checkpoint_sha256",
        "resume_checkpoint_sha256",
        "lineage_digest",
        "schedule_digest",
        "resume_from_receipt_digest",
    ):
        _sha256(manifest.get(field), field=field)
    if action == "START" and (
        manifest.get("resume_checkpoint_sha256") != "0" * 64
        or manifest.get("resume_from_receipt_digest") != "0" * 64
    ):
        _fail("START token may not bind a resume checkpoint or receipt", field="action")
    if action == "RESUME":
        _sha256(manifest.get("resume_checkpoint_sha256"), field="resume_checkpoint_sha256", allow_zero=False)
        _sha256(manifest.get("resume_from_receipt_digest"), field="resume_from_receipt_digest", allow_zero=False)

    issued = _parse_utc(manifest.get("issued_at_utc"), field="execution.issued_at_utc")
    expires = _parse_utc(manifest.get("expires_at_utc"), field="execution.expires_at_utc")
    release_issued = _parse_utc(release_manifest.get("issued_at_utc"), field="release.issued_at_utc")
    release_expires = _parse_utc(release_manifest.get("expires_at_utc"), field="release.expires_at_utc")
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if issued > now + timedelta(minutes=5) or expires <= now:
        _fail("Execution token is not currently valid", field="execution_time_window")
    if expires <= issued or expires - issued > MAX_EXECUTION_TOKEN_LIFETIME:
        _fail("Execution token lifetime must be positive and no more than 24 hours", field="expires_at_utc")
    if issued < release_issued or expires > release_expires:
        _fail("Execution token validity is outside its matrix release window", field="execution_time_window")

    key_records = trust.get("authorized_keys")
    matches = [record for record in key_records if isinstance(record, Mapping) and record.get("key_id") == manifest.get("key_id")] if isinstance(key_records, list) else []
    if len(matches) != 1 or matches[0].get("state") != "ACTIVE":
        _fail("Execution signing key is not uniquely active", field="key_id", observed=manifest.get("key_id"))
    secret = _resolve_secret(trust, matches[0], verification_secret)
    signature = manifest.get("signature")
    if not isinstance(signature, str) or len(signature) != 64:
        _fail("Execution signature is not a SHA-256 MAC", field="signature")
    try:
        bytes.fromhex(signature)
    except ValueError:
        _fail("Execution signature is not hexadecimal", field="signature")
    expected_signature = hmac.new(secret, execution_signature_payload(manifest), hashlib.sha256).hexdigest().upper()
    if not hmac.compare_digest(signature.upper(), expected_signature):
        _fail("Execution token signature verification failed", field="signature")
    return manifest


def claim_formal_execution(
    token: Mapping[str, Any] | str | Path,
    *,
    claim_registry_root: str | Path,
    release: Mapping[str, Any] | str | Path,
    release_trust_policy: Mapping[str, Any] | str | Path,
    expected_release_bindings: Mapping[str, str],
    release_manifest_sha256: str,
    expected_job_bindings: Mapping[str, Any],
    verification_secret: bytes | str | None = None,
    now_utc: datetime | None = None,
) -> Mapping[str, Any]:
    token_path: Path | None = None
    token_bytes: int | None = None
    if isinstance(token, Mapping):
        token_manifest: Mapping[str, Any] = token
        token_sha = stable_digest(token_manifest)
    else:
        token_path = Path(token).resolve()
        if not token_path.is_file() or token_path.is_symlink():
            _fail("Execution token file is missing or indirect", field="execution_token_path")
        raw_token = token_path.read_bytes()
        token_bytes = len(raw_token)
        token_sha = hashlib.sha256(raw_token).hexdigest().upper()
        try:
            parsed_token = json.loads(raw_token.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _fail("Execution token is not canonical UTF-8 JSON", field="execution_token_path")
            raise AssertionError from exc
        if not isinstance(parsed_token, Mapping):
            _fail("Execution token is not a JSON object", field="execution_token_path")
        token_manifest = parsed_token
    root = Path(claim_registry_root).resolve()
    descriptor_path = root / "CLAIM_REGISTRY.json"
    claims_root = (root / "claims").resolve()
    try:
        claims_root.relative_to(root)
    except ValueError as exc:
        _fail("Execution claim directory escapes its registry root", field="claim_registry_root")
        raise AssertionError from exc
    if not descriptor_path.is_file() or not claims_root.is_dir() or descriptor_path.is_symlink() or claims_root.is_symlink():
        _fail("Execution claim registry or claims directory is missing or indirect", field="claim_registry_root", observed=root.as_posix())
    registry_descriptor = load_json(descriptor_path)
    if not isinstance(registry_descriptor, Mapping):
        _fail("Execution claim registry descriptor is not an object", field="claim_registry")
    expected_registry_root_digest = output_root_digest(root)
    if registry_descriptor.get("registry_root_digest") != expected_registry_root_digest:
        _fail(
            "Execution claim registry descriptor was copied away from its owner-provisioned shared root",
            field="registry_root_digest",
            observed=registry_descriptor.get("registry_root_digest"),
            expected=expected_registry_root_digest,
        )
    verified = verify_formal_execution_token(
        token_manifest,
        release=release,
        release_trust_policy=release_trust_policy,
        expected_release_bindings=expected_release_bindings,
        release_manifest_sha256=release_manifest_sha256,
        claim_registry=registry_descriptor,
        expected_job_bindings=expected_job_bindings,
        verification_secret=verification_secret,
        now_utc=now_utc,
    )
    if token_path is not None and (
        not token_path.is_file()
        or token_path.is_symlink()
        or token_path.stat().st_size != token_bytes
        or sha256_file(token_path) != token_sha
    ):
        _fail("Execution token bytes changed during verification", field="execution_token_path")
    nonce_digest = hashlib.sha256(str(verified["nonce"]).encode("utf-8")).hexdigest().upper()
    claim_path = claims_root / f"{nonce_digest}.claim.json"
    job_digest = logical_job_digest(expected_job_bindings)
    heartbeat_path = _heartbeat_path(claims_root, job_digest)
    claimed_at_value = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    claimed_at = claimed_at_value.isoformat().replace("+00:00", "Z")
    claim: dict[str, Any]
    fence: dict[str, Any]
    fence_path: Path
    with _logical_job_control_lock(claims_root, job_digest):
        if claim_path.exists():
            raise SctsrError(
                ErrorCode.FORMAL_EXECUTION_TOKEN_ALREADY_CLAIMED,
                "Execution token nonce has already been claimed; the existing claim is never overwritten",
                artifact_path=claim_path.as_posix(),
                observed=nonce_digest,
                required_action="Stop this process and request a newly signed START or RESUME execution token.",
            )
        fences = _load_fence_chain(claims_root, job_digest)
        action = str(verified["action"])
        if action == "START" and fences:
            raise SctsrError(
                ErrorCode.LOGICAL_JOB_LEASE_ACTIVE,
                "A START attempt already owns this logical job and output root",
                artifact_path=fences[-1][0].as_posix(),
                observed={"logical_job_digest": job_digest, "fence_generation": fences[-1][1]["fence_generation"]},
                required_action="Do not issue another START token; use audited RESUME only after the active lease is stale or failed.",
            )
        if action == "RESUME":
            if not fences:
                _fail(
                    "RESUME cannot claim a logical job with no prior START fence",
                    field="action",
                    observed=action,
                )
            latest_heartbeat = _load_heartbeat(heartbeat_path, latest_fence=fences[-1][1])
            renewed_at = _parse_utc(latest_heartbeat["renewed_at_utc"], field="logical_job_heartbeat.renewed_at_utc")
            if renewed_at > claimed_at_value + timedelta(minutes=5):
                raise SctsrError(
                    ErrorCode.LOGICAL_JOB_FENCE_CORRUPT,
                    "Logical-job heartbeat is implausibly newer than the RESUME claim clock",
                    observed=latest_heartbeat["renewed_at_utc"],
                    expected=claimed_at,
                )
            age_seconds = (claimed_at_value - renewed_at).total_seconds()
            if latest_heartbeat["status"] == "COMPLETE":
                raise SctsrError(
                    ErrorCode.LOGICAL_JOB_LEASE_ACTIVE,
                    "Completed logical job may not be resumed",
                    artifact_path=heartbeat_path.as_posix(),
                    observed=latest_heartbeat,
                )
            if latest_heartbeat["status"] == "ACTIVE" and age_seconds <= LOGICAL_JOB_LEASE_TIMEOUT_SECONDS:
                raise SctsrError(
                    ErrorCode.LOGICAL_JOB_LEASE_ACTIVE,
                    "Logical job still has a fresh active lease",
                    artifact_path=heartbeat_path.as_posix(),
                    observed={"age_seconds": age_seconds, "fence_generation": latest_heartbeat["fence_generation"]},
                    expected={"older_than_seconds": LOGICAL_JOB_LEASE_TIMEOUT_SECONDS},
                    recoverable=True,
                    required_action="Wait for the active attempt or obtain a FAILED heartbeat before issuing RESUME.",
                )
        generation = len(fences) + 1
        previous_fence_digest = "0" * 64 if not fences else str(fences[-1][1]["fence_digest"])
        core = {
            "schema_version": FORMAL_EXECUTION_CLAIM_SCHEMA,
            "status": "CLAIMED",
            "execution_id": verified["execution_id"],
            "execution_nonce_sha256": nonce_digest,
            "execution_token_sha256": token_sha,
            "release_id": verified["release_id"],
            "release_manifest_sha256": verified["release_manifest_sha256"],
            "claim_registry_digest": verified["claim_registry_digest"],
            "job_binding_digest": verified["job_binding_digest"],
            "job_bindings": dict(expected_job_bindings),
            "logical_job_digest": job_digest,
            "fence_generation": generation,
            "previous_fence_digest": previous_fence_digest,
            "claimed_at_utc": claimed_at,
            "claimant_host": socket.gethostname(),
            "claimant_pid": os.getpid(),
        }
        claim = {**core, "claim_digest": stable_digest(core)}
        encoded = canonical_json_bytes(claim)
        fence_path = _fence_path(claims_root, job_digest, generation)
        created_claim = False
        created_fence = False
        previous_heartbeat_bytes = heartbeat_path.read_bytes() if heartbeat_path.is_file() else None
        try:
            try:
                _exclusive_write(claim_path, encoded)
            except FileExistsError as exc:
                raise SctsrError(
                    ErrorCode.FORMAL_EXECUTION_TOKEN_ALREADY_CLAIMED,
                    "Execution token nonce was concurrently claimed for another logical job",
                    artifact_path=claim_path.as_posix(),
                    observed=nonce_digest,
                    required_action="Stop this process and request a new unique execution token.",
                ) from exc
            created_claim = True
            fence_core = {
                "schema_version": LOGICAL_JOB_FENCE_SCHEMA,
                "status": "FENCE_CLAIMED",
                "logical_job_digest": job_digest,
                "fence_generation": generation,
                "previous_fence_digest": previous_fence_digest,
                "execution_id": verified["execution_id"],
                "execution_nonce_sha256": nonce_digest,
                "execution_claim_path": claim_path.as_posix(),
                "execution_claim_sha256": sha256_file(claim_path),
                "job_invariants": logical_job_invariants(expected_job_bindings),
                "claimed_at_utc": claimed_at,
            }
            fence = {**fence_core, "fence_digest": stable_digest(fence_core)}
            _exclusive_write(fence_path, canonical_json_bytes(fence))
            created_fence = True
            atomic_write_json(
                heartbeat_path,
                _heartbeat_payload(
                    status="ACTIVE",
                    job_digest=job_digest,
                    fence=fence,
                    renewed_at=claimed_at_value,
                ),
            )
            _fsync_directory(claims_root)
        except BaseException:
            if previous_heartbeat_bytes is None:
                heartbeat_path.unlink(missing_ok=True)
            else:
                atomic_write_bytes(heartbeat_path, previous_heartbeat_bytes)
            if created_fence:
                fence_path.unlink(missing_ok=True)
            if created_claim:
                claim_path.unlink(missing_ok=True)
            raise
    return {
        "status": "CLAIMED",
        "execution_id": verified["execution_id"],
        "claim_path": claim_path.as_posix(),
        "claim_bytes": claim_path.stat().st_size,
        "claim_sha256": sha256_file(claim_path),
        "claim_digest": claim["claim_digest"],
        "execution_token_sha256": token_sha,
        "job_binding_digest": verified["job_binding_digest"],
        "logical_job_digest": job_digest,
        "fence_generation": generation,
        "fence_claim_path": fence_path.as_posix(),
        "fence_claim_bytes": fence_path.stat().st_size,
        "fence_claim_sha256": sha256_file(fence_path),
        "fence_digest": fence["fence_digest"],
        "lease_heartbeat_path": heartbeat_path.as_posix(),
        "lease_timeout_seconds": LOGICAL_JOB_LEASE_TIMEOUT_SECONDS,
        "execution_token_path": None if token_path is None else token_path.as_posix(),
        "execution_token_bytes": token_bytes,
        "claim_registry_path": descriptor_path.as_posix(),
        "claim_registry_bytes": descriptor_path.stat().st_size,
        "claim_registry_sha256": sha256_file(descriptor_path),
        "claim_registry_digest": stable_digest(registry_descriptor),
    }


def validate_execution_claim_binding(
    binding: Mapping[str, Any],
    *,
    expected_job_bindings: Mapping[str, Any],
    require_token_file: bool = True,
    require_current_fence: bool = True,
) -> Mapping[str, Any]:
    if set(binding) != REQUIRED_EXECUTION_CLAIM_BINDING_FIELDS or binding.get("status") != "CLAIMED":
        _fail("Execution claim binding schema is invalid", field="execution_claim_binding")
    claim_path = Path(str(binding["claim_path"])).resolve()
    if (
        not claim_path.is_file()
        or claim_path.stat().st_size != binding["claim_bytes"]
        or sha256_file(claim_path) != binding["claim_sha256"]
    ):
        _fail("Execution claim bytes changed or disappeared", field="claim_path", observed=claim_path.as_posix())
    claim = load_json(claim_path)
    if not isinstance(claim, Mapping) or set(claim) != REQUIRED_EXECUTION_CLAIM_FIELDS:
        _fail("Execution claim schema is invalid", field="claim_path")
    core = {key: value for key, value in claim.items() if key != "claim_digest"}
    if (
        claim.get("schema_version") != FORMAL_EXECUTION_CLAIM_SCHEMA
        or claim.get("status") != "CLAIMED"
        or claim.get("claim_digest") != stable_digest(core)
        or claim.get("claim_digest") != binding["claim_digest"]
        or claim.get("execution_id") != binding["execution_id"]
        or claim.get("execution_token_sha256") != binding["execution_token_sha256"]
        or claim.get("job_binding_digest") != binding["job_binding_digest"]
        or claim.get("job_bindings") != dict(expected_job_bindings)
        or binding["job_binding_digest"] != stable_digest(dict(expected_job_bindings))
        or claim.get("logical_job_digest") != binding["logical_job_digest"]
        or claim.get("logical_job_digest") != logical_job_digest(expected_job_bindings)
        or claim.get("fence_generation") != binding["fence_generation"]
    ):
        _fail("Execution claim content differs from the requested job", field="execution_claim_binding")
    registry_path = Path(str(binding["claim_registry_path"])).resolve()
    expected_registry_path = (claim_path.parent.parent / "CLAIM_REGISTRY.json").resolve()
    if (
        registry_path != expected_registry_path
        or not registry_path.is_file()
        or registry_path.is_symlink()
        or registry_path.stat().st_size != binding["claim_registry_bytes"]
        or sha256_file(registry_path) != binding["claim_registry_sha256"]
    ):
        _fail("Execution claim registry bytes changed or disappeared", field="claim_registry_path")
    registry = load_json(registry_path)
    if (
        not isinstance(registry, Mapping)
        or stable_digest(registry) != binding["claim_registry_digest"]
        or claim.get("claim_registry_digest") != binding["claim_registry_digest"]
    ):
        _fail("Execution claim registry identity differs from the signed token", field="claim_registry_digest")
    claims_root = claim_path.parent
    fence_path = Path(str(binding["fence_claim_path"])).resolve()
    expected_fence_path = _fence_path(claims_root, str(binding["logical_job_digest"]), int(binding["fence_generation"])).resolve()
    if (
        fence_path != expected_fence_path
        or not fence_path.is_file()
        or fence_path.is_symlink()
        or fence_path.stat().st_size != binding["fence_claim_bytes"]
        or sha256_file(fence_path) != binding["fence_claim_sha256"]
    ):
        _fail("Logical-job fence bytes changed or disappeared", field="fence_claim_path")
    fence = load_json(fence_path)
    if not isinstance(fence, Mapping) or set(fence) != REQUIRED_LOGICAL_JOB_FENCE_FIELDS:
        _fail("Logical-job fence schema is invalid", field="fence_claim_path")
    fence_core = {key: value for key, value in fence.items() if key != "fence_digest"}
    if any(
        (
            fence.get("schema_version") != LOGICAL_JOB_FENCE_SCHEMA,
            fence.get("status") != "FENCE_CLAIMED",
            fence.get("logical_job_digest") != binding["logical_job_digest"],
            fence.get("fence_generation") != binding["fence_generation"],
            fence.get("fence_digest") != stable_digest(fence_core),
            fence.get("fence_digest") != binding["fence_digest"],
            fence.get("execution_id") != binding["execution_id"],
            fence.get("execution_claim_path") != claim_path.as_posix(),
            fence.get("execution_claim_sha256") != binding["claim_sha256"],
            fence.get("job_invariants") != logical_job_invariants(expected_job_bindings),
        )
    ):
        _fail("Logical-job fence does not bind the execution claim", field="fence_claim_path")
    fence_chain = _load_fence_chain(claims_root, str(binding["logical_job_digest"]))
    generation_index = int(binding["fence_generation"]) - 1
    if (
        generation_index < 0
        or generation_index >= len(fence_chain)
        or fence_chain[generation_index][0].resolve() != fence_path
    ):
        _fail("Logical-job fence is absent from its append-only chain", field="fence_claim_path")
    if require_current_fence and fence_chain[-1][0].resolve() != fence_path:
        raise SctsrError(
            ErrorCode.LOGICAL_JOB_FENCED,
            "Execution attempt has been superseded by a newer logical-job fence",
            artifact_path=fence_path.as_posix(),
            observed=binding["fence_generation"],
            expected=fence_chain[-1][1]["fence_generation"],
            required_action="Stop without publishing another epoch; only the latest RESUME attempt may continue.",
        )
    heartbeat_path = Path(str(binding["lease_heartbeat_path"])).resolve()
    expected_heartbeat_path = _heartbeat_path(claims_root, str(binding["logical_job_digest"])).resolve()
    if heartbeat_path != expected_heartbeat_path or binding["lease_timeout_seconds"] != LOGICAL_JOB_LEASE_TIMEOUT_SECONDS:
        _fail("Logical-job heartbeat binding is invalid", field="lease_heartbeat_path")
    if require_current_fence:
        _load_heartbeat(heartbeat_path, latest_fence=fence_chain[-1][1])
    token_path_value = binding.get("execution_token_path")
    if require_token_file:
        token_path = Path(str(token_path_value)).resolve()
        if (
            not isinstance(token_path_value, str)
            or not token_path.is_file()
            or token_path.stat().st_size != binding.get("execution_token_bytes")
            or sha256_file(token_path) != binding["execution_token_sha256"]
        ):
            _fail("Execution token bytes changed or disappeared after claim", field="execution_token_path")
    return claim


def _renew_execution_lease_locked(
    binding: Mapping[str, Any],
    *,
    expected_job_bindings: Mapping[str, Any],
    status: str,
    now_utc: datetime | None,
) -> Mapping[str, Any]:
    validate_execution_claim_binding(
        binding,
        expected_job_bindings=expected_job_bindings,
        require_token_file=True,
        require_current_fence=True,
    )
    fence = load_json(binding["fence_claim_path"])
    heartbeat = _heartbeat_payload(
        status=status,
        job_digest=str(binding["logical_job_digest"]),
        fence=fence,
        renewed_at=(now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc),
    )
    heartbeat_path = Path(str(binding["lease_heartbeat_path"])).resolve()
    atomic_write_json(heartbeat_path, heartbeat)
    _fsync_directory(heartbeat_path.parent)
    return heartbeat


def renew_execution_lease(
    binding: Mapping[str, Any],
    *,
    expected_job_bindings: Mapping[str, Any],
    now_utc: datetime | None = None,
) -> Mapping[str, Any]:
    claims_root = Path(str(binding["claim_path"])).resolve().parent
    with _logical_job_control_lock(claims_root, str(binding["logical_job_digest"])):
        return _renew_execution_lease_locked(
            binding,
            expected_job_bindings=expected_job_bindings,
            status="ACTIVE",
            now_utc=now_utc,
        )


@contextmanager
def execution_fence_guard(
    binding: Mapping[str, Any],
    *,
    expected_job_bindings: Mapping[str, Any],
) -> Iterator[None]:
    """Hold the job control lock across final fence check and epoch publication."""

    claims_root = Path(str(binding["claim_path"])).resolve().parent
    with _logical_job_control_lock(claims_root, str(binding["logical_job_digest"])):
        _renew_execution_lease_locked(
            binding,
            expected_job_bindings=expected_job_bindings,
            status="ACTIVE",
            now_utc=None,
        )
        yield


def publish_execution_claim_snapshot(
    run_root: str | Path,
    binding: Mapping[str, Any],
    *,
    expected_job_bindings: Mapping[str, Any],
) -> Mapping[str, Any]:
    validate_execution_claim_binding(binding, expected_job_bindings=expected_job_bindings, require_token_file=True)
    root = Path(run_root).resolve()
    attempt_root = root / "00_contract" / "execution_attempts" / str(binding["execution_id"])
    if attempt_root.exists():
        _fail("Execution attempt snapshot already exists", field="execution_id", observed=binding["execution_id"])
    attempt_root.mkdir(parents=True)
    token_source = Path(str(binding["execution_token_path"])).resolve()
    claim_source = Path(str(binding["claim_path"])).resolve()
    fence_source = Path(str(binding["fence_claim_path"])).resolve()
    registry_source = Path(str(binding["claim_registry_path"])).resolve()
    token_destination = attempt_root / "EXECUTION_TOKEN.json"
    claim_destination = attempt_root / "EXECUTION_CLAIM.json"
    fence_destination = attempt_root / "LOGICAL_JOB_FENCE.json"
    registry_destination = attempt_root / "EXECUTION_CLAIM_REGISTRY.json"
    binding_destination = attempt_root / "EXECUTION_CLAIM_BINDING.json"
    atomic_write_bytes(token_destination, token_source.read_bytes())
    atomic_write_bytes(claim_destination, claim_source.read_bytes())
    atomic_write_bytes(fence_destination, fence_source.read_bytes())
    atomic_write_bytes(registry_destination, registry_source.read_bytes())
    atomic_write_json(binding_destination, dict(binding))
    copied = {
        "execution_token": (token_destination, binding["execution_token_bytes"], binding["execution_token_sha256"]),
        "execution_claim": (claim_destination, binding["claim_bytes"], binding["claim_sha256"]),
        "logical_job_fence": (
            fence_destination,
            binding["fence_claim_bytes"],
            binding["fence_claim_sha256"],
        ),
        "execution_claim_registry": (
            registry_destination,
            binding["claim_registry_bytes"],
            binding["claim_registry_sha256"],
        ),
    }
    for role, (path, expected_bytes, expected_sha) in copied.items():
        if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha:
            _fail("Execution attempt snapshot differs from claimed bytes", field=role)
    core = {
        "schema_version": EXECUTION_ATTEMPT_SNAPSHOT_SCHEMA,
        "execution_id": binding["execution_id"],
        "job_binding_digest": binding["job_binding_digest"],
        "execution_token": {
            "path": token_destination.relative_to(root).as_posix(),
            "bytes": token_destination.stat().st_size,
            "sha256": sha256_file(token_destination),
        },
        "execution_claim": {
            "path": claim_destination.relative_to(root).as_posix(),
            "bytes": claim_destination.stat().st_size,
            "sha256": sha256_file(claim_destination),
        },
        "logical_job_fence": {
            "path": fence_destination.relative_to(root).as_posix(),
            "bytes": fence_destination.stat().st_size,
            "sha256": sha256_file(fence_destination),
        },
        "execution_claim_registry": {
            "path": registry_destination.relative_to(root).as_posix(),
            "bytes": registry_destination.stat().st_size,
            "sha256": sha256_file(registry_destination),
            "digest": binding["claim_registry_digest"],
        },
        "claim_binding": {
            "path": binding_destination.relative_to(root).as_posix(),
            "bytes": binding_destination.stat().st_size,
            "sha256": sha256_file(binding_destination),
        },
    }
    snapshot = {**core, "snapshot_digest": stable_digest(core)}
    atomic_write_json(attempt_root / "EXECUTION_ATTEMPT_SNAPSHOT.json", snapshot)
    return snapshot


def validate_execution_attempt_snapshot(
    run_root: str | Path,
    *,
    execution_id: str,
    expected_snapshot_digest: str,
    expected_job_binding_digest: str,
    expected_claim_sha256: str,
) -> Mapping[str, Any]:
    """Validate the immutable in-run copy without trusting external paths."""

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{7,127}", execution_id) is None:
        _fail("Execution attempt ID is noncanonical", field="execution_id", observed=execution_id)
    _sha256(expected_snapshot_digest, field="expected_snapshot_digest", allow_zero=False)
    _sha256(expected_job_binding_digest, field="expected_job_binding_digest", allow_zero=False)
    _sha256(expected_claim_sha256, field="expected_claim_sha256", allow_zero=False)
    root = Path(run_root).resolve()
    attempt_root = root / "00_contract" / "execution_attempts" / execution_id
    snapshot_path = attempt_root / "EXECUTION_ATTEMPT_SNAPSHOT.json"
    if not snapshot_path.is_file():
        _fail("Execution attempt snapshot is missing", field="execution_attempt_snapshot")
    snapshot = load_json(snapshot_path)
    required_snapshot_fields = {
        "schema_version",
        "execution_id",
        "job_binding_digest",
        "execution_token",
        "execution_claim",
        "execution_claim_registry",
        "logical_job_fence",
        "claim_binding",
        "snapshot_digest",
    }
    if not isinstance(snapshot, Mapping) or set(snapshot) != required_snapshot_fields:
        _fail("Execution attempt snapshot schema is invalid", field="execution_attempt_snapshot")
    core = {key: value for key, value in snapshot.items() if key != "snapshot_digest"}
    if (
        snapshot.get("schema_version") != EXECUTION_ATTEMPT_SNAPSHOT_SCHEMA
        or snapshot.get("execution_id") != execution_id
        or snapshot.get("snapshot_digest") != stable_digest(core)
        or snapshot.get("snapshot_digest") != expected_snapshot_digest
        or snapshot.get("job_binding_digest") != expected_job_binding_digest
    ):
        _fail("Execution attempt snapshot identity is invalid", field="execution_attempt_snapshot")

    expected_paths = {
        "execution_token": attempt_root / "EXECUTION_TOKEN.json",
        "execution_claim": attempt_root / "EXECUTION_CLAIM.json",
        "logical_job_fence": attempt_root / "LOGICAL_JOB_FENCE.json",
        "execution_claim_registry": attempt_root / "EXECUTION_CLAIM_REGISTRY.json",
        "claim_binding": attempt_root / "EXECUTION_CLAIM_BINDING.json",
    }
    resolved: dict[str, Path] = {}
    for role, expected_path in expected_paths.items():
        record = snapshot.get(role)
        expected_record_fields = {"path", "bytes", "sha256", "digest"} if role == "execution_claim_registry" else {"path", "bytes", "sha256"}
        if not isinstance(record, Mapping) or set(record) != expected_record_fields:
            _fail("Execution attempt file record schema is invalid", field=role)
        path = (root / str(record["path"])).resolve()
        if (
            path != expected_path.resolve()
            or not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            _fail("Execution attempt file bytes are invalid", field=role)
        resolved[role] = path

    binding = load_json(resolved["claim_binding"])
    token = load_json(resolved["execution_token"])
    claim = load_json(resolved["execution_claim"])
    fence = load_json(resolved["logical_job_fence"])
    registry = load_json(resolved["execution_claim_registry"])
    if (
        not isinstance(binding, Mapping)
        or set(binding) != REQUIRED_EXECUTION_CLAIM_BINDING_FIELDS
        or binding.get("status") != "CLAIMED"
        or binding.get("execution_id") != execution_id
        or binding.get("job_binding_digest") != expected_job_binding_digest
        or binding.get("claim_sha256") != expected_claim_sha256
        or binding.get("claim_sha256") != snapshot["execution_claim"]["sha256"]
        or binding.get("fence_claim_sha256") != snapshot["logical_job_fence"]["sha256"]
        or binding.get("execution_token_sha256") != snapshot["execution_token"]["sha256"]
        or binding.get("claim_registry_sha256") != snapshot["execution_claim_registry"]["sha256"]
    ):
        _fail("Execution claim binding copy is invalid", field="claim_binding")
    if not isinstance(token, Mapping) or set(token) != REQUIRED_EXECUTION_TOKEN_FIELDS:
        _fail("Execution token copy schema is invalid", field="execution_token")
    if not isinstance(claim, Mapping) or set(claim) != REQUIRED_EXECUTION_CLAIM_FIELDS:
        _fail("Execution claim copy schema is invalid", field="execution_claim")
    if not isinstance(fence, Mapping) or set(fence) != REQUIRED_LOGICAL_JOB_FENCE_FIELDS:
        _fail("Logical-job fence copy schema is invalid", field="logical_job_fence")
    if not isinstance(registry, Mapping):
        _fail("Execution claim registry copy schema is invalid", field="execution_claim_registry")
    _validate_claim_registry(registry)
    claim_core = {key: value for key, value in claim.items() if key != "claim_digest"}
    registry_digest = stable_digest(registry)
    if (
        token.get("schema_version") != EXECUTION_TOKEN_SCHEMA
        or token.get("authorization") != "SIGNED_SCTSR_V4_FORMAL_EXECUTION"
        or token.get("formal_execution_authorized") is not True
        or token.get("execution_id") != execution_id
        or token.get("job_binding_digest") != expected_job_binding_digest
        or token.get("claim_registry_digest") != registry_digest
        or claim.get("schema_version") != FORMAL_EXECUTION_CLAIM_SCHEMA
        or claim.get("status") != "CLAIMED"
        or claim.get("execution_id") != execution_id
        or claim.get("claim_digest") != stable_digest(claim_core)
        or claim.get("claim_digest") != binding.get("claim_digest")
        or claim.get("execution_token_sha256") != binding.get("execution_token_sha256")
        or claim.get("job_binding_digest") != expected_job_binding_digest
        or claim.get("claim_registry_digest") != registry_digest
        or binding.get("claim_registry_digest") != registry_digest
        or snapshot["execution_claim_registry"].get("digest") != registry_digest
        or token.get("release_manifest_sha256") != claim.get("release_manifest_sha256")
        or fence.get("schema_version") != LOGICAL_JOB_FENCE_SCHEMA
        or fence.get("logical_job_digest") != binding.get("logical_job_digest")
        or fence.get("fence_generation") != binding.get("fence_generation")
        or fence.get("fence_digest") != binding.get("fence_digest")
        or fence.get("execution_claim_sha256") != expected_claim_sha256
    ):
        _fail("Execution token, claim and registry copies do not cross-bind", field="execution_attempt_snapshot")
    return {
        "status": "PASS",
        "execution_id": execution_id,
        "snapshot_digest": snapshot["snapshot_digest"],
        "job_binding_digest": expected_job_binding_digest,
        "claim_sha256": expected_claim_sha256,
    }
