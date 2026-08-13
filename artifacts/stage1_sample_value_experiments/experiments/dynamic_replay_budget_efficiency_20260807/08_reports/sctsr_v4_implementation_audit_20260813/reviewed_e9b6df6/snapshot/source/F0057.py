from __future__ import annotations

import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from .errors import ErrorCode, SctsrError
from .serialization import canonical_json_bytes, load_json


RELEASE_SCHEMA = "stage1.sctsr.formal_release.v1"
TRUST_SCHEMA = "stage1.sctsr.release_trust.v1"
SIGNATURE_ALGORITHM = "HMAC-SHA256"
DEFAULT_SECRET_ENVIRONMENT_VARIABLE = "SCTSR_V4_RELEASE_HMAC_KEY"
MAX_RELEASE_LIFETIME = timedelta(days=7)
MAX_CLOCK_SKEW = timedelta(minutes=5)

REQUIRED_BINDING_FIELDS = (
    "baseline_main_commit",
    "taskbook_blob_sha",
    "source_tree_digest",
    "contract_digest",
    "asset_registry_digest",
    "runtime_config_digest",
    "seed_registry_digest",
)

REQUIRED_RELEASE_FIELDS = frozenset(
    {
        "schema_version",
        "authorization",
        "formal_training_authorized",
        "release_id",
        "key_id",
        "signature_algorithm",
        "signature",
        "issued_at_utc",
        "expires_at_utc",
        "nonce",
        *REQUIRED_BINDING_FIELDS,
    }
)


def _fail(message: str, *, field: str | None = None, observed: Any = None, expected: Any = None) -> None:
    raise SctsrError(
        ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED,
        message,
        failing_field=field,
        observed=observed,
        expected=expected,
        required_action="Provide a future owner-issued release signed by an active out-of-repository key.",
    )


def _load_mapping(value: Mapping[str, Any] | str | Path, *, role: str) -> Mapping[str, Any]:
    loaded: Any = value if isinstance(value, Mapping) else load_json(value)
    if not isinstance(loaded, Mapping):
        _fail(f"{role} must be a JSON object")
    return loaded


def _parse_utc(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail("Release timestamps must use canonical UTC Z notation", field=field, observed=value)
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        _fail("Release timestamp is invalid", field=field, observed=value)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        _fail("Release timestamp is not UTC", field=field, observed=value)
    return parsed.astimezone(timezone.utc)


def release_signature_payload(release: Mapping[str, Any]) -> bytes:
    """Return the exact canonical bytes authenticated by the release authority."""
    return canonical_json_bytes({key: value for key, value in release.items() if key != "signature"})


def _resolve_secret(
    trust: Mapping[str, Any],
    key_record: Mapping[str, Any],
    verification_secret: bytes | str | None,
) -> bytes:
    if verification_secret is None:
        env_name = str(trust.get("secret_environment_variable", DEFAULT_SECRET_ENVIRONMENT_VARIABLE))
        raw = os.environ.get(env_name)
        if raw is None:
            _fail("Release verification secret is unavailable", field="secret_environment_variable", observed=env_name)
        secret = raw.encode("utf-8")
    elif isinstance(verification_secret, str):
        secret = verification_secret.encode("utf-8")
    else:
        secret = bytes(verification_secret)
    if len(secret) < 32:
        _fail("Release verification secret must contain at least 32 bytes", field="verification_secret")
    observed_digest = hashlib.sha256(secret).hexdigest().upper()
    expected_digest = str(key_record.get("secret_sha256", "")).upper()
    if not hmac.compare_digest(observed_digest, expected_digest):
        _fail("Release verification secret is not trusted", field="secret_sha256", observed=observed_digest, expected=expected_digest)
    return secret


def verify_formal_release(
    release: Mapping[str, Any] | str | Path,
    *,
    trust_policy: Mapping[str, Any] | str | Path,
    expected_bindings: Mapping[str, str],
    verification_secret: bytes | str | None = None,
    now_utc: datetime | None = None,
) -> Mapping[str, Any]:
    manifest = _load_mapping(release, role="Release manifest")
    trust = _load_mapping(trust_policy, role="Release trust policy")

    missing = sorted(REQUIRED_RELEASE_FIELDS - set(manifest))
    if missing:
        _fail("Release manifest is incomplete", observed=missing, expected=sorted(REQUIRED_RELEASE_FIELDS))
    if manifest.get("schema_version") != RELEASE_SCHEMA:
        _fail("Release schema is not supported", field="schema_version", observed=manifest.get("schema_version"), expected=RELEASE_SCHEMA)
    if manifest.get("authorization") != "SIGNED_SCTSR_V4_FORMAL_RELEASE" or manifest.get("formal_training_authorized") is not True:
        _fail("Release does not authorize SCTSR v4 formal training")
    if trust.get("schema_version") != TRUST_SCHEMA:
        _fail("Release trust policy schema is not supported", field="trust.schema_version", observed=trust.get("schema_version"), expected=TRUST_SCHEMA)
    if trust.get("required_algorithm") != SIGNATURE_ALGORITHM or manifest.get("signature_algorithm") != SIGNATURE_ALGORITHM:
        _fail("Release signature algorithm is not authorized", field="signature_algorithm", observed=manifest.get("signature_algorithm"), expected=SIGNATURE_ALGORITHM)

    release_id = manifest.get("release_id")
    nonce = manifest.get("nonce")
    key_id = manifest.get("key_id")
    if not isinstance(release_id, str) or not release_id.strip():
        _fail("Release ID is missing", field="release_id")
    if not isinstance(nonce, str) or len(nonce) < 32:
        _fail("Release nonce must contain at least 32 characters", field="nonce")
    if not isinstance(key_id, str) or not key_id.strip():
        _fail("Release key ID is missing", field="key_id")

    expected_keys = set(REQUIRED_BINDING_FIELDS)
    if set(expected_bindings) != expected_keys:
        _fail("Caller did not provide the complete formal identity binding", field="expected_bindings", observed=sorted(expected_bindings), expected=sorted(expected_keys))
    mismatches = {
        field: {"observed": manifest.get(field), "expected": str(expected_bindings[field]).upper()}
        for field in REQUIRED_BINDING_FIELDS
        if str(manifest.get(field, "")).upper() != str(expected_bindings[field]).upper()
    }
    if mismatches:
        _fail("Release identity binding does not match the prepared run", field="release_bindings", observed=mismatches)

    issued = _parse_utc(manifest.get("issued_at_utc"), field="issued_at_utc")
    expires = _parse_utc(manifest.get("expires_at_utc"), field="expires_at_utc")
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if issued > now + MAX_CLOCK_SKEW:
        _fail("Release is not yet valid", field="issued_at_utc", observed=manifest.get("issued_at_utc"))
    if expires <= now:
        _fail("Release has expired", field="expires_at_utc", observed=manifest.get("expires_at_utc"))
    if expires <= issued or expires - issued > MAX_RELEASE_LIFETIME:
        _fail("Release lifetime is invalid", field="expires_at_utc", observed=manifest.get("expires_at_utc"), expected="after issued_at_utc and no more than seven days")

    records = trust.get("authorized_keys")
    if not isinstance(records, list):
        _fail("Trust policy authorized_keys must be a list", field="authorized_keys")
    matches = [record for record in records if isinstance(record, Mapping) and record.get("key_id") == key_id]
    if len(matches) != 1:
        _fail("Release key is not uniquely trusted", field="key_id", observed=key_id)
    key_record = matches[0]
    if key_record.get("state") != "ACTIVE":
        _fail("Release key is not active", field="key_id", observed=key_id)
    key_not_before = _parse_utc(key_record.get("not_before_utc"), field="key.not_before_utc")
    key_not_after = _parse_utc(key_record.get("not_after_utc"), field="key.not_after_utc")
    if issued < key_not_before or expires > key_not_after:
        _fail("Release validity is outside the trusted key window", field="key_id", observed=key_id)

    signature = manifest.get("signature")
    if not isinstance(signature, str) or len(signature) != 64:
        _fail("Release signature is not a SHA-256 MAC", field="signature")
    try:
        bytes.fromhex(signature)
    except ValueError:
        _fail("Release signature is not hexadecimal", field="signature")
    secret = _resolve_secret(trust, key_record, verification_secret)
    expected_signature = hmac.new(secret, release_signature_payload(manifest), hashlib.sha256).hexdigest().upper()
    if not hmac.compare_digest(signature.upper(), expected_signature):
        _fail("Release signature verification failed", field="signature")
    return manifest
