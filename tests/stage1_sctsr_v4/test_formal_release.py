from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timezone

import pytest

from stage1_sctsr_v4.contracts import require_synthetic_or_authorized
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.formal_release import release_signature_payload


def _signed_release(secret: bytes):
    bindings = {
        "baseline_main_commit": "A" * 40,
        "taskbook_blob_sha": "B" * 40,
        "source_tree_digest": "C" * 64,
        "contract_digest": "D" * 64,
        "asset_registry_digest": "E" * 64,
        "runtime_config_digest": "F" * 64,
        "seed_registry_digest": "1" * 64,
    }
    release = {
        "schema_version": "stage1.sctsr.formal_release.v1",
        "authorization": "SIGNED_SCTSR_V4_FORMAL_RELEASE",
        "formal_training_authorized": True,
        "release_id": "OWNER_RELEASE_20260813_001",
        "key_id": "test-owner-key",
        "signature_algorithm": "HMAC-SHA256",
        "signature": "",
        "issued_at_utc": "2026-08-13T00:00:00Z",
        "expires_at_utc": "2026-08-14T00:00:00Z",
        "nonce": "0123456789ABCDEF0123456789ABCDEF",
        **bindings,
    }
    release["signature"] = hmac.new(secret, release_signature_payload(release), hashlib.sha256).hexdigest().upper()
    trust = {
        "schema_version": "stage1.sctsr.release_trust.v1",
        "required_algorithm": "HMAC-SHA256",
        "secret_environment_variable": "UNUSED_IN_TEST",
        "authorized_keys": [
            {
                "key_id": "test-owner-key",
                "state": "ACTIVE",
                "secret_sha256": hashlib.sha256(secret).hexdigest().upper(),
                "not_before_utc": "2026-08-01T00:00:00Z",
                "not_after_utc": "2026-09-01T00:00:00Z",
            }
        ],
    }
    return release, trust, bindings


def test_sa_034_valid_owner_mac_requires_complete_frozen_bindings():
    secret = b"owner-controlled-test-secret-32-bytes-minimum"
    release, trust, bindings = _signed_release(secret)

    require_synthetic_or_authorized(
        "formal",
        release,
        trust_policy=trust,
        expected_bindings=bindings,
        verification_secret=secret,
        now_utc=datetime(2026, 8, 13, 12, tzinfo=timezone.utc),
    )


def test_sa_035_expired_release_is_rejected_even_with_valid_mac():
    secret = b"owner-controlled-test-secret-32-bytes-minimum"
    release, trust, bindings = _signed_release(secret)

    with pytest.raises(SctsrError) as exc:
        require_synthetic_or_authorized(
            "formal",
            release,
            trust_policy=trust,
            expected_bindings=bindings,
            verification_secret=secret,
            now_utc=datetime(2026, 8, 14, 1, tzinfo=timezone.utc),
        )

    assert exc.value.code is ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED


def test_sa_036_release_binding_mismatch_is_rejected():
    secret = b"owner-controlled-test-secret-32-bytes-minimum"
    release, trust, bindings = _signed_release(secret)
    bindings["source_tree_digest"] = "9" * 64

    with pytest.raises(SctsrError) as exc:
        require_synthetic_or_authorized(
            "formal",
            release,
            trust_policy=trust,
            expected_bindings=bindings,
            verification_secret=secret,
            now_utc=datetime(2026, 8, 13, 12, tzinfo=timezone.utc),
        )

    assert exc.value.code is ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED
