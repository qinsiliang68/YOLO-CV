from __future__ import annotations

import hashlib
import hmac
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from stage1_sctsr_v4.formal_release import release_signature_payload, verify_formal_release


def main() -> int:
    secret = b"review-only-owner-secret-at-least-32-bytes"
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
        "release_id": "REVIEW_DUPLICATE_NONCE",
        "key_id": "review-key",
        "signature_algorithm": "HMAC-SHA256",
        "signature": "",
        "issued_at_utc": "2026-08-14T00:00:00Z",
        "expires_at_utc": "2026-08-15T00:00:00Z",
        "nonce": "0123456789ABCDEF0123456789ABCDEF",
        **bindings,
    }
    release["signature"] = hmac.new(
        secret,
        release_signature_payload(release),
        hashlib.sha256,
    ).hexdigest().upper()
    trust = {
        "schema_version": "stage1.sctsr.release_trust.v1",
        "required_algorithm": "HMAC-SHA256",
        "secret_environment_variable": "UNUSED_REVIEW_KEY",
        "authorized_keys": [
            {
                "key_id": "review-key",
                "state": "ACTIVE",
                "secret_sha256": hashlib.sha256(secret).hexdigest().upper(),
                "not_before_utc": "2026-08-01T00:00:00Z",
                "not_after_utc": "2026-09-01T00:00:00Z",
            }
        ],
    }
    now = datetime(2026, 8, 14, 12, tzinfo=timezone.utc)
    first = verify_formal_release(
        release,
        trust_policy=trust,
        expected_bindings=bindings,
        verification_secret=secret,
        now_utc=now,
    )
    second_rejected = False
    second_error = None
    try:
        verify_formal_release(
            release,
            trust_policy=trust,
            expected_bindings=bindings,
            verification_secret=secret,
            now_utc=now,
        )
    except Exception as exc:  # pragma: no cover - review observation path
        second_rejected = True
        second_error = f"{type(exc).__name__}: {exc}"
    result = {
        "schema_version": "stage1.sctsr.review.reproduction.v1",
        "check": "duplicate_release_nonce_rejected",
        "expected": "second verification rejects already-seen release_id/nonce",
        "observed": "second verification rejected" if second_rejected else "second verification accepted",
        "first_release_id": first["release_id"],
        "nonce": release["nonce"],
        "second_error": second_error,
        "status": "PASS" if second_rejected else "FAIL",
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if second_rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())
