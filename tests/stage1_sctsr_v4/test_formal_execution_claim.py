from __future__ import annotations

import hashlib
import hmac
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.formal_execution import (
    CLAIM_REGISTRY_SCHEMA,
    EXECUTION_TOKEN_SCHEMA,
    JOB_BINDING_FIELDS,
    claim_formal_execution,
    execution_signature_payload,
    output_root_digest,
    publish_execution_claim_snapshot,
    validate_execution_claim_binding,
    validate_execution_attempt_snapshot,
    verify_formal_execution_token,
)
from stage1_sctsr_v4.formal_release import release_signature_payload
from stage1_sctsr_v4.serialization import atomic_write_json, stable_digest


SECRET = b"owner-controlled-test-secret-32-bytes-minimum"
NOW = datetime(2026, 8, 14, 12, tzinfo=timezone.utc)


def _release():
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
        "release_id": "OWNER_RELEASE_20260814_001",
        "key_id": "test-owner-key",
        "signature_algorithm": "HMAC-SHA256",
        "signature": "",
        "issued_at_utc": "2026-08-14T00:00:00Z",
        "expires_at_utc": "2026-08-15T00:00:00Z",
        "nonce": "RELEASE_NONCE_0123456789ABCDEF0123456789ABCDEF",
        **bindings,
    }
    release["signature"] = hmac.new(SECRET, release_signature_payload(release), hashlib.sha256).hexdigest().upper()
    trust = {
        "schema_version": "stage1.sctsr.release_trust.v1",
        "required_algorithm": "HMAC-SHA256",
        "secret_environment_variable": "UNUSED_IN_TEST",
        "authorized_keys": [
            {
                "key_id": "test-owner-key",
                "state": "ACTIVE",
                "secret_sha256": hashlib.sha256(SECRET).hexdigest().upper(),
                "not_before_utc": "2026-08-01T00:00:00Z",
                "not_after_utc": "2026-09-01T00:00:00Z",
            }
        ],
    }
    return release, trust, bindings


def _claim_registry(tmp_path):
    root = tmp_path / "shared_claim_registry"
    root.mkdir()
    descriptor = {
        "schema_version": CLAIM_REGISTRY_SCHEMA,
        "registry_id": "SCTSR_SHARED_REGISTRY_TEST_001",
        "mode": "SHARED_MULTI_MACHINE_EXCLUSIVE_CREATE",
        "state": "ACTIVE",
        "experiment_id": "dynamic_replay_budget_efficiency_20260807",
    }
    atomic_write_json(root / "CLAIM_REGISTRY.json", descriptor)
    (root / "claims").mkdir()
    return root, descriptor


def _job(tmp_path, *, action="START", logical_run_id="PARENT_101", training_seed=101):
    values = {
        "action": action,
        "run_role": "COMMON_PARENT",
        "logical_run_id": logical_run_id,
        "arm_id": "COMMON_PARENT_NR",
        "training_seed": training_seed,
        "output_root_digest": output_root_digest(tmp_path / logical_run_id),
        "parent_checkpoint_sha256": "2" * 64,
        "resume_checkpoint_sha256": "0" * 64 if action == "START" else "3" * 64,
        "lineage_digest": stable_digest({"role": "NOT_APPLICABLE_COMMON_PARENT"}),
        "schedule_digest": stable_digest({"role": "COMMON_PARENT_NR", "epochs": [1, 120]}),
        "resume_from_receipt_digest": "0" * 64 if action == "START" else "4" * 64,
    }
    assert set(values) == set(JOB_BINDING_FIELDS)
    return values


def _token(release, registry, job, *, execution_id="EXEC_PARENT_101_START", nonce=None):
    token = {
        "schema_version": EXECUTION_TOKEN_SCHEMA,
        "authorization": "SIGNED_SCTSR_V4_FORMAL_EXECUTION",
        "formal_execution_authorized": True,
        "execution_id": execution_id,
        "key_id": release["key_id"],
        "signature_algorithm": "HMAC-SHA256",
        "signature": "",
        "issued_at_utc": "2026-08-14T11:00:00Z",
        "expires_at_utc": "2026-08-14T13:00:00Z",
        "nonce": nonce or f"EXECUTION_NONCE_{execution_id}_0123456789ABCDEF0123456789ABCDEF",
        "release_id": release["release_id"],
        "release_nonce": release["nonce"],
        "release_manifest_sha256": stable_digest(release),
        "claim_registry_digest": stable_digest(registry),
        "job_binding_digest": stable_digest(job),
        **job,
    }
    token["signature"] = hmac.new(SECRET, execution_signature_payload(token), hashlib.sha256).hexdigest().upper()
    return token


def _verify(token, release, trust, bindings, registry, job):
    return verify_formal_execution_token(
        token,
        release=release,
        release_trust_policy=trust,
        expected_release_bindings=bindings,
        release_manifest_sha256=stable_digest(release),
        claim_registry=registry,
        expected_job_bindings=job,
        verification_secret=SECRET,
        now_utc=NOW,
    )


def test_execution_token_binds_release_registry_and_exact_job(tmp_path):
    release, trust, bindings = _release()
    _root, registry = _claim_registry(tmp_path)
    job = _job(tmp_path)
    token = _token(release, registry, job)

    verified = _verify(token, release, trust, bindings, registry, job)

    assert verified["execution_id"] == "EXEC_PARENT_101_START"
    assert verified["job_binding_digest"] == stable_digest(job)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("arm_id", "T_U"),
        ("training_seed", 102),
        ("output_root_digest", "9" * 64),
        ("action", "RESUME"),
        ("resume_from_receipt_digest", "4" * 64),
    ],
)
def test_execution_token_rejects_job_substitution(tmp_path, field, replacement):
    release, trust, bindings = _release()
    _root, registry = _claim_registry(tmp_path)
    signed_job = _job(tmp_path)
    token = _token(release, registry, signed_job)
    expected_job = {**signed_job, field: replacement}

    with pytest.raises(SctsrError) as caught:
        _verify(token, release, trust, bindings, registry, expected_job)

    assert caught.value.code is ErrorCode.FORMAL_EXECUTION_TOKEN_INVALID


def test_same_execution_token_can_be_claimed_only_once(tmp_path):
    release, trust, bindings = _release()
    claim_root, registry = _claim_registry(tmp_path)
    job = _job(tmp_path)
    token = _token(release, registry, job)
    kwargs = {
        "release": release,
        "release_trust_policy": trust,
        "expected_release_bindings": bindings,
        "release_manifest_sha256": stable_digest(release),
        "expected_job_bindings": job,
        "verification_secret": SECRET,
        "now_utc": NOW,
    }

    first = claim_formal_execution(token, claim_registry_root=claim_root, **kwargs)
    with pytest.raises(SctsrError) as caught:
        claim_formal_execution(token, claim_registry_root=claim_root, **kwargs)

    assert first["status"] == "CLAIMED"
    assert caught.value.code is ErrorCode.FORMAL_EXECUTION_TOKEN_ALREADY_CLAIMED
    assert len(list((claim_root / "claims").glob("*.claim.json"))) == 1


def test_concurrent_claimers_have_exactly_one_winner(tmp_path):
    release, trust, bindings = _release()
    claim_root, registry = _claim_registry(tmp_path)
    job = _job(tmp_path)
    token = _token(release, registry, job)
    barrier = threading.Barrier(8)

    def attempt(_index):
        barrier.wait()
        try:
            claim_formal_execution(
                token,
                claim_registry_root=claim_root,
                release=release,
                release_trust_policy=trust,
                expected_release_bindings=bindings,
                release_manifest_sha256=stable_digest(release),
                expected_job_bindings=job,
                verification_secret=SECRET,
                now_utc=NOW,
            )
        except SctsrError as error:
            return error.code
        return "CLAIMED"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, range(8)))

    assert results.count("CLAIMED") == 1
    assert results.count(ErrorCode.FORMAL_EXECUTION_TOKEN_ALREADY_CLAIMED) == 7
    assert len(list((claim_root / "claims").glob("*.claim.json"))) == 1


def test_one_matrix_release_may_authorize_distinct_job_tokens(tmp_path):
    release, trust, bindings = _release()
    claim_root, registry = _claim_registry(tmp_path)
    first_job = _job(tmp_path, logical_run_id="PARENT_101", training_seed=101)
    second_job = _job(tmp_path, logical_run_id="PARENT_102", training_seed=102)
    first_token = _token(release, registry, first_job, execution_id="EXEC_PARENT_101_START")
    second_token = _token(release, registry, second_job, execution_id="EXEC_PARENT_102_START")

    first = claim_formal_execution(
        first_token,
        claim_registry_root=claim_root,
        release=release,
        release_trust_policy=trust,
        expected_release_bindings=bindings,
        release_manifest_sha256=stable_digest(release),
        expected_job_bindings=first_job,
        verification_secret=SECRET,
        now_utc=NOW,
    )
    second = claim_formal_execution(
        second_token,
        claim_registry_root=claim_root,
        release=release,
        release_trust_policy=trust,
        expected_release_bindings=bindings,
        release_manifest_sha256=stable_digest(release),
        expected_job_bindings=second_job,
        verification_secret=SECRET,
        now_utc=NOW,
    )

    assert first["execution_id"] != second["execution_id"]
    assert len(list((claim_root / "claims").glob("*.claim.json"))) == 2


def test_corrupt_existing_claim_is_never_overwritten(tmp_path):
    release, trust, bindings = _release()
    claim_root, registry = _claim_registry(tmp_path)
    job = _job(tmp_path)
    token = _token(release, registry, job)
    nonce_digest = hashlib.sha256(token["nonce"].encode("utf-8")).hexdigest().upper()
    claim_path = claim_root / "claims" / f"{nonce_digest}.claim.json"
    claim_path.write_bytes(b'{"partial":')

    with pytest.raises(SctsrError) as caught:
        claim_formal_execution(
            token,
            claim_registry_root=claim_root,
            release=release,
            release_trust_policy=trust,
            expected_release_bindings=bindings,
            release_manifest_sha256=stable_digest(release),
            expected_job_bindings=job,
            verification_secret=SECRET,
            now_utc=NOW,
        )

    assert caught.value.code is ErrorCode.FORMAL_EXECUTION_TOKEN_ALREADY_CLAIMED
    assert claim_path.read_bytes() == b'{"partial":'


def test_claim_and_token_are_snapshotted_and_tamper_is_rejected(tmp_path):
    release, trust, bindings = _release()
    claim_root, registry = _claim_registry(tmp_path)
    job = _job(tmp_path)
    token = _token(release, registry, job)
    token_path = tmp_path / "EXECUTION_TOKEN.json"
    atomic_write_json(token_path, token)
    claim = claim_formal_execution(
        token_path,
        claim_registry_root=claim_root,
        release=release,
        release_trust_policy=trust,
        expected_release_bindings=bindings,
        release_manifest_sha256=stable_digest(release),
        expected_job_bindings=job,
        verification_secret=SECRET,
        now_utc=NOW,
    )
    run_root = tmp_path / "run"

    validated = validate_execution_claim_binding(claim, expected_job_bindings=job)
    snapshot = publish_execution_claim_snapshot(run_root, claim, expected_job_bindings=job)
    closeout = validate_execution_attempt_snapshot(
        run_root,
        execution_id=token["execution_id"],
        expected_snapshot_digest=snapshot["snapshot_digest"],
        expected_job_binding_digest=stable_digest(job),
        expected_claim_sha256=claim["claim_sha256"],
    )

    assert validated["status"] == "CLAIMED"
    assert snapshot["execution_id"] == token["execution_id"]
    assert closeout["status"] == "PASS"
    attempt_root = run_root / "00_contract" / "execution_attempts" / token["execution_id"]
    assert (attempt_root / "EXECUTION_TOKEN.json").is_file()
    assert (attempt_root / "EXECUTION_CLAIM.json").is_file()
    assert (attempt_root / "EXECUTION_CLAIM_REGISTRY.json").is_file()

    internal_token = attempt_root / "EXECUTION_TOKEN.json"
    original_internal_token_bytes = internal_token.read_bytes()
    internal_token.write_bytes(b'{}')
    with pytest.raises(SctsrError) as caught:
        validate_execution_attempt_snapshot(
            run_root,
            execution_id=token["execution_id"],
            expected_snapshot_digest=snapshot["snapshot_digest"],
            expected_job_binding_digest=stable_digest(job),
            expected_claim_sha256=claim["claim_sha256"],
        )
    assert caught.value.code is ErrorCode.FORMAL_EXECUTION_TOKEN_INVALID
    internal_token.write_bytes(original_internal_token_bytes)

    registry_path = claim_root / "CLAIM_REGISTRY.json"
    original_registry_bytes = registry_path.read_bytes()
    registry_path.write_bytes(b'{}')
    with pytest.raises(SctsrError) as caught:
        validate_execution_claim_binding(claim, expected_job_bindings=job)
    assert caught.value.code is ErrorCode.FORMAL_EXECUTION_TOKEN_INVALID
    registry_path.write_bytes(original_registry_bytes)

    token_path.write_bytes(b'{}')
    with pytest.raises(SctsrError) as caught:
        validate_execution_claim_binding(claim, expected_job_bindings=job)
    assert caught.value.code is ErrorCode.FORMAL_EXECUTION_TOKEN_INVALID


def test_repository_execution_templates_are_inactive_and_complete(repository_root):
    token = json.loads(
        (repository_root / "configs" / "stage1_sctsr_v4" / "formal_execution_token_schema_v1.json").read_text(
            encoding="utf-8"
        )
    )
    registry = json.loads(
        (repository_root / "configs" / "stage1_sctsr_v4" / "execution_claim_registry_template_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert set(token) == {
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
    assert token["formal_execution_authorized"] is False
    assert registry == {
        "schema_version": CLAIM_REGISTRY_SCHEMA,
        "registry_id": "FUTURE_OWNER_PROVISIONED_SHARED_REGISTRY",
        "mode": "SHARED_MULTI_MACHINE_EXCLUSIVE_CREATE",
        "state": "INACTIVE_TEMPLATE",
        "experiment_id": "dynamic_replay_budget_efficiency_20260807",
    }
