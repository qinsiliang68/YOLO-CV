from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    repository = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(repository))
    sys.path.insert(0, str(repository / "tests" / "stage1_sctsr_v4"))

    from test_formal_execution_claim import NOW, SECRET, _claim_registry, _job, _release, _token
    from stage1_sctsr_v4.errors import SctsrError
    from stage1_sctsr_v4.formal_execution import (
        claim_formal_execution,
        execute_fenced_finalization,
        validate_execution_claim_binding,
    )
    from stage1_sctsr_v4.serialization import atomic_write_json, stable_digest

    evidence: dict[str, object] = {
        "schema_version": "stage1.sctsr.training_output_fix_heartbeat_examples.v1",
        "scientific_role": "SYNTHETIC_CONTROL_PLANE_EVIDENCE_NOT_TRAINING",
    }

    with tempfile.TemporaryDirectory(prefix="sctsr_heartbeat_examples_") as raw_temp:
        temp = Path(raw_temp)

        complete_root = temp / "complete"
        complete_root.mkdir()
        release, trust, release_bindings = _release()
        claim_root, registry = _claim_registry(complete_root)
        complete_job = _job(complete_root)
        complete_token = complete_root / "START.json"
        atomic_write_json(complete_token, _token(release, registry, complete_job, execution_id="EXAMPLE_COMPLETE"))
        common = {
            "claim_registry_root": claim_root,
            "release": release,
            "release_trust_policy": trust,
            "expected_release_bindings": release_bindings,
            "release_manifest_sha256": stable_digest(release),
            "verification_secret": SECRET,
        }
        complete_claim = claim_formal_execution(
            complete_token,
            expected_job_bindings=complete_job,
            now_utc=NOW,
            **common,
        )
        execute_fenced_finalization(
            complete_claim,
            expected_job_bindings=complete_job,
            operation=lambda: {"status": "FORMAL_PARENT_COMPLETE", "receipt_sha256": "A" * 64},
            now_utc=NOW,
        )
        evidence["complete_heartbeat"] = json.loads(Path(complete_claim["lease_heartbeat_path"]).read_text(encoding="utf-8"))

        failed_root = temp / "failed"
        failed_root.mkdir()
        release, trust, release_bindings = _release()
        claim_root, registry = _claim_registry(failed_root)
        failed_job = _job(failed_root)
        failed_token = failed_root / "START.json"
        atomic_write_json(failed_token, _token(release, registry, failed_job, execution_id="EXAMPLE_FAILED"))
        failed_common = {
            "claim_registry_root": claim_root,
            "release": release,
            "release_trust_policy": trust,
            "expected_release_bindings": release_bindings,
            "release_manifest_sha256": stable_digest(release),
            "verification_secret": SECRET,
        }
        failed_claim = claim_formal_execution(
            failed_token,
            expected_job_bindings=failed_job,
            now_utc=NOW,
            **failed_common,
        )

        def fail_operation() -> None:
            raise RuntimeError("synthetic endpoint failure")

        try:
            execute_fenced_finalization(
                failed_claim,
                expected_job_bindings=failed_job,
                operation=fail_operation,
                now_utc=NOW,
            )
        except RuntimeError:
            pass
        evidence["failed_heartbeat"] = json.loads(Path(failed_claim["lease_heartbeat_path"]).read_text(encoding="utf-8"))

        stale_root = temp / "stale"
        stale_root.mkdir()
        release, trust, release_bindings = _release()
        claim_root, registry = _claim_registry(stale_root)
        start_job = _job(stale_root)
        start_token = stale_root / "START.json"
        atomic_write_json(
            start_token,
            _token(
                release,
                registry,
                start_job,
                execution_id="EXAMPLE_STALE_START",
                expires_at_utc="2026-08-14T23:00:00Z",
            ),
        )
        stale_common = {
            "claim_registry_root": claim_root,
            "release": release,
            "release_trust_policy": trust,
            "expected_release_bindings": release_bindings,
            "release_manifest_sha256": stable_digest(release),
            "verification_secret": SECRET,
        }
        start_claim = claim_formal_execution(
            start_token,
            expected_job_bindings=start_job,
            now_utc=NOW,
            **stale_common,
        )
        resume_job = _job(stale_root, action="RESUME")
        resume_token = stale_root / "RESUME.json"
        atomic_write_json(
            resume_token,
            _token(
                release,
                registry,
                resume_job,
                execution_id="EXAMPLE_STALE_RESUME",
                expires_at_utc="2026-08-14T23:00:00Z",
            ),
        )
        resume_claim = claim_formal_execution(
            resume_token,
            expected_job_bindings=resume_job,
            now_utc=datetime(2026, 8, 14, 19, tzinfo=timezone.utc),
            **stale_common,
        )
        try:
            validate_execution_claim_binding(
                start_claim,
                expected_job_bindings=start_job,
                require_token_file=False,
            )
        except SctsrError as error:
            evidence["stale_attempt_rejection"] = {
                "error_code": error.code.value,
                "old_fence_generation": start_claim["fence_generation"],
                "new_fence_generation": resume_claim["fence_generation"],
                "new_execution_id": resume_claim["execution_id"],
            }
        else:
            raise AssertionError("Superseded attempt was not fenced")

    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
