from __future__ import annotations

import pytest

from stage1_sctsr_v4.arm_spec import ArmId
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.formal_cli import validate_resume_identity_pool_binding
from stage1_sctsr_v4.serialization import atomic_write_json, sha256_file, stable_digest


def test_resume_reuses_frozen_pool_binding_and_still_checks_manifest(tmp_path, phase1_schedules):
    schedule = phase1_schedules[ArmId.R2_U]
    run_root = tmp_path / "run"
    pool_root = tmp_path / "pool"
    pool_root.mkdir()
    manifest = pool_root / "POOL_MANIFEST.json"
    atomic_write_json(
        manifest,
        {
            "pool_spec": {
                "pool_role": "R2_MATCHED_RANDOM",
                "identity_digest": schedule.identity_pool_digest,
            },
            "pool_digest": schedule.identity_pool_digest,
        },
    )
    artifact_bindings = {
        "R2_MATCHED_RANDOM": {
            "manifest_path": manifest.resolve().as_posix(),
            "manifest_sha256": sha256_file(manifest),
        }
    }
    binding = {
        "status": "PASS",
        "arm_id": schedule.arm_id.value,
        "pool_roles": ["R2_MATCHED_RANDOM"],
        "pool_digests": {"R2_MATCHED_RANDOM": schedule.identity_pool_digest},
        "artifact_bindings": artifact_bindings,
        "binding_digest": stable_digest(artifact_bindings),
    }
    atomic_write_json(run_root / "IDENTITY_POOL_BINDING.json", binding)

    assert validate_resume_identity_pool_binding(
        run_root=run_root,
        manifest_paths=[manifest],
        schedule=schedule,
    ) == binding

    atomic_write_json(manifest, {"pool_spec": {"pool_role": "R2_MATCHED_RANDOM", "identity_digest": "0" * 64}, "pool_digest": "0" * 64})
    with pytest.raises(SctsrError) as caught:
        validate_resume_identity_pool_binding(run_root=run_root, manifest_paths=[manifest], schedule=schedule)
    assert caught.value.code is ErrorCode.IDENTITY_DIGEST_MISMATCH
