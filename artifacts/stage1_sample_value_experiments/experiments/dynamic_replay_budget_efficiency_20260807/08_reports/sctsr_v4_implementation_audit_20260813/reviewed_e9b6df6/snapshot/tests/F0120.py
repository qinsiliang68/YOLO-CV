from __future__ import annotations

from pathlib import Path

import pytest

from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.formal_cli import (
    FORMAL_AUTHORIZATION_INPUT_ROLES,
    build_external_file_binding,
    validate_external_file_binding,
    validate_prepared_trainer_external_files,
)
from stage1_sctsr_v4.formal_training import publish_formal_input_snapshot
from stage1_sctsr_v4.serialization import sha256_file, stable_digest


def _files(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for role in FORMAL_AUTHORIZATION_INPUT_ROLES:
        path = tmp_path / f"{role}.json"
        path.write_text(f'{{"role":"{role}"}}\n', encoding="utf-8")
        result[role] = path
    return result


def test_external_file_binding_rejects_replaced_authorization_input(tmp_path):
    files = _files(tmp_path)
    binding = build_external_file_binding(files, required_roles=FORMAL_AUTHORIZATION_INPUT_ROLES)
    assert validate_external_file_binding(
        binding,
        required_roles=FORMAL_AUTHORIZATION_INPUT_ROLES,
    )["status"] == "PASS"

    files["asset_registry"].write_text('{"replaced":true}\n', encoding="utf-8")
    with pytest.raises(SctsrError) as caught:
        validate_external_file_binding(binding, required_roles=FORMAL_AUTHORIZATION_INPUT_ROLES)
    assert caught.value.code is ErrorCode.ARTIFACT_VALIDATION_FAILED


def test_formal_input_snapshot_copies_and_binds_every_authorization_file(tmp_path):
    files = _files(tmp_path / "inputs")
    binding = build_external_file_binding(files, required_roles=FORMAL_AUTHORIZATION_INPUT_ROLES)
    run_root = tmp_path / "run"
    run_root.mkdir()

    snapshot = publish_formal_input_snapshot(run_root, binding)

    assert snapshot["schema_version"] == "stage1.sctsr.formal_input_snapshot.v1"
    assert set(snapshot["files"]) == set(FORMAL_AUTHORIZATION_INPUT_ROLES)
    assert (run_root / "00_contract" / "FORMAL_INPUT_SNAPSHOT.json").is_file()
    assert (run_root / "01_assets" / "asset_registry.json").is_file()
    for role, row in snapshot["files"].items():
        copied = run_root / row["snapshot_relative_path"]
        assert copied.is_file()
        assert sha256_file(copied) == row["sha256"] == binding["files"][role]["sha256"]


def test_prepared_trainer_external_file_revalidation_rejects_identity_manifest_drift(tmp_path):
    lock = tmp_path / "lock.json"
    model = tmp_path / "model.pt"
    overrides = tmp_path / "overrides.json"
    identities = tmp_path / "identities.csv"
    for path, content in (
        (lock, "lock\n"),
        (model, "model\n"),
        (overrides, "{}\n"),
        (identities, "sample_id,y_true\na.jpg,0\n"),
    ):
        path.write_text(content, encoding="utf-8")
    manifest_binding = {
        "schema_version": "stage1.sctsr.training_identity_manifest_binding.v1",
        "path": identities.resolve().as_posix(),
        "bytes": identities.stat().st_size,
        "sha256": sha256_file(identities),
        "row_count": 1,
        "base_manifest_sha256": "A" * 64,
        "annotated_pool_identity_count": 0,
        "schedule_digest": None,
    }
    manifest_binding["binding_digest"] = stable_digest(manifest_binding)
    core = {
        "schema_version": "stage1.sctsr.prepared_trainer_binding.v1",
        "canonical_training_lock_path": lock.resolve().as_posix(),
        "canonical_training_lock_sha256": sha256_file(lock),
        "initial_checkpoint_path": model.resolve().as_posix(),
        "initial_checkpoint_sha256": sha256_file(model),
        "trainer_overrides_path": overrides.resolve().as_posix(),
        "trainer_overrides_sha256": sha256_file(overrides),
        "identity_manifest_binding": manifest_binding,
    }
    binding = {**core, "binding_digest": stable_digest(core)}
    assert validate_prepared_trainer_external_files(binding)["status"] == "PASS"

    identities.write_text("sample_id,y_true\na.jpg,1\n", encoding="utf-8")
    with pytest.raises(SctsrError) as caught:
        validate_prepared_trainer_external_files(binding)
    assert caught.value.code is ErrorCode.ARTIFACT_VALIDATION_FAILED
