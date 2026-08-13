from __future__ import annotations

from dataclasses import asdict

import pytest

from stage1_sctsr_v4.arm_spec import ArmId
from stage1_sctsr_v4.columnar import write_zstd_parquet
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.identity_pool import partition_five_groups
from stage1_sctsr_v4.serialization import atomic_write_json, sha256_file
from stage1_sctsr_v4.selection_ledger import write_selection_partition
from stage1_sctsr_v4.synthetic_canary import _selection_rows


def _pool_artifact(tmp_path, pool, fixture, *, denominator: int):
    root = tmp_path / pool.spec.pool_role
    groups = partition_five_groups(pool, base_denominator=denominator)
    rows = [
        {
            **asdict(record),
            "identity_group": group,
            "pool_id": pool.spec.pool_id,
            "pool_role": pool.spec.pool_role,
        }
        for group, records in groups.items()
        for record in records
    ]
    membership = write_zstd_parquet(
        rows,
        root / f"run_id=POOL_{pool.spec.pool_role}" / "epoch=0000" / "identity_group_membership.parquet",
        schema_version="stage1.sctsr.identity_group_membership.v1",
        require_run_epoch_partition=True,
    )
    quota_path = root / "QUOTA_AUDIT.json"
    quota = {
        "schema_version": "stage1.sctsr.identity_pool_quota_audit.v1",
        "pool_role": pool.spec.pool_role,
        "pool_digest": pool.spec.identity_digest,
        "group_counts": {name: len(records) for name, records in groups.items()},
        "pool_build_audit": None,
        "source_bindings": {"fixture": "UNIT_ONLY"},
        "terminal_field_status": "TERMINAL_FIELDS_NOT_LOADED" if pool.spec.pool_role == "R2_MATCHED_RANDOM" else "NOT_APPLICABLE",
        "overlap_with_t": 0 if pool.spec.pool_role == "R2_MATCHED_RANDOM" else "NATURAL_OVERLAP_ALLOWED_OR_SELF",
    }
    atomic_write_json(quota_path, quota)
    selection = write_selection_partition(
        _selection_rows(fixture, pool.spec.pool_role),
        root / "selection" / f"run_id=POOL_{pool.spec.pool_role}" / "epoch=0000" / f"{pool.spec.pool_role}.parquet",
        policy=pool.spec.pool_role,
    )
    manifest = {
        "schema_version": "stage1.sctsr.identity_pool_manifest.v1",
        "pool_spec": asdict(pool.spec),
        "membership": asdict(membership),
        "selection": asdict(selection),
        "quota_audit": {
            "path": quota_path.resolve().as_posix(),
            "bytes": quota_path.stat().st_size,
            "sha256": sha256_file(quota_path),
        },
        "audit": None,
        "group_counts": quota["group_counts"],
        "pool_digest": pool.spec.identity_digest,
        "semantic": pool.spec.selection_semantic,
        "formal_training_started": False,
    }
    path = root / "POOL_MANIFEST.json"
    atomic_write_json(path, manifest)
    return path


def test_pool_artifacts_bind_schedule_bytes_and_roles(tmp_path, synthetic_fixture, phase1_schedules):
    from stage1_sctsr_v4.formal_cli import validate_identity_pool_artifacts

    t = _pool_artifact(tmp_path, synthetic_fixture.t_pool, synthetic_fixture, denominator=synthetic_fixture.base_denominator)
    r2 = _pool_artifact(tmp_path, synthetic_fixture.r2_result.pool, synthetic_fixture, denominator=synthetic_fixture.base_denominator)
    result = validate_identity_pool_artifacts(
        [t, r2],
        schedule=phase1_schedules[ArmId.T_TO_R2_AT_160],
        expected_base_denominator=synthetic_fixture.base_denominator,
        expected_base_manifest_sha256=synthetic_fixture.t_pool.spec.base_manifest_sha256,
    )
    assert result["status"] == "PASS"
    assert result["pool_roles"] == ["R2_MATCHED_RANDOM", "T_STRESS"]


def test_pool_artifacts_reject_schedule_digest_substitution(tmp_path, synthetic_fixture, phase1_schedules):
    from stage1_sctsr_v4.formal_cli import validate_identity_pool_artifacts

    t = _pool_artifact(tmp_path, synthetic_fixture.t_pool, synthetic_fixture, denominator=synthetic_fixture.base_denominator)
    with pytest.raises(SctsrError) as error:
        validate_identity_pool_artifacts(
            [t],
            schedule=phase1_schedules[ArmId.R2_U],
            expected_base_denominator=synthetic_fixture.base_denominator,
            expected_base_manifest_sha256=synthetic_fixture.t_pool.spec.base_manifest_sha256,
        )
    assert error.value.code in {ErrorCode.CONFIGURATION_MISMATCH, ErrorCode.IDENTITY_DIGEST_MISMATCH}


def test_formal_branch_rejects_nonformal_parent_receipt(monkeypatch, tmp_path):
    from stage1_sctsr_v4.formal_training import _validate_parent_completion_receipt

    checkpoint = tmp_path / "parent" / "epoch_0120.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    atomic_write_json(
        checkpoint.parent / "PARENT_RECEIPT.json",
        {
            "status": "IMPLEMENTED_NOT_FORMALLY_RUN",
            "parent_id": "PARENT_7",
            "training_seed": 7,
            "checkpoint_sha256": sha256_file(checkpoint),
            "epoch_end": 120,
            "best_pt_used": False,
        },
    )
    with pytest.raises(SctsrError) as error:
        _validate_parent_completion_receipt(checkpoint, sha256_file(checkpoint), 7, require_formal=True)
    assert error.value.code is ErrorCode.BRANCH_LINEAGE_MISMATCH


def test_formal_run_root_accepts_only_fresh_upstream_trainer_subtree(tmp_path):
    from stage1_sctsr_v4.formal_training import _prepare_run_root_after_upstream_setup

    valid = tmp_path / "valid"
    (valid / "trainer").mkdir(parents=True)
    assert _prepare_run_root_after_upstream_setup(valid, execution_mode="formal").samefile(valid)

    polluted = tmp_path / "polluted"
    (polluted / "trainer").mkdir(parents=True)
    (polluted / "foreign.txt").write_text("not registered", encoding="utf-8")
    with pytest.raises(SctsrError) as error:
        _prepare_run_root_after_upstream_setup(polluted, execution_mode="formal")
    assert error.value.code is ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE
