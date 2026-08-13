from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from stage1_sctsr_v4.columnar import validate_columnar_file, write_zstd_parquet
from stage1_sctsr_v4.filesystem import windows_safe_resolved_path
from stage1_sctsr_v4.recovery import quarantine_inprogress
from stage1_sctsr_v4.synthetic_canary import run_synthetic_canary
from stage1_sctsr_v4.run_validation import validate_run_tree
from stage1_sctsr_v4.serialization import atomic_write_json, load_json, sha256_file


@pytest.mark.skipif(os.name != "nt", reason="Win32 extended path contract")
def test_pyarrow_zstd_partition_survives_registered_path_beyond_max_path(tmp_path: Path):
    root = windows_safe_resolved_path(tmp_path / ("a" * 80) / ("b" * 80) / ("c" * 50))
    path = root / "run_id=run-1" / "epoch=0121" / "part-00000.parquet"
    assert len(str(path).removeprefix("\\\\?\\")) > 260
    try:
        manifest = write_zstd_parquet(
            [{"value": 1}],
            path,
            schema_version="stage1.sctsr.windows_long_path_canary.v1",
            require_run_epoch_partition=True,
        )
        assert validate_columnar_file(path, expected_sha256=manifest.sha256)["status"] == "PASS"
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.skipif(os.name != "nt", reason="Win32 extended path contract")
def test_atomic_json_and_hash_survive_transaction_identity_path_beyond_max_path(tmp_path: Path):
    root = tmp_path / ("registered_experiment_" + "a" * 80) / ("implementation_audit_" + "b" * 80)
    destination = root / "real_data_canary" / "r1" / "03_epoch_transactions" / "epoch_0121.generation_1.inprogress" / "TRANSACTION_IDENTITY.json"
    assert len(str(destination)) > 260
    try:
        atomic_write_json(destination, {"status": "INPROGRESS", "generation": 1})
        assert load_json(destination) == {"status": "INPROGRESS", "generation": 1}
        assert len(sha256_file(destination)) == 64
    finally:
        shutil.rmtree(windows_safe_resolved_path(root), ignore_errors=True)


@pytest.mark.skipif(os.name != "nt", reason="Win32 extended path contract")
def test_quarantine_accepts_unprefixed_registered_root_beyond_max_path(tmp_path: Path):
    root = tmp_path / ("registered_experiment_" + "a" * 80) / ("implementation_audit_" + "b" * 80)
    transaction_root = root / "03_epoch_transactions"
    quarantine_root = root / "09_quarantine"
    partial = windows_safe_resolved_path(transaction_root) / "epoch_0121.generation_1.inprogress"
    try:
        partial.mkdir(parents=True)
        atomic_write_json(partial / "TRANSACTION_IDENTITY.json", {"status": "INPROGRESS", "generation": 1})

        moved = quarantine_inprogress(
            transaction_root,
            quarantine_root,
            reason="ACTUAL_ENGINEERING_CANARY_KILL",
        )

        assert len(moved) == 1
        target = windows_safe_resolved_path(moved[0])
        assert target.is_dir()
        assert load_json(target / "QUARANTINE_RECEIPT.json")["status"] == "QUARANTINED"
        assert not partial.exists()
    finally:
        shutil.rmtree(windows_safe_resolved_path(root), ignore_errors=True)


@pytest.mark.skipif(os.name != "nt", reason="Win32 extended path contract")
def test_complete_synthetic_canary_normalizes_a_long_registered_root(repository_root: Path, tmp_path: Path):
    root = tmp_path / ("registered_experiment_" + "a" * 70) / ("implementation_audit_" + "b" * 70) / "canary"
    probe = root / "04_ledgers" / "selection" / "run_id=SYNTHETIC_SELECTION" / "epoch=0000" / ".T_STRESS.parquet.inprogress"
    assert len(str(probe)) > 260

    receipt = run_synthetic_canary(root, repository_root=repository_root, training_seed=20260606)

    assert receipt["status"] == "PASS"
    assert (windows_safe_resolved_path(root) / "RUN_MANIFEST.json").is_file()
    assert validate_run_tree(root)["status"] == "PASS"
