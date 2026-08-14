from __future__ import annotations

import json
from pathlib import Path

import pytest

from stage1_sctsr_v4 import epoch_transaction
from stage1_sctsr_v4.epoch_transaction import EpochTransaction, validate_receipt_chain
from stage1_sctsr_v4.recovery import validate_recovery_pointer


def _started_transaction(tmp_path: Path, *, epoch: int = 121) -> EpochTransaction:
    transaction = EpochTransaction(tmp_path / "03_epoch_transactions", "run-1", epoch, 1).begin()
    transaction.write_json("value.json", {"epoch": epoch})
    return transaction


def test_post_rename_pre_receipt_failure_quarantines_complete(monkeypatch, tmp_path):
    transaction = _started_transaction(tmp_path)

    def fail_receipt(*_args, **_kwargs):
        raise OSError("injected receipt publication failure")

    monkeypatch.setattr(transaction, "_append_receipt", fail_receipt)
    with pytest.raises(OSError, match="injected receipt"):
        transaction.commit()

    assert not transaction.inprogress.exists()
    assert not transaction.complete.exists()
    quarantined = list((tmp_path / "09_quarantine").glob("*.quarantined.*"))
    assert len(quarantined) == 1
    receipt = json.loads((quarantined[0] / "QUARANTINE_RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["source"] == transaction.complete.as_posix()
    assert receipt["reason"] == "POST_RENAME_PRE_RECEIPT_PUBLICATION_FAILED"
    assert not transaction.receipt_path.exists()
    assert not transaction.artifact_index_path.exists()
    assert not transaction.recovery_pointer_path.exists()


def test_post_receipt_index_failure_repairs_secondary_metadata(monkeypatch, tmp_path):
    transaction = _started_transaction(tmp_path)
    original = transaction._update_artifact_index
    calls = 0

    def fail_index_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected index publication failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(transaction, "_update_artifact_index", fail_index_once)
    with pytest.raises(OSError, match="injected index"):
        transaction.commit()

    assert transaction.complete.is_dir()
    assert not list((tmp_path / "09_quarantine").glob("*.quarantined.*"))
    assert validate_receipt_chain(transaction.receipt_path)["row_count"] == 1
    pointer = validate_recovery_pointer(transaction.recovery_pointer_path)
    assert pointer["epoch"] == 121
    index = json.loads(transaction.artifact_index_path.read_text(encoding="utf-8"))
    assert len(index["epoch_generations"]) == 1


def test_post_index_pointer_failure_repairs_pointer(monkeypatch, tmp_path):
    transaction = _started_transaction(tmp_path)
    original = epoch_transaction.atomic_write_json
    failed = False

    def fail_pointer_once(path, value):
        nonlocal failed
        if Path(path).name == "ROLLING_RECOVERY_POINTER.json" and not failed:
            failed = True
            raise OSError("injected pointer publication failure")
        return original(path, value)

    monkeypatch.setattr(epoch_transaction, "atomic_write_json", fail_pointer_once)
    with pytest.raises(OSError, match="injected pointer"):
        transaction.commit()

    assert transaction.complete.is_dir()
    assert validate_receipt_chain(transaction.receipt_path)["row_count"] == 1
    assert validate_recovery_pointer(transaction.recovery_pointer_path)["epoch"] == 121


def test_next_begin_quarantines_unreceipted_complete_left_by_process_death(tmp_path):
    transaction_root = tmp_path / "03_epoch_transactions"
    orphan = transaction_root / "epoch_0121.generation_1.complete"
    orphan.mkdir(parents=True)
    (orphan / "crash-marker.bin").write_bytes(b"power-loss-after-rename")

    next_transaction = EpochTransaction(transaction_root, "run-1", 122, 1).begin()

    assert next_transaction.inprogress.is_dir()
    assert not orphan.exists()
    quarantined = list((tmp_path / "09_quarantine").glob("*.quarantined.*"))
    assert len(quarantined) == 1
    receipt = json.loads((quarantined[0] / "QUARANTINE_RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["source"] == orphan.as_posix()
    assert receipt["reason"] == "UNRECEIPTED_COMPLETE_RECOVERED_BEFORE_BEGIN"
