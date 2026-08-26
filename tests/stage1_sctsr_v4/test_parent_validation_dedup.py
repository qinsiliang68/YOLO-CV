from __future__ import annotations

import pytest

from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.formal_cli import (
    _consume_fresh_parent_artifact_validation,
    validate_parent_artifact_index,
)
from stage1_sctsr_v4.formal_completion import (
    _validate_formal_completion_with_prevalidated_artifact_index,
    publish_formal_completion,
)
from stage1_sctsr_v4.run_validation import build_artifact_index
from stage1_sctsr_v4.serialization import atomic_write_json, load_json, sha256_file


def _completed_parent(root):
    root.mkdir()
    atomic_write_json(
        root / "PARENT_RECEIPT.json",
        {
            "schema_version": "stage1.sctsr.formal_parent_receipt.v3",
            "status": "FORMAL_PARENT_EPOCHS_COMPLETE_PENDING_FINALIZATION",
            "epoch_start": 1,
            "epoch_end": 120,
            "epoch_receipt_digest": "A" * 64,
            "checkpoint_sha256": "B" * 64,
            "parent_id": "PARENT_17",
            "arm_id": "COMMON_PARENT_NR",
            "training_seed": 17,
            "best_pt_used": False,
        },
    )
    atomic_write_json(
        root / "RUN_MANIFEST.json",
        {
            "execution_mode": "formal",
            "run_role": "COMMON_PARENT",
            "run_id": "PARENT_17",
            "arm_id": "COMMON_PARENT_NR",
            "training_seed": 17,
        },
    )
    atomic_write_json(
        root / "ARTIFACT_INDEX_GENERATIONS.json",
        {"schema_version": "stage1.sctsr.epoch_artifact_index.v1", "epoch_generations": []},
    )
    atomic_write_json(root / "ARTIFACT_INDEX.json", build_artifact_index(root))
    publish_formal_completion(
        root,
        run_role="COMMON_PARENT",
        run_id="PARENT_17",
        arm_id="COMMON_PARENT_NR",
        training_seed=17,
        terminal_epoch=120,
        fixed_checkpoint_sha256="B" * 64,
    )


def test_same_process_parent_artifact_proof_is_one_shot(monkeypatch, tmp_path):
    root = tmp_path / "parent"
    root.mkdir()
    checkpoint = root / "epoch_0120.pt"
    checkpoint.write_bytes(b"checkpoint")
    index = root / "ARTIFACT_INDEX.json"
    atomic_write_json(index, {"schema_version": "stage1.sctsr.artifact_index.v1", "files": [], "artifact_index_digest": "D" * 64})
    checkpoint_sha = sha256_file(checkpoint)
    monkeypatch.setattr(
        "stage1_sctsr_v4.run_validation.validate_run_tree",
        lambda *_args, **_kwargs: {
            "semantic": "formal",
            "run_role": "COMMON_PARENT",
            "fixed_endpoint_checkpoint_sha256": checkpoint_sha,
            "manifest_sha256": "A" * 64,
            "artifact_digest": "B" * 64,
            "receipt_chain_digest": "C" * 64,
            "epoch_transaction_count": 120,
        },
    )

    binding = validate_parent_artifact_index(parent_checkpoint=checkpoint, parent_artifact_index=index)

    kwargs = {
        "parent_checkpoint": checkpoint,
        "parent_sha256": checkpoint_sha,
        "parent_root": root,
    }
    assert _consume_fresh_parent_artifact_validation(binding, **kwargs) is True
    assert _consume_fresh_parent_artifact_validation(binding, **kwargs) is False


def test_prevalidated_completion_skips_only_tree_rebuild(monkeypatch, tmp_path):
    root = tmp_path / "parent"
    _completed_parent(root)
    monkeypatch.setattr(
        "stage1_sctsr_v4.run_validation.build_artifact_index",
        lambda *_args, **_kwargs: pytest.fail("same invocation rebuilt the full parent tree"),
    )

    result = _validate_formal_completion_with_prevalidated_artifact_index(
        root,
        expected_run_role="COMMON_PARENT",
    )

    assert result["status"] == "PASS"
    index = load_json(root / "ARTIFACT_INDEX.json")
    index["artifact_index_digest"] = "0" * 64
    atomic_write_json(root / "ARTIFACT_INDEX.json", index)
    with pytest.raises(SctsrError) as caught:
        _validate_formal_completion_with_prevalidated_artifact_index(root, expected_run_role="COMMON_PARENT")
    assert caught.value.code is ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE
