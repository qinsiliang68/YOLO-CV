from __future__ import annotations

import pytest

from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.formal_completion import publish_formal_completion, validate_formal_completion
from stage1_sctsr_v4.prediction_runtime import prepare_formal_endpoint_publication
from stage1_sctsr_v4.run_validation import build_artifact_index
from stage1_sctsr_v4.serialization import atomic_write_json, load_json, sha256_file


def _prepare_control_files(root, *, role: str):
    root.mkdir()
    state_name = "PARENT_RECEIPT.json" if role == "COMMON_PARENT" else "BRANCH_RECEIPT.json"
    pending_status = (
        "FORMAL_PARENT_EPOCHS_COMPLETE_PENDING_FINALIZATION"
        if role == "COMMON_PARENT"
        else "FORMAL_BRANCH_EPOCHS_COMPLETE_PENDING_ENDPOINT"
    )
    atomic_write_json(
        root / state_name,
        {
            "schema_version": "stage1.sctsr.formal_parent_receipt.v3"
            if role == "COMMON_PARENT"
            else "stage1.sctsr.formal_branch_receipt.v3",
            "status": pending_status,
            "epoch_start": 1 if role == "COMMON_PARENT" else 121,
            "epoch_end": 120 if role == "COMMON_PARENT" else 200,
            "epoch_receipt_digest": "A" * 64,
            "checkpoint_sha256": "B" * 64,
            "fixed_formal_endpoint": {"sha256": "B" * 64},
            "parent_id": "PARENT_17",
            "logical_run_id": "RUN_17_T_U",
            "arm_id": "COMMON_PARENT_NR" if role == "COMMON_PARENT" else "T_U",
            "training_seed": 17,
        },
    )
    atomic_write_json(
        root / "RUN_MANIFEST.json",
        {
            "execution_mode": "formal",
            "run_role": role,
            "run_id": "PARENT_17" if role == "COMMON_PARENT" else "RUN_17_T_U",
            "arm_id": "COMMON_PARENT_NR" if role == "COMMON_PARENT" else "T_U",
            "training_seed": 17,
        },
    )
    atomic_write_json(
        root / "ARTIFACT_INDEX_GENERATIONS.json",
        {"schema_version": "stage1.sctsr.epoch_artifact_index.v1", "epoch_generations": []},
    )
    atomic_write_json(root / "ARTIFACT_INDEX.json", build_artifact_index(root))
    return root / state_name


def test_completion_marker_is_last_atomic_fact_and_detects_later_tamper(tmp_path):
    root = tmp_path / "parent"
    state_path = _prepare_control_files(root, role="COMMON_PARENT")

    receipt = publish_formal_completion(
        root,
        run_role="COMMON_PARENT",
        run_id="PARENT_17",
        arm_id="COMMON_PARENT_NR",
        training_seed=17,
        terminal_epoch=120,
        fixed_checkpoint_sha256="B" * 64,
    )

    marker = root / "FORMAL_COMPLETION_RECEIPT.json"
    assert marker.is_file()
    assert receipt["status"] == "FORMAL_PARENT_COMPLETE"
    assert validate_formal_completion(root, expected_run_role="COMMON_PARENT")["status"] == "PASS"
    assert "FORMAL_COMPLETION_RECEIPT.json" not in {
        row["path"] for row in build_artifact_index(root)["files"]
    }

    state_path.write_text("{}", encoding="utf-8")
    with pytest.raises(SctsrError) as caught:
        validate_formal_completion(root, expected_run_role="COMMON_PARENT")
    assert caught.value.code is ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE


def test_branch_cannot_complete_before_endpoint_receipt_and_final_index(tmp_path):
    root = tmp_path / "branch"
    _prepare_control_files(root, role="BRANCH")

    with pytest.raises(SctsrError) as caught:
        publish_formal_completion(
            root,
            run_role="BRANCH",
            run_id="RUN_17_T_U",
            arm_id="T_U",
            training_seed=17,
            terminal_epoch=200,
            fixed_checkpoint_sha256="B" * 64,
        )
    assert caught.value.code is ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE
    assert not (root / "FORMAL_COMPLETION_RECEIPT.json").exists()

    endpoint = root / "08_receipts" / "FORMAL_ENDPOINT_RECEIPT.json"
    atomic_write_json(endpoint, {"status": "FORMAL_ENDPOINT_COMPLETE_NOT_METHOD_SELECTION"})
    state_path = root / "BRANCH_RECEIPT.json"
    state = load_json(state_path)
    atomic_write_json(
        state_path,
        {
            **state,
            "status": "FORMAL_BRANCH_ENDPOINT_COMPLETE_PENDING_COMMIT",
            "formal_endpoint_evidence": {"status": "PASS"},
        },
    )
    atomic_write_json(root / "ARTIFACT_INDEX.json", build_artifact_index(root))
    receipt = publish_formal_completion(
        root,
        run_role="BRANCH",
        run_id="RUN_17_T_U",
        arm_id="T_U",
        training_seed=17,
        terminal_epoch=200,
        fixed_checkpoint_sha256="B" * 64,
    )

    assert receipt["status"] == "FORMAL_BRANCH_COMPLETE"
    assert receipt["endpoint_receipt_sha256"] == sha256_file(endpoint)
    assert validate_formal_completion(root, expected_run_role="BRANCH")["status"] == "PASS"


def test_partial_endpoint_is_quarantined_before_finalization_retry(tmp_path):
    root = tmp_path / "branch"
    partial = root / "06_predictions" / "run_id=RUN_17_T_U" / "epoch=0200" / "predictions.parquet"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")

    result = prepare_formal_endpoint_publication(
        root,
        run_id="RUN_17_T_U",
        validate_existing=lambda: pytest.fail("partial endpoint must not be reused"),
    )

    assert result["action"] == "BUILD_FRESH"
    assert len(result["quarantined_artifacts"]) == 1
    assert not partial.exists()
    target = root / result["quarantined_artifacts"][0]["target"]
    assert target.read_bytes() == b"partial"


def test_complete_valid_endpoint_is_reused_without_reinference(tmp_path):
    root = tmp_path / "branch"
    prediction_root = root / "06_predictions" / "run_id=RUN_17_T_U" / "epoch=0200"
    evaluation_root = root / "07_evaluation" / "run_id=RUN_17_T_U" / "epoch=0200"
    paths = (
        prediction_root / "split_identity_bundle.json",
        prediction_root / "predictions.parquet",
        prediction_root / "prediction_summary.json",
        evaluation_root / "frontier.parquet",
        evaluation_root / "frontier_summary.json",
        root / "08_receipts" / "FORMAL_ENDPOINT_RECEIPT.json",
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode("ascii"))
    calls = []

    result = prepare_formal_endpoint_publication(
        root,
        run_id="RUN_17_T_U",
        validate_existing=lambda: calls.append("validated") or {"status": "PASS"},
    )

    assert result["action"] == "REUSE_VALID_ENDPOINT"
    assert result["validation"] == {"status": "PASS"}
    assert calls == ["validated"]
    assert not (root / "09_quarantine").exists()
