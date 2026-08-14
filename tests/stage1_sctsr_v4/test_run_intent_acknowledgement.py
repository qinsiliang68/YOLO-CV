from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.run_intent import (
    REQUIRED_ACKNOWLEDGEMENT_STATEMENTS,
    RunIntentContext,
    build_run_intent_acknowledgement,
    build_runbook_manifest,
    build_run_intent_binding,
    publish_run_intent_snapshot,
    validate_run_intent_acknowledgement,
    validate_run_intent_snapshot_chain,
)
from stage1_sctsr_v4.serialization import atomic_write_json, sha256_file


def _context(runbook_sha: str) -> RunIntentContext:
    return RunIntentContext(
        action="START",
        run_role="BRANCH",
        logical_run_id="SCTSR_DISCOVERY_S001_T_U",
        arm_id="T_U",
        training_seed=2026081401,
        dataset_root="C:/data/final_sewerml_dataset",
        output_root="D:/sctsr/SCTSR_DISCOVERY_S001_T_U",
        formal_identity_sha256="1" * 64,
        trainer_overrides_sha256="2" * 64,
        identity_manifest_sha256="3" * 64,
        source_tree_digest="4" * 64,
        contract_digest="5" * 64,
        asset_registry_digest="6" * 64,
        runtime_config_digest="7" * 64,
        dataset_content_identity_digest="8" * 64,
        parent_checkpoint_sha256="9" * 64,
        schedule_digest="A" * 64,
        identity_pool_binding_digest="B" * 64,
        release_manifest_sha256="C" * 64,
        execution_token_sha256="D" * 64,
        claim_registry_root_digest="E" * 64,
        resume_checkpoint_sha256="0" * 64,
        resume_receipt_digest="0" * 64,
        runbook_manifest_sha256=runbook_sha,
    )


def _runbook(tmp_path: Path) -> tuple[Path, tuple[Path, ...]]:
    docs = (tmp_path / "intent.md", tmp_path / "fairness.md")
    docs[0].write_text("intent\n", encoding="utf-8")
    docs[1].write_text("fairness\n", encoding="utf-8")
    output = tmp_path / "RUNBOOK_MANIFEST.json"
    build_runbook_manifest(repository_root=tmp_path, document_paths=docs, output_path=output)
    return output, docs


def _valid_ack(tmp_path: Path) -> tuple[Path, RunIntentContext, Path]:
    runbook, _ = _runbook(tmp_path)
    context = _context(sha256_file(runbook))
    payload = build_run_intent_acknowledgement(
        context=context,
        acknowledgement_id="SCTSR_ACK_DISCOVERY_S001_T_U",
        operator_agent_id="TRAINING_MACHINE_AI_01",
        machine_id="NODE_01",
        created_at_utc=datetime.now(timezone.utc).replace(microsecond=0),
    )
    path = tmp_path / "RUN_INTENT_ACKNOWLEDGEMENT.json"
    atomic_write_json(path, payload)
    return path, context, runbook


def test_valid_run_intent_acknowledgement_binds_job_and_runbook_bytes(tmp_path: Path) -> None:
    path, context, runbook = _valid_ack(tmp_path)

    report = validate_run_intent_acknowledgement(
        path,
        expected_context=context,
        runbook_manifest_path=runbook,
        repository_root=tmp_path,
    )

    assert report["status"] == "PASS"
    assert report["acknowledgement_sha256"] == sha256_file(path)
    assert report["all_required_statements_true"] is True
    assert report["runbook_manifest_sha256"] == sha256_file(runbook)


def test_false_understanding_statement_is_rejected(tmp_path: Path) -> None:
    path, context, runbook = _valid_ack(tmp_path)
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    statement = next(iter(REQUIRED_ACKNOWLEDGEMENT_STATEMENTS))
    payload["statements"][statement] = False
    payload["acknowledgement_digest"] = "0" * 64
    atomic_write_json(path, payload)

    with pytest.raises(SctsrError) as error:
        validate_run_intent_acknowledgement(
            path,
            expected_context=context,
            runbook_manifest_path=runbook,
            repository_root=tmp_path,
        )
    assert error.value.code is ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("training_seed", 2026081499),
        ("arm_id", "R2_U"),
        ("output_root", "D:/sctsr/other"),
        ("parent_checkpoint_sha256", "F" * 64),
        ("schedule_digest", "F" * 64),
        ("execution_token_sha256", "F" * 64),
    ),
)
def test_job_substitution_is_rejected(tmp_path: Path, field: str, replacement: object) -> None:
    path, context, runbook = _valid_ack(tmp_path)

    with pytest.raises(SctsrError) as error:
        validate_run_intent_acknowledgement(
            path,
            expected_context=replace(context, **{field: replacement}),
            runbook_manifest_path=runbook,
            repository_root=tmp_path,
        )
    assert error.value.code is ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED


def test_runbook_document_drift_invalidates_acknowledgement(tmp_path: Path) -> None:
    path, context, runbook = _valid_ack(tmp_path)
    (tmp_path / "intent.md").write_text("changed after acknowledgement\n", encoding="utf-8")

    with pytest.raises(SctsrError) as error:
        validate_run_intent_acknowledgement(
            path,
            expected_context=context,
            runbook_manifest_path=runbook,
            repository_root=tmp_path,
        )
    assert error.value.code is ErrorCode.RUNBOOK_IDENTITY_MISMATCH


def test_acknowledgement_digest_tamper_is_rejected(tmp_path: Path) -> None:
    path, context, runbook = _valid_ack(tmp_path)
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["machine_id"] = "NODE_02"
    atomic_write_json(path, payload)

    with pytest.raises(SctsrError) as error:
        validate_run_intent_acknowledgement(
            path,
            expected_context=context,
            runbook_manifest_path=runbook,
            repository_root=tmp_path,
        )
    assert error.value.code is ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED


def test_published_run_intent_snapshot_is_immutable_and_closeout_verifiable(tmp_path: Path) -> None:
    acknowledgement, context, runbook = _valid_ack(tmp_path)
    validation = validate_run_intent_acknowledgement(
        acknowledgement,
        expected_context=context,
        runbook_manifest_path=runbook,
        repository_root=tmp_path,
    )
    binding = build_run_intent_binding(
        acknowledgement_path=acknowledgement,
        runbook_manifest_path=runbook,
        repository_root=tmp_path,
        context=context,
        validation=validation,
    )
    run_root = tmp_path / "formal_run"
    run_root.mkdir()

    published = publish_run_intent_snapshot(run_root, binding)
    checked = validate_run_intent_snapshot_chain(
        run_root,
        repository_root=tmp_path,
        expected_latest_snapshot_digest=published["snapshot_digest"],
    )

    assert checked["status"] == "PASS"
    assert checked["attempt_count"] == 1
    assert checked["latest_snapshot_digest"] == published["snapshot_digest"]


def test_closeout_rejects_runbook_drift_after_snapshot_publication(tmp_path: Path) -> None:
    acknowledgement, context, runbook = _valid_ack(tmp_path)
    validation = validate_run_intent_acknowledgement(
        acknowledgement,
        expected_context=context,
        runbook_manifest_path=runbook,
        repository_root=tmp_path,
    )
    binding = build_run_intent_binding(
        acknowledgement_path=acknowledgement,
        runbook_manifest_path=runbook,
        repository_root=tmp_path,
        context=context,
        validation=validation,
    )
    run_root = tmp_path / "formal_run"
    run_root.mkdir()
    publish_run_intent_snapshot(run_root, binding)
    (tmp_path / "intent.md").write_text("drift after start\n", encoding="utf-8")

    with pytest.raises(SctsrError) as error:
        validate_run_intent_snapshot_chain(run_root, repository_root=tmp_path)
    assert error.value.code is ErrorCode.RUNBOOK_IDENTITY_MISMATCH


def test_formal_training_rejects_missing_run_intent_before_dataset_iteration(monkeypatch, tmp_path: Path) -> None:
    import stage1_sctsr_v4.formal_training as formal

    monkeypatch.setattr(formal, "require_synthetic_or_authorized", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(formal, "validate_execution_claim_binding", lambda *_args, **_kwargs: {})
    identity = formal.FormalIdentity(
        training_seed=7,
        canonical_training_lock_sha256="1" * 64,
        initial_checkpoint_sha256="2" * 64,
        base_manifest_sha256="3" * 64,
        source_tree_digest="4" * 64,
        runtime_config_digest="5" * 64,
        asset_registry_digest="6" * 64,
        contract_digest="7" * 64,
        seed_registry_digest="8" * 64,
    )

    with pytest.raises(SctsrError) as error:
        formal.run_prepared_common_parent(
            trainer=object(),
            identity=identity,
            output_root=tmp_path / "must_not_exist",
            release_authorization={"unit": True},
            release_expected_bindings={},
            execution_claim_binding={"unit": True},
            prepared_trainer_binding={"unit": True},
            formal_input_binding={"unit": True},
            execution_mode="formal",
        )
    assert error.value.code is ErrorCode.RUN_INTENT_NOT_ACKNOWLEDGED
    assert not (tmp_path / "must_not_exist").exists()
