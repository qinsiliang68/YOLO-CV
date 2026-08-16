from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from stage1_sctsr_v4.checkpointing import build_checkpoint_payload, save_checkpoint_atomic
from stage1_sctsr_v4.arm_spec import ArmId
from stage1_sctsr_v4.branch_lineage import BranchLineage
from stage1_sctsr_v4.epoch_transaction import EpochTransaction, GenerationIdentity
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.fixed_step_runtime import ExponentialMovingAverage
from stage1_sctsr_v4.formal_training import FormalIdentity, run_prepared_branch
from stage1_sctsr_v4.identity_pool import IdentityRecord, identity_digest
from stage1_sctsr_v4.occurrence_ledger import write_occurrence_partition
from stage1_sctsr_v4.recovery import inspect_formal_resume_context, prepare_formal_resume_context
from stage1_sctsr_v4.rng_isolation import capture_global_rng
from stage1_sctsr_v4.serialization import sha256_file, stable_digest
from stage1_sctsr_v4.schedule import build_schedule, schedule_to_dict
from stage1_sctsr_v4.serialization import atomic_write_json


RUN_ID = "RUN_RESUME_T_U"
ARM_ID = "T_U"
SEED = 17
PARENT_SHA = "A" * 64
SOURCE_SHA = "B" * 64
CONTRACT_SHA = "C" * 64
ASSET_SHA = "D" * 64
RUNTIME_SHA = "E" * 64
LOCK_SHA = "F" * 64
INITIAL_SHA = "1" * 64
BASE_SHA = "2" * 64


class FakeScaler:
    def state_dict(self):
        return {}

    def load_state_dict(self, _state):
        return None


class FakeTrainer:
    def __init__(self):
        self.model = torch.nn.Linear(2, 2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)
        self.scaler = FakeScaler()
        self.ema = ExponentialMovingAverage.from_model(self.model)
        self.train_loader = SimpleNamespace(sampler=SimpleNamespace(set_epoch=lambda _epoch: None))


def _occurrence_row(epoch: int) -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "parent_id": "PARENT_17",
        "arm_id": ARM_ID,
        "training_seed": SEED,
        "epoch": epoch,
        "base_batch_index": 0,
        "global_step_before": 120 * 938,
        "occurrence_role": "REPLAY",
        "occurrence_index_in_step": 0,
        "sample_id": "sample-1",
        "y_true": 0,
        "replay_role": "T_STRESS",
        "identity_pool_id": "T",
        "identity_group": "G0",
        "selection_policy": "T_STRESS",
        "selection_reason_code": "PLANNED_REPLAY_STEP_SLOT",
        "oof_fold": 1,
        "oof_group_id": "bucket-1",
        "oof_group_semantic": "FILENAME_BUCKET_SURROGATE_NOT_TRUE_VIDEO_ID",
        "historical_dynamic_bucket": "stable_high",
        "augmentation_seed": 123,
        "augmentation_trace_digest": "3" * 64,
        "replay_count_before": 0,
        "replay_count_after": 1,
        "last_replay_epoch": None,
        "last_replay_epoch_reason": "NEVER_REPLAYED",
        "epochs_since_last_replay": None,
        "epochs_since_last_replay_reason": "NEVER_REPLAYED",
        "logit_normal": 1.0,
        "logit_defect": 0.0,
        "p_defect_raw": 0.26894143,
        "ce_unreduced": 0.3132617,
        "margin_defect_minus_normal": -1.0,
        "predicted_label_argmax": 0,
        "correct_argmax": True,
        "oof_reference_probability": None,
        "oof_reference_reason": "REGISTERED_NOT_AVAILABLE",
        "rho_candidate_signal": None,
        "rho_reason": "REGISTERED_NOT_REPORTED",
        "row_generation": 1,
        "planned_replay_epoch": epoch,
        "planned_replay_epoch_reason": "PRESENT",
        "planned_step_slot": 0,
        "planned_step_slot_reason": "PRESENT",
        "cumulative_replay_count_before": 0,
        "cumulative_replay_count_after": 1,
        "pool_multiplicity_target": 16,
        "schedule_family": "U",
        "fallback_state": "TARGETED",
    }


def _checkpoint_payload(epoch: int) -> dict[str, object]:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    return build_checkpoint_payload(
        model=model,
        ema=ExponentialMovingAverage.from_model(model),
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=FakeScaler(),
        epoch=epoch,
        global_step=epoch * 938,
        base_sampler_generation=epoch,
        canonical_training_lock_sha256=LOCK_SHA,
        initial_checkpoint_sha256=INITIAL_SHA,
        base_manifest_sha256=BASE_SHA,
        training_seed=SEED,
        source_tree_digest=SOURCE_SHA,
        runtime_config_digest=RUNTIME_SHA,
        asset_registry_digest=ASSET_SHA,
    )


def _build_completed_epoch(
    root: Path,
    *,
    epoch: int = 121,
    summary_rng_digest: str | None = None,
) -> tuple[Path, str]:
    transaction_root = root / "03_epoch_transactions"
    start_generation = stable_digest(
        {
            "role": "BRANCH_START",
            "parent_checkpoint_sha256": PARENT_SHA,
            "lineage_digest": "4" * 64,
        }
    )
    start_rng = capture_global_rng().digest()
    transaction = EpochTransaction(
        transaction_root,
        RUN_ID,
        epoch,
        1,
        quarantine_root=root / "09_quarantine",
        identity=GenerationIdentity(
            parent_sha256=PARENT_SHA,
            arm_id=ARM_ID,
            training_seed=SEED,
            source_tree_digest=SOURCE_SHA,
            contract_digest=CONTRACT_SHA,
            asset_registry_digest=ASSET_SHA,
            rng_state_digest=start_rng,
            previous_generation_digest=start_generation,
        ),
    ).begin()
    occurrence_relative = (
        f"04_ledgers/occurrence/run_id={RUN_ID}/epoch={epoch:04d}/part-00000.parquet"
    )
    checkpoint_relative = f"05_checkpoints/rolling_epoch_{epoch:04d}.generation_1.pt"
    summary_relative = "EPOCH_EVIDENCE_SUMMARY.json"
    transaction.required_relative_paths = (
        occurrence_relative,
        checkpoint_relative,
        summary_relative,
    )
    write_occurrence_partition(
        [_occurrence_row(epoch)],
        transaction.path_for(occurrence_relative),
    )
    checkpoint_path = transaction.path_for(checkpoint_relative)
    payload = _checkpoint_payload(epoch)
    checkpoint_sha = save_checkpoint_atomic(checkpoint_path, payload)
    checkpoint_rng = payload["rng_state"].digest()
    transaction.write_json(
        summary_relative,
        {
            "schema_version": "stage1.sctsr.epoch_evidence_summary.v1",
            "status": "VALIDATED_READY_FOR_GENERATION_COMMIT",
            "run_id": RUN_ID,
            "epoch": epoch,
            "generation": 1,
            "checkpoint_sha256": checkpoint_sha,
            "history_after_epoch": {
                "counts": {"sample-1": 1},
                "last_epoch": {"sample-1": epoch},
                "cumulative_occurrences": 1,
            },
            "rng_evidence": {
                "recorder_epoch_start_digest": start_rng,
                "runtime_epoch_start_digest": start_rng,
                "runtime_epoch_end_digest": summary_rng_digest or checkpoint_rng,
                "finalize_entry_digest": summary_rng_digest or checkpoint_rng,
            },
        },
    )
    transaction.commit()
    published = (
        transaction_root
        / f"epoch_{epoch:04d}.generation_1.complete"
        / checkpoint_relative
    )
    return published, start_generation


def _prepare(root: Path, *, start_generation: str):
    return prepare_formal_resume_context(
        run_root=root,
        expected_run_id=RUN_ID,
        expected_arm_id=ARM_ID,
        expected_training_seed=SEED,
        expected_source_tree_digest=SOURCE_SHA,
        expected_contract_digest=CONTRACT_SHA,
        expected_asset_registry_digest=ASSET_SHA,
        expected_previous_checkpoint_sha256=PARENT_SHA,
        expected_previous_generation_digest=start_generation,
        epoch_start=121,
        epoch_end=200,
        minimum_free_bytes=1,
    )


def test_formal_resume_restores_last_complete_checkpoint_history_and_quarantines_partial(tmp_path):
    root = tmp_path / "run"
    checkpoint, start_generation = _build_completed_epoch(root)
    checkpoint_sha_before = sha256_file(checkpoint)
    partial = root / "03_epoch_transactions" / "epoch_0122.generation_1.inprogress"
    partial.mkdir()
    (partial / "half.json").write_text("{", encoding="utf-8")

    state = _prepare(root, start_generation=start_generation)

    assert state.last_complete_epoch == 121
    assert state.resume_epoch == 122
    assert state.checkpoint_path == checkpoint.resolve().as_posix()
    assert state.checkpoint_sha256 == checkpoint_sha_before
    assert state.global_step == 121 * 938
    assert state.history.counts == {"sample-1": 1}
    assert state.history.last_epoch == {"sample-1": 121}
    assert state.history.cumulative_occurrences == 1
    assert len(state.quarantined_partial_paths) == 1
    assert not partial.exists()
    assert sha256_file(checkpoint) == checkpoint_sha_before


def test_resume_inspection_is_read_only_until_execution_claim_succeeds(tmp_path):
    root = tmp_path / "run"
    _checkpoint, start_generation = _build_completed_epoch(root)
    partial = root / "03_epoch_transactions" / "epoch_0122.generation_1.inprogress"
    partial.mkdir()
    half = partial / "half.json"
    half.write_text("{", encoding="utf-8")

    preview = inspect_formal_resume_context(
        run_root=root,
        expected_run_id=RUN_ID,
        expected_arm_id=ARM_ID,
        expected_training_seed=SEED,
        expected_source_tree_digest=SOURCE_SHA,
        expected_contract_digest=CONTRACT_SHA,
        expected_asset_registry_digest=ASSET_SHA,
        expected_previous_checkpoint_sha256=PARENT_SHA,
        expected_previous_generation_digest=start_generation,
        epoch_start=121,
        epoch_end=200,
        minimum_free_bytes=1,
    )

    assert preview.resume_epoch == 122
    assert preview.quarantined_partial_paths == ()
    assert partial.is_dir()
    assert half.read_text(encoding="utf-8") == "{"


@pytest.mark.parametrize("script_name", ["run_common_parent.py", "run_branch.py"])
def test_formal_runner_claims_logical_job_before_mutating_resume_state(script_name, repository_root):
    source = (repository_root / "scripts" / "stage1_sctsr_v4" / script_name).read_text(encoding="utf-8")
    preview_call = source.index("resume_preview = inspect_formal_resume_context(")
    claim_call = source.index("execution_claim = claim_formal_execution(")
    mutate_call = source.index("resume_context = prepare_formal_resume_context(")

    assert preview_call < claim_call < mutate_call


@pytest.mark.parametrize(
    ("script_name", "required_after_claim"),
    [
        (
            "run_common_parent.py",
            ("prepare_formal_resume_context(", "build_prepared_trainer(", "run_prepared_common_parent("),
        ),
        (
            "run_branch.py",
            (
                "prepare_formal_resume_context(",
                "build_prepared_trainer(",
                "run_prepared_branch(",
                'result.pop("_finalization_context")',
                "execute_fenced_finalization(",
            ),
        ),
    ],
)
def test_formal_runner_wraps_every_post_claim_stage_in_terminalizing_phase(
    script_name,
    required_after_claim,
    repository_root,
):
    source = (repository_root / "scripts" / "stage1_sctsr_v4" / script_name).read_text(encoding="utf-8")
    claim_call = source.index("execution_claim = claim_formal_execution(")
    phase_definition = source.index("def execute_claimed_runner_phase():", claim_call)
    phase_call = source.index("return execute_claimed_phase(", phase_definition)

    assert claim_call < phase_definition < phase_call
    for snippet in required_after_claim:
        assert phase_definition < source.index(snippet, phase_definition) < phase_call


def test_formal_resume_rejects_checkpoint_rng_not_bound_by_epoch_summary(tmp_path):
    root = tmp_path / "run"
    _, start_generation = _build_completed_epoch(root, summary_rng_digest="9" * 64)
    with pytest.raises(SctsrError) as captured:
        _prepare(root, start_generation=start_generation)
    assert captured.value.code is ErrorCode.RESUME_GENERATION_MISMATCH


def test_formal_resume_rejects_static_asset_or_generation_chain_mismatch(tmp_path):
    root = tmp_path / "run"
    _build_completed_epoch(root)
    with pytest.raises(SctsrError) as captured:
        prepare_formal_resume_context(
            run_root=root,
            expected_run_id=RUN_ID,
            expected_arm_id=ARM_ID,
            expected_training_seed=SEED,
            expected_source_tree_digest=SOURCE_SHA,
            expected_contract_digest=CONTRACT_SHA,
            expected_asset_registry_digest="8" * 64,
            expected_previous_checkpoint_sha256=PARENT_SHA,
            expected_previous_generation_digest="7" * 64,
            epoch_start=121,
            epoch_end=200,
            minimum_free_bytes=1,
        )
    assert captured.value.code is ErrorCode.RESUME_GENERATION_MISMATCH


def test_formal_resume_rejects_already_complete_run(tmp_path):
    root = tmp_path / "run"
    checkpoint, start_generation = _build_completed_epoch(root, epoch=200)
    assert checkpoint.is_file()
    with pytest.raises(SctsrError) as captured:
        _prepare(root, start_generation=start_generation)
    assert captured.value.code is ErrorCode.RESUME_GENERATION_MISMATCH


def test_terminal_epoch_can_be_inspected_only_for_finalization_recovery(tmp_path):
    root = tmp_path / "run"
    _checkpoint, start_generation = _build_completed_epoch(root, epoch=200)
    kwargs = {
        "run_root": root,
        "expected_run_id": RUN_ID,
        "expected_arm_id": ARM_ID,
        "expected_training_seed": SEED,
        "expected_source_tree_digest": SOURCE_SHA,
        "expected_contract_digest": CONTRACT_SHA,
        "expected_asset_registry_digest": ASSET_SHA,
        "expected_previous_checkpoint_sha256": PARENT_SHA,
        "expected_previous_generation_digest": start_generation,
        "epoch_start": 200,
        "epoch_end": 200,
        "minimum_free_bytes": 1,
    }

    with pytest.raises(SctsrError):
        inspect_formal_resume_context(**kwargs)

    context = inspect_formal_resume_context(
        **kwargs,
        allow_terminal_epoch_for_finalization=True,
    )
    assert context.last_complete_epoch == 200
    assert context.resume_epoch == 201
    assert context.terminal_epoch_complete is True


def test_formal_branch_training_never_writes_complete_status_before_endpoint(repository_root):
    training_source = (repository_root / "stage1_sctsr_v4" / "formal_training.py").read_text(encoding="utf-8")
    runner_source = (repository_root / "scripts" / "stage1_sctsr_v4" / "run_branch.py").read_text(encoding="utf-8")

    assert '"FORMAL_BRANCH_COMPLETE"' not in training_source
    assert runner_source.index("endpoint = publish_formal_endpoint(") < runner_source.index(
        "completion = publish_formal_completion("
    )


def test_branch_runner_continues_at_resume_epoch_without_overwriting_completed_generation(monkeypatch, tmp_path):
    import stage1_sctsr_v4.formal_training as formal

    root = tmp_path / "run"
    completed_checkpoint, start_generation = _build_completed_epoch(root)
    completed_sha = sha256_file(completed_checkpoint)
    context = _prepare(root, start_generation=start_generation)
    identity = FormalIdentity(
        training_seed=SEED,
        canonical_training_lock_sha256=LOCK_SHA,
        initial_checkpoint_sha256=INITIAL_SHA,
        base_manifest_sha256=BASE_SHA,
        source_tree_digest=SOURCE_SHA,
        runtime_config_digest=RUNTIME_SHA,
        asset_registry_digest=ASSET_SHA,
        contract_digest=CONTRACT_SHA,
        seed_registry_digest="5" * 64,
    )
    parent_root = tmp_path / "parent"
    parent_root.mkdir()
    parent_checkpoint = parent_root / "epoch_0120.pt"
    parent_sha = save_checkpoint_atomic(parent_checkpoint, _checkpoint_payload(120))
    atomic_write_json(
        parent_root / "PARENT_RECEIPT.json",
        {
            "status": "FORMAL_PARENT_COMPLETE",
            "parent_id": f"PARENT_{SEED}",
            "training_seed": SEED,
            "checkpoint_sha256": parent_sha,
            "epoch_end": 120,
            "best_pt_used": False,
            "epoch_evidence_enabled": True,
        },
    )
    records = tuple(
        IdentityRecord(f"id-{index}", index % 2, "T_STRESS", "hard", index % 2, f"bucket-{index}")
        for index in range(5)
    )
    groups = {f"G{index}": (records[index],) for index in range(5)}
    schedule = build_schedule(
        ArmId.T_U,
        primary_groups=groups,
        primary_digest=identity_digest(records),
        base_denominator=200,
    )
    lineage = BranchLineage.create(
        logical_run_id=RUN_ID,
        parent_id=f"PARENT_{SEED}",
        parent_checkpoint_path=parent_checkpoint.as_posix(),
        parent_checkpoint_sha256=parent_sha,
        training_seed=SEED,
        arm_id=ARM_ID,
        child_source_tree_digest=SOURCE_SHA,
        child_contract_digest=CONTRACT_SHA,
    )
    release_bindings = {"contract_digest": CONTRACT_SHA}
    original_binding = {
        "schema_version": "stage1.sctsr.prepared_trainer_binding.v1",
        "upstream_binding_digest": "6" * 64,
        "canonical_training_lock_sha256": LOCK_SHA,
        "initial_checkpoint_sha256": INITIAL_SHA,
        "scientific_overrides_digest": "7" * 64,
        "identity_manifest_binding": {"binding_digest": "8" * 64},
        "dataset_binding": {"binding_digest": "9" * 64},
        "training_seed": SEED,
        "output_root": root.as_posix(),
    }
    original_binding["binding_digest"] = stable_digest(original_binding)
    resume_binding = {
        **original_binding,
        "output_root": (root / "10_resume_setup" / "epoch_0122.generation_1").as_posix(),
    }
    resume_binding.pop("binding_digest")
    resume_binding["binding_digest"] = stable_digest(resume_binding)
    pool_binding = {"binding_digest": "A" * 64}
    parent_binding = {"binding_digest": "B" * 64}
    atomic_write_json(root / "FORMAL_IDENTITY.json", asdict(identity))
    atomic_write_json(root / "FORMAL_AUTHORIZATION_BINDING.json", release_bindings)
    atomic_write_json(root / "PREPARED_TRAINER_BINDING.json", original_binding)
    atomic_write_json(root / "BRANCH_LINEAGE.json", asdict(lineage))
    atomic_write_json(root / "SCHEDULE.json", schedule_to_dict(schedule))
    atomic_write_json(root / "IDENTITY_POOL_BINDING.json", pool_binding)
    atomic_write_json(root / "PARENT_ARTIFACT_INDEX_BINDING.json", parent_binding)

    observed_epochs: list[int] = []

    def fake_transactional_epoch(*, epoch, global_step_start, replay_plan, root, **_kwargs):
        observed_epochs.append(epoch)
        checkpoint = root / f"fake_resume_epoch_{epoch:04d}.pt"
        checkpoint.write_bytes(f"checkpoint-{epoch}".encode("ascii"))
        checkpoint_sha = sha256_file(checkpoint)
        return (
            {
                "optimizer_steps": 1,
                "replay_occurrences": replay_plan.planned_replay_occurrences,
                "global_step_end": global_step_start + 1,
                "base_occurrences": 128,
                "ema_updates_delta": 1,
                "base_order_digest": "C" * 64,
                "base_augmentation_digest": "D" * 64,
            },
            {
                "checkpoint_path": checkpoint.as_posix(),
                "checkpoint_sha256": checkpoint_sha,
                "generation_digest": stable_digest({"epoch": epoch}),
            },
        )

    monkeypatch.setattr(formal, "require_synthetic_or_authorized", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(formal, "_base_batch_sizes", lambda _trainer: (128,))
    monkeypatch.setattr(formal, "_revalidate_prepared_dataset_bindings", lambda _binding: None)
    monkeypatch.setattr(formal, "CANONICAL_BASE_STEPS", 1)
    monkeypatch.setattr(formal, "sample_evidence_from_trainer", lambda _trainer: {})
    monkeypatch.setattr(formal, "prepare_counter_domain_base_loader", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(formal, "_run_transactional_epoch", fake_transactional_epoch)
    monkeypatch.setattr(formal, "_publish_complete_logical_timeline", lambda **_kwargs: {"status": "UNIT_ONLY"})
    monkeypatch.setattr(formal, "publish_formal_run_manifest_and_indexes", lambda **_kwargs: {"status": "UNIT_ONLY"})
    monkeypatch.setattr(formal, "validate_execution_claim_binding", lambda *_args, **_kwargs: {"status": "UNIT_ONLY"})
    monkeypatch.setattr(formal, "validate_run_intent_binding", lambda *_args, **_kwargs: {"status": "UNIT_ONLY"})
    monkeypatch.setattr(
        formal,
        "publish_run_intent_snapshot",
        lambda *_args, **_kwargs: {"snapshot_digest": "9" * 64, "acknowledgement_id": "UNIT_RESUME_ACK"},
    )
    monkeypatch.setattr(
        formal,
        "publish_execution_claim_snapshot",
        lambda *_args, **_kwargs: {"snapshot_digest": "1" * 64},
    )
    monkeypatch.setattr(
        formal,
        "validate_formal_input_snapshot",
        lambda *_args, **_kwargs: {"snapshot_digest": "E" * 64},
    )

    receipt = run_prepared_branch(
        trainer=FakeTrainer(),
        identity=identity,
        parent_checkpoint=parent_checkpoint,
        lineage=lineage,
        schedule=schedule,
        replay_batch_provider=lambda *_args: {},
        output_root=root,
        release_authorization={"unit": True},
        execution_mode="formal",
        release_expected_bindings=release_bindings,
        identity_pool_binding=pool_binding,
        parent_artifact_index_binding=parent_binding,
        prepared_trainer_binding=resume_binding,
        formal_input_binding={"binding_digest": "F" * 64},
        execution_claim_binding={
            "execution_id": "UNIT_RESUME_EXECUTION",
            "claim_sha256": "2" * 64,
            "job_binding_digest": "3" * 64,
        },
        run_intent_binding={"unit": True},
        resume_context=context,
    )

    assert observed_epochs[0] == 122
    assert observed_epochs[-1] == 200
    assert receipt["resumed_from_epoch"] == 121
    assert receipt["resume_binding_receipt_digest"]
    assert sha256_file(completed_checkpoint) == completed_sha


@pytest.mark.parametrize("script_name", ["run_common_parent.py", "run_branch.py"])
def test_formal_runner_cli_exposes_explicit_resume_and_isolated_setup_flags(script_name, repository_root):
    script = repository_root / "scripts" / "stage1_sctsr_v4" / script_name
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--resume" in completed.stdout
    assert "--resume-setup-root" in completed.stdout
