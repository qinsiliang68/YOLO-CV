from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import torch

from stage1_sctsr_v4.epoch_transaction import EpochTransaction
from stage1_sctsr_v4.evidence_runtime import EpochEvidenceRecorder, ReplayHistoryState, SampleEvidence
from stage1_sctsr_v4.fixed_step_runtime import OccurrenceEvent
from stage1_sctsr_v4.serialization import atomic_write_bytes, sha256_file
from stage1_sctsr_v4.ultralytics_overlay import UpstreamStepReceipt


SHA = "A" * 64


def _sample(sample_id: str) -> SampleEvidence:
    return SampleEvidence(
        sample_id=sample_id,
        y_true=0,
        replay_role="T_STRESS",
        oof_fold=1,
        oof_group_id="bucket-1",
        historical_dynamic_bucket="stable_high",
        identity_group="G0",
        oof_reference_probability=None,
        oof_reference_reason="REGISTERED_NOT_AVAILABLE",
        rho_candidate_signal=None,
        rho_reason="REGISTERED_NOT_REPORTED",
    )


def _step_receipt() -> UpstreamStepReceipt:
    return UpstreamStepReceipt(
        epoch=121,
        base_step_index=0,
        global_step_before=112_560,
        global_step_after=112_561,
        base_batch_size=4,
        replay_microbatch_size=1,
        base_loss=0.5,
        replay_loss=0.1,
        combined_loss_for_reporting=0.6,
        base_loss_items={"classification_loss": 0.5},
        parameter_grad_norm_before_clip=2.0,
        parameter_grad_norm_after_clip=2.0,
        clip_max_norm=10.0,
        optimizer_step_delta=1,
        learning_rates=(0.001,),
        optimizer_hyperparameters=(
            {
                "group_index": 0,
                "lr": 0.001,
                "initial_lr": 0.001,
                "momentum": 0.9,
                "beta1": None,
                "beta2": None,
                "weight_decay": 0.0005,
                "dampening": 0.0,
                "nesterov": True,
            },
        ),
        amp_scale_before=65_536.0,
        amp_scale_after=65_536.0,
        overflow_or_step_skipped=False,
        ema_updates_before=112_560,
        ema_updates_after=112_561,
        scheduler_state_digest=SHA,
        warmup_progress=1.0,
        ema_update_delta=1,
        rng_before_replay=SHA,
        rng_after_replay=SHA,
        bn_before_replay=SHA,
        bn_after_replay=SHA,
        rng_before_base=SHA,
        replay_rng_fork_digest=SHA,
        base_augmentation_digest=SHA,
        replay_augmentation_digest=SHA,
        replay_rate_numerator=1,
        replay_rate_denominator=4,
        dataloader_wait_seconds=0.25,
        base_forward_seconds=0.01,
        replay_forward_seconds=0.005,
        backward_seconds=0.02,
        optimizer_seconds=0.003,
        write_buffer_bytes=0,
        row_generation=1,
    )


def test_real_epoch_evidence_is_streamed_bound_and_atomically_published(tmp_path: Path):
    transaction_root = tmp_path / "run" / "03_epoch_transactions"
    transaction = EpochTransaction(transaction_root, "run-1", 121, 1).begin()
    samples = {f"sample-{index}": _sample(f"sample-{index}") for index in range(4)}
    recorder = EpochEvidenceRecorder(
        transaction=transaction,
        parent_id="parent-1",
        arm_id="T_U",
        training_seed=17,
        sample_evidence=samples,
        identity_policy="T_STRESS",
        schedule_family="U",
        fallback_state="TARGETED",
        rate_numerator=1,
        rate_denominator=4,
        schedule_digest=SHA,
        identity_pool_digest=SHA,
        pool_multiplicity_targets={"sample-0": 1},
        expected_base_denominator=4,
        expected_optimizer_steps=1,
        global_step_start=112_560,
        history=ReplayHistoryState(),
        artifact_root=tmp_path,
    )
    base_ids = tuple(samples)
    recorder.occurrence_sink(
        OccurrenceEvent(
            occurrence_role="BASE",
            base_step_index=0,
            sample_ids=base_ids,
            labels=torch.zeros(4, dtype=torch.long),
            logits=torch.tensor([[1.0, 0.0]] * 4),
            per_sample_ce=torch.tensor([0.3132617] * 4),
            augmentation_digests=(SHA,) * 4,
            augmentation_seeds=(1, 2, 3, 4),
        )
    )
    recorder.occurrence_sink(
        OccurrenceEvent(
            occurrence_role="REPLAY",
            base_step_index=0,
            sample_ids=("sample-0",),
            labels=torch.zeros(1, dtype=torch.long),
            logits=torch.tensor([[1.0, 0.0]]),
            per_sample_ce=torch.tensor([0.3132617]),
            augmentation_digests=(SHA,),
            augmentation_seeds=(5,),
        )
    )
    recorder.step_sink(_step_receipt())
    atomic_write_bytes(recorder.checkpoint_path, b"immutable epoch checkpoint")
    checkpoint_sha = sha256_file(recorder.checkpoint_path)
    summary = recorder.finalize(
        runtime_result={
            "base_occurrences": 4,
            "replay_occurrences": 1,
            "optimizer_steps": 1,
            "ema_updates_delta": 1,
            "scheduler_epoch_transitions_delta": 1,
            "base_order_digest": SHA,
            "base_augmentation_digest": SHA,
        },
        checkpoint_sha256=checkpoint_sha,
    )
    manifest = transaction.commit()

    assert summary["status"] == "VALIDATED_READY_FOR_GENERATION_COMMIT"
    assert manifest["status"] == "VALIDATED_READY_TO_PUBLISH"
    assert transaction.complete.is_dir()
    assert not transaction.inprogress.exists()
    assert transaction.receipt_path.is_file()
    assert transaction.recovery_pointer_path.is_file()
    assert transaction.artifact_index_path.is_file()
    occurrence = transaction.complete / summary["occurrence_partition"]["path"]
    step = transaction.complete / summary["step_partition"]["path"]
    exposure = transaction.complete / summary["exposure_partition"]["path"]
    telemetry = transaction.complete / summary["telemetry_partition"]["path"]
    assert pq.ParquetFile(occurrence).metadata.num_rows == 5
    assert pq.ParquetFile(step).metadata.num_rows == 1
    assert pq.ParquetFile(exposure).metadata.num_rows == 1
    assert pq.ParquetFile(telemetry).metadata.num_rows >= 1


def test_epoch_evidence_persists_real_dataloader_wait_sum(tmp_path: Path):
    transaction = EpochTransaction(tmp_path / "run" / "03_epoch_transactions", "run-1", 121, 1).begin()
    samples = {f"sample-{index}": _sample(f"sample-{index}") for index in range(4)}
    recorder = EpochEvidenceRecorder(
        transaction=transaction,
        parent_id="parent-1",
        arm_id="T_U",
        training_seed=17,
        sample_evidence=samples,
        identity_policy="T_STRESS",
        schedule_family="U",
        fallback_state="TARGETED",
        rate_numerator=1,
        rate_denominator=4,
        schedule_digest=SHA,
        identity_pool_digest=SHA,
        pool_multiplicity_targets={"sample-0": 1},
        expected_base_denominator=4,
        expected_optimizer_steps=1,
        global_step_start=112_560,
        history=ReplayHistoryState(),
        artifact_root=tmp_path,
    )
    recorder.occurrence_sink(
        OccurrenceEvent(
            occurrence_role="BASE",
            base_step_index=0,
            sample_ids=tuple(samples),
            labels=torch.zeros(4, dtype=torch.long),
            logits=torch.tensor([[1.0, 0.0]] * 4),
            per_sample_ce=torch.tensor([0.3132617] * 4),
            augmentation_digests=(SHA,) * 4,
            augmentation_seeds=(1, 2, 3, 4),
        )
    )
    recorder.occurrence_sink(
        OccurrenceEvent(
            occurrence_role="REPLAY",
            base_step_index=0,
            sample_ids=("sample-0",),
            labels=torch.zeros(1, dtype=torch.long),
            logits=torch.tensor([[1.0, 0.0]]),
            per_sample_ce=torch.tensor([0.3132617]),
            augmentation_digests=(SHA,),
            augmentation_seeds=(5,),
        )
    )
    recorder.step_sink(_step_receipt())
    atomic_write_bytes(recorder.checkpoint_path, b"checkpoint")
    checkpoint_sha = sha256_file(recorder.checkpoint_path)
    summary = recorder.finalize(
        runtime_result={
            "base_occurrences": 4,
            "replay_occurrences": 1,
            "optimizer_steps": 1,
            "ema_updates_delta": 1,
            "scheduler_epoch_transitions_delta": 1,
            "base_order_digest": SHA,
            "base_augmentation_digest": SHA,
        },
        checkpoint_sha256=checkpoint_sha,
    )
    table = pq.ParquetFile(transaction.inprogress / summary["exposure_partition"]["path"]).read().to_pylist()
    assert table[0]["dataloader_wait_seconds"] == 0.25
    recorder.abort()
