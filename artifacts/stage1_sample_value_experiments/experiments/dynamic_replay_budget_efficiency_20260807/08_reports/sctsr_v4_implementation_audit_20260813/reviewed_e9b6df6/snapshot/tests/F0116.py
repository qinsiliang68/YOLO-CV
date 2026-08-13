import pytest

from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.exposure_ledger import build_exposure_row, write_exposure_partition

SHA = "A" * 64


def build(ids=()):
    return build_exposure_row(
        run_id="run-1",
        parent_id="parent-1",
        arm_id="NR",
        training_seed=1,
        epoch=121,
        base_denominator=2000,
        replay_rate_numerator=0 if not ids else 5,
        replay_rate_denominator=1000,
        replay_sample_ids=ids,
        optimizer_steps=16,
        expected_optimizer_steps=16,
        base_order_digest=SHA,
        base_augmentation_digest=SHA,
        schedule_digest=SHA,
        identity_pool_digest=SHA,
        occurrence_partition_sha=SHA,
        step_partition_sha=SHA,
        telemetry_partition_sha=SHA,
        checkpoint_sha=SHA,
        cumulative_occurrences=len(ids),
        ema_updates_delta=16,
        scheduler_epoch_transitions_delta=1,
        write_seconds=0.1,
        dataloader_wait_seconds=0.1,
        training_seconds=1.0,
        evaluation_seconds=0.0,
        disk_bytes_written=100,
        transaction_generation=1,
    )


def test_zero_replay_exposure():
    assert build()["replay_numerator_actual"] == 0


def test_replay_rate_conserved():
    assert build(tuple(f"s{i}" for i in range(10)))["replay_numerator_actual"] == 10


def test_wrong_optimizer_steps_rejected():
    with pytest.raises(SctsrError) as captured:
        kwargs = build()
        kwargs["base_optimizer_steps_actual"] = 15
        write_exposure_partition([kwargs], "run_id=run-1/epoch=0121/part.parquet")
    assert captured.value.code is ErrorCode.BASE_STEP_COUNT_MISMATCH


def test_exposure_partition_written(tmp_path):
    path = tmp_path / "run_id=run-1" / "epoch=0121" / "part-00000.parquet"
    assert write_exposure_partition([build()], path).row_count == 1
