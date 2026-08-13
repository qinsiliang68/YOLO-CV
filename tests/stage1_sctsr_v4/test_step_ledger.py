import pytest

from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.step_ledger import validate_step_rows, write_step_partition
from tests.stage1_sctsr_v4.test_evidence_transaction_hardening import step_row


def test_step_row_valid():
    validate_step_rows([step_row()])


def test_extra_optimizer_step_rejected():
    value = step_row()
    value["optimizer_step_count_delta"] = 2
    with pytest.raises(SctsrError) as captured:
        validate_step_rows([value])
    assert captured.value.code is ErrorCode.REPLAY_ADDED_OPTIMIZER_STEP


def test_step_partition_written(tmp_path):
    path = tmp_path / "run_id=run-1" / "epoch=0121" / "part-00000.parquet"
    assert write_step_partition([step_row()], path).row_count == 1
