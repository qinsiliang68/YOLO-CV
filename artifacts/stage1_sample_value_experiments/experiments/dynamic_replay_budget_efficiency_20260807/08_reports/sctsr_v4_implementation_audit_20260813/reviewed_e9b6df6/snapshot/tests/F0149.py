import pytest

from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.selection_ledger import validate_selection_rows, write_selection_partition
from tests.stage1_sctsr_v4.test_evidence_transaction_hardening import selection_row


def test_selection_row_valid():
    validate_selection_rows([selection_row()], r2=True)


def test_r2_terminal_field_rejected():
    value = selection_row()
    value["loss"] = 1.0
    with pytest.raises(SctsrError) as captured:
        validate_selection_rows([value], r2=True)
    assert captured.value.code is ErrorCode.TERMINAL_FIELD_ACCESS_FORBIDDEN


def test_selection_partition_written(tmp_path):
    path = tmp_path / "run_id=pool-build" / "epoch=0000" / "part-00000.parquet"
    assert write_selection_partition([selection_row()], path, r2=True).row_count == 1
