import pytest

from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.occurrence_ledger import validate_occurrence_rows, write_occurrence_partition
from tests.stage1_sctsr_v4.test_evidence_transaction_hardening import occurrence_row


def test_occurrence_row_valid():
    validate_occurrence_rows([occurrence_row()])


def test_occurrence_missing_field_rejected():
    value = occurrence_row()
    del value["sample_id"]
    with pytest.raises(SctsrError):
        validate_occurrence_rows([value])


def test_candidate_signal_must_be_nonutility():
    value = occurrence_row()
    value["rho_candidate_signal"] = 1.0
    value["rho_reason"] = "UTILITY"
    with pytest.raises(SctsrError) as captured:
        validate_occurrence_rows([value])
    assert captured.value.code is ErrorCode.CANDIDATE_SIGNAL_AS_UTILITY_FORBIDDEN


def test_occurrence_partition_written(tmp_path):
    path = tmp_path / "run_id=run-1" / "epoch=0121" / "part-00000.parquet"
    assert write_occurrence_partition([occurrence_row()], path).row_count == 1
