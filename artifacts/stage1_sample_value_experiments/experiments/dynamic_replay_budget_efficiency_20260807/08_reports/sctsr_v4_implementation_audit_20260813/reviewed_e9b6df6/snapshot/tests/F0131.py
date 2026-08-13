from dataclasses import replace
import pytest
from stage1_sctsr_v4.errors import ErrorCode,SctsrError
from stage1_sctsr_v4.prediction_artifact import sample_label_identity_digest,validate_prediction_rows,write_prediction_artifact

def test_prediction_identity_digest_stable(prediction_rows):assert sample_label_identity_digest(prediction_rows)==sample_label_identity_digest(tuple(reversed(prediction_rows)))

def test_duplicate_prediction_rejected(prediction_rows):
    with pytest.raises(SctsrError) as e:validate_prediction_rows(prediction_rows+(prediction_rows[0],))
    assert e.value.code is ErrorCode.PREDICTION_IDENTITY_MISMATCH

def test_wrong_split_rejected(prediction_rows):
    with pytest.raises(SctsrError):validate_prediction_rows(prediction_rows,expected_split_role='val_op')

def test_formal_predictions_require_e200(tmp_path,prediction_rows):
    bad=tuple(replace(x,checkpoint_epoch=180) for x in prediction_rows)
    with pytest.raises(SctsrError):write_prediction_artifact(bad,tmp_path/'p.parquet',formal_endpoint=True)

def test_prediction_partition_written(tmp_path,prediction_rows):
    path=tmp_path/'run_id=R'/'epoch=0200'/'p.parquet'
    m,s=write_prediction_artifact(prediction_rows,path,formal_endpoint=False);assert m.row_count==len(prediction_rows);assert s['checkpoint_epoch']==200
