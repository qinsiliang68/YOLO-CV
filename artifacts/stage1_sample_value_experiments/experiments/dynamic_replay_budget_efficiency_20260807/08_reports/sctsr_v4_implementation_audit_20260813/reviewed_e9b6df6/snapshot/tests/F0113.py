import math
import pytest
from stage1_sctsr_v4.errors import ErrorCode,SctsrError
from stage1_sctsr_v4.evaluation import compute_tie_safe_frontier,validate_checkpoint_for_evaluation

def test_frontier_has_96_points(prediction_rows):
    points,summary=compute_tie_safe_frontier(prediction_rows,max_fn=95,target_tn=20,checkpoint_sha256='B'*64,prediction_artifact_sha256='D'*64);assert len(points)==96;assert [x.fn_budget for x in points]==list(range(96));assert 0<=summary.raw_frontier_normalized_auc<=1

def test_ties_not_split(prediction_rows):
    points,_=compute_tie_safe_frontier(prediction_rows,max_fn=10,target_tn=20,checkpoint_sha256='B'*64,prediction_artifact_sha256='D'*64);prob_counts={p:sum(x.p_defect_raw==p for x in prediction_rows) for p in {x.p_defect_raw for x in prediction_rows}};assert all(point.tie_size in {0,*prob_counts.values()} for point in points)

def test_unreachable_target_returns_null(prediction_rows):
    _,s=compute_tie_safe_frontier(prediction_rows,target_tn=999999,checkpoint_sha256='B'*64,prediction_artifact_sha256='D'*64);assert not s.target_tn_reachable and s.fn_at_tn68253 is None

def test_best_pt_rejected(tmp_path):
    with pytest.raises(SctsrError) as e:validate_checkpoint_for_evaluation(tmp_path/'best.pt',epoch=200,mode='formal')
    assert e.value.code is ErrorCode.BEST_PT_FORBIDDEN

def test_formal_non_e200_rejected(tmp_path):
    with pytest.raises(SctsrError):validate_checkpoint_for_evaluation(tmp_path/'e180.pt',epoch=180,mode='formal')

def test_max_fn_zero_supported(prediction_rows):
    p,s=compute_tie_safe_frontier(prediction_rows,max_fn=0,target_tn=1,checkpoint_sha256='B'*64,prediction_artifact_sha256='D'*64);assert len(p)==1 and s.raw_frontier_normalized_auc==p[0].normalized_tn
