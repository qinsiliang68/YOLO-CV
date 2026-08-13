from collections import Counter
import pytest
from stage1_sctsr_v4.errors import ErrorCode,SctsrError
from stage1_sctsr_v4.random_controls import build_r2_matched_random
from stage1_sctsr_v4.terminal_field_guard import TerminalFieldGuard

def test_r1_uses_global_universe_and_reports_overlap(synthetic_fixture):
    a=synthetic_fixture.r1_result.audit;assert a.candidate_count==synthetic_fixture.base_denominator;assert a.selected_count==50;assert a.overlap_with_t_count>=0

def test_r2_zero_overlap_exact_quota(synthetic_fixture):
    t={r.sample_id for r in synthetic_fixture.t_pool.records};r2={r.sample_id for r in synthetic_fixture.r2_result.pool.records};assert not t&r2;assert Counter(x.stratum() for x in synthetic_fixture.t_pool.records)==Counter(x.stratum() for x in synthetic_fixture.r2_result.pool.records)

def test_r2_infeasible_fails(synthetic_fixture):
    rows=[r.__dict__ if hasattr(r,'__dict__') else {'sample_id':r.sample_id,'y_true':r.y_true,'replay_role':r.replay_role,'historical_dynamic_bucket':r.historical_dynamic_bucket,'oof_fold':r.oof_fold,'oof_group_id':r.oof_group_id,'group_source':r.group_source,'base_manifest_membership':r.base_manifest_membership} for r in synthetic_fixture.t_pool.records]
    with pytest.raises(SctsrError) as e:build_r2_matched_random(rows,t_pool=synthetic_fixture.t_pool,base_denominator=synthetic_fixture.base_denominator,base_manifest_sha256='A'*64,source_manifest_sha256='B'*64,selection_seed=1,guard=TerminalFieldGuard())
    assert e.value.code is ErrorCode.R2_QUOTA_INFEASIBLE

def test_terminal_field_guard_does_not_enumerate_forbidden():
    class Sentinel(dict):
        def items(self):raise AssertionError('must not enumerate')
        def __iter__(self):raise AssertionError('must not iterate')
    row=Sentinel(sample_id='x',y_true=0,replay_role='N',historical_dynamic_bucket='E',oof_fold=0,oof_group_id='g',loss=999)
    out=TerminalFieldGuard().project_row(row);assert 'loss' not in out and out['sample_id']=='x'
