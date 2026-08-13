import pytest
from stage1_sctsr_v4.errors import ErrorCode,SctsrError
from stage1_sctsr_v4.replay_step_plan import build_replay_step_plan

def test_plan_conserves_occurrences():
    p=build_replay_step_plan(run_id='r',arm_id='T_U',training_seed=1,epoch=121,schedule_family='U',sample_ids=[f's{i}' for i in range(10)],base_batch_sizes=[8]*8)
    assert sum(p.per_step_replay_counts)==10;assert sum(map(len,p.per_step_identity_slices))==10

def test_same_schedule_same_step_skeleton_independent_of_identity():
    a=build_replay_step_plan(run_id='a',arm_id='T_U',training_seed=4,epoch=121,schedule_family='U',sample_ids=[f'a{i}' for i in range(10)],base_batch_sizes=[16]*8)
    b=build_replay_step_plan(run_id='b',arm_id='R2_U',training_seed=4,epoch=121,schedule_family='U',sample_ids=[f'b{i}' for i in range(10)],base_batch_sizes=[16]*8)
    assert a.per_step_replay_counts==b.per_step_replay_counts

def test_tail_batch_cap_checked():
    with pytest.raises(SctsrError) as e:build_replay_step_plan(run_id='r',arm_id='T',training_seed=1,epoch=1,schedule_family='U',sample_ids=['a','b'],base_batch_sizes=[4])
    assert e.value.code is ErrorCode.REPLAY_MICROBATCH_CAP_EXCEEDED

def test_no_replay_plan_valid():
    p=build_replay_step_plan(run_id='r',arm_id='NR',training_seed=1,epoch=1,schedule_family='NR',sample_ids=[],base_batch_sizes=[128,2]);assert p.planned_replay_occurrences==0
