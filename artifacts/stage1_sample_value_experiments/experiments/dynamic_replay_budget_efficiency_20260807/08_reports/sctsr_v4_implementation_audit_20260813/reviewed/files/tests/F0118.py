import pytest, torch
from stage1_sctsr_v4.arm_spec import ArmId
from stage1_sctsr_v4.errors import ErrorCode,SctsrError
from stage1_sctsr_v4.fixed_step_runtime import ExponentialMovingAverage,run_fixed_step_epoch
from stage1_sctsr_v4.replay_step_plan import build_replay_step_plan
from stage1_sctsr_v4.synthetic_fixture import TinyClassifier,make_base_loader,make_replay_provider

def stack(seed=7):
    torch.manual_seed(seed)
    model=TinyClassifier();optimizer=torch.optim.SGD(model.parameters(),lr=.02,momentum=.9)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.0)
    scaler=torch.amp.GradScaler('cpu',enabled=False)
    ema=ExponentialMovingAverage.from_model(model)
    return model,optimizer,scheduler,scaler,ema

def run(fixture,replay_ids,seed=7):
    loader=make_base_loader(fixture,epoch=121)[:2];sizes=[int(x['labels'].shape[0]) for x in loader]
    plan=build_replay_step_plan(run_id='r',arm_id='T_U',training_seed=seed,epoch=121,schedule_family='U',sample_ids=replay_ids,base_batch_sizes=sizes)
    m,o,s,sc,e=stack(seed)
    result=run_fixed_step_epoch(model=m,optimizer=o,base_loader=loader,replay_plan=plan,replay_batch_provider=make_replay_provider(fixture),training_seed=seed,epoch=121,ema=e,scheduler=s,scaler=sc,clip_max_norm=10.0)
    return m,result

def test_replay_does_not_add_optimizer_steps(synthetic_fixture):
    _,result=run(synthetic_fixture,[x.sample_id for x in synthetic_fixture.t_pool.records[:2]])
    assert result.optimizer_steps==2;assert result.replay_occurrences==2;assert result.ema_updates_delta==2

def test_replay_changes_parameter_trajectory(synthetic_fixture):
    a,ra=run(synthetic_fixture,[]);b,rb=run(synthetic_fixture,[x.sample_id for x in synthetic_fixture.t_pool.records[:2]])
    assert ra.base_order_digest==rb.base_order_digest;assert any(not torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))

def test_world_size_gt_one_rejected(synthetic_fixture):
    loader=make_base_loader(synthetic_fixture,epoch=121)[:1];plan=build_replay_step_plan(run_id='r',arm_id='NR',training_seed=1,epoch=121,schedule_family='NR',sample_ids=[],base_batch_sizes=[len(loader[0]['labels'])]);m,o,s,sc,e=stack(1)
    with pytest.raises(SctsrError) as ex:run_fixed_step_epoch(model=m,optimizer=o,base_loader=loader,replay_plan=plan,replay_batch_provider=make_replay_provider(synthetic_fixture),training_seed=1,epoch=121,ema=e,scheduler=s,scaler=sc,world_size=2)
    assert ex.value.code is ErrorCode.DISTRIBUTED_MODE_NOT_SUPPORTED_IN_V4_PHASE1

def test_gradient_accumulation_rejected(synthetic_fixture):
    loader=make_base_loader(synthetic_fixture,epoch=121)[:1];plan=build_replay_step_plan(run_id='r',arm_id='NR',training_seed=1,epoch=121,schedule_family='NR',sample_ids=[],base_batch_sizes=[len(loader[0]['labels'])]);m,o,s,sc,e=stack(1)
    with pytest.raises(SctsrError):run_fixed_step_epoch(model=m,optimizer=o,base_loader=loader,replay_plan=plan,replay_batch_provider=make_replay_provider(synthetic_fixture),training_seed=1,epoch=121,ema=e,scheduler=s,scaler=sc,gradient_accumulation=2)

def test_oom_is_fail_closed(synthetic_fixture):
    loader=make_base_loader(synthetic_fixture,epoch=121)[:1];ids=[synthetic_fixture.t_pool.records[0].sample_id];plan=build_replay_step_plan(run_id='r',arm_id='T_U',training_seed=1,epoch=121,schedule_family='U',sample_ids=ids,base_batch_sizes=[len(loader[0]['labels'])]);m,o,s,sc,e=stack(1)
    def oom(*args,**kwargs):raise RuntimeError('CUDA out of memory')
    with pytest.raises(SctsrError) as ex:run_fixed_step_epoch(model=m,optimizer=o,base_loader=loader,replay_plan=plan,replay_batch_provider=oom,training_seed=1,epoch=121,ema=e,scheduler=s,scaler=sc)
    assert ex.value.code is ErrorCode.OOM_FIXED_CONTRACT_ABORT

def test_replay_ce_is_sum_divided_by_canonical_128():
    torch.manual_seed(5)
    model=torch.nn.Linear(2,2,bias=False);expected=torch.nn.Linear(2,2,bias=False);expected.load_state_dict(model.state_dict())
    base_images=torch.tensor([[1.0,0.0],[0.0,1.0],[1.0,1.0],[-1.0,0.5]]);base_labels=torch.tensor([0,1,0,1])
    replay_images=torch.tensor([[0.25,-0.75]]);replay_labels=torch.tensor([1])
    loader=[{'images':base_images,'labels':base_labels,'sample_ids':('b0','b1','b2','b3'),'augmentation_digests':('a0','a1','a2','a3')}]
    plan=build_replay_step_plan(run_id='r',arm_id='T_U',training_seed=1,epoch=121,schedule_family='U',sample_ids=('r0',),base_batch_sizes=(4,))
    optimizer=torch.optim.SGD(model.parameters(),lr=.1);expected_optimizer=torch.optim.SGD(expected.parameters(),lr=.1)
    def provider(ids,e,s,seed):return {'images':replay_images,'labels':replay_labels,'sample_ids':tuple(ids),'augmentation_digests':('r0aug',)}
    run_fixed_step_epoch(model=model,optimizer=optimizer,base_loader=loader,replay_plan=plan,replay_batch_provider=provider,training_seed=1,epoch=121)
    expected_optimizer.zero_grad();expected_loss=torch.nn.functional.cross_entropy(expected(base_images),base_labels)+torch.nn.functional.cross_entropy(expected(replay_images),replay_labels,reduction='sum')/128;expected_loss.backward();expected_optimizer.step()
    assert all(torch.allclose(a,b,atol=1e-7,rtol=0) for a,b in zip(model.parameters(),expected.parameters()))

def test_generic_runtime_amp_skip_fails_closed():
    class SkipScaler:
        def __init__(self):self.scale_value=2.0
        def scale(self,loss):return loss
        def unscale_(self,optimizer):pass
        def step(self,optimizer):pass
        def update(self):self.scale_value=1.0
        def get_scale(self):return self.scale_value
    model=torch.nn.Linear(2,2);optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    loader=[{'images':torch.ones(4,2),'labels':torch.tensor([0,1,0,1]),'sample_ids':('a','b','c','d'),'augmentation_digests':('1','2','3','4')}]
    plan=build_replay_step_plan(run_id='r',arm_id='NR',training_seed=1,epoch=121,schedule_family='NR',sample_ids=(),base_batch_sizes=(4,))
    with pytest.raises(SctsrError) as exc:run_fixed_step_epoch(model=model,optimizer=optimizer,base_loader=loader,replay_plan=plan,replay_batch_provider=lambda *_:None,training_seed=1,epoch=121,scaler=SkipScaler())
    assert exc.value.code is ErrorCode.OPTIMIZER_STEP_SKIPPED
