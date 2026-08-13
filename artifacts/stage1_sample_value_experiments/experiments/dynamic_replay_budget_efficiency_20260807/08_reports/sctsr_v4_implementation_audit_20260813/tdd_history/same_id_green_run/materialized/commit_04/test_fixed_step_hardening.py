from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from stage1_sctsr_v4.errors import SctsrError
from stage1_sctsr_v4.fixed_step_runtime import ExponentialMovingAverage
from stage1_sctsr_v4.replay_step_plan import build_replay_step_plan
from stage1_sctsr_v4.rng_isolation import derive_counter_seed
from stage1_sctsr_v4.ultralytics_overlay import run_ultralytics_classification_epoch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.bn=torch.nn.BatchNorm1d(4); self.fc=torch.nn.Linear(4,2)
    def forward(self,value):
        if isinstance(value,dict):
            logits=self.fc(self.bn(value['img'])); loss=F.cross_entropy(logits,value['cls'].view(-1).long()); return loss,loss.detach()
        return self.fc(self.bn(value))


class EMA:
    def __init__(self,model):self.inner=ExponentialMovingAverage.from_model(model);self.updates=0
    def update(self,model):self.inner.update(model);self.updates+=1


class SkippingScaler:
    def __init__(self):self.current=2.0
    def scale(self,loss):return loss
    def unscale_(self,optimizer):pass
    def step(self,optimizer):pass
    def update(self):self.current=1.0
    def get_scale(self):return self.current


class Trainer:
    def __init__(self,include_identity=True,scaler=None):
        self.model=Model();self.optimizer=torch.optim.SGD(self.model.parameters(),lr=.01);self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda _:1.0);self.scaler=scaler or torch.amp.GradScaler('cpu',enabled=False);self.ema=EMA(self.model);self.device=torch.device('cpu');self.amp=False;self.batch_size=128;self.world_size=1;self.accumulate=1;self.args=SimpleNamespace(resume='',warmup_epochs=0,nbs=64,warmup_bias_lr=.1,warmup_momentum=.8,momentum=.937,compile=False);self.optimizer_step_calls=0
        batch={'img':torch.randn(128,4),'cls':torch.arange(128)%2,'augmentation_digests':['A'*64]*128}
        if include_identity:batch['sample_ids']=[f'B{i}' for i in range(128)]
        self.train_loader=[batch]
    def preprocess_batch(self,batch):return batch
    def optimizer_step(self):self.optimizer_step_calls+=1;self.scaler.unscale_(self.optimizer);self.scaler.step(self.optimizer);self.scaler.update();self.optimizer.zero_grad();self.ema.update(self.model)


def plan(ids=()):return build_replay_step_plan(run_id='R',arm_id='T_U' if ids else 'NR',training_seed=19,epoch=121,schedule_family='U' if ids else 'NR',sample_ids=ids,base_batch_sizes=[128])
def provider(ids,e,s,seed):return {'img':torch.randn(len(ids),4),'cls':torch.zeros(len(ids),dtype=torch.long),'sample_ids':tuple(ids),'augmentation_digests':('B'*64,)*len(ids)}
def run(trainer,ids=(),provider_fn=provider):return run_ultralytics_classification_epoch(trainer=trainer,replay_plan=plan(ids),replay_batch_provider=provider_fn,training_seed=19,epoch=121,global_step_start=0)


def test_plan_digest_recomputed():
    with pytest.raises(SctsrError):replace(plan(('x','y')),plan_digest='F'*64).validate_structure()

def test_counter_domain_seeds_materialized():
    value=plan(('x','y'));assert value.step_slot_seed==derive_counter_seed('replay_step_slots',19,121);assert value.identity_order_seed==derive_counter_seed('replay_identity_order',19,121)

def test_upstream_optimizer_step_called_once():
    trainer=Trainer();run(trainer,('x','y'));assert trainer.optimizer_step_calls==1

def test_base_identity_required():
    with pytest.raises(SctsrError):run(Trainer(include_identity=False))

def test_replay_order_required():
    def reversed_provider(ids,e,s,seed):
        result=provider(ids,e,s,seed);result['sample_ids']=tuple(reversed(ids));return result
    with pytest.raises(SctsrError):run(Trainer(),('x','y'),reversed_provider)

def test_amp_skip_aborts():
    with pytest.raises(SctsrError):run(Trainer(scaler=SkippingScaler()))
