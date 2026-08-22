from types import SimpleNamespace
import torch
import torch.nn.functional as F
from stage1_sctsr_v4.fixed_step_runtime import ExponentialMovingAverage
from stage1_sctsr_v4.replay_step_plan import build_replay_step_plan
from stage1_sctsr_v4.ultralytics_overlay import run_ultralytics_classification_epoch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.bn=torch.nn.BatchNorm1d(4);self.fc=torch.nn.Linear(4,2)
    def forward(self,x):
        if isinstance(x,dict):
            logits=self.fc(self.bn(x['img']));loss=F.cross_entropy(logits,x['cls'].view(-1).long());return loss,loss.detach()
        return self.fc(self.bn(x))

class EMA:
    def __init__(self,m):self.inner=ExponentialMovingAverage.from_model(m);self.updates=0
    def update(self,m):self.inner.update(m);self.updates+=1

class Trainer:
    def __init__(self):
        self.model=Model();self.optimizer=torch.optim.SGD(self.model.parameters(),lr=.01);self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda _:1.0);self.scaler=torch.amp.GradScaler('cpu');self.ema=EMA(self.model);self.device=torch.device('cpu');self.amp=False;self.batch_size=128;self.world_size=1;self.accumulate=1;self.args=SimpleNamespace(resume='')
        self.train_loader=[{'img':torch.randn(128,4),'cls':torch.arange(128)%2,'sample_ids':[f'b{i}' for i in range(128)],'augmentation_digests':['a']*128}]
    def preprocess_batch(self,b):return b
    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer);torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=10.0)
        self.scaler.step(self.optimizer);self.scaler.update();self.optimizer.zero_grad(set_to_none=True);self.ema.update(self.model)

def test_upstream_overlay_keeps_one_step():
    t=Trainer();plan=build_replay_step_plan(run_id='r',arm_id='T_U',training_seed=1,epoch=121,schedule_family='U',sample_ids=['x','y'],base_batch_sizes=[128])
    def provider(ids,e,s,seed):return {'img':torch.randn(len(ids),4),'cls':torch.zeros(len(ids),dtype=torch.long),'sample_ids':tuple(ids),'augmentation_digests':('r',)*len(ids)}
    receipts=[];r=run_ultralytics_classification_epoch(trainer=t,replay_plan=plan,replay_batch_provider=provider,training_seed=1,epoch=121,global_step_start=0,step_receipt_sink=receipts.append);assert r['optimizer_steps']==1 and r['replay_occurrences']==2 and r['ema_updates_delta']==1
    assert receipts[0].optimizer_step_delta==1 and receipts[0].ema_update_delta==1
    assert receipts[0].combined_loss_for_reporting == receipts[0].base_loss + receipts[0].replay_loss
    assert receipts[0].rng_before_replay==receipts[0].rng_after_replay
    assert receipts[0].bn_before_replay==receipts[0].bn_after_replay
