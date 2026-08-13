from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.fixed_step_runtime import ExponentialMovingAverage
from stage1_sctsr_v4.replay_step_plan import build_replay_step_plan
from stage1_sctsr_v4.rng_isolation import derive_counter_seed
from stage1_sctsr_v4.ultralytics_overlay import run_ultralytics_classification_epoch


SHA_A = "A" * 64
SHA_B = "B" * 64


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(4)
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, value):
        if isinstance(value, dict):
            logits = self.fc(self.bn(value["img"]))
            loss = F.cross_entropy(logits, value["cls"].view(-1).long())
            return loss, loss.detach()
        return self.fc(self.bn(value))


class _Ema:
    def __init__(self, model):
        self.inner = ExponentialMovingAverage.from_model(model)
        self.updates = 0

    def update(self, model):
        self.inner.update(model)
        self.updates += 1


class _Trainer:
    def __init__(self, *, include_identity: bool = True, scaler=None):
        torch.manual_seed(17)
        self.model = _Model()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
        self.scaler = scaler or torch.amp.GradScaler("cpu", enabled=False)
        self.ema = _Ema(self.model)
        self.device = torch.device("cpu")
        self.amp = False
        self.batch_size = 128
        self.world_size = 1
        self.accumulate = 1
        self.args = SimpleNamespace(
            resume="",
            warmup_epochs=0,
            nbs=64,
            warmup_bias_lr=0.1,
            warmup_momentum=0.8,
            momentum=0.937,
            compile=False,
        )
        batch = {
            "img": torch.randn(128, 4),
            "cls": torch.arange(128) % 2,
            "augmentation_digests": [SHA_A] * 128,
        }
        if include_identity:
            batch["sample_ids"] = [f"B{i:03d}" for i in range(128)]
        self.train_loader = [batch]
        self.optimizer_step_calls = 0

    def preprocess_batch(self, batch):
        return batch

    def optimizer_step(self):
        self.optimizer_step_calls += 1
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.ema.update(self.model)


class _SkippingScaler:
    def __init__(self):
        self.current = 2.0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        del optimizer

    def step(self, optimizer):
        del optimizer  # Deliberately simulate an overflow-skipped step.

    def update(self):
        self.current = 1.0

    def get_scale(self):
        return self.current


class _CountingScaler:
    def __init__(self):
        self.inner = torch.amp.GradScaler("cpu", enabled=False)
        self.counts = {"scale": 0, "unscale": 0, "step": 0, "update": 0}

    def scale(self, loss):
        self.counts["scale"] += 1
        return self.inner.scale(loss)

    def unscale_(self, optimizer):
        self.counts["unscale"] += 1
        return self.inner.unscale_(optimizer)

    def step(self, optimizer):
        self.counts["step"] += 1
        return self.inner.step(optimizer)

    def update(self):
        self.counts["update"] += 1
        return self.inner.update()

    def get_scale(self):
        return self.inner.get_scale()


def _plan(sample_ids=()):
    return build_replay_step_plan(
        run_id="RUN",
        arm_id="T_U" if sample_ids else "NR",
        training_seed=19,
        epoch=121,
        schedule_family="U" if sample_ids else "NR",
        sample_ids=sample_ids,
        base_batch_sizes=[128],
    )


def _provider(ids, epoch, step, seed):
    del epoch, step, seed
    return {
        "img": torch.randn(len(ids), 4),
        "cls": torch.zeros(len(ids), dtype=torch.long),
        "sample_ids": tuple(ids),
        "augmentation_digests": (SHA_B,) * len(ids),
    }


def test_sa_075_replay_step_plan_recomputes_content_digest():
    plan = _plan(("R0", "R1"))
    forged = replace(plan, plan_digest="F" * 64)

    with pytest.raises(SctsrError) as exc:
        forged.validate_structure()

    assert exc.value.code is ErrorCode.IDENTITY_DIGEST_MISMATCH


def test_sa_075_step_plan_records_counter_domain_seeds():
    plan = _plan(("R0", "R1"))

    assert plan.step_slot_seed == derive_counter_seed("replay_step_slots", 19, 121)
    assert plan.identity_order_seed == derive_counter_seed("replay_identity_order", 19, 121)
    assert plan.step_slot_seed != plan.training_seed


def test_sa_108_overlay_calls_upstream_optimizer_step_exactly_once():
    trainer = _Trainer()

    result = run_ultralytics_classification_epoch(
        trainer=trainer,
        replay_plan=_plan(("R0", "R1")),
        replay_batch_provider=_provider,
        training_seed=19,
        epoch=121,
        global_step_start=0,
    )

    assert trainer.optimizer_step_calls == 1
    assert result["optimizer_steps"] == 1
    assert result["ema_updates_delta"] == 1


def test_sa_106_base_identity_and_augmentation_trace_are_mandatory():
    trainer = _Trainer(include_identity=False)

    with pytest.raises(SctsrError) as exc:
        run_ultralytics_classification_epoch(
            trainer=trainer,
            replay_plan=_plan(),
            replay_batch_provider=_provider,
            training_seed=19,
            epoch=121,
            global_step_start=0,
        )

    assert exc.value.code is ErrorCode.BASE_ORDER_MISMATCH


def test_sa_109_replay_provider_must_preserve_materialized_identity_order():
    trainer = _Trainer()

    def reordered(ids, epoch, step, seed):
        result = _provider(ids, epoch, step, seed)
        result["sample_ids"] = tuple(reversed(ids))
        return result

    with pytest.raises(SctsrError) as exc:
        run_ultralytics_classification_epoch(
            trainer=trainer,
            replay_plan=_plan(("R0", "R1")),
            replay_batch_provider=reordered,
            training_seed=19,
            epoch=121,
            global_step_start=0,
        )

    assert exc.value.code is ErrorCode.SCHEDULE_EXPOSURE_MISMATCH


def test_sa_108_amp_overflow_skip_aborts_epoch():
    trainer = _Trainer(scaler=_SkippingScaler())

    with pytest.raises(SctsrError) as exc:
        run_ultralytics_classification_epoch(
            trainer=trainer,
            replay_plan=_plan(),
            replay_batch_provider=_provider,
            training_seed=19,
            epoch=121,
            global_step_start=0,
        )

    assert exc.value.code is ErrorCode.OPTIMIZER_STEP_SKIPPED


def test_sa_114_unscale_step_update_are_each_called_once_per_base_step():
    scaler = _CountingScaler()
    trainer = _Trainer(scaler=scaler)

    run_ultralytics_classification_epoch(
        trainer=trainer,
        replay_plan=_plan(("R0", "R1")),
        replay_batch_provider=_provider,
        training_seed=19,
        epoch=121,
        global_step_start=0,
    )

    assert scaler.counts == {"scale": 2, "unscale": 1, "step": 1, "update": 1}


def test_sa_105_warmup_uses_only_base_step_coordinates():
    no_replay = _Trainer()
    replay = _Trainer()
    for trainer in (no_replay, replay):
        trainer.args.warmup_epochs = 3.0
        trainer.lf = lambda epoch: 0.7 + epoch * 0.01

    nr_plan = build_replay_step_plan(
        run_id="RUN",
        arm_id="NR",
        training_seed=19,
        epoch=1,
        schedule_family="NR",
        sample_ids=(),
        base_batch_sizes=(128,),
    )
    nr_result = run_ultralytics_classification_epoch(
        trainer=no_replay,
        replay_plan=nr_plan,
        replay_batch_provider=_provider,
        training_seed=19,
        epoch=1,
        global_step_start=0,
    )
    replay_result = run_ultralytics_classification_epoch(
        trainer=replay,
        replay_plan=build_replay_step_plan(
            run_id="RUN",
            arm_id="T_U",
            training_seed=19,
            epoch=1,
            schedule_family="U",
            sample_ids=("R0", "R1"),
            base_batch_sizes=(128,),
        ),
        replay_batch_provider=_provider,
        training_seed=19,
        epoch=1,
        global_step_start=0,
    )

    assert [group["lr"] for group in no_replay.optimizer.param_groups] == [group["lr"] for group in replay.optimizer.param_groups]
    assert nr_result["optimizer_steps"] == replay_result["optimizer_steps"] == 1
