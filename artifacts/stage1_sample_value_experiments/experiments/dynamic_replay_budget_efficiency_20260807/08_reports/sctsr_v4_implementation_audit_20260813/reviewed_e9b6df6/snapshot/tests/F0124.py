from __future__ import annotations

from types import SimpleNamespace

import torch

from stage1_sctsr_v4.branch_lineage import BranchLineage
from stage1_sctsr_v4.fixed_step_runtime import ExponentialMovingAverage
from stage1_sctsr_v4.formal_training import FormalIdentity, run_prepared_branch, run_prepared_common_parent
from stage1_sctsr_v4.identity_pool import IdentityRecord, identity_digest
from stage1_sctsr_v4.schedule import build_schedule
from stage1_sctsr_v4.arm_spec import ArmId
from stage1_sctsr_v4.serialization import sha256_file


class FakeScaler:
    def state_dict(self):
        return {}

    def load_state_dict(self, _state):
        return None


class FakeTrainer:
    def __init__(self):
        self.model = torch.nn.Linear(2, 2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)
        self.scaler = FakeScaler()
        self.ema = ExponentialMovingAverage.from_model(self.model)
        self.train_loader = SimpleNamespace(sampler=SimpleNamespace(set_epoch=lambda _epoch: None))


def _identity(seed=7):
    token = "A" * 64
    return FormalIdentity(seed, token, token, token, token, token, token)


def _groups():
    records = [IdentityRecord(f"id{i}", i % 2, "NORMAL_REPLAY", "hard", i % 2, f"b{i}") for i in range(5)]
    return {f"G{i}": (records[i],) for i in range(5)}, identity_digest(records)


def test_parent_and_branch_orchestration_preserve_parent_and_never_mislabel_synthetic(monkeypatch, tmp_path):
    import stage1_sctsr_v4.formal_training as formal

    monkeypatch.setattr(formal, "CANONICAL_BASE_STEPS", 1)
    monkeypatch.setattr(formal, "_base_batch_sizes", lambda _trainer: (128,))

    def fake_epoch(*, replay_plan, global_step_start, **_kwargs):
        return {
            "optimizer_steps": 1,
            "replay_occurrences": replay_plan.planned_replay_occurrences,
            "global_step_end": global_step_start + 1,
            "base_occurrences": 128,
            "ema_updates_delta": 1,
            "base_order_digest": "B" * 64,
            "base_augmentation_digest": "C" * 64,
        }

    monkeypatch.setattr(formal, "run_ultralytics_classification_epoch", fake_epoch)

    trainer = FakeTrainer()
    identity = _identity()
    parent_root = tmp_path / "parent"
    parent = run_prepared_common_parent(
        trainer=trainer,
        identity=identity,
        output_root=parent_root,
        release_authorization=None,
        execution_mode="synthetic",
    )
    assert parent["status"] == "IMPLEMENTED_NOT_FORMALLY_RUN"
    assert parent["global_step"] == 120
    parent_path = parent_root / "epoch_0120.pt"
    parent_sha = sha256_file(parent_path)

    groups, digest = _groups()
    schedule = build_schedule(ArmId.T_U, primary_groups=groups, primary_digest=digest, base_denominator=200)
    lineage = BranchLineage.create(
        logical_run_id="RUN_SYNTHETIC_T_U",
        parent_id=parent["parent_id"],
        parent_checkpoint_path=parent_path.as_posix(),
        parent_checkpoint_sha256=parent_sha,
        training_seed=identity.training_seed,
        arm_id="T_U",
        child_source_tree_digest=identity.source_tree_digest,
        child_contract_digest="D" * 64,
    )
    branch = run_prepared_branch(
        trainer=FakeTrainer(),
        identity=identity,
        parent_checkpoint=parent_path,
        lineage=lineage,
        schedule=schedule,
        replay_batch_provider=lambda *_args: {},
        output_root=tmp_path / "branch",
        release_authorization=None,
        execution_mode="synthetic",
    )
    assert branch["status"] == "IMPLEMENTED_NOT_FORMALLY_RUN"
    assert branch["global_step"] == 200
    assert 200 in branch["checkpoints"]
    assert sha256_file(parent_path) == parent_sha
