from dataclasses import replace

import pytest
import torch

from stage1_sctsr_v4.checkpointing import build_checkpoint_payload, validate_checkpoint_payload
from stage1_sctsr_v4.common_parent import CommonParentSpec, validate_parent_checkpoint
from stage1_sctsr_v4.errors import ErrorCode, SctsrError


class _Scaler:
    def state_dict(self):
        return {"scale": 1.0}


class _Ema:
    def __init__(self, model):
        self.ema = torch.nn.Linear(2, 2)
        self.ema.load_state_dict(model.state_dict())
        self.updates = 112_560


def payload(seed: int = 1, epoch: int = 120):
    torch.manual_seed(seed)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    return build_checkpoint_payload(
        model=model,
        ema=_Ema(model),
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=_Scaler(),
        epoch=epoch,
        global_step=112_560,
        base_sampler_generation=120,
        canonical_training_lock_sha256="A" * 64,
        initial_checkpoint_sha256="B" * 64,
        base_manifest_sha256="C" * 64,
        training_seed=seed,
        source_tree_digest="D" * 64,
        runtime_config_digest="E" * 64,
        asset_registry_digest="F" * 64,
    )


def spec(seed: int = 1):
    return CommonParentSpec(
        parent_id="P",
        training_seed=seed,
        canonical_training_lock_sha256="A" * 64,
        initial_checkpoint_sha256="B" * 64,
        base_manifest_sha256="C" * 64,
        source_tree_digest="D" * 64,
        runtime_config_digest="E" * 64,
        asset_registry_digest="F" * 64,
    )


def test_sa_082_complete_parent_payload_valid():
    validate_parent_checkpoint(payload(), spec())


def test_sa_085_parent_wrong_epoch_rejected():
    with pytest.raises(SctsrError):
        validate_checkpoint_payload(payload(epoch=119), expected_epoch=120)


def test_sa_085_parent_wrong_seed_rejected():
    with pytest.raises(SctsrError):
        validate_parent_checkpoint(payload(seed=2), spec(seed=1))


def test_sa_085_parent_wrong_asset_registry_rejected():
    with pytest.raises(SctsrError) as exc:
        validate_parent_checkpoint(payload(), replace(spec(), asset_registry_digest="9" * 64))
    assert exc.value.code is ErrorCode.PARENT_CHECKPOINT_INCOMPLETE


def test_sa_083_parent_wrong_global_step_and_ema_count_rejected():
    wrong_step = payload()
    wrong_step["global_step"] -= 1
    from stage1_sctsr_v4.checkpointing import checkpoint_payload_digest
    wrong_step["checkpoint_payload_digest"] = checkpoint_payload_digest(wrong_step)
    with pytest.raises(SctsrError) as exc:
        validate_parent_checkpoint(wrong_step, spec())
    assert exc.value.code is ErrorCode.BASE_STEP_COUNT_MISMATCH

    wrong_ema = payload()
    wrong_ema["ema_updates"] -= 1
    wrong_ema["checkpoint_payload_digest"] = checkpoint_payload_digest(wrong_ema)
    with pytest.raises(SctsrError) as exc:
        validate_parent_checkpoint(wrong_ema, spec())
    assert exc.value.code is ErrorCode.BASE_STEP_COUNT_MISMATCH


def test_sa_083_parent_spec_requires_runtime_and_asset_identities():
    with pytest.raises(SctsrError):
        replace(spec(), runtime_config_digest="").validate()
    with pytest.raises(SctsrError):
        replace(spec(), asset_registry_digest="").validate()


def test_sa_080_parent_spec_requires_zero_replay():
    spec().validate()
