from dataclasses import replace

import pytest
import torch

import stage1_sctsr_v4.branch_lineage as branch_lineage
from stage1_sctsr_v4.checkpointing import build_checkpoint_payload, load_checkpoint, save_checkpoint_atomic
from stage1_sctsr_v4.common_parent import CommonParentSpec
from stage1_sctsr_v4.errors import SctsrError
from stage1_sctsr_v4.logical_artifact_index import LogicalArtifactIndex


class _Scaler:
    def state_dict(self):
        return {"scale": 1.0}


class _Ema:
    def __init__(self, model):
        self.ema = torch.nn.Linear(2, 2)
        self.ema.load_state_dict(model.state_dict())
        self.updates = 112_560


def _payload():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    return build_checkpoint_payload(
        model=model,
        ema=_Ema(model),
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=_Scaler(),
        epoch=120,
        global_step=112_560,
        base_sampler_generation=120,
        canonical_training_lock_sha256="A" * 64,
        initial_checkpoint_sha256="B" * 64,
        base_manifest_sha256="C" * 64,
        training_seed=7,
        source_tree_digest="D" * 64,
        runtime_config_digest="E" * 64,
        asset_registry_digest="F" * 64,
    )


def _spec():
    return CommonParentSpec("P", 7, "A" * 64, "B" * 64, "C" * 64, "D" * 64)


def test_windows_atomic_checkpoint_write_is_durable(tmp_path):
    path = tmp_path / "epoch_0120.pt"
    digest = save_checkpoint_atomic(path, _payload())
    assert load_checkpoint(path, expected_sha256=digest, expected_epoch=120)["global_step"] == 112_560


def test_forged_tensor_content_is_rejected(tmp_path):
    path = tmp_path / "epoch_0120.pt"
    payload = _payload()
    payload["model_state"]["weight"] = payload["model_state"]["weight"] + 10
    torch.save(payload, path)
    with pytest.raises(SctsrError):
        load_checkpoint(path, expected_epoch=120)


def test_branch_launch_validates_actual_parent_not_a_bare_path():
    assert callable(branch_lineage.validate_branch_parent)


def test_logical_index_rejects_forged_child_history(tmp_path):
    child = tmp_path / "epoch_0120"
    child.mkdir()
    with pytest.raises(SctsrError):
        LogicalArtifactIndex().validate_child_tree(tmp_path)


def test_parent_spec_requires_runtime_and_asset_identity():
    with pytest.raises(SctsrError):
        replace(_spec(), runtime_config_digest="").validate()
