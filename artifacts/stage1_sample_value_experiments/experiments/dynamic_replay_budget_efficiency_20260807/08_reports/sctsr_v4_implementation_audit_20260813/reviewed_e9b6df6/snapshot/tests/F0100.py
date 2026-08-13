from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import stage1_sctsr_v4.branch_lineage as branch_lineage
from stage1_sctsr_v4.branch_lineage import BranchLineage
from stage1_sctsr_v4.checkpointing import build_checkpoint_payload, checkpoint_state_digest, load_checkpoint, save_checkpoint_atomic
from stage1_sctsr_v4.common_parent import CommonParentSpec
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.logical_artifact_index import LogicalArtifactEntry, LogicalArtifactIndex


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
        global_step=112560,
        base_sampler_generation=120,
        canonical_training_lock_sha256="A" * 64,
        initial_checkpoint_sha256="B" * 64,
        base_manifest_sha256="C" * 64,
        training_seed=7,
        source_tree_digest="D" * 64,
        runtime_config_digest="E" * 64,
        asset_registry_digest="F" * 64,
    )


def test_sa_084_windows_atomic_checkpoint_write_is_durable(tmp_path):
    path = tmp_path / "epoch_0120.pt"
    digest = save_checkpoint_atomic(path, _payload())

    assert path.is_file()
    assert load_checkpoint(path, expected_sha256=digest, expected_epoch=120)["global_step"] == 112560
    assert not list(tmp_path.glob("*.tmp"))


def test_sa_087_parent_checkpoint_is_immutable_after_publication(tmp_path):
    path = tmp_path / "epoch_0120.pt"
    save_checkpoint_atomic(path, _payload())

    with pytest.raises(SctsrError) as exc:
        save_checkpoint_atomic(path, _payload())

    assert exc.value.code is ErrorCode.CHILD_MUTATED_PARENT


def test_sa_085_forged_checkpoint_payload_digest_is_rejected(tmp_path):
    path = tmp_path / "epoch_0120.pt"
    payload = _payload()
    payload["model_state"]["weight"] = payload["model_state"]["weight"] + 10
    torch.save(payload, path)

    with pytest.raises(SctsrError) as exc:
        load_checkpoint(path, expected_epoch=120)

    assert exc.value.code is ErrorCode.IDENTITY_DIGEST_MISMATCH


def test_sa_082_checkpoint_digest_covers_optimizer_and_rng_state():
    payload = _payload()
    original_digest = payload["checkpoint_payload_digest"]

    payload["optimizer_state"]["param_groups"][0]["lr"] = 0.2
    with pytest.raises(SctsrError) as exc:
        from stage1_sctsr_v4.checkpointing import validate_checkpoint_payload
        validate_checkpoint_payload(payload)

    assert exc.value.code is ErrorCode.IDENTITY_DIGEST_MISMATCH
    assert payload["checkpoint_payload_digest"] == original_digest


def test_sa_080_same_payload_serializes_to_identical_parent_bytes(tmp_path):
    payload = _payload()

    first = save_checkpoint_atomic(tmp_path / "first" / "epoch_0120.pt", payload)
    second = save_checkpoint_atomic(tmp_path / "second" / "epoch_0120.pt", payload)

    assert first == second


def test_sa_086_lineage_rejects_forged_digest():
    lineage = BranchLineage.create(
        logical_run_id="RUN",
        parent_id="PARENT",
        parent_checkpoint_path="parent.pt",
        parent_checkpoint_sha256="A" * 64,
        training_seed=7,
        arm_id="T_U",
        child_source_tree_digest="B" * 64,
        child_contract_digest="C" * 64,
        created_at_utc="2026-08-13T00:00:00Z",
    )
    forged = replace(lineage, lineage_digest="FORGED")

    with pytest.raises(SctsrError) as exc:
        forged.validate(parent_sha="A" * 64, training_seed=7, arm_id="T_U", source_digest="B" * 64, contract_digest="C" * 64)

    assert exc.value.code is ErrorCode.BRANCH_LINEAGE_MISMATCH


def test_sa_084_to_087_lineage_validates_the_bound_parent_file(tmp_path):
    parent_path = tmp_path / "epoch_0120.pt"
    parent_sha = save_checkpoint_atomic(parent_path, _payload())
    lineage = BranchLineage.create(
        logical_run_id="RUN",
        parent_id="PARENT",
        parent_checkpoint_path=str(parent_path),
        parent_checkpoint_sha256=parent_sha,
        training_seed=7,
        arm_id="T_U",
        child_source_tree_digest="1" * 64,
        child_contract_digest="2" * 64,
        created_at_utc="2026-08-13T00:00:00Z",
    )
    parent_spec = CommonParentSpec(
        parent_id="PARENT",
        training_seed=7,
        canonical_training_lock_sha256="A" * 64,
        initial_checkpoint_sha256="B" * 64,
        base_manifest_sha256="C" * 64,
        source_tree_digest="D" * 64,
        runtime_config_digest="E" * 64,
        asset_registry_digest="F" * 64,
    )

    loaded = branch_lineage.validate_branch_parent(
        lineage,
        parent_spec=parent_spec,
        expected_arm_id="T_U",
        expected_child_source_digest="1" * 64,
        expected_child_contract_digest="2" * 64,
    )

    assert loaded["training_seed"] == 7
    assert load_checkpoint(parent_path, expected_sha256=parent_sha)["checkpoint_payload_digest"] == loaded["checkpoint_payload_digest"]


def test_sa_085_lineage_rejects_parent_with_wrong_seed(tmp_path):
    parent_path = tmp_path / "epoch_0120.pt"
    parent_sha = save_checkpoint_atomic(parent_path, _payload())
    lineage = BranchLineage.create(
        logical_run_id="RUN",
        parent_id="PARENT",
        parent_checkpoint_path=str(parent_path),
        parent_checkpoint_sha256=parent_sha,
        training_seed=8,
        arm_id="T_U",
        child_source_tree_digest="1" * 64,
        child_contract_digest="2" * 64,
        created_at_utc="2026-08-13T00:00:00Z",
    )
    parent_spec = CommonParentSpec(
        parent_id="PARENT",
        training_seed=8,
        canonical_training_lock_sha256="A" * 64,
        initial_checkpoint_sha256="B" * 64,
        base_manifest_sha256="C" * 64,
        source_tree_digest="D" * 64,
        runtime_config_digest="E" * 64,
        asset_registry_digest="F" * 64,
    )

    with pytest.raises(SctsrError) as exc:
        branch_lineage.validate_branch_parent(
            lineage,
            parent_spec=parent_spec,
            expected_arm_id="T_U",
            expected_child_source_digest="1" * 64,
            expected_child_contract_digest="2" * 64,
        )

    assert exc.value.code is ErrorCode.PARENT_CHECKPOINT_INCOMPLETE


def test_sa_085_lineage_rejects_wrong_child_source_binding(tmp_path):
    parent_path = tmp_path / "epoch_0120.pt"
    parent_sha = save_checkpoint_atomic(parent_path, _payload())
    lineage = BranchLineage.create(
        logical_run_id="RUN",
        parent_id="PARENT",
        parent_checkpoint_path=str(parent_path),
        parent_checkpoint_sha256=parent_sha,
        training_seed=7,
        arm_id="T_U",
        child_source_tree_digest="1" * 64,
        child_contract_digest="2" * 64,
        created_at_utc="2026-08-13T00:00:00Z",
    )
    parent_spec = CommonParentSpec(
        parent_id="PARENT",
        training_seed=7,
        canonical_training_lock_sha256="A" * 64,
        initial_checkpoint_sha256="B" * 64,
        base_manifest_sha256="C" * 64,
        source_tree_digest="D" * 64,
        runtime_config_digest="E" * 64,
        asset_registry_digest="F" * 64,
    )

    with pytest.raises(SctsrError) as exc:
        branch_lineage.validate_branch_parent(
            lineage,
            parent_spec=parent_spec,
            expected_arm_id="T_U",
            expected_child_source_digest="9" * 64,
            expected_child_contract_digest="2" * 64,
        )

    assert exc.value.code is ErrorCode.BRANCH_LINEAGE_MISMATCH


def test_sa_087_logical_index_rejects_incomplete_e1_e200_timeline():
    entry = LogicalArtifactEntry("RUN", 120, "PARENT", "PARENT", "parent/e120.pt", "A" * 64, "B" * 64, "C" * 64, "D" * 64)

    with pytest.raises(SctsrError) as exc:
        LogicalArtifactIndex([entry]).validate(require_complete_timeline=True, logical_run_id="RUN")

    assert exc.value.code is ErrorCode.LOGICAL_ARTIFACT_IDENTITY_MISMATCH


def test_sa_088_logical_index_rejects_absolute_or_unhashed_entries():
    entry = LogicalArtifactEntry("RUN", 121, "CHILD", "RUN", "C:/escape/e121.pt", "A", "B", "C", "D")

    with pytest.raises(SctsrError) as exc:
        LogicalArtifactIndex([entry]).validate()

    assert exc.value.code is ErrorCode.LOGICAL_ARTIFACT_IDENTITY_MISMATCH


def test_sa_088_and_089_complete_logical_timeline_maps_parent_then_child():
    entries = []
    for epoch in range(1, 201):
        owner = "PARENT" if epoch <= 120 else "CHILD"
        entries.append(
            LogicalArtifactEntry(
                "RUN",
                epoch,
                owner,
                "PARENT" if owner == "PARENT" else "RUN",
                f"{'02_parent' if owner == 'PARENT' else '03_branch'}/epoch_{epoch:04d}.json",
                "A" * 64,
                "B" * 64,
                "C" * 64,
                "NOT_APPLICABLE_PARENT" if owner == "PARENT" else "D" * 64,
            )
        )

    LogicalArtifactIndex(entries).validate(require_complete_timeline=True, logical_run_id="RUN")


def test_sa_090_child_tree_rejects_forged_prebranch_epoch(tmp_path):
    child_root = tmp_path / "03_branch" / "RUN"
    forged = child_root / "epoch_0120"
    forged.mkdir(parents=True)
    (forged / "metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SctsrError) as exc:
        LogicalArtifactIndex().validate_child_tree(child_root)

    assert exc.value.code is ErrorCode.LOGICAL_ARTIFACT_IDENTITY_MISMATCH


@pytest.mark.parametrize(
    ("tensor", "expected"),
    (
        (torch.tensor([[1.25, -2.5], [0.0, 3.75]], dtype=torch.float32), "A8F593A3685C605EE53C75CE75F58E572635A29204D668AB03EFEEC663F2A248"),
        (torch.tensor(42, dtype=torch.int64), "F9EF575AB39EEB1F2219C9686996A62E6E7FC54B3DC9A811FA18DB994358DC9A"),
        (torch.tensor([1.0, -0.5, 3.0], dtype=torch.bfloat16), "560B4C2D2DAB843EE8E00FD6EAE98679B112D17EB4DACCE1A5E4B0E7405AED32"),
        (torch.arange(12, dtype=torch.float64).reshape(3, 4).t(), "4A0E3AE7821DA77D3F10A12A3DD94691021F337DC3CCD35EC36B0DC3D02CF3B9"),
        (torch.empty((0, 2), dtype=torch.float16), "0C8D0B7F3FF1048136C6CEEEBE61B6EE77AC0CF060DDDD66AB316D08E4BCB189"),
    ),
)
def test_checkpoint_tensor_digest_preserves_frozen_typed_byte_vectors(tensor, expected):
    assert checkpoint_state_digest({"tensor": tensor}) == expected


def test_checkpoint_tensor_digest_uses_bulk_bytes_not_python_storage_iteration(monkeypatch):
    original_storage = torch.Tensor.untyped_storage

    def forbidden_iteration(self):
        raise AssertionError("checkpoint hashing must not materialize bytes from Python storage iteration")

    monkeypatch.setattr(torch.Tensor, "untyped_storage", forbidden_iteration)
    try:
        assert checkpoint_state_digest({"tensor": torch.arange(16, dtype=torch.float32)})
    finally:
        monkeypatch.setattr(torch.Tensor, "untyped_storage", original_storage)
