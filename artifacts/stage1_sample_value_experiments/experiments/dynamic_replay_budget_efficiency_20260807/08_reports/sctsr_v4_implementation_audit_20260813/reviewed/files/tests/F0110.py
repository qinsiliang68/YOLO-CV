import pytest
import torch

from stage1_sctsr_v4.dataset_adapter import DatasetIdentity, IdentityAugmentingDataset, tensor_digest
from stage1_sctsr_v4.errors import ErrorCode, SctsrError


class Base(torch.utils.data.Dataset):
    samples = (("root/a.png", 0), ("root/b.png", 1))
    def __len__(self):
        return 2

    def __getitem__(self, i):
        return {
            "img": torch.full((1, 2, 2), float(i)),
            "cls": torch.tensor(i % 2),
            "im_file": f"root/{'a' if i == 0 else 'b'}.png",
        }


def identities():
    return (
        DatasetIdentity("a", 0, "a.png", "normal", 0, "bucket0", "easy", "G0"),
        DatasetIdentity("b", 1, "b.png", "defect", 1, "bucket1", "hard", "G1"),
    )


def test_identity_wrapper_preserves_length_and_adds_metadata():
    d = IdentityAugmentingDataset(Base(), identities())
    assert len(d) == 2
    item = d[1]
    assert item["sample_id"] == "b"
    assert item["oof_fold"] == 1
    assert item["historical_dynamic_bucket"] == "hard"
    assert len(item["augmentation_digest"]) == 64


def test_identity_wrapper_aligns_manifest_to_physical_dataset_order():
    d = IdentityAugmentingDataset(Base(), tuple(reversed(identities())))
    assert d[0]["sample_id"] == "a"
    assert d[1]["sample_id"] == "b"


def test_replay_batch_is_identity_addressed_and_carries_frozen_metadata():
    d = IdentityAugmentingDataset(Base(), identities())
    length_before = len(d)
    batch = d.replay_batch(["b", "a"])
    assert batch["img"].shape[0] == 2
    assert batch["sample_ids"] == ("b", "a")
    assert batch["cls"].tolist() == [1, 0]
    assert batch["oof_folds"] == (1, 0)
    assert batch["identity_groups"] == ("G1", "G0")
    assert len(d) == length_before == len(d.base_dataset)


def test_source_path_mismatch_fails_closed():
    wrong = (DatasetIdentity("a", 0, "wrong.png"), DatasetIdentity("b", 1, "b.png"))
    with pytest.raises(SctsrError) as exc:
        IdentityAugmentingDataset(Base(), wrong)
    assert exc.value.code is ErrorCode.PREDICTION_IDENTITY_MISMATCH


def test_tensor_digest_is_deterministic():
    x = torch.arange(4).reshape(1, 2, 2)
    assert tensor_digest(x) == tensor_digest(x.clone())
