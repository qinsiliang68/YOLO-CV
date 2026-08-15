import os
from pathlib import Path

import pytest
import torch

from stage1_sctsr_v4.dataset_adapter import (
    DatasetIdentity,
    IdentityAugmentingDataset,
    IdentityFilteringDataset,
    tensor_digest,
    validate_materialized_dataset_bytes,
    revalidate_materialized_dataset_binding,
)
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


def test_materialized_training_file_is_sha_bound_not_filename_label_only(tmp_path: Path):
    expected = tmp_path / "canonical" / "a.png"
    observed = tmp_path / "staging" / "train" / "no_target" / "a.png"
    expected.parent.mkdir(parents=True)
    observed.parent.mkdir(parents=True)
    expected.write_bytes(b"expected-image-bytes")
    observed.write_bytes(b"different-image-bytes")

    class PhysicalBase(torch.utils.data.Dataset):
        samples = ((str(observed), 0),)

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"img": torch.zeros((1, 2, 2)), "cls": torch.tensor(0)}

    wrapped = IdentityAugmentingDataset(
        PhysicalBase(),
        (DatasetIdentity("Det/images/normal_train/a.png", 0, str(expected)),),
    )
    content = {
        "Det/images/normal_train/a.png": {
            "image_bytes": expected.stat().st_size,
            "image_sha256": __import__("hashlib").sha256(expected.read_bytes()).hexdigest().upper(),
        }
    }

    with pytest.raises(SctsrError) as caught:
        validate_materialized_dataset_bytes(wrapped, content, role="train")
    assert caught.value.code is ErrorCode.DATASET_CONTENT_MISMATCH


def test_validation_wrapper_filters_excluded_physical_rows_and_preserves_transform():
    class ValidationBase(torch.utils.data.Dataset):
        samples = (
            ("root/keep_a.png", 0),
            ("root/excluded.png", 0),
            ("root/keep_b.png", 1),
        )
        torch_transforms = object()

        def __len__(self):
            return 3

        def __getitem__(self, index):
            return {"img": torch.full((1,), index), "cls": torch.tensor(self.samples[index][1])}

    identities = (
        DatasetIdentity("Det/images/normal_val_model/keep_a.png", 0, "keep_a.png"),
        DatasetIdentity("Det/images/val_model/keep_b.png", 1, "keep_b.png"),
    )
    wrapped = IdentityFilteringDataset(ValidationBase(), identities)

    assert len(wrapped) == 2
    assert [Path(row[0]).name for row in wrapped.samples] == ["keep_a.png", "keep_b.png"]
    assert [wrapped[index]["cls"].item() for index in range(2)] == [0, 1]
    assert wrapped.torch_transforms is ValidationBase.torch_transforms
    assert tuple(item.sample_id for item in wrapped.identities) == (
        "Det/images/normal_val_model/keep_a.png",
        "Det/images/val_model/keep_b.png",
    )


def test_materialized_binding_detects_post_setup_byte_replacement(tmp_path: Path):
    canonical = tmp_path / "dataset" / "Det" / "images" / "normal_train" / "a.png"
    staged = tmp_path / "dataset" / "train" / "no_target" / "a.png"
    canonical.parent.mkdir(parents=True)
    staged.parent.mkdir(parents=True)
    canonical.write_bytes(b"frozen")
    os.link(canonical, staged)

    class PhysicalBase(torch.utils.data.Dataset):
        samples = ((str(staged), 0),)

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"img": torch.zeros((1, 2, 2)), "cls": torch.tensor(0)}

    sample_id = "Det/images/normal_train/a.png"
    wrapped = IdentityAugmentingDataset(PhysicalBase(), (DatasetIdentity(sample_id, 0, sample_id),))
    content = {
        sample_id: {
            "image_bytes": canonical.stat().st_size,
            "image_sha256": __import__("hashlib").sha256(canonical.read_bytes()).hexdigest().upper(),
        }
    }
    binding = validate_materialized_dataset_bytes(
        wrapped,
        content,
        role="train",
        dataset_root=tmp_path / "dataset",
        evidence_path=tmp_path / "binding" / "train.parquet",
    )

    staged.write_bytes(b"changed-after-setup")
    with pytest.raises(SctsrError) as caught:
        revalidate_materialized_dataset_binding(binding)
    assert caught.value.code is ErrorCode.DATASET_CONTENT_MISMATCH


def test_materialized_binding_rejects_unregistered_extra_file(tmp_path: Path):
    canonical = tmp_path / "dataset" / "Det" / "images" / "normal_train" / "a.png"
    staged = tmp_path / "dataset" / "train" / "no_target" / "a.png"
    canonical.parent.mkdir(parents=True)
    staged.parent.mkdir(parents=True)
    canonical.write_bytes(b"frozen")
    os.link(canonical, staged)
    (staged.parent / "extra.png").write_bytes(b"extra")

    class PhysicalBase(torch.utils.data.Dataset):
        samples = ((str(staged), 0),)

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"img": torch.zeros((1, 2, 2)), "cls": torch.tensor(0)}

    sample_id = "Det/images/normal_train/a.png"
    wrapped = IdentityAugmentingDataset(PhysicalBase(), (DatasetIdentity(sample_id, 0, sample_id),))
    content = {
        sample_id: {
            "image_bytes": canonical.stat().st_size,
            "image_sha256": __import__("hashlib").sha256(canonical.read_bytes()).hexdigest().upper(),
        }
    }
    with pytest.raises(SctsrError) as caught:
        validate_materialized_dataset_bytes(
            wrapped,
            content,
            role="train",
            dataset_root=tmp_path / "dataset",
            evidence_path=tmp_path / "binding" / "train.parquet",
        )
    assert caught.value.code is ErrorCode.DATASET_CONTENT_MISMATCH


def test_materialized_binding_requires_hardlink_to_canonical_source(tmp_path: Path):
    canonical = tmp_path / "dataset" / "Det" / "images" / "normal_train" / "a.png"
    staged = tmp_path / "classification_view" / "train" / "no_target" / "a.png"
    canonical.parent.mkdir(parents=True)
    staged.parent.mkdir(parents=True)
    canonical.write_bytes(b"same-bytes")
    staged.write_bytes(b"same-bytes")

    class PhysicalBase(torch.utils.data.Dataset):
        samples = ((str(staged), 0),)

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"img": torch.zeros((1, 2, 2)), "cls": torch.tensor(0)}

    sample_id = "Det/images/normal_train/a.png"
    wrapped = IdentityAugmentingDataset(PhysicalBase(), (DatasetIdentity(sample_id, 0, sample_id),))
    content = {
        sample_id: {
            "image_bytes": canonical.stat().st_size,
            "image_sha256": __import__("hashlib").sha256(canonical.read_bytes()).hexdigest().upper(),
        }
    }

    with pytest.raises(SctsrError) as caught:
        validate_materialized_dataset_bytes(
            wrapped,
            content,
            role="train",
            dataset_root=tmp_path / "dataset",
            materialized_data_root=tmp_path / "classification_view",
            evidence_path=tmp_path / "binding" / "train.parquet",
        )
    assert caught.value.code is ErrorCode.DATASET_CONTENT_MISMATCH


def test_materialized_binding_rejects_extra_file_added_after_setup(tmp_path: Path):
    canonical = tmp_path / "dataset" / "Det" / "images" / "normal_train" / "a.png"
    staged = tmp_path / "classification_view" / "train" / "no_target" / "a.png"
    canonical.parent.mkdir(parents=True)
    staged.parent.mkdir(parents=True)
    canonical.write_bytes(b"frozen")
    os.link(canonical, staged)

    class PhysicalBase(torch.utils.data.Dataset):
        samples = ((str(staged), 0),)

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"img": torch.zeros((1, 2, 2)), "cls": torch.tensor(0)}

    sample_id = "Det/images/normal_train/a.png"
    wrapped = IdentityAugmentingDataset(PhysicalBase(), (DatasetIdentity(sample_id, 0, sample_id),))
    content = {
        sample_id: {
            "image_bytes": canonical.stat().st_size,
            "image_sha256": __import__("hashlib").sha256(canonical.read_bytes()).hexdigest().upper(),
        }
    }
    binding = validate_materialized_dataset_bytes(
        wrapped,
        content,
        role="train",
        dataset_root=tmp_path / "dataset",
        materialized_data_root=tmp_path / "classification_view",
        evidence_path=tmp_path / "binding" / "train.parquet",
    )
    (staged.parent / "added_after_setup.png").write_bytes(b"unregistered")

    with pytest.raises(SctsrError) as caught:
        revalidate_materialized_dataset_binding(binding)
    assert caught.value.code is ErrorCode.DATASET_CONTENT_MISMATCH


def test_materialized_binding_accepts_separate_hardlink_only_classification_view(tmp_path: Path):
    canonical = tmp_path / "dataset" / "Det" / "images" / "normal_train" / "a.png"
    staged = tmp_path / "classification_view" / "train" / "no_target" / "a.png"
    canonical.parent.mkdir(parents=True)
    staged.parent.mkdir(parents=True)
    canonical.write_bytes(b"frozen")
    os.link(canonical, staged)

    class PhysicalBase(torch.utils.data.Dataset):
        samples = ((str(staged), 0),)

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"img": torch.zeros((1, 2, 2)), "cls": torch.tensor(0)}

    sample_id = "Det/images/normal_train/a.png"
    wrapped = IdentityAugmentingDataset(PhysicalBase(), (DatasetIdentity(sample_id, 0, sample_id),))
    content = {
        sample_id: {
            "image_bytes": canonical.stat().st_size,
            "image_sha256": __import__("hashlib").sha256(canonical.read_bytes()).hexdigest().upper(),
        }
    }

    binding = validate_materialized_dataset_bytes(
        wrapped,
        content,
        role="train",
        dataset_root=tmp_path / "dataset",
        materialized_data_root=tmp_path / "classification_view",
        evidence_path=tmp_path / "binding" / "train.parquet",
    )
    assert binding["canonical_dataset_root"] == (tmp_path / "dataset").resolve().as_posix()
    assert binding["materialized_data_root"] == (tmp_path / "classification_view").resolve().as_posix()
    assert revalidate_materialized_dataset_binding(binding)["status"] == "PASS"
