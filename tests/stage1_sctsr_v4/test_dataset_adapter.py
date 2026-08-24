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


def _single_materialized_dataset(staged: Path, sample_id: str):
    class PhysicalBase(torch.utils.data.Dataset):
        samples = ((str(staged), 0),)

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"img": torch.zeros((1, 2, 2)), "cls": torch.tensor(0)}

    return IdentityAugmentingDataset(PhysicalBase(), (DatasetIdentity(sample_id, 0, sample_id),))


def _role_root_binding_fixture(tmp_path: Path):
    canonical_root = tmp_path / "canonical"
    materialized_root = tmp_path / "classification_view"
    role_root = materialized_root / "train"
    canonical = canonical_root / "Det" / "images" / "normal_train" / "a.png"
    staged = role_root / "no_target" / "a.png"
    canonical.parent.mkdir(parents=True)
    staged.parent.mkdir(parents=True)
    canonical.write_bytes(b"frozen")
    os.link(canonical, staged)
    sample_id = "Det/images/normal_train/a.png"
    content = {
        sample_id: {
            "image_bytes": canonical.stat().st_size,
            "image_sha256": __import__("hashlib").sha256(canonical.read_bytes()).hexdigest().upper(),
        }
    }
    return {
        "canonical_root": canonical_root,
        "materialized_root": materialized_root,
        "role_root": role_root,
        "staged": staged,
        "dataset": _single_materialized_dataset(staged, sample_id),
        "content": content,
    }


def test_materialized_hardlink_is_hashed_only_once(tmp_path: Path, monkeypatch):
    import stage1_sctsr_v4.dataset_adapter as adapter

    fixture = _role_root_binding_fixture(tmp_path)
    original = adapter.sha256_file
    hashed_paths = []

    def count_hash(path):
        hashed_paths.append(Path(path))
        return original(path)

    monkeypatch.setattr(adapter, "sha256_file", count_hash)
    binding = validate_materialized_dataset_bytes(
        fixture["dataset"],
        fixture["content"],
        role="train",
        dataset_root=fixture["canonical_root"],
        materialized_data_root=fixture["materialized_root"],
        materialized_role_root=fixture["role_root"],
        allowed_materialized_role_roots=(fixture["role_root"],),
        evidence_path=tmp_path / "binding" / "train.parquet",
    )

    assert hashed_paths == [fixture["staged"]]
    assert binding["byte_verification_mode"] == "DIRECT_LOADER_SHA256"


def test_materialized_hardlink_reuses_same_invocation_canonical_verification(tmp_path: Path, monkeypatch):
    import stage1_sctsr_v4.dataset_adapter as adapter

    fixture = _role_root_binding_fixture(tmp_path)
    verification = {
        "schema_version": "stage1.sctsr.dataset_content_validation.v1",
        "status": "PASS",
        "dataset_root": fixture["canonical_root"].resolve().as_posix(),
        "physical_verification_enabled": True,
        "physical_files_verified": 1,
    }
    monkeypatch.setattr(adapter, "sha256_file", lambda _path: pytest.fail("hardlink bytes were read twice"))

    binding = validate_materialized_dataset_bytes(
        fixture["dataset"],
        fixture["content"],
        role="train",
        dataset_root=fixture["canonical_root"],
        materialized_data_root=fixture["materialized_root"],
        materialized_role_root=fixture["role_root"],
        allowed_materialized_role_roots=(fixture["role_root"],),
        canonical_physical_verification=verification,
        evidence_path=tmp_path / "binding" / "train.parquet",
    )

    assert binding["byte_verification_mode"] == "REGISTERED_CANONICAL_PHYSICAL_VERIFICATION_REUSED"


def test_materialized_hardlink_rejects_unrelated_canonical_verification(tmp_path: Path):
    fixture = _role_root_binding_fixture(tmp_path)
    verification = {
        "schema_version": "stage1.sctsr.dataset_content_validation.v1",
        "status": "PASS",
        "dataset_root": (tmp_path / "other").resolve().as_posix(),
        "physical_verification_enabled": True,
        "physical_files_verified": 1,
    }

    with pytest.raises(SctsrError) as caught:
        validate_materialized_dataset_bytes(
            fixture["dataset"],
            fixture["content"],
            role="train",
            dataset_root=fixture["canonical_root"],
            materialized_data_root=fixture["materialized_root"],
            materialized_role_root=fixture["role_root"],
            allowed_materialized_role_roots=(fixture["role_root"],),
            canonical_physical_verification=verification,
        )
    assert caught.value.code is ErrorCode.DATASET_CONTENT_MISMATCH


def test_materialized_role_root_rejects_preexisting_sibling_class(tmp_path: Path):
    fixture = _role_root_binding_fixture(tmp_path)
    injected = fixture["role_root"] / "injected_class" / "extra.png"
    injected.parent.mkdir()
    injected.write_bytes(b"extra")

    with pytest.raises(SctsrError) as caught:
        validate_materialized_dataset_bytes(
            fixture["dataset"],
            fixture["content"],
            role="train",
            dataset_root=fixture["canonical_root"],
            materialized_data_root=fixture["materialized_root"],
            materialized_role_root=fixture["role_root"],
            allowed_materialized_role_roots=(fixture["role_root"],),
            evidence_path=tmp_path / "binding" / "train.parquet",
        )
    assert caught.value.code is ErrorCode.DATASET_CONTENT_MISMATCH


def test_materialized_role_root_rejects_preexisting_sibling_role(tmp_path: Path):
    fixture = _role_root_binding_fixture(tmp_path)
    (fixture["materialized_root"] / "injected_role").mkdir()

    with pytest.raises(SctsrError) as caught:
        validate_materialized_dataset_bytes(
            fixture["dataset"],
            fixture["content"],
            role="train",
            dataset_root=fixture["canonical_root"],
            materialized_data_root=fixture["materialized_root"],
            materialized_role_root=fixture["role_root"],
            allowed_materialized_role_roots=(fixture["role_root"],),
            evidence_path=tmp_path / "binding" / "train.parquet",
        )
    assert caught.value.code is ErrorCode.DATASET_CONTENT_MISMATCH


def test_materialized_role_contract_accepts_exact_train_and_val_hardlink_trees(tmp_path: Path):
    canonical_root = tmp_path / "canonical"
    materialized_root = tmp_path / "classification_view"
    train_role = materialized_root / "train"
    val_role = materialized_root / "val"
    train_role.mkdir(parents=True)
    val_role.mkdir(parents=True)
    (materialized_root / "train.cache").write_bytes(b"regular-train-cache")
    (materialized_root / "val.cache").write_bytes(b"regular-val-cache")
    content = {}
    bindings = []
    for role, role_root, source_role in (
        ("train", train_role, "normal_train"),
        ("val_model", val_role, "normal_val_model"),
    ):
        sample_id = f"Det/images/{source_role}/{role}.png"
        canonical = canonical_root / sample_id
        staged = role_root / "no_target" / f"{role}.png"
        canonical.parent.mkdir(parents=True)
        staged.parent.mkdir(parents=True)
        canonical.write_bytes(role.encode("ascii"))
        os.link(canonical, staged)
        content[sample_id] = {
            "image_bytes": canonical.stat().st_size,
            "image_sha256": __import__("hashlib").sha256(canonical.read_bytes()).hexdigest().upper(),
        }
        bindings.append(
            validate_materialized_dataset_bytes(
                _single_materialized_dataset(staged, sample_id),
                content,
                role=role,
                dataset_root=canonical_root,
                materialized_data_root=materialized_root,
                materialized_role_root=role_root,
                allowed_materialized_role_roots=(train_role, val_role),
                evidence_path=tmp_path / "binding" / f"{role}.parquet",
            )
        )

    assert [revalidate_materialized_dataset_binding(binding)["status"] for binding in bindings] == ["PASS", "PASS"]


def test_materialized_role_root_revalidation_rejects_sibling_class_and_role(tmp_path: Path):
    fixture = _role_root_binding_fixture(tmp_path)
    binding = validate_materialized_dataset_bytes(
        fixture["dataset"],
        fixture["content"],
        role="train",
        dataset_root=fixture["canonical_root"],
        materialized_data_root=fixture["materialized_root"],
        materialized_role_root=fixture["role_root"],
        allowed_materialized_role_roots=(fixture["role_root"],),
        evidence_path=tmp_path / "binding" / "train.parquet",
    )

    sibling_class = fixture["role_root"] / "injected_class" / "extra.png"
    sibling_class.parent.mkdir()
    sibling_class.write_bytes(b"extra")
    with pytest.raises(SctsrError) as caught:
        revalidate_materialized_dataset_binding(binding)
    assert caught.value.code is ErrorCode.DATASET_CONTENT_MISMATCH

    sibling_class.unlink()
    sibling_class.parent.rmdir()
    sibling_role = fixture["materialized_root"] / "injected_role"
    sibling_role.mkdir()
    with pytest.raises(SctsrError) as caught:
        revalidate_materialized_dataset_binding(binding)
    assert caught.value.code is ErrorCode.DATASET_CONTENT_MISMATCH


def test_materialized_role_root_checks_every_ancestor_for_reparse_points(tmp_path: Path, monkeypatch):
    import stage1_sctsr_v4.dataset_adapter as adapter

    fixture = _role_root_binding_fixture(tmp_path)
    original = adapter._is_symlink_or_reparse
    poisoned = fixture["role_root"] / "no_target"
    monkeypatch.setattr(
        adapter,
        "_is_symlink_or_reparse",
        lambda path: Path(path) == poisoned or original(Path(path)),
    )

    with pytest.raises(SctsrError) as caught:
        validate_materialized_dataset_bytes(
            fixture["dataset"],
            fixture["content"],
            role="train",
            dataset_root=fixture["canonical_root"],
            materialized_data_root=fixture["materialized_root"],
            materialized_role_root=fixture["role_root"],
            allowed_materialized_role_roots=(fixture["role_root"],),
            evidence_path=tmp_path / "binding" / "train.parquet",
        )
    assert caught.value.code is ErrorCode.DATASET_CONTENT_MISMATCH
