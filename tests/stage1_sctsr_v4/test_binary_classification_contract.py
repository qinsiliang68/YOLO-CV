from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.formal_cli import inspect_ultralytics_classification_caches, validate_binary_classification_contract


CLASSES = ["no_target", "target_defect"]
CLASS_TO_IDX = {"no_target": 0, "target_defect": 1}


def _trainer(tmp_path, *, head_outputs=2, model_names=None, class_to_idx=None):
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    for root in (train_root, val_root):
        for name in CLASSES:
            (root / name).mkdir(parents=True)
    mapping = CLASS_TO_IDX if class_to_idx is None else class_to_idx
    train_image_folder = SimpleNamespace(classes=CLASSES, class_to_idx=mapping)
    val_image_folder = SimpleNamespace(classes=CLASSES, class_to_idx=mapping)
    train_dataset = SimpleNamespace(
        base=train_image_folder,
        samples=[(str(train_root / "no_target" / "a.png"), 0), (str(train_root / "target_defect" / "b.png"), 1)],
    )
    val_dataset = SimpleNamespace(
        base=val_image_folder,
        samples=[(str(val_root / "no_target" / "c.png"), 0), (str(val_root / "target_defect" / "d.png"), 1)],
    )
    model = torch.nn.Sequential(torch.nn.Linear(4, head_outputs))
    model.names = {0: "no_target", 1: "target_defect"} if model_names is None else model_names
    return SimpleNamespace(
        data={"train": str(train_root), "val": str(val_root), "nc": 2, "names": {0: "no_target", 1: "target_defect"}},
        train_loader=SimpleNamespace(dataset=train_dataset),
        test_loader=SimpleNamespace(dataset=val_dataset),
        model=model,
    )


def test_binary_contract_accepts_exact_two_class_training_system(tmp_path):
    report = validate_binary_classification_contract(_trainer(tmp_path), tmp_path)
    assert report["status"] == "PASS"
    assert report["class_counts"]["train"] == {"0": 1, "1": 1}
    assert len(report["binary_contract_digest"]) == 64


def test_binary_contract_reads_frozen_imagefolder_identity_not_dataset_top_level(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.train_loader.dataset.classes = ["spoofed"]
    trainer.train_loader.dataset.class_to_idx = {"spoofed": 0}

    report = validate_binary_classification_contract(trainer, tmp_path)

    assert report["datasets"]["train"]["classes"] == CLASSES
    assert report["datasets"]["train"]["class_to_idx"] == CLASS_TO_IDX


@pytest.mark.parametrize("mutation", ["extra_dir", "swapped_index", "wrong_names", "three_output_head"])
def test_binary_contract_rejects_every_two_class_mismatch(tmp_path, mutation):
    kwargs = {}
    if mutation == "swapped_index":
        kwargs["class_to_idx"] = {"no_target": 1, "target_defect": 0}
    elif mutation == "wrong_names":
        kwargs["model_names"] = {0: "target_defect", 1: "no_target"}
    elif mutation == "three_output_head":
        kwargs["head_outputs"] = 3
    trainer = _trainer(tmp_path, **kwargs)
    if mutation == "extra_dir":
        (tmp_path / "train" / "zzz_extra_class").mkdir()

    with pytest.raises(SctsrError) as caught:
        validate_binary_classification_contract(trainer, tmp_path)
    assert caught.value.code is ErrorCode.CONFIGURATION_MISMATCH


def test_classification_cache_contract_preserves_and_binds_exact_regular_role_caches(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    (tmp_path / "train.cache").write_bytes(b"train-cache")
    (tmp_path / "val.cache").write_bytes(b"val-cache")
    unrelated = tmp_path / "unexpected.cache"
    unrelated.write_bytes(b"must-remain")

    observed = inspect_ultralytics_classification_caches(tmp_path, bind_content=True)

    assert [row["role"] for row in observed] == ["train", "val"]
    assert [row["bytes"] for row in observed] == [11, 9]
    assert all(len(row["sha256"]) == 64 for row in observed)
    assert (tmp_path / "train.cache").read_bytes() == b"train-cache"
    assert (tmp_path / "val.cache").read_bytes() == b"val-cache"
    assert unrelated.read_bytes() == b"must-remain"


def test_classification_cache_pre_setup_inspection_does_not_rehash(tmp_path, monkeypatch):
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    (tmp_path / "train.cache").write_bytes(b"train-cache")
    monkeypatch.setattr("stage1_sctsr_v4.formal_cli.sha256_file", lambda _path: pytest.fail("cache was re-hashed"))

    observed = inspect_ultralytics_classification_caches(tmp_path, bind_content=False)

    assert observed == [{"role": "train", "path": (tmp_path / "train.cache").as_posix(), "bytes": 11}]


def test_classification_cache_contract_rejects_non_regular_role_cache(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    (tmp_path / "train.cache").mkdir()

    with pytest.raises(SctsrError) as caught:
        inspect_ultralytics_classification_caches(tmp_path, bind_content=False)

    assert caught.value.code is ErrorCode.DATASET_CONTENT_MISMATCH
