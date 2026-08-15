from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.formal_cli import validate_binary_classification_contract


CLASSES = ["no_target", "target_defect"]
CLASS_TO_IDX = {"no_target": 0, "target_defect": 1}


def _trainer(tmp_path, *, head_outputs=2, model_names=None, class_to_idx=None):
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    for root in (train_root, val_root):
        for name in CLASSES:
            (root / name).mkdir(parents=True)
    mapping = CLASS_TO_IDX if class_to_idx is None else class_to_idx
    train_dataset = SimpleNamespace(
        classes=CLASSES,
        class_to_idx=mapping,
        samples=[(str(train_root / "no_target" / "a.png"), 0), (str(train_root / "target_defect" / "b.png"), 1)],
    )
    val_dataset = SimpleNamespace(
        classes=CLASSES,
        class_to_idx=mapping,
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
