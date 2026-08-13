from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import sys
import torch

from stage1_sctsr_v4.base_rng import prepare_counter_domain_base_loader
from stage1_sctsr_v4.dataset_adapter import DatasetIdentity, IdentityAugmentingDataset
from stage1_sctsr_v4.rng_isolation import derive_counter_seed


class TinyBase(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.samples = tuple((f"root/{index}.png", index % 2) for index in range(12))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        # The random draw proves the wrapper uses a repeatable augmentation
        # subdomain without mutating the surrounding worker RNG trajectory.
        return {"img": torch.rand(1, 2, 2), "cls": torch.tensor(label), "im_file": path}


def _dataset() -> IdentityAugmentingDataset:
    identities = tuple(
        DatasetIdentity(
            sample_id=f"id-{index}",
            y_true=index % 2,
            source_path=f"{index}.png",
            replay_role="NORMAL_REPLAY",
            oof_fold=index % 5,
            oof_group_id=f"bucket-{index % 3}",
            historical_dynamic_bucket="stable_high",
            identity_group=f"G{index % 5}",
        )
        for index in range(12)
    )
    return IdentityAugmentingDataset(TinyBase(), identities, training_seed=31)


def test_frozen_infinite_loader_is_reset_on_independent_counter_domains(repository_root: Path):
    yolo_root = repository_root / "YOLOv11"
    if str(yolo_root) not in sys.path:
        sys.path.insert(0, str(yolo_root))
    from ultralytics.data.build import build_dataloader

    loader = build_dataloader(_dataset(), batch=4, workers=0, shuffle=True, rank=-1)
    trainer = SimpleNamespace(train_loader=loader)
    def materialize():
        batches = list(loader)
        return (
            tuple(value for batch in batches for value in batch["sample_ids"]),
            tuple(batch["img"].clone() for batch in batches),
        )

    receipt_1 = prepare_counter_domain_base_loader(trainer, training_seed=31, epoch=121)
    order_1, images_1 = materialize()
    receipt_repeat = prepare_counter_domain_base_loader(trainer, training_seed=31, epoch=121)
    order_repeat, images_repeat = materialize()
    receipt_2 = prepare_counter_domain_base_loader(trainer, training_seed=31, epoch=122)
    order_2, _ = materialize()

    assert receipt_1 == receipt_repeat
    assert receipt_1.base_order_seed == derive_counter_seed("base_order", 31, 121)
    assert receipt_1.base_augmentation_seed == derive_counter_seed("base_augmentation", 31, 121)
    assert order_1 == order_repeat
    assert all(torch.equal(left, right) for left, right in zip(images_1, images_repeat, strict=True))
    assert order_2 != order_1
    assert receipt_2.receipt_digest != receipt_1.receipt_digest
