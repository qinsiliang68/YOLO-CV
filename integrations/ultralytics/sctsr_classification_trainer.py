"""Concrete YOLO-CV integration loaded only inside the frozen main checkout.

This file is intentionally outside ``YOLOv11/ultralytics`` so applying the
SCTSR overlay does not modify archived upstream source.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from stage1_sctsr_v4.dataset_adapter import IdentityAugmentingDataset, load_identity_manifest

try:
    from ultralytics.models.yolo.classify.train import ClassificationTrainer
except ImportError as exc:  # pragma: no cover - exercised in the complete YOLO-CV checkout.
    raise RuntimeError("Add <YOLO-CV>/YOLOv11 to PYTHONPATH before importing this adapter") from exc


class SctsrClassificationTrainer(ClassificationTrainer):
    def __init__(self, *args: Any, identity_manifest: str | Path, **kwargs: Any) -> None:
        self._sctsr_identities = load_identity_manifest(identity_manifest)
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        dataset = super().build_dataset(img_path, mode=mode, batch=batch)
        if mode != "train":
            return dataset
        return IdentityAugmentingDataset(dataset, self._sctsr_identities)

    def replay_batch_provider(self, sample_ids, epoch: int, base_step_index: int, training_seed: int):
        del epoch, base_step_index, training_seed
        dataset = self.train_loader.dataset
        if not isinstance(dataset, IdentityAugmentingDataset):
            raise RuntimeError("SCTSR identity-preserving training dataset is not installed")
        return dataset.replay_batch(sample_ids)
