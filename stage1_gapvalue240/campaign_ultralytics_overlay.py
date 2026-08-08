"""Spawn-safe Ultralytics classes used by the dynamic replay trainer."""

from __future__ import annotations

from typing import Any

from ultralytics.data import ClassificationDataset
from ultralytics.models.yolo.classify.train import ClassificationTrainer


class TraceableClassificationDataset(ClassificationDataset):
    """Expose the source path so per-sample process telemetry can join identities."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        sample["im_file"] = str(self.samples[index][0])
        return sample


class RelocatableCampaignClassificationTrainer(ClassificationTrainer):
    """Keep registered paths when a segmented run resumes in its staged workspace."""

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        return TraceableClassificationDataset(
            root=img_path,
            args=self.args,
            augment=mode == "train",
            prefix=mode,
        )

    def check_resume(self, overrides):
        super().check_resume(overrides)
        if self.resume:
            for key in ("project", "name", "data", "exist_ok"):
                if key in overrides:
                    setattr(self.args, key, overrides[key])
            self.args.save_dir = None


__all__ = [
    "RelocatableCampaignClassificationTrainer",
    "TraceableClassificationDataset",
]
