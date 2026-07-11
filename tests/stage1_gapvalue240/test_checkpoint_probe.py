from __future__ import annotations

from pathlib import Path

import pytest
import torch

from stage1_gapvalue240.checkpoint_probe import inspect_checkpoint
from stage1_gapvalue240.errors import ValidationError


def test_checkpoint_probe_distinguishes_reloadable_and_resumable(tmp_path):
    checkpoint = tmp_path / "last.pt"
    torch.save({
        "epoch": 17,
        "optimizer": {"state": {}, "param_groups": []},
        "train_args": {"epochs": 200, "data": "dataset", "model": "base.pt", "batch": 128, "imgsz": 224, "seed": 1},
        "model": {"weight": torch.ones(1)},
    }, checkpoint)

    report = inspect_checkpoint(checkpoint, require_resume_state=True)

    assert report["status"] == "PASS"
    assert report["epoch"] == 17
    assert report["resumable"] is True


def test_checkpoint_probe_rejects_corrupt_or_stripped_resume_state(tmp_path):
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    with pytest.raises(ValidationError, match="cannot be loaded"):
        inspect_checkpoint(corrupt, require_resume_state=False)

    stripped = tmp_path / "stripped.pt"
    torch.save({"epoch": -1, "optimizer": None}, stripped)
    assert inspect_checkpoint(stripped, require_resume_state=False)["status"] == "PASS"
    with pytest.raises(ValidationError, match="not resumable"):
        inspect_checkpoint(stripped, require_resume_state=True)
