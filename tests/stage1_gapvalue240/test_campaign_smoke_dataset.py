from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.campaign_smoke_dataset import (
    LocalSmokeDatasetError,
    prepare_local_smoke_dataset,
)


def _source_dataset(root: Path, *, rows: int = 4) -> Path:
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    frames = {}
    for role, prefix in (
        ("normal_train_manifest.csv", "normal/train"),
        ("train_manifest.csv", "defect/train"),
        ("normal_val_model_manifest.csv", "normal/val"),
        ("val_model_manifest.csv", "defect/val"),
    ):
        values = []
        for index in range(rows):
            relative = f"{prefix}/{index}.jpg"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{role}-{index}".encode())
            values.append(
                {
                    "canonical_image_relpath": relative,
                    "Filename": f"{prefix.replace('/', '_')}_{index}.jpg",
                }
            )
        frames[role] = values
    for name, values in frames.items():
        pd.DataFrame(values).to_csv(manifests / name, index=False)
    return root


def test_prepare_local_smoke_dataset_is_hashed_and_reusable(tmp_path: Path) -> None:
    source = _source_dataset(tmp_path / "source")
    result = prepare_local_smoke_dataset(
        source,
        tmp_path / "smoke",
        train_per_class=3,
        val_per_class=2,
        replay_normal=2,
        batch=4,
    )

    assert result.expected_epoch_samples == 8
    assert result.expected_steps_per_epoch == 2
    assert len(pd.read_csv(result.base_normal_manifest)) == 3
    assert len(pd.read_csv(result.base_defect_manifest)) == 3
    replay = pd.read_csv(result.replay_identity_manifest)
    assert len(replay) == 2
    assert replay.staged_filename.str.startswith("replay__LOCAL_SMOKE__").all()
    assert replay.staged_filename.str.len().max() <= 48
    assert len(pd.read_csv(result.monitor_manifest)) == 4
    validation = json.loads(result.validation_path.read_text(encoding="utf-8"))
    assert validation["status"] == "PASS"
    assert validation["expected_epoch_samples"] == 8
    linked = result.dataset_dir / "train/no_target/normal_train_0.jpg"
    assert os.path.samefile(linked, source / "normal/train/0.jpg")


def test_prepare_local_smoke_dataset_fails_on_short_or_nonempty_target(tmp_path: Path) -> None:
    source = _source_dataset(tmp_path / "source", rows=1)
    with pytest.raises(LocalSmokeDatasetError, match="requires 2 rows"):
        prepare_local_smoke_dataset(
            source,
            tmp_path / "short",
            train_per_class=2,
            val_per_class=1,
            replay_normal=1,
            batch=2,
        )

    target = tmp_path / "nonempty"
    target.mkdir()
    (target / "unexpected.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        prepare_local_smoke_dataset(
            source,
            target,
            train_per_class=1,
            val_per_class=1,
            replay_normal=1,
            batch=2,
        )
