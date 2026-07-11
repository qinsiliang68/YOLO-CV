from __future__ import annotations

import errno
import os
from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.errors import ValidationError
from stage1_gapvalue240.hardlink_staging import (
    StagingExpectedCounts,
    prepare_base_cache,
    storage_preflight,
    staged_replay_session,
)


def _write_image(dataset: Path, relpath: str, payload: bytes) -> dict[str, str]:
    path = dataset / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"canonical_image_relpath": relpath, "Filename": path.name}


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _fixture_manifests(tmp_path: Path) -> tuple[Path, dict[str, Path], StagingExpectedCounts]:
    dataset = tmp_path / "dataset"
    rows = {
        "train_defect": [
            _write_image(dataset, "Det/images/train/d0.png", b"d0"),
            _write_image(dataset, "Det/images/train/d1.png", b"d1"),
        ],
        "train_normal": [
            _write_image(dataset, "Det/images/normal_train/n0.png", b"n0"),
            _write_image(dataset, "Det/images/normal_train/n1.png", b"n1"),
        ],
        "val_defect": [_write_image(dataset, "Det/images/val_model/vd0.png", b"vd0")],
        "val_normal": [_write_image(dataset, "Det/images/normal_val_model/vn0.png", b"vn0")],
    }
    manifests = {
        name: _write_manifest(tmp_path / f"{name}.csv", manifest_rows)
        for name, manifest_rows in rows.items()
    }
    return dataset, manifests, StagingExpectedCounts(2, 2, 1, 1)


def test_base_cache_and_replay_use_hardlinks_then_cleanup(tmp_path):
    dataset, manifests, expected = _fixture_manifests(tmp_path)
    staging_root = tmp_path / "staging"

    cache = prepare_base_cache(dataset, staging_root, manifests, expected_counts=expected)

    assert cache.reused is False
    assert os.path.samefile(
        dataset / "Det/images/train/d0.png",
        cache.dataset_dir / "train/target_defect/d0.png",
    )
    assert os.path.samefile(
        dataset / "Det/images/normal_val_model/vn0.png",
        cache.dataset_dir / "val/no_target/vn0.png",
    )

    defect = pd.read_csv(manifests["train_defect"])
    normal = pd.read_csv(manifests["train_normal"])
    normal_replay = normal.iloc[[0]].copy()
    normal_replay["Filename"] = "replay__RUN_001__00001__n0.png"
    combined_defect = _write_manifest(tmp_path / "run_defect.csv", defect.to_dict("records"))
    combined_normal = _write_manifest(
        tmp_path / "run_normal.csv",
        pd.concat([normal, normal_replay], ignore_index=True).to_dict("records"),
    )

    with staged_replay_session(
        cache,
        combined_defect,
        combined_normal,
        run_slot="RUN_001",
        expected_replay_rows=1,
    ) as staged:
        replay = staged.dataset_dir / "train/no_target/replay__RUN_001__00001__n0.png"
        assert replay.exists()
        assert os.path.samefile(dataset / "Det/images/normal_train/n0.png", replay)
        (staged.dataset_dir / "train.cache").write_bytes(b"cache")
        assert staged.active_journal.exists()

    assert not replay.exists()
    assert not (cache.dataset_dir / "train.cache").exists()
    assert not (cache.dataset_dir / "val.cache").exists()
    assert not staged.active_journal.exists()

    reused = prepare_base_cache(dataset, staging_root, manifests, expected_counts=expected)
    assert reused.reused is True
    assert reused.snapshot_id == cache.snapshot_id


def test_cross_volume_is_rejected_before_any_hardlink(tmp_path, monkeypatch):
    dataset, manifests, expected = _fixture_manifests(tmp_path)
    staging_root = tmp_path / "other_volume" / "staging"
    calls: list[tuple[object, object]] = []

    monkeypatch.setattr(
        "stage1_gapvalue240.hardlink_staging._volume_identity",
        lambda path: "dataset-volume" if Path(path).resolve() == dataset.resolve() else "staging-volume",
    )
    monkeypatch.setattr(os, "link", lambda src, dst: calls.append((src, dst)))

    with pytest.raises(ValidationError, match="same volume"):
        prepare_base_cache(dataset, staging_root, manifests, expected_counts=expected)

    assert calls == []
    assert not (staging_root / "base_cache").exists()


def test_hardlink_failure_never_falls_back_to_copy(tmp_path, monkeypatch):
    dataset, manifests, expected = _fixture_manifests(tmp_path)
    staging_root = tmp_path / "staging"

    def fail_link(_src, _dst):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(os, "link", fail_link)

    with pytest.raises(ValidationError, match="hardlink probe failed"):
        prepare_base_cache(dataset, staging_root, manifests, expected_counts=expected)

    assert not (staging_root / "base_cache").exists()


def test_base_cache_rejects_reserved_replay_filename_prefix(tmp_path):
    dataset, manifests, expected = _fixture_manifests(tmp_path)
    frame = pd.read_csv(manifests["train_normal"])
    frame.loc[0, "Filename"] = "replay__looks_like_runtime_state.png"
    frame.to_csv(manifests["train_normal"], index=False)

    with pytest.raises(ValidationError, match="reserved replay prefix"):
        prepare_base_cache(dataset, tmp_path / "staging", manifests, expected_counts=expected)


def test_storage_preflight_enforces_free_space_and_file_ceiling(tmp_path, monkeypatch):
    dataset, manifests, _ = _fixture_manifests(tmp_path)
    staging = tmp_path / "staging"
    output = tmp_path / "output"

    with pytest.raises(ValidationError, match="file ceiling"):
        storage_preflight(
            dataset_root=dataset,
            staging_root=staging,
            output_root=output,
            hardlink_probe_manifest=manifests["train_defect"],
            expected_staging_files=10,
            maximum_staging_files=9,
            minimum_staging_free_bytes=0,
            minimum_output_free_bytes=0,
        )

    fake_usage = type("Usage", (), {"total": 100, "used": 95, "free": 5})()
    monkeypatch.setattr("stage1_gapvalue240.hardlink_staging.shutil.disk_usage", lambda _path: fake_usage)
    with pytest.raises(ValidationError, match="free space"):
        storage_preflight(
            dataset_root=dataset,
            staging_root=staging,
            output_root=output,
            hardlink_probe_manifest=manifests["train_defect"],
            expected_staging_files=10,
            maximum_staging_files=20,
            minimum_staging_free_bytes=6,
            minimum_output_free_bytes=6,
        )
