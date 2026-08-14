from __future__ import annotations

import csv
from pathlib import Path

import pytest

from stage1_sctsr_v4.engineering_canary import (
    ENGINEERING_CANARY_SEMANTIC,
    select_engineering_canary_samples,
    validate_engineering_canary_report,
)
from stage1_sctsr_v4.errors import SctsrError


def _write_manifest(path: Path, *, split: str, names: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("split", "Filename", "canonical_image_relpath"),
        )
        writer.writeheader()
        for name in names:
            writer.writerow(
                {
                    "split": split,
                    "Filename": name,
                    "canonical_image_relpath": f"Det/images/{split}/{name}",
                }
            )


def test_select_engineering_canary_samples_is_train_only_and_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_manifest(root / "manifests" / "train_manifest.csv", split="train", names=("d2.png", "d1.png"))
    _write_manifest(
        root / "manifests" / "normal_train_manifest.csv",
        split="normal_train",
        names=("n2.png", "n1.png"),
    )
    for split, names in {"train": ("d2.png", "d1.png"), "normal_train": ("n2.png", "n1.png")}.items():
        directory = root / "Det" / "images" / split
        directory.mkdir(parents=True)
        for name in names:
            (directory / name).write_bytes(name.encode("ascii"))

    first = select_engineering_canary_samples(root, per_class=2)
    second = select_engineering_canary_samples(root, per_class=2)

    assert first == second
    assert [(sample.sample_id, sample.label, sample.split_role) for sample in first] == [
        ("d2.png", 1, "train"),
        ("d1.png", 1, "train"),
        ("n2.png", 0, "normal_train"),
        ("n1.png", 0, "normal_train"),
    ]
    assert {sample.split_role for sample in first} == {"train", "normal_train"}
    assert all(
        Path(sample.image_path).relative_to(root.resolve()).parts[:3]
        in {("Det", "images", "train"), ("Det", "images", "normal_train")}
        for sample in first
    )
    assert all(len(sample.image_sha256) == 64 and sample.image_bytes > 0 for sample in first)


def test_select_engineering_canary_samples_rejects_non_train_manifest_role(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_manifest(root / "manifests" / "train_manifest.csv", split="test", names=("d.png",))
    _write_manifest(root / "manifests" / "normal_train_manifest.csv", split="normal_train", names=("n.png",))
    for split, name in (("test", "d.png"), ("normal_train", "n.png")):
        path = root / "Det" / "images" / split / name
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")

    with pytest.raises(SctsrError):
        select_engineering_canary_samples(root, per_class=1)


def test_engineering_canary_report_is_fail_closed() -> None:
    report = {
        "schema_version": "stage1.sctsr.engineering_canary.v1",
        "status": "PASS",
        "scientific_role": ENGINEERING_CANARY_SEMANTIC,
        "real_image_count": 4,
        "real_image_bytes_verified": True,
        "real_yolo_weight_verified": True,
        "forward_passed": True,
        "base_backward_passed": True,
        "replay_backward_passed": True,
        "optimizer_step_count": 1,
        "ema_update_count": 1,
        "parquet": {"canonical_parquet": True, "compression": "ZSTD"},
        "checkpoint": {"save_passed": True, "reload_passed": True},
        "recovery": {"corrupt_checkpoint_rejected": True, "partial_generation_quarantined": True},
        "frontier_point_count": 96,
        "formal_training_started": False,
        "assignments_generated": False,
        "engineering_gate_generated": False,
        "pilot_release_generated": False,
        "blind_holdout_opened": False,
        "test_accessed": False,
        "method_effectiveness_claimed": False,
    }
    assert validate_engineering_canary_report(report)["status"] == "PASS"
    report["test_accessed"] = True
    with pytest.raises(SctsrError):
        validate_engineering_canary_report(report)
