from __future__ import annotations

import csv
from pathlib import Path

import pytest

from run_real_data_engineering_canary import CanaryInputError, ensure_repository_import_path, select_canary_samples


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("split", "source_image_path", "canonical_image_relpath", "Defect"),
        )
        writer.writeheader()
        writer.writerows(rows)


def test_select_canary_samples_uses_only_registered_train_rows_and_t_identity(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    paths = []
    for index in range(5):
        path = images / f"{index}.png"
        path.write_bytes(f"image-{index}".encode())
        paths.append(path)
    defect = tmp_path / "defect.csv"
    normal = tmp_path / "normal.csv"
    _write_manifest(
        defect,
        [
            {"split": "train", "source_image_path": str(paths[0]), "canonical_image_relpath": "D0", "Defect": "1"},
            {"split": "train", "source_image_path": str(paths[1]), "canonical_image_relpath": "D1", "Defect": "1"},
        ],
    )
    _write_manifest(
        normal,
        [
            {"split": "normal_train", "source_image_path": str(paths[2]), "canonical_image_relpath": "N0", "Defect": "0"},
            {"split": "normal_train", "source_image_path": str(paths[3]), "canonical_image_relpath": "N1", "Defect": "0"},
            {"split": "normal_train", "source_image_path": str(paths[4]), "canonical_image_relpath": "T0", "Defect": "0"},
        ],
    )
    t_manifest = tmp_path / "t.csv"
    with t_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "y_true", "replay_role"))
        writer.writeheader()
        writer.writerow({"sample_id": "T0", "y_true": "0", "replay_role": "normal_replay"})

    selected = select_canary_samples(
        defect_manifest=defect,
        normal_manifest=normal,
        treatment_manifest=t_manifest,
        base_per_label=2,
    )

    assert [row.sample_id for row in selected.base] == ["D0", "D1", "N0", "N1"]
    assert selected.replay.sample_id == "T0"
    assert selected.replay.role == "T_STRESS_ENGINEERING_CANARY"
    assert {row.split for row in (*selected.base, selected.replay)} == {"train", "normal_train"}
    assert len({row.sample_id for row in (*selected.base, selected.replay)}) == 5
    assert all(len(row.image_sha256) == 64 and row.image_bytes > 0 for row in (*selected.base, selected.replay))


def test_select_canary_samples_rejects_forbidden_or_missing_train_material(tmp_path: Path):
    image = tmp_path / "x.png"
    image.write_bytes(b"x")
    defect = tmp_path / "defect.csv"
    normal = tmp_path / "normal.csv"
    _write_manifest(
        defect,
        [{"split": "val_op", "source_image_path": str(image), "canonical_image_relpath": "D0", "Defect": "1"}],
    )
    _write_manifest(
        normal,
        [{"split": "normal_train", "source_image_path": str(image), "canonical_image_relpath": "T0", "Defect": "0"}],
    )
    t_manifest = tmp_path / "t.csv"
    t_manifest.write_text("sample_id,y_true,replay_role\nT0,0,normal_replay\n", encoding="utf-8")

    with pytest.raises(CanaryInputError, match="train-only"):
        select_canary_samples(
            defect_manifest=defect,
            normal_manifest=normal,
            treatment_manifest=t_manifest,
            base_per_label=1,
        )


def test_deep_entrypoint_registers_the_repository_root_for_imports(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("sys.path", ["deep/tools/path"])
    observed = ensure_repository_import_path(tmp_path)
    assert observed == tmp_path.resolve()
    assert __import__("sys").path[0] == str(tmp_path.resolve())
