from __future__ import annotations

import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_stage1_oof_folds import (  # noqa: E402
    assign_group_disjoint_folds,
    group_key_for_row,
    write_fold_manifests,
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_explicit_group_is_never_split_across_folds() -> None:
    rows = []
    for group_id in range(12):
        for idx in range(3):
            rows.append(
                {
                    "Filename": f"{group_id:04d}_{idx}.png",
                    "pipe_id": f"pipe-{group_id}",
                    "oof_y_true": "1" if group_id % 2 == 0 else "0",
                }
            )

    assigned, _, group_summaries = assign_group_disjoint_folds(
        rows,
        n_folds=4,
        seed=7,
        group_columns=("pipe_id",),
        filename_bucket_size=1000,
    )

    folds_by_group: dict[str, set[int]] = {}
    for row in assigned:
        folds_by_group.setdefault(row["oof_group_id"], set()).add(int(row["oof_fold"]))

    assert all(len(folds) == 1 for folds in folds_by_group.values())
    assert len(group_summaries) == 12


def test_numeric_filename_bucket_keeps_nearby_frames_together() -> None:
    row_a = {"Filename": "00145723.png", "oof_y_true": "0"}
    row_b = {"Filename": "00145724.png", "oof_y_true": "0"}
    row_c = {"Filename": "00146724.png", "oof_y_true": "0"}

    key_a = group_key_for_row(row_a, group_columns=(), filename_bucket_size=1000)
    key_b = group_key_for_row(row_b, group_columns=(), filename_bucket_size=1000)
    key_c = group_key_for_row(row_c, group_columns=(), filename_bucket_size=1000)

    assert key_a.group_id == key_b.group_id
    assert key_a.group_id != key_c.group_id
    assert key_a.source == "numeric_filename_bucket"


def test_stratified_group_assignment_preserves_all_rows() -> None:
    rows = []
    for group_id in range(30):
        y_true = "1" if group_id % 3 == 0 else "0"
        rows.append(
            {
                "Filename": f"{group_id:08d}.png",
                "pipe_id": f"group-{group_id}",
                "oof_y_true": y_true,
            }
        )

    assigned, fold_summaries, _ = assign_group_disjoint_folds(
        rows,
        n_folds=10,
        seed=20260606,
        group_columns=("pipe_id",),
        filename_bucket_size=1000,
    )

    assert len(assigned) == len(rows)
    assert sum(item.total_n for item in fold_summaries) == len(rows)
    assert {int(row["oof_fold"]) for row in assigned} == set(range(10))


def test_fold_manifests_use_holdout_as_validation(tmp_path: Path) -> None:
    assigned = [
        {"Filename": "target_fold0.png", "oof_y_true": "1", "oof_fold": "0"},
        {"Filename": "normal_fold0.png", "oof_y_true": "0", "oof_fold": "0"},
        {"Filename": "target_fold1.png", "oof_y_true": "1", "oof_fold": "1"},
        {"Filename": "normal_fold1.png", "oof_y_true": "0", "oof_fold": "1"},
    ]
    output_root = tmp_path / "oof"

    write_fold_manifests(
        output_root=output_root,
        assigned_rows=assigned,
        original_dataset_root=tmp_path / "dataset",
        n_folds=2,
        fieldnames=("Filename", "oof_y_true", "oof_fold"),
    )

    fold_00 = output_root / "folds" / "fold_00" / "manifests"
    train_target = read_csv_rows(fold_00 / "train_manifest.csv")
    train_normal = read_csv_rows(fold_00 / "normal_train_manifest.csv")
    val_target = read_csv_rows(fold_00 / "val_model_manifest.csv")
    val_normal = read_csv_rows(fold_00 / "normal_val_model_manifest.csv")
    holdout_target = read_csv_rows(fold_00 / "holdout_manifest.csv")
    holdout_normal = read_csv_rows(fold_00 / "normal_holdout_manifest.csv")

    assert [row["Filename"] for row in train_target] == ["target_fold1.png"]
    assert [row["Filename"] for row in train_normal] == ["normal_fold1.png"]
    assert val_target == holdout_target
    assert val_normal == holdout_normal
    assert [row["Filename"] for row in val_target] == ["target_fold0.png"]
    assert [row["Filename"] for row in val_normal] == ["normal_fold0.png"]
