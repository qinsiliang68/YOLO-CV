# -*- coding: utf-8 -*-
"""Build group-disjoint OOF fold manifests for Stage-1 sample scoring.

This script does not train a model. It only creates reproducible 10-fold
manifests so each fold can be trained and predicted independently on separate
machines.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_DATASET_ROOT = Path("data") / "final_sewerml_dataset"
DEFAULT_OUTPUT_ROOT = Path("artifacts") / "stage1_oof_folds_10fold_20260617"
DEFAULT_GROUP_COLUMNS = (
    "pipe_id",
    "inspection_id",
    "video_id",
    "PipeId",
    "InspectionId",
    "VideoId",
    "pipe",
    "inspection",
    "video",
)
ADDED_COLUMNS = (
    "oof_source_manifest",
    "oof_y_true",
    "oof_fold",
    "oof_group_id",
    "oof_group_source",
    "oof_group_size",
    "oof_group_positive_n",
    "oof_group_negative_n",
)


@dataclass(frozen=True)
class GroupKey:
    group_id: str
    source: str


@dataclass
class MutableGroup:
    group_id: str
    source: str
    rows: list[dict[str, str]]
    positive_n: int = 0
    negative_n: int = 0

    @property
    def total_n(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class GroupSummary:
    group_id: str
    source: str
    fold: int
    total_n: int
    positive_n: int
    negative_n: int


@dataclass(frozen=True)
class FoldSummary:
    fold: int
    total_n: int
    positive_n: int
    negative_n: int
    train_total_n: int
    train_positive_n: int
    train_negative_n: int


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def write_csv(path: Path, rows: Sequence[dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def merge_fieldnames(*fieldname_lists: Iterable[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for fieldnames in fieldname_lists:
        for name in fieldnames:
            if name not in seen:
                merged.append(name)
                seen.add(name)
    for name in ADDED_COLUMNS:
        if name not in seen:
            merged.append(name)
            seen.add(name)
    return merged


def normalize_group_columns(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return DEFAULT_GROUP_COLUMNS
    return tuple(part.strip() for part in value.split(",") if part.strip())


def numeric_token_from_row(row: dict[str, str]) -> int | None:
    candidates = (
        row.get("Filename", ""),
        Path(row.get("canonical_image_relpath", "")).name,
        Path(row.get("source_image_path", "")).name,
    )
    for candidate in candidates:
        stem = Path(candidate).stem
        tokens = re.findall(r"\d+", stem)
        if tokens:
            return int(max(tokens, key=len))
    return None


def stable_short_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def group_key_for_row(
    row: dict[str, str],
    *,
    group_columns: Sequence[str],
    filename_bucket_size: int,
) -> GroupKey:
    for column in group_columns:
        value = row.get(column, "").strip()
        if value:
            return GroupKey(group_id=f"{column}:{value}", source=f"explicit:{column}")

    token = numeric_token_from_row(row)
    if token is not None and filename_bucket_size > 0:
        bucket = token // filename_bucket_size
        return GroupKey(
            group_id=f"filename_bucket_{filename_bucket_size}:{bucket}",
            source="numeric_filename_bucket",
        )

    fallback = (
        row.get("canonical_image_relpath")
        or row.get("source_image_path")
        or row.get("Filename")
        or json.dumps(row, sort_keys=True, ensure_ascii=True)
    )
    return GroupKey(group_id=f"row_hash:{stable_short_hash(fallback)}", source="row_hash_fallback")


def assign_group_disjoint_folds(
    rows: Sequence[dict[str, str]],
    *,
    n_folds: int,
    seed: int,
    group_columns: Sequence[str],
    filename_bucket_size: int,
) -> tuple[list[dict[str, str]], list[FoldSummary], list[GroupSummary]]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if not rows:
        raise ValueError("rows must not be empty")

    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as exc:  # pragma: no cover - project dependency guard
        raise RuntimeError("scikit-learn is required for stratified group OOF folds") from exc

    groups: dict[str, MutableGroup] = {}
    annotated_rows: list[dict[str, str]] = []
    for row in rows:
        key = group_key_for_row(row, group_columns=group_columns, filename_bucket_size=filename_bucket_size)
        group = groups.setdefault(key.group_id, MutableGroup(group_id=key.group_id, source=key.source, rows=[]))
        copied = dict(row)
        copied["oof_group_id"] = key.group_id
        copied["oof_group_source"] = key.source
        annotated_rows.append(copied)
        group.rows.append(copied)
        if str(row.get("oof_y_true", "")).strip() == "1":
            group.positive_n += 1
        else:
            group.negative_n += 1

    total_n = len(rows)
    positive_n = sum(group.positive_n for group in groups.values())
    negative_n = total_n - positive_n
    fold_totals = [0] * n_folds
    fold_positives = [0] * n_folds
    fold_negatives = [0] * n_folds
    group_to_fold: dict[str, int] = {}

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y = [int(row["oof_y_true"]) for row in annotated_rows]
    group_ids = [row["oof_group_id"] for row in annotated_rows]

    for fold, (_, holdout_indices) in enumerate(
        splitter.split(X=list(range(len(annotated_rows))), y=y, groups=group_ids)
    ):
        for row_index in holdout_indices:
            row = annotated_rows[int(row_index)]
            group_id = row["oof_group_id"]
            previous_fold = group_to_fold.setdefault(group_id, fold)
            if previous_fold != fold:
                raise RuntimeError(f"Group split across folds: {group_id}")
            row["oof_fold"] = str(fold)

    for group_id, group in groups.items():
        fold = group_to_fold[group_id]
        fold_totals[fold] += group.total_n
        fold_positives[fold] += group.positive_n
        fold_negatives[fold] += group.negative_n

    assigned: list[dict[str, str]] = []
    group_summaries: list[GroupSummary] = []
    for group in groups.values():
        fold = group_to_fold[group.group_id]
        group_summaries.append(
            GroupSummary(
                group_id=group.group_id,
                source=group.source,
                fold=fold,
                total_n=group.total_n,
                positive_n=group.positive_n,
                negative_n=group.negative_n,
            )
        )
        for row in group.rows:
            row["oof_fold"] = str(fold)
            row["oof_group_size"] = str(group.total_n)
            row["oof_group_positive_n"] = str(group.positive_n)
            row["oof_group_negative_n"] = str(group.negative_n)
            assigned.append(row)

    fold_summaries = [
        FoldSummary(
            fold=fold,
            total_n=fold_totals[fold],
            positive_n=fold_positives[fold],
            negative_n=fold_negatives[fold],
            train_total_n=total_n - fold_totals[fold],
            train_positive_n=positive_n - fold_positives[fold],
            train_negative_n=negative_n - fold_negatives[fold],
        )
        for fold in range(n_folds)
    ]
    return assigned, fold_summaries, sorted(group_summaries, key=lambda item: (item.fold, item.group_id))


def add_source_columns(rows: list[dict[str, str]], *, source_manifest: str, y_true: int) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        copied = dict(row)
        copied["oof_source_manifest"] = source_manifest
        copied["oof_y_true"] = str(y_true)
        output.append(copied)
    return output


def write_fold_manifests(
    *,
    output_root: Path,
    assigned_rows: Sequence[dict[str, str]],
    original_dataset_root: Path,
    n_folds: int,
    fieldnames: Sequence[str],
) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    for fold in range(n_folds):
        fold_rows = [row for row in assigned_rows if int(row["oof_fold"]) == fold]
        train_rows = [row for row in assigned_rows if int(row["oof_fold"]) != fold]

        train_target = [row for row in train_rows if row["oof_y_true"] == "1"]
        train_normal = [row for row in train_rows if row["oof_y_true"] == "0"]
        holdout_target = [row for row in fold_rows if row["oof_y_true"] == "1"]
        holdout_normal = [row for row in fold_rows if row["oof_y_true"] == "0"]

        fold_manifest_dir = output_root / "folds" / f"fold_{fold:02d}" / "manifests"
        write_csv(fold_manifest_dir / "train_manifest.csv", train_target, fieldnames)
        write_csv(fold_manifest_dir / "normal_train_manifest.csv", train_normal, fieldnames)
        write_csv(fold_manifest_dir / "holdout_manifest.csv", holdout_target, fieldnames)
        write_csv(fold_manifest_dir / "normal_holdout_manifest.csv", holdout_normal, fieldnames)
        write_csv(fold_manifest_dir / "val_model_manifest.csv", holdout_target, fieldnames)
        write_csv(fold_manifest_dir / "normal_val_model_manifest.csv", holdout_normal, fieldnames)

        jobs.append(
            {
                "fold": str(fold),
                "manifest_dir": str(fold_manifest_dir),
                "train_target_n": str(len(train_target)),
                "train_normal_n": str(len(train_normal)),
                "holdout_target_n": str(len(holdout_target)),
                "holdout_normal_n": str(len(holdout_normal)),
                "suggested_train_command": (
                    "uv run python scripts/train_stage1_cls_sweep.py "
                    "--mode full --models l --epochs 200 --imgsz 224 --batch 128 --workers 4 "
                    "--save-period 1 --keep-data --device 0 "
                    f"--dataset-root {original_dataset_root} "
                    f"--manifest-dir {fold_manifest_dir} "
                    f"--work-root data/stage1_oof_workdir/fold_{fold:02d} "
                    f"--runs-root YOLOv11/runs/stage1_oof_10fold/fold_{fold:02d}"
                ),
            }
        )
    return jobs


def write_summary_files(
    *,
    output_root: Path,
    assigned_rows: Sequence[dict[str, str]],
    fold_summaries: Sequence[FoldSummary],
    group_summaries: Sequence[GroupSummary],
    fieldnames: Sequence[str],
    jobs: Sequence[dict[str, str]],
    metadata: dict[str, object],
) -> None:
    write_csv(output_root / "train_oof_assignments.csv", assigned_rows, fieldnames)
    write_csv(
        output_root / "fold_summary.csv",
        [summary.__dict__ for summary in fold_summaries],
        (
            "fold",
            "total_n",
            "positive_n",
            "negative_n",
            "train_total_n",
            "train_positive_n",
            "train_negative_n",
        ),
    )
    write_csv(
        output_root / "group_summary.csv",
        [summary.__dict__ for summary in group_summaries],
        ("group_id", "source", "fold", "total_n", "positive_n", "negative_n"),
    )
    write_csv(
        output_root / "fold_jobs.csv",
        jobs,
        (
            "fold",
            "manifest_dir",
            "train_target_n",
            "train_normal_n",
            "holdout_target_n",
            "holdout_normal_n",
            "suggested_train_command",
        ),
    )
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_oof_folds(
    *,
    dataset_root: Path,
    output_root: Path,
    n_folds: int,
    seed: int,
    group_columns: Sequence[str],
    filename_bucket_size: int,
    overwrite: bool,
) -> None:
    manifest_dir = dataset_root / "manifests"
    target_rows, target_fields = read_csv(manifest_dir / "train_manifest.csv")
    normal_rows, normal_fields = read_csv(manifest_dir / "normal_train_manifest.csv")
    rows = add_source_columns(target_rows, source_manifest="train_manifest.csv", y_true=1)
    rows.extend(add_source_columns(normal_rows, source_manifest="normal_train_manifest.csv", y_true=0))
    fieldnames = merge_fieldnames(target_fields, normal_fields)

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Use --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    assigned, fold_summaries, group_summaries = assign_group_disjoint_folds(
        rows,
        n_folds=n_folds,
        seed=seed,
        group_columns=group_columns,
        filename_bucket_size=filename_bucket_size,
    )
    jobs = write_fold_manifests(
        output_root=output_root,
        assigned_rows=assigned,
        original_dataset_root=dataset_root,
        n_folds=n_folds,
        fieldnames=fieldnames,
    )

    group_sources: dict[str, int] = {}
    for summary in group_summaries:
        group_sources[summary.source] = group_sources.get(summary.source, 0) + 1

    metadata = {
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "manifest_dir": str(manifest_dir),
        "output_root": str(output_root),
        "n_folds": n_folds,
        "seed": seed,
        "group_columns": list(group_columns),
        "filename_bucket_size": filename_bucket_size,
        "total_rows": len(assigned),
        "positive_rows": sum(1 for row in assigned if row["oof_y_true"] == "1"),
        "negative_rows": sum(1 for row in assigned if row["oof_y_true"] == "0"),
        "group_count": len(group_summaries),
        "group_sources": group_sources,
        "note": (
            "Explicit pipe/inspection/video columns are used when present. "
            "This dataset currently has no such fields in the train manifests, "
            "so numeric filename buckets are used as a conservative surrogate "
            "to keep nearby frames in the same OOF fold. Each fold trains on "
            "the other 9/10 rows and writes its own 1/10 holdout as both "
            "holdout_* and val_model_* manifests."
        ),
    }
    write_summary_files(
        output_root=output_root,
        assigned_rows=assigned,
        fold_summaries=fold_summaries,
        group_summaries=group_summaries,
        fieldnames=fieldnames,
        jobs=jobs,
        metadata=metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument(
        "--group-columns",
        default=",".join(DEFAULT_GROUP_COLUMNS),
        help="Comma-separated explicit group-id columns, tried in order before filename bucketing.",
    )
    parser.add_argument(
        "--filename-bucket-size",
        type=int,
        default=1000,
        help="Fallback numeric filename bucket size when no explicit group id exists.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_oof_folds(
        dataset_root=args.dataset_root.resolve(),
        output_root=args.output_root.resolve(),
        n_folds=args.folds,
        seed=args.seed,
        group_columns=normalize_group_columns(args.group_columns),
        filename_bucket_size=args.filename_bucket_size,
        overwrite=args.overwrite,
    )
    print(f"wrote_oof_folds={args.output_root.resolve()}")


if __name__ == "__main__":
    main()
