# -*- coding: utf-8 -*-
"""Predict Stage-1 OOF holdouts and write sample-difficulty tables.

Training a fold only creates ``best.pt`` and training curves.  This script is
the missing post-training step: for each completed fold, load that fold's
``best.pt``, predict only its held-out manifests, and write per-image raw
probabilities plus the user-facing difficulty coordinate:

``wrong_confidence_raw = 1 - confidence assigned to the true label``

The raw probability is used because the 0.4-0.6 decision-boundary view only
makes sense before calibration or deployment-prior adjustment.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_stage1_cls_gate import (  # noqa: E402
    EvalConfig,
    PREDICTION_COLUMNS,
    Paths,
    collect_artifact_rows,
    ensure_yolo_import,
    file_sha256,
    manifest_row_to_record,
    predict_records,
    read_manifest,
    write_csv,
    write_json,
)


DEFAULT_DATASET_ROOT = Path("data") / "final_sewerml_dataset"
DEFAULT_OOF_ROOT = Path("artifacts") / "stage1_oof_folds_10fold_20260617"
DEFAULT_RUNS_ROOT = Path("YOLOv11") / "runs" / "stage1_oof_10fold"
DEFAULT_YOLO_ROOT = Path("YOLOv11")
DEFAULT_OUTPUT_ROOT = Path("artifacts") / "stage1_oof_predictions_20260621"

DIFFICULTY_COLUMNS = (
    "oof_fold",
    "human_fold",
    "fold_run_dir",
    "weights",
    "true_confidence_raw",
    "wrong_confidence_raw",
    "difficulty_bucket_raw",
)

OOF_PREDICTION_COLUMNS = (*DIFFICULTY_COLUMNS, *PREDICTION_COLUMNS)


@dataclass(frozen=True)
class FoldJob:
    fold: int
    manifest_dir: Path
    weights: Path
    run_dir: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def parse_fold_spec(value: str, *, base: int) -> list[int]:
    folds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid fold range: {part}")
            folds.extend(range(start, end + 1))
        else:
            folds.append(int(part))

    output: list[int] = []
    seen: set[int] = set()
    for raw_fold in folds:
        fold = raw_fold - base
        if fold < 0 or fold > 9:
            raise ValueError(f"Fold out of range after base conversion: raw={raw_fold}, fold={fold}")
        if fold not in seen:
            output.append(fold)
            seen.add(fold)
    if not output:
        raise ValueError("No folds selected")
    return output


def discover_fold_weights(runs_root: Path, fold: int) -> tuple[Path, Path]:
    fold_root = runs_root / f"fold_{fold:02d}"
    if not fold_root.exists():
        raise FileNotFoundError(f"Missing fold run root: {fold_root}")

    candidates = sorted(
        (path for path in fold_root.glob("*/weights/best.pt") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No best.pt found under: {fold_root}")
    weights = candidates[0]
    run_dir = weights.parents[1]
    return weights, run_dir


def build_fold_jobs(*, folds: Iterable[int], oof_root: Path, runs_root: Path) -> list[FoldJob]:
    jobs: list[FoldJob] = []
    for fold in folds:
        manifest_dir = oof_root / "folds" / f"fold_{fold:02d}" / "manifests"
        for filename in ("val_model_manifest.csv", "normal_val_model_manifest.csv"):
            path = manifest_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing fold holdout manifest: {path}")
        weights, run_dir = discover_fold_weights(runs_root, fold)
        jobs.append(FoldJob(fold=fold, manifest_dir=manifest_dir, weights=weights, run_dir=run_dir))
    return jobs


def load_fold_holdout_records(job: FoldJob, dataset_root: Path) -> list[dict[str, str]]:
    defect_rows = read_manifest(job.manifest_dir / "val_model_manifest.csv")
    normal_rows = read_manifest(job.manifest_dir / "normal_val_model_manifest.csv")
    records = [manifest_row_to_record(row, "oof_holdout", 1, dataset_root) for row in defect_rows]
    records.extend(manifest_row_to_record(row, "oof_holdout", 0, dataset_root) for row in normal_rows)
    return records


def difficulty_bucket(wrong_confidence: float) -> str:
    if wrong_confidence >= 0.9:
        return "confidently_wrong"
    if wrong_confidence > 0.6:
        return "wrong_not_confident"
    if wrong_confidence >= 0.4:
        return "decision_boundary"
    if wrong_confidence > 0.1:
        return "correct_not_confident"
    return "confidently_correct"


def add_difficulty_columns(rows: list[dict[str, str]], job: FoldJob) -> list[dict[str, str]]:
    enriched_rows: list[dict[str, str]] = []
    for row in rows:
        y_true = int(row["y_true"])
        p_defect = float(row["p_defect_raw"])
        p_normal = float(row["p_normal_raw"])
        true_confidence = p_defect if y_true == 1 else p_normal
        wrong_confidence = 1.0 - true_confidence
        enriched = dict(row)
        enriched.update(
            {
                "oof_fold": f"{job.fold:02d}",
                "human_fold": str(job.fold + 1),
                "fold_run_dir": str(job.run_dir),
                "weights": str(job.weights),
                "true_confidence_raw": f"{true_confidence:.10f}",
                "wrong_confidence_raw": f"{wrong_confidence:.10f}",
                "difficulty_bucket_raw": difficulty_bucket(wrong_confidence),
            }
        )
        enriched_rows.append(enriched)
    return enriched_rows


def summarize_difficulty(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summary: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        key = (row["human_fold"], row["difficulty_bucket_raw"])
        item = summary.setdefault(key, {"count": 0, "defect": 0, "normal": 0})
        item["count"] += 1
        if row["y_true"] == "1":
            item["defect"] += 1
        else:
            item["normal"] += 1

    output: list[dict[str, str]] = []
    bucket_order = {
        "confidently_wrong": 0,
        "wrong_not_confident": 1,
        "decision_boundary": 2,
        "correct_not_confident": 3,
        "confidently_correct": 4,
    }
    for (human_fold, bucket), counts in sorted(summary.items(), key=lambda item: (int(item[0][0]), bucket_order[item[0][1]])):
        output.append(
            {
                "human_fold": human_fold,
                "difficulty_bucket_raw": bucket,
                "count": str(counts["count"]),
                "defect": str(counts["defect"]),
                "normal": str(counts["normal"]),
            }
        )
    return output


def write_histogram(rows: list[dict[str, str]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    values = [float(row["wrong_confidence_raw"]) for row in rows]
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=180)
    ax.hist(values, bins=np.linspace(0, 1, 51), color="#5975a4", alpha=0.82, edgecolor="white", linewidth=0.7)
    ax.axvspan(0.4, 0.6, color="#f2c14e", alpha=0.28, label="decision boundary 0.4-0.6")
    ax.axvspan(0.9, 1.0, color="#d1495b", alpha=0.18, label="confidently wrong >=0.9")
    ax.axvline(0.5, color="#333333", linestyle="--", linewidth=1.1)
    ax.set_title("Stage-1 OOF Sample Difficulty Distribution", fontsize=14, pad=14)
    ax.set_xlabel("wrong_confidence_raw = 1 - raw confidence assigned to the true label")
    ax.set_ylabel("image count")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper center", ncols=2, frameon=False)

    total = len(values)
    boundary = sum(0.4 <= value <= 0.6 for value in values)
    confidently_wrong = sum(value >= 0.9 for value in values)
    wrong_side = sum(value > 0.5 for value in values)
    box = (
        f"total={total:,}\n"
        f"wrong side >0.5={wrong_side:,} ({wrong_side / total:.2%})\n"
        f"boundary 0.4-0.6={boundary:,} ({boundary / total:.2%})\n"
        f"confidently wrong >=0.9={confidently_wrong:,} ({confidently_wrong / total:.2%})"
    )
    ax.text(
        0.985,
        0.56,
        box,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#c8c8c8", alpha=0.92),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)


def write_artifact_manifest_for_output(output_root: Path) -> None:
    rows = collect_artifact_rows(output_root)
    write_csv(output_root / "artifact_manifest.csv", rows, ("relative_path", "size_bytes", "sha256", "modified_at_local"))
    write_json(
        output_root / "artifact_manifest.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "output_root": str(output_root),
            "artifact_count": len(rows),
            "artifacts": rows,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", required=True, help="Fold list/range, e.g. 0-7 with --fold-base 0 or 1-8 with --fold-base 1.")
    parser.add_argument("--fold-base", type=int, choices=(0, 1), default=0, help="Interpret --folds as zero-based or human one-based.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--oof-root", type=Path, default=DEFAULT_OOF_ROOT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--yolo-root", type=Path, default=DEFAULT_YOLO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and write run_config.json, but do not predict.")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    folds = parse_fold_spec(args.folds, base=args.fold_base)
    dataset_root = resolve_path(args.dataset_root, root).resolve()
    oof_root = resolve_path(args.oof_root, root).resolve()
    runs_root = resolve_path(args.runs_root, root).resolve()
    yolo_root = resolve_path(args.yolo_root, root).resolve()
    output_root = resolve_path(args.output_root, root).resolve()

    if output_root.exists() and any(output_root.iterdir()) and not args.exist_ok:
        raise FileExistsError(f"Output root already exists and is not empty: {output_root}. Use --exist-ok to append/rewrite.")
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = build_fold_jobs(folds=folds, oof_root=oof_root, runs_root=runs_root)
    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(root),
        "dataset_root": str(dataset_root),
        "oof_root": str(oof_root),
        "runs_root": str(runs_root),
        "yolo_root": str(yolo_root),
        "output_root": str(output_root),
        "folds_zero_based": [job.fold for job in jobs],
        "folds_human": [job.fold + 1 for job in jobs],
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "dry_run": args.dry_run,
        "jobs": [
            {
                "fold": job.fold,
                "human_fold": job.fold + 1,
                "manifest_dir": str(job.manifest_dir),
                "run_dir": str(job.run_dir),
                "weights": str(job.weights),
                "weights_sha256": file_sha256(job.weights),
            }
            for job in jobs
        ],
    }
    write_json(output_root / "run_config.json", run_config)

    print(f"output_root={output_root}")
    for job in jobs:
        print(f"fold_{job.fold:02d} human_fold={job.fold + 1} weights={job.weights}")

    if args.dry_run:
        write_artifact_manifest_for_output(output_root)
        print("dry_run=true; prediction skipped")
        return 0

    YOLO = ensure_yolo_import(yolo_root)
    all_rows: list[dict[str, str]] = []
    started = time.time()

    for job in jobs:
        records = load_fold_holdout_records(job, dataset_root)
        cfg = EvalConfig(
            weights=job.weights,
            run_name=f"oof_fold_{job.fold:02d}",
            splits=("oof_holdout",),
            seed=20260606,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            limit_per_class=None,
            target_recall=0.995,
            deployment_defect_prevalence=0.10,
            dry_run=False,
            exist_ok=True,
        )
        print(f"predict fold_{job.fold:02d} human_fold={job.fold + 1} images={len(records)}")
        model = YOLO(str(job.weights))
        fold_rows = add_difficulty_columns(predict_records(model, records, cfg), job)
        write_csv(output_root / f"predictions_fold_{job.fold:02d}.csv", fold_rows, OOF_PREDICTION_COLUMNS)
        all_rows.extend(fold_rows)

    write_csv(output_root / "oof_predictions_merged.csv", all_rows, OOF_PREDICTION_COLUMNS)
    write_csv(
        output_root / "difficulty_summary.csv",
        summarize_difficulty(all_rows),
        ("human_fold", "difficulty_bucket_raw", "count", "defect", "normal"),
    )
    if not args.no_plot:
        write_histogram(all_rows, output_root / "wrong_confidence_hist.png")

    write_artifact_manifest_for_output(output_root)
    print(f"merged_predictions={output_root / 'oof_predictions_merged.csv'}")
    print(f"difficulty_summary={output_root / 'difficulty_summary.csv'}")
    print(f"duration_sec={time.time() - started:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
