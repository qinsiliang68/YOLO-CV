# -*- coding: utf-8 -*-
"""Run Stage-1 OOF fold training jobs.

This wrapper keeps the original training script immutable while making the
machine commands short and repeatable. It runs folds sequentially.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


DEFAULT_TRAIN_PYTHON = Path(r"C:\Users\ASUS\Desktop\ssh\AI\projects\YOLO-CV\.venv\Scripts\python.exe")
DEFAULT_BOOTSTRAP_SOURCE_ROOT = Path(r"C:\Users\ASUS\Desktop\ssh\AI\projects\YOLO-CV")
DEFAULT_OOF_ROOT = Path("artifacts") / "stage1_oof_folds_10fold_20260617"
DEFAULT_WORK_ROOT = Path("data") / "stage1_oof_workdir"
DEFAULT_RUNS_ROOT = Path("YOLOv11") / "runs" / "stage1_oof_10fold"
DEFAULT_YOLO_ROOT = Path("YOLOv11")
YOLO_ASSET_FILES = ("bus.jpg", "zidane.jpg")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def parse_folds(value: str) -> list[int]:
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
    unique_folds = []
    seen = set()
    for fold in folds:
        if fold < 0 or fold > 9:
            raise ValueError(f"Fold out of range 0..9: {fold}")
        if fold not in seen:
            unique_folds.append(fold)
            seen.add(fold)
    if not unique_folds:
        raise ValueError("No folds selected")
    return unique_folds


def model_keys(models: str) -> list[str]:
    keys = [item.strip() for item in models.split(",") if item.strip()]
    if not keys:
        raise ValueError("No models selected")
    for key in keys:
        if key not in {"n", "s", "m", "l", "x"}:
            raise ValueError(f"Unsupported model key: {key}")
    return keys


def ensure_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def copy_if_missing(src: Path, dst: Path, description: str) -> None:
    if dst.exists():
        return
    ensure_file(src, description)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"bootstrapped {description}: {dst}")


def bootstrap_runtime_files(
    *,
    root: Path,
    yolo_root: Path,
    source_root: Path,
    models: list[str],
) -> None:
    for key in models:
        filename = f"yolo11{key}-cls.pt"
        copy_if_missing(source_root / filename, root / filename, filename)

    source_assets = source_root / "YOLOv11" / "ultralytics" / "assets"
    target_assets = yolo_root / "ultralytics" / "assets"
    for filename in YOLO_ASSET_FILES:
        copy_if_missing(source_assets / filename, target_assets / filename, f"YOLO asset {filename}")


def build_command(
    *,
    train_python: Path,
    train_script: Path,
    args: argparse.Namespace,
    dataset_root: Path,
    manifest_dir: Path,
    work_root: Path,
    runs_root: Path,
    yolo_root: Path,
) -> list[str]:
    command = [
        str(train_python),
        str(train_script),
        "--mode",
        args.mode,
        "--models",
        args.models,
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--save-period",
        str(args.save_period),
        "--device",
        args.device,
        "--dataset-root",
        str(dataset_root),
        "--manifest-dir",
        str(manifest_dir),
        "--work-root",
        str(work_root),
        "--runs-root",
        str(runs_root),
        "--yolo-root",
        str(yolo_root),
    ]
    if args.keep_data:
        command.append("--keep-data")
    if args.exist_ok:
        command.append("--exist-ok")
    if args.train_per_class is not None:
        command.extend(["--train-per-class", str(args.train_per_class)])
    if args.val_per_class is not None:
        command.extend(["--val-per-class", str(args.val_per_class)])
    if args.training_dry_run:
        command.append("--dry-run")
    return command


def validate_fold_manifests(oof_root: Path, folds: list[int]) -> None:
    required = (
        "train_manifest.csv",
        "normal_train_manifest.csv",
        "val_model_manifest.csv",
        "normal_val_model_manifest.csv",
    )
    for fold in folds:
        manifest_dir = oof_root / "folds" / f"fold_{fold:02d}" / "manifests"
        for filename in required:
            ensure_file(manifest_dir / filename, f"fold_{fold:02d} manifest {filename}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", required=True, help="Fold list/range, e.g. 0-3 or 4,5,6,7.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--models", default="l")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--train-per-class", type=int, default=None)
    parser.add_argument("--val-per-class", type=int, default=None)
    parser.add_argument("--train-python", type=Path, default=DEFAULT_TRAIN_PYTHON)
    parser.add_argument("--bootstrap-source-root", type=Path, default=DEFAULT_BOOTSTRAP_SOURCE_ROOT)
    parser.add_argument("--oof-root", type=Path, default=DEFAULT_OOF_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--yolo-root", type=Path, default=DEFAULT_YOLO_ROOT)
    parser.add_argument("--keep-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap-runtime-files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--training-dry-run", action="store_true", help="Pass --dry-run to train_stage1_cls_sweep.py.")
    parser.add_argument("--print-only", action="store_true", help="Print the fold commands without executing them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    folds = parse_folds(args.folds)
    keys = model_keys(args.models)

    train_script = root / "scripts" / "train_stage1_cls_sweep.py"
    train_python = resolve_path(args.train_python, root)
    dataset_root = resolve_path(Path(args.dataset_root), root)
    oof_root = resolve_path(args.oof_root, root)
    work_root_base = resolve_path(args.work_root, root)
    runs_root_base = resolve_path(args.runs_root, root)
    yolo_root = resolve_path(args.yolo_root, root)
    source_root = resolve_path(args.bootstrap_source_root, root)

    ensure_file(train_script, "training script")
    ensure_file(train_python, "training Python")
    ensure_file(dataset_root / "manifests" / "train_manifest.csv", "dataset train manifest")
    ensure_file(oof_root / "fold_summary.csv", "OOF fold summary")
    validate_fold_manifests(oof_root, folds)

    if args.bootstrap_runtime_files and not args.print_only:
        bootstrap_runtime_files(root=root, yolo_root=yolo_root, source_root=source_root, models=keys)

    for fold in folds:
        fold_name = f"fold_{fold:02d}"
        manifest_dir = oof_root / "folds" / fold_name / "manifests"
        command = build_command(
            train_python=train_python,
            train_script=train_script,
            args=args,
            dataset_root=dataset_root,
            manifest_dir=manifest_dir,
            work_root=work_root_base / fold_name,
            runs_root=runs_root_base / fold_name,
            yolo_root=yolo_root,
        )
        print(f"=== {fold_name} ===")
        print(subprocess.list2cmdline(command))
        if args.print_only:
            continue
        completed = subprocess.run(command, cwd=root)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
