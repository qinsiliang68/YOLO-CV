# -*- coding: utf-8 -*-
"""Continue the interrupted Stage-1 OOF jobs on the Windows training nodes.

This script is intentionally node-specific. It knows the emergency recovery
state from 2026-06-19 and runs the remaining sequence after validating that
training inputs are on C and run outputs resolve to a non-C disk.
Run it from the repository root on node 13 or node 18.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


TARGET_EPOCHS = 200
DEFAULT_SAVE_PERIOD = 1
DEFAULT_TRAIN_PYTHON = Path(r"C:\Users\ASUS\Desktop\ssh\AI\venvs\yolo-cv\Scripts\python.exe")
DEFAULT_DATASET_ROOT = Path(r"C:\Users\ASUS\Desktop\ssh\AI\datasets\final_sewerml_dataset")
DEFAULT_RUNS_ROOT = Path("YOLOv11") / "runs" / "stage1_oof_10fold"
DEFAULT_YOLO_ROOT = Path("YOLOv11")


@dataclass(frozen=True)
class NodePlan:
    node: str
    interrupted_fold: int
    interrupted_run: str
    remaining_folds: tuple[int, ...]
    work_root: Path


NODE_PLANS = {
    "13": NodePlan(
        node="13",
        interrupted_fold=5,
        interrupted_run="full_yolo11l_cls_20260618-075627",
        remaining_folds=(6, 7),
        work_root=Path(r"C:\Users\ASUS\Desktop\ssh\AI\workdirs\stage1_oof_workdir"),
    ),
    "18": NodePlan(
        node="18",
        interrupted_fold=1,
        interrupted_run="full_yolo11l_cls_20260618-082829",
        remaining_folds=(2, 3),
        work_root=Path(r"C:\Users\ASUS\Desktop\ssh\AI\workdirs\stage1_oof_workdir_cdrive"),
    ),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def real_path(path: Path) -> Path:
    return Path(os.path.realpath(path))


def drive_upper(path: Path) -> str:
    return real_path(path).drive.upper()


def require_drive(path: Path, drive: str, description: str) -> None:
    actual = drive_upper(path)
    if actual != drive.upper():
        raise RuntimeError(f"{description} must resolve to {drive.upper()}, got {actual}: {path} -> {real_path(path)}")


def require_not_drive(path: Path, drive: str, description: str) -> None:
    actual = drive_upper(path)
    if actual == drive.upper():
        raise RuntimeError(f"{description} must not resolve to {drive.upper()}: {path} -> {real_path(path)}")


def result_epoch(run_dir: Path) -> int | None:
    results = run_dir / "results.csv"
    if not results.exists():
        return None
    last_epoch: int | None = None
    with results.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or row[0].strip().lower() in {"epoch", "epochs"}:
                continue
            try:
                last_epoch = int(float(row[0]))
            except ValueError:
                continue
    return last_epoch


def fold_dir(root: Path, fold: int) -> Path:
    return root / f"fold_{fold:02d}"


def run_dirs_for_fold(runs_root: Path, fold: int) -> list[Path]:
    folder = fold_dir(runs_root, fold)
    if not folder.exists():
        return []
    return sorted(
        [item for item in folder.iterdir() if item.is_dir() and item.name != "_logs"],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )


def fold_complete(runs_root: Path, fold: int) -> bool:
    for run_dir in run_dirs_for_fold(runs_root, fold):
        epoch = result_epoch(run_dir)
        weights = run_dir / "weights"
        if epoch is not None and epoch >= TARGET_EPOCHS and (weights / "best.pt").exists() and (weights / "last.pt").exists():
            return True
    return False


def print_fold_status(runs_root: Path, folds: tuple[int, ...]) -> None:
    print("=== fold status ===", flush=True)
    for fold in folds:
        dirs = run_dirs_for_fold(runs_root, fold)
        if not dirs:
            print(f"fold_{fold:02d}: no run directory", flush=True)
            continue
        for run_dir in dirs[:3]:
            weights = run_dir / "weights"
            print(
                f"fold_{fold:02d}: {run_dir.name} "
                f"epoch={result_epoch(run_dir)} "
                f"best={(weights / 'best.pt').exists()} "
                f"last={(weights / 'last.pt').exists()}",
                flush=True,
            )


def validate_storage_layout(*, dataset_root: Path, work_root: Path, runs_root: Path) -> None:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Missing dataset root: {dataset_root}")
    if not runs_root.exists():
        raise FileNotFoundError(f"Missing runs root: {runs_root}")
    work_root.mkdir(parents=True, exist_ok=True)

    require_drive(dataset_root, "C:", "dataset_root")
    require_drive(work_root, "C:", "work_root")
    require_not_drive(runs_root, "C:", "runs_root")


def prepare_fold_workdir(
    *,
    root: Path,
    fold: int,
    dataset_root: Path,
    work_root: Path,
    runs_root: Path,
    yolo_root: Path,
    save_period: int,
    device: str,
    workers: int,
) -> None:
    dataset_dir = work_root / f"fold_{fold:02d}" / "full"
    if dataset_dir.exists():
        print(f"fold_{fold:02d} workdir already exists: {dataset_dir}", flush=True)
        return

    sys.path.insert(0, str(root))
    from scripts.train_stage1_cls_sweep import Paths, TrainConfig, prepare_cls_dataset  # noqa: PLC0415

    manifest_dir = (
        root
        / "artifacts"
        / "stage1_oof_folds_10fold_20260617"
        / "folds"
        / f"fold_{fold:02d}"
        / "manifests"
    )
    paths = Paths(
        repo_root=root,
        yolo_root=yolo_root,
        dataset_root=dataset_root,
        manifest_dir=manifest_dir,
        work_root=work_root / f"fold_{fold:02d}",
        runs_root=runs_root / f"fold_{fold:02d}",
    )
    cfg = TrainConfig(
        mode="full",
        models=("l",),
        seed=20260606,
        epochs=TARGET_EPOCHS,
        imgsz=224,
        batch=128,
        workers=workers,
        save_period=save_period,
        device=device,
        rebuild_data=True,
        train_per_class=None,
        val_per_class=None,
        dry_run=True,
        exist_ok=False,
    )
    print(f"preparing fold_{fold:02d} workdir on C: {dataset_dir}", flush=True)
    prepare_cls_dataset(paths, cfg)


def detect_node(root: Path) -> str:
    runs_root = root / DEFAULT_RUNS_ROOT
    has_18_work = (root / "data" / "stage1_oof_workdir_cdrive").exists()
    has_13_work = (root / "data" / "stage1_oof_workdir").exists()
    has_18_runs = fold_dir(runs_root, 0).exists() or fold_dir(runs_root, 1).exists()
    has_13_runs = fold_dir(runs_root, 4).exists() or fold_dir(runs_root, 5).exists()

    if has_18_runs and not has_13_runs:
        return "18"
    if has_13_runs and not has_18_runs:
        return "13"
    if has_18_work and not has_13_work:
        return "18"
    if has_13_work and not has_18_work:
        return "13"
    raise RuntimeError("Could not auto-detect node. Re-run with --node 13 or --node 18.")


def run_command(command: list[str], *, cwd: Path) -> None:
    print("=== run ===", flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=cwd)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)


def resume_interrupted_fold(
    *,
    root: Path,
    plan: NodePlan,
    train_python: Path,
    runs_root: Path,
    device: str,
    workers: int,
    save_period: int,
    print_only: bool,
) -> None:
    fold = plan.interrupted_fold
    if fold_complete(runs_root, fold):
        print(f"fold_{fold:02d} already complete; skip resume.", flush=True)
        return

    checkpoint = fold_dir(runs_root, fold) / plan.interrupted_run / "weights" / "last.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing resume checkpoint: {checkpoint}")
    resume_data = resolve_path(plan.work_root, root) / f"fold_{fold:02d}" / "full"
    if not resume_data.exists() and not print_only:
        raise FileNotFoundError(f"Missing prepared resume data: {resume_data}")

    code = (
        "from pathlib import Path\n"
        "import os, sys\n"
        f"repo = Path(r'{root}')\n"
        "os.chdir(repo)\n"
        "sys.path.insert(0, str(repo / 'YOLOv11'))\n"
        "from ultralytics import YOLO\n"
        f"checkpoint = Path(r'{checkpoint}')\n"
        f"data = Path(r'{resume_data}')\n"
        "print(f'resuming from {checkpoint}', flush=True)\n"
        "print(f'resume data={data}', flush=True)\n"
        f"YOLO(str(checkpoint)).train(resume=True, data=str(data), save_period={save_period}, device={device!r}, workers={workers})\n"
    )
    command = [str(train_python), "-c", code]
    if print_only:
        print(subprocess.list2cmdline(command), flush=True)
        return
    run_command(command, cwd=root)
    if not fold_complete(runs_root, fold):
        raise RuntimeError(f"fold_{fold:02d} resume returned successfully but fold is not complete yet.")


def run_remaining_folds(
    *,
    root: Path,
    plan: NodePlan,
    wrapper_python: Path,
    dataset_root: Path,
    runs_root: Path,
    device: str,
    workers: int,
    save_period: int,
    print_only: bool,
) -> None:
    pending = tuple(fold for fold in plan.remaining_folds if not fold_complete(runs_root, fold))
    if not pending:
        print("remaining folds already complete; nothing to run.", flush=True)
        return

    command = [
        str(wrapper_python),
        "scripts\\run_stage1_oof_folds_20260617.py",
        "--folds",
        ",".join(str(fold) for fold in pending),
        "--dataset-root",
        str(dataset_root),
        "--work-root",
        str(resolve_path(plan.work_root, root)),
        "--save-period",
        str(save_period),
        "--device",
        device,
        "--workers",
        str(workers),
    ]
    if print_only:
        print(subprocess.list2cmdline(command), flush=True)
        return
    run_command(command, cwd=root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", choices=("auto", "13", "18"), default="auto")
    parser.add_argument("--train-python", type=Path, default=DEFAULT_TRAIN_PYTHON)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--yolo-root", type=Path, default=DEFAULT_YOLO_ROOT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-period", type=int, default=DEFAULT_SAVE_PERIOD)
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    node = detect_node(root) if args.node == "auto" else args.node
    plan = NODE_PLANS[node]
    train_python = resolve_path(args.train_python, root)
    wrapper_python = Path(sys.executable)
    dataset_root = resolve_path(args.dataset_root, root)
    runs_root = resolve_path(args.runs_root, root)
    yolo_root = resolve_path(args.yolo_root, root)
    work_root = resolve_path(plan.work_root, root)

    print(f"repo_root={root}", flush=True)
    print(f"detected_node={node}", flush=True)
    print(f"dataset_root={dataset_root}", flush=True)
    print(f"dataset_root_real={real_path(dataset_root)}", flush=True)
    print(f"work_root={work_root}", flush=True)
    print(f"work_root_real={real_path(work_root)}", flush=True)
    print(f"runs_root={runs_root}", flush=True)
    print(f"runs_root_real={real_path(runs_root)}", flush=True)
    print(f"save_period={args.save_period}", flush=True)
    print_fold_status(runs_root, (plan.interrupted_fold, *plan.remaining_folds))

    if not train_python.exists():
        raise FileNotFoundError(f"Missing training Python: {train_python}")
    validate_storage_layout(dataset_root=dataset_root, work_root=work_root, runs_root=runs_root)
    if args.print_only:
        print(f"would prepare fold_{plan.interrupted_fold:02d} workdir before resume", flush=True)
    else:
        prepare_fold_workdir(
            root=root,
            fold=plan.interrupted_fold,
            dataset_root=dataset_root,
            work_root=work_root,
            runs_root=runs_root,
            yolo_root=yolo_root,
            save_period=args.save_period,
            device=args.device,
            workers=args.workers,
        )

    resume_interrupted_fold(
        root=root,
        plan=plan,
        train_python=train_python,
        runs_root=runs_root,
        device=args.device,
        workers=args.workers,
        save_period=args.save_period,
        print_only=args.print_only,
    )
    run_remaining_folds(
        root=root,
        plan=plan,
        wrapper_python=wrapper_python,
        dataset_root=dataset_root,
        runs_root=runs_root,
        device=args.device,
        workers=args.workers,
        save_period=args.save_period,
        print_only=args.print_only,
    )
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
