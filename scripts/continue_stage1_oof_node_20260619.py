# -*- coding: utf-8 -*-
"""Continue the interrupted Stage-1 OOF jobs on the Windows training nodes.

This script is intentionally node-specific. It knows the emergency recovery
state from 2026-06-19 and runs the remaining safe sequence with save_period=-1.
Run it from the repository root on node 13 or node 18.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


TARGET_EPOCHS = 200
DEFAULT_TRAIN_PYTHON = Path(r"C:\Users\ASUS\Desktop\ssh\AI\projects\YOLO-CV\.venv\Scripts\python.exe")
DEFAULT_DATASET_ROOT = Path(r"C:\Users\ASUS\Desktop\ssh\AI\datasets\final_sewerml_dataset")
DEFAULT_RUNS_ROOT = Path("YOLOv11") / "runs" / "stage1_oof_10fold"


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
        work_root=Path("data") / "stage1_oof_workdir",
    ),
    "18": NodePlan(
        node="18",
        interrupted_fold=1,
        interrupted_run="full_yolo11l_cls_20260618-082829",
        remaining_folds=(2, 3),
        work_root=Path("data") / "stage1_oof_workdir_cdrive",
    ),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


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
    print_only: bool,
) -> None:
    fold = plan.interrupted_fold
    if fold_complete(runs_root, fold):
        print(f"fold_{fold:02d} already complete; skip resume.", flush=True)
        return

    checkpoint = fold_dir(runs_root, fold) / plan.interrupted_run / "weights" / "last.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing resume checkpoint: {checkpoint}")

    code = (
        "from pathlib import Path\n"
        "import os, sys\n"
        f"repo = Path(r'{root}')\n"
        "os.chdir(repo)\n"
        "sys.path.insert(0, str(repo / 'YOLOv11'))\n"
        "from ultralytics import YOLO\n"
        f"checkpoint = Path(r'{checkpoint}')\n"
        "print(f'resuming from {checkpoint}', flush=True)\n"
        f"YOLO(str(checkpoint)).train(resume=True, save_period=-1, device={device!r}, workers={workers})\n"
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
        "-1",
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
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
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

    print(f"repo_root={root}", flush=True)
    print(f"detected_node={node}", flush=True)
    print(f"dataset_root={dataset_root}", flush=True)
    print(f"work_root={resolve_path(plan.work_root, root)}", flush=True)
    print(f"runs_root={runs_root}", flush=True)
    print_fold_status(runs_root, (plan.interrupted_fold, *plan.remaining_folds))

    if not train_python.exists():
        raise FileNotFoundError(f"Missing training Python: {train_python}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Missing dataset root: {dataset_root}")

    resume_interrupted_fold(
        root=root,
        plan=plan,
        train_python=train_python,
        runs_root=runs_root,
        device=args.device,
        workers=args.workers,
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
        print_only=args.print_only,
    )
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
