from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable


SEED = 20260606
CLASS_NAMES = ("no_target", "target_defect")
MODEL_KEYS = ("n", "s", "m", "l", "x")
MODEL_WEIGHTS = {
    "n": "yolo11n-cls.pt",
    "s": "yolo11s-cls.pt",
    "m": "yolo11m-cls.pt",
    "l": "yolo11l-cls.pt",
    "x": "yolo11x-cls.pt",
}

SMOKE_TRAIN_PER_CLASS = 96
SMOKE_VAL_PER_CLASS = 48
SMOKE_EPOCHS = 2
FULL_EPOCHS = 200
DEFAULT_IMGSZ = 224


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    yolo_root: Path
    dataset_root: Path
    manifest_dir: Path
    work_root: Path
    runs_root: Path


@dataclass(frozen=True)
class TrainConfig:
    mode: str
    models: tuple[str, ...]
    seed: int
    epochs: int
    imgsz: int
    batch: int
    workers: int
    device: str
    rebuild_data: bool
    train_per_class: int | None
    val_per_class: int | None
    dry_run: bool
    exist_ok: bool


@dataclass(frozen=True)
class DatasetCounts:
    train_no_target: int
    train_target_defect: int
    val_no_target: int
    val_target_defect: int


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def build_paths(args: argparse.Namespace) -> Paths:
    repo_root = repo_root_from_script()
    yolo_root = Path(args.yolo_root).resolve() if args.yolo_root else repo_root / "YOLOv11"
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else repo_root / "data" / "final_sewerml_dataset"
    manifest_dir = dataset_root / "manifests"
    work_root = Path(args.work_root).resolve() if args.work_root else repo_root / "data" / "stage1_cls_workdir"
    runs_root = Path(args.runs_root).resolve() if args.runs_root else yolo_root / "runs" / "stage1_cls_sweep"
    return Paths(
        repo_root=repo_root,
        yolo_root=yolo_root,
        dataset_root=dataset_root,
        manifest_dir=manifest_dir,
        work_root=work_root,
        runs_root=runs_root,
    )


def parse_models(value: str | None, mode: str) -> tuple[str, ...]:
    raw = value or os.environ.get("STAGE1_MODELS")
    if not raw:
        return ("n",) if mode == "smoke" else MODEL_KEYS
    models = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    unknown = [m for m in models if m not in MODEL_KEYS]
    if unknown:
        raise ValueError(f"Unknown model key(s): {unknown}. Valid keys: {MODEL_KEYS}")
    return models


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def canonical_image_path(row: dict[str, str], dataset_root: Path) -> Path:
    rel = row.get("canonical_image_relpath", "")
    if rel:
        candidate = dataset_root / Path(rel)
        if candidate.exists():
            return candidate
    source = row.get("source_image_path", "")
    if source:
        candidate = Path(source)
        if candidate.exists():
            return candidate
    filename = row.get("Filename", "<missing filename>")
    raise FileNotFoundError(f"Image not found for {filename}")


def choose_rows(rows: list[dict[str, str]], count: int | None, seed: int, salt: str) -> list[dict[str, str]]:
    if count is None or count >= len(rows):
        return list(rows)
    rng = random.Random(f"{seed}:{salt}")
    return rng.sample(rows, count)


def assert_safe_generated_path(path: Path, allowed_root: Path) -> None:
    path = path.resolve()
    allowed_root = allowed_root.resolve()
    if path == allowed_root:
        raise ValueError(f"Refusing to remove work root itself: {path}")
    if allowed_root not in path.parents:
        raise ValueError(f"Refusing to remove path outside generated work root: {path}")


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def populate_class_dir(rows: list[dict[str, str]], class_dir: Path, dataset_root: Path) -> str:
    methods: set[str] = set()
    seen: set[str] = set()
    for row in rows:
        filename = row["Filename"]
        if filename in seen:
            raise ValueError(f"Duplicate filename in selected rows: {filename}")
        seen.add(filename)
        src = canonical_image_path(row, dataset_root)
        method = link_or_copy(src, class_dir / filename)
        methods.add(method)
    return "+".join(sorted(methods))


def prepare_cls_dataset(paths: Paths, cfg: TrainConfig) -> tuple[Path, DatasetCounts]:
    train_target_rows = read_manifest(paths.manifest_dir / "train_manifest.csv")
    train_no_target_rows = read_manifest(paths.manifest_dir / "normal_train_manifest.csv")
    val_target_rows = read_manifest(paths.manifest_dir / "val_model_manifest.csv")
    val_no_target_rows = read_manifest(paths.manifest_dir / "normal_val_model_manifest.csv")

    if cfg.mode == "smoke":
        train_target_rows = choose_rows(train_target_rows, cfg.train_per_class, cfg.seed, "smoke-train-target")
        train_no_target_rows = choose_rows(train_no_target_rows, cfg.train_per_class, cfg.seed, "smoke-train-no-target")
        val_target_rows = choose_rows(val_target_rows, cfg.val_per_class, cfg.seed, "smoke-val-target")
        val_no_target_rows = choose_rows(val_no_target_rows, cfg.val_per_class, cfg.seed, "smoke-val-no-target")

    dataset_dir = paths.work_root / cfg.mode
    if cfg.rebuild_data and dataset_dir.exists():
        assert_safe_generated_path(dataset_dir, paths.work_root)
        shutil.rmtree(dataset_dir)

    split_rows = {
        ("train", "target_defect"): train_target_rows,
        ("train", "no_target"): train_no_target_rows,
        ("val", "target_defect"): val_target_rows,
        ("val", "no_target"): val_no_target_rows,
    }
    for (split, class_name), rows in split_rows.items():
        method = populate_class_dir(rows, dataset_dir / split / class_name, paths.dataset_root)
        print(f"prepared {split}/{class_name}: {len(rows)} images ({method})")

    counts = DatasetCounts(
        train_no_target=len(train_no_target_rows),
        train_target_defect=len(train_target_rows),
        val_no_target=len(val_no_target_rows),
        val_target_defect=len(val_target_rows),
    )
    return dataset_dir, counts


def import_local_ultralytics(yolo_root: Path):
    if not yolo_root.exists():
        raise FileNotFoundError(f"Missing YOLOv11 root: {yolo_root}")
    sys.path.insert(0, str(yolo_root))
    from ultralytics import YOLO  # noqa: PLC0415

    return YOLO


def weight_path(paths: Paths, model_key: str) -> Path:
    filename = MODEL_WEIGHTS[model_key]
    candidates = [
        paths.repo_root / filename,
        paths.yolo_root / filename,
        paths.yolo_root / "weights" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing model weight {filename}; checked: {candidates}")


def summary_path(paths: Paths, mode: str) -> Path:
    name = "smoke_summary.csv" if mode == "smoke" else "summary.csv"
    return paths.runs_root / name


def append_summary(paths: Paths, cfg: TrainConfig, counts: DatasetCounts, row: dict[str, str]) -> None:
    path = summary_path(paths, cfg.mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp",
        "mode",
        "model_key",
        "model_weight",
        "epochs",
        "imgsz",
        "batch",
        "workers",
        "device",
        "seed",
        "dataset_dir",
        "train_no_target",
        "train_target_defect",
        "val_no_target",
        "val_target_defect",
        "run_dir",
        "best_pt_exists",
        "last_pt_exists",
        "results_csv_exists",
        "args_yaml_exists",
        "status",
        "error",
        "duration_sec",
    ]
    full_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": cfg.mode,
        "epochs": str(cfg.epochs),
        "imgsz": str(cfg.imgsz),
        "batch": str(cfg.batch),
        "workers": str(cfg.workers),
        "device": cfg.device,
        "seed": str(cfg.seed),
        "train_no_target": str(counts.train_no_target),
        "train_target_defect": str(counts.train_target_defect),
        "val_no_target": str(counts.val_no_target),
        "val_target_defect": str(counts.val_target_defect),
        **row,
    }
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(full_row)


def train_one_model(model_key: str, paths: Paths, cfg: TrainConfig, dataset_dir: Path, counts: DatasetCounts) -> Path:
    started = time.time()
    model_weight = weight_path(paths, model_key)
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{cfg.mode}_yolo11{model_key}_cls_{run_stamp}"
    run_dir = paths.runs_root / run_name
    status = "failed"
    error = ""
    try:
        if cfg.dry_run:
            print(f"[dry-run] would train {model_key} with {model_weight}")
            status = "dry_run"
        else:
            YOLO = import_local_ultralytics(paths.yolo_root)
            model = YOLO(str(model_weight))
            model.train(
                data=str(dataset_dir),
                epochs=cfg.epochs,
                imgsz=cfg.imgsz,
                batch=cfg.batch,
                workers=cfg.workers,
                device=cfg.device,
                project=str(paths.runs_root),
                name=run_name,
                exist_ok=cfg.exist_ok,
                seed=cfg.seed,
                deterministic=True,
                cache=False,
                val=True,
                plots=True,
                verbose=True,
                task="classify",
            )
            status = "ok"
    except Exception as exc:  # keep summary even when smoke exposes a bug
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        print(traceback.format_exc())
        raise
    finally:
        duration = time.time() - started
        append_summary(
            paths,
            cfg,
            counts,
            {
                "model_key": model_key,
                "model_weight": str(model_weight),
                "dataset_dir": str(dataset_dir),
                "run_dir": str(run_dir),
                "best_pt_exists": str((run_dir / "weights" / "best.pt").exists()),
                "last_pt_exists": str((run_dir / "weights" / "last.pt").exists()),
                "results_csv_exists": str((run_dir / "results.csv").exists()),
                "args_yaml_exists": str((run_dir / "args.yaml").exists()),
                "status": status,
                "error": error,
                "duration_sec": f"{duration:.2f}",
            },
        )
    return run_dir


def train_yolo11n_cls(paths: Paths, cfg: TrainConfig, dataset_dir: Path, counts: DatasetCounts) -> Path:
    return train_one_model("n", paths, cfg, dataset_dir, counts)


def train_yolo11s_cls(paths: Paths, cfg: TrainConfig, dataset_dir: Path, counts: DatasetCounts) -> Path:
    return train_one_model("s", paths, cfg, dataset_dir, counts)


def train_yolo11m_cls(paths: Paths, cfg: TrainConfig, dataset_dir: Path, counts: DatasetCounts) -> Path:
    return train_one_model("m", paths, cfg, dataset_dir, counts)


def train_yolo11l_cls(paths: Paths, cfg: TrainConfig, dataset_dir: Path, counts: DatasetCounts) -> Path:
    return train_one_model("l", paths, cfg, dataset_dir, counts)


def train_yolo11x_cls(paths: Paths, cfg: TrainConfig, dataset_dir: Path, counts: DatasetCounts) -> Path:
    return train_one_model("x", paths, cfg, dataset_dir, counts)


TRAINERS: dict[str, Callable[[Paths, TrainConfig, Path, DatasetCounts], Path]] = {
    "n": train_yolo11n_cls,
    "s": train_yolo11s_cls,
    "m": train_yolo11m_cls,
    "l": train_yolo11l_cls,
    "x": train_yolo11x_cls,
}


def run_selected_models(paths: Paths, cfg: TrainConfig) -> list[Path]:
    dataset_dir, counts = prepare_cls_dataset(paths, cfg)
    print(f"dataset_dir={dataset_dir}")
    print(f"counts={counts}")

    run_dirs = []
    for model_key in cfg.models:
        print(f"=== train yolo11{model_key}-cls ({cfg.mode}) ===")
        run_dirs.append(TRAINERS[model_key](paths, cfg, dataset_dir, counts))
    return run_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 YOLO11-cls binary training sweep.")
    parser.add_argument("--mode", choices=("smoke", "full"), default=os.environ.get("STAGE1_MODE", "smoke"))
    parser.add_argument("--models", default=None, help="Comma-separated model keys, e.g. n,s,m or l,x.")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("STAGE1_SEED", SEED)))
    parser.add_argument("--epochs", type=int, default=None, help="Defaults to 2 for smoke and 200 for full.")
    parser.add_argument("--imgsz", type=int, default=int(os.environ.get("STAGE1_IMGSZ", DEFAULT_IMGSZ)))
    parser.add_argument("--batch", type=int, default=int(os.environ.get("STAGE1_BATCH", 8)))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("STAGE1_WORKERS", 0)))
    parser.add_argument("--device", default=os.environ.get("STAGE1_DEVICE", "0"))
    parser.add_argument("--train-per-class", type=int, default=None)
    parser.add_argument("--val-per-class", type=int, default=None)
    parser.add_argument("--keep-data", action="store_true", help="Reuse generated classification workdir.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare data and write dry-run summary without training.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow Ultralytics to reuse an existing run name.")
    parser.add_argument("--dataset-root", default=os.environ.get("STAGE1_DATASET_ROOT"))
    parser.add_argument("--work-root", default=os.environ.get("STAGE1_WORK_ROOT"))
    parser.add_argument("--runs-root", default=os.environ.get("STAGE1_RUNS_ROOT"))
    parser.add_argument("--yolo-root", default=os.environ.get("STAGE1_YOLO_ROOT"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = parse_models(args.models, args.mode)
    epochs = args.epochs if args.epochs is not None else (SMOKE_EPOCHS if args.mode == "smoke" else FULL_EPOCHS)
    train_per_class = args.train_per_class
    val_per_class = args.val_per_class
    if args.mode == "smoke":
        train_per_class = train_per_class or SMOKE_TRAIN_PER_CLASS
        val_per_class = val_per_class or SMOKE_VAL_PER_CLASS
    cfg = TrainConfig(
        mode=args.mode,
        models=models,
        seed=args.seed,
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        rebuild_data=not args.keep_data,
        train_per_class=train_per_class,
        val_per_class=val_per_class,
        dry_run=args.dry_run,
        exist_ok=args.exist_ok,
    )
    paths = build_paths(args)
    print(f"repo_root={paths.repo_root}")
    print(f"yolo_root={paths.yolo_root}")
    print(f"dataset_root={paths.dataset_root}")
    print(f"runs_root={paths.runs_root}")
    print(f"mode={cfg.mode} models={','.join(cfg.models)} epochs={cfg.epochs}")
    run_dirs = run_selected_models(paths, cfg)
    print("completed runs:")
    for run_dir in run_dirs:
        print(run_dir)
    print(f"summary={summary_path(paths, cfg.mode)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
