from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
YOLOV11_ROOT = REPO_ROOT / "YOLOv11"
DEFAULT_CONFIG = YOLOV11_ROOT / "configs" / "runtime" / "cls_cls6_sweep.json"
DEFAULT_ENTRY_CONFIG = YOLOV11_ROOT / "configs" / "runtime" / "main_entry.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
BUILTIN_DEFAULT_CONFIG = {
    "source_dataset": r"C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\data\sewerml_cls6_train7200",
    "project": r"C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\runs\cls_source_uniform",
    "recycle_root": r"C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\_recycle_bin\cls_source_uniform",
    "temp_config_dir": r"C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\$out\generated_configs\cls6_sweep",
    "models": [
        "yolo11n-cls.pt",
        "yolo11s-cls.pt",
        "yolo11m-cls.pt",
        "yolo11l-cls.pt",
        "yolo11x-cls.pt",
    ],
    "epochs": 100,
    "imgsz": 640,
    "batch": 32,
    "workers": 4,
    "device": "0",
    "pretrained": True,
    "patience": 20,
    "optimizer": "auto",
    "cache": False,
    "resume": False,
    "collect_batch": 16,
    "run_name_suffix": "cls6_train7200_uniform",
}
BUILTIN_STAGE1_ENTRY_CONFIG = {
    "task": "stage1_gate_s_hn",
    "score_device": "0",
    "top_k": 22,
    "score_batch": 1,
    "score_chunk_size": 32,
}
STAGE1_HN_TASKS = {
    "stage1_gate_l_hn": {
        "weights": r"C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\runs\cls_gate_source\yolo11l_gate2_train7200\weights\best.pt",
        "data_root": r"C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\datasets\sewerml_gate2_train7200",
        "output_dir": r"C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\research\materials\stage1_hn\yolo11l_gate2_train7200",
        "hn_dataset": r"C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\datasets\stage1_gate_hn_backflow\yolo11l_gate2_hn02",
        "train_config": r".\YOLOv11\configs\runtime\stage1_gate_l_hn.json",
        "label": "yolo11l-cls",
    },
    "stage1_gate_s_hn": {
        "weights": r"C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\runs\cls_gate_source\yolo11s_gate2_train7200\weights\best.pt",
        "data_root": r"C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\datasets\sewerml_gate2_train7200",
        "output_dir": r"C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\research\materials\stage1_hn\yolo11s_gate2_train7200",
        "hn_dataset": r"C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\datasets\stage1_gate_hn_backflow\yolo11s_gate2_hn02",
        "train_config": r".\YOLOv11\configs\runtime\stage1_gate_s_hn.json",
        "label": "yolo11s-cls",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified launcher for the current YOLO-CV training task."
    )
    parser.add_argument(
        "--config",
        default="",
        help="Config JSON for the selected task. For cls6_sweep this is the sweep config; otherwise the active entry config is used.",
    )
    parser.add_argument(
        "--task",
        choices=("auto", "cls6_sweep", "stage1_gate_l_hn", "stage1_gate_s_hn"),
        default="auto",
        help="Task to run. 'auto' reads YOLOv11/configs/runtime/main_entry.json.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without training.")
    parser.add_argument("--rerun", action="store_true", help="Archive existing runs and rerun all configured models.")
    return parser.parse_args()


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Config must be a JSON object: {path}")
    return payload


def load_sweep_config(path: Path) -> dict:
    config = dict(BUILTIN_DEFAULT_CONFIG)
    if path.exists():
        config.update(load_json(path))
        print_step("config", f"loaded sweep config: {path}")
    else:
        print_step("config", f"missing sweep config, using built-in defaults: {path}")
    return config


def load_entry_config(path: Path) -> dict:
    config = dict(BUILTIN_STAGE1_ENTRY_CONFIG)
    if path.exists():
        config.update(load_json(path))
        print_step("config", f"loaded entry config: {path}")
    else:
        print_step("config", f"missing entry config, using built-in defaults: {path}")
    return config


def resolve_path(value: str | None, *, base: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        return base.resolve()
    path = Path(text)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def resolve_str(value: object, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def run_python(script: str, args: list[str], dry_run: bool) -> None:
    cmd = [sys.executable, str(REPO_ROOT / script), *args]
    print_step("run", " ".join(f'"{part}"' if " " in part else part for part in cmd))
    if dry_run:
        return

    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".ultralytics"))
    if os.name == "nt":
        env.setdefault("PIN_MEMORY", "False")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)


def split_image_count(root: Path, split: str) -> int:
    split_root = root / split
    if not split_root.exists():
        return 0
    return sum(1 for path in split_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def ensure_source_dataset(source_root: Path) -> None:
    train_count = split_image_count(source_root, "train")
    val_count = split_image_count(source_root, "val")
    if train_count == 0 or val_count == 0:
        raise SystemExit(
            "Source six-class dataset is not ready.\n"
            f"Expected images under:\n  {source_root / 'train'}\n  {source_root / 'val'}"
        )
    print_step("data", f"source cls6 dataset ready: train={train_count} val={val_count}")


def archive_existing_run(run_dir: Path, recycle_root: Path, dry_run: bool) -> None:
    if not run_dir.exists():
        return
    destination = recycle_root / run_dir.name
    print_step("archive", f"{run_dir} -> {destination}")
    if dry_run:
        return
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(destination))


def build_run_name(model_name: str, suffix: str) -> str:
    stem = Path(model_name).stem.replace("-cls", "")
    return f"{stem}_{suffix}"


def build_run_config(cfg: dict, model_name: str, source_dataset: Path, project_dir: Path, run_name: str) -> dict:
    return {
        "task": "classify",
        "model": model_name,
        "data": str(source_dataset),
        "epochs": int(cfg.get("epochs", 100)),
        "imgsz": int(cfg.get("imgsz", 640)),
        "batch": int(cfg.get("batch", 32)),
        "device": resolve_str(cfg.get("device"), "0"),
        "workers": int(cfg.get("workers", 4)),
        "project": str(project_dir),
        "name": run_name,
        "pretrained": bool(cfg.get("pretrained", True)),
        "patience": int(cfg.get("patience", 20)),
        "optimizer": resolve_str(cfg.get("optimizer"), "auto"),
        "cache": bool(cfg.get("cache", False)),
        "resume": bool(cfg.get("resume", False)),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_cls6_sweep(cfg: dict, dry_run: bool, rerun: bool) -> None:
    source_dataset = resolve_path(cfg.get("source_dataset"), base=YOLOV11_ROOT / "datasets" / "sewerml_cls6_train7200")
    project_dir = resolve_path(cfg.get("project"), base=YOLOV11_ROOT / "runs" / "cls_source_uniform")
    recycle_root = resolve_path(cfg.get("recycle_root"), base=REPO_ROOT / "_recycle_bin" / "cls_source_uniform")
    temp_config_dir = resolve_path(cfg.get("temp_config_dir"), base=REPO_ROOT / "$out" / "generated_configs" / "cls6_sweep")
    models = cfg.get("models") or []
    suffix = resolve_str(cfg.get("run_name_suffix"), "cls6_train7200_uniform")
    collect_batch = int(cfg.get("collect_batch", 16) or 16)

    if not isinstance(models, list) or not models:
        raise SystemExit("Config field 'models' must be a non-empty array.")

    ensure_source_dataset(source_dataset)

    for model_name in models:
        run_name = build_run_name(str(model_name), suffix)
        run_dir = project_dir / run_name
        temp_config_path = temp_config_dir / f"{run_name}.json"

        if rerun:
            archive_existing_run(run_dir, recycle_root, dry_run=dry_run)

        run_cfg = build_run_config(cfg, str(model_name), source_dataset, project_dir, run_name)
        print_step(
            "train",
            f"model={model_name} data={source_dataset} epochs={run_cfg['epochs']} batch={run_cfg['batch']} imgsz={run_cfg['imgsz']} workers={run_cfg['workers']}",
        )
        if not dry_run:
            write_json(temp_config_path, run_cfg)

        run_python(
            "scripts/cls_pretrain.py",
            ["--config", str(temp_config_path if not dry_run else temp_config_path)],
            dry_run=dry_run,
        )
        run_python(
            "scripts/collect_cls_raw_materials.py",
            [
                "--config",
                str(temp_config_path if not dry_run else temp_config_path),
                "--run-dir",
                str(run_dir),
                "--data",
                str(source_dataset),
                "--batch",
                str(collect_batch),
                "--device",
                resolve_str(cfg.get("device"), "0"),
                "--normal-class",
                "Normal",
            ],
            dry_run=dry_run,
        )

    print_step("summary", f"runs -> {project_dir}")
    print_step("summary", f"materials -> {REPO_ROOT / 'research' / 'materials'}")


def run_stage1_hn(task_name: str, entry_cfg: dict, dry_run: bool) -> None:
    task_cfg = STAGE1_HN_TASKS.get(task_name)
    if task_cfg is None:
        raise SystemExit(f"Unsupported stage-1 task: {task_name}")

    score_device = resolve_str(entry_cfg.get("score_device"), "0")
    top_k = str(int(entry_cfg.get("top_k", 22) or 22))
    score_batch = str(int(entry_cfg.get("score_batch", 1) or 1))
    score_chunk_size = str(int(entry_cfg.get("score_chunk_size", 32) or 32))

    print_step("task", f"{task_name} ({task_cfg['label']})")
    run_python(
        "scripts/stage1_score_train_normals.py",
        [
            "--weights",
            task_cfg["weights"],
            "--data-root",
            task_cfg["data_root"],
            "--output-dir",
            task_cfg["output_dir"],
            "--device",
            score_device,
            "--imgsz",
            "640",
            "--batch",
            score_batch,
            "--chunk-size",
            score_chunk_size,
            "--top-k",
            top_k,
        ],
        dry_run=dry_run,
    )
    run_python(
        "scripts/stage1_build_hn_dataset.py",
        [
            "--source-dataset",
            task_cfg["data_root"],
            "--scores-csv",
            str(Path(task_cfg["output_dir"]) / "top_false_positive_normals.csv"),
            "--output-dataset",
            task_cfg["hn_dataset"],
            "--top-k",
            top_k,
            "--repeat",
            "1",
            "--link-mode",
            "hardlink",
        ],
        dry_run=dry_run,
    )
    run_python(
        "scripts/stage1_gate_train.py",
        ["--config", task_cfg["train_config"]],
        dry_run=dry_run,
    )


def main() -> None:
    args = parse_args()
    if args.task == "cls6_sweep":
        config_path = resolve_path(args.config, base=DEFAULT_CONFIG) if args.config else DEFAULT_CONFIG
        config = load_sweep_config(config_path)
        run_cls6_sweep(config, dry_run=args.dry_run, rerun=args.rerun)
        return

    entry_config_path = resolve_path(args.config, base=DEFAULT_ENTRY_CONFIG) if args.config else DEFAULT_ENTRY_CONFIG
    entry_cfg = load_entry_config(entry_config_path)
    task_name = resolve_str(entry_cfg.get("task"), BUILTIN_STAGE1_ENTRY_CONFIG["task"]) if args.task == "auto" else args.task

    if task_name == "cls6_sweep":
        sweep_path = resolve_path(entry_cfg.get("cls6_sweep_config"), base=DEFAULT_CONFIG)
        sweep_cfg = load_sweep_config(sweep_path)
        run_cls6_sweep(sweep_cfg, dry_run=args.dry_run, rerun=args.rerun)
        return

    run_stage1_hn(task_name, entry_cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
