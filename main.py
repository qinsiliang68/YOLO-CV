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
DEFAULT_CONFIG = YOLOV11_ROOT / "configs" / "runtime" / "cls_gate2_sweep.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
BUILTIN_DEFAULT_CONFIG = {
    "source_dataset": r"C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\data\sewerml_cls6_train7200",
    "gate2_dataset": r"C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\datasets\sewerml_gate2_train7200",
    "project": r"C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\runs\cls_gate_source",
    "recycle_root": r"C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\_recycle_bin\cls_gate_source",
    "single_run_config": r"C:\Users\ASUS\Desktop\YOLOv11\YOLO-CV\YOLOv11\configs\runtime\cls_source_gate2.json",
    "dataset_materialization": "hardlink",
    "models": [
        "yolo11n-cls.pt",
        "yolo11s-cls.pt",
        "yolo11m-cls.pt",
        "yolo11l-cls.pt",
        "yolo11x-cls.pt",
    ],
    "epochs": 200,
    "imgsz": 640,
    "batch": 32,
    "collect_batch": 16,
    "device": "0",
    "normal_class": "Normal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the direct Normal/Abnormal gate baseline sweep and collect tracked experiment materials."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Sweep config JSON. In normal use you only need to edit dataset and project paths inside this file.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without training.")
    parser.add_argument("--rerun", action="store_true", help="Rebuild the binary dataset and rerun all configured models.")
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


def has_images(root: Path) -> bool:
    if not root.exists():
        return False
    return any(path.suffix.lower() in IMAGE_SUFFIXES for path in root.rglob("*"))


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


def materialize_gate2_dataset(source_root: Path, gate_root: Path, mode: str, rerun: bool, dry_run: bool) -> None:
    ready = split_image_count(gate_root, "train") > 0 and split_image_count(gate_root, "val") > 0
    if ready and not rerun:
        print_step(
            "data",
            f"gate2 dataset ready: train={split_image_count(gate_root, 'train')} val={split_image_count(gate_root, 'val')}",
        )
        return

    if rerun and gate_root.exists() and not dry_run:
        shutil.rmtree(gate_root)

    print_step("data", f"build gate2 dataset: {source_root} -> {gate_root}")
    if dry_run:
        return

    for split_root in sorted(path for path in source_root.iterdir() if path.is_dir()):
        split = split_root.name
        for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            mapped_class = "Normal" if class_dir.name == "Normal" else "Abnormal"
            for image_path in sorted(path for path in class_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
                target_path = gate_root / split / mapped_class / image_path.name
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if target_path.exists():
                    continue
                if mode == "hardlink":
                    try:
                        os.link(image_path, target_path)
                        continue
                    except OSError:
                        pass
                shutil.copy2(image_path, target_path)

    print_step(
        "data",
        f"gate2 dataset built: train={split_image_count(gate_root, 'train')} val={split_image_count(gate_root, 'val')}",
    )


def build_train_command(cfg: dict, model_name: str, source_dataset: Path, run_name: str, project_dir: Path) -> list[str]:
    command = [
        "--config",
        str(resolve_path(cfg.get("single_run_config"), base=YOLOV11_ROOT / "configs" / "runtime" / "cls_source_gate2.json")),
        "--data",
        str(source_dataset),
        "--model",
        model_name,
        "--project",
        str(project_dir),
        "--name",
        run_name,
    ]
    for key, flag in (("epochs", "--epochs"), ("batch", "--batch"), ("imgsz", "--imgsz")):
        value = int(cfg.get(key, 0) or 0)
        if value > 0:
            command.extend([flag, str(value)])
    device = resolve_str(cfg.get("device"), "")
    if device:
        command.extend(["--device", device])
    return command


def build_collect_command(cfg: dict, run_dir: Path, source_dataset: Path) -> list[str]:
    command = [
        "--config",
        str(resolve_path(cfg.get("single_run_config"), base=YOLOV11_ROOT / "configs" / "runtime" / "cls_source_gate2.json")),
        "--weights",
        str(run_dir / "weights" / "best.pt"),
        "--data",
        str(source_dataset),
        "--batch",
        str(int(cfg.get("collect_batch", 16) or 16)),
        "--normal-class",
        resolve_str(cfg.get("normal_class"), "Normal"),
    ]
    device = resolve_str(cfg.get("device"), "")
    if device:
        command.extend(["--device", device])
    return command


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


def run_gate2_sweep(cfg: dict, dry_run: bool, rerun: bool) -> None:
    source_dataset = resolve_path(cfg.get("source_dataset"), base=YOLOV11_ROOT / "datasets" / "sewerml_cls6_train7200")
    gate2_dataset = resolve_path(cfg.get("gate2_dataset"), base=YOLOV11_ROOT / "datasets" / "sewerml_gate2_train7200")
    project_dir = resolve_path(cfg.get("project"), base=YOLOV11_ROOT / "runs" / "cls_gate_source")
    recycle_root = resolve_path(cfg.get("recycle_root"), base=REPO_ROOT / "_recycle_bin" / "cls_gate_source")
    mode = resolve_str(cfg.get("dataset_materialization"), "hardlink")
    models = cfg.get("models") or []
    if not isinstance(models, list) or not models:
        raise SystemExit("Config field 'models' must be a non-empty array.")

    ensure_source_dataset(source_dataset)
    materialize_gate2_dataset(source_dataset, gate2_dataset, mode=mode, rerun=rerun, dry_run=dry_run)

    for model_name in models:
        run_name = f"{Path(str(model_name)).stem.replace('-cls', '')}_gate2_train7200"
        run_dir = project_dir / run_name
        if rerun:
            archive_existing_run(run_dir, recycle_root, dry_run=dry_run)

        print_step("train", f"model={model_name} dataset={gate2_dataset}")
        run_python(
            "scripts/cls_pretrain.py",
            build_train_command(cfg, str(model_name), gate2_dataset, run_name, project_dir),
            dry_run=dry_run,
        )
        run_python(
            "scripts/collect_cls_raw_materials.py",
            build_collect_command(cfg, run_dir, gate2_dataset),
            dry_run=dry_run,
        )

    print_step("summary", f"runs -> {project_dir}")
    print_step("summary", f"materials -> {REPO_ROOT / 'research' / 'materials'}")


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config, base=DEFAULT_CONFIG)
    config = load_sweep_config(config_path)
    run_gate2_sweep(config, dry_run=args.dry_run, rerun=args.rerun)


if __name__ == "__main__":
    main()
