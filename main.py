from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
YOLOV11_ROOT = REPO_ROOT / "YOLOv11"
DATA_ROOT = REPO_ROOT / "data"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
STAGE_ORDER = ("source", "target", "cam", "pseudobox")

SOURCE_DATASET = REPO_ROOT.parent / "sewerml_cls6_train7200"
TARGET_DATASET = YOLOV11_ROOT / "datasets" / "struct6_cls_target"
SOURCE_RUN_ROOT = YOLOV11_ROOT / "runs" / "cls_source"
TARGET_RUN_ROOT = YOLOV11_ROOT / "runs" / "cls_target"
SOURCE_RUN_NAME = "yolo11m_cls6_train7200"
TARGET_RUN_NAME = "struct6_target_finetune"
SOURCE_CONFIG_FILE = YOLOV11_ROOT / "configs" / "runtime" / "cls_source_cls6.json"
TARGET_CONFIG_FILE = YOLOV11_ROOT / "configs" / "runtime" / "cls_target_struct6.json"
DEFAULT_TARGET_MODEL_FROM_SOURCE = "runs/cls_source/yolo11m_cls6_train7200/weights/best.pt"
CAM_OUTPUT_ROOT = DATA_ROOT / "local" / "cam_outputs"
TARGET_LABEL_MANIFEST = DATA_ROOT / "local" / "labels_cls" / "val_manifest.csv"
PSEUDO_DATASET = YOLOV11_ROOT / "datasets" / "struct6_det_pseudo"
CAM_THRESHOLD_FILE = REPO_ROOT / "research" / "cam_threshold_template.json"
RUN_RECYCLE_ROOT = REPO_ROOT / "_recycle_bin" / "runs"
STRUCT6_CLASSES = (
    "Normal",
    "CrackBreak",
    "SurfaceDamage",
    "Deformation",
    "JointDislocation",
    "Intrusion",
    "Infiltration",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the default SewerML -> Struct6 classification, CAM, and pseudo-box pipeline."
    )
    parser.add_argument("--device", default="", help="Optional device override passed to Ultralytics, e.g. 0 or cpu.")
    parser.add_argument(
        "--cam-device",
        default="",
        help="Torch device for CAM export, e.g. cuda:0 or cpu. Defaults to a CUDA device when training uses a GPU.",
    )
    parser.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink", help="How to materialize derived images.")
    parser.add_argument("--rerun", action="store_true", help="Delete generated outputs for each stage before rerunning it.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned commands and exit.")
    parser.add_argument(
        "--stop-after",
        choices=STAGE_ORDER,
        default="source",
        help="Stop after the selected stage. Defaults to source pretraining only.",
    )
    parser.add_argument("--source-epochs", type=int, default=-1, help="Optional override for source pretraining epochs.")
    parser.add_argument("--source-batch", type=int, default=-1, help="Optional override for source pretraining batch size.")
    parser.add_argument("--source-imgsz", type=int, default=-1, help="Optional override for source pretraining image size.")
    parser.add_argument("--target-epochs", type=int, default=-1, help="Optional override for target fine-tuning epochs.")
    parser.add_argument("--target-batch", type=int, default=-1, help="Optional override for target fine-tuning batch size.")
    parser.add_argument("--target-imgsz", type=int, default=-1, help="Optional override for target fine-tuning image size.")
    parser.add_argument("--cam-alpha", type=float, default=0.45, help="Overlay alpha for CAM export.")
    parser.add_argument("--cam-limit", type=int, default=0, help="Optional CAM image limit for debugging.")
    parser.add_argument("--default-threshold", type=float, default=0.45, help="Default CAM threshold for pseudo boxes.")
    parser.add_argument("--min-area-ratio", type=float, default=0.001, help="Minimum pseudo-box area ratio to keep.")
    parser.add_argument("--max-area-ratio", type=float, default=0.85, help="Maximum pseudo-box area ratio to keep.")
    parser.add_argument("--max-boxes", type=int, default=1, help="Maximum pseudo boxes kept per image.")
    return parser.parse_args()


def stage_enabled(stop_after: str, stage: str) -> bool:
    return STAGE_ORDER.index(stage) <= STAGE_ORDER.index(stop_after)


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def load_runtime_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise SystemExit(f"Runtime config must be a JSON object: {path}")
    return config


def resolve_yolov11_path(value: object, default: Path) -> Path:
    text = str(value).strip() if value is not None else ""
    if not text:
        return default
    path = Path(text)
    if path.is_absolute():
        return path
    return (YOLOV11_ROOT / path).resolve()


def resolve_runtime_int(config: dict[str, object], key: str, default: int) -> int:
    value = config.get(key)
    if isinstance(value, int) and value > 0:
        return value
    return default


def source_runtime_config() -> dict[str, object]:
    if not SOURCE_CONFIG_FILE.exists():
        raise SystemExit(f"Missing source runtime config: {SOURCE_CONFIG_FILE}")
    return load_runtime_config(SOURCE_CONFIG_FILE)


def source_dataset_root() -> Path:
    return resolve_yolov11_path(source_runtime_config().get("data"), SOURCE_DATASET)


def source_project_root() -> Path:
    return resolve_yolov11_path(source_runtime_config().get("project"), SOURCE_RUN_ROOT)


def source_run_name() -> str:
    name = str(source_runtime_config().get("name") or SOURCE_RUN_NAME).strip()
    return name or SOURCE_RUN_NAME


def source_run_dir() -> Path:
    return source_project_root() / source_run_name()


def source_weights_path() -> Path:
    return source_run_dir() / "weights" / "best.pt"


def source_imgsz_value(args: argparse.Namespace) -> int:
    if args.source_imgsz > 0:
        return args.source_imgsz
    return resolve_runtime_int(source_runtime_config(), "imgsz", 640)


def target_runtime_config() -> dict[str, object]:
    if not TARGET_CONFIG_FILE.exists():
        raise SystemExit(f"Missing target runtime config: {TARGET_CONFIG_FILE}")
    return load_runtime_config(TARGET_CONFIG_FILE)


def target_dataset_root() -> Path:
    return resolve_yolov11_path(target_runtime_config().get("data"), TARGET_DATASET)


def target_project_root() -> Path:
    return resolve_yolov11_path(target_runtime_config().get("project"), TARGET_RUN_ROOT)


def target_run_name() -> str:
    name = str(target_runtime_config().get("name") or TARGET_RUN_NAME).strip()
    return name or TARGET_RUN_NAME


def target_run_dir() -> Path:
    return target_project_root() / target_run_name()


def target_weights_path() -> Path:
    return target_run_dir() / "weights" / "best.pt"


def target_model_override_from_source() -> str | None:
    value = str(target_runtime_config().get("model") or "").strip().replace("\\", "/")
    if not value or value == DEFAULT_TARGET_MODEL_FROM_SOURCE:
        return str(source_weights_path())
    return None


def target_imgsz_value(args: argparse.Namespace) -> int:
    if args.target_imgsz > 0:
        return args.target_imgsz
    return resolve_runtime_int(target_runtime_config(), "imgsz", 640)


def build_classify_command_args(
    config_path: Path,
    epochs: int,
    batch: int,
    imgsz: int,
    device: str,
) -> list[str]:
    command = ["--config", str(config_path)]
    if epochs > 0:
        command.extend(["--epochs", str(epochs)])
    if batch > 0:
        command.extend(["--batch", str(batch)])
    if imgsz > 0:
        command.extend(["--imgsz", str(imgsz)])
    if device.strip():
        command.extend(["--device", device])
    return command


def infer_cam_device(device: str, explicit: str) -> str:
    if explicit:
        return explicit
    stripped = device.strip().lower()
    if stripped in {"", "0", "1", "2", "3"} or stripped.isdigit():
        return "cuda:0"
    if stripped.startswith("cuda"):
        return device
    return "cpu"


def has_images(root: Path) -> bool:
    if not root.exists():
        return False
    return any(path.suffix.lower() in IMAGE_SUFFIXES for path in root.rglob("*"))


def split_image_count(root: Path, split: str) -> int:
    split_root = root / split
    if not split_root.exists():
        return 0
    return sum(1 for path in split_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def target_dataset_ready(root: Path) -> bool:
    return split_image_count(root, "train") > 0 and split_image_count(root, "val") > 0


def remove_tree(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink():
        path.unlink()
        return
    shutil.rmtree(path)


def archive_existing_tree(path: Path, bucket: str, dry_run: bool) -> Path | None:
    if not path.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = RUN_RECYCLE_ROOT / bucket / f"{timestamp}_{path.name}"
    print_step("archive", f"{path} -> {destination}")
    if dry_run:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    return destination


def restore_archived_tree(archived_path: Path | None, original_path: Path, bucket: str) -> None:
    if archived_path is None or not archived_path.exists():
        return

    if original_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        interrupted_destination = RUN_RECYCLE_ROOT / bucket / f"{timestamp}_{original_path.name}"
        interrupted_destination.parent.mkdir(parents=True, exist_ok=True)
        print_step("archive", f"{original_path} -> {interrupted_destination}")
        shutil.move(str(original_path), str(interrupted_destination))

    original_path.parent.mkdir(parents=True, exist_ok=True)
    print_step("restore", f"{archived_path} -> {original_path}")
    shutil.move(str(archived_path), str(original_path))


def safe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def script_path(relative: str) -> Path:
    return REPO_ROOT / relative


def run_python(script: str, args: list[str], dry_run: bool) -> None:
    cmd = [sys.executable, str(script_path(script)), *args]
    print_step("run", " ".join(f'"{part}"' if " " in part else part for part in cmd))
    if not dry_run:
        env = os.environ.copy()
        env.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".ultralytics"))
        # Windows + CUDA classification runs can trip PyTorch pin-memory threads with
        # "CUDA error: resource already mapped" on larger batches. Default to the
        # safer path for the one-click pipeline.
        if os.name == "nt":
            env.setdefault("PIN_MEMORY", "False")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)


def require_source_dataset() -> None:
    dataset_root = source_dataset_root()
    train_count = split_image_count(dataset_root, "train")
    val_count = split_image_count(dataset_root, "val")
    if train_count == 0 or val_count == 0:
        raise SystemExit(
            "Source dataset is not ready.\n"
            f"Expected images under {dataset_root / 'train'} and {dataset_root / 'val'}."
        )
    print_step("data", f"source dataset ready: train={train_count} val={val_count}")


def find_manifest(labels_root: Path, split: str) -> Path | None:
    for candidate in (
        labels_root / f"{split}_manifest.csv",
        labels_root / f"{split}.csv",
    ):
        if candidate.exists():
            return candidate
    return None


def first_existing_path(row: dict[str, str], images_root: Path) -> Path | None:
    candidates: list[Path] = []
    source_path = (row.get("source_path") or "").strip()
    if source_path:
        candidates.append(Path(source_path))

    relative_path = (row.get("relative_path") or row.get("image") or "").strip()
    if relative_path:
        candidates.append(images_root / relative_path)

    filename = (row.get("filename") or "").strip()
    if filename:
        candidates.append(images_root / filename)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def normalize_class_name(row: dict[str, str]) -> str:
    value = (row.get("assigned_class") or row.get("class") or row.get("label") or "").strip()
    if not value:
        raise ValueError(f"Missing class column in manifest row: {row}")
    return value


def build_target_dataset_from_manifests(dataset_root: Path, mode: str, rerun: bool, dry_run: bool) -> bool:
    candidates = (
        (DATA_ROOT / "local" / "images", DATA_ROOT / "local" / "labels_cls"),
        (DATA_ROOT / "foshan" / "images", DATA_ROOT / "foshan" / "labels_cls"),
    )
    for images_root, labels_root in candidates:
        train_manifest = find_manifest(labels_root, "train")
        val_manifest = find_manifest(labels_root, "val")
        if train_manifest is None or val_manifest is None:
            continue

        manifests = {"train": train_manifest, "val": val_manifest}
        test_manifest = find_manifest(labels_root, "test")
        if test_manifest is not None:
            manifests["test"] = test_manifest

        if rerun and not dry_run:
            remove_tree(dataset_root)

        print_step("data", f"building target dataset from manifests in {labels_root}")
        if dry_run:
            return True
        for split, manifest_path in manifests.items():
            with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    src = first_existing_path(row, images_root)
                    if src is None:
                        continue
                    class_name = normalize_class_name(row)
                    dst = dataset_root / split / class_name / src.name
                    safe_link_or_copy(src, dst, mode)
        return target_dataset_ready(dataset_root)
    return False


def ensure_target_dataset(mode: str, rerun: bool, dry_run: bool) -> None:
    dataset_root = target_dataset_root()
    if target_dataset_ready(dataset_root):
        print_step(
            "data",
            f"target dataset ready: train={split_image_count(dataset_root, 'train')} val={split_image_count(dataset_root, 'val')}",
        )
        return

    if build_target_dataset_from_manifests(dataset_root=dataset_root, mode=mode, rerun=rerun, dry_run=dry_run):
        print_step(
            "data",
            "target dataset manifests found and can be materialized automatically"
            if dry_run
            else f"target dataset built: train={split_image_count(dataset_root, 'train')} val={split_image_count(dataset_root, 'val')}",
        )
        return

    raise SystemExit(
        "Target classification dataset is not ready.\n"
        f"Expected either:\n"
        f"  1. structured data under {dataset_root}\n"
        f"  2. manifests plus images under {DATA_ROOT / 'local' / 'labels_cls'} and {DATA_ROOT / 'local' / 'images'}\n"
        f"  3. manifests plus images under {DATA_ROOT / 'foshan' / 'labels_cls'} and {DATA_ROOT / 'foshan' / 'images'}"
    )


def write_val_manifest(dataset_root: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    val_root = dataset_root / "val"
    rows: list[dict[str, str]] = []
    for class_dir in sorted(path for path in val_root.iterdir() if path.is_dir()):
        for image_path in sorted(path for path in class_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
            rows.append(
                {
                    "filename": image_path.name,
                    "relative_path": image_path.relative_to(val_root).as_posix(),
                    "assigned_class": class_dir.name,
                }
            )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "relative_path", "assigned_class"])
        writer.writeheader()
        writer.writerows(rows)
    print_step("data", f"wrote target val manifest: {output_path}")


def run_source_stage(args: argparse.Namespace) -> None:
    resolved_source_run_dir = source_run_dir()
    archived_run_dir = archive_existing_tree(resolved_source_run_dir, bucket="cls_source", dry_run=args.dry_run)

    print_step("source", f"loading config: {SOURCE_CONFIG_FILE}")
    print_step("source", "starting source training; the first torch import may take a few seconds")
    try:
        run_python(
            "scripts/cls_pretrain.py",
            build_classify_command_args(
                config_path=SOURCE_CONFIG_FILE,
                epochs=args.source_epochs,
                batch=args.source_batch,
                imgsz=args.source_imgsz,
                device=args.device,
            ),
            dry_run=args.dry_run,
        )
    except BaseException:
        if not args.dry_run:
            restore_archived_tree(archived_run_dir, resolved_source_run_dir, bucket="cls_source_interrupted")
        raise


def run_source_materials_stage(args: argparse.Namespace) -> None:
    weights_path = source_weights_path()
    if not weights_path.exists() and not args.dry_run:
        raise SystemExit(f"Missing source weights for raw material export: {weights_path}")
    command = [
        "--config",
        str(SOURCE_CONFIG_FILE),
        "--weights",
        str(weights_path),
        "--batch",
        "16",
    ]
    if args.device.strip():
        command.extend(["--device", args.device])
    run_python("scripts/collect_cls_raw_materials.py", command, dry_run=args.dry_run)


def run_target_stage(args: argparse.Namespace) -> None:
    resolved_target_run_dir = target_run_dir()
    target_weights = target_weights_path()
    source_model_override = target_model_override_from_source()
    if args.rerun and not args.dry_run:
        remove_tree(resolved_target_run_dir)
    if target_weights.exists():
        print_step("target", f"skip existing weights: {target_weights}")
        return

    print_step("target", f"loading config: {TARGET_CONFIG_FILE}")
    if source_model_override is not None:
        print_step("target", f"using source weights: {source_model_override}")

    command = build_classify_command_args(
        config_path=TARGET_CONFIG_FILE,
        epochs=args.target_epochs,
        batch=args.target_batch,
        imgsz=args.target_imgsz,
        device=args.device,
    )
    if source_model_override is not None:
        command.extend(["--model", source_model_override])

    run_python("scripts/cls_finetune_target.py", command, dry_run=args.dry_run)


def run_cam_stage(args: argparse.Namespace, cam_device: str) -> None:
    target_dataset = target_dataset_root()
    target_weights = target_weights_path()
    manifest_path = CAM_OUTPUT_ROOT / "manifest.csv"
    if args.rerun and not args.dry_run:
        remove_tree(CAM_OUTPUT_ROOT)

    if args.dry_run:
        print_step("data", f"would write target val manifest: {TARGET_LABEL_MANIFEST}")
    else:
        write_val_manifest(target_dataset, TARGET_LABEL_MANIFEST)
    if manifest_path.exists():
        print_step("cam", f"skip existing CAM manifest: {manifest_path}")
        return
    if not target_weights.exists() and not args.dry_run:
        raise SystemExit(f"Missing target weights for CAM export: {target_weights}")

    command = [
        "--weights",
        str(target_weights),
        "--source",
        str(target_dataset / "val"),
        "--output",
        str(CAM_OUTPUT_ROOT),
        "--label-manifest",
        str(TARGET_LABEL_MANIFEST),
        "--imgsz",
        str(target_imgsz_value(args)),
        "--alpha",
        str(args.cam_alpha),
        "--device",
        cam_device,
    ]
    if args.cam_limit > 0:
        command.extend(["--limit", str(args.cam_limit)])
    run_python("scripts/export_cam.py", command, dry_run=args.dry_run)


def run_pseudobox_stage(args: argparse.Namespace) -> None:
    manifest_path = PSEUDO_DATASET / "manifest.csv"
    if args.rerun and not args.dry_run:
        remove_tree(PSEUDO_DATASET)
    if manifest_path.exists():
        print_step("pseudobox", f"skip existing pseudo-box manifest: {manifest_path}")
        return
    cam_manifest = CAM_OUTPUT_ROOT / "manifest.csv"
    if not cam_manifest.exists() and not args.dry_run:
        raise SystemExit(f"Missing CAM manifest for pseudo-box generation: {cam_manifest}")

    run_python(
        "scripts/cam_to_pseudobox.py",
        [
            "--cam-manifest",
            str(cam_manifest),
            "--output",
            str(PSEUDO_DATASET),
            "--thresholds",
            str(CAM_THRESHOLD_FILE),
            "--default-threshold",
            str(args.default_threshold),
            "--min-area-ratio",
            str(args.min_area_ratio),
            "--max-area-ratio",
            str(args.max_area_ratio),
            "--max-boxes",
            str(args.max_boxes),
            "--mode",
            args.mode,
            "--keep-normal",
        ],
        dry_run=args.dry_run,
    )


def print_summary() -> None:
    summary = {
        "source_weights": source_weights_path(),
        "target_weights": target_weights_path(),
        "cam_manifest": CAM_OUTPUT_ROOT / "manifest.csv",
        "pseudo_manifest": PSEUDO_DATASET / "manifest.csv",
    }
    for name, path in summary.items():
        status = "ready" if path.exists() else "pending"
        print_step("summary", f"{name}: {status} -> {path}")


def main() -> None:
    args = parse_args()
    cam_device = infer_cam_device(args.device, args.cam_device)

    require_source_dataset()
    if stage_enabled(args.stop_after, "target"):
        ensure_target_dataset(mode=args.mode, rerun=args.rerun, dry_run=args.dry_run)

    if stage_enabled(args.stop_after, "source"):
        run_source_stage(args)
        run_source_materials_stage(args)
    if stage_enabled(args.stop_after, "target"):
        run_target_stage(args)
    if stage_enabled(args.stop_after, "cam"):
        run_cam_stage(args, cam_device)
    if stage_enabled(args.stop_after, "pseudobox"):
        run_pseudobox_stage(args)

    print_summary()


if __name__ == "__main__":
    main()
