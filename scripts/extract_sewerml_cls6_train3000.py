from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


CLASS_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Deformation", ("DE",)),
    ("Roots", ("RO",)),
    ("JointAnomaly", ("FS", "IS")),
    ("WallDamage", ("RB", "OB", "PF")),
    ("DepositAttachment", ("AF", "BE")),
]
ALL_SOURCE_LABELS = ("RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE", "FO", "GR", "PH", "PB", "OS", "OP", "OK")
HOLDOUT_LABELS = ("IN", "FO", "GR", "PH", "PB", "OS", "OP", "OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a compact 3000-image single-label SewerML classification dataset directly from the raw library."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(r"C:\GitHub\YOLO-CV\data\sewerml"),
        help="Raw SewerML workspace root containing annotations and images_all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\GitHub\YOLO-CV\YOLOv11\datasets\sewerml_cls6_train3000"),
        help="Target dataset root.",
    )
    parser.add_argument(
        "--csv-name",
        default="SewerML_Train.csv",
        help="Annotation CSV used as the source pool.",
    )
    parser.add_argument("--train-per-class", type=int, default=450, help="Training images per class.")
    parser.add_argument("--val-per-class", type=int, default=50, help="Validation images per class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for stable sampling.")
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Store extracted files as hardlinks or full copies.",
    )
    parser.add_argument("--clean", action="store_true", help="Remove an existing output directory before rebuilding.")
    return parser.parse_args()


def is_positive(row: dict[str, str], label: str) -> bool:
    return str(row.get(label, "0")).strip() == "1"


def is_normal(row: dict[str, str]) -> bool:
    if is_positive(row, "ND"):
        return True
    return str(row.get("Defect", "1")).strip() == "0" and not any(is_positive(row, label) for label in ALL_SOURCE_LABELS)


def assigned_class(row: dict[str, str]) -> str | None:
    if any(is_positive(row, label) for label in HOLDOUT_LABELS):
        return None
    if is_normal(row):
        return "Normal"
    for class_name, labels in CLASS_RULES:
        if any(is_positive(row, label) for label in labels):
            return class_name
    return None


def safe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["split", "class_name", "image_name", "source_csv", "source_labels", "image_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    annotations_csv = args.base_dir / "annotations" / args.csv_name
    images_dir = args.base_dir / "images_all"
    if not annotations_csv.exists():
        raise SystemExit(f"Missing annotation CSV: {annotations_csv}")
    if not images_dir.exists():
        raise SystemExit(f"Missing images directory: {images_dir}")

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    total_per_class = args.train_per_class + args.val_per_class
    rng = random.Random(args.seed)
    pools: dict[str, list[dict[str, str]]] = defaultdict(list)

    with annotations_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_name = row["Filename"]
            image_path = images_dir / image_name
            if not image_path.exists():
                continue
            class_name = assigned_class(row)
            if not class_name:
                continue
            source_labels = [label for label in ALL_SOURCE_LABELS if is_positive(row, label)]
            pools[class_name].append(
                {
                    "class_name": class_name,
                    "image_name": image_name,
                    "image_path": str(image_path),
                    "source_labels": "|".join(source_labels) if source_labels else "ND",
                }
            )

    expected_classes = ["Normal"] + [name for name, _ in CLASS_RULES]
    for class_name in expected_classes:
        available = len(pools[class_name])
        if available < total_per_class:
            raise SystemExit(
                f"Class {class_name} has only {available} usable raw images, fewer than the requested {total_per_class}."
            )
        rng.shuffle(pools[class_name])

    train_manifest: list[dict[str, str]] = []
    val_manifest: list[dict[str, str]] = []

    for class_name in expected_classes:
        selected = pools[class_name][:total_per_class]
        train_rows = selected[: args.train_per_class]
        val_rows = selected[args.train_per_class :]
        for split_name, rows, manifest in (
            ("train", train_rows, train_manifest),
            ("val", val_rows, val_manifest),
        ):
            for row in rows:
                src = Path(row["image_path"])
                dst = args.output_dir / split_name / class_name / row["image_name"]
                safe_link_or_copy(src, dst, args.mode)
                manifest.append(
                    {
                        "split": split_name,
                        "class_name": class_name,
                        "image_name": row["image_name"],
                        "source_csv": annotations_csv.name,
                        "source_labels": row["source_labels"],
                        "image_path": row["image_path"],
                    }
                )

    write_manifest(args.output_dir / "manifests" / "train.csv", train_manifest)
    write_manifest(args.output_dir / "manifests" / "val.csv", val_manifest)

    summary_lines = [
        "# SewerML CLS6 Train3000",
        "",
        "This dataset is extracted directly from the raw SewerML library under `data/sewerml`.",
        "",
        f"- Source CSV: `{annotations_csv.name}`",
        f"- Mode: `{args.mode}`",
        f"- Train per class: `{args.train_per_class}`",
        f"- Val per class: `{args.val_per_class}`",
        f"- Seed: `{args.seed}`",
        "",
        "## Classes",
        "",
    ]
    for class_name in expected_classes:
        summary_lines.append(f"- `{class_name}`: `{total_per_class}` images")
    (args.output_dir / "README.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[done] dataset written to {args.output_dir}")
    for class_name in expected_classes:
        print(f"[class] {class_name}: {total_per_class}")


if __name__ == "__main__":
    main()
