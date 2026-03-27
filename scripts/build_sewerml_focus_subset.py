from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build smaller single-label focus subsets from aligned SewerML manifests."
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path(r"C:\GitHub\YOLO-CV\YOLOv11\datasets"),
        help="Root that contains the aligned SewerML datasets.",
    )
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to place focus subset images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio for the compact subsets.",
    )
    parser.add_argument(
        "--cls3-per-class",
        type=int,
        default=6000,
        help="Target total samples per class for the 3-class focus subset.",
    )
    parser.add_argument(
        "--cls6-per-class",
        type=int,
        default=1500,
        help="Target total samples per class for the 6-class focus subset.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previously generated focus subsets before rebuilding.",
    )
    return parser.parse_args()


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


def clean_outputs(datasets_root: Path) -> None:
    for name in ("sewerml_hla_cls3_focus", "sewerml_hla_cls6_focus"):
        path = datasets_root / name
        if path.exists():
            shutil.rmtree(path)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def choose_label(main_classes: str, priority: list[str]) -> str | None:
    labels = [label for label in main_classes.split(";") if label]
    if not labels:
        return None
    for label in priority:
        if label in labels:
            return label
    return None


def select_rows(
    rows: list[dict[str, str]],
    priority: list[str],
    per_class: int,
    rng: random.Random,
) -> dict[str, list[dict[str, str]]]:
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("holdout_labels"):
            continue
        label = choose_label(row.get("main_classes", ""), priority)
        if label is None:
            continue
        buckets[label].append(row)

    selected: dict[str, list[dict[str, str]]] = {}
    for label in priority:
        candidates = buckets.get(label, [])
        rng.shuffle(candidates)
        selected[label] = candidates[: min(per_class, len(candidates))]
    return selected


def create_subset(
    name: str,
    selected: dict[str, list[dict[str, str]]],
    images_root: Path,
    datasets_root: Path,
    val_ratio: float,
    mode: str,
) -> None:
    subset_root = datasets_root / name
    manifest_root = subset_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)

    train_manifest = manifest_root / "train.csv"
    val_manifest = manifest_root / "val.csv"
    fields = ["filename", "assigned_class", "source_main_classes", "source_labels", "water_level"]

    with (
        train_manifest.open("w", encoding="utf-8", newline="") as train_out,
        val_manifest.open("w", encoding="utf-8", newline="") as val_out,
    ):
        train_writer = csv.DictWriter(train_out, fieldnames=fields)
        val_writer = csv.DictWriter(val_out, fieldnames=fields)
        train_writer.writeheader()
        val_writer.writeheader()

        for label, rows in selected.items():
            val_count = max(1, int(round(len(rows) * val_ratio))) if len(rows) > 1 else 0
            val_rows = rows[:val_count]
            train_rows = rows[val_count:]

            for split_name, split_rows, writer in (
                ("train", train_rows, train_writer),
                ("val", val_rows, val_writer),
            ):
                split_dir = subset_root / split_name / label
                split_dir.mkdir(parents=True, exist_ok=True)
                for row in split_rows:
                    filename = row["filename"]
                    src = images_root / filename
                    safe_link_or_copy(src, split_dir / filename, mode)
                    writer.writerow(
                        {
                            "filename": filename,
                            "assigned_class": label,
                            "source_main_classes": row["main_classes"],
                            "source_labels": row["source_labels"],
                            "water_level": row["water_level"],
                        }
                    )


def print_summary(name: str, selected: dict[str, list[dict[str, str]]], val_ratio: float) -> None:
    counts = Counter({label: len(rows) for label, rows in selected.items()})
    print(f"[focus] {name}")
    print(f"  selected totals: {dict(counts)}")
    train_counts = {
        label: len(rows) - (max(1, int(round(len(rows) * val_ratio))) if len(rows) > 1 else 0)
        for label, rows in selected.items()
    }
    val_counts = {
        label: (max(1, int(round(len(rows) * val_ratio))) if len(rows) > 1 else 0)
        for label, rows in selected.items()
    }
    print(f"  train split: {train_counts}")
    print(f"  val split: {val_counts}")


def main() -> None:
    args = parse_args()
    if args.clean:
        clean_outputs(args.datasets_root)

    rng = random.Random(args.seed)
    images_root = Path(r"C:\GitHub\YOLO-CV\data\sewerml\images_all")

    cls3_rows = load_manifest(args.datasets_root / "sewerml_hla_cls3" / "manifests" / "Train.csv")
    cls6_rows = load_manifest(args.datasets_root / "sewerml_hla_cls6" / "manifests" / "Train.csv")

    cls3_priority = ["FunctionalDefect", "StructuralDefect", "Normal"]
    cls6_priority = [
        "Roots",
        "Deformation",
        "DepositAttachment",
        "WallDamage",
        "JointAnomaly",
        "Normal",
    ]

    cls3_selected = select_rows(cls3_rows, cls3_priority, args.cls3_per_class, rng)
    cls6_selected = select_rows(cls6_rows, cls6_priority, args.cls6_per_class, rng)

    create_subset(
        name="sewerml_hla_cls3_focus",
        selected=cls3_selected,
        images_root=images_root,
        datasets_root=args.datasets_root,
        val_ratio=args.val_ratio,
        mode=args.mode,
    )
    create_subset(
        name="sewerml_hla_cls6_focus",
        selected=cls6_selected,
        images_root=images_root,
        datasets_root=args.datasets_root,
        val_ratio=args.val_ratio,
        mode=args.mode,
    )

    print_summary("sewerml_hla_cls3_focus", cls3_selected, args.val_ratio)
    print_summary("sewerml_hla_cls6_focus", cls6_selected, args.val_ratio)


if __name__ == "__main__":
    main()
