from __future__ import annotations

import argparse
import csv
import os
import shutil
import zipfile
from pathlib import Path


DEFECT_LABELS = [
    "RB",
    "OB",
    "PF",
    "DE",
    "FS",
    "IS",
    "RO",
    "IN",
    "AF",
    "BE",
    "FO",
    "GR",
    "PH",
    "PB",
    "OS",
    "OP",
    "OK",
]
SPLITS = ("Train", "Val", "Test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare SewerML images and organize them into class folders using official CSV annotations."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(r"C:\GitHub\YOLO-CV\data\sewerml"),
        help="SewerML workspace root.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract all images from zip archives into images_all before organizing.",
    )
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to place files into class folders. hardlink saves disk space on NTFS.",
    )
    parser.add_argument(
        "--layout",
        choices=("per_class", "grouped"),
        default="per_class",
        help="per_class duplicates images into every positive class folder. grouped keeps single-damage folders and puts multi-damage images under MultiDamage/<combo>/.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=list(SPLITS),
        choices=SPLITS,
        help="Which official splits to organize.",
    )
    return parser.parse_args()


def ensure_structure(base_dir: Path) -> tuple[Path, Path, Path, Path]:
    archives_dir = base_dir / "archives"
    images_dir = base_dir / "images_all"
    annotations_dir = base_dir / "annotations"
    by_class_dir = base_dir / "by_class"

    for path in (archives_dir, images_dir, annotations_dir, by_class_dir):
        path.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        for label in ["Normal", *DEFECT_LABELS]:
            (by_class_dir / split / label).mkdir(parents=True, exist_ok=True)

    return archives_dir, images_dir, annotations_dir, by_class_dir


def ensure_grouped_structure(base_dir: Path) -> Path:
    grouped_dir = base_dir / "by_damage_pattern"
    for split in SPLITS:
        (grouped_dir / split / "Normal").mkdir(parents=True, exist_ok=True)
        (grouped_dir / split / "MultiDamage").mkdir(parents=True, exist_ok=True)
        for label in DEFECT_LABELS:
            (grouped_dir / split / label).mkdir(parents=True, exist_ok=True)
    return grouped_dir


def resolve_csv_path(annotations_dir: Path, split: str) -> Path | None:
    candidates = [
        annotations_dir / f"{split}13.csv",
        annotations_dir / f"SewerML_{split}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


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


def extract_archives(archives_dir: Path, images_dir: Path) -> None:
    archives = sorted(archives_dir.glob("*.zip"))
    if not archives:
        print(f"[extract] no zip archives found in {archives_dir}")
        return

    for archive in archives:
        print(f"[extract] {archive.name}")
        with zipfile.ZipFile(archive) as zf:
            image_members = [m for m in zf.infolist() if Path(m.filename).suffix.lower() in IMAGE_SUFFIXES]
            total = len(image_members)
            for idx, member in enumerate(image_members, start=1):
                target = images_dir / Path(member.filename).name
                if target.exists():
                    continue
                with zf.open(member) as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
                if idx % 5000 == 0 or idx == total:
                    print(f"  extracted {idx}/{total}")


def organize_split(csv_path: Path, images_dir: Path, split_dir: Path, mode: str) -> None:
    print(f"[organize] {csv_path.name}")
    created = 0
    missing = 0
    normal_count = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path} has no header row")

        available_labels = [label for label in DEFECT_LABELS if label in reader.fieldnames]
        required_columns = {"Filename", "Defect", *available_labels}
        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"{csv_path.name} missing columns: {sorted(missing_columns)}")

        for row in reader:
            filename = Path(row["Filename"]).name
            src = images_dir / filename
            if not src.exists():
                missing += 1
                continue

            positive_labels = [label for label in available_labels if str(row.get(label, "0")).strip() == "1"]

            if str(row.get("Defect", "0")).strip() == "0" or not positive_labels:
                safe_link_or_copy(src, split_dir / "Normal" / filename, mode)
                created += 1
                normal_count += 1

            for label in positive_labels:
                safe_link_or_copy(src, split_dir / label / filename, mode)
                created += 1

    print(f"  created entries: {created}")
    print(f"  normal entries: {normal_count}")
    if missing:
        print(f"  missing source images: {missing}")


def organize_split_grouped(csv_path: Path, images_dir: Path, split_dir: Path, mode: str) -> None:
    print(f"[organize-grouped] {csv_path.name}")
    created = 0
    missing = 0
    normal_count = 0
    multi_count = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path} has no header row")

        available_labels = [label for label in DEFECT_LABELS if label in reader.fieldnames]
        required_columns = {"Filename", "Defect", *available_labels}
        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"{csv_path.name} missing columns: {sorted(missing_columns)}")

        for row in reader:
            filename = Path(row["Filename"]).name
            src = images_dir / filename
            if not src.exists():
                missing += 1
                continue

            positive_labels = [label for label in available_labels if str(row.get(label, "0")).strip() == "1"]

            if str(row.get("Defect", "0")).strip() == "0" or not positive_labels:
                safe_link_or_copy(src, split_dir / "Normal" / filename, mode)
                created += 1
                normal_count += 1
                continue

            if len(positive_labels) == 1:
                safe_link_or_copy(src, split_dir / positive_labels[0] / filename, mode)
                created += 1
                continue

            combo = "__".join(positive_labels)
            safe_link_or_copy(src, split_dir / "MultiDamage" / combo / filename, mode)
            created += 1
            multi_count += 1

    print(f"  created entries: {created}")
    print(f"  normal entries: {normal_count}")
    print(f"  multi-damage entries: {multi_count}")
    if missing:
        print(f"  missing source images: {missing}")


def main() -> None:
    args = parse_args()
    archives_dir, images_dir, annotations_dir, by_class_dir = ensure_structure(args.base_dir)
    grouped_dir = ensure_grouped_structure(args.base_dir) if args.layout == "grouped" else None

    if args.extract:
        extract_archives(archives_dir, images_dir)

    found_csv = False
    for split in args.splits:
        csv_path = resolve_csv_path(annotations_dir, split)
        if csv_path is None:
            print(f"[skip] missing annotation file for split: {split}")
            continue
        found_csv = True
        if args.layout == "grouped":
            organize_split_grouped(csv_path, images_dir, grouped_dir / split, args.mode)
        else:
            organize_split(csv_path, images_dir, by_class_dir / split, args.mode)

    if not found_csv:
        print("[done] no official CSV found yet.")
        print(f"Place Train13.csv / Val13.csv / Test13.csv into: {annotations_dir}")


if __name__ == "__main__":
    main()
