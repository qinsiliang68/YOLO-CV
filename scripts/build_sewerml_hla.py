from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import Counter
from pathlib import Path


SPLITS = ("Train", "Val", "Test")
ALL_SOURCE_LABELS = (
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
)
STRUCTURAL_LABELS = ("RB", "OB", "PF", "DE", "FS", "IS", "IN")
FUNCTIONAL_LABELS = ("RO", "AF", "BE", "FO")
CONSTRUCTION_LABELS = ("GR", "PH", "PB", "OS", "OP", "OK")

CLS3_NAME = "sewerml_hla_cls3"
CLS6_NAME = "sewerml_hla_cls6"
HOLDOUT_NAME = "research_holdout"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hierarchical label alignment datasets from official SewerML CSV files."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(r"C:\GitHub\YOLO-CV\data\sewerml"),
        help="SewerML raw workspace root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"C:\GitHub\YOLO-CV\YOLOv11\datasets"),
        help="Where aligned training datasets should be created.",
    )
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to place derived files. hardlink is preferred on NTFS.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["Train", "Val"],
        choices=SPLITS,
        help="Official SewerML splits to process.",
    )
    parser.add_argument(
        "--sixclass-in-mode",
        choices=("holdout", "joint"),
        default="holdout",
        help="How to treat IN in the 6-class scheme.",
    )
    parser.add_argument(
        "--sixclass-fo-mode",
        choices=("holdout", "deposit"),
        default="holdout",
        help="How to treat FO in the 6-class scheme.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previously generated aligned datasets before rebuilding.",
    )
    return parser.parse_args()


def resolve_csv_path(annotations_dir: Path, split: str) -> Path | None:
    candidates = (
        annotations_dir / f"{split}13.csv",
        annotations_dir / f"SewerML_{split}.csv",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def is_positive(row: dict[str, str], label: str) -> bool:
    return str(row.get(label, "0")).strip() == "1"


def source_labels(row: dict[str, str]) -> list[str]:
    return [label for label in ALL_SOURCE_LABELS if is_positive(row, label)]


def is_normal(row: dict[str, str]) -> bool:
    if is_positive(row, "ND"):
        return True
    if str(row.get("Defect", "1")).strip() == "0" and not source_labels(row):
        return True
    return False


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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_cls3_labels(row: dict[str, str]) -> list[str]:
    if is_normal(row):
        return ["Normal"]

    labels: list[str] = []
    if any(is_positive(row, label) for label in STRUCTURAL_LABELS):
        labels.append("StructuralDefect")
    if any(is_positive(row, label) for label in FUNCTIONAL_LABELS):
        labels.append("FunctionalDefect")
    return labels


def cls3_holdouts(row: dict[str, str]) -> list[str]:
    reasons: list[str] = []
    if any(is_positive(row, label) for label in CONSTRUCTION_LABELS):
        reasons.append("ConstructionInfo")
    return reasons


def build_cls6_labels(row: dict[str, str], in_mode: str, fo_mode: str) -> list[str]:
    if is_normal(row):
        return ["Normal"]

    labels: list[str] = []
    if any(is_positive(row, label) for label in ("RB", "OB", "PF")):
        labels.append("WallDamage")
    joint_labels = ["FS", "IS"]
    if in_mode == "joint":
        joint_labels.append("IN")
    if any(is_positive(row, label) for label in joint_labels):
        labels.append("JointAnomaly")
    if is_positive(row, "DE"):
        labels.append("Deformation")
    deposit_labels = ["AF", "BE"]
    if fo_mode == "deposit":
        deposit_labels.append("FO")
    if any(is_positive(row, label) for label in deposit_labels):
        labels.append("DepositAttachment")
    if is_positive(row, "RO"):
        labels.append("Roots")
    return labels


def cls6_holdouts(row: dict[str, str], in_mode: str, fo_mode: str) -> list[str]:
    reasons: list[str] = []
    if in_mode == "holdout" and is_positive(row, "IN"):
        reasons.append("IN")
    if fo_mode == "holdout" and is_positive(row, "FO"):
        reasons.append("FO")
    if any(is_positive(row, label) for label in CONSTRUCTION_LABELS):
        reasons.append("ConstructionInfo")
    return reasons


def clean_outputs(output_root: Path, holdout_root: Path) -> None:
    for path in (output_root / CLS3_NAME, output_root / CLS6_NAME, holdout_root):
        if path.exists():
            shutil.rmtree(path)


def write_manifest_header(writer: csv.DictWriter) -> None:
    writer.writeheader()


def create_scheme_dirs(output_root: Path) -> tuple[Path, Path]:
    cls3_root = ensure_dir(output_root / CLS3_NAME)
    cls6_root = ensure_dir(output_root / CLS6_NAME)
    for split in ("Train", "Val"):
        for name in ("Normal", "StructuralDefect", "FunctionalDefect"):
            ensure_dir(cls3_root / "images" / split / name)
        for name in (
            "Normal",
            "WallDamage",
            "JointAnomaly",
            "Deformation",
            "DepositAttachment",
            "Roots",
        ):
            ensure_dir(cls6_root / "images" / split / name)
        ensure_dir(cls3_root / "manifests")
        ensure_dir(cls6_root / "manifests")
    return cls3_root, cls6_root


def create_holdout_dirs(holdout_root: Path) -> None:
    for scheme in ("cls3", "cls6"):
        for split in ("Train", "Val"):
            ensure_dir(holdout_root / scheme / "images" / split)
        ensure_dir(holdout_root / scheme / "manifests")


def build_split(
    csv_path: Path,
    images_dir: Path,
    split: str,
    cls3_root: Path,
    cls6_root: Path,
    holdout_root: Path,
    mode: str,
    in_mode: str,
    fo_mode: str,
) -> None:
    print(f"[build] {split} <- {csv_path.name}")

    cls3_manifest = cls3_root / "manifests" / f"{split}.csv"
    cls6_manifest = cls6_root / "manifests" / f"{split}.csv"
    cls3_holdout_manifest = holdout_root / "cls3" / "manifests" / f"{split}.csv"
    cls6_holdout_manifest = holdout_root / "cls6" / "manifests" / f"{split}.csv"

    counters = {
        "rows": 0,
        "missing_images": 0,
        "cls3_main_rows": 0,
        "cls6_main_rows": 0,
        "cls3_holdout_only": 0,
        "cls6_holdout_only": 0,
    }
    cls3_class_counts: Counter[str] = Counter()
    cls6_class_counts: Counter[str] = Counter()
    cls3_holdout_counts: Counter[str] = Counter()
    cls6_holdout_counts: Counter[str] = Counter()

    with (
        csv_path.open("r", encoding="utf-8-sig", newline="") as source,
        cls3_manifest.open("w", encoding="utf-8", newline="") as cls3_out,
        cls6_manifest.open("w", encoding="utf-8", newline="") as cls6_out,
        cls3_holdout_manifest.open("w", encoding="utf-8", newline="") as cls3_holdout_out,
        cls6_holdout_manifest.open("w", encoding="utf-8", newline="") as cls6_holdout_out,
    ):
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path} has no header row")

        required_fields = {"Filename", "Defect", "ND", *ALL_SOURCE_LABELS}
        missing_fields = required_fields - set(reader.fieldnames)
        if missing_fields:
            raise ValueError(f"{csv_path.name} missing columns: {sorted(missing_fields)}")

        manifest_fields = [
            "filename",
            "split",
            "main_classes",
            "source_labels",
            "holdout_labels",
            "defect",
            "water_level",
        ]
        cls3_writer = csv.DictWriter(cls3_out, fieldnames=manifest_fields)
        cls6_writer = csv.DictWriter(cls6_out, fieldnames=manifest_fields)
        cls3_holdout_writer = csv.DictWriter(cls3_holdout_out, fieldnames=manifest_fields)
        cls6_holdout_writer = csv.DictWriter(cls6_holdout_out, fieldnames=manifest_fields)
        write_manifest_header(cls3_writer)
        write_manifest_header(cls6_writer)
        write_manifest_header(cls3_holdout_writer)
        write_manifest_header(cls6_holdout_writer)

        for row in reader:
            counters["rows"] += 1
            filename = Path(row["Filename"]).name
            src = images_dir / filename
            if not src.exists():
                counters["missing_images"] += 1
                continue

            labels = source_labels(row)
            cls3_labels = build_cls3_labels(row)
            cls6_labels = build_cls6_labels(row, in_mode=in_mode, fo_mode=fo_mode)
            cls3_reasons = cls3_holdouts(row)
            cls6_reasons = cls6_holdouts(row, in_mode=in_mode, fo_mode=fo_mode)

            manifest_row = {
                "filename": filename,
                "split": split,
                "source_labels": ";".join(labels),
                "defect": row.get("Defect", ""),
                "water_level": row.get("WaterLevel", ""),
            }

            if cls3_labels:
                counters["cls3_main_rows"] += 1
                for label in cls3_labels:
                    safe_link_or_copy(
                        src,
                        cls3_root / "images" / split / label / filename,
                        mode,
                    )
                    cls3_class_counts[label] += 1
                cls3_writer.writerow(
                    {
                        **manifest_row,
                        "main_classes": ";".join(cls3_labels),
                        "holdout_labels": ";".join(cls3_reasons),
                    }
                )
            elif cls3_reasons:
                counters["cls3_holdout_only"] += 1
                for reason in cls3_reasons:
                    safe_link_or_copy(
                        src,
                        holdout_root / "cls3" / "images" / split / reason / filename,
                        mode,
                    )
                    cls3_holdout_counts[reason] += 1
                cls3_holdout_writer.writerow(
                    {
                        **manifest_row,
                        "main_classes": "",
                        "holdout_labels": ";".join(cls3_reasons),
                    }
                )

            if cls6_labels:
                counters["cls6_main_rows"] += 1
                for label in cls6_labels:
                    safe_link_or_copy(
                        src,
                        cls6_root / "images" / split / label / filename,
                        mode,
                    )
                    cls6_class_counts[label] += 1
                cls6_writer.writerow(
                    {
                        **manifest_row,
                        "main_classes": ";".join(cls6_labels),
                        "holdout_labels": ";".join(cls6_reasons),
                    }
                )
            elif cls6_reasons:
                counters["cls6_holdout_only"] += 1
                for reason in cls6_reasons:
                    safe_link_or_copy(
                        src,
                        holdout_root / "cls6" / "images" / split / reason / filename,
                        mode,
                    )
                    cls6_holdout_counts[reason] += 1
                cls6_holdout_writer.writerow(
                    {
                        **manifest_row,
                        "main_classes": "",
                        "holdout_labels": ";".join(cls6_reasons),
                    }
                )

    print(f"  rows scanned: {counters['rows']}")
    print(f"  missing source images: {counters['missing_images']}")
    print(f"  cls3 included rows: {counters['cls3_main_rows']}")
    print(f"  cls3 holdout-only rows: {counters['cls3_holdout_only']}")
    print(f"  cls3 class counts: {dict(cls3_class_counts)}")
    if cls3_holdout_counts:
        print(f"  cls3 holdout counts: {dict(cls3_holdout_counts)}")
    print(f"  cls6 included rows: {counters['cls6_main_rows']}")
    print(f"  cls6 holdout-only rows: {counters['cls6_holdout_only']}")
    print(f"  cls6 class counts: {dict(cls6_class_counts)}")
    if cls6_holdout_counts:
        print(f"  cls6 holdout counts: {dict(cls6_holdout_counts)}")


def main() -> None:
    args = parse_args()
    annotations_dir = args.base_dir / "annotations"
    images_dir = args.base_dir / "images_all"
    holdout_root = args.base_dir / HOLDOUT_NAME

    if args.clean:
        clean_outputs(args.output_root, holdout_root)

    cls3_root, cls6_root = create_scheme_dirs(args.output_root)
    create_holdout_dirs(holdout_root)

    found = False
    for split in args.splits:
        csv_path = resolve_csv_path(annotations_dir, split)
        if csv_path is None:
            print(f"[skip] missing annotation file for split: {split}")
            continue

        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
        if not set(ALL_SOURCE_LABELS).issubset(fieldnames):
            print(f"[skip] {csv_path.name} does not contain training labels.")
            continue

        found = True
        build_split(
            csv_path=csv_path,
            images_dir=images_dir,
            split=split,
            cls3_root=cls3_root,
            cls6_root=cls6_root,
            holdout_root=holdout_root,
            mode=args.mode,
            in_mode=args.sixclass_in_mode,
            fo_mode=args.sixclass_fo_mode,
        )

    if not found:
        print("[done] no labeled SewerML split was processed.")
        return

    print(f"[done] cls3 dataset: {cls3_root}")
    print(f"[done] cls6 dataset: {cls6_root}")
    print(f"[done] holdout manifests: {holdout_root}")


if __name__ == "__main__":
    main()
