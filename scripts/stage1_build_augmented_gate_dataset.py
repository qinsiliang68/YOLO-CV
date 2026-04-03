from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build augmented stage-1 gate datasets for max-filter experiments.")
    parser.add_argument("--source-dataset", required=True, help="Original binary gate dataset root.")
    parser.add_argument("--scores-csv", required=True, help="CSV exported by stage1_score_train_samples.py.")
    parser.add_argument("--output-dataset", required=True, help="Output dataset root.")
    parser.add_argument("--normal-class", default="Normal", help="Class treated as normal.")
    parser.add_argument("--hard-negative-top-k", type=int, default=0, help="Top-K hard negatives to duplicate.")
    parser.add_argument("--hard-negative-repeat", type=int, default=0, help="Extra copies per selected hard negative.")
    parser.add_argument("--hard-positive-top-k", type=int, default=0, help="Top-K hard positives to duplicate.")
    parser.add_argument("--hard-positive-repeat", type=int, default=0, help="Extra copies per selected hard positive.")
    parser.add_argument("--abnormal-repeat-all", type=int, default=0, help="Extra copies for every abnormal train image.")
    parser.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink", help="File materialization mode.")
    return parser.parse_args()


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def materialize(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def mirror_dataset(source_root: Path, target_root: Path, mode: str) -> int:
    count = 0
    for split_dir in ("train", "val"):
        split_root = source_root / split_dir
        if not split_root.exists():
            continue
        for image_path in split_root.rglob("*"):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            target = target_root / image_path.relative_to(source_root)
            materialize(image_path, target, mode)
            count += 1
    return count


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def duplicate_selected(rows: list[dict], target_root: Path, repeat: int, mode: str, suffix_tag: str) -> list[dict]:
    duplicated: list[dict] = []
    for row in rows:
        src = Path(row["img_path"])
        if not src.exists():
            continue
        relative = Path(row["img_rel_path"])
        stem = relative.stem
        suffix = relative.suffix
        for rep in range(1, repeat + 1):
            target = target_root / relative.parent / f"{stem}_{suffix_tag}{rep}{suffix}"
            materialize(src, target, mode)
            duplicated.append(
                {
                    "source_path": str(src),
                    "duplicated_path": str(target),
                    "gt_label": row["gt_label"],
                    "pred_label": row["pred_label"],
                    "p_abnormal": row["p_abnormal"],
                    "hardness_score": row["hardness_score"],
                    "duplication_type": suffix_tag,
                }
            )
    return duplicated


def duplicate_all_abnormal(source_root: Path, target_root: Path, normal_class: str, repeat: int, mode: str) -> list[dict]:
    duplicated: list[dict] = []
    train_root = source_root / "train"
    if not train_root.exists():
        return duplicated
    for class_dir in sorted(path for path in train_root.iterdir() if path.is_dir() and path.name != normal_class):
        images = sorted(path for path in class_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        for image_path in images:
            relative = image_path.relative_to(source_root)
            stem = relative.stem
            suffix = relative.suffix
            for rep in range(1, repeat + 1):
                target = target_root / relative.parent / f"{stem}_abn{rep}{suffix}"
                materialize(image_path, target, mode)
                duplicated.append(
                    {
                        "source_path": str(image_path),
                        "duplicated_path": str(target),
                        "gt_label": class_dir.name,
                        "pred_label": "",
                        "p_abnormal": "",
                        "hardness_score": "",
                        "duplication_type": "abn",
                    }
                )
    return duplicated


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_dataset).resolve()
    target_root = Path(args.output_dataset).resolve()
    scores_csv = Path(args.scores_csv).resolve()
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    rows = load_rows(scores_csv)
    mirrored = mirror_dataset(source_root, target_root, args.link_mode)

    hard_negative_rows = []
    if args.hard_negative_top_k > 0 and args.hard_negative_repeat > 0:
        candidates = [row for row in rows if row["gt_label"] == args.normal_class]
        hard_negative_rows = sorted(candidates, key=lambda item: float(item["p_abnormal"]), reverse=True)[: args.hard_negative_top_k]
    hard_positive_rows = []
    if args.hard_positive_top_k > 0 and args.hard_positive_repeat > 0:
        candidates = [row for row in rows if row["gt_label"] != args.normal_class]
        hard_positive_rows = sorted(candidates, key=lambda item: float(item["p_abnormal"]))[: args.hard_positive_top_k]

    duplicated = []
    duplicated.extend(duplicate_selected(hard_negative_rows, target_root, args.hard_negative_repeat, args.link_mode, "hn"))
    duplicated.extend(duplicate_selected(hard_positive_rows, target_root, args.hard_positive_repeat, args.link_mode, "hp"))
    duplicated.extend(duplicate_all_abnormal(source_root, target_root, args.normal_class, args.abnormal_repeat_all, args.link_mode))

    summary = {
        "source_dataset": str(source_root),
        "output_dataset": str(target_root),
        "scores_csv": str(scores_csv),
        "normal_class": args.normal_class,
        "hard_negative_top_k": args.hard_negative_top_k,
        "hard_negative_repeat": args.hard_negative_repeat,
        "hard_positive_top_k": args.hard_positive_top_k,
        "hard_positive_repeat": args.hard_positive_repeat,
        "abnormal_repeat_all": args.abnormal_repeat_all,
        "mirrored_images": mirrored,
        "duplicated_images": len(duplicated),
        "link_mode": args.link_mode,
    }
    (target_root / "augmentation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (target_root / "augmentation_duplications.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["source_path", "duplicated_path", "gt_label", "pred_label", "p_abnormal", "hardness_score", "duplication_type"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(duplicated)
    print_step("done", f"wrote {target_root}")


if __name__ == "__main__":
    main()
