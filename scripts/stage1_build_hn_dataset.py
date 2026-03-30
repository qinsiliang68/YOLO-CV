from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a hard-negative-augmented stage-1 gate dataset.")
    parser.add_argument("--source-dataset", required=True, help="Original binary gate dataset root.")
    parser.add_argument("--scores-csv", required=True, help="CSV exported by stage1_score_train_normals.py.")
    parser.add_argument("--output-dataset", required=True, help="Output dataset root.")
    parser.add_argument("--top-k", type=int, default=200, help="Number of hard negatives to duplicate.")
    parser.add_argument("--repeat", type=int, default=2, help="Extra copies per selected hard negative.")
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


def load_top_rows(path: Path, top_k: int) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows_sorted = sorted(rows, key=lambda item: float(item["p_abnormal"]), reverse=True)
    return rows_sorted[:top_k]


def duplicate_hn(rows: list[dict], source_root: Path, target_root: Path, repeat: int, mode: str) -> list[dict]:
    duplicated: list[dict] = []
    for row in rows:
        src = Path(row["img_path"])
        if not src.exists():
            continue
        relative = Path(row["img_rel_path"])
        stem = relative.stem
        suffix = relative.suffix
        for rep in range(1, repeat + 1):
            target = target_root / relative.parent / f"{stem}_hn{rep}{suffix}"
            materialize(src, target, mode)
            duplicated.append(
                {
                    "source_path": str(src),
                    "duplicated_path": str(target),
                    "p_abnormal": row["p_abnormal"],
                    "heuristic_group": row.get("heuristic_group", ""),
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

    mirrored = mirror_dataset(source_root, target_root, args.link_mode)
    top_rows = load_top_rows(scores_csv, args.top_k)
    duplicated = duplicate_hn(top_rows, source_root, target_root, args.repeat, args.link_mode)
    summary = {
        "source_dataset": str(source_root),
        "output_dataset": str(target_root),
        "scores_csv": str(scores_csv),
        "top_k": args.top_k,
        "repeat": args.repeat,
        "mirrored_images": mirrored,
        "duplicated_hn_images": len(duplicated),
        "link_mode": args.link_mode,
    }
    (target_root / "hn_build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (target_root / "hn_duplications.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["source_path", "duplicated_path", "p_abnormal", "heuristic_group"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(duplicated)
    print_step("done", f"wrote {target_root}")


if __name__ == "__main__":
    main()
