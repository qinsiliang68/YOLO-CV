from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed-budget weighted stage-1 gate dataset from lite sample scores.")
    parser.add_argument("--source-dataset", required=True, help="Original binary gate dataset root.")
    parser.add_argument("--candidate-scores-csv", required=True, help="Per-setting candidate score CSV with duplication counts.")
    parser.add_argument("--output-dataset", required=True, help="Output dataset root.")
    parser.add_argument("--setting-name", required=True, help="Human-readable setting name.")
    parser.add_argument("--setting-id", required=True, help="Compact setting identifier, e.g. A2.")
    parser.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink", help="File materialization mode.")
    parser.add_argument("--expected-duplication-total", type=int, default=-1, help="Expected replay budget for duplication rows.")
    return parser.parse_args()


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    for split_name in ("train", "val"):
        split_root = source_root / split_name
        if not split_root.exists():
            continue
        for image_path in split_root.rglob("*"):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            materialize(image_path, target_root / image_path.relative_to(source_root), mode)
            count += 1
    return count


def duplicate_selected_rows(rows: list[dict[str, str]], source_root: Path, target_root: Path, setting_id: str, mode: str) -> list[dict[str, Any]]:
    duplicated: list[dict[str, Any]] = []
    for row in rows:
        duplication_count = int(row["duplication_count"])
        if duplication_count <= 0:
            continue
        relative = Path(str(row["img_rel_path"]))
        source_path = source_root / relative
        if not source_path.exists():
            continue
        stem = relative.stem
        suffix = relative.suffix
        for replica_index in range(1, duplication_count + 1):
            target_path = target_root / relative.parent / f"{stem}_{setting_id.lower()}{replica_index:03d}{suffix}"
            materialize(source_path, target_path, mode)
            duplicated.append(
                {
                    "image_id": row["image_id"],
                    "img_rel_path": row["img_rel_path"],
                    "source_path": str(source_path),
                    "duplicated_path": str(target_path),
                    "duplication_index": replica_index,
                    "duplication_count_total": duplication_count,
                    "calibrated_p": row["calibrated_p"],
                    "R": row["R"],
                    "C": row["C"],
                    "D": row["D"],
                    "S": row["S"],
                    "pi": row["pi"],
                }
            )
    return duplicated


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_dataset).resolve()
    target_root = Path(args.output_dataset).resolve()
    scores_rows = load_csv_rows(Path(args.candidate_scores_csv).resolve())
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    mirrored = mirror_dataset(source_root, target_root, args.link_mode)
    duplicated = duplicate_selected_rows(scores_rows, source_root, target_root, args.setting_id, args.link_mode)
    duplication_total = len(duplicated)
    if args.expected_duplication_total >= 0 and duplication_total != int(args.expected_duplication_total):
        raise SystemExit(
            f"{args.setting_id}: duplicated_images={duplication_total} does not match expected "
            f"{int(args.expected_duplication_total)}"
        )
    write_csv(
        target_root / "weighted_replay.csv",
        [
            "image_id",
            "img_rel_path",
            "source_path",
            "duplicated_path",
            "duplication_index",
            "duplication_count_total",
            "calibrated_p",
            "R",
            "C",
            "D",
            "S",
            "pi",
        ],
        duplicated,
    )
    summary = {
        "setting_id": args.setting_id,
        "setting_name": args.setting_name,
        "source_dataset": str(source_root),
        "output_dataset": str(target_root),
        "candidate_scores_csv": str(Path(args.candidate_scores_csv).resolve()),
        "mirrored_images": mirrored,
        "duplicated_images": duplication_total,
        "expected_duplicated_images": int(args.expected_duplication_total) if args.expected_duplication_total >= 0 else None,
        "duplication_total_matches_expected": None if args.expected_duplication_total < 0 else duplication_total == int(args.expected_duplication_total),
        "link_mode": args.link_mode,
    }
    (target_root / "weighted_replay_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print_step("done", f"{args.setting_name}: wrote {target_root}")


if __name__ == "__main__":
    main()
