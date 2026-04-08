from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild a stage-1 formal HN dataset view from a base dataset and a duplication CSV."
    )
    parser.add_argument("--source-dataset", required=True, help="Base binary gate dataset root.")
    parser.add_argument("--duplication-csv", required=True, help="CSV listing HN duplicate source/target pairs.")
    parser.add_argument("--output-dataset", required=True, help="Output dataset root to create.")
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "copy"],
        default="copy",
        help="Materialization mode for mirrored and duplicated files.",
    )
    return parser.parse_args()


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


def iter_image_paths(root: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes]


def mirror_base_dataset(source_root: Path, target_root: Path, mode: str) -> int:
    count = 0
    for src in iter_image_paths(source_root):
        dst = target_root / src.relative_to(source_root)
        materialize(src, dst, mode)
        count += 1
    return count


def load_duplications(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def apply_duplications(source_root: Path, target_root: Path, rows: list[dict[str, str]], mode: str) -> int:
    count = 0
    for row in rows:
        src = source_root / Path(row["source_rel_path"])
        dst = target_root / Path(row["target_rel_path"])
        if not src.exists():
            raise FileNotFoundError(f"Missing source file listed in duplication CSV: {src}")
        materialize(src, dst, mode)
        count += 1
    return count


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_dataset).resolve()
    duplication_csv = Path(args.duplication_csv).resolve()
    target_root = Path(args.output_dataset).resolve()
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    mirrored = mirror_base_dataset(source_root, target_root, args.link_mode)
    duplications = load_duplications(duplication_csv)
    duplicated = apply_duplications(source_root, target_root, duplications, args.link_mode)
    print(
        f"rebuilt {target_root} from {source_root} using {duplication_csv} "
        f"(mirrored={mirrored}, duplicated={duplicated})"
    )


if __name__ == "__main__":
    main()
