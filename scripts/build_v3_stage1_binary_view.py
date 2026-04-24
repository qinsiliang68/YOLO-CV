"""
build_v3_stage1_binary_view.py — Make a 2-class (Normal/Defect) view of the
existing v3_stage1 7-folder dataset using NTFS hardlinks (no extra disk space).

Input:  data/sewerml_gate_v3_stage1/{train,val_cal,val_op,test}/
            {AF,DE,FS,Normal,OB,PF,RB}/*.png
Output: data/sewerml_gate_v3_stage1_binary/{train,val_cal,val_op,test}/
            {Normal,Defect}/*.png   (hardlinks — same inodes as originals)

Paths default to <repo_root>/data/... so this script is portable across machines.

Run:
    uv run python scripts/build_v3_stage1_binary_view.py           # incremental
    uv run python scripts/build_v3_stage1_binary_view.py --clean   # rebuild
"""
import argparse
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = REPO_ROOT / "data" / "sewerml_gate_v3_stage1"
DEFAULT_DST = REPO_ROOT / "data" / "sewerml_gate_v3_stage1_binary"
DEFECT_CLASSES = ["PF", "DE", "FS", "RB", "AF", "OB"]
SPLITS = ["train", "val_cal", "val_op", "test"]


def _link_or_skip(src: Path, dst: Path, counts: dict):
    if dst.exists():
        counts["skipped"] += 1
        return
    try:
        os.link(src, dst)
        counts["linked"] += 1
    except OSError as e:
        raise RuntimeError(
            f"hardlink failed: {src} -> {dst} ({e}). "
            f"Both paths must be on the same NTFS volume."
        ) from e


def build_split(split: str, src: Path, dst: Path) -> dict:
    src_split = src / split
    dst_split = dst / split
    normal_dst = dst_split / "Normal"
    defect_dst = dst_split / "Defect"
    normal_dst.mkdir(parents=True, exist_ok=True)
    defect_dst.mkdir(parents=True, exist_ok=True)

    counts = {"Normal": 0, "Defect": 0, "linked": 0, "skipped": 0}

    normal_src = src_split / "Normal"
    if not normal_src.exists():
        raise FileNotFoundError(normal_src)
    for p in normal_src.iterdir():
        if p.is_file():
            _link_or_skip(p, normal_dst / p.name, counts)
            counts["Normal"] += 1

    for cls in DEFECT_CLASSES:
        cls_dir = src_split / cls
        if not cls_dir.exists():
            continue
        for p in cls_dir.iterdir():
            if p.is_file():
                _link_or_skip(p, defect_dst / p.name, counts)
                counts["Defect"] += 1

    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help=f"7-folder source dataset (default: {DEFAULT_SRC})")
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST,
                    help=f"2-folder hardlink view destination (default: {DEFAULT_DST})")
    ap.add_argument("--clean", action="store_true", help="remove DST before building")
    args = ap.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()

    if not src.exists():
        raise FileNotFoundError(src)

    if args.clean and dst.exists():
        print(f"[setup] removing {dst}")
        shutil.rmtree(dst)

    dst.mkdir(parents=True, exist_ok=True)
    print(f"[setup] SRC = {src}")
    print(f"[setup] DST = {dst}\n")

    total = {"Normal": 0, "Defect": 0, "linked": 0, "skipped": 0}
    for split in SPLITS:
        c = build_split(split, src, dst)
        print(f"[{split:8s}] Normal={c['Normal']:>6d}  Defect={c['Defect']:>6d}  "
              f"linked={c['linked']:>6d}  skipped={c['skipped']:>6d}")
        for k in total:
            total[k] += c[k]

    print(f"\n[TOTAL   ] Normal={total['Normal']:>6d}  Defect={total['Defect']:>6d}  "
          f"linked={total['linked']:>6d}  skipped={total['skipped']:>6d}")
    print(f"\n[DONE] 2-class view ready at {dst}")


if __name__ == "__main__":
    main()
