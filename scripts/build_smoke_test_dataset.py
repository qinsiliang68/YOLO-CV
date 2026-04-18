"""
build_smoke_test_dataset.py — create tiny miniature of v3 Stage 1 data for local
pipeline smoke testing (~10 images per class per split).

Source: C:/Users/28898/Desktop/sewerml_gate_v3_stage1/
Target: C:/Users/28898/Desktop/sewerml_gate_v3_smoke/

Structure mirrors the real export so pipeline code paths are identical.
"""
import shutil
from pathlib import Path

SRC = Path(r"C:/Users/28898/Desktop/sewerml_gate_v3_stage1")
DST = Path(r"C:/Users/28898/Desktop/sewerml_gate_v3_smoke")

PER_CLASS_PER_SPLIT = 10
SPLITS = ["train", "val_cal", "val_op", "test"]
FOLDERS = ["Normal", "PF", "DE", "RB", "AF", "OB", "FS"]


def main():
    if DST.exists():
        print(f"[setup] removing existing {DST}")
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    # copy manifests (just for reference, smoke test doesn't need them)
    if (SRC / "manifests").exists():
        shutil.copytree(SRC / "manifests", DST / "manifests")

    total = 0
    for split in SPLITS:
        for folder in FOLDERS:
            src_dir = SRC / split / folder
            dst_dir = DST / split / folder
            dst_dir.mkdir(parents=True, exist_ok=True)
            if not src_dir.exists():
                continue
            files = sorted(f for f in src_dir.iterdir() if f.name.endswith(".png"))
            n_take = min(PER_CLASS_PER_SPLIT, len(files))
            for f in files[:n_take]:
                shutil.copyfile(f, dst_dir / f.name)
            total += n_take
            print(f"  {split}/{folder}: took {n_take}/{len(files)}")
    print(f"\n[DONE] {total} files copied to {DST}")


if __name__ == "__main__":
    main()
