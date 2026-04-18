"""
Export the 52k images referenced by v1/ manifests into a portable layout
for cold-storage backup or cross-machine transfer.

Target layout (7-class folder, rarity-priority multi-label assignment):
    DEST/
    ├── manifests/     (copies of the 7 v1/ metadata files)
    ├── train/{Normal, PF, DE, RB, AF, OB, FS}/     24,000
    ├── val_cal/{same}/                              2,400
    ├── val_op/{same}/                               5,600
    └── test/{same}/                                20,000

Priority order for multi-label frames (rarest first):
    PF > DE > RB > AF > OB > FS

A frame with multiple main-task labels is placed in the folder of its
RAREST label only (appears exactly once on disk).

For binary gate training: load Normal as y=0, union of the 6 defect
folders as y=1. No dedup needed.

For future 6-class / detection work: per-class folder gives "primary"
class for each frame; multi-label info is preserved in manifests/*_ids.csv.

Run:
    uv run python scripts/export_v1_images.py
"""
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

SRC_IMG_DIR = Path(r"C:/baidunetdiskdownload/sewerml_train_images")
V1_DIR = Path(r"C:/GitHub/YOLO-CV/research/materials/stage1_formal/manifests/v1")
DEST = Path(r"C:/Users/28898/Desktop/sewerml_gate_v1")

SPLITS = ["train", "val_cal", "val_op", "test"]
# Rarity priority (rarest first). A multi-label frame is assigned to the
# folder of its rarest main-task label only. Natural rarity order in
# Sewer-ML Train (approximate): PF ~16k < DE ~19k < RB ~46k < AF ~75k
# < OB ~184k < FS ~284k.
DEFECT_PRIORITY = ["PF", "DE", "RB", "AF", "OB", "FS"]
ALL_FOLDERS = ["Normal"] + DEFECT_PRIORITY
N_WORKERS = 8


def copy_one(src: Path, dst: Path):
    shutil.copyfile(src, dst)
    return dst


def assign_folder(row) -> str:
    if row["y"] == 0:
        return "Normal"
    for cls in DEFECT_PRIORITY:
        if row.get(cls, 0) == 1:
            return cls
    raise ValueError(f"row {row['image_id']} has y=1 but no main class: "
                     f"{dict((c, row.get(c)) for c in DEFECT_PRIORITY)}")


def export_split(name: str):
    df = pd.read_csv(V1_DIR / f"{name}_ids.csv", dtype={"image_id": str})
    df["image_id"] = df["image_id"].str.zfill(8)

    for folder in ALL_FOLDERS:
        (DEST / name / folder).mkdir(parents=True, exist_ok=True)

    tasks = []
    folder_counts = {f: 0 for f in ALL_FOLDERS}
    for _, row in df.iterrows():
        folder = assign_folder(row)
        folder_counts[folder] += 1
        src = SRC_IMG_DIR / f"{row.image_id}.png"
        dst = DEST / name / folder / f"{row.image_id}.png"
        tasks.append((src, dst))

    t0 = time.time()
    done = 0
    errors = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(copy_one, s, d): s for s, d in tasks}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                errors.append((futs[fut], repr(e)))
            done += 1
            if done % 2000 == 0:
                dt = time.time() - t0
                rate = done / dt if dt > 0 else 0
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                print(f"  [{name}] {done}/{len(tasks)}  {rate:.0f} files/s  ETA {eta:.0f}s")

    dt = time.time() - t0
    cnt_str = "  ".join(f"{f}={folder_counts[f]}" for f in ALL_FOLDERS)
    print(f"[{name}] DONE in {dt:.1f}s  (total {len(df)})")
    print(f"[{name}]   {cnt_str}")
    if errors:
        print(f"[{name}] ERRORS: {len(errors)} files failed, first 3: {errors[:3]}")
        raise RuntimeError(f"{len(errors)} copy failures in split {name}")


def copy_manifests():
    manifests_dir = DEST / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    files = list(V1_DIR.iterdir())
    for f in files:
        if f.is_file():
            shutil.copyfile(f, manifests_dir / f.name)
    print(f"[manifests] copied {len(files)} files to {manifests_dir}")


def main():
    # Clean existing desktop copy (only v1 subtree, nothing else on desktop)
    if DEST.exists():
        print(f"[setup] removing existing {DEST}")
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    t0 = time.time()
    copy_manifests()
    for split in SPLITS:
        export_split(split)
    print(f"\n[TOTAL] {time.time() - t0:.1f}s, target: {DEST}")


if __name__ == "__main__":
    main()
