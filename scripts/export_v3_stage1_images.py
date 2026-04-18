"""
Export v3 Stage 1 (binary gate) training package to desktop.

Structure (ready for YOLO11-cls training on 4 GPU machines):
    DEST/
    ├── manifests/                         (all 6 v3 CSVs + manifest.json + README)
    ├── train/     24,000 (normal_stage1_ids + defect_ids)
    │   ├── Normal/    12,000
    │   ├── PF/         2,000
    │   ├── DE/         2,000
    │   ├── RB/         2,000
    │   ├── AF/         2,000
    │   ├── OB/         2,000
    │   └── FS/         2,000
    ├── val_cal/   2,400 (7 folders, natural distribution)
    ├── val_op/    5,600 (7 folders, natural)
    └── test/     20,000 (7 folders, natural)

Rarity priority for multi-label frames: PF > DE > RB > AF > OB > FS
Each frame goes into exactly one folder (its rarest main-task label).

Run:
    uv run python scripts/export_v3_stage1_images.py
"""
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

SRC_IMG_DIR = Path(r"C:/baidunetdiskdownload/sewerml_train_images")
V3_DIR = Path(r"C:/GitHub/YOLO-CV/research/materials/stage1_formal/manifests/v3")
DEST = Path(r"C:/Users/28898/Desktop/sewerml_gate_v3_stage1")

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
    raise ValueError(f"row {row['image_id']} has y=1 but no main class")


def load_split(name: str) -> pd.DataFrame:
    """
    For 'train': concatenate defect_ids.csv + normal_stage1_ids.csv.
    For val_cal / val_op / test: load the single CSV directly.
    """
    if name == "train":
        defect = pd.read_csv(V3_DIR / "defect_ids.csv", dtype={"image_id": str})
        normal = pd.read_csv(V3_DIR / "normal_stage1_ids.csv", dtype={"image_id": str})
        df = pd.concat([defect, normal], ignore_index=True)
    else:
        df = pd.read_csv(V3_DIR / f"{name}_ids.csv", dtype={"image_id": str})
    df["image_id"] = df["image_id"].str.zfill(8)
    return df


def export_split(name: str):
    df = load_split(name)

    for folder in ALL_FOLDERS:
        (DEST / name / folder).mkdir(parents=True, exist_ok=True)

    tasks = []
    counts = {f: 0 for f in ALL_FOLDERS}
    for _, row in df.iterrows():
        folder = assign_folder(row)
        counts[folder] += 1
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
    cnt_str = "  ".join(f"{f}={counts[f]}" for f in ALL_FOLDERS)
    print(f"[{name}] DONE in {dt:.1f}s  (total {len(df)})")
    print(f"[{name}]   {cnt_str}")
    if errors:
        print(f"[{name}] ERRORS: {len(errors)}")
        raise RuntimeError(f"{len(errors)} copy failures in split {name}")


def copy_manifests():
    manifests_dir = DEST / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    files = [f for f in V3_DIR.iterdir() if f.is_file()]
    for f in files:
        shutil.copyfile(f, manifests_dir / f.name)
    print(f"[manifests] copied {len(files)} files to {manifests_dir}")


def main():
    if DEST.exists():
        print(f"[setup] removing existing {DEST}")
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    t0 = time.time()
    copy_manifests()
    for split in ["train", "val_cal", "val_op", "test"]:
        export_split(split)
    print(f"\n[TOTAL] {time.time() - t0:.1f}s, target: {DEST}")


if __name__ == "__main__":
    main()
