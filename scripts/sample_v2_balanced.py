"""
sample_v2_balanced.py — v2 sub-class balanced sampling (1:6 ratio).

Design intent: the dataset is built primarily for essay4 (object detection),
where each defect class needs equal representation to avoid class bias.
The binary gate task (essay3) reuses the same data as a class-balanced
comparison against v1's natural-distribution training.

Train: 24,500 frames, 7 equal buckets (3,500 each)
    - Normal (y=0):  3,500
    - PF, DE, RB, AF, OB, FS (y=1):  3,500 each under rarity priority
    - Normal : defect_total = 1 : 6 (user-specified)

Val / Test: copied from v1 (natural distribution for evaluation realism).

Integrity: v2 train image_ids are drawn from a pool that EXCLUDES v1
val_cal/val_op/test ids, guaranteeing v2 train disjoint from v1 val/test.

Run:
    uv run python scripts/sample_v2_balanced.py
"""
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "v2"
SEED = 20260606

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = REPO_ROOT / "YOLOv11" / "datasets" / "sewerml_annotations" / "SewerML_Train.csv"
IMG_DIR = Path(r"C:/baidunetdiskdownload/sewerml_train_images")
V1_DIR = REPO_ROOT / "research" / "materials" / "stage1_formal" / "manifests" / "v1"
OUT_DIR = REPO_ROOT / "research" / "materials" / "stage1_formal" / "manifests" / "v2"

MAIN_CLASSES = ["PF", "DE", "FS", "RB", "AF", "OB"]
HOLDOUT_CLASSES = ["BE", "RO", "IN", "FO"]
# Rarity priority: PF (rarest) > DE > RB > AF > OB > FS
PRIORITY = ["PF", "DE", "RB", "AF", "OB", "FS"]

K_PER_CLASS = 3_500          # per-class target (7 classes total: Normal + 6 defects)
N_NORMAL = K_PER_CLASS
N_TRAIN = K_PER_CLASS * (len(MAIN_CLASSES) + 1)   # 7 × 3,500 = 24,500


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_v1_excluded_ids():
    excluded = set()
    for name in ["val_cal", "val_op", "test"]:
        df = pd.read_csv(V1_DIR / f"{name}_ids.csv", dtype={"image_id": str})
        df["image_id"] = df["image_id"].str.zfill(8)
        excluded.update(df.image_id.tolist())
    print(f"[exclude] {len(excluded):,} image_ids from v1 val/test")
    return excluded


def load_and_filter(csv_path, excluded_ids):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    n_raw = len(df)
    print(f"[load] raw: {n_raw:,}")

    df = df[(df["OK"] == 0) & (df["PH"] == 0)].copy()
    print(f"[filter] after OK/PH drop: {len(df):,}")

    has_main = (df[MAIN_CLASSES].sum(axis=1) >= 1)
    is_normal = (df["Defect"] == 0)
    df = df[has_main | is_normal].copy()
    print(f"[filter] after non-main-defect exclude: {len(df):,}")

    df["y"] = (df[MAIN_CLASSES].sum(axis=1) >= 1).astype(int)
    df["image_id"] = df["Filename"].str.replace(r"\.png$", "", regex=True)

    before = len(df)
    df = df[~df["image_id"].isin(excluded_ids)].copy()
    print(f"[filter] after v1 val/test exclude: {len(df):,} (dropped {before - len(df):,})")

    return df


def sample_balanced_train(df, seed):
    rng = np.random.default_rng(seed)
    selected_dfs = []
    assigned_ids = set()

    print()
    print(f"[sample] per-class K={K_PER_CLASS} under rarity priority:")
    for cls in PRIORITY:
        mask = (df[cls] == 1) & (~df["image_id"].isin(assigned_ids))
        pool = df[mask]
        if len(pool) < K_PER_CLASS:
            raise RuntimeError(f"{cls}: pool size {len(pool):,} < K {K_PER_CLASS:,}")
        indices = rng.permutation(len(pool))[:K_PER_CLASS]
        selected = pool.iloc[indices].copy()
        selected_dfs.append(selected)
        assigned_ids.update(selected.image_id.tolist())
        print(f"  {cls}: pool={len(pool):>6,}  ->  selected {K_PER_CLASS:,}")

    normal_pool = df[(df["y"] == 0) & (~df["image_id"].isin(assigned_ids))]
    if len(normal_pool) < N_NORMAL:
        raise RuntimeError(f"Normal: pool size {len(normal_pool):,} < N {N_NORMAL:,}")
    indices = rng.permutation(len(normal_pool))[:N_NORMAL]
    selected_normal = normal_pool.iloc[indices].copy()
    selected_dfs.append(selected_normal)
    print(f"  Normal: pool={len(normal_pool):>6,}  ->  selected {N_NORMAL:,}")

    train = pd.concat(selected_dfs, ignore_index=True)
    perm = rng.permutation(len(train))
    return train.iloc[perm].reset_index(drop=True)


def copy_v1_valtest(out_dir):
    print()
    for name in ["val_cal", "val_op", "test"]:
        src = V1_DIR / f"{name}_ids.csv"
        dst = out_dir / f"{name}_ids.csv"
        shutil.copyfile(src, dst)
        print(f"[copy] v1/{name}_ids.csv  ->  v2/  (sha={sha256_of_file(dst)[:12]}...)")


def assert_integrity(train_df, img_dir):
    # 1. size exact
    assert len(train_df) == N_TRAIN, f"train {len(train_df)} != {N_TRAIN}"

    # 2. no duplicates within train
    assert train_df.image_id.is_unique, "duplicate image_ids in train"

    # 3. y distribution matches 1:6
    n_y1 = int(train_df["y"].sum())
    n_y0 = N_TRAIN - n_y1
    assert n_y1 == K_PER_CLASS * len(MAIN_CLASSES), f"y=1 count {n_y1} != {K_PER_CLASS*6}"
    assert n_y0 == N_NORMAL, f"y=0 count {n_y0} != {N_NORMAL}"

    # 4. disjoint from v1 val/test
    for name in ["val_cal", "val_op", "test"]:
        v1 = pd.read_csv(V1_DIR / f"{name}_ids.csv", dtype={"image_id": str})
        v1["image_id"] = v1["image_id"].str.zfill(8)
        overlap = set(train_df.image_id) & set(v1.image_id)
        assert not overlap, f"v2 train overlaps v1/{name}: {len(overlap)}"

    # 5. disk presence
    print("[assert] scanning disk...")
    disk = set()
    for entry in os.scandir(img_dir):
        if entry.name.endswith(".png"):
            disk.add(entry.name[:-4])
    missing = set(train_df.image_id) - disk
    assert not missing, f"{len(missing)} missing, e.g. {list(missing)[:3]}"

    # 6. seed
    assert SEED == 20260606
    print("[assert] ALL 6 CHECKS PASSED")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[setup] output: {OUT_DIR}")

    excluded = load_v1_excluded_ids()
    df = load_and_filter(SOURCE_CSV, excluded)
    train = sample_balanced_train(df, SEED)

    print()
    print(f"[train] total {len(train):,}  y=1 {int(train.y.sum()):,}  y=0 {int((train.y==0).sum()):,}")
    print(f"[train] per-class breakdown (rarity-priority assigned):")
    assigned = set()
    for cls in PRIORITY:
        mask = (train[cls] == 1) & (~train["image_id"].isin(assigned))
        n = int(mask.sum())
        assigned.update(train.loc[mask, "image_id"].tolist())
        print(f"         {cls}: {n:,}")
    n_normal_in_train = int((train["y"] == 0).sum())
    print(f"         Normal: {n_normal_in_train:,}")

    # write train
    cols = ["image_id", "y", "WaterLevel"] + MAIN_CLASSES + HOLDOUT_CLASSES + ["Defect"]
    train_path = OUT_DIR / "train_ids.csv"
    train[cols].to_csv(train_path, index=False)
    train_sha = sha256_of_file(train_path)
    print()
    print(f"[write] train_ids.csv: {len(train):,} rows  sha={train_sha[:12]}...")

    copy_v1_valtest(OUT_DIR)

    # cooccurrence (train only)
    rows = []
    for m in MAIN_CLASSES:
        for h in HOLDOUT_CLASSES:
            rows.append({"split": "train", "main": m, "holdout": h,
                         "count": int(((train[m] == 1) & (train[h] == 1)).sum())})
    pd.DataFrame(rows).to_csv(OUT_DIR / "cooccurrence_matrix.csv", index=False)
    print(f"[cooc] written")

    print()
    assert_integrity(train, IMG_DIR)

    manifest = {
        "protocol_version": VERSION,
        "seed": SEED,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_strategy": (
            f"sub-class balanced: {K_PER_CLASS} per main class (rarity-priority) "
            f"+ {N_NORMAL} Normal = {N_TRAIN} total (Normal:defect_total = 1:6)"
        ),
        "val_test_source": "v1 (natural distribution, copied for eval-time realism)",
        "splits": {
            "train":   {"n_frames": len(train),
                        "y_ratio": round(float(train.y.mean()), 4),
                        "sha256": train_sha},
            "val_cal": {"source": "v1/val_cal_ids.csv",
                        "sha256": sha256_of_file(OUT_DIR / "val_cal_ids.csv")},
            "val_op":  {"source": "v1/val_op_ids.csv",
                        "sha256": sha256_of_file(OUT_DIR / "val_op_ids.csv")},
            "test":    {"source": "v1/test_ids.csv",
                        "sha256": sha256_of_file(OUT_DIR / "test_ids.csv")},
        },
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[manifest] {OUT_DIR / 'manifest.json'}")

    (OUT_DIR / "README.md").write_text(
        f"# v2 Sampling Output (Sub-class Balanced Train, 1:6)\n\n"
        f"Generated: {manifest['generated_at']}\n"
        f"Seed: {SEED}\n\n"
        f"## Strategy\n\n"
        f"- train ({N_TRAIN:,}): 7 equal buckets, {K_PER_CLASS} each\n"
        f"  - Normal (y=0): {N_NORMAL:,}\n"
        f"  - PF/DE/FS/RB/AF/OB (y=1): {K_PER_CLASS:,} each under rarity priority\n"
        f"  - Normal:defect_total = 1:6\n"
        f"- val_cal / val_op / test: copied from v1 (natural distribution)\n\n"
        f"## Purpose\n\n"
        f"- essay3 (binary gate): head-to-head comparison against v1 natural training\n"
        f"- essay4 (object detection): primary balanced training data for 6-class localization\n\n"
        f"## Integrity\n\n"
        f"- v2 train drawn from pool excluding v1 val/test ids\n"
        f"- v2 train disjoint from v1 val_cal/val_op/test (asserted)\n"
        f"- Same seed as v1 (20260606); v1 val/test IDs identical via copy\n\n"
        f"## WARNING - test_ids.csv\n\n"
        f"**test_ids.csv must NOT be read during development.**\n"
        f"See repo-root LEAKAGE_AUDIT.md L1-8.\n",
        encoding="utf-8",
    )

    print(f"\n[DONE] all outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
