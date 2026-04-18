"""
sample_v1.py — Sewer-ML gate binary task sampling (protocol v1).

Implements sampling_protocol_v1:
- Source: SewerML_Train.csv (single pool)
- Method: frame-level simple random sampling, seed=20260606
- No stratification, no groupwise, no per-inspection cap
- Outputs: 4 split CSVs + cooccurrence matrix + manifest.json + README.md
- Validation: 6 integrity assertions, fail-fast

Run:
    uv run python scripts/sample_v1.py
"""
import hashlib
import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---- fixed configuration (mirrors sampling_protocol_v1.yaml) ----
VERSION = "v1"
SEED = 20260606

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = REPO_ROOT / "YOLOv11" / "datasets" / "sewerml_annotations" / "SewerML_Train.csv"
IMG_DIR = Path(r"C:/baidunetdiskdownload/sewerml_train_images")
OUT_DIR = REPO_ROOT / "research" / "materials" / "stage1_formal" / "manifests" / "v1"
PROTOCOL_FILE = REPO_ROOT / "SAMPLING_PROTOCOL.md"

MAIN_CLASSES = ["PF", "DE", "FS", "RB", "AF", "OB"]
HOLDOUT_CLASSES = ["BE", "RO", "IN", "FO"]

TARGETS = {
    "train":    24_000,
    "val_cal":   2_400,
    "val_op":    5_600,
    "test":     20_000,
}
N_TOTAL = sum(TARGETS.values())  # 52,000


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_and_filter(csv_path: Path):
    print(f"[load] {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    n_raw = len(df)
    print(f"[load] raw rows: {n_raw:,}")

    # Step 1: drop quality-problematic frames (OK or PH flagged)
    df_q = df[(df["OK"] == 0) & (df["PH"] == 0)].copy()
    n_q = len(df_q)
    print(f"[filter] after OK/PH drop: {n_q:,} (dropped {n_raw - n_q:,} = {(n_raw - n_q)/n_raw:.1%})")

    # Step 2: keep only frames with unambiguous binary label
    #   - has_main (any of 6 main classes = 1) → y = 1
    #   - pure normal (Defect = 0) → y = 0
    #   - drop "defect but no main" (only VA/GR/PB/OS/OP/BE/RO/IN/FO/IS labeled)
    has_main = (df_q[MAIN_CLASSES].sum(axis=1) >= 1)
    is_normal = (df_q["Defect"] == 0)
    keep_mask = has_main | is_normal
    df_k = df_q[keep_mask].copy()
    n_k = len(df_k)
    print(f"[filter] after non-main-defect exclude: {n_k:,} (dropped {n_q - n_k:,})")

    df_k["y"] = (df_k[MAIN_CLASSES].sum(axis=1) >= 1).astype(int)
    df_k["image_id"] = df_k["Filename"].str.replace(r"\.png$", "", regex=True)

    y1 = int(df_k["y"].sum())
    y0 = n_k - y1
    print(f"[pool] y=0: {y0:,} ({y0/n_k:.1%}) | y=1: {y1:,} ({y1/n_k:.1%})")

    return df_k, {
        "n_raw": n_raw,
        "n_after_quality": n_q,
        "n_pool": n_k,
        "y_ratio_in_pool": round(float(df_k["y"].mean()), 4),
    }


def split_frames(df: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    sampled = df.iloc[perm[:N_TOTAL]].reset_index(drop=True)

    cursor = 0
    splits = {}
    for name in ["train", "val_cal", "val_op", "test"]:
        target = TARGETS[name]
        splits[name] = sampled.iloc[cursor:cursor + target].copy()
        cursor += target
    return splits


def build_cooccurrence(splits):
    rows = []
    for split_name, df in splits.items():
        for m in MAIN_CLASSES:
            for h in HOLDOUT_CLASSES:
                count = int(((df[m] == 1) & (df[h] == 1)).sum())
                rows.append({"split": split_name, "main": m, "holdout": h, "count": count})
    return pd.DataFrame(rows)


def write_splits(splits, out_dir: Path):
    cols = ["image_id", "y", "WaterLevel"] + MAIN_CLASSES + HOLDOUT_CLASSES + ["Defect"]
    sha_map = {}
    for name, df in splits.items():
        path = out_dir / f"{name}_ids.csv"
        df[cols].to_csv(path, index=False)
        sha_map[name] = {
            "n_frames": len(df),
            "y_ratio": round(float(df["y"].mean()), 4),
            "sha256": sha256_of_file(path),
        }
        print(f"[write] {path.name}: {len(df):,} rows, sha256={sha_map[name]['sha256'][:12]}...")
    return sha_map


def assert_integrity(splits, img_dir: Path):
    # 1. pairwise image_id disjoint across splits
    for (a_n, a), (b_n, b) in itertools.combinations(splits.items(), 2):
        overlap = set(a.image_id) & set(b.image_id)
        assert not overlap, f"image_id leak {a_n} x {b_n}: {len(overlap)} overlapping"

    # 2. no duplicate image_id within any split
    for name, df in splits.items():
        assert df.image_id.is_unique, f"{name}: duplicate image_ids"

    # 3. y_ratio in plausible band per split
    for name, df in splits.items():
        r = df.y.mean()
        assert 0.20 <= r <= 0.80, f"{name}: y_ratio={r:.3f} outside [0.20, 0.80]"

    # 4. size exactly equals target
    for name, df in splits.items():
        assert len(df) == TARGETS[name], f"{name}: size={len(df)} != target {TARGETS[name]}"

    # 5. all sampled image files physically present on disk
    print("[assert] scanning disk image set...")
    disk_images = set()
    for entry in os.scandir(img_dir):
        if entry.name.endswith(".png"):
            disk_images.add(entry.name[:-4])
    print(f"[assert] {len(disk_images):,} png files on disk")
    for name, df in splits.items():
        missing = set(df.image_id) - disk_images
        assert not missing, f"{name}: {len(missing)} images missing on disk, e.g. {list(missing)[:3]}"

    # 6. seed recorded
    assert SEED == 20260606

    print("[assert] ALL 6 CHECKS PASSED")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[setup] output dir: {OUT_DIR}")

    df, pool_stats = load_and_filter(SOURCE_CSV)
    splits = split_frames(df, SEED)

    print()
    for name, d in splits.items():
        print(f"[split] {name:<8} n={len(d):>6,}  y_ratio={d.y.mean():.4f}")

    print()
    sha_map = write_splits(splits, OUT_DIR)

    cooc = build_cooccurrence(splits)
    cooc_path = OUT_DIR / "cooccurrence_matrix.csv"
    cooc.to_csv(cooc_path, index=False)
    print(f"[cooc] written {len(cooc)} rows to {cooc_path.name}")

    print()
    assert_integrity(splits, IMG_DIR)

    manifest = {
        "protocol_version": VERSION,
        "protocol_sha256": sha256_of_file(PROTOCOL_FILE) if PROTOCOL_FILE.exists() else "n/a",
        "seed": SEED,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_pool": {"csv_path": str(SOURCE_CSV), **pool_stats},
        "splits": sha_map,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[manifest] {OUT_DIR / 'manifest.json'}")

    (OUT_DIR / "README.md").write_text(
        f"# v1 Sampling Output\n\n"
        f"Generated: {manifest['generated_at']}\n"
        f"Seed: {SEED}\n"
        f"Protocol: v1 (see repo root SAMPLING_PROTOCOL.md)\n\n"
        f"## WARNING - test_ids.csv\n\n"
        f"**test_ids.csv must NOT be read during development.**\n"
        f"Any dev script reading test_ids.csv must set env FINAL_EVAL=1 or abort.\n"
        f"See LEAKAGE_AUDIT.md L1-8.\n",
        encoding="utf-8",
    )

    print(f"\n[DONE] all outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
