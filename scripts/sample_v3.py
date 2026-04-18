"""
sample_v3.py — v3 two-stage balanced sampling.

Design philosophy: train-side intervention, eval-side observation.

Stage 1 (essay3, binary gate):
    - 12,000 train = 6,000 defect + 6,000 Normal (1:1 balanced)
    - 1000/main class × 6, rarity-priority assignment
    - Addresses rare-class underlearning in v1 natural (PF 208 → 1000)

Stage 2 (essay4, object detection, future):
    - 8,000 train = 6,000 defect (same) + 2,000 Normal (1:3 pos:neg)
    - normal_stage2 is STRICT SUBSET of normal_stage1 (first 2,000)
    - One defect sample serves both stages; only Normal count differs

Val / Test: copied from v1 verbatim (natural distribution).
    - test: mirrors deployment
    - val: when train is balanced, natural val is the only unbiased ruler
      that exposes training-induced bias during epoch selection

Run:
    uv run python scripts/sample_v3.py
"""
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "v3"
SEED = 20260606

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = REPO_ROOT / "YOLOv11" / "datasets" / "sewerml_annotations" / "SewerML_Train.csv"
IMG_DIR = Path(r"C:/baidunetdiskdownload/sewerml_train_images")
V1_DIR = REPO_ROOT / "research" / "materials" / "stage1_formal" / "manifests" / "v1"
OUT_DIR = REPO_ROOT / "research" / "materials" / "stage1_formal" / "manifests" / "v3"

MAIN_CLASSES = ["PF", "DE", "FS", "RB", "AF", "OB"]
HOLDOUT_CLASSES = ["BE", "RO", "IN", "FO"]
PRIORITY = ["PF", "DE", "RB", "AF", "OB", "FS"]

K_PER_CLASS = 1_000
N_NORMAL_STAGE1 = 6_000
N_NORMAL_STAGE2 = 2_000


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_v1_excluded():
    excluded = set()
    for name in ["val_cal", "val_op", "test"]:
        df = pd.read_csv(V1_DIR / f"{name}_ids.csv", dtype={"image_id": str})
        df["image_id"] = df["image_id"].str.zfill(8)
        excluded.update(df.image_id.tolist())
    print(f"[exclude] {len(excluded):,} image_ids from v1 val/test")
    return excluded


def load_and_filter(csv_path, excluded):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"[load] raw: {len(df):,}")
    df = df[(df["OK"] == 0) & (df["PH"] == 0)].copy()
    print(f"[filter] after OK/PH: {len(df):,}")
    has_main = (df[MAIN_CLASSES].sum(axis=1) >= 1)
    is_normal = (df["Defect"] == 0)
    df = df[has_main | is_normal].copy()
    print(f"[filter] after non-main-defect: {len(df):,}")
    df["y"] = (df[MAIN_CLASSES].sum(axis=1) >= 1).astype(int)
    df["image_id"] = df["Filename"].str.replace(r"\.png$", "", regex=True)
    before = len(df)
    df = df[~df["image_id"].isin(excluded)].copy()
    print(f"[filter] after v1 val/test exclude: {len(df):,} (dropped {before - len(df):,})")
    return df


def sample_defects(df, seed):
    rng = np.random.default_rng(seed)
    dfs = []
    assigned = set()
    print(f"\n[defect] K={K_PER_CLASS} per class (rarity priority):")
    for cls in PRIORITY:
        mask = (df[cls] == 1) & (~df["image_id"].isin(assigned))
        pool = df[mask]
        if len(pool) < K_PER_CLASS:
            raise RuntimeError(f"{cls}: pool {len(pool):,} < K {K_PER_CLASS}")
        indices = rng.permutation(len(pool))[:K_PER_CLASS]
        sel = pool.iloc[indices].copy()
        dfs.append(sel)
        assigned.update(sel.image_id.tolist())
        print(f"  {cls}: pool={len(pool):>6,}  ->  {K_PER_CLASS:,}")
    defects = pd.concat(dfs, ignore_index=True)
    return defects, assigned


def sample_normal(df, defect_ids, seed):
    rng = np.random.default_rng(seed + 1)
    pool = df[(df["y"] == 0) & (~df["image_id"].isin(defect_ids))]
    if len(pool) < N_NORMAL_STAGE1:
        raise RuntimeError(f"Normal: pool {len(pool)} < {N_NORMAL_STAGE1}")
    indices = rng.permutation(len(pool))[:N_NORMAL_STAGE1]
    stage1 = pool.iloc[indices].reset_index(drop=True)
    stage2 = stage1.iloc[:N_NORMAL_STAGE2].copy().reset_index(drop=True)
    print(f"\n[normal] stage1: pool={len(pool):,}  ->  {N_NORMAL_STAGE1:,}")
    print(f"[normal] stage2: first {N_NORMAL_STAGE2:,} of stage1 (strict subset)")
    return stage1, stage2


def copy_v1_valtest(out_dir):
    result = {}
    for name in ["val_cal", "val_op", "test"]:
        src = V1_DIR / f"{name}_ids.csv"
        dst = out_dir / f"{name}_ids.csv"
        shutil.copyfile(src, dst)
        sha = sha256_of_file(dst)
        result[name] = sha
        print(f"[copy] v1/{name}_ids.csv  ->  v3/  sha={sha[:12]}...")
    return result


def assert_integrity(defects, normal_stage1, normal_stage2, img_dir):
    # 1. sizes
    assert len(defects) == K_PER_CLASS * 6, f"defects {len(defects)} != {K_PER_CLASS*6}"
    assert len(normal_stage1) == N_NORMAL_STAGE1
    assert len(normal_stage2) == N_NORMAL_STAGE2

    d_ids = set(defects.image_id)
    s1_ids = set(normal_stage1.image_id)
    s2_ids = set(normal_stage2.image_id)

    # 2. stage2 strict subset of stage1
    assert s2_ids.issubset(s1_ids), "stage2 Normal not subset of stage1"

    # 3. defect vs normal disjoint
    assert not (d_ids & s1_ids), f"defect overlaps normal_stage1: {len(d_ids & s1_ids)}"

    # 4. uniqueness within each pool
    assert defects.image_id.is_unique, "duplicate image_ids in defects"
    assert normal_stage1.image_id.is_unique, "duplicate image_ids in normal_stage1"

    # 5. disjoint from v1 val/test
    for name in ["val_cal", "val_op", "test"]:
        v1 = pd.read_csv(V1_DIR / f"{name}_ids.csv", dtype={"image_id": str})
        v1_ids = set(v1.image_id.str.zfill(8))
        assert not (d_ids & v1_ids), f"defects overlap v1/{name}"
        assert not (s1_ids & v1_ids), f"normal_stage1 overlaps v1/{name}"

    # 6. disk presence
    print("[assert] scanning disk...")
    disk = set()
    for entry in os.scandir(img_dir):
        if entry.name.endswith(".png"):
            disk.add(entry.name[:-4])
    missing_d = d_ids - disk
    missing_n = s1_ids - disk
    assert not missing_d, f"defects missing on disk: {len(missing_d)}, e.g. {list(missing_d)[:3]}"
    assert not missing_n, f"normal missing on disk: {len(missing_n)}"

    # 7. seed
    assert SEED == 20260606
    print("[assert] ALL 7 CHECKS PASSED")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[setup] output: {OUT_DIR}")

    excluded = load_v1_excluded()
    df = load_and_filter(SOURCE_CSV, excluded)

    defects, defect_ids = sample_defects(df, SEED)
    normal_stage1, normal_stage2 = sample_normal(df, defect_ids, SEED)

    # shuffle defects for storage (avoid class-ordering artifact)
    rng_out = np.random.default_rng(SEED + 2)
    defects = defects.iloc[rng_out.permutation(len(defects))].reset_index(drop=True)

    print(f"\n[summary]")
    print(f"  defect_ids:        {len(defects):>6,}  y=1 fixed")
    print(f"  normal_stage1_ids: {len(normal_stage1):>6,}  y=0 fixed")
    print(f"  normal_stage2_ids: {len(normal_stage2):>6,}  y=0 (subset)")
    print(f"  stage 1 train:     {len(defects)+len(normal_stage1):>6,}  ratio 1:1")
    print(f"  stage 2 train:     {len(defects)+len(normal_stage2):>6,}  ratio 3:1 (defect:normal)")

    cols = ["image_id", "y", "WaterLevel"] + MAIN_CLASSES + HOLDOUT_CLASSES + ["Defect"]
    defects[cols].to_csv(OUT_DIR / "defect_ids.csv", index=False)
    normal_stage1[cols].to_csv(OUT_DIR / "normal_stage1_ids.csv", index=False)
    normal_stage2[cols].to_csv(OUT_DIR / "normal_stage2_ids.csv", index=False)

    sha_d = sha256_of_file(OUT_DIR / "defect_ids.csv")
    sha_s1 = sha256_of_file(OUT_DIR / "normal_stage1_ids.csv")
    sha_s2 = sha256_of_file(OUT_DIR / "normal_stage2_ids.csv")
    print(f"\n[write] defect_ids.csv         sha={sha_d[:12]}...")
    print(f"[write] normal_stage1_ids.csv  sha={sha_s1[:12]}...")
    print(f"[write] normal_stage2_ids.csv  sha={sha_s2[:12]}...")

    print()
    val_sha = copy_v1_valtest(OUT_DIR)

    # cooccurrence (defect pool)
    rows = []
    for m in MAIN_CLASSES:
        for h in HOLDOUT_CLASSES:
            rows.append({"split": "defect_pool", "main": m, "holdout": h,
                         "count": int(((defects[m] == 1) & (defects[h] == 1)).sum())})
    pd.DataFrame(rows).to_csv(OUT_DIR / "cooccurrence_matrix.csv", index=False)
    print(f"[cooc] written")

    print()
    assert_integrity(defects, normal_stage1, normal_stage2, IMG_DIR)

    manifest = {
        "protocol_version": VERSION,
        "seed": SEED,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "design_philosophy": "train-side intervention, eval-side observation",
        "stages": {
            "stage1_binary_gate_essay3": {
                "train_files": ["defect_ids.csv", "normal_stage1_ids.csv"],
                "n_total": K_PER_CLASS * 6 + N_NORMAL_STAGE1,
                "pos_neg_ratio": "1:1",
            },
            "stage2_detection_essay4_future": {
                "train_files": ["defect_ids.csv", "normal_stage2_ids.csv"],
                "n_total": K_PER_CLASS * 6 + N_NORMAL_STAGE2,
                "pos_neg_ratio": "3:1 defect:normal (normal as background)",
                "note": "normal_stage2 is strict subset of normal_stage1 (first 2000 after shuffle)",
            },
        },
        "val_test_source": "v1 (natural distribution, copied verbatim)",
        "pools": {
            "defect_ids":        {"n": len(defects),       "k_per_class": K_PER_CLASS, "sha256": sha_d},
            "normal_stage1_ids": {"n": len(normal_stage1), "sha256": sha_s1},
            "normal_stage2_ids": {"n": len(normal_stage2), "sha256": sha_s2,
                                   "note": "strict subset of normal_stage1"},
            "val_cal": {"source": "v1", "sha256": val_sha["val_cal"]},
            "val_op":  {"source": "v1", "sha256": val_sha["val_op"]},
            "test":    {"source": "v1", "sha256": val_sha["test"]},
        },
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[manifest] written")

    readme = f"""# v3 Sampling Output (Two-stage Balanced)

Generated: {manifest['generated_at']}
Seed: {SEED}

## Design Philosophy

**Train-side intervention, eval-side observation.**

### Stage 1 (essay3, binary gate) — 12,000 train

- 6,000 defect (1000/class × 6) + 6,000 Normal
- 1:1 pos:neg balance
- Each main class guaranteed 1,000 (vs v1 natural's PF=208/DE=402)

### Stage 2 (essay4, object detection, future) — 8,000 train

- 6,000 defect (same pool as Stage 1) + 2,000 Normal
- 1:3 pos:neg (Normal as background)
- normal_stage2 is STRICT SUBSET of normal_stage1 (first 2,000 after shuffle)

## Val / Test (copied from v1)

Natural distribution preserved:
- **test**: mirrors deployment; only interpretation for real-world performance
- **val**: when train is balanced, natural val is the only unbiased ruler
  that exposes training-induced bias during epoch selection

## Integrity

- v3 defects + normal drawn from pool EXCLUDING v1 val/test ids
- 7 assertions pass: size, subset, disjoint, uniqueness, disk, seed

## WARNING — test_ids.csv

**test_ids.csv must NOT be read during development.**
See repo-root LEAKAGE_AUDIT.md L1-8.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(f"\n[DONE] outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
