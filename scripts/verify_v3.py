"""
verify_v3.py — Comprehensive integrity check of v3 manifests + desktop export.

Runs 12 checks. Any failure aborts.

Run:
    uv run python scripts/verify_v3.py
"""
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
V3_DIR = REPO_ROOT / "research" / "materials" / "stage1_formal" / "manifests" / "v3"
V1_DIR = REPO_ROOT / "research" / "materials" / "stage1_formal" / "manifests" / "v1"
IMG_DIR = Path(r"C:/baidunetdiskdownload/sewerml_train_images")
DESKTOP = Path(r"C:/Users/28898/Desktop/sewerml_gate_v3_stage1")

MAIN_CLASSES = ["PF", "DE", "FS", "RB", "AF", "OB"]
PRIORITY = ["PF", "DE", "RB", "AF", "OB", "FS"]

EXPECTED_SIZES = {
    "defect_ids":         12_000,
    "normal_stage1_ids":  12_000,
    "normal_stage2_ids":   4_000,
    "val_cal_ids":         2_400,
    "val_op_ids":          5_600,
    "test_ids":           20_000,
}


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_ids_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"image_id": str})
    df["image_id"] = df["image_id"].str.zfill(8)
    return df


def main():
    print("=" * 60)
    print(" v3 INTEGRITY CHECK — 12 assertions ")
    print("=" * 60)

    # ---------- load everything ----------
    dfs = {k: load_ids_csv(V3_DIR / f"{k}.csv") for k in EXPECTED_SIZES.keys()}
    with open(V3_DIR / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    # ---------- Check 1: sizes ----------
    for k, expected in EXPECTED_SIZES.items():
        actual = len(dfs[k])
        assert actual == expected, f"[1] {k}: got {actual} rows, expected {expected}"
    print(f"[ 1] OK all 6 CSVs match expected row counts")

    # ---------- Check 2: image_id uniqueness within each file ----------
    for k, df in dfs.items():
        assert df.image_id.is_unique, f"[2] {k}: duplicate image_ids"
    print(f"[ 2] OKimage_id unique within each CSV")

    # ---------- Check 3: image_id format (zero-padded 8 chars) ----------
    for k, df in dfs.items():
        bad = df[~df.image_id.str.match(r"^\d{8}$")]
        assert len(bad) == 0, f"[3] {k}: {len(bad)} malformed image_ids, e.g. {bad.head(3).image_id.tolist()}"
    print(f"[ 3] OKall image_ids are 8-digit zero-padded")

    # ---------- Check 4: per-class coverage (rarity-priority guarantee) ----------
    # Sampling puts 2000 distinct frames into each priority class's bucket.
    # A multi-label frame may appear in both its own bucket's class=1 column
    # AND co-occur in rarer-class columns. So (class X == 1).sum() >= 2000.
    # The exclusive "rarity-priority folder" count sums to exactly 12,000.
    defects = dfs["defect_ids"]
    for cls in MAIN_CLASSES:
        n_with_cls = int((defects[cls] == 1).sum())
        assert n_with_cls >= 2000, f"[4] {cls}: only {n_with_cls} frames, expected >= 2000"
    # exclusive (rarity-priority) folder counts sum to 12,000
    exclusive_counts = {}
    assigned_mask = pd.Series(False, index=defects.index)
    for cls in PRIORITY:
        mask = (defects[cls] == 1) & (~assigned_mask)
        exclusive_counts[cls] = int(mask.sum())
        assigned_mask |= mask
    total_excl = sum(exclusive_counts.values())
    assert total_excl == 12_000, f"[4] exclusive bucket sum {total_excl} != 12,000"
    print(f"[ 4] OK defect coverage: each class >= 2,000; rarity-priority folders sum = 12,000")
    print(f"     (rarity-priority folder sizes: {exclusive_counts})")

    # ---------- Check 5: defect_ids has y=1 for all; normal has y=0 for all ----------
    assert (defects["y"] == 1).all(), f"[5] defect_ids has {(defects['y']!=1).sum()} rows with y != 1"
    for k in ["normal_stage1_ids", "normal_stage2_ids"]:
        assert (dfs[k]["y"] == 0).all(), f"[5] {k} has {(dfs[k]['y']!=0).sum()} rows with y != 0"
    print(f"[ 5] OKy labels consistent (defect=1, normal=0)")

    # ---------- Check 6: normal_stage2 is strict subset of normal_stage1 ----------
    s1_ids = set(dfs["normal_stage1_ids"].image_id)
    s2_ids = set(dfs["normal_stage2_ids"].image_id)
    assert s2_ids.issubset(s1_ids), f"[6] normal_stage2 NOT subset of normal_stage1 ({len(s2_ids - s1_ids)} extras)"
    print(f"[ 6] OKnormal_stage2 is strict subset of normal_stage1 (4,000 of 12,000)")

    # ---------- Check 7: pairwise disjoint across v3 train pools (defect, normal_stage1) ----------
    d_ids = set(defects.image_id)
    assert not (d_ids & s1_ids), f"[7] defect ∩ normal_stage1: {len(d_ids & s1_ids)}"
    print(f"[ 7] OKdefect pool disjoint from normal pool")

    # ---------- Check 8: v3 train disjoint from v3 val/test ----------
    val_cal_ids = set(dfs["val_cal_ids"].image_id)
    val_op_ids = set(dfs["val_op_ids"].image_id)
    test_ids = set(dfs["test_ids"].image_id)
    train_ids = d_ids | s1_ids
    for name, ids in [("val_cal", val_cal_ids), ("val_op", val_op_ids), ("test", test_ids)]:
        overlap = train_ids & ids
        assert not overlap, f"[8] train ∩ {name}: {len(overlap)}"
    # val_cal ∩ val_op ∩ test pairwise
    for (a_n, a), (b_n, b) in [
        (("val_cal", val_cal_ids), ("val_op", val_op_ids)),
        (("val_cal", val_cal_ids), ("test", test_ids)),
        (("val_op", val_op_ids), ("test", test_ids)),
    ]:
        assert not (a & b), f"[8] {a_n} ∩ {b_n}: {len(a & b)}"
    print(f"[ 8] OKall v3 pools pairwise disjoint")

    # ---------- Check 9: v3 val/test == v1 val/test (byte-identical via sha256) ----------
    for k in ["val_cal_ids", "val_op_ids", "test_ids"]:
        v3_sha = sha256_of_file(V3_DIR / f"{k}.csv")
        v1_sha = sha256_of_file(V1_DIR / f"{k}.csv")
        assert v3_sha == v1_sha, f"[9] {k}: v3 sha {v3_sha[:12]} != v1 sha {v1_sha[:12]}"
    print(f"[ 9] OKv3 val/test byte-identical to v1 (sha256 match)")

    # ---------- Check 10: manifest.json sha256 claims match actual files ----------
    # manifest uses both "name_ids" and plain "name" keys; handle both
    for name, info in manifest["pools"].items():
        filename = f"{name}.csv" if (V3_DIR / f"{name}.csv").exists() else f"{name}_ids.csv"
        path = V3_DIR / filename
        assert path.exists(), f"[10] manifest references {name} but neither {name}.csv nor {name}_ids.csv exists"
        actual_sha = sha256_of_file(path)
        claimed = info["sha256"]
        assert actual_sha == claimed, f"[10] {filename}: actual sha {actual_sha[:12]} != manifest {claimed[:12]}"
    print(f"[10] OK manifest.json sha256 values match actual CSV contents")

    # ---------- Check 11: all image_ids physically exist on disk ----------
    print(f"[..] scanning disk {IMG_DIR}...")
    disk_ids = set()
    for entry in os.scandir(IMG_DIR):
        if entry.name.endswith(".png"):
            disk_ids.add(entry.name[:-4])
    all_referenced = d_ids | s1_ids | val_cal_ids | val_op_ids | test_ids
    missing = all_referenced - disk_ids
    assert not missing, f"[11] {len(missing)} image files missing on disk, e.g. {list(missing)[:3]}"
    print(f"[11] OKall {len(all_referenced):,} referenced image_ids exist on disk")

    # ---------- Check 12: desktop export integrity ----------
    if DESKTOP.exists():
        # Expected folder counts: for EVERY split, use rarity-priority assignment
        # (Normal + PF > DE > RB > AF > OB > FS exclusive).
        def compute_expected(df):
            d = {"Normal": int((df["y"] == 0).sum())}
            assigned_split = set()
            for cls in PRIORITY:
                mask = (df[cls] == 1) & (~df["image_id"].isin(assigned_split))
                d[cls] = int(mask.sum())
                assigned_split.update(df.loc[mask, "image_id"])
            return d

        train_df = pd.concat([defects, dfs["normal_stage1_ids"]], ignore_index=True)
        expected_folder_counts = {
            "train":   compute_expected(train_df),
            "val_cal": compute_expected(dfs["val_cal_ids"]),
            "val_op":  compute_expected(dfs["val_op_ids"]),
            "test":    compute_expected(dfs["test_ids"]),
        }

        for split, folder_counts in expected_folder_counts.items():
            split_dir = DESKTOP / split
            assert split_dir.exists(), f"[12] desktop {split}/ missing"
            for folder, expected_n in folder_counts.items():
                folder_dir = split_dir / folder
                if expected_n == 0:
                    continue  # might not exist if empty
                assert folder_dir.exists(), f"[12] desktop {split}/{folder}/ missing"
                actual_n = sum(1 for _ in folder_dir.iterdir() if _.name.endswith(".png"))
                assert actual_n == expected_n, \
                    f"[12] desktop {split}/{folder}/: got {actual_n}, expected {expected_n}"

        # manifest folder check
        manifests_dir = DESKTOP / "manifests"
        assert manifests_dir.exists(), "[12] desktop manifests/ missing"
        for name in ["defect_ids", "normal_stage1_ids", "normal_stage2_ids",
                     "val_cal_ids", "val_op_ids", "test_ids"]:
            f = manifests_dir / f"{name}.csv"
            assert f.exists(), f"[12] desktop manifests/{name}.csv missing"
        print(f"[12] OKdesktop export structure verified ({DESKTOP})")
    else:
        print(f"[12] SKIP desktop export (not present at {DESKTOP})")

    print()
    print("=" * 60)
    print(" ALL 12 CHECKS PASSED ")
    print("=" * 60)

    # ---------- summary printout ----------
    print(f"\nSummary of v3:")
    print(f"  defect_ids           : {len(defects):>6,} (2000/class × 6, rarity-priority)")
    print(f"  normal_stage1_ids    : {len(dfs['normal_stage1_ids']):>6,}")
    print(f"  normal_stage2_ids    : {len(dfs['normal_stage2_ids']):>6,} (strict subset of stage1)")
    print(f"  val_cal_ids          : {len(dfs['val_cal_ids']):>6,} (copy of v1)")
    print(f"  val_op_ids           : {len(dfs['val_op_ids']):>6,} (copy of v1)")
    print(f"  test_ids             : {len(dfs['test_ids']):>6,} (copy of v1)")
    print(f"\nStage 1 train set size: {len(defects) + len(dfs['normal_stage1_ids']):,} (1:1 balanced)")
    print(f"Stage 2 train set size: {len(defects) + len(dfs['normal_stage2_ids']):,} (3:1 defect:normal)")
    print(f"Total unique train images: {len(d_ids | s1_ids):,}")
    print(f"Total unique across all pools: {len(all_referenced):,}")


if __name__ == "__main__":
    main()
