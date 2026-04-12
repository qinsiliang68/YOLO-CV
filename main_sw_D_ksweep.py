"""
D 信号 k-sweep + 滑窗扫描

对 k ∈ {5, 10, 15, 20, 30} 各做一轮 21 窗口滑窗扫描。
先用 train_embeddings.npy 重算每个 k 下的 D 值,再按 D 排序做滑窗。
总计 5 × 21 = 105 run。

Usage:
    uv run main_sw_D_ksweep.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

POOL_MASTER = REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_info_sampling_lite" / "score_inputs" / "candidate_pool_master.csv"
SW_ROOT = REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_sliding_window"
SW_RESULTS = REPO_ROOT / "research" / "results" / "stage1_formal" / "gate_sliding_window"

# possible locations for train embeddings
EMBEDDING_CANDIDATES = [
    REPO_ROOT / "$out" / "scratch" / "stage1_formal" / "gate_info_sampling_lite" / "teacher_train_features" / "train_embeddings.npy",
    REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_info_sampling_lite" / "score_inputs" / "train_embeddings.npy",
    REPO_ROOT / "research" / "materials" / "yolo11m_gate2_train7200" / "val_embeddings.npy",
]

FEATURE_CSV_CANDIDATES = [
    REPO_ROOT / "$out" / "scratch" / "stage1_formal" / "gate_info_sampling_lite" / "teacher_train_features" / "train_features.csv",
    REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_info_sampling_lite" / "score_inputs" / "train_features.csv",
]

K_VALUES = [5, 10, 15, 20, 30]


def load_pool() -> list[dict]:
    with open(POOL_MASTER, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def find_file(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def compute_D_for_k(pool: list[dict], embeddings: np.ndarray,
                    feature_ids: list[str], k: int) -> dict[str, float]:
    """Compute kNN density D for each pool sample at given k."""
    pool_ids = {r["image_id"] for r in pool}

    # find embedding indices for pool samples
    id_to_embed_idx = {}
    for idx, fid in enumerate(feature_ids):
        if fid in pool_ids:
            id_to_embed_idx[fid] = idx

    if not id_to_embed_idx:
        raise ValueError("No pool samples found in embedding file")

    # extract pool embeddings
    pool_id_list = [pid for pid in pool_ids if pid in id_to_embed_idx]
    pool_indices = [id_to_embed_idx[pid] for pid in pool_id_list]
    pool_embeds = embeddings[pool_indices]

    # normalize for cosine similarity
    norms = np.linalg.norm(pool_embeds, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    pool_embeds_normed = pool_embeds / norms

    # compute pairwise cosine similarity
    sim_matrix = pool_embeds_normed @ pool_embeds_normed.T

    # for each sample, get mean of top-k neighbors (excluding self)
    D_values = {}
    for i, pid in enumerate(pool_id_list):
        sims = sim_matrix[i].copy()
        sims[i] = -1  # exclude self
        topk_indices = np.argpartition(sims, -k)[-k:]
        topk_sims = sims[topk_indices]
        D_values[pid] = float(np.mean(topk_sims))

    # normalize to [0, 1]
    vals = list(D_values.values())
    vmin, vmax = min(vals), max(vals)
    if vmax > vmin:
        D_values = {k: (v - vmin) / (vmax - vmin) for k, v in D_values.items()}

    return D_values


def main():
    print("=" * 60)
    print("  D Signal k-Sweep + Sliding Window Pipeline")
    print(f"  k values: {K_VALUES}")
    print(f"  Total: {len(K_VALUES)} × 21 windows = {len(K_VALUES) * 21} runs")
    print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    pool = load_pool()
    print(f"\n  Pool: {len(pool)} samples")

    # find embeddings
    embed_path = find_file(EMBEDDING_CANDIDATES)
    if embed_path is None:
        print("\n  ERROR: train_embeddings.npy not found at any expected location:")
        for p in EMBEDDING_CANDIDATES:
            print(f"    {p}")
        print("\n  Please ensure feature extraction has been run for hn00 teacher.")
        print("  Alternatively, copy train_embeddings.npy to one of the above paths.")
        sys.exit(1)

    print(f"  Embeddings: {embed_path}")
    embeddings = np.load(str(embed_path))
    print(f"  Shape: {embeddings.shape}")

    # find feature CSV to get image IDs matching embedding indices
    feat_csv_path = find_file(FEATURE_CSV_CANDIDATES)
    if feat_csv_path:
        with open(feat_csv_path, "r", encoding="utf-8-sig") as f:
            feature_ids = [row.get("image_id", row.get("path", "")) for row in csv.DictReader(f)]
        print(f"  Feature IDs from: {feat_csv_path} ({len(feature_ids)} entries)")
    else:
        # try to reconstruct from embedding_index in pool master
        print("  WARNING: train_features.csv not found, using embedding_index from pool master")
        max_idx = max(int(r.get("embedding_index", 0)) for r in pool)
        feature_ids = [f"unknown_{i}" for i in range(max_idx + 1)]
        for r in pool:
            idx = int(r.get("embedding_index", 0))
            feature_ids[idx] = r["image_id"]

    # import sliding window pipeline
    from scripts.run_sliding_window_pipeline import (
        load_teacher_checkpoint, build_id_to_filename, define_windows,
        run_experiment, SW_RESULTS as results_dir
    )

    teacher_ckpt = load_teacher_checkpoint()
    id_to_filename = build_id_to_filename(pool)

    all_results = []
    total_runs = len(K_VALUES) * 21
    run_counter = 0
    t_start = time.time()

    for k in K_VALUES:
        print(f"\n{'#' * 60}")
        print(f"  Computing D values for k={k}...")
        print(f"{'#' * 60}")

        D_values = compute_D_for_k(pool, embeddings, feature_ids, k)

        # create pool copy with updated D values
        pool_with_new_D = []
        for r in pool:
            rc = dict(r)
            rc["D"] = D_values.get(r["image_id"], 0.0)
            pool_with_new_D.append(rc)

        # sort by new D, descending
        pool_sorted = sorted(pool_with_new_D, key=lambda r: float(r["D"]), reverse=True)

        # define windows with k-specific experiment IDs
        windows = define_windows("D")
        for w in windows:
            w["id"] = f"sw_Dk{k:02d}_w{w['window_idx']:02d}_r{w['rank_start']:03d}_{w['rank_end']:03d}"
            w["signal"] = f"D_k{k}"

        for exp in windows:
            run_counter += 1
            print(f"\n  [{run_counter}/{total_runs}] k={k}, {exp['id']}")
            try:
                result = run_experiment(exp, pool_sorted, id_to_filename, teacher_ckpt, "0")
                result["k"] = k
                all_results.append(result)
            except Exception as exc:
                print(f"    FAILED: {exc}")
                import traceback
                traceback.print_exc()
                all_results.append({"experiment_id": exp["id"], "k": k, "error": str(exc)})

    # write summary
    total_hours = (time.time() - t_start) / 3600
    print(f"\n{'=' * 60}")
    print(f"  k-Sweep complete. Total: {total_hours:.1f}h")
    print(f"{'=' * 60}\n")

    summary_path = results_dir / "D_ksweep_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment_id", "signal", "k", "window_idx", "rank_start", "rank_end",
                  "center_pct", "n_unique", "signal_min", "signal_max", "signal_mean",
                  "spec_at_r995", "spec_at_r990", "best_epoch", "train_hours"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([r for r in all_results if "error" not in r])

    # print peak per k
    print("\n  Peak per k:")
    for k in K_VALUES:
        k_results = [r for r in all_results if r.get("k") == k and r.get("spec_at_r995") is not None]
        if k_results:
            peak = max(k_results, key=lambda r: r["spec_at_r995"])
            print(f"    k={k:2d}: peak at rank {peak['rank_start']}-{peak['rank_end']}, "
                  f"Spec@R99.5={peak['spec_at_r995']:.4f}")

    print(f"\n  Summary: {summary_path}")


if __name__ == "__main__":
    main()
