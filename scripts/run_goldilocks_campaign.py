"""
Goldilocks Campaign: 78 full runs — 定峰 + 对照 + 组合验证

Phase 1 定峰 (66 run):
  R: 11 sliding windows (step=20, width=50)
  T: 11 sliding windows (step=20, width=50)
  D: 33 sliding windows (k={8,15,25} × 11 windows)
  C: 11 sliding windows (step=20, width=50)

Phase 2 对照 (3 run):
  Rand50-S1/S2/S3: random 50 from top250, replay to 151

Phase 3 组合验证 (9 run):
  F-R, F-T, F-D, F-C: 单信号 peak 窗口
  F-RT, F-RD, F-TD: 双信号交集
  F-RTD: 三信号交集
  F-RTDC: 四信号交集

所有 run: yolo11m-cls, hn00 teacher, top250 pool, budget=151, 200 epoch

Usage:
    python scripts/run_goldilocks_campaign.py --phase all --device 0
    python scripts/run_goldilocks_campaign.py --phase peak --signal R --device 0
    python scripts/run_goldilocks_campaign.py --phase peak --signal D --device 0
    python scripts/run_goldilocks_campaign.py --phase control --device 0
    python scripts/run_goldilocks_campaign.py --phase combine --device 0
    python scripts/run_goldilocks_campaign.py --list
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

DATASET_SOURCE = REPO_ROOT / "YOLOv11" / "datasets" / "sewerml_gate2_train7200"
SPLIT_CSV = REPO_ROOT / "research" / "materials" / "stage1_formal" / "manifests" / "val_cal_op_split.csv"
POOL_MASTER = REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_bucket_pilot" / "score_inputs" / "candidate_pool_master.csv"
TEACHER_MANIFEST = REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_hn_m_sweep" / "hn00" / "best_epoch_manifest.json"
HN00_PER_EPOCH = REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_hn_m_sweep" / "hn00" / "per_epoch_gate"

CAMP_ROOT = REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_goldilocks_campaign"
CAMP_RESULTS = REPO_ROOT / "research" / "results" / "stage1_formal" / "gate_goldilocks_campaign"
CAMP_DATASETS = REPO_ROOT / "YOLOv11" / "datasets" / "stage1_goldilocks"
CAMP_RUNS = REPO_ROOT / "YOLOv11" / "runs" / "stage1_goldilocks"

WINDOW_SIZE = 50
STEP_SIZE = 20
REPLAY_COUNT = 151
K_VALUES = [8, 15, 25]
T_CHECKPOINTS = [40, 60, 78, 100, 120, 140, 160]
SHARED_PROTOCOL = dict(epochs=200, imgsz=640, batch=24, workers=4,
                       optimizer="auto", patience=0, save_period=1,
                       pretrained=True, cache=False)


# ══════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════

def load_pool() -> list[dict]:
    with open(POOL_MASTER, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_teacher_checkpoint() -> str:
    manifest = json.loads(TEACHER_MANIFEST.read_text(encoding="utf-8"))
    cp = manifest["checkpoint_path"]
    p = Path(cp)
    if p.exists():
        return str(p.resolve())
    for marker in ("YOLO-CV/", "YOLOv11/"):
        idx = cp.replace("\\", "/").find(marker)
        if idx >= 0:
            local = REPO_ROOT / cp.replace("\\", "/")[idx + len("YOLO-CV/"):]
            if local.exists():
                return str(local.resolve())
    return cp


def build_id_to_filename(pool: list[dict]) -> dict[str, str]:
    mapping = {}
    for row in pool:
        img_id = row["image_id"]
        rel_path = row.get("img_rel_path", "")
        if rel_path:
            mapping[img_id] = Path(rel_path).name
        else:
            mapping[img_id] = f"{img_id.replace('Normal_', '')}.png"
    return mapping


# ══════════════════════════════════════════════════════════
# Signal computation
# ══════════════════════════════════════════════════════════

def compute_T_signal(pool: list[dict]) -> dict[str, float]:
    """Compute training dynamics signal T using multi-checkpoint boundary relevance."""
    cache_path = CAMP_ROOT / "T_signal_cache.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    print("  Computing T signal from multi-checkpoint boundary relevance...")
    pool_ids = {r["image_id"] for r in pool}

    # collect per-sample logits across target checkpoints
    sample_logits: dict[str, dict[int, float]] = {sid: {} for sid in pool_ids}
    tau_per_epoch: dict[int, float] = {}

    if HN00_PER_EPOCH.exists():
        for ep in T_CHECKPOINTS:
            epoch_dir = HN00_PER_EPOCH / f"epoch_{ep:03d}"
            if not epoch_dir.exists():
                continue
            # get tau_r995 for this epoch
            cal_summary = epoch_dir / "calibration_summary.json"
            op_json = epoch_dir / "threshold_operating_points_calibrated.json"
            if op_json.exists():
                ops = json.loads(op_json.read_text(encoding="utf-8"))
                tau = float(ops.get("recall_ge_99_5", {}).get("threshold", 0.28))
            else:
                tau = 0.28
            tau_per_epoch[ep] = tau

            pred_csv = epoch_dir / "val_op_predictions_calibrated.csv"
            if not pred_csv.exists():
                continue
            # build mapping from short img_id (e.g. "00467510") to pool image_id ("Normal_00467510")
            with open(pred_csv, "r", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    raw_id = row.get("img_id", "")
                    # try both formats: with and without "Normal_" prefix
                    full_id = f"Normal_{raw_id}" if not raw_id.startswith("Normal_") else raw_id
                    if full_id in sample_logits:
                        score = float(row.get("p_abnormal_calibrated", row.get("abnormal_conf", 0)))
                        score = max(1e-6, min(1 - 1e-6, score))
                        sample_logits[full_id][ep] = math.log(score / (1 - score))

    if not tau_per_epoch:
        print("  WARNING: per_epoch_gate not found, using R as proxy for T")
        T_values = {r["image_id"]: float(r["R"]) for r in pool}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(T_values, indent=2), encoding="utf-8")
        return T_values

    beta_T = 2.0  # bandwidth for boundary relevance
    sigma_T_sq = 0.5  # variance penalty

    T_values = {}
    for img_id in pool_ids:
        logits = sample_logits[img_id]
        if len(logits) < 3:
            T_values[img_id] = 0.0
            continue

        # boundary relevance at each checkpoint
        b_vals = []
        for ep, logit_val in logits.items():
            tau = tau_per_epoch.get(ep, 0.28)
            tau_logit = math.log(max(tau, 1e-6) / max(1 - tau, 1e-6))
            b = math.exp(-abs(logit_val - tau_logit) / beta_T)
            b_vals.append(b)

        b_arr = np.array(b_vals)
        b_mean = float(np.mean(b_arr))
        b_std = float(np.std(b_arr))

        # T = mean boundary relevance × stability penalty
        T_values[img_id] = b_mean * math.exp(-b_std ** 2 / sigma_T_sq)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(T_values, indent=2), encoding="utf-8")
    print(f"  T signal computed for {len(T_values)} samples")
    return T_values


def compute_D_for_k(pool: list[dict], k: int) -> dict[str, float]:
    """Recompute D signal at given k using embeddings."""
    cache_path = CAMP_ROOT / f"D_k{k:02d}_cache.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    # find embeddings
    embed_candidates = [
        REPO_ROOT / "$out" / "scratch" / "stage1_formal" / "gate_info_sampling_lite" / "teacher_train_features" / "train_embeddings.npy",
        REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_bucket_pilot" / "score_inputs" / "train_embeddings.npy",
    ]
    embed_path = None
    for p in embed_candidates:
        if p.exists():
            embed_path = p
            break

    if embed_path is None:
        print(f"  WARNING: embeddings not found, using existing D values for k={k}")
        D_values = {r["image_id"]: float(r["D"]) for r in pool}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(D_values, indent=2), encoding="utf-8")
        return D_values

    print(f"  Computing D signal for k={k} from embeddings...")
    embeddings = np.load(str(embed_path))

    # find feature CSV for index mapping
    feat_csv_candidates = [
        REPO_ROOT / "$out" / "scratch" / "stage1_formal" / "gate_info_sampling_lite" / "teacher_train_features" / "train_features.csv",
        REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_bucket_pilot" / "score_inputs" / "train_features.csv",
    ]
    feature_ids = None
    for p in feat_csv_candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8-sig") as f:
                feature_ids = [row.get("image_id", row.get("path", "")) for row in csv.DictReader(f)]
            break

    if feature_ids is None:
        # use embedding_index from pool
        max_idx = max(int(r.get("embedding_index", 0)) for r in pool)
        feature_ids = [f"unknown_{i}" for i in range(max_idx + 1)]
        for r in pool:
            feature_ids[int(r.get("embedding_index", 0))] = r["image_id"]

    pool_ids = {r["image_id"] for r in pool}
    id_to_idx = {fid: idx for idx, fid in enumerate(feature_ids) if fid in pool_ids}
    pool_id_list = [pid for pid in pool_ids if pid in id_to_idx]
    pool_indices = [id_to_idx[pid] for pid in pool_id_list]
    pool_embeds = embeddings[pool_indices]

    norms = np.linalg.norm(pool_embeds, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    pool_embeds_normed = pool_embeds / norms
    sim_matrix = pool_embeds_normed @ pool_embeds_normed.T

    D_values = {}
    for i, pid in enumerate(pool_id_list):
        sims = sim_matrix[i].copy()
        sims[i] = -1
        topk_idx = np.argpartition(sims, -k)[-k:]
        D_values[pid] = float(np.mean(sims[topk_idx]))

    # normalize to [0,1]
    vals = list(D_values.values())
    vmin, vmax = min(vals), max(vals)
    if vmax > vmin:
        D_values = {k_: (v - vmin) / (vmax - vmin) for k_, v in D_values.items()}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(D_values, indent=2), encoding="utf-8")
    print(f"  D(k={k}) computed for {len(D_values)} samples")
    return D_values


# ══════════════════════════════════════════════════════════
# Experiment definitions
# ══════════════════════════════════════════════════════════

def define_sliding_windows() -> list[dict]:
    """Generate all 66 sliding window experiments."""
    experiments = []
    pool_size = 250

    def make_windows(signal: str, suffix: str = ""):
        wins = []
        w_idx = 0
        start = 1
        while start + WINDOW_SIZE - 1 <= pool_size:
            end = start + WINDOW_SIZE - 1
            exp_id = f"{signal}{suffix}-W{w_idx + 1:02d}"
            wins.append({
                "id": exp_id, "phase": "peak", "signal": signal,
                "signal_key": f"{signal}{suffix}",
                "rank_start": start, "rank_end": end,
                "center_pct": round(((start + end) / 2.0 - 0.5) / pool_size * 100, 1),
            })
            start += STEP_SIZE
            w_idx += 1
        return wins

    experiments.extend(make_windows("R"))
    experiments.extend(make_windows("T"))
    for k in K_VALUES:
        experiments.extend(make_windows("D", f"-k{k:02d}"))
    experiments.extend(make_windows("C"))
    return experiments


def define_controls() -> list[dict]:
    """3 random size-matched controls."""
    return [
        {"id": f"Rand50-S{i}", "phase": "control", "signal": "random", "seed": 42 + i}
        for i in range(1, 4)
    ]


def define_combinations() -> list[dict]:
    """9 combination verification runs."""
    combos = ["R", "T", "D", "C", "RT", "RD", "TD", "RTD", "RTDC"]
    return [{"id": f"F-{c}", "phase": "combine", "signal": c} for c in combos]


# ══════════════════════════════════════════════════════════
# Dataset building + training + evaluation
# ══════════════════════════════════════════════════════════

def build_dataset_view(exp_id: str, selected_ids: list[str],
                       id_to_filename: dict[str, str]) -> Path:
    ds_root = CAMP_DATASETS / exp_id
    marker_path = ds_root / "_built_marker.json"
    if marker_path.exists():
        return ds_root

    train_normal = ds_root / "train" / "Normal"
    train_abnormal = ds_root / "train" / "Abnormal"
    val_normal = ds_root / "val" / "Normal"
    val_abnormal = ds_root / "val" / "Abnormal"
    for d in [train_normal, train_abnormal, val_normal, val_abnormal]:
        d.mkdir(parents=True, exist_ok=True)

    src = {
        "tn": DATASET_SOURCE / "train" / "Normal",
        "ta": DATASET_SOURCE / "train" / "Abnormal",
        "vn": DATASET_SOURCE / "val" / "Normal",
        "va": DATASET_SOURCE / "val" / "Abnormal",
    }

    def link_dir(s: Path, d: Path):
        for img in s.iterdir():
            t = d / img.name
            if not t.exists():
                try:
                    os.link(str(img), str(t))
                except OSError:
                    shutil.copy2(str(img), str(t))

    link_dir(src["ta"], train_abnormal)
    link_dir(src["tn"], train_normal)
    link_dir(src["va"], val_abnormal)
    link_dir(src["vn"], val_normal)

    n_unique = len(selected_ids)
    for copy_idx in range(REPLAY_COUNT):
        sid = selected_ids[copy_idx % n_unique]
        filename = id_to_filename.get(sid, f"{sid}.png")
        src_file = src["tn"] / filename
        if not src_file.exists():
            alt = list(src["tn"].glob(f"*{sid.replace('Normal_', '')}*"))
            if alt:
                src_file = alt[0]
            else:
                continue
        dst = train_normal / f"_replay_{exp_id}_{copy_idx:04d}_{src_file.name}"
        if not dst.exists():
            try:
                os.link(str(src_file), str(dst))
            except OSError:
                shutil.copy2(str(src_file), str(dst))

    marker_path.write_text(json.dumps({
        "experiment_id": exp_id, "n_unique": n_unique,
        "replay": REPLAY_COUNT, "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")
    return ds_root


def run_training(exp_id: str, ds_root: Path, teacher_ckpt: str, device: str) -> Path:
    run_dir = CAMP_RUNS / exp_id
    last_pt = run_dir / "weights" / "last.pt"
    if last_pt.exists():
        try:
            import torch
            state = torch.load(str(last_pt), map_location="cpu", weights_only=False)
            completed_epoch = state.get("epoch", 0)
            if completed_epoch >= SHARED_PROTOCOL["epochs"] - 1:
                print(f"    training complete ({completed_epoch+1} epochs), skipping")
                return run_dir
            else:
                print(f"    incomplete training ({completed_epoch+1}/{SHARED_PROTOCOL['epochs']} epochs), retraining")
        except Exception:
            print(f"    last.pt exists but unreadable, retraining")
    from ultralytics import YOLO
    model = YOLO(teacher_ckpt)
    model.train(data=str(ds_root), project=str(CAMP_RUNS), name=exp_id,
                exist_ok=True, device=device, seed=0, **SHARED_PROTOCOL)
    return run_dir


def run_gate_eval(exp_id: str, run_dir: Path, dataset_root: Path) -> dict[str, Any]:
    summary_dir = CAMP_ROOT / exp_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    if (summary_dir / "best_epoch_manifest.json").exists():
        m = json.loads((summary_dir / "best_epoch_manifest.json").read_text(encoding="utf-8"))
        print(f"    eval exists: Spec@R99.5={m.get('spec_at_r995', '?')}")
        return m
    eval_script = SCRIPTS_DIR / "stage1_formal_gate_epoch_eval.py"
    import subprocess
    result = subprocess.run([
        sys.executable, str(eval_script),
        "--run-dir", str(run_dir),
        "--data-root", str(dataset_root),
        "--summary-dir", str(summary_dir),
        "--split-csv", str(SPLIT_CSV),
    ], cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"    WARNING: gate eval failed (exit {result.returncode})")
    if (summary_dir / "best_epoch_manifest.json").exists():
        m = json.loads((summary_dir / "best_epoch_manifest.json").read_text(encoding="utf-8"))
        print(f"    eval complete: Spec@R99.5={m.get('spec_at_r995', '?')}")
        return m
    return {}


# ══════════════════════════════════════════════════════════
# Sorting helpers
# ══════════════════════════════════════════════════════════

def sort_pool(pool: list[dict], signal_key: str,
              T_values: dict | None = None,
              D_values_by_k: dict | None = None) -> list[dict]:
    """Sort pool by signal, descending."""
    if signal_key == "T":
        pool_copy = [dict(r, T=T_values.get(r["image_id"], 0.0)) for r in pool]
        return sorted(pool_copy, key=lambda r: float(r["T"]), reverse=True)
    elif signal_key.startswith("D-k"):
        k = int(signal_key.split("k")[1])
        D_vals = D_values_by_k[k]
        pool_copy = [dict(r, D=D_vals.get(r["image_id"], 0.0)) for r in pool]
        return sorted(pool_copy, key=lambda r: float(r["D"]), reverse=True)
    else:
        return sorted(pool, key=lambda r: float(r.get(signal_key, 0)), reverse=True)


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Goldilocks Campaign: 78 full runs")
    parser.add_argument("--phase", default="all", help="all, peak, control, combine")
    parser.add_argument("--signal", default="", help="Filter peak phase by signal: R, T, D, C")
    parser.add_argument("--device", default="0")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    sw_exps = define_sliding_windows()
    ctrl_exps = define_controls()
    combo_exps = define_combinations()

    if args.list:
        print(f"\nGoldilocks Campaign — 78 runs\n")
        print(f"  Peak (定峰): {len(sw_exps)} runs")
        for e in sw_exps:
            print(f"    {e['id']:20s}  rank {e['rank_start']:3d}-{e['rank_end']:3d}  center={e['center_pct']}%")
        print(f"\n  Control (对照): {len(ctrl_exps)} runs")
        for e in ctrl_exps:
            print(f"    {e['id']}")
        print(f"\n  Combine (组合验证): {len(combo_exps)} runs")
        for e in combo_exps:
            print(f"    {e['id']}")
        print(f"\n  Total: {len(sw_exps) + len(ctrl_exps) + len(combo_exps)}")
        return

    pool = load_pool()
    teacher_ckpt = load_teacher_checkpoint()
    id_to_fn = build_id_to_filename(pool)

    print(f"\n{'='*60}")
    print(f"  Goldilocks Campaign")
    print(f"  Pool: {len(pool)}, Teacher: {Path(teacher_ckpt).name}")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # pre-compute signals
    T_values = None
    D_values_by_k: dict[int, dict[str, float]] = {}

    need_T = args.phase in ("all", "peak") and (not args.signal or args.signal == "T")
    need_D = args.phase in ("all", "peak") and (not args.signal or args.signal == "D")

    if need_T or args.phase in ("all", "combine"):
        T_values = compute_T_signal(pool)
    if need_D or args.phase in ("all", "combine"):
        for k in K_VALUES:
            D_values_by_k[k] = compute_D_for_k(pool, k)

    # pre-sort pools (only for available signals)
    sorted_pools: dict[str, list[dict]] = {}
    for sig_key in ["R", "T", "C"] + [f"D-k{k:02d}" for k in K_VALUES]:
        if sig_key == "T" and T_values is None:
            sorted_pools[sig_key] = sorted(pool, key=lambda r: float(r.get("R", 0)), reverse=True)
        elif sig_key.startswith("D-k"):
            k_val = int(sig_key.split("k")[1])
            if k_val in D_values_by_k:
                sorted_pools[sig_key] = sort_pool(pool, sig_key, T_values, D_values_by_k)
            else:
                sorted_pools[sig_key] = sorted(pool, key=lambda r: float(r.get("D", 0)), reverse=True)
        else:
            sorted_pools[sig_key] = sort_pool(pool, sig_key, T_values, D_values_by_k)

    all_results = []
    peak_results_path = CAMP_RESULTS / "peak_results.json"
    t_start = time.time()

    def save_peak_results(results: list[dict]):
        """Persist peak results to disk so combine phase can find them across calls."""
        existing = {}
        if peak_results_path.exists():
            existing = json.loads(peak_results_path.read_text(encoding="utf-8"))
        for r in results:
            if r.get("phase") == "peak" and r.get("spec_at_r995") is not None:
                key = r.get("signal_key", r.get("signal", ""))
                if key not in existing or r["spec_at_r995"] > existing[key].get("spec_at_r995", -1):
                    existing[key] = {
                        "signal_key": key, "rank_start": r["rank_start"],
                        "rank_end": r["rank_end"], "spec_at_r995": r["spec_at_r995"],
                        "id": r["id"],
                    }
        peak_results_path.parent.mkdir(parents=True, exist_ok=True)
        peak_results_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def load_peak_results() -> dict:
        if peak_results_path.exists():
            return json.loads(peak_results_path.read_text(encoding="utf-8"))
        return {}

    # ── Phase 1: Peak (sliding windows) ──
    if args.phase in ("all", "peak"):
        exps = sw_exps
        if args.signal:
            exps = [e for e in exps if e["signal"] == args.signal.upper()]
        for i, exp in enumerate(exps):
            print(f"\n  [Peak {i+1}/{len(exps)}] {exp['id']}")
            ps = sorted_pools[exp["signal_key"]]
            selected = ps[exp["rank_start"]-1:exp["rank_end"]]
            selected_ids = [r["image_id"] for r in selected]
            ds = build_dataset_view(exp["id"], selected_ids, id_to_fn)
            t0 = time.time()
            rd = run_training(exp["id"], ds, teacher_ckpt, args.device)
            m = run_gate_eval(exp["id"], rd, ds)
            all_results.append({**exp, "spec_at_r995": m.get("spec_at_r995"),
                                "best_epoch": m.get("epoch"),
                                "train_hours": round((time.time()-t0)/3600, 2)})
        save_peak_results(all_results)

    # ── Phase 2: Controls ──
    if args.phase in ("all", "control"):
        for i, exp in enumerate(ctrl_exps):
            print(f"\n  [Control {i+1}/{len(ctrl_exps)}] {exp['id']}")
            rng = random.Random(exp["seed"])
            selected_ids = [r["image_id"] for r in rng.sample(pool, WINDOW_SIZE)]
            ds = build_dataset_view(exp["id"], selected_ids, id_to_fn)
            t0 = time.time()
            rd = run_training(exp["id"], ds, teacher_ckpt, args.device)
            m = run_gate_eval(exp["id"], rd, ds)
            all_results.append({**exp, "spec_at_r995": m.get("spec_at_r995"),
                                "best_epoch": m.get("epoch"),
                                "train_hours": round((time.time()-t0)/3600, 2)})

    # ── Phase 3: Combinations ──
    if args.phase in ("all", "combine"):
        # load peak results from file (persisted by peak phase, possibly from another machine)
        saved_peaks = load_peak_results()
        peak_ids: dict[str, set[str]] = {}

        for sig_key in ["R", "T", "C"]:
            if sig_key in saved_peaks:
                pk = saved_peaks[sig_key]
                ps = sorted_pools.get(sig_key)
                if ps:
                    peak_ids[sig_key] = {r["image_id"] for r in ps[pk["rank_start"]-1:pk["rank_end"]]}
                    print(f"    Peak {sig_key}: rank {pk['rank_start']}-{pk['rank_end']} (Spec={pk['spec_at_r995']:.4f})")

        # D: find best k from saved peaks
        best_d_key = None
        best_d_spec = -1
        for sk, pk in saved_peaks.items():
            if sk.startswith("D-k") and pk.get("spec_at_r995", -1) > best_d_spec:
                best_d_spec = pk["spec_at_r995"]
                best_d_key = sk
        if best_d_key and best_d_key in sorted_pools:
            pk = saved_peaks[best_d_key]
            ps = sorted_pools[best_d_key]
            peak_ids["D"] = {r["image_id"] for r in ps[pk["rank_start"]-1:pk["rank_end"]]}
            print(f"    Peak D: {best_d_key} rank {pk['rank_start']}-{pk['rank_end']} (Spec={best_d_spec:.4f})")

        for i, exp in enumerate(combo_exps):
            print(f"\n  [Combine {i+1}/{len(combo_exps)}] {exp['id']}")
            combo_signals = list(exp["signal"])  # e.g. "RTD" → ["R","T","D"]

            missing = [s for s in combo_signals if s not in peak_ids]
            if missing:
                print(f"    SKIP: peak not found for {missing}. Run peak phase first.")
                continue

            if len(combo_signals) == 1:
                selected_set = peak_ids[combo_signals[0]]
            else:
                selected_set = set.intersection(*[peak_ids[s] for s in combo_signals])

            if len(selected_set) < 5:
                # intersection too small, use union ranked by frequency (most nominated first)
                from collections import Counter
                counter = Counter()
                for s in combo_signals:
                    counter.update(peak_ids[s])
                selected_ids = [sid for sid, _ in counter.most_common(WINDOW_SIZE)]
                selected_set = set(selected_ids)
            else:
                selected_ids = list(selected_set)[:WINDOW_SIZE]
            if len(selected_ids) < WINDOW_SIZE:
                # pad with highest-ranked samples from R
                for r in sorted_pools["R"]:
                    if r["image_id"] not in selected_set:
                        selected_ids.append(r["image_id"])
                    if len(selected_ids) >= WINDOW_SIZE:
                        break

            ds = build_dataset_view(exp["id"], selected_ids, id_to_fn)
            t0 = time.time()
            rd = run_training(exp["id"], ds, teacher_ckpt, args.device)
            m = run_gate_eval(exp["id"], rd)
            all_results.append({**exp, "spec_at_r995": m.get("spec_at_r995"),
                                "n_selected": len(selected_ids),
                                "best_epoch": m.get("epoch"),
                                "train_hours": round((time.time()-t0)/3600, 2)})

    # ── Summary ──
    total_h = (time.time() - t_start) / 3600
    print(f"\n{'='*60}")
    print(f"  Campaign complete. {total_h:.1f}h")
    print(f"{'='*60}\n")

    summary_path = CAMP_RESULTS / "goldilocks_campaign_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "phase", "signal", "signal_key", "rank_start", "rank_end",
              "center_pct", "spec_at_r995", "best_epoch", "train_hours"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)

    print(f"  Summary: {summary_path}")
    for r in all_results:
        spec = r.get("spec_at_r995", "?")
        print(f"  {r['id']:20s}  Spec@R99.5={spec}")


if __name__ == "__main__":
    main()
