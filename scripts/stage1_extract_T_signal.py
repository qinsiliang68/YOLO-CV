"""
Extract Training Dynamics (T) signal for top-250 candidate pool.

For each checkpoint in T_CHECKPOINTS, runs inference on the 250 candidate
training normal samples, computes calibrated abnormal probability, then
calculates per-sample boundary relevance and trajectory consistency.

Must run BEFORE the Goldilocks campaign if real T signal is needed.
Without this, T degrades to R proxy.

Usage:
    python scripts/stage1_extract_T_signal.py --device 0

Output:
    research/materials/stage1_formal/gate_goldilocks_campaign/T_signal_cache.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

POOL_MASTER = REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_bucket_pilot" / "score_inputs" / "candidate_pool_master.csv"
DATASET_SOURCE = REPO_ROOT / "YOLOv11" / "datasets" / "sewerml_gate2_train7200"
TEACHER_RUN_DIR = REPO_ROOT / "YOLOv11" / "runs" / "stage1_formal_gate" / "yolo11m_gate2_formal_200ep"
SPLIT_CSV = REPO_ROOT / "research" / "materials" / "stage1_formal" / "manifests" / "val_cal_op_split.csv"
OUTPUT_DIR = REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_goldilocks_campaign"

T_CHECKPOINTS = [40, 60, 78, 100, 120, 140, 160]
BETA_T = 2.0
SIGMA_T_SQ = 0.5


def load_pool() -> list[dict]:
    with open(POOL_MASTER, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def get_checkpoint_path(epoch: int) -> Path:
    weights_dir = TEACHER_RUN_DIR / "weights"
    candidates = [
        weights_dir / f"epoch_{epoch:03d}.pt",
        weights_dir / f"epoch_{epoch}.pt",
        weights_dir / f"epoch{epoch:03d}.pt",
        weights_dir / f"epoch{epoch}.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Checkpoint for epoch {epoch} not found in {weights_dir}")


def infer_pool_samples(checkpoint_path: Path, pool: list[dict], device: str) -> tuple[dict[str, float], float]:
    """
    Run inference on pool samples + val_cal for temperature calibration.
    Returns (pool_scores, temperature).
    """
    from ultralytics import YOLO
    import torch

    model = YOLO(str(checkpoint_path))

    # Step 1: get val_cal predictions for temperature fitting
    val_dir = DATASET_SOURCE / "val"
    if not val_dir.exists():
        print(f"    WARNING: val dir not found, using T=1.0")
        temperature = 1.0
    else:
        # run prediction on val set
        results_val = model.predict(
            source=str(val_dir),
            device=device, imgsz=640, batch=24, verbose=False,
        )

        # load split CSV to identify cal vs op
        cal_ids = set()
        with open(SPLIT_CSV, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("cal_or_op", "") == "cal":
                    cal_ids.add(row.get("img_id", ""))

        # collect cal logits and labels
        cal_logits = []
        cal_labels = []
        for r in results_val:
            img_name = Path(r.path).stem
            if img_name not in cal_ids:
                continue
            probs = r.probs
            if probs is None:
                continue
            # get abnormal probability (class index depends on training)
            p_data = probs.data.cpu()
            if p_data.shape[0] == 2:
                logit_abnormal = float(torch.log(p_data[0] / p_data[1])) if p_data[1] > 1e-8 else 0.0
            else:
                logit_abnormal = 0.0
            # determine label from directory
            parent_name = Path(r.path).parent.name
            label = 1 if parent_name != "Normal" else 0
            cal_logits.append(logit_abnormal)
            cal_labels.append(label)

        if len(cal_logits) >= 10:
            # fit temperature using simple grid search on NLL
            logits_t = torch.tensor(cal_logits, dtype=torch.float32)
            labels_t = torch.tensor(cal_labels, dtype=torch.float32)
            best_T = 1.0
            best_nll = float("inf")
            for T_cand in np.arange(0.5, 5.0, 0.05):
                scaled = logits_t / T_cand
                probs_scaled = torch.sigmoid(scaled)
                nll = -torch.mean(labels_t * torch.log(probs_scaled + 1e-8) +
                                  (1 - labels_t) * torch.log(1 - probs_scaled + 1e-8))
                if nll < best_nll:
                    best_nll = nll
                    best_T = T_cand
            temperature = float(best_T)
        else:
            temperature = 1.0

    print(f"    Temperature: {temperature:.4f}")

    # Step 2: run inference on pool samples (training normals)
    train_normal_dir = DATASET_SOURCE / "train" / "Normal"
    scores = {}

    for i, row in enumerate(pool):
        img_id = row["image_id"]
        filename = Path(row.get("img_rel_path", "")).name
        img_path = train_normal_dir / filename

        if not img_path.exists():
            scores[img_id] = 0.5
            continue

        results = model.predict(str(img_path), device=device, imgsz=640, verbose=False)
        if results and len(results) > 0:
            probs = results[0].probs
            if probs is not None:
                p_data = probs.data.cpu()
                if p_data.shape[0] == 2:
                    logit = float(torch.log(p_data[0] / p_data[1])) if p_data[1] > 1e-8 else 0.0
                else:
                    logit = 0.0
                scaled_logit = logit / temperature
                p_calibrated = float(torch.sigmoid(torch.tensor(scaled_logit)))
                scores[img_id] = p_calibrated
            else:
                scores[img_id] = 0.5
        else:
            scores[img_id] = 0.5

        if (i + 1) % 50 == 0:
            print(f"    scored {i+1}/{len(pool)} samples")

    return scores, temperature


def main():
    parser = argparse.ArgumentParser(description="Extract T signal for Goldilocks campaign")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    pool = load_pool()
    if not pool:
        sys.exit("ERROR: candidate_pool_master.csv has no data rows")
    print(f"Pool: {len(pool)} samples")
    print(f"Checkpoints: {T_CHECKPOINTS}")

    checkpoint_scores: dict[int, dict[str, float]] = {}
    checkpoint_taus: dict[int, float] = {}

    for ep in T_CHECKPOINTS:
        print(f"\n  Checkpoint epoch {ep}:")
        try:
            ckpt_path = get_checkpoint_path(ep)
            print(f"    path: {ckpt_path}")
        except FileNotFoundError as e:
            print(f"    SKIP: {e}")
            continue

        scores, temperature = infer_pool_samples(ckpt_path, pool, args.device)
        checkpoint_scores[ep] = scores

        # estimate tau from calibrated scores
        score_vals = list(scores.values())
        tau = float(np.percentile(score_vals, 70))
        checkpoint_taus[ep] = tau
        print(f"    scored {len(scores)} samples, tau_est={tau:.4f}, T={temperature:.4f}")

    if len(checkpoint_scores) < 3:
        print(f"\n  ERROR: only {len(checkpoint_scores)} checkpoints available, need >= 3")
        sys.exit(1)

    # compute T signal
    print(f"\n  Computing T signal from {len(checkpoint_scores)} checkpoints...")
    T_values = {}

    for row in pool:
        img_id = row["image_id"]
        b_vals = []
        for ep, scores in checkpoint_scores.items():
            if img_id not in scores:
                continue
            p = scores[img_id]
            p = max(1e-6, min(1 - 1e-6, p))
            logit = math.log(p / (1 - p))
            tau = checkpoint_taus[ep]
            tau = max(1e-6, min(1 - 1e-6, tau))
            tau_logit = math.log(tau / (1 - tau))
            b = math.exp(-abs(logit - tau_logit) / BETA_T)
            b_vals.append(b)

        if len(b_vals) >= 3:
            b_arr = np.array(b_vals)
            b_mean = float(np.mean(b_arr))
            b_std = float(np.std(b_arr))
            T_values[img_id] = b_mean * math.exp(-b_std ** 2 / SIGMA_T_SQ)
        else:
            T_values[img_id] = 0.0

    output_path = OUTPUT_DIR / "T_signal_cache.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(T_values, indent=2), encoding="utf-8")

    vals = [v for v in T_values.values() if v > 0]
    print(f"\n  T signal extracted: {len(T_values)} samples ({len(vals)} nonzero)")
    if vals:
        print(f"  min={min(vals):.6f}, max={max(vals):.6f}, mean={np.mean(vals):.6f}")
    print(f"  saved → {output_path}")


if __name__ == "__main__":
    main()
