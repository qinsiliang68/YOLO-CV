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
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
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
    """Find checkpoint file for given epoch."""
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


def run_inference_on_pool(checkpoint_path: Path, pool: list[dict], device: str) -> dict[str, float]:
    """Run inference on pool samples using given checkpoint, return calibrated scores."""
    from ultralytics import YOLO

    model = YOLO(str(checkpoint_path))

    # calibrate temperature on val_cal
    from temperature_scale_gate_runs import fit_temperature, prediction_view
    from collect_cls_raw_materials import collect_validation_predictions

    # get val predictions for temperature fitting
    val_preds = collect_validation_predictions(
        model_path=str(checkpoint_path),
        data_root=str(DATASET_SOURCE),
        split="val",
        device=device,
        imgsz=640,
        batch=24,
    )

    # load split info
    split_rows = {}
    with open(SPLIT_CSV, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            split_rows[row.get("img_id", row.get("image_id", ""))] = row.get("cal_or_op", "")

    cal_preds = [p for p in val_preds if split_rows.get(p.get("img_id", ""), "") == "cal"]

    if cal_preds:
        T_temp = fit_temperature(cal_preds)
    else:
        T_temp = 1.0

    # now run inference on pool samples (training normals)
    scores = {}
    train_normal_dir = DATASET_SOURCE / "train" / "Normal"

    for row in pool:
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
                p_abnormal = float(probs.data[0])  # assuming class 0 = Abnormal
                # apply temperature scaling
                logit = math.log(max(p_abnormal, 1e-6) / max(1 - p_abnormal, 1e-6))
                scaled_logit = logit / T_temp
                p_calibrated = 1.0 / (1.0 + math.exp(-scaled_logit))
                scores[img_id] = p_calibrated
            else:
                scores[img_id] = 0.5
        else:
            scores[img_id] = 0.5

    return scores


def main():
    parser = argparse.ArgumentParser(description="Extract T signal for Goldilocks campaign")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    pool = load_pool()
    print(f"Pool: {len(pool)} samples")

    # collect per-checkpoint scores
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

        scores = run_inference_on_pool(ckpt_path, pool, args.device)
        checkpoint_scores[ep] = scores

        # estimate tau_r995 from the scores distribution
        score_vals = sorted(scores.values(), reverse=True)
        # tau_r995 ≈ threshold where 99.5% of abnormals are above
        # for normal samples, use a rough estimate based on score distribution
        tau = float(np.percentile(list(scores.values()), 70))  # rough proxy
        checkpoint_taus[ep] = tau
        print(f"    scored {len(scores)} samples, tau_est={tau:.4f}")

    if len(checkpoint_scores) < 3:
        print(f"\n  ERROR: only {len(checkpoint_scores)} checkpoints available, need at least 3")
        print(f"  T signal cannot be computed. Campaign will use R as proxy.")
        return

    # compute T signal
    print(f"\n  Computing T signal from {len(checkpoint_scores)} checkpoints...")
    T_values = {}
    pool_ids = [r["image_id"] for r in pool]

    for img_id in pool_ids:
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

    # save
    output_path = OUTPUT_DIR / "T_signal_cache.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(T_values, indent=2), encoding="utf-8")

    # summary
    vals = list(T_values.values())
    print(f"\n  T signal extracted for {len(T_values)} samples")
    print(f"  min={min(vals):.6f}, max={max(vals):.6f}, mean={np.mean(vals):.6f}")
    print(f"  saved → {output_path}")


if __name__ == "__main__":
    main()
