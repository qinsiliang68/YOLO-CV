"""Evaluate all 55 best checkpoints on the extended val set (low VRAM mode)."""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path

# Ensure scripts/ is on the path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from collect_cls_raw_materials import collect_validation_predictions
from temperature_scale_gate_runs import (
    attach_calibrated_scores,
    build_op_metrics,
    fit_temperature,
    prediction_view,
)

import torch


def safe_prob_to_logit(prob: float, eps: float = 1e-6) -> float:
    clipped = min(max(prob, eps), 1.0 - eps)
    return math.log(clipped / (1.0 - clipped))


def evaluate_single_checkpoint(
    checkpoint_path: Path,
    data_root: Path,
    split_lookup: dict[str, str],
    normal_class: str = "Normal",
    device: str = "0",
    batch: int = 1,
    imgsz: int = 640,
    eps: float = 1e-6,
    bins: int = 10,
    max_iter: int = 200,
) -> dict:
    """Run inference + calibration + threshold search on one checkpoint."""
    _, prediction_rows, _, _ = collect_validation_predictions(
        weights_path=checkpoint_path,
        data_root=data_root,
        normal_class=normal_class,
        device=device,
        batch=batch,
        imgsz=imgsz,
        collect_embeddings=False,
    )

    enriched = []
    for row in prediction_rows:
        img_rel = str(row["img_rel_path"]).replace("\\", "/")
        subset = split_lookup.get(img_rel, "")
        abnormal_conf = float(row.get("abnormal_conf") or row.get("p_abnormal") or 0.0)
        enriched.append({
            **row,
            "subset": subset,
            "y_true": 0 if row["gt_label"] == normal_class else 1,
            "p_abnormal_raw": abnormal_conf,
            "logit_raw": safe_prob_to_logit(abnormal_conf, eps),
        })

    val_cal = [r for r in enriched if r["subset"] == "val-cal"]
    val_op = [r for r in enriched if r["subset"] == "val-op"]
    if not val_cal or not val_op:
        return {"error": "no cal/op split match"}

    logits = torch.tensor([float(r["logit_raw"]) for r in val_cal], dtype=torch.float64)
    labels = torch.tensor([float(r["y_true"]) for r in val_cal], dtype=torch.float64)
    temperature, _, _ = fit_temperature(logits=logits, labels=labels, max_iter=max_iter)

    val_op_calibrated = attach_calibrated_scores(val_op, temperature)
    calibrated_view = prediction_view(val_op_calibrated, "p_abnormal_calibrated", normal_class)
    _, threshold_summary, _, _ = build_op_metrics(calibrated_view, normal_class, bins)

    op_995 = threshold_summary["operating_points"]["recall_ge_99_5"]
    op_990 = threshold_summary["operating_points"]["recall_ge_99_0"]

    return {
        "temperature_T": round(float(temperature), 6),
        "tau_r995": round(float(op_995["threshold"]), 6),
        "tau_r990": round(float(op_990["threshold"]), 6),
        "spec_at_r995": round(float(op_995["specificity"]), 6),
        "spec_at_r990": round(float(op_990["specificity"]), 6),
        "prec_at_r990": round(float(op_990["precision"]), 6),
        "ptr_at_r990": round(float(op_990["ptr"]), 6),
        "tn_at_r995": int(op_995["tn"]),
        "fn_at_r995": int(op_995["fn"]),
        "tn_at_r990": int(op_990["tn"]),
        "fn_at_r990": int(op_990["fn"]),
        "n_cal": len(val_cal),
        "n_op": len(val_op),
        "n_op_normal": sum(1 for r in val_op if r["y_true"] == 0),
    }


def main():
    repo_root = Path(__file__).resolve().parent.parent
    ckpt_base = repo_root / "YOLOv11" / "checkpoint_export" / "best_checkpoints"
    data_root = repo_root / "YOLOv11" / "datasets" / "sewerml_gate2_extended_binary"
    split_csv = data_root / "val_cal_op_split.csv"
    output_csv = repo_root / "research" / "results" / "stage1_formal" / "extended_eval_results.csv"
    output_json = repo_root / "research" / "results" / "stage1_formal" / "extended_eval_results.json"

    # Load split lookup
    split_lookup = {}
    with open(split_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            split_lookup[row["img_rel_path"].replace("\\", "/")] = row["subset"]

    # Collect all checkpoints
    models = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]
    checkpoints = []
    for model in models:
        model_dir = ckpt_base / model
        if not model_dir.exists():
            continue
        for pt_file in sorted(model_dir.glob("*.pt")):
            # Parse: yolo11n_gate2_formal_hn00_epoch_076.pt
            name = pt_file.stem
            parts = name.split("_")
            # Find hn ratio
            hn = [p for p in parts if p.startswith("hn")]
            ratio = hn[0] if hn else "hn00"
            checkpoints.append({
                "model": model,
                "ratio": ratio,
                "checkpoint": str(pt_file),
                "name": name,
            })

    print(f"Total checkpoints: {len(checkpoints)}")
    print(f"Data root: {data_root}")
    print(f"Split CSV: {split_csv}")
    print(f"Output: {output_csv}")
    print()

    results = []
    fields = [
        "model", "ratio", "name",
        "spec_at_r995", "spec_at_r990", "prec_at_r990", "ptr_at_r990",
        "tau_r995", "tau_r990", "temperature_T",
        "tn_at_r995", "fn_at_r995", "tn_at_r990", "fn_at_r990",
        "n_cal", "n_op", "n_op_normal",
    ]

    for i, ckpt in enumerate(checkpoints, 1):
        print(f"[{i}/{len(checkpoints)}] {ckpt['model']} {ckpt['ratio']}...")
        t0 = time.time()

        try:
            metrics = evaluate_single_checkpoint(
                checkpoint_path=Path(ckpt["checkpoint"]),
                data_root=data_root,
                split_lookup=split_lookup,
                device="0",
                batch=1,
                imgsz=640,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            metrics = {"error": str(e)}

        elapsed = time.time() - t0
        row = {**ckpt, **metrics}
        results.append(row)
        print(f"  spec995={metrics.get('spec_at_r995','ERR')} "
              f"tn={metrics.get('tn_at_r995','?')}/{metrics.get('n_op_normal','?')} "
              f"({elapsed:.0f}s)")

        # Write incrementally
        with open(output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in results:
                w.writerow(r)

        # Clear GPU cache
        torch.cuda.empty_cache()

    # Write final JSON
    with open(output_json, "w") as f:
        json.dump({"total": len(results), "results": results}, f, indent=2)

    print(f"\nDone! {len(results)} checkpoints evaluated.")
    print(f"Results: {output_csv}")


if __name__ == "__main__":
    main()
