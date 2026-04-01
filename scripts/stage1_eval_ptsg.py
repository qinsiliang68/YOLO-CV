from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from collect_cls_raw_materials import (
    DEFAULT_THRESHOLDS,
    build_calibration_rows,
    build_threshold_summary,
    compute_binary_threshold_rows,
    write_csv,
    write_json,
)


DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 1.0
DEFAULT_GAMMA = 0.5
DEFAULT_EPS = 1e-8
DEFAULT_NORMAL_CLASS = "Normal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate post-hoc PTSG variants on stage-1 gate materials.")
    parser.add_argument("--val-features-csv", required=True, help="Validation feature index CSV.")
    parser.add_argument("--val-embeddings-npy", required=True, help="Validation embedding matrix.")
    parser.add_argument("--val-split-csv", required=True, help="CSV defining val-cal / val-op subsets.")
    parser.add_argument("--normal-proto", required=True, help="Base normal prototype (.npy).")
    parser.add_argument("--abnormal-proto", required=True, help="Base abnormal prototype (.npy).")
    parser.add_argument("--hn-aware-normal-proto", default="", help="Optional HN-aware normal prototype (.npy).")
    parser.add_argument("--output-dir", required=True, help="Output directory for PTSG artifacts.")
    parser.add_argument("--normal-class", default=DEFAULT_NORMAL_CLASS, help="Class treated as normal.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Safe-normal score alpha.")
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="Safe-normal score beta.")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="Safe-normal score gamma.")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="Small epsilon for divisions/logs.")
    parser.add_argument("--max-iter", type=int, default=200, help="LBFGS max iterations for temperature fitting.")
    return parser.parse_args()


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_prob_to_logit(prob: float, eps: float) -> float:
    clipped = min(max(prob, eps), 1.0 - eps)
    return math.log(clipped / (1.0 - clipped))


def binary_entropy(prob: float, eps: float) -> float:
    clipped = min(max(prob, eps), 1.0 - eps)
    entropy = -(clipped * math.log(clipped) + (1.0 - clipped) * math.log(1.0 - clipped))
    return entropy / math.log(2.0)


def load_split_lookup(path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = str(row["img_rel_path"]).replace("\\", "/")
            lookup[key] = str(row["subset"]).strip()
    return lookup


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int) -> float:
    criterion = torch.nn.BCEWithLogitsLoss()
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(logits / torch.exp(log_temperature), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).item())


def choose_best_variant(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def score_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        spec_995 = float(row["spec_at_r995"]) if row["spec_at_r995"] != "" else -1.0
        spec_990 = float(row["spec_at_r990"]) if row["spec_at_r990"] != "" else -1.0
        prec_990 = float(row["prec_at_r990"]) if row["prec_at_r990"] != "" else -1.0
        ptr_990 = float(row["ptr_at_r990"]) if row["ptr_at_r990"] != "" else 999.0
        return (spec_995, spec_990, prec_990, -ptr_990)

    return max(summary_rows, key=score_key)


def variant_score(
    *,
    variant: str,
    p_abn_cal: float,
    trust_normal: float,
    trust_normal_hn: float | None,
    uncertainty: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    if variant == "P0":
        return p_abn_cal
    if variant == "P1":
        safe_normal = sigmoid(alpha * (1.0 - p_abn_cal) - gamma * uncertainty)
        return 1.0 - safe_normal
    if variant == "P2":
        safe_normal = sigmoid(alpha * (1.0 - p_abn_cal) + beta * trust_normal)
        return 1.0 - safe_normal
    if variant == "P3":
        safe_normal = sigmoid(alpha * (1.0 - p_abn_cal) + beta * trust_normal - gamma * uncertainty)
        return 1.0 - safe_normal
    if variant == "P4":
        trust_value = trust_normal if trust_normal_hn is None else trust_normal_hn
        safe_normal = sigmoid(alpha * (1.0 - p_abn_cal) + beta * trust_value - gamma * uncertainty)
        return 1.0 - safe_normal
    raise ValueError(f"Unknown variant: {variant}")


def main() -> None:
    args = parse_args()
    val_rows = load_rows(Path(args.val_features_csv).resolve())
    val_embeddings = np.load(Path(args.val_embeddings_npy).resolve())
    split_lookup = load_split_lookup(Path(args.val_split_csv).resolve())
    normal_proto = np.load(Path(args.normal_proto).resolve()).astype(np.float32)
    abnormal_proto = np.load(Path(args.abnormal_proto).resolve()).astype(np.float32)
    normal_proto_hn = np.load(Path(args.hn_aware_normal_proto).resolve()).astype(np.float32) if args.hn_aware_normal_proto else None
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    enriched_rows: list[dict[str, Any]] = []
    for row in val_rows:
        img_rel_path = str(row["img_rel_path"]).replace("\\", "/")
        subset = split_lookup.get(img_rel_path)
        if subset not in {"val-cal", "val-op"}:
            continue
        embedding = val_embeddings[int(row["embedding_index"])].astype(np.float32)
        p_abn_raw = float(row["p_abnormal_raw"])
        logit_raw = float(row.get("logit_abnormal") or safe_prob_to_logit(p_abn_raw, args.eps))
        d_n = float(np.linalg.norm(embedding - normal_proto))
        d_a = float(np.linalg.norm(embedding - abnormal_proto))
        trust_normal = d_a / (d_n + d_a + args.eps)
        trust_normal_hn = None
        if normal_proto_hn is not None:
            d_n_hn = float(np.linalg.norm(embedding - normal_proto_hn))
            trust_normal_hn = d_a / (d_n_hn + d_a + args.eps)
        enriched_rows.append(
            {
                **row,
                "img_rel_path": img_rel_path,
                "subset": subset,
                "y_true": 0 if row["gt_label"] == args.normal_class else 1,
                "p_abnormal_raw": p_abn_raw,
                "logit_raw": logit_raw,
                "uncertainty_entropy": binary_entropy(p_abn_raw, args.eps),
                "d_normal": d_n,
                "d_abnormal": d_a,
                "trust_normal": trust_normal,
                "trust_normal_hn": trust_normal_hn,
            }
        )

    val_cal_rows = [row for row in enriched_rows if row["subset"] == "val-cal"]
    val_op_rows = [row for row in enriched_rows if row["subset"] == "val-op"]
    if not val_cal_rows or not val_op_rows:
        raise SystemExit("Both val-cal and val-op rows are required for PTSG evaluation.")

    logits = torch.tensor([float(row["logit_raw"]) for row in val_cal_rows], dtype=torch.float64)
    labels = torch.tensor([float(row["y_true"]) for row in val_cal_rows], dtype=torch.float64)
    temperature = fit_temperature(logits, labels, args.max_iter)
    print_step("temperature", f"T={temperature:.8f}")

    long_threshold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    variants = [
        ("P0", "calibrated p_abnormal"),
        ("P1", "p_abnormal + uncertainty"),
        ("P2", "p_abnormal + trust"),
        ("P3", "p_abnormal + trust + uncertainty"),
        ("P4", "P3 + HN-aware normal bank"),
    ]

    val_op_prediction_views: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in variants}
    for row in val_op_rows:
        scaled_logit = float(row["logit_raw"]) / temperature
        p_abn_cal = sigmoid(scaled_logit)
        base_prediction = {
            "img_id": row["img_id"],
            "img_rel_path": row["img_rel_path"],
            "gt_label": row["gt_label"],
            "subset": row["subset"],
            "y_true": row["y_true"],
            "p_abnormal_raw": round(float(row["p_abnormal_raw"]), 12),
            "p_abnormal_cal": round(p_abn_cal, 12),
            "uncertainty_entropy": round(float(row["uncertainty_entropy"]), 12),
            "d_normal": round(float(row["d_normal"]), 12),
            "d_abnormal": round(float(row["d_abnormal"]), 12),
            "trust_normal": round(float(row["trust_normal"]), 12),
            "trust_normal_hn": "" if row["trust_normal_hn"] is None else round(float(row["trust_normal_hn"]), 12),
        }
        for variant_name, _ in variants:
            abnormal_score = variant_score(
                variant=variant_name,
                p_abn_cal=p_abn_cal,
                trust_normal=float(row["trust_normal"]),
                trust_normal_hn=None if row["trust_normal_hn"] is None else float(row["trust_normal_hn"]),
                uncertainty=float(row["uncertainty_entropy"]),
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
            )
            safe_score = 1.0 - abnormal_score
            base_prediction[f"{variant_name}_abnormal_score"] = round(abnormal_score, 12)
            base_prediction[f"{variant_name}_safe_score"] = round(safe_score, 12)
            val_op_prediction_views[variant_name].append(
                {
                    "abnormal_conf": abnormal_score,
                    "gt_label": row["gt_label"],
                    "is_abnormal": bool(row["y_true"]),
                }
            )
        prediction_rows.append(base_prediction)

    for variant_name, variant_desc in variants:
        threshold_rows = compute_binary_threshold_rows(val_op_prediction_views[variant_name], DEFAULT_THRESHOLDS)
        threshold_summary = build_threshold_summary(threshold_rows, val_op_prediction_views[variant_name], args.normal_class)
        calibration_rows, calibration_summary = build_calibration_rows(val_op_prediction_views[variant_name], args.normal_class)

        for threshold_row in threshold_rows:
            long_threshold_rows.append({"variant": variant_name, "variant_desc": variant_desc, **threshold_row})

        op_995 = threshold_summary["operating_points"]["recall_ge_99_5"]
        op_990 = threshold_summary["operating_points"]["recall_ge_99_0"]
        summary_row = {
            "variant": variant_name,
            "description": variant_desc,
            "temperature": round(temperature, 8),
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "auroc": threshold_summary["auroc_exact"],
            "average_precision": threshold_summary["average_precision_exact"],
            "ece": calibration_summary["ece"],
            "brier_score": calibration_summary["brier_score"],
            "spec_at_r995": "" if op_995 is None else op_995["specificity"],
            "spec_at_r990": "" if op_990 is None else op_990["specificity"],
            "prec_at_r990": "" if op_990 is None else op_990["precision"],
            "ptr_at_r990": "" if op_990 is None else op_990["ptr"],
            "threshold_at_r995": "" if op_995 is None else op_995["threshold"],
            "threshold_at_r990": "" if op_990 is None else op_990["threshold"],
        }
        summary_rows.append(summary_row)

        variant_dir = output_dir / variant_name.lower()
        variant_dir.mkdir(parents=True, exist_ok=True)
        write_csv(variant_dir / "threshold_sweep.csv", list(threshold_rows[0].keys()), threshold_rows)
        write_json(variant_dir / "threshold_summary.json", threshold_summary)
        if calibration_rows:
            write_csv(variant_dir / "calibration_curve.csv", list(calibration_rows[0].keys()), calibration_rows)
        write_json(variant_dir / "calibration_summary.json", calibration_summary)

    best_variant = choose_best_variant(summary_rows)
    write_csv(output_dir / "ptsg_val_op_predictions.csv", list(prediction_rows[0].keys()), prediction_rows)
    write_csv(output_dir / "ptsg_threshold_sweep.csv", list(long_threshold_rows[0].keys()), long_threshold_rows)
    write_csv(output_dir / "ptsg_summary.csv", list(summary_rows[0].keys()), summary_rows)
    write_json(
        output_dir / "best_ptsg_config.json",
        {
            "best_variant": best_variant["variant"],
            "description": best_variant["description"],
            "temperature": best_variant["temperature"],
            "alpha": best_variant["alpha"],
            "beta": best_variant["beta"],
            "gamma": best_variant["gamma"],
            "ranking_rule": [
                "Spec@R99.5 descending",
                "Spec@R99.0 descending",
                "Prec@R99.0 descending",
                "PTR@R99.0 ascending",
            ],
        },
    )

    summary_lines = [
        "# Stage-1 PTSG Summary",
        "",
        f"- Temperature: `{temperature:.8f}`",
        f"- Alpha/Beta/Gamma: `{args.alpha} / {args.beta} / {args.gamma}`",
        f"- Best variant: `{best_variant['variant']}` ({best_variant['description']})",
        "",
        "| Variant | Description | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | ECE | Brier |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        summary_lines.append(
            "| {variant} | {description} | {spec_at_r995} | {spec_at_r990} | {prec_at_r990} | {ptr_at_r990} | {ece} | {brier_score} |".format(
                **row
            )
        )
    (output_dir / "ptsg_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print_step("done", f"wrote {output_dir}")


if __name__ == "__main__":
    main()
