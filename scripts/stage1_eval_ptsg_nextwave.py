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
DEFAULT_DELTA = 0.5
DEFAULT_EPS = 1e-8
DEFAULT_NORMAL_CLASS = "Normal"
DEFAULT_K_VALUES = (4, 8)
DEFAULT_SEED = 20260330
DEFAULT_MAX_ITER = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate next-wave multi-prototype PTSG variants.")
    parser.add_argument("--train-features-csv", required=True, help="Training feature index CSV.")
    parser.add_argument("--train-embeddings-npy", required=True, help="Training embeddings matrix.")
    parser.add_argument("--val-features-csv", required=True, help="Validation feature index CSV.")
    parser.add_argument("--val-embeddings-npy", required=True, help="Validation embeddings matrix.")
    parser.add_argument("--val-split-csv", required=True, help="CSV defining val-cal / val-op subsets.")
    parser.add_argument("--output-dir", required=True, help="Output directory for next-wave PTSG artifacts.")
    parser.add_argument("--normal-class", default=DEFAULT_NORMAL_CLASS, help="Class treated as normal.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Safe-normal score alpha.")
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="Safe-normal score beta.")
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA, help="Margin trust delta.")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="Small epsilon for divisions/logs.")
    parser.add_argument("--k-values", nargs="+", type=int, default=list(DEFAULT_K_VALUES), help="Normal prototype counts.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="KMeans random seed.")
    parser.add_argument("--kmeans-max-iter", type=int, default=DEFAULT_MAX_ITER, help="KMeans max iterations.")
    parser.add_argument("--temperature-max-iter", type=int, default=200, help="LBFGS max iterations for temperature fitting.")
    return parser.parse_args()


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_split_lookup(path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = str(row["img_rel_path"]).replace("\\", "/")
            lookup[key] = str(row["subset"]).strip()
    return lookup


def safe_prob_to_logit(prob: float, eps: float) -> float:
    clipped = min(max(prob, eps), 1.0 - eps)
    return math.log(clipped / (1.0 - clipped))


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


def mean_prototype(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        raise SystemExit("Cannot build prototype from empty embedding set.")
    return embeddings.mean(axis=0, dtype=np.float64).astype(np.float32)


def kmeans_plus_plus_init(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n_samples = data.shape[0]
    if n_samples < k:
        raise SystemExit(f"KMeans requires at least {k} normal embeddings, got {n_samples}.")
    centroids = np.empty((k, data.shape[1]), dtype=np.float32)
    first_idx = int(rng.integers(0, n_samples))
    centroids[0] = data[first_idx]
    closest_sq = np.sum((data - centroids[0]) ** 2, axis=1)
    for idx in range(1, k):
        total = float(closest_sq.sum())
        if total <= 0.0:
            centroids[idx] = data[int(rng.integers(0, n_samples))]
            continue
        probs = closest_sq / total
        next_idx = int(rng.choice(n_samples, p=probs))
        centroids[idx] = data[next_idx]
        new_sq = np.sum((data - centroids[idx]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)
    return centroids


def run_kmeans(data: np.ndarray, k: int, seed: int, max_iter: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centroids = kmeans_plus_plus_init(data, k, rng)
    assignments = np.full((data.shape[0],), -1, dtype=np.int32)
    for _ in range(max_iter):
        distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        new_assignments = distances.argmin(axis=1).astype(np.int32)
        if np.array_equal(assignments, new_assignments):
            assignments = new_assignments
            break
        assignments = new_assignments
        for index in range(k):
            mask = assignments == index
            if not np.any(mask):
                centroids[index] = data[int(rng.integers(0, data.shape[0]))]
                continue
            centroids[index] = data[mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    return centroids.astype(np.float32), assignments


def min_distance_to_centroids(vector: np.ndarray, centroids: np.ndarray) -> float:
    distances = np.linalg.norm(centroids - vector[None, :], axis=1)
    return float(distances.min())


def build_variant_score(
    *,
    variant: str,
    p_abn_cal: float,
    trust_single: float,
    trust_multi: float | None,
    margin_multi: float | None,
    alpha: float,
    beta: float,
    delta: float,
) -> float:
    if variant == "P2":
        safe_normal = sigmoid(alpha * (1.0 - p_abn_cal) + beta * trust_single)
        return 1.0 - safe_normal
    if variant in {"P5a", "P5b"}:
        safe_normal = sigmoid(alpha * (1.0 - p_abn_cal) + beta * float(trust_multi))
        return 1.0 - safe_normal
    if variant in {"P6a", "P6b"}:
        safe_normal = sigmoid(alpha * (1.0 - p_abn_cal) + beta * float(trust_multi) + delta * float(margin_multi))
        return 1.0 - safe_normal
    raise ValueError(f"Unknown variant: {variant}")


def choose_best_variant(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def score_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(row["spec_at_r995"]),
            float(row["spec_at_r990"]),
            float(row["prec_at_r990"]),
            -float(row["ptr_at_r990"]),
        )

    return max(summary_rows, key=score_key)


def metric_delta_text(value: float) -> str:
    return f"{value:+.4f}"


def build_auto_conclusion(summary_rows: list[dict[str, Any]], baseline_variant: str = "P2") -> tuple[str, str]:
    baseline = next(row for row in summary_rows if row["variant"] == baseline_variant)
    best_candidate: dict[str, Any] | None = None
    for row in summary_rows:
        if row["variant"] == baseline_variant:
            continue
        spec995_gain = float(row["spec_at_r995"]) - float(baseline["spec_at_r995"])
        tn995_gain = int(row["tn_at_r995"]) - int(baseline["tn_at_r995"])
        fn995_delta = int(row["fn_at_r995"]) - int(baseline["fn_at_r995"])
        tn990_gain = int(row["tn_at_r990"]) - int(baseline["tn_at_r990"])
        prec990_ok = float(row["prec_at_r990"]) >= float(baseline["prec_at_r990"])
        worthwhile = (
            spec995_gain >= 0.01
            or (tn995_gain >= 1 and fn995_delta <= 0)
            or (tn990_gain >= 2 and prec990_ok)
        )
        if worthwhile:
            if best_candidate is None or choose_best_variant([best_candidate, row])["variant"] == row["variant"]:
                best_candidate = row
    if best_candidate is None:
        return (
            "stop_at_stage1",
            "P5/P6 do not clearly beat P2; multi-prototype trust does not add enough value in this round.",
        )
    return (
        "worth_continue",
        f"{best_candidate['variant']} beats the continue rule relative to P2 and is the best selector candidate for the next step.",
    )


def write_metric_plot(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print_step("warn", f"skip plot generation: {exc}")
        return

    variants = [row["variant"] for row in summary_rows]
    metric_specs = [
        ("spec_at_r995", "Spec@R99.5"),
        ("spec_at_r990", "Spec@R99.0"),
        ("prec_at_r990", "Prec@R99.0"),
        ("ptr_at_r990", "PTR@R99.0"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, (key, title) in zip(axes.flat, metric_specs, strict=True):
        values = [float(row[key]) for row in summary_rows]
        axis.bar(variants, values, color=["#4E79A7", "#59A14F", "#76B7B2", "#F28E2B", "#E15759"])
        axis.set_title(title)
        if key == "ptr_at_r990":
            axis.set_ylim(min(values) * 0.995, max(values) * 1.01)
        else:
            axis.set_ylim(min(values) * 0.98, max(values) * 1.02)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    train_rows = load_rows(Path(args.train_features_csv).resolve())
    train_embeddings = np.load(Path(args.train_embeddings_npy).resolve())
    val_rows = load_rows(Path(args.val_features_csv).resolve())
    val_embeddings = np.load(Path(args.val_embeddings_npy).resolve())
    split_lookup = load_split_lookup(Path(args.val_split_csv).resolve())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    normal_train_indices = [int(row["embedding_index"]) for row in train_rows if row["gt_label"] == args.normal_class]
    abnormal_train_indices = [int(row["embedding_index"]) for row in train_rows if row["gt_label"] != args.normal_class]
    normal_train_embeddings = train_embeddings[normal_train_indices].astype(np.float32)
    abnormal_train_embeddings = train_embeddings[abnormal_train_indices].astype(np.float32)

    single_normal_proto = mean_prototype(normal_train_embeddings)
    abnormal_proto = mean_prototype(abnormal_train_embeddings)
    np.save(output_dir / "single_normal_proto.npy", single_normal_proto)
    np.save(output_dir / "single_abnormal_proto.npy", abnormal_proto)

    multi_proto_lookup: dict[int, np.ndarray] = {}
    for k_value in sorted(set(int(value) for value in args.k_values)):
        centroids, assignments = run_kmeans(normal_train_embeddings, k_value, args.seed, args.kmeans_max_iter)
        multi_proto_lookup[k_value] = centroids
        np.save(output_dir / f"normal_multi_proto_k{k_value}.npy", centroids)
        np.save(output_dir / f"normal_multi_proto_k{k_value}_assignments.npy", assignments)

    enriched_rows: list[dict[str, Any]] = []
    for row in val_rows:
        img_rel_path = str(row["img_rel_path"]).replace("\\", "/")
        subset = split_lookup.get(img_rel_path)
        if subset not in {"val-cal", "val-op"}:
            continue
        embedding = val_embeddings[int(row["embedding_index"])].astype(np.float32)
        p_abn_raw = float(row["p_abnormal_raw"])
        logit_raw = float(row.get("logit_abnormal") or safe_prob_to_logit(p_abn_raw, args.eps))
        d_single_n = float(np.linalg.norm(embedding - single_normal_proto))
        d_a = float(np.linalg.norm(embedding - abnormal_proto))
        trust_single = d_a / (d_single_n + d_a + args.eps)
        row_payload: dict[str, Any] = {
            **row,
            "img_rel_path": img_rel_path,
            "subset": subset,
            "y_true": 0 if row["gt_label"] == args.normal_class else 1,
            "p_abnormal_raw": p_abn_raw,
            "logit_raw": logit_raw,
            "d_single_normal": d_single_n,
            "d_abnormal": d_a,
            "trust_single": trust_single,
        }
        for k_value, centroids in multi_proto_lookup.items():
            d_multi_n = min_distance_to_centroids(embedding, centroids)
            row_payload[f"d_multi_normal_k{k_value}"] = d_multi_n
            row_payload[f"trust_multi_k{k_value}"] = d_a / (d_multi_n + d_a + args.eps)
            row_payload[f"margin_multi_k{k_value}"] = (d_a - d_multi_n) / (d_a + d_multi_n + args.eps)
        enriched_rows.append(row_payload)

    val_cal_rows = [row for row in enriched_rows if row["subset"] == "val-cal"]
    val_op_rows = [row for row in enriched_rows if row["subset"] == "val-op"]
    if not val_cal_rows or not val_op_rows:
        raise SystemExit("Both val-cal and val-op rows are required for next-wave PTSG evaluation.")

    logits = torch.tensor([float(row["logit_raw"]) for row in val_cal_rows], dtype=torch.float64)
    labels = torch.tensor([float(row["y_true"]) for row in val_cal_rows], dtype=torch.float64)
    temperature = fit_temperature(logits, labels, args.temperature_max_iter)
    print_step("temperature", f"T={temperature:.8f}")

    variants = [
        ("P2", "single-prototype trust", None),
        ("P5a", "K4 multi-prototype trust", 4),
        ("P5b", "K8 multi-prototype trust", 8),
        ("P6a", "K4 multi-prototype + margin trust", 4),
        ("P6b", "K8 multi-prototype + margin trust", 8),
    ]

    prediction_rows: list[dict[str, Any]] = []
    long_threshold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    val_op_prediction_views: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in variants}

    for row in val_op_rows:
        p_abn_cal = sigmoid(float(row["logit_raw"]) / temperature)
        base_row = {
            "img_id": row["img_id"],
            "img_rel_path": row["img_rel_path"],
            "gt_label": row["gt_label"],
            "subset": row["subset"],
            "y_true": row["y_true"],
            "p_abnormal_raw": round(float(row["p_abnormal_raw"]), 12),
            "p_abnormal_cal": round(p_abn_cal, 12),
            "trust_single": round(float(row["trust_single"]), 12),
        }
        for variant_name, _, k_value in variants:
            trust_multi = None if k_value is None else float(row[f"trust_multi_k{k_value}"])
            margin_multi = None if k_value is None else float(row[f"margin_multi_k{k_value}"])
            abnormal_score = build_variant_score(
                variant=variant_name,
                p_abn_cal=p_abn_cal,
                trust_single=float(row["trust_single"]),
                trust_multi=trust_multi,
                margin_multi=margin_multi,
                alpha=args.alpha,
                beta=args.beta,
                delta=args.delta,
            )
            base_row[f"{variant_name}_abnormal_score"] = round(abnormal_score, 12)
            base_row[f"{variant_name}_safe_score"] = round(1.0 - abnormal_score, 12)
            val_op_prediction_views[variant_name].append(
                {
                    "abnormal_conf": abnormal_score,
                    "gt_label": row["gt_label"],
                    "is_abnormal": bool(row["y_true"]),
                }
            )
        prediction_rows.append(base_row)

    for variant_name, variant_desc, k_value in variants:
        threshold_rows = compute_binary_threshold_rows(val_op_prediction_views[variant_name], DEFAULT_THRESHOLDS)
        threshold_summary = build_threshold_summary(threshold_rows, val_op_prediction_views[variant_name], args.normal_class)
        calibration_rows, calibration_summary = build_calibration_rows(val_op_prediction_views[variant_name], args.normal_class)
        for threshold_row in threshold_rows:
            long_threshold_rows.append({"variant": variant_name, "variant_desc": variant_desc, **threshold_row})
        op_995 = threshold_summary["operating_points"]["recall_ge_99_5"]
        op_990 = threshold_summary["operating_points"]["recall_ge_99_0"]
        summary_rows.append(
            {
                "variant": variant_name,
                "description": variant_desc,
                "k_normal_proto": "" if k_value is None else k_value,
                "temperature": round(temperature, 8),
                "alpha": args.alpha,
                "beta": args.beta,
                "delta": args.delta,
                "auroc": threshold_summary["auroc_exact"],
                "average_precision": threshold_summary["average_precision_exact"],
                "ece": calibration_summary["ece"],
                "brier_score": calibration_summary["brier_score"],
                "spec_at_r995": op_995["specificity"],
                "spec_at_r990": op_990["specificity"],
                "prec_at_r990": op_990["precision"],
                "ptr_at_r990": op_990["ptr"],
                "threshold_at_r995": op_995["threshold"],
                "threshold_at_r990": op_990["threshold"],
                "tp_at_r995": op_995["tp"],
                "fn_at_r995": op_995["fn"],
                "fp_at_r995": op_995["fp"],
                "tn_at_r995": op_995["tn"],
                "tp_at_r990": op_990["tp"],
                "fn_at_r990": op_990["fn"],
                "fp_at_r990": op_990["fp"],
                "tn_at_r990": op_990["tn"],
            }
        )
        variant_dir = output_dir / variant_name.lower()
        variant_dir.mkdir(parents=True, exist_ok=True)
        write_csv(variant_dir / "threshold_sweep.csv", list(threshold_rows[0].keys()), threshold_rows)
        write_json(variant_dir / "threshold_summary.json", threshold_summary)
        if calibration_rows:
            write_csv(variant_dir / "calibration_curve.csv", list(calibration_rows[0].keys()), calibration_rows)
        write_json(variant_dir / "calibration_summary.json", calibration_summary)

    best_variant = choose_best_variant(summary_rows)
    verdict_label, verdict_detail = build_auto_conclusion(summary_rows)

    write_csv(output_dir / "ptsg_nextwave_predictions.csv", list(prediction_rows[0].keys()), prediction_rows)
    write_csv(output_dir / "ptsg_nextwave_threshold_sweep.csv", list(long_threshold_rows[0].keys()), long_threshold_rows)
    write_csv(output_dir / "ptsg_nextwave_summary.csv", list(summary_rows[0].keys()), summary_rows)
    write_json(
        output_dir / "best_ptsg_nextwave_config.json",
        {
            "best_variant": best_variant["variant"],
            "description": best_variant["description"],
            "k_normal_proto": best_variant["k_normal_proto"],
            "temperature": best_variant["temperature"],
            "alpha": best_variant["alpha"],
            "beta": best_variant["beta"],
            "delta": best_variant["delta"],
            "verdict": verdict_label,
            "verdict_detail": verdict_detail,
            "ranking_rule": [
                "Spec@R99.5 descending",
                "Spec@R99.0 descending",
                "Prec@R99.0 descending",
                "PTR@R99.0 ascending",
            ],
        },
    )

    p2_row = next(row for row in summary_rows if row["variant"] == "P2")
    summary_lines = [
        "# Stage-1 PTSG Next Wave Summary",
        "",
        f"- Temperature: `{temperature:.8f}`",
        f"- Alpha/Beta/Delta: `{args.alpha} / {args.beta} / {args.delta}`",
        f"- Best variant: `{best_variant['variant']}` ({best_variant['description']})",
        f"- Verdict: `{verdict_label}`",
        f"- Detail: {verdict_detail}",
        "",
        "| Variant | Description | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | TN@R99.5 | FN@R99.5 | TN@R99.0 | FN@R99.0 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        summary_lines.append(
            "| {variant} | {description} | {spec_at_r995} | {spec_at_r990} | {prec_at_r990} | {ptr_at_r990} | {tn_at_r995} | {fn_at_r995} | {tn_at_r990} | {fn_at_r990} |".format(
                **row
            )
        )
    summary_lines.extend(["", "## Relative To P2", ""])
    for row in summary_rows:
        if row["variant"] == "P2":
            continue
        summary_lines.append(
            f"- `{row['variant']}`: "
            f"Spec@R99.5 {metric_delta_text(float(row['spec_at_r995']) - float(p2_row['spec_at_r995']))}, "
            f"Spec@R99.0 {metric_delta_text(float(row['spec_at_r990']) - float(p2_row['spec_at_r990']))}, "
            f"Prec@R99.0 {metric_delta_text(float(row['prec_at_r990']) - float(p2_row['prec_at_r990']))}, "
            f"PTR@R99.0 {metric_delta_text(float(row['ptr_at_r990']) - float(p2_row['ptr_at_r990']))}, "
            f"TN@R99.5 {int(row['tn_at_r995']) - int(p2_row['tn_at_r995']):+d}, "
            f"FN@R99.5 {int(row['fn_at_r995']) - int(p2_row['fn_at_r995']):+d}, "
            f"TN@R99.0 {int(row['tn_at_r990']) - int(p2_row['tn_at_r990']):+d}, "
            f"FN@R99.0 {int(row['fn_at_r990']) - int(p2_row['fn_at_r990']):+d}"
        )
    (output_dir / "ptsg_nextwave_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    write_metric_plot(summary_rows, output_dir / "ptsg_nextwave_metrics.png")
    print_step("done", f"wrote {output_dir}")


if __name__ == "__main__":
    main()
