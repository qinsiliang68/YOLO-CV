from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import matplotlib
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


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATERIALS_ROOT = REPO_ROOT / "research" / "materials"
DEFAULT_RUNS = ["yolo11l_gate2_train7200", "yolo11m_gate2_train7200"]
DEFAULT_OUTPUT_SUBDIR = "calibration_ts"
DEFAULT_SEED = 20260330
DEFAULT_CAL_FRACTION = 0.30
DEFAULT_BINS = 10
DEFAULT_EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run temperature scaling on existing gate raw materials without retraining."
    )
    parser.add_argument(
        "--materials-root",
        default=str(DEFAULT_MATERIALS_ROOT),
        help="Root directory containing run material folders.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=DEFAULT_RUNS,
        help="Run directories under materials root to calibrate.",
    )
    parser.add_argument(
        "--output-subdir",
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Per-run output subdirectory for calibration artifacts.",
    )
    parser.add_argument("--normal-class", default="Normal", help="Class treated as normal.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for stratified split.")
    parser.add_argument(
        "--cal-fraction",
        type=float,
        default=DEFAULT_CAL_FRACTION,
        help="Fraction of validation set reserved for temperature fitting.",
    )
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS, help="Number of calibration bins.")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="Probability clamp epsilon.")
    parser.add_argument("--max-iter", type=int, default=200, help="LBFGS max iterations.")
    return parser.parse_args()


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def load_prediction_rows(path: Path, normal_class: str, eps: float) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise SystemExit(f"No rows in {path}")

    enriched: list[dict[str, Any]] = []
    for row in rows:
        probability_raw = row.get("p_abnormal") or row.get("abnormal_conf") or row.get("top1_prob")
        if probability_raw is None or probability_raw == "":
            continue
        p_abnormal = float(probability_raw)
        p_abnormal = min(max(p_abnormal, eps), 1.0 - eps)
        logit_raw = math.log(p_abnormal / (1.0 - p_abnormal))
        gt_label = str(row["gt_label"])
        enriched_row = dict(row)
        enriched_row["y_true"] = 0 if gt_label == normal_class else 1
        enriched_row["p_abnormal_raw"] = round(p_abnormal, 12)
        enriched_row["logit_raw"] = round(logit_raw, 12)
        enriched.append(enriched_row)
    return enriched


def stratified_split(
    rows: list[dict[str, Any]],
    *,
    cal_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < cal_fraction < 1.0:
        raise SystemExit(f"cal_fraction must be between 0 and 1, got {cal_fraction}")

    groups: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for row in rows:
        groups[int(row["y_true"])].append(row)

    rng = random.Random(seed)
    val_cal: list[dict[str, Any]] = []
    val_op: list[dict[str, Any]] = []

    for label, group in groups.items():
        group_sorted = sorted(group, key=lambda item: (item["img_id"], item["img_rel_path"]))
        rng.shuffle(group_sorted)
        cal_count = int(round(len(group_sorted) * cal_fraction))
        if len(group_sorted) > 1:
            cal_count = max(1, min(len(group_sorted) - 1, cal_count))
        val_cal.extend(group_sorted[:cal_count])
        val_op.extend(group_sorted[cal_count:])

    val_cal = sorted(val_cal, key=lambda item: int(item["row_id"]))
    val_op = sorted(val_op, key=lambda item: int(item["row_id"]))
    return val_cal, val_op


def rows_to_model_inputs(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.tensor([float(row["logit_raw"]) for row in rows], dtype=torch.float64)
    labels = torch.tensor([float(row["y_true"]) for row in rows], dtype=torch.float64)
    return logits, labels


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_iter: int,
) -> tuple[float, float, float]:
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

    before_nll = float(criterion(logits, labels).item())
    optimizer.step(closure)
    temperature = float(torch.exp(log_temperature).item())
    after_nll = float(criterion(logits / temperature, labels).item())
    return temperature, before_nll, after_nll


def attach_calibrated_scores(rows: list[dict[str, Any]], temperature: float) -> list[dict[str, Any]]:
    calibrated: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        scaled_logit = float(row["logit_raw"]) / temperature
        p_calibrated = 1.0 / (1.0 + math.exp(-scaled_logit))
        new_row["temperature"] = round(temperature, 12)
        new_row["logit_scaled"] = round(scaled_logit, 12)
        new_row["p_abnormal_calibrated"] = round(p_calibrated, 12)
        calibrated.append(new_row)
    return calibrated


def prediction_view(rows: list[dict[str, Any]], score_field: str, normal_class: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                "abnormal_conf": float(row[score_field]),
                "gt_label": row["gt_label"],
                "is_abnormal": row["gt_label"] != normal_class,
            }
        )
    return output


def build_op_metrics(rows: list[dict[str, Any]], normal_class: str, bins: int) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    threshold_rows = compute_binary_threshold_rows(rows, DEFAULT_THRESHOLDS)
    threshold_summary = build_threshold_summary(threshold_rows, rows, normal_class)
    calibration_rows, calibration_summary = build_calibration_rows(rows, normal_class, bins=bins)
    return threshold_rows, threshold_summary, calibration_rows, calibration_summary


def write_split_rows(path: Path, rows: list[dict[str, Any]], extra_fields: list[str] | None = None) -> None:
    fieldnames = [
        "row_id",
        "img_id",
        "split",
        "gt_label",
        "y_true",
        "img_rel_path",
        "pred_label",
        "correct",
        "p_normal",
        "p_abnormal",
        "abnormal_conf",
        "p_abnormal_raw",
        "logit_raw",
    ]
    if extra_fields:
        fieldnames.extend(extra_fields)
    rows_to_write: list[dict[str, Any]] = []
    for row in rows:
        item = {key: row.get(key, "") for key in fieldnames}
        rows_to_write.append(item)
    write_csv(path, fieldnames, rows_to_write)


def flatten_operating_points(summary: dict[str, Any]) -> dict[str, float | str]:
    operating_points = summary.get("operating_points", {})
    result: dict[str, float | str] = {
        "auroc": float(summary.get("auroc_exact", 0.0)),
        "average_precision": float(summary.get("average_precision_exact", 0.0)),
    }
    for name, row in operating_points.items():
        prefix = name
        if row is None:
            result[f"{prefix}_threshold"] = ""
            result[f"{prefix}_specificity"] = ""
            result[f"{prefix}_precision"] = ""
            result[f"{prefix}_ptr"] = ""
            continue
        result[f"{prefix}_threshold"] = float(row["threshold"])
        result[f"{prefix}_specificity"] = float(row["specificity"])
        result[f"{prefix}_precision"] = float(row["precision"])
        result[f"{prefix}_ptr"] = float(row["ptr"])
    return result


def build_operating_point_comparison(before_summary: dict[str, Any], after_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = ["recall_ge_99_5", "recall_ge_99_0", "recall_ge_98_0"]
    for key in keys:
        before = before_summary.get("operating_points", {}).get(key)
        after = after_summary.get("operating_points", {}).get(key)
        rows.append(
            {
                "operating_point": key,
                "before_threshold": "" if before is None else before["threshold"],
                "after_threshold": "" if after is None else after["threshold"],
                "before_specificity": "" if before is None else before["specificity"],
                "after_specificity": "" if after is None else after["specificity"],
                "before_precision": "" if before is None else before["precision"],
                "after_precision": "" if after is None else after["precision"],
                "before_ptr": "" if before is None else before["ptr"],
                "after_ptr": "" if after is None else after["ptr"],
            }
        )
    return rows


def save_figure(fig: plt.Figure, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_reliability(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]], output_prefix: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, rows, title in zip(
        axes,
        (before_rows, after_rows),
        ("校准前", "校准后"),
        strict=True,
    ):
        xs = []
        ys = []
        for row in rows:
            if row["avg_confidence"] is None or row["empirical_positive_rate"] is None:
                continue
            xs.append(float(row["avg_confidence"]))
            ys.append(float(row["empirical_positive_rate"]))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0)
        if xs:
            ax.bar(xs, ys, width=0.08, color="#4C78A8", alpha=0.75)
            ax.plot(xs, ys, marker="o", color="#F58518", linewidth=1.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("平均置信度")
        ax.set_title(title)
    axes[0].set_ylabel("经验异常比例")
    fig.suptitle("Reliability Diagram 对比", fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_prefix)


def plot_ece_brier(before_summary: dict[str, Any], after_summary: dict[str, Any], output_prefix: Path) -> None:
    labels = ["ECE", "Brier"]
    before_values = [float(before_summary["ece"]), float(before_summary["brier_score"])]
    after_values = [float(after_summary["ece"]), float(after_summary["brier_score"])]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(x - width / 2, before_values, width=width, label="校准前", color="#4C78A8")
    ax.bar(x + width / 2, after_values, width=width, label="校准后", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("指标值")
    ax.set_title("ECE 与 Brier Score 对比")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_prefix)


def plot_threshold_sweep(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]], output_prefix: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for rows, label, color in (
        (before_rows, "校准前", "#4C78A8"),
        (after_rows, "校准后", "#F58518"),
    ):
        recalls = [float(row["recall"]) for row in rows]
        specificities = [float(row["specificity"]) for row in rows]
        precisions = [float(row["precision"]) for row in rows]
        axes[0].plot(recalls, specificities, label=label, color=color, linewidth=1.5)
        axes[1].plot(recalls, precisions, label=label, color=color, linewidth=1.5)

    axes[0].set_xlabel("召回率")
    axes[0].set_ylabel("特异度")
    axes[0].set_title("Specificity-Recall 曲线")
    axes[1].set_xlabel("召回率")
    axes[1].set_ylabel("精确率")
    axes[1].set_title("Precision-Recall 曲线")
    for ax in axes:
        ax.set_xlim(0.9, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(frameon=False)
    fig.suptitle("Calibration 前后 Threshold Sweep 对比", fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_prefix)


def run_single(
    materials_root: Path,
    run_name: str,
    *,
    output_subdir: str,
    normal_class: str,
    seed: int,
    cal_fraction: float,
    bins: int,
    eps: float,
    max_iter: int,
) -> dict[str, Any]:
    run_dir = materials_root / run_name
    prediction_path = run_dir / "val_predictions.csv"
    if not prediction_path.exists():
        raise SystemExit(f"Missing val_predictions.csv for {run_name}: {prediction_path}")

    output_dir = run_dir / output_subdir
    rows = load_prediction_rows(prediction_path, normal_class, eps)
    val_cal, val_op = stratified_split(rows, cal_fraction=cal_fraction, seed=seed)
    logits_cal, labels_cal = rows_to_model_inputs(val_cal)
    temperature, nll_before, nll_after = fit_temperature(logits_cal, labels_cal, max_iter=max_iter)
    val_op_calibrated = attach_calibrated_scores(val_op, temperature)

    val_op_before_view = prediction_view(val_op, "p_abnormal_raw", normal_class)
    val_op_after_view = prediction_view(val_op_calibrated, "p_abnormal_calibrated", normal_class)

    threshold_before_rows, threshold_before_summary, calibration_before_rows, calibration_before_summary = build_op_metrics(
        val_op_before_view, normal_class, bins
    )
    threshold_after_rows, threshold_after_summary, calibration_after_rows, calibration_after_summary = build_op_metrics(
        val_op_after_view, normal_class, bins
    )

    write_split_rows(output_dir / "val_cal.csv", val_cal)
    write_split_rows(output_dir / "val_op.csv", val_op)
    write_split_rows(
        output_dir / "val_op_predictions_calibrated.csv",
        val_op_calibrated,
        extra_fields=["temperature", "logit_scaled", "p_abnormal_calibrated"],
    )
    write_csv(output_dir / "threshold_sweep_before.csv", list(threshold_before_rows[0].keys()), threshold_before_rows)
    write_csv(output_dir / "threshold_sweep_calibrated.csv", list(threshold_after_rows[0].keys()), threshold_after_rows)
    write_json(output_dir / "threshold_operating_points_before.json", threshold_before_summary)
    write_json(output_dir / "threshold_operating_points_calibrated.json", threshold_after_summary)
    write_csv(output_dir / "calibration_curve_before.csv", list(calibration_before_rows[0].keys()), calibration_before_rows)
    write_csv(output_dir / "calibration_curve_after.csv", list(calibration_after_rows[0].keys()), calibration_after_rows)
    write_json(output_dir / "calibration_summary_before.json", calibration_before_summary)
    write_json(output_dir / "calibration_summary_after.json", calibration_after_summary)
    write_csv(
        output_dir / "operating_point_comparison.csv",
        list(build_operating_point_comparison(threshold_before_summary, threshold_after_summary)[0].keys()),
        build_operating_point_comparison(threshold_before_summary, threshold_after_summary),
    )

    y_true_cal = [int(row["y_true"]) for row in val_cal]
    val_cal_counts = {
        "abnormal": sum(y_true_cal),
        "normal": len(y_true_cal) - sum(y_true_cal),
    }
    y_true_op = [int(row["y_true"]) for row in val_op]
    val_op_counts = {
        "abnormal": sum(y_true_op),
        "normal": len(y_true_op) - sum(y_true_op),
    }
    temperature_payload = {
        "run_name": run_name,
        "seed": seed,
        "cal_fraction": cal_fraction,
        "bins": bins,
        "eps": eps,
        "temperature": round(temperature, 6),
        "fit_method": "temperature_scaling_from_probability_derived_logits",
        "fit_target": "binary_bce_with_logits",
        "nll_before_fit": round(nll_before, 6),
        "nll_after_fit": round(nll_after, 6),
        "val_cal_count": len(val_cal),
        "val_op_count": len(val_op),
        "val_cal_class_counts": val_cal_counts,
        "val_op_class_counts": val_op_counts,
    }
    write_json(output_dir / "temperature_scaling.json", temperature_payload)

    plot_reliability(calibration_before_rows, calibration_after_rows, output_dir / "reliability_diagram_before_after")
    plot_ece_brier(calibration_before_summary, calibration_after_summary, output_dir / "ece_brier_before_after")
    plot_threshold_sweep(threshold_before_rows, threshold_after_rows, output_dir / "threshold_sweep_before_after")

    before_flat = flatten_operating_points(threshold_before_summary)
    after_flat = flatten_operating_points(threshold_after_summary)
    summary_row = {
        "run_name": run_name,
        "temperature": round(temperature, 6),
        "val_cal_count": len(val_cal),
        "val_op_count": len(val_op),
        "val_cal_abnormal": val_cal_counts["abnormal"],
        "val_cal_normal": val_cal_counts["normal"],
        "val_op_abnormal": val_op_counts["abnormal"],
        "val_op_normal": val_op_counts["normal"],
        "ece_before": calibration_before_summary["ece"],
        "ece_after": calibration_after_summary["ece"],
        "brier_before": calibration_before_summary["brier_score"],
        "brier_after": calibration_after_summary["brier_score"],
        "auroc_before": before_flat["auroc"],
        "auroc_after": after_flat["auroc"],
        "ap_before": before_flat["average_precision"],
        "ap_after": after_flat["average_precision"],
        "spec_r995_before": before_flat["recall_ge_99_5_specificity"],
        "spec_r995_after": after_flat["recall_ge_99_5_specificity"],
        "spec_r990_before": before_flat["recall_ge_99_0_specificity"],
        "spec_r990_after": after_flat["recall_ge_99_0_specificity"],
        "prec_r990_before": before_flat["recall_ge_99_0_precision"],
        "prec_r990_after": after_flat["recall_ge_99_0_precision"],
        "ptr_r990_before": before_flat["recall_ge_99_0_ptr"],
        "ptr_r990_after": after_flat["recall_ge_99_0_ptr"],
    }
    print_step(
        "done",
        (
            f"{run_name}: T={summary_row['temperature']:.4f}, "
            f"ECE {summary_row['ece_before']:.4f}->{summary_row['ece_after']:.4f}, "
            f"Spec@R99.0 {summary_row['spec_r990_before']:.4f}->{summary_row['spec_r990_after']:.4f}"
        ),
    )
    return summary_row


def main() -> None:
    args = parse_args()
    materials_root = Path(args.materials_root).resolve()
    summary_rows = []
    for run_name in args.runs:
        summary_rows.append(
            run_single(
                materials_root,
                run_name,
                output_subdir=args.output_subdir,
                normal_class=args.normal_class,
                seed=args.seed,
                cal_fraction=args.cal_fraction,
                bins=args.bins,
                eps=args.eps,
                max_iter=args.max_iter,
            )
        )

    summary_path = materials_root / "gate2_temperature_scaling_summary.csv"
    write_csv(summary_path, list(summary_rows[0].keys()), summary_rows)
    print_step("summary", f"wrote {summary_path}")


if __name__ == "__main__":
    main()
