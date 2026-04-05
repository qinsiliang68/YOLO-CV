from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib
import torch

from collect_cls_raw_materials import collect_validation_predictions, write_csv, write_json
from temperature_scale_gate_runs import (
    attach_calibrated_scores,
    build_op_metrics,
    fit_temperature,
    prediction_view,
    write_split_rows,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


SUMMARY_FIELDS = [
    "epoch",
    "checkpoint_file",
    "checkpoint_path",
    "temperature_T",
    "tau_r995",
    "tau_r990",
    "spec_at_r995",
    "spec_at_r990",
    "prec_at_r990",
    "ptr_at_r990",
    "tn_at_r995",
    "fn_at_r995",
    "tn_at_r990",
    "fn_at_r990",
]

INDEX_FIELDS = [
    "epoch",
    "checkpoint_file",
    "checkpoint_path",
    "checkpoint_exists",
    "evaluated",
    "temperature_T",
    "tau_r995",
    "tau_r990",
    "spec_at_r995",
    "spec_at_r990",
    "prec_at_r990",
    "ptr_at_r990",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate formal binary gate checkpoints with external gate-aware metrics.")
    parser.add_argument("--run-dir", required=True, help="Training run directory.")
    parser.add_argument("--data-root", required=True, help="Binary gate dataset root.")
    parser.add_argument("--summary-dir", required=True, help="Formal material output directory.")
    parser.add_argument("--split-csv", required=True, help="Val-cal / val-op split CSV.")
    parser.add_argument("--normal-class", default="Normal", help="Normal class name.")
    parser.add_argument("--device", default="0", help="Inference device.")
    parser.add_argument("--batch", type=int, default=24, help="Evaluation batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Probability clamp epsilon.")
    parser.add_argument("--bins", type=int, default=10, help="Calibration bins.")
    parser.add_argument("--max-iter", type=int, default=200, help="LBFGS max iterations.")
    return parser.parse_args()


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_split_lookup(path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            lookup[str(row["img_rel_path"]).replace("\\", "/")] = str(row["subset"]).strip()
    return lookup


def safe_prob_to_logit(prob: float, eps: float) -> float:
    clipped = min(max(prob, eps), 1.0 - eps)
    return math.log(clipped / (1.0 - clipped))


def checkpoint_epoch(path: Path) -> int | None:
    match = re.fullmatch(r"epoch(\d+)\.pt", path.name)
    if not match:
        return None
    return int(match.group(1)) + 1


def formal_checkpoint_path(path: Path) -> Path:
    epoch = checkpoint_epoch(path)
    if epoch is None:
        return path
    alias = path.with_name(f"epoch_{epoch:03d}.pt")
    return alias if alias.exists() else path


def list_epoch_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    weights_dir = run_dir / "weights"
    pairs: list[tuple[int, Path]] = []
    for path in sorted(weights_dir.glob("epoch*.pt")):
        epoch = checkpoint_epoch(path)
        if epoch is not None:
            pairs.append((epoch, path))
    return sorted(pairs, key=lambda item: item[0])


def choose_best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def key_fn(row: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(row["spec_at_r995"]),
            float(row["spec_at_r990"]),
            float(row["prec_at_r990"]),
            -float(row["ptr_at_r990"]),
        )

    return max(rows, key=key_fn)


def plot_metric_dashboard(rows: list[dict[str, Any]], output_prefix: Path) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    metrics = [
        ("spec_at_r995", "Spec@R99.5", "#355C7D"),
        ("spec_at_r990", "Spec@R99.0", "#6C5B7B"),
        ("prec_at_r990", "Prec@R99.0", "#F67280"),
        ("ptr_at_r990", "PTR@R99.0", "#C06C84"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.0), sharex=True)
    for ax, (field, title, color) in zip(axes.flatten(), metrics, strict=True):
        values = [float(row[field]) for row in rows]
        ax.plot(epochs, values, color=color, linewidth=1.6, marker="o", markersize=3.0)
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    fig.suptitle("Formal gate checkpoint scan", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_md(rows: list[dict[str, Any]], best_row: dict[str, Any]) -> str:
    lines = [
        "# Formal Stage-1 Gate Epoch Summary",
        "",
        "| Epoch | Spec@R99.5 | Spec@R99.0 | Prec@R99.0 | PTR@R99.0 | Tau@R99.5 | Tau@R99.0 | TN@R99.5 | FN@R99.5 | TN@R99.0 | FN@R99.0 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {epoch} | {spec_at_r995} | {spec_at_r990} | {prec_at_r990} | {ptr_at_r990} | {tau_r995} | {tau_r990} | {tn_at_r995} | {fn_at_r995} | {tn_at_r990} | {fn_at_r990} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            f"- Best epoch: `{best_row['epoch']}`",
            f"- Best checkpoint: `{best_row['checkpoint_path']}`",
            "- Ranking rule: `Spec@R99.5 -> Spec@R99.0 -> Prec@R99.0 -> PTR@R99.0 ascending`",
        ]
    )
    return "\n".join(lines) + "\n"


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    data_root: Path,
    split_lookup: dict[str, str],
    normal_class: str,
    device: str,
    batch: int,
    imgsz: int,
    eps: float,
    bins: int,
    max_iter: int,
    temp_dir: Path,
) -> dict[str, Any]:
    _, prediction_rows, _, _ = collect_validation_predictions(
        weights_path=checkpoint_path,
        data_root=data_root,
        normal_class=normal_class,
        device=device,
        batch=batch,
        imgsz=imgsz,
        collect_embeddings=False,
    )

    enriched_rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        img_rel_path = str(row["img_rel_path"]).replace("\\", "/")
        subset = split_lookup.get(img_rel_path, "")
        abnormal_conf = float(row["abnormal_conf"] or row["p_abnormal"] or 0.0)
        enriched_rows.append(
            {
                **row,
                "subset": subset,
                "y_true": 0 if row["gt_label"] == normal_class else 1,
                "p_abnormal_raw": abnormal_conf,
                "logit_raw": safe_prob_to_logit(abnormal_conf, eps),
            }
        )

    val_cal = [row for row in enriched_rows if row["subset"] == "val-cal"]
    val_op = [row for row in enriched_rows if row["subset"] == "val-op"]
    if not val_cal or not val_op:
        raise SystemExit("Val-cal / val-op split did not match predictions.")

    logits = torch.tensor([float(row["logit_raw"]) for row in val_cal], dtype=torch.float64)
    labels = torch.tensor([float(row["y_true"]) for row in val_cal], dtype=torch.float64)
    temperature, _, _ = fit_temperature(logits=logits, labels=labels, max_iter=max_iter)
    val_op_calibrated = attach_calibrated_scores(val_op, temperature)
    calibrated_view = prediction_view(val_op_calibrated, "p_abnormal_calibrated", normal_class)
    threshold_rows, threshold_summary, calibration_rows, calibration_summary = build_op_metrics(
        calibrated_view, normal_class, bins
    )

    epoch_tag = checkpoint_epoch(checkpoint_path) or 0
    epoch_dir = temp_dir / f"epoch_{epoch_tag:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    write_split_rows(epoch_dir / "val_cal.csv", val_cal)
    write_split_rows(
        epoch_dir / "val_op_predictions_calibrated.csv",
        val_op_calibrated,
        extra_fields=["temperature", "logit_scaled", "p_abnormal_calibrated"],
    )
    if threshold_rows:
        write_csv(epoch_dir / "threshold_sweep_calibrated.csv", list(threshold_rows[0].keys()), threshold_rows)
    write_json(epoch_dir / "threshold_operating_points_calibrated.json", threshold_summary)
    if calibration_rows:
        write_csv(epoch_dir / "calibration_curve.csv", list(calibration_rows[0].keys()), calibration_rows)
    write_json(epoch_dir / "calibration_summary.json", calibration_summary)

    op_995 = threshold_summary["operating_points"]["recall_ge_99_5"]
    op_990 = threshold_summary["operating_points"]["recall_ge_99_0"]
    formal_path = formal_checkpoint_path(checkpoint_path)
    return {
        "epoch": checkpoint_epoch(checkpoint_path),
        "checkpoint_file": formal_path.name,
        "checkpoint_path": str(formal_path),
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
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    data_root = Path(args.data_root).resolve()
    summary_dir = Path(args.summary_dir).resolve()
    temp_dir = summary_dir / "per_epoch_gate"
    summary_dir.mkdir(parents=True, exist_ok=True)
    split_lookup = load_split_lookup(Path(args.split_csv).resolve())

    existing_rows = load_rows(summary_dir / "epoch_gate_summary.csv")
    completed_epochs = {int(float(row["epoch"])) for row in existing_rows if row.get("epoch")}
    summary_by_epoch = {int(float(row["epoch"])): row for row in existing_rows if row.get("epoch")}

    for epoch, checkpoint_path in list_epoch_checkpoints(run_dir):
        if epoch in completed_epochs:
            continue
        print_step("eval", f"epoch={epoch} checkpoint={checkpoint_path.name}")
        row = evaluate_checkpoint(
            checkpoint_path,
            data_root=data_root,
            split_lookup=split_lookup,
            normal_class=args.normal_class,
            device=args.device,
            batch=args.batch,
            imgsz=args.imgsz,
            eps=args.eps,
            bins=args.bins,
            max_iter=args.max_iter,
            temp_dir=temp_dir,
        )
        summary_by_epoch[epoch] = row
        rows = [summary_by_epoch[key] for key in sorted(summary_by_epoch)]
        write_csv(summary_dir / "epoch_gate_summary.csv", SUMMARY_FIELDS, rows)
        write_json(
            summary_dir / "epoch_gate_summary.json",
            {
                "ranking_rule": [
                    "Spec@R99.5 descending",
                    "Spec@R99.0 descending",
                    "Prec@R99.0 descending",
                    "PTR@R99.0 ascending",
                ],
                "rows": rows,
            },
        )
        best_row = choose_best_row(rows)
        write_json(summary_dir / "best_epoch_manifest.json", best_row)
        (summary_dir / "epoch_gate_summary.md").write_text(build_md(rows, best_row), encoding="utf-8")
        plot_metric_dashboard(rows, summary_dir / "epoch_gate_dashboard")

    final_rows = load_rows(summary_dir / "epoch_gate_summary.csv")
    final_by_epoch = {int(float(row["epoch"])): row for row in final_rows if row.get("epoch")}
    index_rows = []
    for epoch, checkpoint_path in list_epoch_checkpoints(run_dir):
        summary_row = final_by_epoch.get(epoch)
        formal_path = formal_checkpoint_path(checkpoint_path)
        index_rows.append(
            {
                "epoch": epoch,
                "checkpoint_file": formal_path.name,
                "checkpoint_path": str(formal_path),
                "checkpoint_exists": int(checkpoint_path.exists()),
                "evaluated": int(summary_row is not None),
                "temperature_T": "" if summary_row is None else summary_row["temperature_T"],
                "tau_r995": "" if summary_row is None else summary_row["tau_r995"],
                "tau_r990": "" if summary_row is None else summary_row["tau_r990"],
                "spec_at_r995": "" if summary_row is None else summary_row["spec_at_r995"],
                "spec_at_r990": "" if summary_row is None else summary_row["spec_at_r990"],
                "prec_at_r990": "" if summary_row is None else summary_row["prec_at_r990"],
                "ptr_at_r990": "" if summary_row is None else summary_row["ptr_at_r990"],
            }
        )
    if index_rows:
        write_csv(summary_dir / "all_checkpoints_index.csv", INDEX_FIELDS, index_rows)
    print_step("done", f"wrote {summary_dir}")


if __name__ == "__main__":
    main()
