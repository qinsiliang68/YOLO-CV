from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from collect_cls_raw_materials import (
    DEFAULT_THRESHOLDS,
    build_threshold_summary,
    collect_validation_predictions,
    compute_binary_threshold_rows,
    write_csv,
    write_json,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


SUMMARY_FIELDS = ["epoch", "checkpoint_file", "checkpoint_path", "accuracy", "auroc", "auprc"]
INDEX_FIELDS = ["epoch", "checkpoint_file", "checkpoint_path", "checkpoint_exists", "evaluated", "accuracy", "auroc", "auprc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate formal cls6 checkpoints.")
    parser.add_argument("--run-dir", required=True, help="Training run directory.")
    parser.add_argument("--data-root", required=True, help="Cls6 dataset root.")
    parser.add_argument("--summary-dir", required=True, help="Formal material output directory.")
    parser.add_argument("--normal-class", default="Normal", help="Normal class name.")
    parser.add_argument("--device", default="0", help="Inference device.")
    parser.add_argument("--batch", type=int, default=24, help="Evaluation batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    return parser.parse_args()


def print_step(name: str, detail: str) -> None:
    print(f"[{name}] {detail}")


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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
    def key_fn(row: dict[str, Any]) -> tuple[float, float, float]:
        return (float(row["accuracy"]), float(row["auroc"]), float(row["auprc"]))

    return max(rows, key=key_fn)


def build_md(rows: list[dict[str, Any]], best_row: dict[str, Any]) -> str:
    lines = [
        "# Formal Stage-1 Cls6 Epoch Summary",
        "",
        "| Epoch | Accuracy | AUROC | AUPRC |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append("| {epoch} | {accuracy} | {auroc} | {auprc} |".format(**row))
    lines.extend(
        [
            "",
            f"- Best epoch: `{best_row['epoch']}`",
            f"- Best checkpoint: `{best_row['checkpoint_path']}`",
            "- Ranking rule: `Accuracy -> AUROC -> AUPRC`",
        ]
    )
    return "\n".join(lines) + "\n"


def plot_metric_dashboard(rows: list[dict[str, Any]], output_prefix: Path) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    metrics = [("accuracy", "Accuracy", "#355C7D"), ("auroc", "AUROC", "#6C5B7B"), ("auprc", "AUPRC", "#F67280")]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), sharex=True)
    for ax, (field, title, color) in zip(axes, metrics, strict=True):
        values = [float(row[field]) for row in rows]
        ax.plot(epochs, values, color=color, linewidth=1.6, marker="o", markersize=3.0)
        ax.set_title(title)
        ax.grid(alpha=0.2)
    for ax in axes:
        ax.set_xlabel("Epoch")
    fig.suptitle("Formal cls6 checkpoint scan", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    data_root: Path,
    normal_class: str,
    device: str,
    batch: int,
    imgsz: int,
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
    accuracy = sum(int(row["correct"]) for row in prediction_rows) / max(len(prediction_rows), 1)
    scored = [
        {
            "abnormal_conf": float(row["abnormal_conf"] or row["p_abnormal"] or 0.0),
            "gt_label": row["gt_label"],
            "is_abnormal": row["gt_label"] != normal_class,
        }
        for row in prediction_rows
    ]
    threshold_rows = compute_binary_threshold_rows(scored, DEFAULT_THRESHOLDS)
    threshold_summary = build_threshold_summary(threshold_rows, scored, normal_class)
    return {
        "epoch": checkpoint_epoch(checkpoint_path),
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_path": str(formal_checkpoint_path(checkpoint_path)),
        "accuracy": round(float(accuracy), 6),
        "auroc": round(float(threshold_summary["auroc_exact"]), 6),
        "auprc": round(float(threshold_summary["average_precision_exact"]), 6),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    data_root = Path(args.data_root).resolve()
    summary_dir = Path(args.summary_dir).resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)

    existing_rows = load_rows(summary_dir / "epoch_cls6_summary.csv")
    completed_epochs = {int(float(row["epoch"])) for row in existing_rows if row.get("epoch")}
    summary_by_epoch = {int(float(row["epoch"])): row for row in existing_rows if row.get("epoch")}

    for epoch, checkpoint_path in list_epoch_checkpoints(run_dir):
        if epoch in completed_epochs:
            continue
        print_step("eval", f"epoch={epoch} checkpoint={checkpoint_path.name}")
        row = evaluate_checkpoint(
            checkpoint_path,
            data_root=data_root,
            normal_class=args.normal_class,
            device=args.device,
            batch=args.batch,
            imgsz=args.imgsz,
        )
        summary_by_epoch[epoch] = row
        rows = [summary_by_epoch[key] for key in sorted(summary_by_epoch)]
        write_csv(summary_dir / "epoch_cls6_summary.csv", SUMMARY_FIELDS, rows)
        write_json(
            summary_dir / "epoch_cls6_summary.json",
            {
                "ranking_rule": ["Accuracy descending", "AUROC descending", "AUPRC descending"],
                "rows": rows,
            },
        )
        best_row = choose_best_row(rows)
        write_json(summary_dir / "best_epoch_manifest.json", best_row)
        (summary_dir / "epoch_cls6_summary.md").write_text(build_md(rows, best_row), encoding="utf-8")
        plot_metric_dashboard(rows, summary_dir / "epoch_cls6_dashboard")

    final_rows = load_rows(summary_dir / "epoch_cls6_summary.csv")
    final_by_epoch = {int(float(row["epoch"])): row for row in final_rows if row.get("epoch")}
    index_rows = []
    for epoch, checkpoint_path in list_epoch_checkpoints(run_dir):
        summary_row = final_by_epoch.get(epoch)
        index_rows.append(
            {
                "epoch": epoch,
                "checkpoint_file": checkpoint_path.name,
                "checkpoint_path": str(formal_checkpoint_path(checkpoint_path)),
                "checkpoint_exists": int(checkpoint_path.exists()),
                "evaluated": int(summary_row is not None),
                "accuracy": "" if summary_row is None else summary_row["accuracy"],
                "auroc": "" if summary_row is None else summary_row["auroc"],
                "auprc": "" if summary_row is None else summary_row["auprc"],
            }
        )
    if index_rows:
        write_csv(summary_dir / "all_checkpoints_index.csv", INDEX_FIELDS, index_rows)
    print_step("done", f"wrote {summary_dir}")


if __name__ == "__main__":
    main()
