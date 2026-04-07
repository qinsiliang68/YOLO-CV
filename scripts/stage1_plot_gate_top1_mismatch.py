from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a readable mismatch figure between trainer-selected and gate-selected checkpoints."
    )
    parser.add_argument(
        "--m-csv",
        default="research/results/stage1_formal/capacity_scan/appendix/tables/yolo11m_gate_epoch_summary.csv",
        help="Epoch summary CSV for yolo11m-cls.",
    )
    parser.add_argument(
        "--l-csv",
        default="research/results/stage1_formal/capacity_scan/appendix/tables/yolo11l_gate_epoch_summary.csv",
        help="Epoch summary CSV for yolo11l-cls.",
    )
    parser.add_argument(
        "--output",
        default="research/results/stage1_formal/capacity_scan/paper_main/figures/fig_stage1_gate_top1_vs_spec_dualaxis.png",
        help="Output figure path.",
    )
    return parser.parse_args()


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"epoch", "Spec@R99.5", "top1_acc"}
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"Missing columns in {path}: {sorted(missing)}")
    return frame.sort_values("epoch").reset_index(drop=True)


def draw_panel(ax: plt.Axes, frame: pd.DataFrame, title: str) -> None:
    top1_ax = ax.twinx()
    spec_color = "#2f5b84"
    top1_color = "#ef7d88"

    gate_idx = frame["Spec@R99.5"].idxmax()
    top1_idx = frame["top1_acc"].idxmax()
    gate_row = frame.loc[gate_idx]
    top1_row = frame.loc[top1_idx]

    ax.plot(frame["epoch"], frame["Spec@R99.5"], color=spec_color, linewidth=2.0, label="Spec@R99.5")
    top1_ax.plot(frame["epoch"], frame["top1_acc"], color=top1_color, linewidth=1.8, label="Top1 accuracy")

    ax.axvline(gate_row["epoch"], color=spec_color, linestyle="--", linewidth=1.4, alpha=0.9)
    top1_ax.axvline(top1_row["epoch"], color=top1_color, linestyle=":", linewidth=1.6, alpha=0.95)

    ax.scatter(
        [gate_row["epoch"]],
        [gate_row["Spec@R99.5"]],
        s=70,
        color=spec_color,
        zorder=5,
    )
    top1_ax.scatter(
        [top1_row["epoch"]],
        [top1_row["top1_acc"]],
        s=76,
        color=top1_color,
        marker="D",
        zorder=6,
    )

    ax.annotate(
        f"gate-best: ep{int(gate_row['epoch'])}",
        xy=(gate_row["epoch"], gate_row["Spec@R99.5"]),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=9,
        color=spec_color,
    )
    top1_ax.annotate(
        f"top1-best: ep{int(top1_row['epoch'])}",
        xy=(top1_row["epoch"], top1_row["top1_acc"]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        color=top1_color,
    )

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Spec@R99.5", fontsize=11)
    top1_ax.set_ylabel("Top1 Accuracy", fontsize=11)
    ax.grid(True, alpha=0.25)

    ax.set_xlim(1, int(frame["epoch"].max()))
    ax.tick_params(labelsize=10)
    top1_ax.tick_params(labelsize=10)

    handles = [
        plt.Line2D([0], [0], color=spec_color, linewidth=2.0, label="Spec@R99.5"),
        plt.Line2D([0], [0], color=top1_color, linewidth=1.8, label="Top1 accuracy"),
        plt.Line2D([0], [0], color=spec_color, linestyle="--", linewidth=1.4, label="gate-best epoch"),
        plt.Line2D([0], [0], color=top1_color, linestyle=":", linewidth=1.6, label="top1-best epoch"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, frameon=True)


def main() -> None:
    args = parse_args()
    m_frame = load_frame(Path(args.m_csv))
    l_frame = load_frame(Path(args.l_csv))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)

    draw_panel(axes[0], m_frame, "yolo11m-cls")
    draw_panel(axes[1], l_frame, "yolo11l-cls")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
