"""Build a single 4-panel figure overlaying HN sweep curves from multiple capacity models.

Each panel plots one of the four gate metrics (Spec@R99.5 / Spec@R99.0 / Prec@R99.0 / PTR@R99.0)
against HN ratio (%). One curve per model is drawn, with the winner ratio highlighted by a
star marker on each curve. The script is data-source agnostic: pass any number of
hn_sweep_summary.csv files via --model.

Example:
    uv run python scripts/stage1_build_hn_capacity_curves_panel.py \
        --model n yolo11n-cls research/results/stage1_formal/gate_hn_n_sweep/hn_sweep_summary.csv \
        --model s yolo11s-cls research/results/stage1_formal/gate_hn_s_sweep/hn_sweep_summary.csv \
        --model m yolo11m-cls research/results/stage1_formal/gate_hn_m_sweep/hn_sweep_summary.csv \
        --output-png research/results/stage1_formal/gate_hn_paper/figures/fig_hn_capacity_curves_panel.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

METRIC_COLUMNS = {
    "spec_at_r995": "Spec@R99.5",
    "spec_at_r990": "Spec@R99.0",
    "prec_at_r990": "Prec@R99.0",
    "ptr_at_r990": "PTR@R99.0",
}

LOWER_IS_BETTER = {"ptr_at_r990"}

# ColorBrewer Set1 — designed for maximum pairwise hue contrast across 5
# qualitative classes. Used widely in NeurIPS / ICML / CVPR figures because
# every pair of colors is unambiguously distinguishable in print.
# m (the main model) is placed on red to draw attention on every panel.
CAPACITY_PALETTE = {
    "n": "#377EB8",  # blue       (Set1[1])
    "s": "#4DAF4A",  # green      (Set1[2])
    "m": "#E41A1C",  # red        (Set1[0]) -- main model
    "l": "#FF7F00",  # orange     (Set1[4])
    "x": "#984EA3",  # purple     (Set1[3])
}

# Per-model line widths. Main model gets a slightly thicker line.
MODEL_LINEWIDTH = {
    "n": 2.0,
    "s": 2.0,
    "m": 2.6,
    "l": 2.0,
    "x": 2.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay HN sweep curves from multiple models onto a single 4-panel figure.",
    )
    parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        metavar=("TAG", "LABEL", "SUMMARY_CSV"),
        required=True,
        help="Repeatable. Each occurrence is (model_tag, display_label, hn_sweep_summary.csv).",
    )
    parser.add_argument(
        "--output-png",
        required=True,
        help="Output PNG path. Parent dir will be created.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional output CSV path; defaults to PNG path with .csv extension.",
    )
    parser.add_argument(
        "--suptitle",
        default="HN ratio sweep across capacity",
        help="Figure suptitle.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[12.0, 8.0],
        metavar=("W", "H"),
        help="Figure size in inches.",
    )
    parser.add_argument(
        "--layout",
        choices=["four-panel", "insight"],
        default="insight",
        help="four-panel = legacy 4 metric grid; insight = main Spec@R99.5 + side best-epoch panel.",
    )
    parser.add_argument(
        "--main-baseline-tag",
        default="m",
        help="Model tag whose hn00 / winner are drawn as horizontal reference lines in insight layout.",
    )
    return parser.parse_args()


def load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["ratio_percent"] = df["ratio_percent"].astype(int)
    df["best_epoch"] = df["best_epoch"].astype(int)
    for col in METRIC_COLUMNS:
        df[col] = df[col].astype(float)
    return df.sort_values("ratio_percent").reset_index(drop=True)


def compute_rank(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.sort_values(
        by=["spec_at_r995", "spec_at_r990", "prec_at_r990", "ptr_at_r990", "ratio_percent"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    rank_map = {ratio_id: idx + 1 for idx, ratio_id in enumerate(ranked["ratio_id"].tolist())}
    df = df.copy()
    df["formal_rank"] = df["ratio_id"].map(rank_map).astype(int)
    return df


def pick_color(tag: str, fallback_index: int) -> str:
    if tag in CAPACITY_PALETTE:
        return CAPACITY_PALETTE[tag]
    cmap = plt.get_cmap("tab10")
    return cmap(fallback_index % 10)


def build_combined_csv(per_model: List[Tuple[str, str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for tag, label, df in per_model:
        winner_id = df.sort_values("formal_rank").iloc[0]["ratio_id"]
        for _, row in df.iterrows():
            rows.append(
                {
                    "model_tag": tag,
                    "model_label": label,
                    "ratio_id": row["ratio_id"],
                    "ratio_percent": int(row["ratio_percent"]),
                    "best_epoch": int(row["best_epoch"]),
                    "spec_at_r995": float(row["spec_at_r995"]),
                    "spec_at_r990": float(row["spec_at_r990"]),
                    "prec_at_r990": float(row["prec_at_r990"]),
                    "ptr_at_r990": float(row["ptr_at_r990"]),
                    "formal_rank": int(row["formal_rank"]),
                    "is_winner": row["ratio_id"] == winner_id,
                }
            )
    return pd.DataFrame(rows)


def apply_venue_rcparams() -> None:
    """Top-tier venue (NeurIPS/ICML/CVPR) figure style.

    - Sans-serif font, consistent across all elements.
    - Slightly larger axis label / tick / legend font sizes than mpl default.
    - Solid axis spines on left and bottom only.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 15,
        "axes.linewidth": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def render_insight_panel(
    per_model: List[Tuple[str, str, pd.DataFrame]],
    png_path: Path,
    suptitle: str,
    figsize: Tuple[float, float],
    main_baseline_tag: str,
) -> None:
    """Two-panel insight-driven layout.

    Left (main): Spec@R99.5 vs HN ratio for all models, with two horizontal
        reference lines drawn from the main_baseline_tag model:
            - main_baseline_tag's hn00  (= "main baseline")
            - main_baseline_tag's winner (= "global best")
        Each model's winner ratio is highlighted with a star and a text label.
    Right (side): Best-epoch vs HN ratio for the same models, exposing the
        small-model-overfits-at-epoch-1 phenomenon. Points where best_epoch=1
        are emphasized.
    """
    apply_venue_rcparams()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.85, 1.0]},
        constrained_layout=True,
    )

    main_ax, side_ax = axes[0], axes[1]
    common_xticks = sorted(set(per_model[0][2]["ratio_percent"].tolist()))

    # ----- Main panel: Spec@R99.5 vs ratio -----
    main_metric = "spec_at_r995"
    metric_label = METRIC_COLUMNS[main_metric]

    # Draw the two horizontal reference lines first (so data lines plot on top)
    base_df = next((df for tag, _, df in per_model if tag == main_baseline_tag), None)
    if base_df is not None:
        base_color = pick_color(main_baseline_tag, 0)
        global_best_row = base_df.sort_values("formal_rank").iloc[0]
        baseline_row = base_df.loc[base_df["ratio_id"] == "hn00"].iloc[0]
        global_best_val = float(global_best_row[main_metric])
        baseline_val = float(baseline_row[main_metric])

        main_ax.axhline(
            y=global_best_val,
            color=base_color,
            linestyle="--",
            linewidth=1.6,
            alpha=0.55,
            zorder=1,
        )
        main_ax.text(
            common_xticks[-1] + 0.4,
            global_best_val,
            f" global best\n m+{global_best_row['ratio_id']} ({global_best_val:.4f})",
            fontsize=10,
            color=base_color,
            verticalalignment="center",
            horizontalalignment="left",
        )
        main_ax.axhline(
            y=baseline_val,
            color=base_color,
            linestyle=":",
            linewidth=1.4,
            alpha=0.45,
            zorder=1,
        )
        main_ax.text(
            common_xticks[-1] + 0.4,
            baseline_val,
            f" main baseline\n m+hn00 ({baseline_val:.4f})",
            fontsize=10,
            color=base_color,
            verticalalignment="center",
            horizontalalignment="left",
        )

    for idx, (tag, label, df) in enumerate(per_model):
        color = pick_color(tag, idx)
        linewidth = MODEL_LINEWIDTH.get(tag, 2.0)
        x = df["ratio_percent"]
        y = df[main_metric]
        main_ax.plot(
            x,
            y,
            marker="o",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=linewidth,
            color=color,
            label=label,
            zorder=3,
        )
        winner = df.sort_values("formal_rank").iloc[0]
        main_ax.scatter(
            [winner["ratio_percent"]],
            [winner[main_metric]],
            facecolor=color,
            edgecolor="black",
            linewidth=1.4,
            s=360,
            marker="*",
            zorder=10,
        )
        # Inline ratio_id label next to each star
        main_ax.annotate(
            f"{winner['ratio_id']}",
            xy=(winner["ratio_percent"], winner[main_metric]),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color=color,
            zorder=11,
        )

    main_ax.set_title("(a)  Spec@R99.5 vs HN ratio  ($\\uparrow$ better)", fontsize=13, pad=8, loc="left")
    main_ax.set_xlabel("HN ratio (%)")
    main_ax.set_ylabel("Spec@R99.5")
    main_ax.set_xticks(common_xticks)
    main_ax.set_xlim(min(common_xticks) - 0.8, max(common_xticks) + 4.2)
    main_ax.grid(axis="y", alpha=0.30, linewidth=0.7, linestyle="--", zorder=0)
    main_ax.set_axisbelow(True)
    main_ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        fontsize=10,
        handlelength=2.0,
    )

    # ----- Side panel: best epoch vs ratio -----
    for idx, (tag, label, df) in enumerate(per_model):
        color = pick_color(tag, idx)
        linewidth = MODEL_LINEWIDTH.get(tag, 2.0)
        x = df["ratio_percent"]
        y = df["best_epoch"]
        side_ax.plot(
            x,
            y,
            marker="o",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=linewidth,
            color=color,
            label=label,
            zorder=3,
        )
        # Highlight best_epoch=1 points (overfit-at-first-epoch symptom)
        early_mask = df["best_epoch"] == 1
        if early_mask.any():
            side_ax.scatter(
                df.loc[early_mask, "ratio_percent"],
                df.loc[early_mask, "best_epoch"],
                facecolor=color,
                edgecolor="black",
                linewidth=1.2,
                s=140,
                marker="X",
                zorder=10,
            )

    side_ax.set_title("(b)  Best epoch vs HN ratio", fontsize=13, pad=8, loc="left")
    side_ax.set_xlabel("HN ratio (%)")
    side_ax.set_ylabel("Best epoch")
    side_ax.set_xticks(common_xticks[::2])  # less crowded
    side_ax.set_xlim(min(common_xticks) - 0.8, max(common_xticks) + 0.8)
    side_ax.set_yscale("symlog", linthresh=10)
    side_ax.grid(axis="y", alpha=0.30, linewidth=0.7, linestyle="--", zorder=0)
    side_ax.set_axisbelow(True)
    # Mark the y=1 line so the "overfit at epoch 1" baseline is obvious
    side_ax.axhline(y=1, color="#888888", linewidth=0.8, linestyle=":", alpha=0.7, zorder=1)
    side_ax.text(
        common_xticks[-1],
        1,
        " best_ep=1\n (overfit at\n  first epoch)",
        fontsize=9,
        color="#444444",
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.04)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_panel(per_model: List[Tuple[str, str, pd.DataFrame]], png_path: Path, suptitle: str, figsize: Tuple[float, float]) -> None:
    apply_venue_rcparams()

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()

    common_xticks = sorted(set(per_model[0][2]["ratio_percent"].tolist()))

    for ax, metric in zip(axes, METRIC_COLUMNS):
        for idx, (tag, label, df) in enumerate(per_model):
            color = pick_color(tag, idx)
            linewidth = MODEL_LINEWIDTH.get(tag, 2.0)
            x = df["ratio_percent"]
            y = df[metric]
            ax.plot(
                x,
                y,
                marker="o",
                markersize=6,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.8,
                linewidth=linewidth,
                color=color,
                label=label,
                zorder=3,
            )
            winner_row = df.sort_values("formal_rank").iloc[0]
            ax.scatter(
                [winner_row["ratio_percent"]],
                [winner_row[metric]],
                facecolor=color,
                edgecolor="black",
                linewidth=1.4,
                s=320,
                marker="*",
                zorder=10,
            )

        title_suffix = "  ($\\downarrow$ better)" if metric in LOWER_IS_BETTER else "  ($\\uparrow$ better)"
        ax.set_title(METRIC_COLUMNS[metric] + title_suffix, fontsize=13, pad=8)
        ax.set_xlabel("HN ratio (%)")
        ax.set_ylabel(METRIC_COLUMNS[metric])
        ax.set_xticks(common_xticks)
        ax.set_xlim(min(common_xticks) - 0.8, max(common_xticks) + 0.8)
        ax.grid(axis="y", alpha=0.30, linewidth=0.7, linestyle="--", zorder=0)
        ax.set_axisbelow(True)
        # Slight outer margin for breathing room
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#444444")

    # Legend at top-center, single row, frameless
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.04),
        frameon=False,
        fontsize=11,
        handlelength=2.4,
        columnspacing=2.2,
    )
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.10)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    per_model: List[Tuple[str, str, pd.DataFrame]] = []
    for tag, label, csv_path_str in args.model:
        csv_path = Path(csv_path_str)
        if not csv_path.exists():
            raise FileNotFoundError(f"summary csv not found: {csv_path}")
        df = compute_rank(load_summary(csv_path))
        per_model.append((tag, label, df))

    png_path = Path(args.output_png)
    csv_path = Path(args.output_csv) if args.output_csv else png_path.with_suffix(".csv")

    combined = build_combined_csv(per_model)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path, index=False)

    if args.layout == "insight":
        # Insight layout uses a wider, shorter canvas
        figsize = (14.0, 5.5) if tuple(args.figsize) == (12.0, 8.0) else tuple(args.figsize)
        render_insight_panel(
            per_model,
            png_path,
            args.suptitle,
            figsize,
            args.main_baseline_tag,
        )
    else:
        render_panel(per_model, png_path, args.suptitle, tuple(args.figsize))

    print(f"[ok] wrote panel csv: {csv_path}")
    print(f"[ok] wrote panel png: {png_path}")
    for tag, label, df in per_model:
        winner = df.sort_values("formal_rank").iloc[0]
        print(
            f"  - {label:<14} winner={winner['ratio_id']:<5} "
            f"spec995={winner['spec_at_r995']:.4f} best_ep={int(winner['best_epoch'])}"
        )


if __name__ == "__main__":
    main()
