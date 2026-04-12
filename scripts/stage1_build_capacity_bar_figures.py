"""Regenerate the two capacity-scan bar figures used by essay2.tex chapter 4
in venue (NeurIPS / ICML / CVPR) style. Replaces the legacy outputs whose
panel titles contained unsupported unicode arrow characters that fell back
to "?" glyphs in the rendered PNGs.

Outputs:
    research/results/stage1_formal/capacity_scan/paper_main/figures/
        fig_stage1_gate_capacity_bar.png
        fig_stage1_cls6_capacity_bar.png

Usage:
    uv run python scripts/stage1_build_capacity_bar_figures.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_CSV = REPO_ROOT / "research/results/stage1_formal/gate_capacity/binary_gate_capacity_summary.csv"
FIG_DIR = REPO_ROOT / "research/results/stage1_formal/capacity_scan/paper_main/figures"

# Display order across the capacity gradient (small -> large).
MODEL_ORDER: List[str] = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]

# ColorBrewer Set1 high-contrast palette aligned with the rest of the
# essay's figures (m on red so the main model draws the eye).
MODEL_COLOR: Dict[str, str] = {
    "yolo11n": "#377EB8",  # blue
    "yolo11s": "#4DAF4A",  # green
    "yolo11m": "#E41A1C",  # red (main model)
    "yolo11l": "#FF7F00",  # orange
    "yolo11x": "#984EA3",  # purple
}

# Six-class source-side capacity scan (frozen reference values, kept in
# sync with table 4.4 in essay2.tex). Stored here because the cls6 path
# does not currently emit a single roll-up CSV.
CLS6_ROWS: List[Dict[str, float]] = [
    {"model": "yolo11n", "best_epoch": 117, "accuracy": 0.6833, "auroc": 0.9421, "auprc": 0.9870},
    {"model": "yolo11s", "best_epoch": 167, "accuracy": 0.6917, "auroc": 0.9406, "auprc": 0.9854},
    {"model": "yolo11m", "best_epoch": 130, "accuracy": 0.6958, "auroc": 0.9387, "auprc": 0.9848},
    {"model": "yolo11l", "best_epoch": 104, "accuracy": 0.6944, "auroc": 0.9400, "auprc": 0.9862},
    {"model": "yolo11x", "best_epoch": 161, "accuracy": 0.7000, "auroc": 0.9484, "auprc": 0.9885},
]


def apply_venue_rcparams() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def short_name(full: str) -> str:
    """yolo11m-cls -> yolo11m, leave yolo11m unchanged."""
    return full.replace("-cls", "")


def load_gate_summary() -> pd.DataFrame:
    df = pd.read_csv(GATE_CSV)
    df["model_short"] = df["model"].apply(short_name)
    df = df.set_index("model_short").reindex(MODEL_ORDER).reset_index()
    return df


def annotate_bars(ax, bars, fmt: str = "{:.4f}", offset: float = 0.005) -> None:
    for rect in bars:
        height = rect.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
        )


def render_bar_panel(
    metrics: Sequence[Tuple[str, str, bool]],
    rows: List[Dict],
    output_path: Path,
    suptitle: str,
    figsize: Tuple[float, float] = (12.0, 7.5),
) -> None:
    """Render a 2x2 grid of bar charts.

    metrics: sequence of (column_name, display_name, higher_is_better) tuples.
    rows: list of dicts in MODEL_ORDER order, with one entry per model.
    """
    apply_venue_rcparams()

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()

    models = [row["model"] for row in rows]
    colors = [MODEL_COLOR[m] for m in models]
    x = list(range(len(models)))

    for ax, (col, display_name, higher_is_better) in zip(axes, metrics):
        values = [float(row[col]) for row in rows]
        bars = ax.bar(
            x,
            values,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        annotate_bars(ax, bars)

        direction = "(higher is better)" if higher_is_better else "(lower is better)"
        ax.set_title(f"{display_name}  {direction}", fontsize=12, pad=8, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0)
        ax.set_ylabel(display_name)
        ax.grid(axis="y", alpha=0.30, linewidth=0.7, linestyle="--", zorder=0)
        ax.set_axisbelow(True)
        # Tighten y-limits so the bar differences are visible without wasting
        # vertical space, while always anchoring at 0 for fair visual scale.
        y_min = 0.0
        y_max = max(values) * 1.13 if max(values) > 0 else 1.0
        ax.set_ylim(y_min, y_max)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.02)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ok] {output_path}")


def build_gate_bar() -> None:
    df = load_gate_summary()
    rows = [
        {
            "model": row["model_short"],
            "spec_at_r995": row["spec_at_r995"],
            "spec_at_r990": row["spec_at_r990"],
            "prec_at_r990": row["prec_at_r990"],
            "ptr_at_r990": row["ptr_at_r990"],
        }
        for _, row in df.iterrows()
    ]
    metrics = [
        ("spec_at_r995", "Spec@R99.5", True),
        ("spec_at_r990", "Spec@R99.0", True),
        ("prec_at_r990", "Prec@R99.0", True),
        ("ptr_at_r990", "PTR@R99.0", False),
    ]
    render_bar_panel(
        metrics,
        rows,
        FIG_DIR / "fig_stage1_gate_capacity_bar.png",
        suptitle="Stage-1 direct binary-gate formal capacity scan",
    )


def build_cls6_bar() -> None:
    metrics = [
        ("accuracy", "Accuracy", True),
        ("auroc", "AUROC", True),
        ("auprc", "AUPRC", True),
        ("best_epoch", "Best epoch", False),
    ]
    render_bar_panel(
        metrics,
        CLS6_ROWS,
        FIG_DIR / "fig_stage1_cls6_capacity_bar.png",
        suptitle="Stage-1 six-class source-domain capacity scan (auxiliary view)",
    )


def main() -> None:
    build_gate_bar()
    build_cls6_bar()


if __name__ == "__main__":
    main()
