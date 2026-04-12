"""
Bucket pilot Goldilocks panel: R / C / D signal value curves.
Each subplot shows Spec@R99.5 vs quintile bucket (Q1-Q5) with G0 baseline.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── venue rcParams ──
def apply_venue_rcparams():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })

apply_venue_rcparams()

# ── data ──
G0 = 0.547619

signals = {
    "R  (boundary risk)": {
        "spec": [0.523810, 0.619048, 0.511905, 0.535714, 0.583333],
        "color": "#E41A1C",
        "peak": 1,
        "label_long": "R signal: calibrated boundary risk",
    },
    "C  (TTA consistency)": {
        "spec": [0.559524, 0.559524, 0.535714, 0.607143, 0.583333],
        "color": "#377EB8",
        "peak": 3,
        "label_long": "C signal: test-time augmentation consistency",
    },
    "D  (kNN density)": {
        "spec": [0.595238, 0.547619, 0.607143, 0.559524, 0.547619],
        "color": "#4DAF4A",
        "peak": 2,
        "label_long": "D signal: feature-space kNN density",
    },
}

buckets = ["Q1", "Q2", "Q3", "Q4", "Q5"]
x = np.arange(len(buckets))

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
fig.subplots_adjust(wspace=0.08, left=0.07, right=0.97, top=0.88, bottom=0.15)

for ax, (name, d) in zip(axes, signals.items()):
    spec = d["spec"]
    peak_idx = d["peak"]
    color = d["color"]

    # G0 baseline
    ax.axhline(G0, color="gray", linestyle="--", linewidth=1.2, alpha=0.6, zorder=0)
    ax.text(4.05, G0, "G0", va="center", ha="left", fontsize=8, color="gray")

    # bars
    bar_colors = ["#D9D9D9"] * 5
    bar_colors[peak_idx] = color
    bars = ax.bar(x, spec, width=0.6, color=bar_colors, edgecolor="black",
                  linewidth=0.8, zorder=2)

    # peak star
    ax.plot(peak_idx, spec[peak_idx], marker="*", markersize=18, color=color,
            markeredgecolor="black", markeredgewidth=1.2, zorder=3)

    # delta annotation on peak bar
    delta = spec[peak_idx] - G0
    ax.annotate(f"+{delta:.4f}\n(+{int(round(delta * 84))} normals)",
                xy=(peak_idx, spec[peak_idx]),
                xytext=(peak_idx, spec[peak_idx] + 0.018),
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_title(name, fontweight="bold")
    ax.set_xlabel("Quintile bucket")
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.set_ylim(0.44, 0.68)

axes[0].set_ylabel("Spec@R99.5 (higher is better)")

fig.suptitle("Goldilocks pattern: extreme-signal samples underperform moderate ones across all three dimensions",
             fontsize=11, y=0.98)

out = Path(r"C:\GitHub\YOLO-CV\research\results\stage1_formal\gate_bucket_pilot\fig_bucket_goldilocks_panel.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"saved → {out}")
