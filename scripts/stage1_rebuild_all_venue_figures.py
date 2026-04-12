"""
Rebuild all key figures to venue standard (Automation in Construction / CVPR).
- ColorBrewer Set1 palette
- ★ winner markers with black edge
- Baseline reference lines
- No top/right spines, ticks outward
- DPI 300, pdf.fonttype=42
- Sans-serif (Arial/DejaVu Sans)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Venue rcParams ──
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

PALETTE = {
    "n": "#377EB8",
    "s": "#4DAF4A",
    "m": "#E41A1C",
    "l": "#FF7F00",
    "x": "#984EA3",
}
MODEL_ORDER = ["n", "s", "m", "l", "x"]
MODEL_LABELS = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]

OUT_DIR = Path(r"C:\GitHub\YOLO-CV\research\results\stage1_formal")


# ════════════════════════════════════════════════════════════════
# FIGURE 1: Capacity scan bar chart (gate view) — 2-panel layout
# ════════════════════════════════════════════════════════════════
def build_capacity_bar():
    spec995 = [0.5119, 0.5238, 0.5952, 0.5238, 0.5476]
    spec990 = [0.5833, 0.5357, 0.6548, 0.5714, 0.5952]
    winner_idx = 2  # m

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    fig.subplots_adjust(wspace=0.28, left=0.08, right=0.97, top=0.88, bottom=0.18)

    for ax, vals, metric in [(ax1, spec995, "Spec@R99.5"), (ax2, spec990, "Spec@R99.0")]:
        colors = [PALETTE[m] for m in MODEL_ORDER]
        bars = ax.bar(range(5), vals, width=0.6, color=colors,
                      edgecolor="black", linewidth=0.6, zorder=2)

        # winner star
        ax.plot(winner_idx, vals[winner_idx], marker="*", markersize=22,
                color=PALETTE["m"], markeredgecolor="black", markeredgewidth=1.2, zorder=3)

        # value labels
        for i, v in enumerate(vals):
            weight = "bold" if i == winner_idx else "normal"
            ax.text(i, v + 0.008, f"{v:.4f}", ha="center", va="bottom",
                    fontsize=9, fontweight=weight)

        ax.set_xticks(range(5))
        ax.set_xticklabels(MODEL_LABELS, fontsize=10)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}  (higher is better)", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
        ax.set_ylim(0.42, 0.72)

    fig.suptitle("Binary-gate capacity scan under ceteris paribus protocol",
                 fontsize=12, y=0.96)

    out = OUT_DIR / "capacity_scan/paper_main/figures/fig_stage1_gate_capacity_bar.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[1] saved → {out}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# FIGURE 2: HN capacity curves panel (4 models: n/s/m/l)
# ════════════════════════════════════════════════════════════════
def build_hn_capacity_curves():
    ratios = list(range(0, 22, 2))  # 0,2,...,20
    ratio_pct = [r for r in ratios]

    data = {
        "n": {"spec": [0.5119,0.5595,0.5238,0.5357,0.5833,0.5476,0.5476,0.5238,0.5476,0.5714,0.5595],
              "best_ep": [76,43,72,1,100,1,1,87,1,1,1],
              "winner": 4, "winner_ratio": "hn08"},
        "s": {"spec": [0.5238,0.5119,0.5595,0.5000,0.5357,0.5833,0.5238,0.5357,0.5476,0.5476,0.5952],
              "best_ep": [77,8,28,54,1,36,1,75,1,1,66],
              "winner": 10, "winner_ratio": "hn20"},
        "m": {"spec": [0.5952,0.6429,0.5595,0.5952,0.5833,0.5238,0.5357,0.6429,0.5714,0.5476,0.5595],
              "best_ep": [78,51,144,96,1,29,115,74,118,48,1],
              "winner": 7, "winner_ratio": "hn14"},
        "l": {"spec": [None,None,None,None,0.5238,0.5238,0.5238,0.5119,0.5000,0.4762,0.5238],
              "best_ep": [None,None,None,None,14,48,5,86,54,1,5],
              "winner": 4, "winner_ratio": "hn08"},
    }

    lw = {"n": 2.0, "s": 2.0, "m": 2.6, "l": 2.0}
    global_best = 0.6429
    m_baseline = 0.5952

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                    gridspec_kw={"width_ratios": [1.85, 1]})
    fig.subplots_adjust(wspace=0.22, left=0.07, right=0.97, top=0.88, bottom=0.13)

    # ── Panel (a): Spec@R99.5 vs ratio ──
    for model in ["n", "s", "m", "l"]:
        d = data[model]
        xs, ys = [], []
        for i, v in enumerate(d["spec"]):
            if v is not None:
                xs.append(ratio_pct[i])
                ys.append(v)
        ax1.plot(xs, ys, marker="o", markersize=5, color=PALETTE[model],
                 linewidth=lw[model], label=f"yolo11{model}-cls", zorder=2)
        # winner star
        wi = d["winner"]
        ax1.plot(ratio_pct[wi], d["spec"][wi], marker="*", markersize=20,
                 color=PALETTE[model], markeredgecolor="black",
                 markeredgewidth=1.2, zorder=3)
        ax1.annotate(d["winner_ratio"], xy=(ratio_pct[wi], d["spec"][wi]),
                     xytext=(ratio_pct[wi]+0.8, d["spec"][wi]+0.012),
                     fontsize=8, color=PALETTE[model], fontweight="bold")

    ax1.axhline(global_best, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.text(20.5, global_best, f"global best\nm+hn14 ({global_best})", fontsize=7,
             va="center", color="gray")
    ax1.axhline(m_baseline, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax1.text(20.5, m_baseline, f"m+hn00\n({m_baseline})", fontsize=7,
             va="center", color="gray")

    ax1.set_xlabel("HN ratio (%)")
    ax1.set_ylabel("Spec@R99.5")
    ax1.set_title("(a) Spec@R99.5 vs HN ratio  (higher is better)", fontweight="bold")
    ax1.set_xticks(ratio_pct)
    ax1.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax1.legend(loc="lower left", framealpha=0.9, edgecolor="#CCCCCC")
    ax1.set_ylim(0.44, 0.68)

    # ── Panel (b): Best epoch vs ratio ──
    for model in ["n", "s", "m", "l"]:
        d = data[model]
        xs, ys = [], []
        markers_x, markers_y = [], []
        for i, v in enumerate(d["best_ep"]):
            if v is not None:
                xs.append(ratio_pct[i])
                ys.append(v)
                if v == 1:
                    markers_x.append(ratio_pct[i])
                    markers_y.append(v)
        ax2.plot(xs, ys, marker="o", markersize=5, color=PALETTE[model],
                 linewidth=lw[model], label=f"yolo11{model}-cls", zorder=2)
        if markers_x:
            ax2.scatter(markers_x, markers_y, marker="X", s=100,
                       color=PALETTE[model], edgecolors="black",
                       linewidths=0.8, zorder=3)

    ax2.set_xlabel("HN ratio (%)")
    ax2.set_ylabel("Best epoch")
    ax2.set_title("(b) Best epoch vs HN ratio", fontweight="bold")
    ax2.set_xticks(ratio_pct)
    ax2.set_yscale("log")
    ax2.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax2.text(20.5, 1.3, "best_ep=1\n(X marks)", fontsize=7, color="gray")

    out = OUT_DIR / "gate_hn_paper/figures/fig_hn_capacity_curves_panel.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[2] saved → {out}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# FIGURE 3: Lite ablation bar — 2-panel (Spec995 + Spec990)
# ════════════════════════════════════════════════════════════════
def build_lite_ablation_bar():
    settings = ["A0\nhn00", "A1\nuniform\nhn14", "A2\nR only", "A3\nR+C", "A4\nR+C+D"]
    spec995 = [0.5952, 0.5952, 0.5595, 0.5595, 0.5714]
    spec990 = [0.6548, 0.6190, 0.5952, 0.5833, 0.6071]
    baseline_idx = 1  # A1 uniform_hn14

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(wspace=0.28, left=0.08, right=0.97, top=0.88, bottom=0.22)

    for ax, vals, metric in [(ax1, spec995, "Spec@R99.5"), (ax2, spec990, "Spec@R99.0")]:
        colors = []
        for i in range(5):
            if i <= 1:
                colors.append("#377EB8")  # baselines
            else:
                colors.append("#E41A1C")  # weighted variants
        bars = ax.bar(range(5), vals, width=0.6, color=colors,
                      edgecolor="black", linewidth=0.6, zorder=2)

        # baseline reference
        ax.axhline(vals[baseline_idx], color="gray", linestyle="--",
                   linewidth=1.0, alpha=0.6)
        ax.text(4.4, vals[baseline_idx], f"A1={vals[baseline_idx]:.4f}",
                fontsize=7, color="gray", va="center")

        for i, v in enumerate(vals):
            delta = v - vals[baseline_idx]
            sign = "+" if delta >= 0 else ""
            label = f"{v:.4f}\n({sign}{delta:.4f})"
            ax.text(i, v + 0.005, label, ha="center", va="bottom", fontsize=8)

        ax.set_xticks(range(5))
        ax.set_xticklabels(settings, fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}  (higher is better)", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
        ax.set_ylim(0.50, 0.70)

    fig.suptitle("Lite ablation A0--A4: weighted variants (red) vs uniform baselines (blue)",
                 fontsize=11, y=0.96)

    out = OUT_DIR / "gate_info_sampling_lite/figures/fig_lite_ablation_main_barpanel.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[3] saved → {out}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# FIGURE 4: HN gain decay curve (NEW — cross-capacity insight)
# ════════════════════════════════════════════════════════════════
def build_hn_gain_decay():
    params = [5.4, 10, 25, 24]  # M params (l=24M, but order by capacity tier)
    delta = [0.0714, 0.0714, 0.0476, 0.0]
    models = ["n (5.4M)", "s (10M)", "m (25M)", "l (24M)"]
    colors = [PALETTE["n"], PALETTE["s"], PALETTE["m"], PALETTE["l"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.bar(range(4), delta, width=0.55, color=colors,
           edgecolor="black", linewidth=0.8, zorder=2)

    for i, (d, m) in enumerate(zip(delta, models)):
        label = f"+{d:.4f}" if d > 0 else "0"
        ax.text(i, d + 0.003, label, ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("$\\Delta$Spec@R99.5 (best ratio vs hn00)")
    ax.set_title("HN gain diminishes to zero with increasing capacity\n"
                 "(under shared training protocol)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax.set_ylim(-0.01, 0.09)

    out = OUT_DIR / "gate_hn_paper/figures/fig_hn_gain_decay_bar.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[4] saved → {out}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# RUN ALL
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    build_capacity_bar()
    build_hn_capacity_curves()
    build_lite_ablation_bar()
    build_hn_gain_decay()
    print("\nAll venue figures rebuilt.")
