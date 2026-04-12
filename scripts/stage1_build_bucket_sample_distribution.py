"""
250-sample R / C / D distribution with quintile bucket boundaries.
Highlights the Goldilocks zone (peak bucket) vs extreme buckets.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path

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

# ── load data ──
master = Path(r"C:\GitHub\YOLO-CV\research\materials\stage1_formal\gate_info_sampling_lite\score_inputs\candidate_pool_master.csv")
rows = list(csv.DictReader(open(master)))

R_vals = np.array([float(r["R"]) for r in rows])
C_vals = np.array([float(r["C"]) for r in rows])
D_vals = np.array([float(r["D"]) for r in rows])

def assign_quintiles(vals):
    order = np.argsort(vals)[::-1]  # descending: rank 1 = highest value
    q = np.zeros(len(vals), dtype=int)
    for i, idx in enumerate(order):
        q[idx] = i // 50  # 0=Q1, 1=Q2, ..., 4=Q5
    return q

R_q = assign_quintiles(R_vals)
C_q = assign_quintiles(C_vals)
D_q = assign_quintiles(D_vals)

# ── bucket pilot results (Spec@R99.5) ──
G0 = 0.547619
bucket_spec = {
    "R": [0.523810, 0.619048, 0.511905, 0.535714, 0.583333],
    "C": [0.559524, 0.559524, 0.535714, 0.607143, 0.583333],
    "D": [0.595238, 0.547619, 0.607143, 0.559524, 0.547619],
}
peak_bucket = {"R": 1, "C": 3, "D": 2}  # 0-indexed

# ── colors ──
q_colors_base = ["#D9D9D9"] * 5
signal_colors = {"R": "#E41A1C", "C": "#377EB8", "D": "#4DAF4A"}

fig, axes = plt.subplots(2, 3, figsize=(14, 9),
                         gridspec_kw={"height_ratios": [1.6, 1]})
fig.subplots_adjust(hspace=0.35, wspace=0.3, left=0.07, right=0.97, top=0.93, bottom=0.07)

signals = [
    ("R  (boundary risk)", R_vals, R_q, "R"),
    ("C  (TTA consistency)", C_vals, C_q, "C"),
    ("D  (kNN density)", D_vals, D_q, "D"),
]

for col, (title, vals, quints, key) in enumerate(signals):
    ax_top = axes[0, col]
    ax_bot = axes[1, col]
    pk = peak_bucket[key]
    color = signal_colors[key]

    # ── top: strip plot (jittered 1D) ──
    for qi in range(5):
        mask = quints == qi
        y = vals[mask]
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(y))
        c = color if qi == pk else "#AAAAAA"
        alpha = 0.9 if qi == pk else 0.4
        edgecolor = "black" if qi == pk else "none"
        ax_top.scatter(np.full_like(y, qi) + jitter, y, s=25, c=c,
                       alpha=alpha, edgecolors=edgecolor, linewidths=0.4, zorder=2)

    # bucket boundaries
    sorted_vals = np.sort(vals)[::-1]
    for b in [50, 100, 150, 200]:
        if b < len(sorted_vals):
            boundary = sorted_vals[b - 1]
            ax_top.axhline(boundary, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax_top.set_xticks(range(5))
    ax_top.set_xticklabels(["Q1\n(extreme)", "Q2", "Q3", "Q4", "Q5\n(mild)"])
    ax_top.set_ylabel(f"{key} score")
    ax_top.set_title(title, fontweight="bold")
    ax_top.grid(axis="y", linestyle="--", alpha=0.2, zorder=0)

    # highlight peak bucket label
    ax_top.annotate(f"Q{pk+1}\nGoldilocks",
                    xy=(pk, np.mean(vals[quints == pk])),
                    xytext=(pk + 0.5, np.max(vals) * 0.95),
                    fontsize=9, fontweight="bold", color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                    ha="center")

    # ── bottom: bar chart (Spec@R99.5 per bucket) ──
    specs = bucket_spec[key]
    bar_colors = ["#D9D9D9"] * 5
    bar_colors[pk] = color
    ax_bot.bar(range(5), specs, width=0.6, color=bar_colors,
               edgecolor="black", linewidth=0.8, zorder=2)
    ax_bot.axhline(G0, color="gray", linestyle="--", linewidth=1.2, alpha=0.6)
    ax_bot.text(4.35, G0, "G0", va="center", fontsize=8, color="gray")

    ax_bot.plot(pk, specs[pk], marker="*", markersize=16, color=color,
                markeredgecolor="black", markeredgewidth=1.0, zorder=3)

    delta = specs[pk] - G0
    ax_bot.annotate(f"+{delta:.4f}", xy=(pk, specs[pk]),
                    xytext=(pk, specs[pk] + 0.012),
                    ha="center", fontsize=9, fontweight="bold", color=color)

    ax_bot.set_xticks(range(5))
    ax_bot.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax_bot.set_ylabel("Spec@R99.5")
    ax_bot.set_ylim(0.44, 0.68)
    ax_bot.grid(axis="y", linestyle="--", alpha=0.2, zorder=0)

fig.suptitle("250 candidate samples: signal distribution (top) and gate performance by quintile (bottom)",
             fontsize=12, y=0.97)

out = Path(r"C:\GitHub\YOLO-CV\research\results\stage1_formal\gate_bucket_pilot\fig_bucket_sample_distribution.png")
fig.savefig(out, bbox_inches="tight")
print(f"saved -> {out}")
