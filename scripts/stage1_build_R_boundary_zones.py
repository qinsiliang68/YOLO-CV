"""
R signal (boundary risk) sample distribution with annotated zones:
  - Outer skin (Q1): extreme boundary, noisy, LOW value
  - Subcutaneous muscle (Q2): moderate boundary, HIGH value
  - Deep tissue (Q3-Q5): easy samples, majority, moderate/low value

Shows 250 candidate samples sorted by R value, colored by zone,
with the Spec@R99.5 result annotated for each zone.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import csv
from pathlib import Path

def apply_venue_rcparams():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
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

# sort descending (rank 1 = highest R = most boundary)
order = np.argsort(R_vals)[::-1]
R_sorted = R_vals[order]
ranks = np.arange(1, 251)

# ── zone definitions ──
# Q1: rank 1-50  (outer skin)
# Q2: rank 51-100 (subcutaneous muscle)
# Q3: rank 101-150 (transition)
# Q4: rank 151-200 (easy)
# Q5: rank 201-250 (easiest)

zones = [
    (0, 50, "#E41A1C", 0.25, "Q1: outer skin\n(extreme boundary)\nSpec@R99.5 = 0.5238\n< G0 baseline"),
    (50, 100, "#FF7F00", 0.85, "Q2: subcutaneous muscle\n(moderate boundary)\nSpec@R99.5 = 0.6190\nBEST (+0.0714 vs G0)"),
    (100, 150, "#FFEDA0", 0.18, "Q3: transition zone\nSpec@R99.5 = 0.5119\nWORST"),
    (150, 200, "#A8DDB5", 0.18, "Q4: easy samples\nSpec@R99.5 = 0.5357"),
    (200, 250, "#377EB8", 0.25, "Q5: easiest (core)\nSpec@R99.5 = 0.5833"),
]

# ── figure: two panels ──
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                gridspec_kw={"height_ratios": [2.2, 1]},
                                sharex=False)
fig.subplots_adjust(hspace=0.30, left=0.10, right=0.88, top=0.92, bottom=0.08)

# ── Panel 1: R value distribution with zone shading ──
for start, end, color, alpha, label in zones:
    ax1.axvspan(start + 0.5, end + 0.5, color=color, alpha=alpha, zorder=0)
    ax1.scatter(ranks[start:end], R_sorted[start:end], s=18, c=color,
                edgecolors="black", linewidths=0.3, zorder=2, alpha=0.9)

# zone boundary lines
for b in [50, 100, 150, 200]:
    ax1.axvline(b + 0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

# zone labels on top
zone_labels = [
    (25, "OUTER SKIN\n(Q1: noisy boundary)", "#E41A1C"),
    (75, "MUSCLE\n(Q2: high-value)", "#CC6600"),
    (125, "Q3", "#888888"),
    (175, "Q4", "#888888"),
    (225, "CORE\n(Q5: easy)", "#377EB8"),
]
for xpos, text, color in zone_labels:
    ax1.text(xpos, R_sorted[0] * 1.05, text, ha="center", va="bottom",
             fontsize=9, fontweight="bold", color=color)

ax1.set_ylabel("R score (boundary risk)")
ax1.set_xlabel("Sample rank (sorted by R, descending)")
ax1.set_title("250 candidate samples: boundary risk distribution with value zones",
              fontweight="bold", fontsize=12)
ax1.grid(axis="y", linestyle="--", alpha=0.2, zorder=0)

# ── Panel 2: Spec@R99.5 bar chart by quintile ──
G0 = 0.547619
specs = [0.523810, 0.619048, 0.511905, 0.535714, 0.583333]
bar_colors = ["#E41A1C", "#FF7F00", "#FFEDA0", "#A8DDB5", "#377EB8"]
bar_labels = ["Q1\n(skin)", "Q2\n(muscle)", "Q3", "Q4", "Q5\n(core)"]

bars = ax2.bar(range(5), specs, width=0.6, color=bar_colors,
               edgecolor="black", linewidth=0.8, zorder=2)

ax2.axhline(G0, color="gray", linestyle="--", linewidth=1.2, alpha=0.6)
ax2.text(4.4, G0, f"G0 = {G0:.4f}", va="center", fontsize=8, color="gray")

# star on winner
ax2.plot(1, specs[1], marker="*", markersize=20, color="#FF7F00",
         markeredgecolor="black", markeredgewidth=1.2, zorder=3)

# delta annotations
for i, s in enumerate(specs):
    delta = s - G0
    sign = "+" if delta >= 0 else ""
    color = "black"
    if i == 1:
        color = "#CC6600"
    ax2.text(i, s + 0.005, f"{sign}{delta:.4f}", ha="center", va="bottom",
             fontsize=9, fontweight="bold" if i == 1 else "normal", color=color)

ax2.set_xticks(range(5))
ax2.set_xticklabels(bar_labels)
ax2.set_ylabel("Spec@R99.5")
ax2.set_ylim(0.44, 0.68)
ax2.set_title("Gate performance by R-quintile bucket (higher is better)", fontsize=11)
ax2.grid(axis="y", linestyle="--", alpha=0.2, zorder=0)

out = Path(r"C:\GitHub\YOLO-CV\research\results\stage1_formal\gate_bucket_pilot\fig_R_boundary_zones.png")
fig.savefig(out, bbox_inches="tight")
print(f"saved -> {out}")

# also copy to desktop for preview
import shutil
desktop = Path(r"C:\Users\28898\Desktop\fig_R_boundary_zones.png")
shutil.copy2(out, desktop)
print(f"copied -> {desktop}")
