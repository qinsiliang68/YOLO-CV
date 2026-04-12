"""
Venue-standard figure rebuild v2.
Based on top-venue survey: heatmap for dense grids, grouped bar for small comparisons,
line+curve for Goldilocks, dot plot for ablation. No 4-panel identical bars.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def venue_rc():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
        "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.2, "xtick.direction": "out", "ytick.direction": "out",
    })

venue_rc()

PAL = {"n":"#377EB8","s":"#4DAF4A","m":"#E41A1C","l":"#FF7F00","x":"#984EA3"}
OUT = Path(r"C:\GitHub\YOLO-CV\research\results\stage1_formal")

# ════════════════════════════════════════════════════════════
# FIG 1: HN sweep heatmap — 4 models × 11 ratios, ΔSpec@R99.5
# ════════════════════════════════════════════════════════════
def build_hn_heatmap():
    ratios = [f"hn{r:02d}" for r in range(0,22,2)]
    models = ["yolo11n", "yolo11s", "yolo11m", "yolo11l"]

    baselines = {"yolo11n": 0.5119, "yolo11s": 0.5238, "yolo11m": 0.5952, "yolo11l": 0.5238}
    raw = {
        "yolo11n": [0.5119,0.5595,0.5238,0.5357,0.5833,0.5476,0.5476,0.5238,0.5476,0.5714,0.5595],
        "yolo11s": [0.5238,0.5119,0.5595,0.5000,0.5357,0.5833,0.5238,0.5357,0.5476,0.5476,0.5952],
        "yolo11m": [0.5952,0.6429,0.5595,0.5952,0.5833,0.5238,0.5357,0.6429,0.5714,0.5476,0.5595],
        "yolo11l": [np.nan,np.nan,np.nan,np.nan,0.5238,0.5238,0.5238,0.5119,0.5000,0.4762,0.5238],
    }

    delta = np.full((4, 11), np.nan)
    for i, m in enumerate(models):
        for j in range(11):
            if not np.isnan(raw[m][j]):
                delta[i, j] = raw[m][j] - baselines[m]

    fig, ax = plt.subplots(figsize=(12, 3.8))
    fig.subplots_adjust(left=0.12, right=0.92, top=0.88, bottom=0.15)

    vmax = 0.08
    im = ax.imshow(delta, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(4):
        for j in range(11):
            v = delta[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
            else:
                sign = "+" if v > 0 else ""
                color = "white" if abs(v) > 0.04 else "black"
                weight = "bold" if v == np.nanmax(delta[i, :]) and v > 0 else "normal"
                ax.text(j, i, f"{sign}{v:.3f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight=weight)

    ax.set_xticks(range(11))
    ax.set_xticklabels(ratios, fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel("HN ratio")
    ax.set_title("$\\Delta$Spec@R99.5 relative to hn00 baseline: four capacity tiers under shared protocol",
                 fontweight="bold", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("$\\Delta$Spec@R99.5", fontsize=10)

    out = OUT / "gate_hn_paper/figures/fig_hn_sweep_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[1] heatmap → {out}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════
# FIG 2: Capacity scan — grouped bar (gate + cls6 dual-row)
# ════════════════════════════════════════════════════════════
def build_capacity_grouped():
    models = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]
    colors = [PAL[m[-1]] for m in models]

    gate_995 = [0.5119, 0.5238, 0.5952, 0.5238, 0.5476]
    gate_990 = [0.5833, 0.5357, 0.6548, 0.5714, 0.5952]
    cls6_acc = [0.6833, 0.6917, 0.6917, 0.6833, 0.7000]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [1.2, 0.8]})
    fig.subplots_adjust(hspace=0.35, left=0.10, right=0.95, top=0.93, bottom=0.08)

    # ── Top: Gate view (grouped bar: Spec995 + Spec990) ──
    ax = axes[0]
    x = np.arange(5)
    w = 0.35
    b1 = ax.bar(x - w/2, gate_995, w, color=colors, edgecolor="black", linewidth=0.6,
                label="Spec@R99.5", alpha=0.9, zorder=2)
    b2 = ax.bar(x + w/2, gate_990, w, color=colors, edgecolor="black", linewidth=0.6,
                label="Spec@R99.0", alpha=0.5, hatch="//", zorder=2)

    for i in range(5):
        fw995 = "bold" if i == 2 else "normal"
        fw990 = "bold" if i == 2 else "normal"
        ax.text(x[i] - w/2, gate_995[i] + 0.005, f"{gate_995[i]:.4f}", ha="center",
                va="bottom", fontsize=8, fontweight=fw995)
        ax.text(x[i] + w/2, gate_990[i] + 0.005, f"{gate_990[i]:.4f}", ha="center",
                va="bottom", fontsize=8, fontweight=fw990)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Specificity")
    ax.set_title("(a) Binary-gate view: yolo11m ranks 1st (Spec@R99.5 = 0.5952)", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.2, zorder=0)
    ax.set_ylim(0.42, 0.72)

    # ── Bottom: CLS6 view (single bar: Acc) ──
    ax2 = axes[1]
    ax2.bar(x, cls6_acc, width=0.5, color=colors, edgecolor="black", linewidth=0.6, zorder=2)
    for i in range(5):
        fw = "bold" if i == 4 else "normal"
        ax2.text(i, cls6_acc[i] + 0.003, f"{cls6_acc[i]:.4f}", ha="center",
                 va="bottom", fontsize=9, fontweight=fw)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylabel("Accuracy")
    ax2.set_title("(b) Six-class source view: yolo11x ranks 1st (Acc = 0.7000) — cross-view misalignment",
                  fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.2, zorder=0)
    ax2.set_ylim(0.65, 0.72)

    out = OUT / "capacity_scan/paper_main/figures/fig_stage1_gate_capacity_bar.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[2] capacity grouped → {out}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════
# FIG 3: Lite ablation — horizontal dot plot (Cleveland style)
# ════════════════════════════════════════════════════════════
def build_lite_dotplot():
    settings = ["A0 (hn00 baseline)", "A1 (uniform hn14)", "A2 (R only)",
                "A3 (R+C)", "A4 (R+C+D)"]
    spec995 = [0.5952, 0.5952, 0.5595, 0.5595, 0.5714]
    baseline = spec995[1]  # A1

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.subplots_adjust(left=0.30, right=0.92, top=0.88, bottom=0.12)

    y = np.arange(len(settings))[::-1]
    colors = []
    for i, v in enumerate(spec995):
        if i <= 1:
            colors.append("#377EB8")
        elif v >= baseline:
            colors.append("#4DAF4A")
        else:
            colors.append("#E41A1C")

    ax.axvline(baseline, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, zorder=0)
    ax.text(baseline + 0.001, 4.4, f"A1 baseline\n({baseline:.4f})", fontsize=8, color="gray")

    for i in range(5):
        ax.plot([baseline, spec995[i]], [y[i], y[i]], color=colors[i],
                linewidth=1.5, alpha=0.5, zorder=1)

    ax.scatter(spec995, y, s=120, c=colors, edgecolors="black", linewidths=0.8, zorder=3)

    for i in range(5):
        delta = spec995[i] - baseline
        sign = "+" if delta > 0 else ""
        label = f"{spec995[i]:.4f} ({sign}{delta:.4f})"
        ax.text(spec995[i] + 0.003, y[i], label, va="center", fontsize=9,
                fontweight="bold" if abs(delta) > 0.001 else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels(settings, fontsize=10)
    ax.set_xlabel("Spec@R99.5")
    ax.set_title("Lite A0--A4: weighted variants (red) do not outperform uniform baseline (blue)",
                 fontweight="bold", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.2, zorder=0)
    ax.set_xlim(0.54, 0.62)

    out = OUT / "gate_info_sampling_lite/figures/fig_lite_ablation_main_barpanel.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[3] lite dotplot → {out}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════
# FIG 4: Goldilocks — line + fitted quadratic for each signal
# ════════════════════════════════════════════════════════════
def build_goldilocks_curves():
    G0 = 0.547619
    signals = {
        "R (boundary risk)": {
            "spec": [0.5238, 0.6190, 0.5119, 0.5357, 0.5833],
            "color": "#E41A1C", "peak": 1},
        "C (TTA consistency)": {
            "spec": [0.5595, 0.5595, 0.5357, 0.6071, 0.5833],
            "color": "#377EB8", "peak": 3},
        "D (kNN density)": {
            "spec": [0.5952, 0.5476, 0.6071, 0.5595, 0.5476],
            "color": "#4DAF4A", "peak": 2},
    }
    x = np.arange(5)
    x_smooth = np.linspace(0, 4, 50)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.07, right=0.97, top=0.85, bottom=0.15)

    for ax, (name, d) in zip(axes, signals.items()):
        spec = d["spec"]
        pk = d["peak"]
        color = d["color"]

        coeffs = np.polyfit(x, spec, 2)
        y_fit = np.polyval(coeffs, x_smooth)

        ax.fill_between(x_smooth, G0, y_fit, where=(y_fit > G0),
                        color=color, alpha=0.08, zorder=0)
        ax.plot(x_smooth, y_fit, color=color, linewidth=1.5, alpha=0.4, zorder=1)

        ax.axhline(G0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.text(4.1, G0, "G0", fontsize=8, color="gray", va="center")

        markers = ["o"] * 5
        sizes = [60] * 5
        sizes[pk] = 140
        for i in range(5):
            ec = "black" if i == pk else "gray"
            lw = 1.5 if i == pk else 0.5
            ax.scatter(x[i], spec[i], s=sizes[i], c=color if i == pk else "#CCCCCC",
                       edgecolors=ec, linewidths=lw, zorder=3)

        delta = spec[pk] - G0
        ax.annotate(f"Q{pk+1}: +{delta:.4f}\n(+{int(round(delta*84))} normals)",
                    xy=(pk, spec[pk]), xytext=(pk+0.6, spec[pk]+0.02),
                    fontsize=9, fontweight="bold", color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

        if spec[0] < G0:
            ax.annotate(f"Q1: {spec[0]-G0:+.4f}", xy=(0, spec[0]),
                        xytext=(0.5, spec[0]-0.02), fontsize=8, color="#999999",
                        arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))

        ax.set_xticks(x)
        ax.set_xticklabels(["Q1\n(extreme)", "Q2", "Q3", "Q4", "Q5\n(mild)"])
        ax.set_title(name, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.2, zorder=0)
        ax.set_ylim(0.47, 0.66)

    axes[0].set_ylabel("Spec@R99.5")
    fig.suptitle("Non-monotonic value--signal response (Goldilocks pattern): "
                 "peak performance at moderate quintiles, not extremes",
                 fontsize=11, y=0.97)

    out = OUT / "gate_bucket_pilot/fig_bucket_goldilocks_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[4] goldilocks curves → {out}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════
# FIG 5: Goldilocks attribution — lite π vs actual Spec
# ════════════════════════════════════════════════════════════
def build_goldilocks_attribution():
    import csv
    G0 = 0.547619
    R_spec = [0.5238, 0.6190, 0.5119, 0.5357, 0.5833]

    master_path = Path(r"C:\GitHub\YOLO-CV\research\materials\stage1_formal\gate_info_sampling_lite\score_inputs\candidate_pool_master.csv")
    rows = list(csv.DictReader(open(master_path)))
    R_vals = sorted([float(r["R"]) for r in rows], reverse=True)
    quintile_means = [np.mean(R_vals[i*50:(i+1)*50]) for i in range(5)]
    quintile_pi = np.array(quintile_means) ** 2
    quintile_pi = quintile_pi / quintile_pi.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(wspace=0.30, left=0.08, right=0.95, top=0.85, bottom=0.15)

    x = np.arange(5)
    labels = ["Q1\n(extreme)", "Q2", "Q3", "Q4", "Q5\n(mild)"]

    ax1.bar(x, quintile_pi, width=0.6, color="#E41A1C", alpha=0.7,
            edgecolor="black", linewidth=0.6)
    for i, v in enumerate(quintile_pi):
        ax1.text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Sampling probability $\\pi$")
    ax1.set_title("(a) Where lite A4 spends budget\n(concentrated on Q1)", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.2, zorder=0)

    colors = ["#E41A1C" if v < G0 else "#4DAF4A" for v in R_spec]
    ax2.bar(x, R_spec, width=0.6, color=colors, edgecolor="black", linewidth=0.6, zorder=2)
    ax2.axhline(G0, color="gray", linestyle="--", linewidth=1.2, alpha=0.6)
    ax2.text(4.3, G0, f"G0={G0:.4f}", fontsize=8, color="gray")
    for i, v in enumerate(R_spec):
        delta = v - G0
        sign = "+" if delta >= 0 else ""
        ax2.text(i, v + 0.005, f"{sign}{delta:.4f}", ha="center", fontsize=9,
                 fontweight="bold" if i == 1 else "normal",
                 color=colors[i])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Spec@R99.5")
    ax2.set_title("(b) Where value actually is\n(peak at Q2, Q1 below baseline)", fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.2, zorder=0)
    ax2.set_ylim(0.47, 0.66)

    fig.suptitle("Goldilocks attribution: lite spends budget on lowest-value quintile",
                 fontsize=12, y=0.98)

    out = OUT / "gate_bucket_pilot/fig_goldilocks_attribution.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"[5] attribution → {out}")
    plt.close(fig)


if __name__ == "__main__":
    build_hn_heatmap()
    build_capacity_grouped()
    build_lite_dotplot()
    build_goldilocks_curves()
    build_goldilocks_attribution()
    print("\nAll v2 venue figures rebuilt.")
