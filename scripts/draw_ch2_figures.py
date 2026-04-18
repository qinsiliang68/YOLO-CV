"""Draw Ch2 technical theory figures for essay3."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from pathlib import Path

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.1,
})

OUT = Path(r"C:\GitHub\YOLO-CV\essay\img")
OUT.mkdir(parents=True, exist_ok=True)

# Set1-inspired palette
C_BLUE   = "#377EB8"
C_RED    = "#E41A1C"
C_GREEN  = "#4DAF4A"
C_ORANGE = "#FF7F00"
C_PURPLE = "#984EA3"
C_GREY   = "#999999"


# =====================================================
# FIG 1: Reliability diagram before/after T-scaling
# =====================================================
def fig_reliability():
    np.random.seed(0)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0))

    def plot_panel(ax, calibrated, title):
        bins = np.linspace(0, 1, 11)
        centers = (bins[:-1] + bins[1:]) / 2
        if calibrated:
            acc = centers + np.random.normal(0, 0.02, len(centers))
            acc = np.clip(acc, 0, 1)
        else:
            # overconfident: when conf is high, acc is lower
            acc = np.where(centers < 0.5,
                           centers - 0.05,
                           centers - 0.12 * (centers - 0.5) * 2)
            acc += np.random.normal(0, 0.015, len(centers))
            acc = np.clip(acc, 0, 1)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.6, label="perfect")
        ax.bar(centers, acc, width=0.09, color=C_BLUE, alpha=0.7,
               edgecolor="black", linewidth=0.6, label="empirical accuracy")
        gap = centers - acc
        ax.bar(centers, gap, width=0.09, bottom=acc, color=C_RED, alpha=0.5,
               edgecolor="none", label="gap")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("confidence  $\\hat{p}$")
        ax.set_ylabel("empirical accuracy")
        ax.set_title(title)
        ax.legend(loc="upper left", framealpha=0.9)
        ax.set_aspect("equal")

    plot_panel(axes[0], False, "(a) before temperature scaling (overconfident)")
    plot_panel(axes[1], True, "(b) after temperature scaling (calibrated)")

    fig.tight_layout()
    path = OUT / "ch2_reliability_diagram.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


# =====================================================
# FIG 2: Threshold search under recall constraint
# =====================================================
def fig_threshold_search():
    np.random.seed(1)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    # simulate two distributions
    x = np.linspace(0, 1, 500)
    # normal class: low p_abnormal (most mass near 0)
    pdf_normal = (1 / (0.15 * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-0.22)/0.15)**2)
    # abnormal class: high p_abnormal (most mass near 1)
    pdf_abnormal = (1 / (0.18 * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-0.72)/0.18)**2)

    ax.fill_between(x, 0, pdf_normal, color=C_BLUE, alpha=0.35, label="$p_{\\rm cal}(x\\mid y=0)$ (normal)")
    ax.fill_between(x, 0, pdf_abnormal, color=C_RED, alpha=0.35, label="$p_{\\rm cal}(x\\mid y=1)$ (abnormal)")
    ax.plot(x, pdf_normal, color=C_BLUE, linewidth=1.8)
    ax.plot(x, pdf_abnormal, color=C_RED, linewidth=1.8)

    tau = 0.40
    ax.axvline(tau, color="black", linestyle="--", linewidth=1.6, zorder=5)
    ax.annotate(r"$\tau_{R_0}$",
                xy=(tau, 2.1), xytext=(tau+0.05, 2.3),
                fontsize=12, fontweight="bold")

    # shaded regions
    mask_spec = x < tau
    ax.fill_between(x[mask_spec], 0, pdf_normal[mask_spec], color=C_BLUE, alpha=0.6)
    mask_recall = x >= tau
    ax.fill_between(x[mask_recall], 0, pdf_abnormal[mask_recall], color=C_RED, alpha=0.6)

    ax.text(0.18, 1.2, "Spec@R_0\n(TN)", ha="center", fontsize=10, color=C_BLUE,
            fontweight="bold")
    ax.text(0.82, 1.0, "Recall $\\geq R_0$\n(TP)", ha="center", fontsize=10, color=C_RED,
            fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3.3)
    ax.set_xlabel("calibrated probability  $p_{\\rm cal}(x)$")
    ax.set_ylabel("density")
    ax.set_title("Threshold search under recall constraint")
    ax.legend(loc="upper center", framealpha=0.9)

    fig.tight_layout()
    path = OUT / "ch2_threshold_search.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


# =====================================================
# FIG 3: Goldilocks non-monotonic curve
# =====================================================
def fig_goldilocks():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    rank = np.linspace(0, 100, 200)
    # inverted-U (Goldilocks)
    value = 0.55 + 0.08 * np.exp(-((rank-25)/15)**2) - 0.04 * (rank/100)

    ax.plot(rank, value, color=C_PURPLE, linewidth=2.4)
    ax.fill_between(rank, 0.48, value, color=C_PURPLE, alpha=0.12)

    # regions
    ax.axvspan(0, 10, alpha=0.15, color=C_RED, label="extreme hard\n(noisy / mislabeled)")
    ax.axvspan(10, 50, alpha=0.15, color=C_GREEN, label="Goldilocks zone\n(moderate difficulty)")
    ax.axvspan(50, 100, alpha=0.15, color=C_BLUE, label="easy\n(low marginal gain)")

    # peak marker
    peak_idx = np.argmax(value)
    ax.plot(rank[peak_idx], value[peak_idx], marker="*", markersize=22,
            color="gold", markeredgecolor="black", markeredgewidth=1.0, zorder=5)
    ax.annotate("peak value\n$\\approx$ rank 20--30",
                xy=(rank[peak_idx], value[peak_idx]),
                xytext=(45, value[peak_idx]+0.02),
                fontsize=10,
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0))

    ax.set_xlim(0, 100)
    ax.set_ylim(0.48, 0.66)
    ax.set_xlabel("sample rank percentile by signal value (%)")
    ax.set_ylabel("training value (downstream Spec@R$_0$)")
    ax.set_title("Goldilocks effect: non-monotonic value--signal response")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.15)

    fig.tight_layout()
    path = OUT / "ch2_goldilocks_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


# =====================================================
# FIG 4: Wilson CI half-width vs sample size
# =====================================================
def fig_wilson_ci():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    n = np.logspace(np.log10(50), np.log10(30000), 200)
    p = 0.50  # worst-case (maximum variance) for illustration
    # Wilson approximation: half-width ≈ 1.96 sqrt(p(1-p)/n) for moderate n
    hw = 1.96 * np.sqrt(p*(1-p)/n) * 100  # in pp

    ax.plot(n, hw, color=C_BLUE, linewidth=2.2)
    ax.fill_between(n, 0, hw, color=C_BLUE, alpha=0.12)

    # mark typical points (with custom offsets to avoid overlap)
    points = [
        (84,    "$n=84$  (older protocols)",   (0.35, 2.5)),
        (707,   "$n=707$  (this work val)",    (0.55, 2.2)),
        (1000,  "$n=1000$  (this work test)",  (0.80, -1.8)),
        (10000, "$n=10000$  (large-scale)",    (0.28, -1.6)),
    ]
    for n_val, label, (dx_frac, dy) in points:
        hw_val = 1.96 * np.sqrt(p*(1-p)/n_val) * 100
        ax.plot(n_val, hw_val, marker="o", markersize=8, color=C_RED, zorder=5)
        ax.annotate(label, xy=(n_val, hw_val),
                    xytext=(n_val*(1+dx_frac), hw_val+dy),
                    fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    ax.set_xscale("log")
    ax.set_xlim(50, 30000)
    ax.set_ylim(0, 14)
    ax.set_xlabel("number of normal samples in evaluation set  $n$")
    ax.set_ylabel("95% Wilson CI half-width (percentage points)")
    ax.set_title("Confidence interval width vs evaluation sample size ($p=0.5$ worst case)")
    ax.grid(alpha=0.15, which="both")

    fig.tight_layout()
    path = OUT / "ch2_wilson_ci_vs_n.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


# =====================================================
# FIG 5: RDTC four signals conceptual panel
# =====================================================
def fig_rdtc_panel():
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0))
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    # R: boundary distance
    ax = axes[0, 0]
    theta = np.linspace(-0.8, 0.8, 100)
    xb = np.sin(theta)*0.8
    yb = theta*0.9
    ax.plot(xb, yb, color=C_GREY, linewidth=1.8, linestyle="--", label="decision boundary")
    # points at various distances
    np.random.seed(2)
    for i in range(30):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        # distance to boundary (crude)
        d = abs(x - np.sin(y)*0.8)
        if d < 0.12:
            ax.plot(x, y, "o", color=C_RED, markersize=10, markeredgecolor="black", markeredgewidth=0.8)
        elif d < 0.35:
            ax.plot(x, y, "o", color=C_ORANGE, markersize=7, alpha=0.8)
        else:
            ax.plot(x, y, "o", color=C_BLUE, markersize=5, alpha=0.5)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_title("$R$: boundary risk")
    ax.text(-1.1, -1.1, "$R(x) = |p_{\\rm cal}(x) - 0.5|^{-1}$",
            fontsize=10, fontstyle="italic")
    ax.set_xticks([]); ax.set_yticks([])

    # D: kNN density
    ax = axes[0, 1]
    np.random.seed(3)
    # dense cluster
    cluster1 = np.random.normal(-0.4, 0.15, (20, 2))
    # sparse
    cluster2 = np.random.uniform(-0.8, 0.8, (8, 2))
    cluster2 += [0.5, 0.3]
    ax.scatter(cluster1[:, 0], cluster1[:, 1], color=C_BLUE, s=60, alpha=0.6, label="dense region (high $D$)")
    ax.scatter(cluster2[:, 0], cluster2[:, 1], color=C_ORANGE, s=60, alpha=0.6, label="sparse region (low $D$)")
    # mark center points
    ax.scatter([-0.4], [0.0], marker="*", s=300, color="gold", edgecolor="black", linewidth=1, zorder=5)
    ax.scatter([0.7], [0.5], marker="*", s=300, color="gold", edgecolor="black", linewidth=1, zorder=5)
    ax.set_xlim(-1.2, 1.5); ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_title("$D$: $k$NN density in feature space")
    ax.text(-1.1, -1.15, "$D(x) = (1/k)\\sum_{y \\in k{\\rm NN}(x)} {\\rm sim}(\\phi(x),\\phi(y))$",
            fontsize=9, fontstyle="italic")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])

    # T: training dynamics across epochs
    ax = axes[1, 0]
    epochs = np.arange(1, 21)
    np.random.seed(4)
    # stable sample
    stable = 0.85 + np.random.normal(0, 0.02, 20)
    # fluctuating sample (high T value)
    fluct = 0.5 + 0.3*np.sin(epochs*0.6) + np.random.normal(0, 0.05, 20)
    # easy sample
    easy = 0.95 + np.random.normal(0, 0.01, 20)

    ax.plot(epochs, easy, color=C_BLUE, linewidth=1.8, marker="o", markersize=4, label="easy (low $T$)")
    ax.plot(epochs, stable, color=C_GREEN, linewidth=1.8, marker="s", markersize=4, label="stable")
    ax.plot(epochs, fluct, color=C_RED, linewidth=2.2, marker="^", markersize=5, label="fluctuating (high $T$)")
    ax.set_xlabel("training epoch  $t$")
    ax.set_ylabel("$p_{\\rm cal,t}(x)$")
    ax.set_xlim(1, 20); ax.set_ylim(0, 1.05)
    ax.set_title("$T$: training dynamics (cross-epoch)")
    ax.text(1, -0.2, "$T(x) = {\\rm Var}_t[\\,p_{\\rm cal,t}(x)\\,]$",
            fontsize=10, fontstyle="italic", transform=ax.get_xaxis_transform())
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.15)

    # C: consistency under TTA
    ax = axes[1, 1]
    augmentations = ["orig", "flip", "rot", "crop", "color", "blur"]
    pos = np.arange(len(augmentations))
    # consistent: low variance
    cons = np.array([0.78, 0.80, 0.76, 0.79, 0.77, 0.80])
    # inconsistent: high variance
    incons = np.array([0.65, 0.35, 0.72, 0.28, 0.55, 0.41])
    ax.plot(pos, cons, color=C_BLUE, linewidth=2.0, marker="o", markersize=8, label="consistent (high $C$)")
    ax.plot(pos, incons, color=C_RED, linewidth=2.0, marker="^", markersize=8, label="inconsistent (low $C$)")
    ax.fill_between(pos, cons - 0.03, cons + 0.03, color=C_BLUE, alpha=0.15)
    ax.set_xticks(pos)
    ax.set_xticklabels(augmentations, rotation=20, ha="right")
    ax.set_ylabel("$p_{\\rm cal}(a(x))$ under augmentation $a$")
    ax.set_xlim(-0.5, 5.5); ax.set_ylim(0, 1.0)
    ax.set_title("$C$: consistency under TTA")
    ax.text(-0.5, -0.25, "$C(x) = 1 - {\\rm Var}_{a \\sim \\mathcal{A}}[\\,p_{\\rm cal}(a(x))\\,]$",
            fontsize=10, fontstyle="italic", transform=ax.get_xaxis_transform())
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.15)

    path = OUT / "ch2_rdtc_four_signals.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  {path.name}")


# =====================================================
# Run all
# =====================================================
if __name__ == "__main__":
    print("Drawing Ch2 figures...")
    fig_reliability()
    fig_threshold_search()
    fig_goldilocks()
    fig_wilson_ci()
    fig_rdtc_panel()
    print("Done.")
