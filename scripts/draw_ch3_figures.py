"""Draw Ch3 dataset & protocol figures for essay3."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from pathlib import Path

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.1,
})

OUT = Path(r"C:\GitHub\YOLO-CV\essay\img")
OUT.mkdir(parents=True, exist_ok=True)

C_BLUE   = "#377EB8"
C_RED    = "#E41A1C"
C_GREEN  = "#4DAF4A"
C_ORANGE = "#FF7F00"
C_PURPLE = "#984EA3"
C_BROWN  = "#A65628"
C_PINK   = "#F781BF"
C_GREY   = "#999999"


# =====================================================
# FIG 3.1: Data funnel - SewerML -> filtered pool -> splits
# =====================================================
def fig_data_funnel():
    fig, ax = plt.subplots(figsize=(11, 6.0))

    stages = [
        ("SewerML Train 训练集\nn = 1,040,000",              1.00, C_GREY,   "原始",       "original"),
        ("剔除 OK + PH\n(图像质量问题)\nn ≈ 860,000",         0.84, C_BROWN,  "≈ 17.5%",   "drop OK/PH"),
        ("剔除仅具 holdout 类帧\nn ≈ 820,000",                0.80, C_ORANGE, "≈ 4%",      "drop holdout"),
        ("采样池 sampling pool\n(本文可用)\nn ≈ 820,000",     0.80, C_PURPLE, "随机抽样",   "random sampling"),
    ]

    y_positions = np.linspace(0.85, 0.28, len(stages))
    box_height = 0.10

    for (name, w, color, left_lbl, right_lbl), y in zip(stages, y_positions):
        left = 0.5 - w/2
        rect = Rectangle((left, y - box_height/2), w, box_height,
                         facecolor=color, alpha=0.78, edgecolor="black", linewidth=0.8)
        ax.add_patch(rect)
        ax.text(0.5, y, name, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        ax.text(left - 0.015, y, left_lbl, ha="right", va="center",
                fontsize=9, color=color)
        ax.text(left + w + 0.015, y, right_lbl, ha="left", va="center",
                fontsize=9, color=color)

    for y_from, y_to in zip(y_positions[:-1], y_positions[1:]):
        ax.annotate("", xy=(0.5, y_to + box_height/2 + 0.005),
                    xytext=(0.5, y_from - box_height/2 - 0.005),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    final_y = y_positions[-1]
    splits = [
        ("train\n24,000",   C_BLUE,  0.15),
        ("val-cal\n2,400",  C_GREEN, 0.38),
        ("val-op\n5,600",   C_GREEN, 0.61),
        ("test\n20,000",    C_RED,   0.85),
    ]
    split_y = 0.06
    split_w = 0.14
    split_h = 0.09
    for name, color, x in splits:
        rect = Rectangle((x - split_w/2, split_y - split_h/2), split_w, split_h,
                         facecolor=color, alpha=0.8, edgecolor="black", linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x, split_y, name, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        ax.annotate("", xy=(x, split_y + split_h/2 + 0.005),
                    xytext=(0.5, final_y - box_height/2 - 0.005),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.7))

    ax.set_xlim(-0.10, 1.15)
    ax.set_ylim(-0.02, 1.00)
    ax.set_aspect("auto")
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    ax.set_title("SewerML 到本文数据集的构建漏斗（数字为示意值）", pad=10)

    path = OUT / "ch3_data_funnel.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


# =====================================================
# FIG 3.2: CJJ class distribution in sampling pool
# =====================================================
def fig_cjj_distribution():
    fig, ax = plt.subplots(figsize=(9.5, 4.5))

    # approximate numbers based on our earlier analysis
    classes = [
        ("Normal\n正常", 550000, C_BLUE),
        ("Joint\n错口 CK (FS)", 284000, C_RED),
        ("Obstacle\n障碍物 ZW (OB)", 184000, C_ORANGE),
        ("Deposit\n沉积 CJ (AF)", 75000, C_GREEN),
        ("Corrosion\n腐蚀 FS (RB)", 46000, C_PURPLE),
        ("Deformation\n变形 BX (DE)", 19000, C_BROWN),
        ("Pipe break\n破裂 PL (PF)", 16000, C_PINK),
    ]

    names = [c[0] for c in classes]
    counts = [c[1] for c in classes]
    colors = [c[2] for c in classes]

    x = np.arange(len(classes))
    bars = ax.bar(x, counts, color=colors, alpha=0.8, edgecolor="black", linewidth=0.6)

    # annotate value on top
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02,
                f"{n:,}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(5000, 1e6)
    ax.set_ylabel("该类别在 SewerML 训练集中出现的图像数（log scale）")
    ax.set_title("CJJ 6 类 + 正常类在 SewerML 中的自然分布（图像可属多类，非互斥）")
    ax.grid(alpha=0.15, which="major", axis="y")

    fig.tight_layout()
    path = OUT / "ch3_cjj_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


# =====================================================
# FIG 3.3: Wilson CI table as visual comparison
# =====================================================
def fig_ci_comparison():
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    scenarios = [
        ("previous protocols\n(small val)", 84, "old", C_GREY),
        ("this work\nval-op", 3800, "new-val", C_GREEN),
        ("this work\ntest set", 13400, "new-test", C_BLUE),
    ]

    names = [s[0] for s in scenarios]
    ns = [s[1] for s in scenarios]
    colors = [s[3] for s in scenarios]

    # Wilson half-width at p=0.5 worst case
    hws = [1.96 * np.sqrt(0.25 / n) * 100 for n in ns]

    x = np.arange(len(scenarios))
    bars = ax.bar(x, hws, color=colors, alpha=0.8, edgecolor="black", linewidth=0.6, width=0.5)

    for bar, hw, n in zip(bars, hws, ns):
        ax.text(bar.get_x() + bar.get_width()/2, hw + 0.3,
                f"$\\pm${hw:.2f}pp\n($n={n}$)", ha="center", fontsize=10, fontweight="bold")

    # add line at 3pp threshold
    ax.axhline(3, color=C_RED, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(2.3, 3.2, "target $\\pm 3$pp threshold", color=C_RED, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 13)
    ax.set_ylabel("95% Wilson CI 半宽度 (pp)")
    ax.set_title("不同评估集规模下 Spec@R$_0$ 估计的置信区间宽度（$p = 0.5$ 最坏情形）")
    ax.grid(alpha=0.15, axis="y")

    fig.tight_layout()
    path = OUT / "ch3_ci_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


if __name__ == "__main__":
    print("Drawing Ch3 figures...")
    fig_data_funnel()
    fig_cjj_distribution()
    fig_ci_comparison()
    print("Done.")
