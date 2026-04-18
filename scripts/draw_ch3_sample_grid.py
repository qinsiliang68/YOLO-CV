"""从 SewerML 里为每类抽取代表样本，拼成 Ch3 类别示例图。"""
import csv
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pathlib import Path

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
    "axes.unicode_minus": False,
})

ANNOT = Path(r"C:\GitHub\YOLO-CV\YOLOv11\datasets\sewerml_annotations\SewerML_Train.csv")
IMG_DIR = Path(r"C:\baidunetdiskdownload\sewerml_train_images")
OUT = Path(r"C:\GitHub\YOLO-CV\essay\img\ch3_sample_grid.png")

# 映射：CJJ 类别 -> 对应 SewerML 代码 + 排除其他干扰
CLASS_SPEC = [
    ("正常 Normal",            "Defect",  0, [],                 (0.2, 0.4, 0.7)),
    ("破裂 PL (PF)",           "PF",      1, ["DE","FS","RB","OB"], (0.9, 0.2, 0.2)),
    ("变形 BX (DE)",           "DE",      1, ["PF","FS","RB","OB"], (0.6, 0.3, 0.7)),
    ("错口 CK (FS)",           "FS",      1, ["PF","DE","RB","OB"], (0.2, 0.6, 0.3)),
    ("腐蚀 FS (RB)",           "RB",      1, ["PF","DE","FS","OB"], (0.9, 0.5, 0.1)),
    ("沉积 CJ (AF)",           "AF",      1, ["BE","PF","DE"],      (0.5, 0.3, 0.1)),
    ("障碍物 ZW (OB)",         "OB",      1, ["PF","DE","FS","RB"], (0.3, 0.3, 0.3)),
]


def find_representative(target_col, target_val, exclude_cols, n_candidates=200, n_show=3, seed=42):
    """从 SewerML CSV 里找 target_col==target_val 且 exclude_cols 全为 0 的样本。"""
    random.seed(seed)
    candidates = []
    with open(ANNOT, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # filter for quality (drop OK/PH)
            if row.get("OK", "0") == "1" or row.get("PH", "0") == "1":
                continue
            # must match target
            if target_col == "Defect":
                if int(row.get("Defect", "0")) != target_val:
                    continue
            else:
                if int(row.get(target_col, "0")) != target_val:
                    continue
            # must exclude other strong defect codes (for cleanest single-label shots)
            ok_exclude = True
            for c in exclude_cols:
                if int(row.get(c, "0")) == 1:
                    ok_exclude = False
                    break
            if not ok_exclude:
                continue
            fname = row["Filename"].replace(".png", "")
            candidates.append(fname)
            if len(candidates) >= n_candidates:
                break
    if not candidates:
        return []
    # pick n_show from front (cleaner labeling) with some randomization
    random.shuffle(candidates)
    return candidates[:n_show]


def main():
    n_show = 3
    rows = len(CLASS_SPEC)
    fig, axes = plt.subplots(rows, n_show, figsize=(n_show*2.8, rows*2.2))
    fig.subplots_adjust(hspace=0.25, wspace=0.05, left=0.18, right=0.98, top=0.98, bottom=0.02)

    for i, (name, col, val, excl, title_color) in enumerate(CLASS_SPEC):
        ids = find_representative(col, val, excl, n_show=n_show, seed=42 + i)
        print(f"{name}: found {len(ids)} samples")
        for j in range(n_show):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if j < len(ids):
                img_path = IMG_DIR / f"{ids[j]}.png"
                if img_path.exists():
                    img = imread(str(img_path))
                    ax.imshow(img)
                    ax.text(0.02, 0.96, ids[j], transform=ax.transAxes,
                            fontsize=7, color="white",
                            bbox=dict(facecolor="black", alpha=0.55, pad=1, edgecolor="none"),
                            va="top")
                else:
                    ax.text(0.5, 0.5, f"{ids[j]}\n(not on disk)", ha="center", va="center",
                             transform=ax.transAxes, fontsize=8)
            else:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=9)
        # row label on left
        axes[i, 0].annotate(name, xy=(-0.08, 0.5), xycoords="axes fraction",
                            ha="right", va="center", fontsize=11, fontweight="bold",
                            color=title_color)

    fig.savefig(OUT, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
