"""为 Ch3 每个 CJJ 类别生成独立的 3 图横排示例。"""
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
OUT_DIR = Path(r"C:\GitHub\YOLO-CV\essay\img")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# key, display name, target column, target value, exclude list, suffix for filename
CLASS_SPEC = [
    ("normal",       "Normal（正常）",        "Defect",  0, []),
    ("break",        "PL 破裂 (PF)",          "PF",      1, ["DE","FS","RB","OB"]),
    ("deformation",  "BX 变形 (DE)",          "DE",      1, ["PF","FS","RB","OB"]),
    ("dislocation",  "CK 错口 (FS)",          "FS",      1, ["PF","DE","RB","OB"]),
    ("corrosion",    "FS 腐蚀 (RB)",          "RB",      1, ["PF","DE","FS","OB"]),
    ("deposit",      "CJ 沉积 (AF)",          "AF",      1, ["BE","PF","DE"]),
    ("obstacle",     "ZW 障碍物 (OB)",        "OB",      1, ["PF","DE","FS","RB"]),
]


def find_representative(target_col, target_val, exclude_cols, n_candidates=300, n_show=3, seed=42):
    random.seed(seed)
    candidates = []
    with open(ANNOT, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("OK", "0") == "1" or row.get("PH", "0") == "1":
                continue
            if target_col == "Defect":
                if int(row.get("Defect", "0")) != target_val:
                    continue
            else:
                if int(row.get(target_col, "0")) != target_val:
                    continue
            ok = True
            for c in exclude_cols:
                if int(row.get(c, "0")) == 1:
                    ok = False
                    break
            if not ok:
                continue
            fname = row["Filename"].replace(".png", "")
            candidates.append(fname)
            if len(candidates) >= n_candidates:
                break
    if not candidates:
        return []
    random.shuffle(candidates)
    return candidates[:n_show]


def make_per_class_figure(key, name, col, val, excl, seed_offset):
    n_show = 3
    ids = find_representative(col, val, excl, n_show=n_show, seed=100 + seed_offset)
    fig, axes = plt.subplots(1, n_show, figsize=(n_show * 2.6, 2.3))
    fig.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.94, bottom=0.02)
    for j in range(n_show):
        ax = axes[j]
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
                        bbox=dict(facecolor="black", alpha=0.55, pad=1.5, edgecolor="none"),
                        va="top")

    out = OUT_DIR / f"ch3_class_{key}.png"
    fig.savefig(out, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  {out.name}  ({name}, {len(ids)} shown)")


def main():
    print("Generating per-class figures...")
    for i, (key, name, col, val, excl) in enumerate(CLASS_SPEC):
        make_per_class_figure(key, name, col, val, excl, seed_offset=i)
    print("Done.")


if __name__ == "__main__":
    main()
