from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(r"C:\GitHub\YOLO-CV")
OUT_DIR = ROOT / "essay" / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _setup(fig_w: float, fig_h: float):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _box(ax, x, y, w, h, text, fc="#F7F7F7", ec="#2F4F4F", text_size=11, lw=1.4):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=text_size,
        wrap=True,
        color="#1B1B1B",
    )


def _arrow(ax, x1, y1, x2, y2, color="#4A6FA5", lw=1.8, style="-|>"):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color=color,
    )
    ax.add_patch(arr)


def save(fig, stem: str):
    pdf_path = OUT_DIR / f"{stem}.pdf"
    png_path = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


def fig_ch1_timeline():
    fig, ax = _setup(14, 4.6)
    y = 0.50
    _arrow(ax, 0.05, y, 0.95, y, color="#355C7D", lw=2.4, style="->")

    nodes = [
        (0.08, "2014\n传统视觉\nHOG + SVM"),
        (0.22, "2018\nCNN 分类\n与早期深度\n学习检测"),
        (0.36, "2019-2020\n层次分类\n综述与系统\n研究"),
        (0.50, "2021\nSewer-ML\n基准、跟踪\n与特征融合"),
        (0.66, "2023\n迁移学习\n与改进型\nYOLO 研究"),
        (0.80, "2024-2025\n弱监督定位\n多模态、轻量化\n与实时方法"),
        (0.92, "本文工作\n改进 YOLOv11 +\n跨域训练"),
    ]
    colors = ["#D9EAF7", "#E6F4D7", "#FFF0C9", "#FDE2E4", "#D9EAF7", "#E6F4D7", "#F4D6FF"]
    for idx, (x, text) in enumerate(nodes):
        _box(ax, x - 0.06, y - 0.12, 0.12, 0.24, text, fc=colors[idx], text_size=10)
        ax.plot([x, x], [y - 0.04, y + 0.04], color="#355C7D", lw=2)

    ax.text(0.08, 0.16, "人工 CCTV 判读", fontsize=10, ha="center", color="#555555")
    ax.text(0.36, 0.16, "深度学习成为主线", fontsize=10, ha="center", color="#555555")
    ax.text(0.66, 0.16, "基准数据 + 迁移 + 弱监督", fontsize=10, ha="center", color="#555555")
    ax.text(0.92, 0.16, "场景化优化与落地", fontsize=10, ha="center", color="#555555")
    save(fig, "ch1_research_evolution_timeline")


def fig_ch1_problem_chain():
    fig, ax = _setup(12.5, 7.2)
    _box(ax, 0.06, 0.68, 0.24, 0.16, "工程现实\n海量 CCTV 视频\n正常帧比例高\n人工复核成本大", fc="#D9EAF7")
    _box(ax, 0.38, 0.68, 0.24, 0.16, "视觉难点\n缺陷目标小\n纹理弱\n背景复杂", fc="#E6F4D7")
    _box(ax, 0.70, 0.68, 0.24, 0.16, "数据瓶颈\n目标域框标注\n代价高\n质量不均", fc="#FFF0C9")

    _box(ax, 0.06, 0.38, 0.24, 0.16, "现有路径 1\n直接做检测\n对精标依赖高", fc="#FDE2E4")
    _box(ax, 0.38, 0.38, 0.24, 0.16, "现有路径 2\n分类 / 基准研究\n但难以精确定位", fc="#FDE2E4")
    _box(ax, 0.70, 0.38, 0.24, 0.16, "现有路径 3\n迁移 / 弱监督\n有效但链路分散", fc="#FDE2E4")

    _box(ax, 0.24, 0.10, 0.52, 0.16, "本文路线\n源域分类预训练 → 目标域适配 → CAM 候选区域 → 改进 YOLOv11 检测", fc="#F4D6FF", text_size=12, ec="#6C3483")

    for x in [0.18, 0.50, 0.82]:
        _arrow(ax, x, 0.68, x, 0.54, color="#4A6FA5")
    _arrow(ax, 0.18, 0.38, 0.42, 0.26, color="#6C3483")
    _arrow(ax, 0.50, 0.38, 0.50, 0.26, color="#6C3483")
    _arrow(ax, 0.82, 0.38, 0.58, 0.26, color="#6C3483")
    save(fig, "ch1_problem_chain")


def fig_ch2_supervision():
    fig, ax = _setup(13.5, 5.5)
    items = [
        (0.04, "源域\n单标签\n分类训练", "#D9EAF7"),
        (0.20, "目标域\n分类\n微调", "#E6F4D7"),
        (0.36, "CAM /\nGrad-CAM\n热力图", "#FFF0C9"),
        (0.52, "伪框生成\n阈值化 +\n连通域", "#FDE2E4"),
        (0.68, "复核框\n少量人工\n修正", "#EAE0FF"),
        (0.84, "检测器\n训练与\n迭代", "#D6F5F3"),
    ]
    y = 0.42
    for x, label, color in items:
        _box(ax, x, y, 0.12, 0.22, label, fc=color, text_size=11)
    for x1, x2 in [(0.16, 0.20), (0.32, 0.36), (0.48, 0.52), (0.64, 0.68), (0.80, 0.84)]:
        _arrow(ax, x1, y + 0.11, x2, y + 0.11, color="#4A6FA5", lw=2.0)

    ax.text(0.17, 0.18, "粗粒度语义监督", fontsize=10, ha="center", color="#555555")
    ax.text(0.50, 0.18, "候选区域监督", fontsize=10, ha="center", color="#555555")
    ax.text(0.83, 0.18, "细粒度检测监督", fontsize=10, ha="center", color="#555555")
    save(fig, "ch2_supervision_continuum")


def fig_ch2_theory_map():
    fig, ax = _setup(13.5, 7.0)
    ax.text(0.16, 0.82, "任务难点", fontsize=13, weight="bold", ha="center")
    ax.text(0.50, 0.82, "理论基础", fontsize=13, weight="bold", ha="center")
    ax.text(0.84, 0.82, "方法响应", fontsize=13, weight="bold", ha="center")

    left = [
        "小目标 / 细长型\n结构缺陷",
        "背景噪声\n与反光干扰",
        "类别不均衡\n与难样本",
        "域差显著\n且标注有限",
    ]
    mid = [
        "小目标检测\n与上下文建模",
        "注意力机制\n与伪显著抑制",
        "损失函数设计\n与标签分配",
        "迁移学习\nCAM 与弱监督定位",
    ]
    right = [
        "增强特征\n表达能力",
        "注意力模块\n场景适配",
        "检测头 + 损失\n协同优化",
        "源域预训练 +\n目标域适配",
    ]

    ys = [0.66, 0.49, 0.32, 0.15]
    colors = ["#D9EAF7", "#E6F4D7", "#FFF0C9", "#FDE2E4"]
    for i, y in enumerate(ys):
        _box(ax, 0.06, y, 0.20, 0.11, left[i], fc=colors[i])
        _box(ax, 0.40, y, 0.20, 0.11, mid[i], fc="#F7F7F7")
        _box(ax, 0.74, y, 0.20, 0.11, right[i], fc="#EAE0FF")
        _arrow(ax, 0.26, y + 0.055, 0.40, y + 0.055, color="#4A6FA5")
        _arrow(ax, 0.60, y + 0.055, 0.74, y + 0.055, color="#6C3483")

    save(fig, "ch2_theory_support_map")


def fig_ch2_domain_gap():
    fig, ax = _setup(12.8, 6.4)
    _box(ax, 0.09, 0.36, 0.26, 0.20, "源域\n公开基准 /\n源域分类集", fc="#D9EAF7", text_size=12)
    _box(ax, 0.65, 0.36, 0.26, 0.20, "目标域\n本地巡检图像 /\n检测数据集", fc="#EAE0FF", text_size=12)

    center_items = [
        (0.42, 0.68, "设备参数 /\n照明条件"),
        (0.42, 0.54, "管材差异 /\n壁面纹理"),
        (0.42, 0.40, "缺陷外观 /\n严重程度"),
        (0.42, 0.26, "标注口径 /\n监督粒度"),
    ]
    for x, y, txt in center_items:
        _box(ax, x, y, 0.16, 0.10, txt, fc="#FFF0C9", text_size=10)
        _arrow(ax, 0.35, 0.46, x, y + 0.05, color="#4A6FA5", lw=1.4)
        _arrow(ax, x + 0.16, y + 0.05, 0.65, 0.46, color="#6C3483", lw=1.4)

    ax.text(0.50, 0.11, "需要：源域先验 + 目标域适配 + 低标注增强", ha="center", fontsize=11, color="#333333")
    save(fig, "ch2_domain_gap_factors")


def fig_ch2_yolo11_architecture():
    fig, ax = _setup(13.2, 5.6)
    _box(ax, 0.05, 0.30, 0.14, 0.18, "输入图像", fc="#D9EAF7", text_size=12)
    _box(ax, 0.24, 0.26, 0.18, 0.26, "主干网络\n卷积与残差模块\n提取多层特征", fc="#E6F4D7", text_size=12)
    _box(ax, 0.47, 0.26, 0.18, 0.26, "颈部融合\n多尺度特征聚合\nP3 / P4 / P5", fc="#FFF0C9", text_size=12)
    _box(ax, 0.70, 0.26, 0.18, 0.26, "检测头\n分类 + 回归\n多尺度输出", fc="#FDE2E4", text_size=12)
    _box(ax, 0.91, 0.30, 0.06, 0.18, "结果", fc="#EAE0FF", text_size=12)

    _arrow(ax, 0.19, 0.39, 0.24, 0.39, color="#4A6FA5", lw=2.0)
    _arrow(ax, 0.42, 0.39, 0.47, 0.39, color="#4A6FA5", lw=2.0)
    _arrow(ax, 0.65, 0.39, 0.70, 0.39, color="#4A6FA5", lw=2.0)
    _arrow(ax, 0.88, 0.39, 0.91, 0.39, color="#4A6FA5", lw=2.0)

    ax.text(0.33, 0.62, "低层细节特征", fontsize=10, ha="center", color="#555555")
    ax.text(0.56, 0.62, "多尺度语义融合", fontsize=10, ha="center", color="#555555")
    ax.text(0.79, 0.62, "目标类别与位置预测", fontsize=10, ha="center", color="#555555")
    save(fig, "ch2_yolo11_architecture")


def main():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig_ch1_timeline()
    fig_ch1_problem_chain()
    fig_ch2_supervision()
    fig_ch2_theory_map()
    fig_ch2_domain_gap()
    fig_ch2_yolo11_architecture()


if __name__ == "__main__":
    main()
