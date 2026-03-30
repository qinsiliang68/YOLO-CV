from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare stage-1 hard-negative analysis assets from existing raw materials.")
    parser.add_argument("--materials-root", default="research/materials", help="Root raw-material directory.")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["yolo11l_gate2_train7200", "yolo11s_gate2_train7200"],
        help="Gate runs used for hard-negative analysis assets.",
    )
    parser.add_argument("--output-root", default="research/materials/stage1_hn", help="Output directory.")
    parser.add_argument("--top-k", type=int, default=60, help="Top-K false-positive normals to export.")
    return parser.parse_args()


def heuristic_group(image_path: Path) -> tuple[str, str]:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        hsv = rgb.convert("HSV")
        hsv_np = np.asarray(hsv, dtype=np.float32)
        gray = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0

    mean_v = float(hsv_np[..., 2].mean())
    std_v = float(hsv_np[..., 2].std())
    mean_s = float(hsv_np[..., 1].mean())
    bright_ratio = float((hsv_np[..., 2] > 245).mean())
    dark_ratio = float((hsv_np[..., 2] < 45).mean())
    grad_y, grad_x = np.gradient(gray)
    edge_ratio = float((np.hypot(grad_x, grad_y) > 0.18).mean())

    if bright_ratio > 0.06 and std_v > 35:
        return "反光", "高亮区域比例偏高且亮度波动明显。"
    if dark_ratio > 0.42:
        return "阴影", "低亮度区域占比较高。"
    if mean_s < 35 and std_v < 28:
        return "水膜", "饱和度较低且整体亮度变化平缓。"
    if edge_ratio > 0.18:
        return "接缝", "边缘密度偏高，疑似结构边界触发误报。"
    if std_v > 45 and mean_s > 40:
        return "污渍", "亮度与颜色波动同时较强。"
    if edge_ratio > 0.12:
        return "纹理伪异常", "纹理边缘较密，疑似结构纹理干扰。"
    return "其他", "未命中显著启发式模式。"


def export_gallery(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, start=1):
        source = Path(row["img_path"])
        if not source.exists():
            continue
        target = output_dir / f"{index:03d}_{float(row['score']):.4f}_{row['heuristic_group']}_{source.name}"
        shutil.copy2(source, target)


def process_run(materials_root: Path, run_name: str, output_root: Path, top_k: int) -> dict:
    run_dir = materials_root / run_name
    fp_path = run_dir / "fp_normal.csv"
    split_train_path = run_dir / "split_train.csv"
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with fp_path.open("r", encoding="utf-8-sig", newline="") as handle:
        fp_rows = list(csv.DictReader(handle))
    rows_sorted = sorted(fp_rows, key=lambda item: float(item["abnormal_conf"]), reverse=True)

    top_rows: list[dict] = []
    for row in rows_sorted[:top_k]:
        source = Path(row["img_path"])
        if source.exists():
            group, reason = heuristic_group(source)
        else:
            group, reason = ("其他", "本地工作区缺少对应图像，仅保留数值记录。")
        top_rows.append(
            {
                "img_path": row["img_path"],
                "img_rel_path": row["img_rel_path"],
                "gt": row["gt_label"],
                "pred": row["pred_label"],
                "score": float(row["abnormal_conf"]),
                "top1_prob": float(row["top1_prob"]),
                "top2_label": row["top2_label"],
                "top2_prob": float(row["top2_prob"]),
                "top3_label": row["top3_label"],
                "top3_prob": float(row["top3_prob"]),
                "heuristic_group": group,
                "heuristic_reason": reason,
                "training_use": "no",
            }
        )

    if top_rows:
        with (output_dir / "top_false_positive_normals.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(top_rows[0].keys()))
            writer.writeheader()
            writer.writerows(top_rows)
        export_gallery(top_rows[:24], output_dir / "hardest_normal_gallery")

    train_normal_rows: list[dict] = []
    with split_train_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            relative_path = row["relative_path"].replace("\\", "/")
            if row["label"] == "Normal" or "/Normal/" in relative_path:
                train_normal_rows.append(row)
    if train_normal_rows:
        with (output_dir / "train_normal_pool_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(train_normal_rows[0].keys()))
            writer.writeheader()
            writer.writerows(train_normal_rows)

    counts: dict[str, int] = {}
    for row in top_rows:
        counts[row["heuristic_group"]] = counts.get(row["heuristic_group"], 0) + 1
    (output_dir / "heuristic_group_summary.json").write_text(
        json.dumps(counts, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "run_name": run_name,
        "top_k_exported": len(top_rows),
        "train_normal_pool": len(train_normal_rows),
        "heuristic_counts": counts,
    }


def write_plan(output_root: Path, summaries: list[dict]) -> None:
    lines = [
        "# Stage-1 Hard Negative Plan",
        "",
        "## 原则",
        "- 本目录下的 `top_false_positive_normals.csv` 与 `hardest_normal_gallery/` 来自现有验证侧误报 normal，仅用于展示、分析与误报模式归纳。",
        "- 验证侧误报样本不得直接回流训练，避免污染 val-op。",
        "- 真正可回流的 hard negative 必须来自训练侧 normal 池或额外 normal 池；后续统一通过 `stage1_score_train_normals.py` 在 `train/Normal` 上重新打分生成。",
        "",
        "## 当前已准备材料",
    ]
    for summary in summaries:
        lines.append(
            f"- `{summary['run_name']}`：导出验证侧误报 normal {summary['top_k_exported']} 张，训练侧 normal 池 {summary['train_normal_pool']} 张。"
        )
    lines.extend(
        [
            "",
            "## 回流建议",
            "- 可回流：后续在 `train/Normal` 中重新打分后选出的高置信 normal 误报样本。",
            "- 仅展示或分析：当前 `fp_normal.csv` 导出的验证侧 hardest normal 画廊。",
            "- 若本地工作区缺少原始图像，当前 gallery 可能为空；不影响数值筛选与训练机上的正式构建。",
            "",
            "## 后续执行顺序",
            "1. 用确定的主模型和第二模型在 `train/Normal` 上重新打分。",
            "2. 按分数排序选择 top-k 训练侧 hard negatives。",
            "3. 通过 `stage1_build_hn_dataset.py` 生成带 HN 重复样本的新数据集。",
            "4. 再启动 HN、Weighted BCE 和 Focal 版本训练。",
        ]
    )
    (output_root / "stage1_hn_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    materials_root = Path(args.materials_root).resolve()
    output_root = Path(args.output_root).resolve()
    summaries = [process_run(materials_root, run_name, output_root, args.top_k) for run_name in args.runs]
    write_plan(output_root, summaries)


if __name__ == "__main__":
    main()
