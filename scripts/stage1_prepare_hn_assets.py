from __future__ import annotations

import argparse
import csv
import json
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
        "## Principles",
        "- Keep ranked CSV manifests such as `top_false_positive_normals.csv`; do not copy classification images into the repo by default.",
        "- Validation-side false-positive normals are for analysis only and must not be mixed back into training.",
        "- Reusable hard negatives must be regenerated from `train/Normal` using the formal scoring pipeline.",
        "",
        "## Prepared Materials",
    ]
    for summary in summaries:
        lines.append(
            f"- `{summary['run_name']}`: exported {summary['top_k_exported']} ranked false-positive normals and indexed {summary['train_normal_pool']} train normals."
        )
    lines.extend(
        [
            "",
            "## Backflow Guidance",
            "- Use only re-scored train-normal candidates for formal hard-negative replay.",
            "- Keep image paths and scores in CSV form; render qualitative panels only on demand outside the repo.",
            "- Missing local image files do not affect the ranked manifests or formal training preparation.",
            "",
            "## Next Steps",
            "1. Rescore `train/Normal` with the selected formal backbone.",
            "2. Select high-risk train normals by ranked score rather than copied image galleries.",
            "3. Build the HN dataset view from the ranked manifests.",
            "4. Launch the formal HN training variants.",
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
