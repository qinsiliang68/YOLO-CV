from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot summary figures for the stage-1 max-filter suite.")
    parser.add_argument("--summary-csv", required=True, help="Path to stage1_maxfilter_suite_summary.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for result figures")
    parser.add_argument("--essay-dir", default="", help="Optional thesis figure directory")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str) -> float:
    return float(value) if value not in ("", None) else float("nan")


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def plot_best_metrics(rows: list[dict[str, str]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["label"] for row in rows]
    short_labels = ["H0", "Selective", "HardMix", "WBCE", "Focal", "DefectOS"]
    metrics = [
        ("best_spec_at_r995", "Spec@R99.5", "#355C7D"),
        ("best_spec_at_r990", "Spec@R99.0", "#C06C84"),
        ("best_prec_at_r990", "Prec@R99.0", "#6C5B7B"),
        ("best_ptr_at_r990", "PTR@R99.0", "#F67280"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    axes = axes.flatten()
    x = list(range(len(labels)))

    for axis, (key, title, color) in zip(axes, metrics, strict=True):
        values = [as_float(row[key]) for row in rows]
        bars = axis.bar(x, values, color=color, alpha=0.88)
        axis.set_title(title, fontsize=12)
        axis.set_xticks(x, short_labels, rotation=18)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        axis.set_axisbelow(True)
        if "ptr" not in key:
            axis.set_ylim(0.35, 0.95)
        else:
            axis.set_ylim(0.88, 0.915)
        for idx, (bar, value) in enumerate(zip(bars, values, strict=True)):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
            if idx == 0:
                continue
            delta = value - values[0]
            delta_text = f"{delta:+.4f}"
            axis.text(bar.get_x() + bar.get_width() / 2, value - 0.02 if "ptr" not in key else value - 0.004, delta_text, ha="center", va="bottom", fontsize=8, color="#222222")

    fig.suptitle("Stage-1 Max-Filter Suite: Best Operating-Point Metrics", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_p0_p2_behavior(rows: list[dict[str, str]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plot_rows = [row for row in rows if row["label"] != "H0 current best hn02 + P2"]
    short_labels = ["Selective", "HardMix", "WBCE", "Focal", "DefectOS"]
    x = list(range(len(plot_rows)))
    width = 0.34

    p0_r995 = [as_float(row["p0_spec_at_r995"]) for row in plot_rows]
    p2_r995 = [as_float(row["p2_spec_at_r995"]) for row in plot_rows]
    p0_r990 = [as_float(row["p0_spec_at_r990"]) for row in plot_rows]
    p2_r990 = [as_float(row["p2_spec_at_r990"]) for row in plot_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=False)
    palette = {"p0": "#4C78A8", "p2": "#E45756"}

    for axis, p0_vals, p2_vals, title in (
        (axes[0], p0_r995, p2_r995, "P0 vs P2 on Spec@R99.5"),
        (axes[1], p0_r990, p2_r990, "P0 vs P2 on Spec@R99.0"),
    ):
        axis.bar([v - width / 2 for v in x], p0_vals, width=width, color=palette["p0"], label="P0")
        axis.bar([v + width / 2 for v in x], p2_vals, width=width, color=palette["p2"], label="P2")
        axis.set_title(title, fontsize=12)
        axis.set_xticks(x, short_labels, rotation=18)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        axis.set_axisbelow(True)
        axis.set_ylim(0.35, 0.62)
        for idx, (p0_val, p2_val) in enumerate(zip(p0_vals, p2_vals, strict=True)):
            axis.text(idx - width / 2, p0_val + 0.005, f"{p0_val:.4f}", ha="center", va="bottom", fontsize=8)
            axis.text(idx + width / 2, p2_val + 0.005, f"{p2_val:.4f}", ha="center", va="bottom", fontsize=8)
            axis.text(idx, max(p0_val, p2_val) + 0.028, f"Δ {p2_val - p0_val:+.4f}", ha="center", va="bottom", fontsize=8, color="#333333")

    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("How Much Trust Still Helps After Each Training Strategy", fontsize=14, y=1.03)
    fig.tight_layout()
    save_figure(fig, output_path)


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    essay_dir = Path(args.essay_dir).resolve() if args.essay_dir else None

    rows = load_rows(summary_csv)
    best_metrics_path = output_dir / "stage1_maxfilter_suite_best_metrics.png"
    p0_p2_path = output_dir / "stage1_maxfilter_suite_p0_p2_behavior.png"
    plot_best_metrics(rows, best_metrics_path)
    plot_p0_p2_behavior(rows, p0_p2_path)

    if essay_dir is not None:
        essay_dir.mkdir(parents=True, exist_ok=True)
        for src in (best_metrics_path, best_metrics_path.with_suffix(".pdf"), p0_p2_path, p0_p2_path.with_suffix(".pdf")):
            dst = essay_dir / src.name
            dst.write_bytes(src.read_bytes())

    print(f"[done] wrote {output_dir}")


if __name__ == "__main__":
    main()
