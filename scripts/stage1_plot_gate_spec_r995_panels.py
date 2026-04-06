from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


MODEL_ORDER = ["yolo11n-cls", "yolo11s-cls", "yolo11m-cls", "yolo11l-cls", "yolo11x-cls"]
MODEL_COLORS = {
    "yolo11n-cls": "#4C78A8",
    "yolo11s-cls": "#F58518",
    "yolo11m-cls": "#54A24B",
    "yolo11l-cls": "#E45756",
    "yolo11x-cls": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-model Spec@R99.5 epoch curves for the formal gate scan.")
    parser.add_argument("--tables-dir", required=True, help="Directory containing yolo11*_gate_epoch_summary.csv tables.")
    parser.add_argument("--main-table", required=True, help="Main gate capacity table with best epochs.")
    parser.add_argument("--output-dir", required=True, help="Directory for the generated per-model figures.")
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_model_name(raw: str) -> str:
    return raw.strip()


def main() -> None:
    args = parse_args()
    tables_dir = Path(args.tables_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    main_rows = load_csv_rows(Path(args.main_table).resolve())
    best_epoch_by_model = {
        normalize_model_name(row["Model"]): int(row["Best Epoch"])
        for row in main_rows
        if row.get("Model") and row.get("Best Epoch")
    }

    curve_data: dict[str, tuple[list[int], list[float]]] = {}
    all_values: list[float] = []
    for model in MODEL_ORDER:
        csv_path = tables_dir / f"{model.replace('-cls', '')}_gate_epoch_summary.csv"
        rows = load_csv_rows(csv_path)
        epochs = [int(row["epoch"]) for row in rows]
        values = [float(row["Spec@R99.5"]) for row in rows]
        curve_data[model] = (epochs, values)
        all_values.extend(values)

    y_min = min(all_values)
    y_max = max(all_values)
    y_pad = max((y_max - y_min) * 0.08, 0.015)
    y_limits = (max(0.0, y_min - y_pad), min(1.0, y_max + y_pad))

    for model in MODEL_ORDER:
        epochs, values = curve_data[model]
        best_epoch = best_epoch_by_model[model]
        best_idx = epochs.index(best_epoch)
        best_value = values[best_idx]

        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.plot(epochs, values, color=MODEL_COLORS[model], linewidth=2.0)
        ax.axvline(best_epoch, color=MODEL_COLORS[model], linestyle="--", linewidth=1.2, alpha=0.8)
        ax.scatter([best_epoch], [best_value], color=MODEL_COLORS[model], edgecolor="white", linewidth=0.9, s=42, zorder=3)
        ax.annotate(
            f"best@{best_epoch}",
            xy=(best_epoch, best_value),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=8,
            color=MODEL_COLORS[model],
        )
        ax.set_title(f"{model}  Spec@R99.5", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Spec@R99.5")
        ax.set_xlim(1, max(epochs))
        ax.set_ylim(*y_limits)
        ax.grid(alpha=0.22)
        fig.tight_layout()
        fig.savefig(output_dir / f"fig_stage1_gate_spec_r995_{model.replace('-cls', '')}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
