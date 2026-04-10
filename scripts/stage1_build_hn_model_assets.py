from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRIC_COLUMNS = {
    "spec_at_r995": "Spec@R99.5",
    "spec_at_r990": "Spec@R99.0",
    "prec_at_r990": "Prec@R99.0",
    "ptr_at_r990": "PTR@R99.0",
}

LOWER_IS_BETTER = {"ptr_at_r990"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HN paper assets for a single model.")
    parser.add_argument("--summary-csv", required=True, help="hn_sweep_summary.csv path")
    parser.add_argument("--materials-root", required=True, help="Directory containing hn00/hn02/... epoch summaries")
    parser.add_argument("--output-root", required=True, help="Output directory for tables/figures")
    parser.add_argument("--model-tag", required=True, help="Short tag for output stems, e.g. n or m")
    parser.add_argument("--model-label", required=True, help="Display label, e.g. yolo11n-cls")
    parser.add_argument(
        "--representative-ratios",
        nargs="+",
        help="Representative ratios for epoch dynamics. Default: hn00, winner, max ratio.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ratio_percent"] = df["ratio_percent"].astype(int)
    df["best_epoch"] = df["best_epoch"].astype(int)
    for col in METRIC_COLUMNS:
        df[col] = df[col].astype(float)
    return df.sort_values("ratio_percent").reset_index(drop=True)


def compute_rank_and_delta(df: pd.DataFrame) -> pd.DataFrame:
    base = df.loc[df["ratio_id"] == "hn00"].iloc[0]
    ranked = df.sort_values(
        by=["spec_at_r995", "spec_at_r990", "prec_at_r990", "ptr_at_r990", "ratio_percent"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    rank_map = {ratio_id: idx + 1 for idx, ratio_id in enumerate(ranked["ratio_id"].tolist())}
    df = df.copy()
    df["formal_rank"] = df["ratio_id"].map(rank_map).astype(int)
    for metric in METRIC_COLUMNS:
        df[f"delta_{metric}"] = df[metric] - float(base[metric])
    return df


def ensure_dirs(output_root: Path) -> tuple[Path, Path]:
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir


def write_table_csv(df: pd.DataFrame, tables_dir: Path, model_tag: str) -> Path:
    export = df[
        [
            "ratio_id",
            "best_epoch",
            "spec_at_r995",
            "spec_at_r990",
            "prec_at_r990",
            "ptr_at_r990",
            "formal_rank",
            "delta_spec_at_r995",
            "delta_spec_at_r990",
            "delta_prec_at_r990",
            "delta_ptr_at_r990",
        ]
    ].copy()
    export.columns = [
        "Ratio",
        "Best Epoch",
        "Spec@R99.5",
        "Spec@R99.0",
        "Prec@R99.0",
        "PTR@R99.0",
        "Rank",
        "Delta Spec@R99.5",
        "Delta Spec@R99.0",
        "Delta Prec@R99.0",
        "Delta PTR@R99.0",
    ]
    path = tables_dir / f"table_hn_{model_tag}_main_ranking.csv"
    export.to_csv(path, index=False)
    return path


def build_ratio_metric_panel(df: pd.DataFrame, figures_dir: Path, model_tag: str, model_label: str) -> None:
    csv_path = figures_dir / f"fig_hn_{model_tag}_ratio_metric_curves_panel.csv"
    png_path = figures_dir / f"fig_hn_{model_tag}_ratio_metric_curves_panel.png"
    df[
        ["ratio_id", "ratio_percent", "spec_at_r995", "spec_at_r990", "prec_at_r990", "ptr_at_r990", "formal_rank"]
    ].to_csv(csv_path, index=False)

    winner = df.sort_values("formal_rank").iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes = axes.flatten()
    x = df["ratio_percent"]
    colors = {
        "spec_at_r995": "#1f77b4",
        "spec_at_r990": "#2ca02c",
        "prec_at_r990": "#d62728",
        "ptr_at_r990": "#9467bd",
    }
    for ax, metric in zip(axes, METRIC_COLUMNS):
        ax.plot(x, df[metric], marker="o", linewidth=2, color=colors[metric])
        ax.scatter(
            [winner["ratio_percent"]],
            [winner[metric]],
            color="#ff8c00",
            s=90,
            marker="*",
            zorder=3,
            label=f"winner: {winner['ratio_id']}",
        )
        ax.axvline(winner["ratio_percent"], color="#ff8c00", linestyle="--", linewidth=1)
        ax.set_title(METRIC_COLUMNS[metric])
        ax.set_xlabel("HN ratio (%)")
        ax.set_ylabel(METRIC_COLUMNS[metric])
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"{model_label}: HN sweep metric panel", fontsize=13)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_best_epoch_plot(df: pd.DataFrame, figures_dir: Path, model_tag: str, model_label: str) -> None:
    csv_path = figures_dir / f"fig_hn_{model_tag}_best_epoch_vs_ratio.csv"
    png_path = figures_dir / f"fig_hn_{model_tag}_best_epoch_vs_ratio.png"
    df[["ratio_id", "ratio_percent", "best_epoch", "formal_rank"]].to_csv(csv_path, index=False)

    winner = df.sort_values("formal_rank").iloc[0]
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.plot(df["ratio_percent"], df["best_epoch"], marker="o", linewidth=2, color="#1f77b4")
    ax.scatter(
        [winner["ratio_percent"]],
        [winner["best_epoch"]],
        color="#ff8c00",
        s=95,
        marker="*",
        zorder=3,
        label=f"winner: {winner['ratio_id']}",
    )
    ax.axvline(winner["ratio_percent"], color="#ff8c00", linestyle="--", linewidth=1)
    ax.set_xlabel("HN ratio (%)")
    ax.set_ylabel("Best epoch")
    ax.set_title(f"{model_label}: best epoch vs. HN ratio")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def pick_representative_ratios(df: pd.DataFrame, ratios: list[str] | None) -> list[str]:
    if ratios:
        return ratios
    winner_ratio = df.sort_values("formal_rank").iloc[0]["ratio_id"]
    max_ratio = df.sort_values("ratio_percent").iloc[-1]["ratio_id"]
    ordered = []
    for ratio in ["hn00", winner_ratio, max_ratio]:
        if ratio not in ordered:
            ordered.append(ratio)
    return ordered


def load_epoch_frame(materials_root: Path, ratio_id: str) -> pd.DataFrame:
    path = materials_root / ratio_id / "epoch_gate_summary.csv"
    df = pd.read_csv(path)
    df["epoch"] = df["epoch"].astype(int)
    for metric in METRIC_COLUMNS:
        df[metric] = df[metric].astype(float)
    df["ratio_id"] = ratio_id
    return df


def build_epoch_dynamics_panel(
    df: pd.DataFrame,
    materials_root: Path,
    figures_dir: Path,
    model_tag: str,
    model_label: str,
    representative_ratios: list[str],
) -> None:
    csv_path = figures_dir / f"fig_hn_{model_tag}_epoch_dynamics_panel.csv"
    png_path = figures_dir / f"fig_hn_{model_tag}_epoch_dynamics_panel.png"

    frames = [load_epoch_frame(materials_root, ratio_id) for ratio_id in representative_ratios]
    long_df = pd.concat(frames, ignore_index=True)
    long_df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes = axes.flatten()
    colors = {"hn00": "#1f77b4", representative_ratios[1]: "#ff8c00", representative_ratios[-1]: "#2ca02c"}
    metric_order = list(METRIC_COLUMNS.keys())
    for ax, metric in zip(axes, metric_order):
        for ratio_id in representative_ratios:
            ratio_df = long_df.loc[long_df["ratio_id"] == ratio_id].sort_values("epoch")
            ax.plot(
                ratio_df["epoch"],
                ratio_df[metric],
                linewidth=1.6,
                label=ratio_id,
                color=colors.get(ratio_id),
            )
        ax.set_title(METRIC_COLUMNS[metric])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(METRIC_COLUMNS[metric])
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        f"{model_label}: representative HN dynamics ({', '.join(representative_ratios)})",
        fontsize=13,
    )
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    materials_root = Path(args.materials_root)
    output_root = Path(args.output_root)
    tables_dir, figures_dir = ensure_dirs(output_root)

    df = compute_rank_and_delta(load_summary(summary_csv))
    write_table_csv(df, tables_dir, args.model_tag)
    build_ratio_metric_panel(df, figures_dir, args.model_tag, args.model_label)
    build_best_epoch_plot(df, figures_dir, args.model_tag, args.model_label)
    representative_ratios = pick_representative_ratios(df, args.representative_ratios)
    build_epoch_dynamics_panel(
        df,
        materials_root,
        figures_dir,
        args.model_tag,
        args.model_label,
        representative_ratios,
    )


if __name__ == "__main__":
    main()
