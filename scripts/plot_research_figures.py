from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pipeline_common import REPO_ROOT


RAW_LABELS = [
    "VA",
    "RB",
    "OB",
    "PF",
    "DE",
    "FS",
    "IS",
    "RO",
    "IN",
    "AF",
    "BE",
    "FO",
    "GR",
    "PH",
    "PB",
    "OS",
    "OP",
    "OK",
    "ND",
]
SOURCE_ACTIVE_LABELS = [label for label in RAW_LABELS if label not in {"VA", "ND"}]
STRUCTURAL_RAW = ["RB", "OB", "PF", "DE", "FS", "IS", "IN"]
FUNCTIONAL_RAW = ["RO", "AF", "BE", "FO"]
CONSTRUCTION_RAW = ["GR", "PH", "PB", "OS", "OP", "OK"]

RAW_GROUPS = {
    "VA": "Metadata",
    "RB": "Structural",
    "OB": "Structural",
    "PF": "Structural",
    "DE": "Structural",
    "FS": "Structural",
    "IS": "Structural",
    "RO": "Functional",
    "IN": "Structural",
    "AF": "Functional",
    "BE": "Functional",
    "FO": "Functional",
    "GR": "Construction",
    "PH": "Construction",
    "PB": "Construction",
    "OS": "Construction",
    "OP": "Construction",
    "OK": "Construction",
    "ND": "Normal",
}

GROUP_COLORS = {
    "Structural": "#C44E52",
    "Functional": "#4C72B0",
    "Construction": "#55A868",
    "Metadata": "#8172B2",
    "Normal": "#CCB974",
}

HLA3_MAP = {
    "Normal": ["ND"],
    "StructuralDefect": STRUCTURAL_RAW,
    "FunctionalDefect": FUNCTIONAL_RAW,
}

HLA6_MAP = {
    "Normal": ["ND"],
    "WallDamage": ["RB", "OB", "PF"],
    "JointAnomaly": ["FS", "IS"],
    "Deformation": ["DE"],
    "DepositAttachment": ["AF", "BE"],
    "Roots": ["RO"],
}

STRUCT6_MAP = {
    "CrackBreak": ["RB"],
    "SurfaceDamage": ["OB", "PF"],
    "Deformation": ["DE"],
    "JointDislocation": ["FS"],
    "Intrusion": ["IS"],
    "Infiltration": ["IN"],
}

STRUCT6_COLORS = {
    "CrackBreak": "#B22222",
    "SurfaceDamage": "#D95F02",
    "Deformation": "#7570B3",
    "JointDislocation": "#1B9E77",
    "Intrusion": "#E7298A",
    "Infiltration": "#4E79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reusable experiment figures for the sewer research pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sewerml = subparsers.add_parser("sewerml", help="Plot SewerML annotation statistics and alignment figures.")
    sewerml.add_argument(
        "--annotations-dir",
        type=Path,
        default=REPO_ROOT / "data" / "sewerml" / "annotations",
        help="Directory containing SewerML_Train.csv and SewerML_Val.csv.",
    )
    sewerml.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "figures" / "sewerml",
        help="Directory for generated SewerML figures.",
    )
    sewerml.add_argument(
        "--splits",
        nargs="+",
        default=["Train", "Val"],
        choices=["Train", "Val", "Test"],
        help="Official splits to load.",
    )
    sewerml.add_argument("--dpi", type=int, default=220, help="Figure export DPI.")

    cam = subparsers.add_parser("cam-review", help="Plot CAM review summaries from the review template CSV.")
    cam.add_argument(
        "--review-csv",
        type=Path,
        default=REPO_ROOT / "research" / "cam_review_template.csv",
        help="CSV exported from the CAM review table.",
    )
    cam.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "figures" / "cam_review",
        help="Directory for generated CAM review figures.",
    )
    cam.add_argument("--dpi", type=int, default=220, help="Figure export DPI.")

    metrics = subparsers.add_parser("train-metrics", help="Plot YOLO training curves from a results.csv file.")
    metrics.add_argument("--results-csv", type=Path, required=True, help="Path to YOLO results.csv.")
    metrics.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "figures" / "training" / "training_curves.png",
        help="Output image path.",
    )
    metrics.add_argument("--title", default="", help="Optional figure title.")
    metrics.add_argument("--dpi", type=int, default=220, help="Figure export DPI.")

    return parser.parse_args()


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "#F7F5F2",
            "axes.facecolor": "#F7F5F2",
            "savefig.facecolor": "#F7F5F2",
            "axes.edgecolor": "#C8C3BA",
            "grid.color": "#D8D3CB",
            "axes.labelcolor": "#252320",
            "xtick.color": "#252320",
            "ytick.color": "#252320",
            "text.color": "#252320",
            "axes.titleweight": "bold",
            "axes.titlesize": 17,
        }
    )


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_sewerml_splits(annotations_dir: Path, splits: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in splits:
        csv_path = annotations_dir / f"SewerML_{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing SewerML annotation file: {csv_path}")
        frame = pd.read_csv(csv_path, encoding="utf-8-sig")
        frame["Split"] = split
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def positive_counts(frame: pd.DataFrame, labels: list[str]) -> pd.Series:
    return frame[labels].fillna(0).astype(int).sum(axis=0).sort_values(ascending=False)


def count_any(frame: pd.DataFrame, labels: list[str]) -> int:
    return int(frame[labels].fillna(0).astype(int).any(axis=1).sum())


def aligned_counts(frame: pd.DataFrame, mapping: dict[str, list[str]]) -> pd.Series:
    counts = {}
    for name, labels in mapping.items():
        if name == "Normal":
            counts[name] = int(frame["ND"].fillna(0).astype(int).sum())
        else:
            counts[name] = count_any(frame, labels)
    return pd.Series(counts)


def build_struct6_binary(frame: pd.DataFrame) -> pd.DataFrame:
    data = {}
    for name, labels in STRUCT6_MAP.items():
        data[name] = frame[labels].fillna(0).astype(int).any(axis=1).astype(int)
    return pd.DataFrame(data)


def annotate_barh(ax: plt.Axes, values: pd.Series) -> None:
    max_value = max(float(values.max()), 1.0)
    for patch, value in zip(ax.patches, values.values):
        ax.text(
            patch.get_width() + max_value * 0.01,
            patch.get_y() + patch.get_height() / 2,
            f"{int(value):,}",
            va="center",
            ha="left",
            fontsize=10,
        )


def plot_raw_label_distribution(frame: pd.DataFrame, output_path: Path, dpi: int) -> None:
    counts = positive_counts(frame, RAW_LABELS)
    colors = [GROUP_COLORS[RAW_GROUPS[label]] for label in counts.index]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.0)
    ax.invert_yaxis()
    ax.set_title("SewerML Raw Label Distribution (Train + Val)")
    ax.set_xlabel("Positive label occurrences")
    ax.set_ylabel("Raw labels")
    annotate_barh(ax, counts)

    legend_items = []
    for group_name, color in GROUP_COLORS.items():
        legend_items.append(plt.Rectangle((0, 0), 1, 1, color=color, label=group_name))
    ax.legend(handles=legend_items, frameon=False, ncol=3, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_alignment_summary(frame: pd.DataFrame, output_path: Path, dpi: int) -> None:
    cls3_counts = aligned_counts(frame, HLA3_MAP)
    cls6_counts = aligned_counts(frame, HLA6_MAP)
    struct6_counts = aligned_counts(frame, STRUCT6_MAP)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    summary_specs = [
        ("Three-Class Alignment", cls3_counts, sns.color_palette("crest", n_colors=len(cls3_counts))),
        ("Six-Class Alignment", cls6_counts, sns.color_palette("flare", n_colors=len(cls6_counts))),
        (
            "Final Structural-6 Source View",
            struct6_counts,
            [STRUCT6_COLORS[name] for name in struct6_counts.index],
        ),
    ]

    for ax, (title, counts, colors) in zip(axes, summary_specs):
        total = counts.sum()
        ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.0)
        ax.set_title(title)
        ax.set_ylabel("Images / positive groups")
        ax.tick_params(axis="x", rotation=25)
        for idx, value in enumerate(counts.values):
            ratio = 100.0 * float(value) / max(float(total), 1.0)
            ax.text(idx, value, f"{int(value):,}\n{ratio:.1f}%", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Hierarchical Label Alignment Summary", y=1.02, fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_struct6_cooccurrence(frame: pd.DataFrame, output_path: Path, dpi: int) -> None:
    struct6_binary = build_struct6_binary(frame)
    matrix = struct6_binary.to_numpy(dtype=float)
    counts = matrix.sum(axis=0)

    conditional = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=float)
    for row_idx in range(matrix.shape[1]):
        positives = counts[row_idx]
        if positives == 0:
            continue
        conditional[row_idx, :] = matrix[matrix[:, row_idx] == 1].mean(axis=0) * 100.0

    conditional_frame = pd.DataFrame(conditional, index=STRUCT6_MAP.keys(), columns=STRUCT6_MAP.keys())

    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        conditional_frame,
        cmap=sns.color_palette(["#F7F5F2", "#E5C07B", "#D95F02", "#7F2704"], as_cmap=True),
        annot=True,
        fmt=".1f",
        linewidths=1.0,
        linecolor="#F7F5F2",
        cbar_kws={"label": "P(column | row) %"},
        ax=ax,
    )
    ax.set_title("Structural-6 Conditional Co-Occurrence")
    ax.set_xlabel("Also present defect")
    ax.set_ylabel("Conditioned defect")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_waterlevel_profile(frame: pd.DataFrame, output_path: Path, dpi: int) -> None:
    relevant = frame.copy()
    relevant["WaterLevel"] = pd.to_numeric(relevant["WaterLevel"], errors="coerce")
    summary = (
        relevant.groupby("WaterLevel")
        .apply(
            lambda part: pd.Series(
                {
                    "StructuralDefect": count_any(part, STRUCTURAL_RAW),
                    "FunctionalDefect": count_any(part, FUNCTIONAL_RAW),
                    "Normal": int(part["ND"].fillna(0).astype(int).sum()),
                }
            )
        )
        .reset_index()
        .sort_values("WaterLevel")
    )

    long_summary = summary.melt(id_vars="WaterLevel", var_name="Category", value_name="Count")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=long_summary,
        x="WaterLevel",
        y="Count",
        hue="Category",
        style="Category",
        markers=True,
        dashes=False,
        linewidth=2.6,
        palette={"StructuralDefect": "#C44E52", "FunctionalDefect": "#4C72B0", "Normal": "#CCB974"},
        ax=ax,
    )
    ax.set_title("Water-Level Profile vs Label Groups")
    ax.set_xlabel("Annotated water level")
    ax.set_ylabel("Images")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_group_share_donut(frame: pd.DataFrame, output_path: Path, dpi: int) -> None:
    group_counts = pd.Series(
        {
            "Structural": int(frame[STRUCTURAL_RAW].fillna(0).astype(int).sum().sum()),
            "Functional": int(frame[FUNCTIONAL_RAW].fillna(0).astype(int).sum().sum()),
            "Construction": int(frame[CONSTRUCTION_RAW].fillna(0).astype(int).sum().sum()),
            "Metadata": int(frame["VA"].fillna(0).astype(int).sum()),
            "Normal": int(frame["ND"].fillna(0).astype(int).sum()),
        }
    )

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    colors = [GROUP_COLORS[name] for name in group_counts.index]
    wedges, _texts, autotexts = ax.pie(
        group_counts.values,
        labels=group_counts.index,
        colors=colors,
        startangle=90,
        wedgeprops={"width": 0.44, "edgecolor": "#F7F5F2", "linewidth": 2},
        autopct=lambda pct: f"{pct:.1f}%",
        pctdistance=0.78,
    )
    for autotext in autotexts:
        autotext.set_color("#252320")
        autotext.set_fontsize(11)
        autotext.set_weight("bold")
    ax.set_title("Raw Label Group Share", pad=20)
    total = int(group_counts.sum())
    ax.text(0, 0, f"{total:,}\noccurrences", ha="center", va="center", fontsize=17, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_multilabel_cardinality(frame: pd.DataFrame, output_path: Path, dpi: int) -> None:
    working = frame.copy()
    working["positive_count"] = working[SOURCE_ACTIVE_LABELS].fillna(0).astype(int).sum(axis=1)
    summary = (
        working.groupby(["Split", "positive_count"])
        .size()
        .reset_index(name="images")
        .sort_values(["Split", "positive_count"])
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    sns.lineplot(
        data=summary,
        x="positive_count",
        y="images",
        hue="Split",
        style="Split",
        markers=True,
        dashes=False,
        linewidth=2.5,
        palette={"Train": "#4C72B0", "Val": "#C44E52", "Test": "#55A868"},
        ax=ax,
    )
    ax.set_title("Multi-Label Cardinality Per Image")
    ax.set_xlabel("Number of positive raw labels in one image")
    ax.set_ylabel("Images")
    ax.legend(frameon=False)

    stats = working.groupby("Split")["positive_count"].mean().to_dict()
    subtitle = " | ".join(f"{split} mean={value:.2f}" for split, value in stats.items())
    ax.text(0.99, 0.98, subtitle, transform=ax.transAxes, ha="right", va="top", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_struct6_by_split(frame: pd.DataFrame, output_path: Path, dpi: int) -> None:
    records = []
    for split, part in frame.groupby("Split"):
        for class_name, labels in STRUCT6_MAP.items():
            records.append(
                {
                    "Split": split,
                    "Class": class_name,
                    "Count": count_any(part, labels),
                }
            )
    summary = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    sns.barplot(
        data=summary,
        x="Class",
        y="Count",
        hue="Split",
        palette={"Train": "#4C72B0", "Val": "#C44E52", "Test": "#55A868"},
        edgecolor="white",
        ax=ax,
    )
    ax.set_title("Structural-6 Class Balance By Split")
    ax.set_xlabel("")
    ax.set_ylabel("Images")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False, title="")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_top_label_pairs(frame: pd.DataFrame, output_path: Path, dpi: int, top_n: int = 12) -> None:
    active = frame[SOURCE_ACTIVE_LABELS].fillna(0).astype(int)
    pair_counts: list[tuple[str, int]] = []
    for idx, left in enumerate(SOURCE_ACTIVE_LABELS):
        for right in SOURCE_ACTIVE_LABELS[idx + 1 :]:
            count = int((active[left] & active[right]).sum())
            if count > 0:
                pair_counts.append((f"{left} + {right}", count))

    pair_counts.sort(key=lambda item: item[1], reverse=True)
    top_pairs = pd.Series(dict(pair_counts[:top_n]))
    top_pairs = top_pairs.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.barh(top_pairs.index, top_pairs.values, color="#4C72B0", edgecolor="white", linewidth=1.0)
    ax.set_title(f"Top {len(top_pairs)} Raw Label Co-Occurrence Pairs")
    ax.set_xlabel("Images")
    ax.set_ylabel("Label pair")
    annotate_barh(ax, top_pairs)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_sewerml_figures(args: argparse.Namespace) -> None:
    output_dir = ensure_output_dir(args.output_dir)
    frame = load_sewerml_splits(args.annotations_dir, args.splits)

    plot_raw_label_distribution(frame, output_dir / "01_raw_label_distribution.png", args.dpi)
    plot_alignment_summary(frame, output_dir / "02_alignment_summary.png", args.dpi)
    plot_struct6_cooccurrence(frame, output_dir / "03_struct6_cooccurrence.png", args.dpi)
    plot_waterlevel_profile(frame, output_dir / "04_waterlevel_profile.png", args.dpi)
    plot_group_share_donut(frame, output_dir / "05_group_share_donut.png", args.dpi)
    plot_multilabel_cardinality(frame, output_dir / "06_multilabel_cardinality.png", args.dpi)
    plot_struct6_by_split(frame, output_dir / "07_struct6_by_split.png", args.dpi)
    plot_top_label_pairs(frame, output_dir / "08_top_label_pairs.png", args.dpi)

    print(f"[done] SewerML figures written to {output_dir}")


def load_cam_review(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No CAM review rows found in {path}")
    frame = pd.DataFrame(rows)
    numeric_cols = ["inspected_total", "direct_use", "minor_edit_use", "unusable", "usable_rate", "cam_threshold"]
    for column in numeric_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["usable_rate_pct"] = np.where(
        frame["inspected_total"] > 0,
        (frame["direct_use"] + frame["minor_edit_use"]) / frame["inspected_total"] * 100.0,
        0.0,
    )
    frame["unusable_pct"] = np.where(
        frame["inspected_total"] > 0,
        frame["unusable"] / frame["inspected_total"] * 100.0,
        0.0,
    )
    frame["direct_pct"] = np.where(
        frame["inspected_total"] > 0,
        frame["direct_use"] / frame["inspected_total"] * 100.0,
        0.0,
    )
    frame["minor_pct"] = np.where(
        frame["inspected_total"] > 0,
        frame["minor_edit_use"] / frame["inspected_total"] * 100.0,
        0.0,
    )
    return frame


def plot_cam_review(frame: pd.DataFrame, output_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.8))
    x = np.arange(len(frame))
    ax.bar(x, frame["direct_pct"], label="Direct Use", color="#1B9E77")
    ax.bar(x, frame["minor_pct"], bottom=frame["direct_pct"], label="Minor Edit", color="#E6AB02")
    ax.bar(
        x,
        frame["unusable_pct"],
        bottom=frame["direct_pct"] + frame["minor_pct"],
        label="Unusable",
        color="#C44E52",
    )

    ax.axhline(70, color="#1F78B4", linestyle="--", linewidth=1.5, label="CAM main-route line")
    ax.axhline(40, color="#6A3D9A", linestyle=":", linewidth=1.5, label="CAM/manual switch line")

    ax.set_xticks(x)
    ax.set_xticklabels(frame["class_name"], rotation=20)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Review outcome share (%)")
    ax.set_title("CAM Review Summary")
    ax.legend(frameon=False, ncol=2, loc="upper right")

    for idx, row in frame.iterrows():
        ax.text(
            idx,
            min(row["usable_rate_pct"] + 3, 98),
            f"{row['usable_rate_pct']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_cam_review_figure(args: argparse.Namespace) -> None:
    output_dir = ensure_output_dir(args.output_dir)
    frame = load_cam_review(args.review_csv)
    plot_cam_review(frame, output_dir / "cam_review_summary.png", args.dpi)
    print(f"[done] CAM review figure written to {output_dir}")


def normalize_results_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def preferred_metric_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    available = [column for column in preferred if column in frame.columns]
    if available:
        return available

    numeric_cols = []
    for column in frame.columns:
        if column.lower() == "epoch":
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().any():
            numeric_cols.append(column)
    return numeric_cols[:8]


def plot_training_metrics(frame: pd.DataFrame, output_path: Path, title: str, dpi: int) -> None:
    frame = normalize_results_columns(frame)
    if "epoch" not in frame.columns:
        frame.insert(0, "epoch", np.arange(len(frame)))

    metric_columns = preferred_metric_columns(frame)
    if not metric_columns:
        raise ValueError("No numeric metric columns were found in the results.csv file.")

    ncols = 2
    nrows = int(np.ceil(len(metric_columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, column in zip(axes_flat, metric_columns):
        values = pd.to_numeric(frame[column], errors="coerce")
        ax.plot(frame["epoch"], values, color="#4C72B0", linewidth=2.3)
        ax.fill_between(frame["epoch"], values, alpha=0.16, color="#4C72B0")
        ax.set_title(column)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")

    for ax in axes_flat[len(metric_columns) :]:
        ax.axis("off")

    if title:
        fig.suptitle(title, y=1.01, fontsize=20, fontweight="bold")
    else:
        fig.suptitle("YOLO Training Curves", y=1.01, fontsize=20, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_training_figure(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.results_csv)
    plot_training_metrics(frame, args.output, args.title, args.dpi)
    print(f"[done] Training figure written to {args.output}")


def main() -> None:
    args = parse_args()
    set_plot_style()

    if args.command == "sewerml":
        generate_sewerml_figures(args)
    elif args.command == "cam-review":
        generate_cam_review_figure(args)
    else:
        generate_training_figure(args)


if __name__ == "__main__":
    main()
