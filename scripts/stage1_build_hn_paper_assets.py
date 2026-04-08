from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "research" / "results" / "stage1_formal"
MATERIALS_ROOT = REPO_ROOT / "research" / "materials" / "stage1_formal"
OUTPUT_ROOT = RESULTS_ROOT / "gate_hn_paper"

DERIVED_DIR = OUTPUT_ROOT / "derived"
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"
APPENDIX_DIR = OUTPUT_ROOT / "appendix"
CAPTIONS_DIR = OUTPUT_ROOT / "captions"
MANIFESTS_DIR = OUTPUT_ROOT / "manifests"

HN_M_SUMMARY = RESULTS_ROOT / "gate_hn_m_sweep" / "hn_sweep_summary.csv"
HN_X_SUMMARY = RESULTS_ROOT / "gate_hn_x_crosscheck" / "hn_sweep_summary.csv"
HN_OVERVIEW = RESULTS_ROOT / "gate_hn_overview" / "stage1_formal_hn_overview.csv"
HN_CROSS_COMPARE = RESULTS_ROOT / "gate_hn_overview" / "table_hn_cross_capacity_compare.csv"
HN_REGISTRY = RESULTS_ROOT / "gate_hn_overview" / "hn_best_checkpoint_registry.csv"
BASELINE_SUMMARY = RESULTS_ROOT / "gate_capacity" / "binary_gate_capacity_summary.csv"
INGEST_MANIFEST = RESULTS_ROOT / "gate_hn_overview" / "ingest_manifest.json"

HN_M_MATERIALS = MATERIALS_ROOT / "gate_hn_m_sweep"
HN_X_MATERIALS = MATERIALS_ROOT / "gate_hn_x_crosscheck"
HN_ASSETS = MATERIALS_ROOT / "gate_hn_assets" / "yolo11m_train_normal_scores"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    }
)


@dataclass
class CaptionSpec:
    stem: str
    question: str
    sources: list[str]
    rule: str
    finding: str
    limitation: str


def ensure_dirs() -> None:
    for path in (OUTPUT_ROOT, DERIVED_DIR, TABLES_DIR, FIGURES_DIR, APPENDIX_DIR, CAPTIONS_DIR, MANIFESTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def fmt4(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.4f}"


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def rank_gate_rows(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.sort_values(
        by=["spec_at_r995", "spec_at_r990", "prec_at_r990", "ptr_at_r990", "ratio_percent"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["formal_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_md_table(path: Path, title: str, df: pd.DataFrame, notes: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [str(col) for col in df.columns]
    rows = []
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, (float, np.floating)):
                values.append(fmt4(value))
            else:
                values.append(str(value))
        rows.append(values)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        table_lines.append("| " + " | ".join(row) + " |")
    lines = [f"# {title}", "", *table_lines]
    if notes:
        lines.extend(["", "Notes:"])
        lines.extend([f"- {note}" for note in notes])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_caption(spec: CaptionSpec) -> None:
    path = CAPTIONS_DIR / f"{spec.stem}.md"
    content = "\n".join(
        [
            f"# {spec.stem}",
            "",
            f"1. This asset answers: {spec.question}",
            "2. Source files:",
            *[f"   - {item}" for item in spec.sources],
            f"3. Ranking/selection rule: {spec.rule}",
            f"4. Key finding: {spec.finding}",
            f"5. Limitation: {spec.limitation}",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def read_csv_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_inputs() -> dict[str, Any]:
    ingest = json.loads(INGEST_MANIFEST.read_text(encoding="utf-8"))
    return {
        "m": read_csv_file(HN_M_SUMMARY),
        "x": read_csv_file(HN_X_SUMMARY),
        "overview": read_csv_file(HN_OVERVIEW),
        "cross": read_csv_file(HN_CROSS_COMPARE),
        "registry": read_csv_file(HN_REGISTRY),
        "baseline": read_csv_file(BASELINE_SUMMARY),
        "ingest": ingest,
        "train_normal_scores": read_csv_file(HN_ASSETS / "train_normal_scores.csv"),
        "top_false_positive_normals": read_csv_file(HN_ASSETS / "top_false_positive_normals.csv"),
        "assets_summary": json.loads((HN_ASSETS / "summary.json").read_text(encoding="utf-8")),
    }


def build_derived(inputs: dict[str, Any]) -> dict[str, pd.DataFrame]:
    m_df = inputs["m"].copy()
    x_df = inputs["x"].copy()
    baseline_df = inputs["baseline"].copy()

    m_ranked = rank_gate_rows(m_df)
    rank_map = m_ranked.set_index("ratio_id")["formal_rank"].to_dict()
    winner_ratio = str(m_ranked.iloc[0]["ratio_id"])
    anchor = m_df.loc[m_df["ratio_id"] == "hn00"].iloc[0]
    m_df["formal_rank"] = m_df["ratio_id"].map(rank_map)
    m_df["hn_ratio"] = m_df["ratio_id"]
    m_df["delta_Spec@R99.5_vs_hn00"] = m_df["spec_at_r995"] - float(anchor["spec_at_r995"])
    m_df["delta_Spec@R99.0_vs_hn00"] = m_df["spec_at_r990"] - float(anchor["spec_at_r990"])
    m_df["delta_Prec@R99.0_vs_hn00"] = m_df["prec_at_r990"] - float(anchor["prec_at_r990"])
    m_df["delta_PTR@R99.0_vs_hn00"] = m_df["ptr_at_r990"] - float(anchor["ptr_at_r990"])
    m_df["PTR_improved_flag"] = m_df["delta_PTR@R99.0_vs_hn00"] < 0
    m_df["winner_flag"] = m_df["ratio_id"] == winner_ratio
    m_df["delta_best_epoch_vs_hn00"] = m_df["best_epoch"] - int(anchor["best_epoch"])

    x_anchor = x_df.loc[x_df["ratio_id"] == "hn00"].iloc[0]
    x_df["delta_vs_hn00"] = (
        "delta Spec@R99.5="
        + (x_df["spec_at_r995"] - float(x_anchor["spec_at_r995"])).map(fmt4)
        + "; delta Spec@R99.0="
        + (x_df["spec_at_r990"] - float(x_anchor["spec_at_r990"])).map(fmt4)
        + "; delta Prec@R99.0="
        + (x_df["prec_at_r990"] - float(x_anchor["prec_at_r990"])).map(fmt4)
        + "; delta PTR@R99.0="
        + (x_df["ptr_at_r990"] - float(x_anchor["ptr_at_r990"])).map(fmt4)
    )

    baseline_lookup = baseline_df.set_index("model")
    yolo11m_baseline = baseline_lookup.loc["yolo11m-cls"]
    yolo11x_baseline = baseline_lookup.loc["yolo11x-cls"]
    yolo11m_best = m_ranked.iloc[0]
    yolo11x_hn02 = x_df.loc[x_df["ratio_id"] == "hn02"].iloc[0]

    bridge_rows = [
        {
            "setting": "yolo11m baseline (hn00)",
            "best_epoch": int(yolo11m_baseline["best_epoch"]),
            "Spec@R99.5": float(yolo11m_baseline["spec_at_r995"]),
            "Spec@R99.0": float(yolo11m_baseline["spec_at_r990"]),
            "Prec@R99.0": float(yolo11m_baseline["prec_at_r990"]),
            "PTR@R99.0": float(yolo11m_baseline["ptr_at_r990"]),
            "delta_vs_same_model_hn00": "baseline anchor",
        },
        {
            "setting": f"yolo11m + {yolo11m_best['ratio_id']}",
            "best_epoch": int(yolo11m_best["best_epoch"]),
            "Spec@R99.5": float(yolo11m_best["spec_at_r995"]),
            "Spec@R99.0": float(yolo11m_best["spec_at_r990"]),
            "Prec@R99.0": float(yolo11m_best["prec_at_r990"]),
            "PTR@R99.0": float(yolo11m_best["ptr_at_r990"]),
            "delta_vs_same_model_hn00": (
                f"delta Spec@R99.5={fmt4(float(yolo11m_best['spec_at_r995']) - float(yolo11m_baseline['spec_at_r995']))}; "
                f"delta Spec@R99.0={fmt4(float(yolo11m_best['spec_at_r990']) - float(yolo11m_baseline['spec_at_r990']))}; "
                f"delta Prec@R99.0={fmt4(float(yolo11m_best['prec_at_r990']) - float(yolo11m_baseline['prec_at_r990']))}; "
                f"delta PTR@R99.0={fmt4(float(yolo11m_best['ptr_at_r990']) - float(yolo11m_baseline['ptr_at_r990']))}"
            ),
        },
        {
            "setting": "yolo11x baseline (hn00)",
            "best_epoch": int(yolo11x_baseline["best_epoch"]),
            "Spec@R99.5": float(yolo11x_baseline["spec_at_r995"]),
            "Spec@R99.0": float(yolo11x_baseline["spec_at_r990"]),
            "Prec@R99.0": float(yolo11x_baseline["prec_at_r990"]),
            "PTR@R99.0": float(yolo11x_baseline["ptr_at_r990"]),
            "delta_vs_same_model_hn00": "baseline anchor",
        },
        {
            "setting": "yolo11x + hn02",
            "best_epoch": int(yolo11x_hn02["best_epoch"]),
            "Spec@R99.5": float(yolo11x_hn02["spec_at_r995"]),
            "Spec@R99.0": float(yolo11x_hn02["spec_at_r990"]),
            "Prec@R99.0": float(yolo11x_hn02["prec_at_r990"]),
            "PTR@R99.0": float(yolo11x_hn02["ptr_at_r990"]),
            "delta_vs_same_model_hn00": (
                f"delta Spec@R99.5={fmt4(float(yolo11x_hn02['spec_at_r995']) - float(yolo11x_baseline['spec_at_r995']))}; "
                f"delta Spec@R99.0={fmt4(float(yolo11x_hn02['spec_at_r990']) - float(yolo11x_baseline['spec_at_r990']))}; "
                f"delta Prec@R99.0={fmt4(float(yolo11x_hn02['prec_at_r990']) - float(yolo11x_baseline['prec_at_r990']))}; "
                f"delta PTR@R99.0={fmt4(float(yolo11x_hn02['ptr_at_r990']) - float(yolo11x_baseline['ptr_at_r990']))}"
            ),
        },
    ]

    joined = inputs["overview"].copy().merge(
        inputs["registry"][["model", "ratio_id", "exported_checkpoint_path", "checkpoint_exists"]],
        on=["model", "ratio_id"],
        how="left",
    )
    return {
        "hn_m_main_derived": m_df,
        "hn_x_main_derived": x_df,
        "hn_overview_joined": joined,
        "hn_anchor_baseline_joined": pd.DataFrame(bridge_rows),
        "hn_m_ranked": m_ranked,
    }


def export_derived(derived: dict[str, pd.DataFrame]) -> None:
    for name, df in derived.items():
        save_csv(DERIVED_DIR / f"{name}.csv", df)


def build_main_tables(derived: dict[str, pd.DataFrame]) -> list[CaptionSpec]:
    captions: list[CaptionSpec] = []
    m_df = derived["hn_m_main_derived"].copy()
    m_table = m_df[
        [
            "model",
            "hn_ratio",
            "best_epoch",
            "spec_at_r995",
            "spec_at_r990",
            "prec_at_r990",
            "ptr_at_r990",
            "formal_rank",
            "delta_Spec@R99.5_vs_hn00",
            "delta_Spec@R99.0_vs_hn00",
            "delta_Prec@R99.0_vs_hn00",
            "delta_PTR@R99.0_vs_hn00",
            "winner_flag",
            "PTR_improved_flag",
        ]
    ].rename(
        columns={
            "spec_at_r995": "Spec@R99.5",
            "spec_at_r990": "Spec@R99.0",
            "prec_at_r990": "Prec@R99.0",
            "ptr_at_r990": "PTR@R99.0",
        }
    )
    save_csv(TABLES_DIR / "table_hn_m_main_ranking.csv", m_table)
    save_md_table(TABLES_DIR / "table_hn_m_main_ranking.md", "table_hn_m_main_ranking", m_table)
    captions.append(
        CaptionSpec(
            "table_hn_m_main_ranking",
            "Which HN ratio is the formal winner for yolo11m under the recall-constrained stage-1 gate objective.",
            [rel(HN_M_SUMMARY)],
            "Rows are ranked by Spec@R99.5, Spec@R99.0, Prec@R99.0, and PTR@R99.0.",
            f"The current formal winner is {derived['hn_m_ranked'].iloc[0]['ratio_id']}, indicating a non-monotonic sweet spot rather than a monotonic ratio effect.",
            "This table operates on formal summary rows rather than raw PT checkpoint archives.",
        )
    )

    delta_table = m_df[
        [
            "hn_ratio",
            "best_epoch",
            "delta_Spec@R99.5_vs_hn00",
            "delta_Spec@R99.0_vs_hn00",
            "delta_Prec@R99.0_vs_hn00",
            "delta_PTR@R99.0_vs_hn00",
            "delta_best_epoch_vs_hn00",
        ]
    ].rename(
        columns={
            "delta_Spec@R99.5_vs_hn00": "delta_Spec@R99.5",
            "delta_Spec@R99.0_vs_hn00": "delta_Spec@R99.0",
            "delta_Prec@R99.0_vs_hn00": "delta_Prec@R99.0",
            "delta_PTR@R99.0_vs_hn00": "delta_PTR@R99.0",
        }
    )
    save_csv(TABLES_DIR / "table_hn_m_delta_vs_hn00_compact.csv", delta_table)
    save_md_table(TABLES_DIR / "table_hn_m_delta_vs_hn00_compact.md", "table_hn_m_delta_vs_hn00_compact", delta_table)
    captions.append(
        CaptionSpec(
            "table_hn_m_delta_vs_hn00_compact",
            "How much each HN ratio changes the formal gate metrics relative to the no-HN anchor.",
            [rel(HN_M_SUMMARY)],
            "All deltas are computed as absolute differences against hn00.",
            "The delta view makes the HN gain profile interpretable without repeating the raw values.",
            "Positive delta on PTR indicates a worse pass-through rate because PTR is lower-is-better.",
        )
    )

    bridge_df = derived["hn_anchor_baseline_joined"]
    save_csv(TABLES_DIR / "table_hn_baseline_vs_hn_anchor_compare.csv", bridge_df)
    save_md_table(TABLES_DIR / "table_hn_baseline_vs_hn_anchor_compare.md", "table_hn_baseline_vs_hn_anchor_compare", bridge_df)
    captions.append(
        CaptionSpec(
            "table_hn_baseline_vs_hn_anchor_compare",
            "How the HN results connect back to the already-completed baseline capacity scan.",
            [rel(BASELINE_SUMMARY), rel(HN_M_SUMMARY), rel(HN_X_SUMMARY)],
            "Baseline rows come from the completed formal capacity scan; HN rows come from the formal HN sweep summaries.",
            "This bridge table shows whether HN improves the same model relative to its own hn00 anchor instead of reopening the backbone selection question.",
            "yolo11x is only a light cross-capacity check because only hn00 and hn02 are available.",
        )
    )

    x_df = derived["hn_x_main_derived"].copy()
    x_table = x_df[
        ["model", "ratio_id", "best_epoch", "spec_at_r995", "spec_at_r990", "prec_at_r990", "ptr_at_r990", "delta_vs_hn00"]
    ].rename(
        columns={
            "ratio_id": "hn_ratio",
            "spec_at_r995": "Spec@R99.5",
            "spec_at_r990": "Spec@R99.0",
            "prec_at_r990": "Prec@R99.0",
            "ptr_at_r990": "PTR@R99.0",
        }
    )
    save_csv(TABLES_DIR / "table_hn_x_crosscheck_main.csv", x_table)
    save_md_table(TABLES_DIR / "table_hn_x_crosscheck_main.md", "table_hn_x_crosscheck_main", x_table)
    captions.append(
        CaptionSpec(
            "table_hn_x_crosscheck_main",
            "Whether the HN direction observed on the main model remains directionally consistent on the second model.",
            [rel(HN_X_SUMMARY)],
            "This table is a light cross-capacity validation and does not claim a full sweep on yolo11x.",
            "The yolo11x rows show only hn00 and hn02, which is sufficient for a directionality check but not for a full sweet-spot search.",
            "No full x-side ratio sweep is available in the current working set.",
        )
    )
    return captions


def _save_figure(
    fig: plt.Figure,
    stem: str,
    source_df: pd.DataFrame,
    captions: list[CaptionSpec],
    question: str,
    sources: list[str],
    finding: str,
    limitation: str,
    rule: str,
) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    save_csv(FIGURES_DIR / f"{stem}.csv", source_df)
    captions.append(CaptionSpec(stem, question, sources, rule, finding, limitation))


def load_epoch_summary(root: Path, ratio_id: str) -> pd.DataFrame:
    return read_csv_file(root / ratio_id / "epoch_gate_summary.csv")


def build_main_figures(derived: dict[str, pd.DataFrame]) -> list[CaptionSpec]:
    captions: list[CaptionSpec] = []
    m_df = derived["hn_m_main_derived"].copy().sort_values("ratio_percent")
    winner = derived["hn_m_ranked"].iloc[0]
    winner_ratio = str(winner["ratio_id"])
    metrics = [
        ("spec_at_r995", "Spec@R99.5", False),
        ("spec_at_r990", "Spec@R99.0", False),
        ("prec_at_r990", "Prec@R99.0", False),
        ("ptr_at_r990", "PTR@R99.0", True),
    ]

    panel_fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    for ax, (metric, label, lower_better) in zip(axes.flat, metrics):
        ax.plot(m_df["ratio_percent"], m_df[metric], marker="o", linewidth=2, color="#204b57")
        ax.scatter([int(winner["ratio_percent"])], [float(winner[metric])], color="#b6400b", s=45, label=f"winner: {winner_ratio}", zorder=5)
        ax.scatter([0], [float(m_df.loc[m_df["ratio_id"] == "hn00", metric].iloc[0])], color="#2a9d8f", s=40, label="hn00", zorder=5)
        ax.set_title(f"{label} vs HN ratio")
        ax.set_xlabel("HN ratio (%)")
        ax.set_ylabel(label + (" (lower is better)" if lower_better else ""))
        ax.legend(loc="best")
    _save_figure(
        panel_fig,
        "fig_hn_m_ratio_metric_curves_panel",
        m_df[["ratio_id", "ratio_percent", *[m for m, _, _ in metrics], "formal_rank"]],
        captions,
        "How the four formal gate metrics evolve across the yolo11m HN ratio sweep.",
        [rel(HN_M_SUMMARY)],
        f"The sweep exhibits a non-monotonic profile with {winner_ratio} as the current formal winner.",
        "This panel summarizes per-ratio winners and does not visualize per-epoch trajectories.",
        "Formal ranking follows Spec@R99.5 > Spec@R99.0 > Prec@R99.0 > PTR@R99.0.",
    )

    for metric, label, lower_better in metrics:
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        ax.plot(m_df["ratio_percent"], m_df[metric], marker="o", linewidth=2, color="#204b57")
        ax.scatter([int(winner["ratio_percent"])], [float(winner[metric])], color="#b6400b", s=45, zorder=5)
        ax.scatter([0], [float(m_df.loc[m_df["ratio_id"] == "hn00", metric].iloc[0])], color="#2a9d8f", s=40, zorder=5)
        ax.set_xlabel("HN ratio (%)")
        ax.set_ylabel(label + (" (lower is better)" if lower_better else ""))
        ax.set_title(f"{label} vs HN ratio")
        ax.annotate(winner_ratio, (int(winner["ratio_percent"]), float(winner[metric])), xytext=(5, 8), textcoords="offset points")
        stem = {
            "spec_at_r995": "fig_hn_m_ratio_metric_curves_spec995",
            "spec_at_r990": "fig_hn_m_ratio_metric_curves_spec990",
            "prec_at_r990": "fig_hn_m_ratio_metric_curves_prec990",
            "ptr_at_r990": "fig_hn_m_ratio_metric_curves_ptr990",
        }[metric]
        _save_figure(
            fig,
            stem,
            m_df[["ratio_id", "ratio_percent", metric, "formal_rank"]],
            captions,
            f"How {label} changes as the yolo11m HN ratio increases from 0 to 20.",
            [rel(HN_M_SUMMARY)],
            f"{label} does not improve monotonically, which supports the sweet-spot interpretation.",
            "This figure isolates one metric and should be interpreted together with the other three formal metrics.",
            "Rows are interpreted under the formal gate-aware rule; PTR remains lower-is-better.",
        )

        anchor_val = float(m_df.loc[m_df["ratio_id"] == "hn00", metric].iloc[0])
        delta_df = m_df[["ratio_id", "ratio_percent", metric]].copy()
        delta_df["delta"] = delta_df[metric] - anchor_val
        delta_fig, delta_ax = plt.subplots(figsize=(5.6, 3.8))
        delta_ax.axhline(0.0, color="black", linewidth=1)
        delta_ax.plot(delta_df["ratio_percent"], delta_df["delta"], marker="o", linewidth=2, color="#8d6e63")
        delta_ax.set_xlabel("HN ratio (%)")
        delta_ax.set_ylabel(f"delta {label} vs hn00")
        delta_ax.set_title(f"delta {label} vs hn00")
        appendix_stem = {
            "spec_at_r995": "fig_hn_m_ratio_metric_curves_spec995_delta",
            "spec_at_r990": "fig_hn_m_ratio_metric_curves_spec990_delta",
            "prec_at_r990": "fig_hn_m_ratio_metric_curves_prec990_delta",
            "ptr_at_r990": "fig_hn_m_ratio_metric_curves_ptr990_delta",
        }[metric]
        delta_fig.tight_layout()
        delta_fig.savefig(APPENDIX_DIR / f"{appendix_stem}.png", dpi=220, bbox_inches="tight")
        plt.close(delta_fig)
        save_csv(APPENDIX_DIR / f"{appendix_stem}.csv", delta_df[["ratio_id", "ratio_percent", "delta"]])
        captions.append(
            CaptionSpec(
                appendix_stem,
                f"How the delta in {label} behaves relative to the hn00 anchor.",
                [rel(HN_M_SUMMARY)],
                "The hn00 row is treated as the anchor and all values are absolute differences.",
                "The delta view shows the gain pattern more clearly than the raw-value plot.",
                "Only the metric-specific delta is shown here; formal ranking still uses the full four-metric rule.",
            )
        )

    ratio_ids = list(m_df["ratio_id"])
    if winner_ratio == "hn20":
        tail_ratio = "hn16" if "hn16" in ratio_ids else ratio_ids[-2]
    else:
        tail_ratio = "hn20" if "hn20" in ratio_ids else ratio_ids[-1]
    rep_ratios = ["hn00", winner_ratio, tail_ratio]
    rep_ratios = list(dict.fromkeys(rep_ratios))
    ratio_frames = {ratio: load_epoch_summary(HN_M_MATERIALS, ratio) for ratio in rep_ratios}
    epoch_metrics = [
        ("spec_at_r995", "Spec@R99.5", False, "fig_hn_m_epoch_dynamics_spec995"),
        ("spec_at_r990", "Spec@R99.0", False, "fig_hn_m_epoch_dynamics_spec990"),
        ("prec_at_r990", "Prec@R99.0", False, "fig_hn_m_epoch_dynamics_prec990"),
        ("ptr_at_r990", "PTR@R99.0", True, "fig_hn_m_epoch_dynamics_ptr990"),
    ]
    panel_fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8))
    panel_rows = None
    colors = {"hn00": "#2a9d8f", winner_ratio: "#b6400b", tail_ratio: "#6d597a"}
    for ax, (metric, label, lower_better, stem) in zip(axes.flat, epoch_metrics):
        rows = []
        for ratio in rep_ratios:
            frame = ratio_frames[ratio]
            color = colors.get(ratio, "#204b57")
            ax.plot(frame["epoch"], frame[metric], linewidth=1.8, label=ratio, color=color)
            best_idx = frame[metric].idxmin() if lower_better else frame[metric].idxmax()
            best_epoch = int(frame.loc[best_idx, "epoch"])
            best_value = float(frame.loc[best_idx, metric])
            ax.scatter([best_epoch], [best_value], color=color, s=22, zorder=5)
            ax.annotate(f"{ratio}:{best_epoch}", (best_epoch, best_value), xytext=(4, 5), textcoords="offset points", fontsize=7)
            rows.extend(
                {
                    "ratio_id": ratio,
                    "epoch": int(row["epoch"]),
                    label: float(row[metric]),
                }
                for _, row in frame[["epoch", metric]].iterrows()
            )
        ax.set_title(f"{label} vs epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label + (" (lower is better)" if lower_better else ""))
        ax.legend(loc="best")
        if stem == "fig_hn_m_epoch_dynamics_spec995":
            panel_rows = pd.DataFrame(rows)
        fig, single_ax = plt.subplots(figsize=(5.8, 3.9))
        for ratio in rep_ratios:
            frame = ratio_frames[ratio]
            color = colors.get(ratio, "#204b57")
            single_ax.plot(frame["epoch"], frame[metric], linewidth=1.8, label=ratio, color=color)
            best_idx = frame[metric].idxmin() if lower_better else frame[metric].idxmax()
            best_epoch = int(frame.loc[best_idx, "epoch"])
            best_value = float(frame.loc[best_idx, metric])
            single_ax.scatter([best_epoch], [best_value], color=color, s=22, zorder=5)
            single_ax.annotate(f"{ratio}:{best_epoch}", (best_epoch, best_value), xytext=(4, 5), textcoords="offset points", fontsize=7)
        single_ax.set_title(f"{label} vs epoch")
        single_ax.set_xlabel("Epoch")
        single_ax.set_ylabel(label + (" (lower is better)" if lower_better else ""))
        single_ax.legend(loc="best")
        _save_figure(
            fig,
            stem,
            pd.DataFrame(rows),
            captions,
            f"How {label} evolves across training epochs for the baseline, winner, and high-ratio tail HN regimes.",
            [rel(HN_M_MATERIALS / "hn00" / "epoch_gate_summary.csv"), rel(HN_M_MATERIALS / winner_ratio / "epoch_gate_summary.csv"), rel(HN_M_MATERIALS / tail_ratio / "epoch_gate_summary.csv")],
            f"The epoch dynamics show that HN changes the training trajectory itself rather than only shifting the final selected checkpoint.",
            "Only three representative ratios are shown here to keep the comparison interpretable.",
            "Representative ratios are hn00, the formal winner, and a high-ratio tail setting.",
        )
    _save_figure(
        panel_fig,
        "fig_hn_m_epoch_dynamics_panel",
        panel_rows if panel_rows is not None else pd.DataFrame(),
        captions,
        "How the main formal gate metrics evolve across epochs for representative HN ratios.",
        [rel(HN_M_MATERIALS / "hn00" / "epoch_gate_summary.csv"), rel(HN_M_MATERIALS / winner_ratio / "epoch_gate_summary.csv"), rel(HN_M_MATERIALS / tail_ratio / "epoch_gate_summary.csv")],
        "The representative-ratio panel summarizes how HN affects the training dynamics beyond the final best-epoch snapshot.",
        "The panel is limited to representative ratios and does not replace the full ratio-level summaries.",
        "Representative ratios are selected as hn00, the formal winner, and a high-ratio tail condition.",
    )

    cross_compare = read_csv_file(HN_CROSS_COMPARE)
    cross_compare["ratio_percent"] = cross_compare["ratio_id"].astype(str).str.replace("hn", "", regex=False).astype(int)
    slope_metrics = [
        ("spec_at_r995", "Spec@R99.5", False, "fig_hn_cross_capacity_slope_spec995"),
        ("spec_at_r990", "Spec@R99.0", False, "fig_hn_cross_capacity_slope_spec990"),
        ("prec_at_r990", "Prec@R99.0", False, "fig_hn_cross_capacity_slope_prec990"),
        ("ptr_at_r990", "PTR@R99.0", True, "fig_hn_cross_capacity_slope_ptr990"),
    ]
    panel_fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.6))
    panel_df = None
    for ax, (metric, label, lower_better, stem) in zip(axes.flat, slope_metrics):
        rows = []
        for model, color in [("yolo11m-cls", "#204b57"), ("yolo11x-cls", "#b6400b")]:
            model_df = cross_compare.loc[cross_compare["model"] == model].sort_values("ratio_percent")
            ax.plot(model_df["ratio_percent"], model_df[metric], marker="o", linewidth=2, label=model.replace("-cls", ""), color=color)
            rows.extend(
                {
                    "model": model,
                    "ratio_id": str(row["ratio_id"]),
                    "ratio_percent": int(row["ratio_percent"]),
                    label: float(row[metric]),
                }
                for _, row in model_df[["ratio_id", "ratio_percent", metric]].iterrows()
            )
        ax.set_title(f"{label}: hn00 to hn02")
        ax.set_xlabel("HN ratio (%)")
        ax.set_ylabel(label + (" (lower is better)" if lower_better else ""))
        ax.set_xticks([0, 2], ["hn00", "hn02"])
        ax.legend(loc="best")
        if stem == "fig_hn_cross_capacity_slope_spec995":
            panel_df = pd.DataFrame(rows)
        fig, single_ax = plt.subplots(figsize=(5.8, 3.9))
        for model, color in [("yolo11m-cls", "#204b57"), ("yolo11x-cls", "#b6400b")]:
            model_df = cross_compare.loc[cross_compare["model"] == model].sort_values("ratio_percent")
            single_ax.plot(model_df["ratio_percent"], model_df[metric], marker="o", linewidth=2, label=model.replace("-cls", ""), color=color)
        single_ax.set_title(f"{label}: hn00 to hn02")
        single_ax.set_xlabel("HN ratio (%)")
        single_ax.set_ylabel(label + (" (lower is better)" if lower_better else ""))
        single_ax.set_xticks([0, 2], ["hn00", "hn02"])
        single_ax.legend(loc="best")
        _save_figure(
            fig,
            stem,
            pd.DataFrame(rows),
            captions,
            f"Whether the directional effect of HN from hn00 to hn02 is consistent between yolo11m and yolo11x for {label}.",
            [rel(HN_CROSS_COMPARE), rel(HN_M_SUMMARY), rel(HN_X_SUMMARY)],
            "The cross-capacity slope view shows whether HN behaves directionally similarly on the main model and the second model.",
            "yolo11x only has hn00 and hn02, so this is a light validation rather than a full sweep.",
            "The comparison is restricted to hn00 and hn02 under the same formal gate-aware rule.",
        )
    _save_figure(
        panel_fig,
        "fig_hn_cross_capacity_slope_panel",
        panel_df if panel_df is not None else pd.DataFrame(),
        captions,
        "Whether the HN direction from hn00 to hn02 is reproducible across the main and second models.",
        [rel(HN_CROSS_COMPARE), rel(HN_M_SUMMARY), rel(HN_X_SUMMARY)],
        "The panel supports a directional cross-capacity validation rather than a claim of x-side full-sweep optimality.",
        "Only hn00 and hn02 are available for yolo11x in the current working set.",
        "The comparison uses the existing cross-capacity summary without any new inference run.",
    )

    best_epoch_df = m_df[["ratio_id", "ratio_percent", "best_epoch", "formal_rank"]].copy()
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    ax.plot(best_epoch_df["ratio_percent"], best_epoch_df["best_epoch"], marker="o", linewidth=2, color="#204b57")
    ax.scatter([int(winner["ratio_percent"])], [int(winner["best_epoch"])], color="#b6400b", s=45, zorder=5)
    ax.annotate(winner_ratio, (int(winner["ratio_percent"]), int(winner["best_epoch"])), xytext=(4, 8), textcoords="offset points")
    ax.set_xlabel("HN ratio (%)")
    ax.set_ylabel("Gate-best epoch")
    ax.set_title("Gate-best epoch vs HN ratio")
    _save_figure(
        fig,
        "fig_hn_best_epoch_vs_ratio",
        best_epoch_df,
        captions,
        "How the location of the gate-best epoch changes across HN ratios.",
        [rel(HN_M_SUMMARY)],
        "The best-epoch position shifts with ratio, which indicates that HN changes the training regime rather than only the final score.",
        "This figure tracks only the selected gate-best epoch and not the full per-epoch trajectory.",
        "Best epochs are taken from the formal HN sweep summary after gate-aware checkpoint selection.",
    )

    return captions


def build_appendix_assets(inputs: dict[str, Any], derived: dict[str, pd.DataFrame]) -> list[CaptionSpec]:
    captions: list[CaptionSpec] = []
    ingest = inputs["ingest"]
    registry = inputs["registry"].copy()
    registry["hn_ratio"] = registry["ratio_id"]
    registry["selected_ckpt_name / path_stub"] = registry["exported_checkpoint_path"].fillna(registry["best_checkpoint_path"]).map(lambda x: Path(str(x)).name if pd.notna(x) else "")
    registry["notes"] = np.where(registry["checkpoint_exists"].astype(str).isin(["1", "True", "true"]), "checkpoint recorded", "checkpoint missing")
    clean_registry = registry[
        ["model", "hn_ratio", "best_epoch", "selected_ckpt_name / path_stub", "spec_at_r995", "spec_at_r990", "prec_at_r990", "ptr_at_r990", "notes"]
    ].rename(
        columns={
            "spec_at_r995": "Spec@R99.5",
            "spec_at_r990": "Spec@R99.0",
            "prec_at_r990": "Prec@R99.0",
            "ptr_at_r990": "PTR@R99.0",
        }
    )
    save_csv(APPENDIX_DIR / "table_hn_best_checkpoint_registry_clean.csv", clean_registry)
    save_md_table(APPENDIX_DIR / "table_hn_best_checkpoint_registry_clean.md", "table_hn_best_checkpoint_registry_clean", clean_registry)
    captions.append(
        CaptionSpec(
            "table_hn_best_checkpoint_registry_clean",
            "Where the selected HN checkpoints are recorded for later reproducibility and reuse.",
            [rel(HN_REGISTRY)],
            "This table is a cleaned view of the formal HN checkpoint registry.",
            "The registry provides a compact mapping from model and ratio to the selected formal checkpoint.",
            "Checkpoint paths point to archived locations and may not all exist in the current local repo working set.",
        )
    )

    coverage_rows = []
    raw_cov = {
        ratio: "repo summaries/manifests only; fuller non-PT source archive existed"
        for ratio in ingest["coverage"]["yolo11m_raw_plus_summary_ratios"]
    }
    raw_cov.update({ratio: "repo summaries/manifests only" for ratio in ingest["coverage"]["yolo11m_manifest_only_ratios"]})
    x_cov = {ratio: "repo summaries/manifests only" for ratio in ingest["coverage"]["yolo11x_manifest_only_ratios"]}
    for model, root, mapping in [("yolo11m-cls", HN_M_MATERIALS, raw_cov), ("yolo11x-cls", HN_X_MATERIALS, x_cov)]:
        for ratio_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            coverage_rows.append(
                {
                    "model": model,
                    "hn_ratio": ratio_dir.name,
                    "has_epoch_gate_summary": (ratio_dir / "epoch_gate_summary.csv").exists(),
                    "has_all_checkpoints_index": (ratio_dir / "all_checkpoints_index.csv").exists(),
                    "has_best_epoch_manifest": (ratio_dir / "best_epoch_manifest.json").exists(),
                    "has_raw_per_epoch_tree": (ratio_dir / "per_epoch_gate").exists(),
                    "has_pt": False,
                    "material_level_note": mapping.get(ratio_dir.name, "formal summary/manifests"),
                }
            )
    coverage_df = pd.DataFrame(coverage_rows)
    save_csv(APPENDIX_DIR / "table_hn_ratio_coverage_manifest.csv", coverage_df)
    save_md_table(APPENDIX_DIR / "table_hn_ratio_coverage_manifest.md", "table_hn_ratio_coverage_manifest", coverage_df)
    captions.append(
        CaptionSpec(
            "table_hn_ratio_coverage_manifest",
            "Which ratio-level HN materials are currently available in the repo working set.",
            [rel(HN_M_MATERIALS), rel(HN_X_MATERIALS), rel(INGEST_MANIFEST)],
            "Coverage is derived from the current repo paths, with source-archive provenance carried over only in the note field.",
            "The table reports repo working-set availability rather than the fuller external source archive.",
            "PT checkpoints and raw per-epoch trees are intentionally tracked as unavailable in this repo working set.",
        )
    )

    return captions


def build_appendix_visuals(inputs: dict[str, Any], derived: dict[str, pd.DataFrame]) -> list[CaptionSpec]:
    captions: list[CaptionSpec] = []
    heat_df = derived["hn_m_main_derived"].sort_values("ratio_percent")
    heat_cols = ["spec_at_r995", "spec_at_r990", "prec_at_r990", "ptr_at_r990", "best_epoch"]
    heat_data = heat_df.set_index("ratio_id")[heat_cols]
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    im = ax.imshow(heat_data.values, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(heat_data.index)), heat_data.index)
    ax.set_xticks(np.arange(len(heat_cols)), ["Spec@R99.5", "Spec@R99.0", "Prec@R99.0", "PTR@R99.0", "Best Epoch"], rotation=20, ha="right")
    for i in range(heat_data.shape[0]):
        for j in range(heat_data.shape[1]):
            ax.text(j, i, fmt4(heat_data.values[i, j]), ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(APPENDIX_DIR / "fig_hn_overview_heatmap_raw.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    save_csv(APPENDIX_DIR / "fig_hn_overview_heatmap_raw.csv", heat_data.reset_index())
    captions.append(
        CaptionSpec(
            "fig_hn_overview_heatmap_raw",
            "How all HN ratios compare jointly across the four formal gate metrics and the selected best epoch.",
            [rel(HN_M_SUMMARY)],
            "Rows are ratios and columns are formal summary metrics plus best epoch.",
            "The raw heatmap provides a compact whole-sweep view of the HN landscape.",
            "The metrics have different scales, so the panel is best used as a qualitative scan rather than a numeric substitute for the tables.",
        )
    )

    delta_heat = heat_df.set_index("ratio_id")[["delta_Spec@R99.5_vs_hn00", "delta_Spec@R99.0_vs_hn00", "delta_Prec@R99.0_vs_hn00", "delta_PTR@R99.0_vs_hn00"]]
    fig, ax = plt.subplots(figsize=(6.1, 4.6))
    im = ax.imshow(delta_heat.values, aspect="auto", cmap="coolwarm")
    ax.set_yticks(np.arange(len(delta_heat.index)), delta_heat.index)
    ax.set_xticks(np.arange(delta_heat.shape[1]), ["Delta Spec@R99.5", "Delta Spec@R99.0", "Delta Prec@R99.0", "Delta PTR@R99.0"], rotation=20, ha="right")
    for i in range(delta_heat.shape[0]):
        for j in range(delta_heat.shape[1]):
            ax.text(j, i, fmt4(delta_heat.values[i, j]), ha="center", va="center", color="black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(APPENDIX_DIR / "fig_hn_overview_heatmap_delta.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    save_csv(APPENDIX_DIR / "fig_hn_overview_heatmap_delta.csv", delta_heat.reset_index())
    captions.append(
        CaptionSpec(
            "fig_hn_overview_heatmap_delta",
            "How each HN ratio differs from the hn00 anchor across the four formal gate metrics.",
            [rel(HN_M_SUMMARY)],
            "Rows are ratios and all values are absolute deltas relative to hn00.",
            "The delta heatmap highlights the sweet-spot region more clearly than the raw-value view.",
            "PTR remains lower-is-better, so a negative delta is favorable.",
        )
    )

    normal_df = inputs["train_normal_scores"].copy()
    top_df = inputs["top_false_positive_normals"].copy()
    normal_df["p_abnormal"] = pd.to_numeric(normal_df["p_abnormal"])
    top_df["p_abnormal"] = pd.to_numeric(top_df["p_abnormal"])
    threshold = float(top_df["p_abnormal"].min())
    stats_df = pd.DataFrame(
        [
            {
                "normal_total": int(len(normal_df)),
                "selected_top_false_positive_count": int(len(top_df)),
                "selected_ratio": float(len(top_df) / len(normal_df)),
                "score_min": float(top_df["p_abnormal"].min()),
                "score_median": float(top_df["p_abnormal"].median()),
                "score_max": float(top_df["p_abnormal"].max()),
                "score_p95": float(top_df["p_abnormal"].quantile(0.95)),
                "score_p99": float(top_df["p_abnormal"].quantile(0.99)),
                "selection_threshold_min_score": threshold,
            }
        ]
    )
    save_csv(APPENDIX_DIR / "table_hn_hard_normal_selection_stats.csv", stats_df)
    save_md_table(APPENDIX_DIR / "table_hn_hard_normal_selection_stats.md", "table_hn_hard_normal_selection_stats", stats_df)
    captions.append(
        CaptionSpec(
            "table_hn_hard_normal_selection_stats",
            "What portion of the train-normal pool is selected into the hard-normal candidate set and what the selected-pool score range looks like.",
            [rel(HN_ASSETS / "train_normal_scores.csv"), rel(HN_ASSETS / "top_false_positive_normals.csv"), rel(HN_ASSETS / "summary.json")],
            "The selected hard normals are summarized using the existing score assets rather than any new inference run.",
            "The table reports selected-pool score statistics, while the threshold column records the minimum selected score.",
            "Heuristic labels in the source CSV are not used here because the key evidence is the score-tail behavior.",
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    axes[0].hist(normal_df["p_abnormal"], bins=30, color="#457b9d", alpha=0.85)
    axes[0].axvline(threshold, color="#b6400b", linestyle="--", label="selection threshold")
    axes[0].set_title("All train-normal scores")
    axes[0].set_xlabel("Predicted abnormal score")
    axes[0].set_ylabel("Count")
    axes[0].legend(loc="best")
    tail_mask = normal_df["p_abnormal"] >= normal_df["p_abnormal"].quantile(0.9)
    axes[1].hist(normal_df.loc[tail_mask, "p_abnormal"], bins=20, color="#2a9d8f", alpha=0.85)
    axes[1].axvline(threshold, color="#b6400b", linestyle="--")
    axes[1].set_title("Tail zoom (top 10%)")
    axes[1].set_xlabel("Predicted abnormal score")
    sorted_scores = np.sort(normal_df["p_abnormal"].to_numpy())
    axes[2].plot(np.linspace(0, 1, len(sorted_scores), endpoint=True), sorted_scores, color="#6d597a", linewidth=2)
    axes[2].axhline(threshold, color="#b6400b", linestyle="--")
    axes[2].set_title("Empirical score profile")
    axes[2].set_xlabel("Normalized rank")
    axes[2].set_ylabel("Predicted abnormal score")
    fig.tight_layout()
    fig.savefig(APPENDIX_DIR / "fig_hn_hard_normal_score_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    save_csv(APPENDIX_DIR / "fig_hn_hard_normal_score_distribution.csv", pd.DataFrame({"p_abnormal": normal_df["p_abnormal"]}))
    captions.append(
        CaptionSpec(
            "fig_hn_hard_normal_score_distribution",
            "Where the selected hard normals sit within the train-normal abnormal-score distribution.",
            [rel(HN_ASSETS / "train_normal_scores.csv"), rel(HN_ASSETS / "top_false_positive_normals.csv")],
            "The threshold is taken from the minimum score among the selected top false-positive normals.",
            "The selected HN pool comes from the high-score tail rather than ad hoc manual picking.",
            "This provenance view does not itself evaluate whether the selected normals are semantically diverse.",
        )
    )

    gallery_dir = HN_ASSETS / "hardest_normal_gallery"
    gallery_paths = sorted(gallery_dir.glob("*.png"))[:12]
    if gallery_paths:
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        for ax, img_path in zip(axes.flat, gallery_paths):
            with Image.open(img_path) as img:
                ax.imshow(img)
            ax.axis("off")
            parts = img_path.stem.split("_")
            rank = parts[0]
            score = parts[1] if len(parts) > 1 else ""
            short_id = parts[-1] if len(parts) > 2 else img_path.stem
            ax.set_title(f"#{rank} | {score}\\n{short_id}", fontsize=8)
        for ax in axes.flat[len(gallery_paths) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(APPENDIX_DIR / "fig_hn_hardest_normal_gallery_panel.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        gallery_rows = pd.DataFrame([{"image_name": p.name, "rank": i + 1, "score_stub": p.stem.split('_')[1] if len(p.stem.split('_')) > 1 else ''} for i, p in enumerate(gallery_paths)])
        save_csv(APPENDIX_DIR / "fig_hn_hardest_normal_gallery_panel.csv", gallery_rows)
        captions.append(
            CaptionSpec(
                "fig_hn_hardest_normal_gallery_panel",
                "What the top-ranked hard-normal candidates look like qualitatively.",
                [rel(gallery_dir), rel(HN_ASSETS / "top_false_positive_normals.csv")],
                "The panel uses the highest-ranked gallery images already exported by the HN asset builder.",
                "The gallery provides visual evidence that the HN source pool is tied to concrete difficult normal cases.",
                "This panel is illustrative and should be interpreted together with the formal score-distribution statistics.",
            )
        )

    unavailable_path = APPENDIX_DIR / "table_hn_top1_vs_gatebest_by_ratio_unavailable.md"
    unavailable_path.write_text(
        "# table_hn_top1_vs_gatebest_by_ratio\\n\\nThe current HN material level does not contain a reliable trainer-side per-ratio top1 summary in `all_checkpoints_index.csv`.\\nTherefore this conditional artifact is intentionally not generated.\\n",
        encoding="utf-8",
    )
    captions.append(
        CaptionSpec(
            "table_hn_top1_vs_gatebest_by_ratio_unavailable",
            "Whether HN preserves the top1-best versus gate-best mismatch across ratios.",
            [rel(HN_M_MATERIALS / "hn02" / "all_checkpoints_index.csv"), rel(HN_X_MATERIALS / "hn02" / "all_checkpoints_index.csv")],
            "Generation is conditional on the presence of reliable trainer-side top1 fields in the ratio-level checkpoint index.",
            "The current HN working set does not include those trainer-side fields, so no ratio-level top1 mismatch table is emitted.",
            "This absence is a material-level limitation rather than a modeling claim.",
        )
    )
    return captions


def build_asset_manifest(captions: list[CaptionSpec]) -> None:
    entries = []
    for spec in captions:
        stem = spec.stem
        matches = []
        for directory in (TABLES_DIR, FIGURES_DIR, APPENDIX_DIR):
            for path in directory.glob(f"{stem}*"):
                if path.is_file():
                    matches.append(rel(path))
        if (CAPTIONS_DIR / f"{stem}.md").exists():
            matches.append(rel(CAPTIONS_DIR / f"{stem}.md"))
        entries.append(
            {
                "stem": stem,
                "sources": spec.sources,
                "rule": spec.rule,
                "finding": spec.finding,
                "limitation": spec.limitation,
                "outputs": matches,
            }
        )
    (MANIFESTS_DIR / "paper_assets_manifest.json").write_text(json.dumps({"assets": entries}, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# paper_assets_manifest", ""]
    for entry in entries:
        lines.append(f"## {entry['stem']}")
        lines.append("")
        lines.append("- Outputs:")
        for out in entry["outputs"]:
            lines.append(f"  - `{out}`")
        lines.append("- Sources:")
        for src in entry["sources"]:
            lines.append(f"  - `{src}`")
        lines.append(f"- Rule: {entry['rule']}")
        lines.append(f"- Key finding: {entry['finding']}")
        lines.append(f"- Limitation: {entry['limitation']}")
        lines.append("")
    (MANIFESTS_DIR / "paper_assets_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    inputs = load_inputs()
    derived = build_derived(inputs)
    export_derived(derived)
    captions: list[CaptionSpec] = []
    captions.extend(build_main_tables(derived))
    captions.extend(build_main_figures(derived))
    captions.extend(build_appendix_assets(inputs, derived))
    captions.extend(build_appendix_visuals(inputs, derived))
    for spec in captions:
        write_caption(spec)
    build_asset_manifest(captions)
    print(f"[done] wrote HN paper assets to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
