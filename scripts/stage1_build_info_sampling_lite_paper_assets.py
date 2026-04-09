from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from pipeline_common import REPO_ROOT, YOLOV11_ROOT
from stage1_formal_capacity_suite import resolve_path, resolve_str


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

SETTING_ID_MAP = {
    "weighted_hn14_risk_only": "A2",
    "weighted_hn14_risk_consistency": "A3",
    "weighted_hn14_risk_consistency_density": "A4",
}

SETTING_LABEL_MAP = {
    "A0": "A0\nhn00",
    "A1": "A1\nuniform_hn14",
    "A2": "A2\nrisk_only",
    "A3": "A3\nrisk+consistency",
    "A4": "A4\nrisk+consistency+density",
}


@dataclass
class CaptionSpec:
    stem: str
    question: str
    sources: list[str]
    rule: str
    finding: str
    limitation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready assets for the stage-1 effective-information lite ablation suite.")
    parser.add_argument("--config", required=True, help="Info-sampling-lite runtime config.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON object expected: {path}")
    return payload


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def resolve_repo_anchored_path(raw_path: str | Path, fallback: Path | None = None) -> Path:
    path = Path(str(raw_path))
    if path.exists():
        return path
    if fallback is not None and fallback.exists():
        return fallback
    parts = list(path.parts)
    for anchor in ("research", "YOLOv11", "$out", "_recycle_bin"):
        if anchor in parts:
            anchor_index = parts.index(anchor)
            candidate = REPO_ROOT.joinpath(*parts[anchor_index:])
            if candidate.exists():
                return candidate
    return fallback if fallback is not None else path


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def fmt4(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.4f}"


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_md_table(path: Path, title: str, df: pd.DataFrame, notes: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [str(col) for col in df.columns]
    rows: list[list[str]] = []
    for _, row in df.iterrows():
        values: list[str] = []
        for column in df.columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, (float, np.floating)):
                values.append(fmt4(value))
            else:
                values.append(str(value))
        rows.append(values)
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    if notes:
        lines.extend(["", "Notes:"])
        lines.extend([f"- {note}" for note in notes])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_caption(captions_dir: Path, spec: CaptionSpec) -> None:
    path = captions_dir / f"{spec.stem}.md"
    path.write_text(
        "\n".join(
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
        ),
        encoding="utf-8",
    )


def rank_gate_rows(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.sort_values(
        by=["Spec@R99.5", "Spec@R99.0", "Prec@R99.0", "PTR@R99.0", "setting_order"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["formal_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def score_dir_from_cfg(cfg: dict[str, Any], materials_root: Path) -> Path:
    return resolve_path(cfg.get("score_output_dir"), base=materials_root / "score_inputs")


def build_context(cfg: dict[str, Any]) -> dict[str, Path]:
    materials_root = resolve_path(cfg.get("materials_root"), base=REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_info_sampling_lite")
    results_root = resolve_path(cfg.get("results_dir"), base=REPO_ROOT / "research" / "results" / "stage1_formal" / "gate_info_sampling_lite")
    return {
        "materials_root": materials_root,
        "results_root": results_root,
        "score_dir": score_dir_from_cfg(cfg, materials_root),
        "gate_hn_results": resolve_path(cfg.get("hn_summary_csv"), base=REPO_ROOT / "research" / "results" / "stage1_formal" / "gate_hn_m_sweep" / "hn_sweep_summary.csv"),
        "teacher_summary_dir": resolve_path(cfg.get("teacher_summary_dir"), base=REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_hn_m_sweep" / "hn00"),
        "uniform_summary_dir": resolve_path(cfg.get("uniform_anchor_summary_dir"), base=REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_hn_m_sweep" / "hn14"),
        "source_dataset": resolve_path(cfg.get("source_dataset"), base=YOLOV11_ROOT / "datasets" / "sewerml_gate2_train7200"),
        "pool_summary_json": resolve_path(cfg.get("pool_summary_json"), base=REPO_ROOT),
        "suite_context_json": results_root / "suite_context.json",
        "setup_audit_json": results_root / "setup_audit.json",
        "scratch_cleanup_json": results_root / "scratch_cleanup.json",
    }


def load_setting_best(summary_dir: Path) -> dict[str, Any]:
    best_path = summary_dir / "best_epoch_manifest.json"
    if not best_path.exists():
        raise SystemExit(f"Missing best_epoch_manifest.json: {summary_dir}")
    return load_json(best_path)


def load_epoch_summary(summary_dir: Path) -> pd.DataFrame:
    path = summary_dir / "epoch_gate_summary.csv"
    if not path.exists():
        raise SystemExit(f"Missing epoch_gate_summary.csv: {summary_dir}")
    return pd.read_csv(path)


def lookup_hn_row(hn_df: pd.DataFrame, ratio_id: str) -> pd.Series:
    matched = hn_df.loc[hn_df["ratio_id"] == ratio_id]
    if not matched.empty:
        return matched.iloc[0]
    overview_csv = REPO_ROOT / "research" / "results" / "stage1_formal" / "gate_hn_overview" / "stage1_formal_hn_overview.csv"
    if overview_csv.exists():
        overview_df = pd.read_csv(overview_csv)
        matched = overview_df.loc[(overview_df["model"] == "yolo11m-cls") & (overview_df["ratio_id"] == ratio_id)]
        if not matched.empty:
            return matched.iloc[0]
    raise SystemExit(f"Missing ratio `{ratio_id}` in HN summary inputs.")


def build_ablation_rows(cfg: dict[str, Any], ctx: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, Path]]:
    hn_df = pd.read_csv(ctx["gate_hn_results"])
    suite_ctx = load_json(ctx["suite_context_json"])
    settings = cfg.get("settings") or []
    setting_lookup = {resolve_str(item.get("setting_id"), ""): item for item in settings}
    budget_count = int(suite_ctx["fixed_budget_count"])
    budget_anchor = resolve_str(cfg.get("budget_anchor_ratio_id"), "hn14")

    hn00_row = lookup_hn_row(hn_df, resolve_str(cfg.get("teacher_ratio_id"), "hn00"))
    hn14_row = lookup_hn_row(hn_df, resolve_str(cfg.get("uniform_anchor_ratio_id"), "hn14"))

    row_bank: list[dict[str, Any]] = [
        {
            "setting": "A0_hn00",
            "setting_id": "A0",
            "setting_name": "hn00",
            "teacher": "reused hn00 teacher",
            "pool": "none",
            "budget_anchor": budget_anchor,
            "best_epoch": int(hn00_row["best_epoch"]),
            "Spec@R99.5": float(hn00_row["spec_at_r995"]),
            "Spec@R99.0": float(hn00_row["spec_at_r990"]),
            "Prec@R99.0": float(hn00_row["prec_at_r990"]),
            "PTR@R99.0": float(hn00_row["ptr_at_r990"]),
            "setting_order": 0,
        },
        {
            "setting": "A1_uniform_hn14",
            "setting_id": "A1",
            "setting_name": "uniform_hn14",
            "teacher": "reused uniform HN14",
            "pool": "fixed top250 hard-normal pool",
            "budget_anchor": budget_anchor,
            "best_epoch": int(hn14_row["best_epoch"]),
            "Spec@R99.5": float(hn14_row["spec_at_r995"]),
            "Spec@R99.0": float(hn14_row["spec_at_r990"]),
            "Prec@R99.0": float(hn14_row["prec_at_r990"]),
            "PTR@R99.0": float(hn14_row["ptr_at_r990"]),
            "setting_order": 1,
        },
    ]
    summary_dirs: dict[str, Path] = {
        "A0": ctx["teacher_summary_dir"],
        "A1": ctx["uniform_summary_dir"],
    }

    for index, suite_row in enumerate(suite_ctx["rows"], start=2):
        setting_id = resolve_str(suite_row.get("setting_id"), "")
        if setting_id not in setting_lookup:
            continue
        setting_name = resolve_str(suite_row.get("setting_name"), setting_id.lower())
        summary_dir = resolve_repo_anchored_path(suite_row["summary_dir"], ctx["materials_root"] / setting_name)
        best = load_setting_best(summary_dir)
        row_bank.append(
            {
                "setting": f"{setting_id}_{setting_name}",
                "setting_id": setting_id,
                "setting_name": setting_name,
                "teacher": "hn00 best checkpoint",
                "pool": "fixed top250 hard-normal pool",
                "budget_anchor": budget_anchor,
                "best_epoch": int(best["epoch"]),
                "Spec@R99.5": float(best["spec_at_r995"]),
                "Spec@R99.0": float(best["spec_at_r990"]),
                "Prec@R99.0": float(best["prec_at_r990"]),
                "PTR@R99.0": float(best["ptr_at_r990"]),
                "setting_order": index,
            }
        )
        summary_dirs[setting_id] = summary_dir

    main_df = pd.DataFrame(row_bank)
    main_df = rank_gate_rows(main_df)
    main_df["budget_anchor"] = main_df["budget_anchor"].map(lambda _value: f"{budget_anchor} ({budget_count} extras)")
    return main_df, summary_dirs


def build_delta_tables(main_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hn00 = main_df.loc[main_df["setting_id"] == "A0"].iloc[0]
    hn14 = main_df.loc[main_df["setting_id"] == "A1"].iloc[0]
    base_cols = ["setting", "setting_id", "setting_name", "best_epoch"]

    def delta_frame(anchor: pd.Series, anchor_name: str) -> pd.DataFrame:
        df = main_df[base_cols].copy()
        df["delta_Spec@R99.5"] = main_df["Spec@R99.5"] - float(anchor["Spec@R99.5"])
        df["delta_Spec@R99.0"] = main_df["Spec@R99.0"] - float(anchor["Spec@R99.0"])
        df["delta_Prec@R99.0"] = main_df["Prec@R99.0"] - float(anchor["Prec@R99.0"])
        df["delta_PTR@R99.0"] = main_df["PTR@R99.0"] - float(anchor["PTR@R99.0"])
        df["delta_best_epoch"] = main_df["best_epoch"] - int(anchor["best_epoch"])
        df["anchor"] = anchor_name
        return df

    return delta_frame(hn00, "hn00"), delta_frame(hn14, "uniform_hn14")


def build_score_component_stats(score_dir: Path) -> pd.DataFrame:
    path = score_dir / "table_score_component_stats.csv"
    if not path.exists():
        raise SystemExit(f"Missing score component stats: {path}")
    return pd.read_csv(path)


def build_setting_metric_rows(main_df: pd.DataFrame) -> pd.DataFrame:
    return main_df[
        ["setting", "teacher", "pool", "budget_anchor", "best_epoch", "Spec@R99.5", "Spec@R99.0", "Prec@R99.0", "PTR@R99.0", "formal_rank"]
    ].copy()


def build_epoch_panel(summary_dirs: dict[str, Path], target: Path) -> None:
    metrics = [
        ("spec_at_r995", "Spec@R99.5"),
        ("spec_at_r990", "Spec@R99.0"),
        ("prec_at_r990", "Prec@R99.0"),
        ("ptr_at_r990", "PTR@R99.0"),
    ]
    mapping = [("A0", "hn00", "#355C7D"), ("A1", "uniform_hn14", "#6C5B7B"), ("A4", "weighted_risk_consistency_density", "#F67280")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (field, title) in zip(axes.flatten(), metrics, strict=True):
        for setting_id, label, color in mapping:
            rows = load_epoch_summary(summary_dirs[setting_id])
            ax.plot(rows["epoch"], rows[field], label=label, color=color, linewidth=1.6)
            best_idx = rows[field].idxmax() if field != "ptr_at_r990" else rows[field].idxmin()
            best_row = rows.loc[best_idx]
            ax.scatter([best_row["epoch"]], [best_row[field]], color=color, s=28, zorder=5)
        ax.set_title(title if field != "ptr_at_r990" else "PTR@R99.0 (lower is better)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    axes[0, 0].legend(loc="best")
    fig.suptitle("Epoch dynamics for representative settings", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(target, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_main_barpanel(main_df: pd.DataFrame, target: Path) -> None:
    settings = main_df["setting"].tolist()
    x = np.arange(len(settings))
    metrics = [
        ("Spec@R99.5", "#355C7D"),
        ("Spec@R99.0", "#6C5B7B"),
        ("Prec@R99.0", "#F67280"),
        ("PTR@R99.0", "#C06C84"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, color) in zip(axes.flatten(), metrics, strict=True):
        values = main_df[metric].to_numpy(dtype=float)
        ax.bar(x, values, color=color, width=0.65)
        ax.set_xticks(x, settings, rotation=20, ha="right")
        ax.set_title(metric if metric != "PTR@R99.0" else "PTR@R99.0 (lower is better)")
    fig.suptitle("Main ablation metrics under the formal gate-aware protocol", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(target, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_budget_concentration_tripanel(score_stats: pd.DataFrame, target: Path) -> pd.DataFrame:
    plot_df = score_stats.copy()
    plot_df["setting_id"] = plot_df["setting"].map(SETTING_ID_MAP)
    plot_df = plot_df.loc[plot_df["setting_id"].isin(["A2", "A3", "A4"])].copy()
    plot_df["setting_label"] = plot_df["setting_id"].map(SETTING_LABEL_MAP)
    plot_df = plot_df.sort_values("setting_id").reset_index(drop=True)

    metrics = [
        ("effective_count", "Effective Count", "#355C7D"),
        ("gini", "Gini", "#F67280"),
        ("top10_cumulative_pi", "Top10 Cumulative $\\pi$", "#6C5B7B"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    x = np.arange(len(plot_df))
    for ax, (column, title, color) in zip(axes, metrics, strict=True):
        values = plot_df[column].to_numpy(dtype=float)
        bars = ax.bar(x, values, color=color, width=0.62)
        ax.set_xticks(x, plot_df["setting_label"].tolist())
        ax.set_title(title)
        value_max = max(values) if len(values) else 1.0
        ax.set_ylim(0, value_max * 1.18 if value_max > 0 else 1.0)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + value_max * 0.03,
                fmt4(value),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("Replay-budget concentration diagnostics for weighted variants", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(target, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_df[
        ["setting_id", "setting", "effective_count", "gini", "top10_cumulative_pi", "nonzero_probability_count"]
    ].copy()


def build_score_distribution_panel(score_dir: Path, target: Path) -> None:
    settings = [("A2", "risk_only"), ("A3", "risk_consistency"), ("A4", "risk_consistency_density")]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for row_index, (setting_id, title) in enumerate(settings):
        df = pd.read_csv(score_dir / f"{setting_id}_candidate_pool_scores.csv")
        axes[row_index, 0].hist(df["S"], bins=20, color="#355C7D", alpha=0.85)
        axes[row_index, 0].set_title(f"{setting_id} score distribution ({title})")
        axes[row_index, 1].hist(df["pi"], bins=20, color="#F67280", alpha=0.85)
        axes[row_index, 1].set_title(f"{setting_id} sampling probability distribution")
    fig.tight_layout()
    fig.savefig(target, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_pool_width_threshold_panel(
    score_dir: Path,
    target: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool_manifest = load_json(score_dir / "pool_source_manifest.json")
    pool_scores_path = resolve_repo_anchored_path(
        pool_manifest["pool_scores_csv"],
        REPO_ROOT / "research" / "materials" / "stage1_formal" / "gate_hn_assets" / "yolo11m_train_normal_scores" / "train_normal_scores.csv",
    )
    top_k = int(pool_manifest["candidate_top_k"])
    tau_r995 = float(pool_manifest["teacher_tau_r995"])
    tau_r990 = float(pool_manifest["teacher_tau_r990"])

    pool_df = pd.read_csv(pool_scores_path)
    if "p_abnormal" not in pool_df.columns:
        raise SystemExit(f"Expected `p_abnormal` in pool score csv: {pool_scores_path}")

    ranked_df = pool_df.sort_values("p_abnormal", ascending=False).reset_index(drop=True).copy()
    ranked_df["rank_desc"] = np.arange(1, len(ranked_df) + 1)
    ranked_df["in_top250"] = ranked_df["rank_desc"] <= top_k
    ranked_df["image_stub"] = ranked_df["img_rel_path"].astype(str).map(lambda value: Path(value).stem)

    top_cutoff = float(ranked_df.loc[top_k - 1, "p_abnormal"])
    top_median = float(ranked_df.loc[ranked_df["in_top250"], "p_abnormal"].median())
    top_ratio = float(top_k / len(ranked_df))

    markers_df = pd.DataFrame(
        [
            {"marker": "tau_r995", "value": tau_r995},
            {"marker": "tau_r990", "value": tau_r990},
            {"marker": "top250_cutoff", "value": top_cutoff},
            {"marker": "top250_median", "value": top_median},
            {"marker": "top250_ratio", "value": top_ratio},
            {"marker": "total_train_normals", "value": float(len(ranked_df))},
            {"marker": "candidate_top_k", "value": float(top_k)},
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    hist_ax = axes[0]
    hist_ax.hist(ranked_df["p_abnormal"], bins=32, color="#355C7D", alpha=0.85)
    hist_ax.axvspan(top_cutoff, ranked_df["p_abnormal"].max(), color="#F67280", alpha=0.14, label="top250 span")
    hist_ax.axvline(tau_r995, color="#C06C84", linestyle="--", linewidth=1.7, label=r"$\tau_{99.5}$")
    hist_ax.axvline(tau_r990, color="#6C5B7B", linestyle="--", linewidth=1.7, label=r"$\tau_{99.0}$")
    hist_ax.axvline(top_cutoff, color="#F67280", linestyle="-", linewidth=1.7, label="top250 cutoff")
    hist_ax.set_title("Train-normal score distribution")
    hist_ax.set_xlabel("Teacher abnormal score")
    hist_ax.set_ylabel("Count")
    hist_ax.legend(loc="upper right")

    rank_ax = axes[1]
    rank_ax.plot(ranked_df["rank_desc"], ranked_df["p_abnormal"], color="#355C7D", linewidth=1.8)
    rank_ax.axvspan(1, top_k, color="#F67280", alpha=0.14, label=f"top250 ({top_ratio:.2%})")
    rank_ax.axhline(tau_r995, color="#C06C84", linestyle="--", linewidth=1.5, label=r"$\tau_{99.5}$")
    rank_ax.axhline(tau_r990, color="#6C5B7B", linestyle="--", linewidth=1.5, label=r"$\tau_{99.0}$")
    rank_ax.axhline(top_cutoff, color="#F67280", linestyle="-", linewidth=1.5, label="top250 cutoff")
    rank_ax.scatter([top_k], [top_cutoff], color="#F67280", s=32, zorder=5)
    rank_ax.set_title("Descending rank view of the hard-normal pool")
    rank_ax.set_xlabel("Rank among train-normal scores (descending)")
    rank_ax.set_ylabel("Teacher abnormal score")
    rank_ax.legend(loc="upper right")

    fig.suptitle("Hard-normal pool width relative to the formal operating thresholds", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(target, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return ranked_df[["rank_desc", "img_rel_path", "p_abnormal", "in_top250", "image_stub"]].copy(), markers_df


def build_overlap_figure(score_dir: Path, target: Path) -> pd.DataFrame:
    uniform_df = pd.read_csv(score_dir / "uniform_hn14_reference.csv")
    uniform_set = set(uniform_df["image_id"].astype(str))
    rows: list[dict[str, Any]] = []
    for setting_id in ("A2", "A3", "A4"):
        df = pd.read_csv(score_dir / f"{setting_id}_candidate_pool_scores.csv")
        selected = set(df.loc[df["selected_flag"] > 0, "image_id"].astype(str))
        rows.append(
            {
                "setting_id": setting_id,
                "overlap_count": len(uniform_set & selected),
                "weighted_only_count": len(selected - uniform_set),
                "uniform_only_count": len(uniform_set - selected),
            }
        )
    overlap_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(overlap_df))
    width = 0.24
    ax.bar(x - width, overlap_df["overlap_count"], width=width, label="overlap", color="#355C7D")
    ax.bar(x, overlap_df["weighted_only_count"], width=width, label="weighted-only", color="#F67280")
    ax.bar(x + width, overlap_df["uniform_only_count"], width=width, label="uniform-only", color="#6C5B7B")
    ax.set_xticks(x, overlap_df["setting_id"].tolist())
    ax.set_ylabel("Count")
    ax.set_title("Uniform HN14 vs weighted replay overlap")
    for offset, column in ((-width, "overlap_count"), (0, "weighted_only_count"), (width, "uniform_only_count")):
        for xpos, value in zip(x + offset, overlap_df[column].to_numpy(dtype=float), strict=True):
            ax.text(xpos, value + 2, f"{int(value)}", ha="center", va="bottom", fontsize=8)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(target, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return overlap_df


def build_a1_vs_a4_spec995_delta(summary_dirs: dict[str, Path], target: Path) -> pd.DataFrame:
    a1 = load_epoch_summary(summary_dirs["A1"])[["epoch", "spec_at_r995"]].rename(columns={"spec_at_r995": "a1_spec_at_r995"})
    a4 = load_epoch_summary(summary_dirs["A4"])[["epoch", "spec_at_r995"]].rename(columns={"spec_at_r995": "a4_spec_at_r995"})
    joined = a1.merge(a4, on="epoch", how="inner")
    joined["delta_spec_at_r995"] = joined["a4_spec_at_r995"] - joined["a1_spec_at_r995"]

    best_a1 = joined.loc[joined["a1_spec_at_r995"].idxmax()]
    best_a4 = joined.loc[joined["a4_spec_at_r995"].idxmax()]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2]})

    top_ax = axes[0]
    top_ax.plot(joined["epoch"], joined["a1_spec_at_r995"], color="#6C5B7B", linewidth=1.8, label="A1 uniform_hn14")
    top_ax.plot(joined["epoch"], joined["a4_spec_at_r995"], color="#F67280", linewidth=1.8, label="A4 risk+consistency+density")
    top_ax.scatter([best_a1["epoch"]], [best_a1["a1_spec_at_r995"]], color="#6C5B7B", s=34, zorder=5)
    top_ax.scatter([best_a4["epoch"]], [best_a4["a4_spec_at_r995"]], color="#F67280", s=34, zorder=5)
    top_ax.set_ylabel("Spec@R99.5")
    top_ax.set_title("A1 vs A4 on the primary metric")
    top_ax.legend(loc="best")

    delta_ax = axes[1]
    delta_ax.axhline(0.0, color="#444444", linewidth=1.0)
    delta_ax.plot(joined["epoch"], joined["delta_spec_at_r995"], color="#355C7D", linewidth=1.8)
    delta_ax.fill_between(
        joined["epoch"],
        joined["delta_spec_at_r995"],
        0.0,
        where=joined["delta_spec_at_r995"] >= 0.0,
        color="#355C7D",
        alpha=0.20,
    )
    delta_ax.fill_between(
        joined["epoch"],
        joined["delta_spec_at_r995"],
        0.0,
        where=joined["delta_spec_at_r995"] < 0.0,
        color="#F67280",
        alpha=0.20,
    )
    delta_ax.set_xlabel("Epoch")
    delta_ax.set_ylabel(r"$\Delta$Spec@R99.5")
    delta_ax.set_title("A4 - A1 delta curve")

    fig.suptitle("Primary-metric gap between the uniform anchor and the best weighted variant", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(target, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return joined


def render_contact_sheet(rows: pd.DataFrame, source_dataset: Path, title: str) -> Image.Image:
    thumb_w, thumb_h = 120, 90
    label_h = 52
    cols, rows_n = 4, 3
    padding = 10
    title_h = 34
    width = cols * (thumb_w + padding) + padding
    height = title_h + rows_n * (thumb_h + label_h + padding) + padding
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((padding, 8), title, fill="black", font=font)
    for index, (_, row) in enumerate(rows.iterrows()):
        col = index % cols
        row_id = index // cols
        if row_id >= rows_n:
            break
        x0 = padding + col * (thumb_w + padding)
        y0 = title_h + row_id * (thumb_h + label_h + padding)
        rel_path = Path(str(row["img_rel_path"]))
        image_path = source_dataset / rel_path
        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((thumb_w, thumb_h))
            thumb = Image.new("RGB", (thumb_w, thumb_h), "#f2f2f2")
            offset = ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2)
            thumb.paste(image, offset)
        except Exception:
            thumb = Image.new("RGB", (thumb_w, thumb_h), "#efefef")
            thumb_draw = ImageDraw.Draw(thumb)
            thumb_draw.rectangle((0, 0, thumb_w - 1, thumb_h - 1), outline="#999999")
            thumb_draw.text((8, 8), "missing", fill="#333333", font=font)
        canvas.paste(thumb, (x0, y0))
        text = [
            str(row["image_id"])[:18],
            f"p={float(row['calibrated_p']):.3f} S={float(row['S']):.3f}",
            f"pi={float(row['pi']):.3f}",
        ]
        draw.multiline_text((x0, y0 + thumb_h + 4), "\n".join(text), fill="black", font=font, spacing=2)
    return canvas


def build_gallery_panel(score_dir: Path, source_dataset: Path, target: Path, top_n: int) -> None:
    settings = [("A2", "risk_only"), ("A3", "risk_consistency"), ("A4", "risk_consistency_density")]
    sheets: list[Image.Image] = []
    for setting_id, _title in settings:
        df = pd.read_csv(score_dir / f"{setting_id}_candidate_pool_scores.csv")
        ranked = df.sort_values(by="S", ascending=False).reset_index(drop=True)
        top_sheet = render_contact_sheet(ranked.head(top_n), source_dataset, f"{setting_id} top-{top_n}")
        bottom_sheet = render_contact_sheet(ranked.tail(top_n).sort_values(by="S", ascending=True), source_dataset, f"{setting_id} bottom-{top_n}")
        row_canvas = Image.new("RGB", (top_sheet.width + bottom_sheet.width + 20, max(top_sheet.height, bottom_sheet.height)), "white")
        row_canvas.paste(top_sheet, (0, 0))
        row_canvas.paste(bottom_sheet, (top_sheet.width + 20, 0))
        sheets.append(row_canvas)

    total_height = sum(sheet.height for sheet in sheets) + 20 * max(len(sheets) - 1, 0)
    max_width = max(sheet.width for sheet in sheets)
    final = Image.new("RGB", (max_width, total_height), "white")
    y = 0
    for sheet in sheets:
        final.paste(sheet, (0, y))
        y += sheet.height + 20
    final.save(target)


def build_ratio_coverage(materials_root: Path, target_csv: Path, target_md: Path) -> None:
    rows: list[dict[str, Any]] = []
    for setting_name in ("weighted_hn14_risk_only", "weighted_hn14_risk_consistency", "weighted_hn14_risk_consistency_density"):
        summary_dir = materials_root / setting_name
        rows.append(
            {
                "setting_name": setting_name,
                "has_epoch_gate_summary": (summary_dir / "epoch_gate_summary.csv").exists(),
                "has_all_checkpoints_index": (summary_dir / "all_checkpoints_index.csv").exists(),
                "has_best_epoch_manifest": (summary_dir / "best_epoch_manifest.json").exists(),
                "has_pt": (summary_dir / "best_epoch_manifest.json").exists(),
                "material_level_note": "repo working set keeps summaries/manifests and checkpoint pointers; PT lives in training runs/checkpoint storage",
            }
        )
    df = pd.DataFrame(rows)
    save_csv(target_csv, df)
    save_md_table(target_md, "table_ratio_coverage_manifest", df)


def build_summary_docs(
    *,
    output_root: Path,
    main_df: pd.DataFrame,
    delta_vs_hn00: pd.DataFrame,
    delta_vs_uniform: pd.DataFrame,
    score_stats: pd.DataFrame,
    overlap_df: pd.DataFrame,
    ctx: dict[str, Path],
    cfg: dict[str, Any],
) -> None:
    a1 = main_df.loc[main_df["setting_id"] == "A1"].iloc[0]
    a2 = main_df.loc[main_df["setting_id"] == "A2"].iloc[0]
    a3 = main_df.loc[main_df["setting_id"] == "A3"].iloc[0]
    a4 = main_df.loc[main_df["setting_id"] == "A4"].iloc[0]
    best_row = main_df.sort_values("formal_rank").iloc[0]
    criterion_1 = float(a4["Spec@R99.5"]) > float(a1["Spec@R99.5"]) and float(a4["Spec@R99.0"]) >= float(a1["Spec@R99.0"]) - 0.005
    criterion_3 = float(a3["Spec@R99.5"]) >= float(a2["Spec@R99.5"]) or float(a4["Spec@R99.5"]) >= float(a2["Spec@R99.5"])
    if criterion_1 and criterion_3:
        next_step = "L1: move to a static full version by adding the objective-alignment gradient term G."
    elif abs(float(a4["Spec@R99.5"]) - float(a1["Spec@R99.5"])) <= 0.005 and criterion_3:
        next_step = "L2: keep the lite formulation and run a very small robustness check on alpha/kappa or a yolo11s diagnostic transfer."
    else:
        next_step = "L3: revisit teacher choice, fixed-pool width, or whether consistency/density over-suppressed true hard negatives before attempting the full version."

    summary_lines = [
        "# SUMMARY_gate_info_sampling_lite",
        "",
        "## Experiment Groups",
        "- A0: hn00 baseline (reused).",
        "- A1: uniform_hn14 (reused).",
        "- A2: weighted_hn14_risk_only (new training).",
        "- A3: weighted_hn14_risk_consistency (new training).",
        "- A4: weighted_hn14_risk_consistency_density (new training).",
        "",
        "## Score Definition",
        f"- alpha: `{float(cfg.get('alpha', 2.0) or 2.0):.2f}`",
        f"- kappa: `{float(cfg.get('kappa', 2.0) or 2.0):.2f}`",
        f"- density_k: `{int(cfg.get('density_k', 15) or 15)}`",
        "- A2 score: `S = R`",
        "- A3 score: `S = sqrt(R * C)`",
        "- A4 score: `S = (R * C * D)^(1/3)`",
        "",
        "## Budget Alignment",
        f"- fixed pool: `{int(load_json(ctx['pool_summary_json'])['top_k'])}` hard-normal candidates",
        f"- fixed replay budget: `{load_json(ctx['suite_context_json'])['fixed_budget_count']}` extra normal replays, matched to uniform_hn14",
        "",
        "## Key Result",
        f"- best lite setting: `{best_row['setting_name']}` (formal rank `{int(best_row['formal_rank'])}`)",
        f"- A4 vs A1 delta Spec@R99.5: `{delta_vs_uniform.loc[delta_vs_uniform['setting_id'] == 'A4', 'delta_Spec@R99.5'].iloc[0]:+.4f}`",
        f"- A4 vs A1 delta Spec@R99.0: `{delta_vs_uniform.loc[delta_vs_uniform['setting_id'] == 'A4', 'delta_Spec@R99.0'].iloc[0]:+.4f}`",
        f"- A4 vs A1 delta Prec@R99.0: `{delta_vs_uniform.loc[delta_vs_uniform['setting_id'] == 'A4', 'delta_Prec@R99.0'].iloc[0]:+.4f}`",
        f"- A4 vs A1 delta PTR@R99.0: `{delta_vs_uniform.loc[delta_vs_uniform['setting_id'] == 'A4', 'delta_PTR@R99.0'].iloc[0]:+.4f}`",
        "",
        "## Interpretation",
        f"- A4 better than uniform_hn14 on Spec@R99.5 with limited Spec@R99.0 damage: `{criterion_1}`",
        f"- A3/A4 outperform or match A2 on the primary metric: `{criterion_3}`",
        f"- overlap reference file: `{rel(output_root / 'appendix' / 'table_uniform_vs_weighted_overlap.csv')}`",
        "",
        "## Next Step Recommendation",
        f"- {next_step}",
        "",
    ]
    (output_root / "SUMMARY_gate_info_sampling_lite.md").write_text("\n".join(summary_lines), encoding="utf-8")

    reproduce_lines = [
        "# REPRODUCE_gate_info_sampling_lite",
        "",
        "## Sources",
        f"- source dataset: `{ctx['source_dataset']}`",
        f"- hn summary: `{ctx['gate_hn_results']}`",
        f"- top250 pool summary: `{ctx['pool_summary_json']}`",
        f"- suite context: `{ctx['suite_context_json']}`",
        "",
        "## Teacher",
        f"- teacher ratio: `{resolve_str(cfg.get('teacher_ratio_id'), 'hn00')}`",
        f"- teacher summary dir: `{ctx['teacher_summary_dir']}`",
        "",
        "## One-Click Command",
        "- `uv run main.py --task stage1_formal_gate_info_sampling_lite`",
        "- add `--rerun` to archive and rebuild the suite",
        "",
        "## Key Output Paths",
        f"- score inputs: `{score_dir_from_cfg(cfg, ctx['materials_root'])}`",
        f"- material summaries: `{ctx['materials_root']}`",
        f"- result package: `{ctx['results_root']}`",
        "",
    ]
    (output_root / "REPRODUCE_gate_info_sampling_lite.md").write_text("\n".join(reproduce_lines), encoding="utf-8")

    artifacts_lines = [
        "# ARTIFACTS_gate_info_sampling_lite",
        "",
        "## Tables",
        *[f"- `{name}`" for name in sorted(path.name for path in (output_root / "tables").glob("*"))],
        "",
        "## Figures",
        *[f"- `{name}`" for name in sorted(path.name for path in (output_root / "figures").glob("*"))],
        "",
        "## Appendix",
        *[f"- `{name}`" for name in sorted(path.name for path in (output_root / "appendix").glob("*"))],
        "",
        "## Captions",
        *[f"- `{name}`" for name in sorted(path.name for path in (output_root / "captions").glob("*"))],
        "",
        "## Scratch Cleanup",
        f"- cleanup manifest: `{ctx['scratch_cleanup_json']}`",
        "- large temporary feature exports were removed after score generation and are not kept in the formal working set.",
        "",
    ]
    (output_root / "ARTIFACTS_gate_info_sampling_lite.md").write_text("\n".join(artifacts_lines), encoding="utf-8")


def build_appendix_manifests(output_root: Path, ctx: dict[str, Path]) -> None:
    score_dir = ctx["score_dir"]
    pool_json = load_json(score_dir / "pool_source_manifest.json")
    feature_json = load_json(score_dir / "feature_extraction_manifest.json")
    tta_json = load_json(score_dir / "tta_manifest.json")

    for stem, payload in (
        ("pool_source_manifest", pool_json),
        ("feature_extraction_manifest", feature_json),
        ("tta_manifest", tta_json),
    ):
        path_json = output_root / "appendix" / f"{stem}.json"
        path_md = output_root / "appendix" / f"{stem}.md"
        path_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        lines = [f"# {stem}", ""]
        for key, value in payload.items():
            lines.append(f"- {key}: `{value}`")
        path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    (output_root / "appendix" / "score_formula.md").write_text(
        "\n".join(
            [
                "# score_formula",
                "",
                "- A2: `S = R`",
                "- A3: `S = sqrt(R * C)`",
                "- A4: `S = (R * C * D)^(1/3)`",
                "- replay probability: `pi_i = (S_i + eps)^kappa / sum_j (S_j + eps)^kappa`",
                "- ranking rule remains `Spec@R99.5 > Spec@R99.0 > Prec@R99.0 > PTR@R99.0`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    cfg = load_json(resolve_path(args.config, base=YOLOV11_ROOT / "configs" / "runtime"))
    ctx = build_context(cfg)
    output_root = ctx["results_root"]
    derived_dir = output_root / "derived"
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    appendix_dir = output_root / "appendix"
    captions_dir = output_root / "captions"
    manifests_dir = output_root / "manifests"
    ensure_dirs(output_root, derived_dir, tables_dir, figures_dir, appendix_dir, captions_dir, manifests_dir)

    main_df, summary_dirs = build_ablation_rows(cfg, ctx)
    delta_vs_hn00, delta_vs_uniform = build_delta_tables(main_df)
    score_stats = build_score_component_stats(ctx["score_dir"])

    save_csv(derived_dir / "ablation_main_joined.csv", main_df)
    save_csv(derived_dir / "ablation_delta_vs_hn00.csv", delta_vs_hn00)
    save_csv(derived_dir / "ablation_delta_vs_uniform_hn14.csv", delta_vs_uniform)

    main_table = build_setting_metric_rows(main_df)
    save_csv(tables_dir / "table_lite_ablation_main.csv", main_table)
    save_md_table(
        tables_dir / "table_lite_ablation_main.md",
        "table_lite_ablation_main",
        main_table,
        notes=[
            "A0 and A1 are reused anchors from the completed formal HN sweep.",
            "A2/A3/A4 reuse the fixed top250 pool and the hn14-equivalent budget, and only change pool-internal replay probability.",
        ],
    )
    save_csv(tables_dir / "table_lite_ablation_delta_vs_hn00.csv", delta_vs_hn00)
    save_md_table(tables_dir / "table_lite_ablation_delta_vs_hn00.md", "table_lite_ablation_delta_vs_hn00", delta_vs_hn00)
    save_csv(tables_dir / "table_lite_ablation_delta_vs_uniform_hn14.csv", delta_vs_uniform)
    save_md_table(tables_dir / "table_lite_ablation_delta_vs_uniform_hn14.md", "table_lite_ablation_delta_vs_uniform_hn14", delta_vs_uniform)
    save_csv(tables_dir / "table_score_component_stats.csv", score_stats)
    save_md_table(tables_dir / "table_score_component_stats.md", "table_score_component_stats", score_stats)

    build_main_barpanel(main_df, figures_dir / "fig_lite_ablation_main_barpanel.png")
    build_epoch_panel(summary_dirs, figures_dir / "fig_lite_ablation_epoch_dynamics_panel.png")
    budget_diag_df = build_budget_concentration_tripanel(score_stats, figures_dir / "fig_budget_concentration_tripanel.png")
    build_score_distribution_panel(ctx["score_dir"], figures_dir / "fig_score_distribution_panel.png")
    pool_width_curve_df, pool_width_markers_df = build_pool_width_threshold_panel(
        ctx["score_dir"],
        figures_dir / "fig_pool_width_threshold_panel.png",
    )
    build_gallery_panel(
        ctx["score_dir"],
        ctx["source_dataset"],
        figures_dir / "fig_top_bottom_gallery_panel.png",
        top_n=int(cfg.get("panel_top_n", 12) or 12),
    )
    overlap_df = build_overlap_figure(ctx["score_dir"], figures_dir / "fig_uniform_vs_weighted_overlap.png")
    a1_a4_delta_df = build_a1_vs_a4_spec995_delta(summary_dirs, figures_dir / "fig_a1_vs_a4_spec995_delta.png")
    save_csv(appendix_dir / "table_uniform_vs_weighted_overlap.csv", overlap_df)
    save_md_table(appendix_dir / "table_uniform_vs_weighted_overlap.md", "table_uniform_vs_weighted_overlap", overlap_df)
    save_csv(appendix_dir / "table_budget_concentration_diagnostics.csv", budget_diag_df)
    save_md_table(appendix_dir / "table_budget_concentration_diagnostics.md", "table_budget_concentration_diagnostics", budget_diag_df)
    save_csv(appendix_dir / "table_pool_width_threshold_markers.csv", pool_width_markers_df)
    save_md_table(appendix_dir / "table_pool_width_threshold_markers.md", "table_pool_width_threshold_markers", pool_width_markers_df)
    save_csv(derived_dir / "pool_width_ranked_normals.csv", pool_width_curve_df)
    save_csv(derived_dir / "a1_vs_a4_spec995_delta.csv", a1_a4_delta_df)

    build_ratio_coverage(
        ctx["materials_root"],
        appendix_dir / "table_info_sampling_material_coverage.csv",
        appendix_dir / "table_info_sampling_material_coverage.md",
    )
    build_appendix_manifests(output_root, ctx)

    write_caption(
        captions_dir,
        CaptionSpec(
            "table_lite_ablation_main",
            "Whether fixed-budget weighted replay can outperform hn00 and uniform_hn14 under the formal gate-aware rule.",
            [rel(derived_dir / "ablation_main_joined.csv")],
            "Rows are ranked by Spec@R99.5, Spec@R99.0, Prec@R99.0, and PTR@R99.0.",
            f"The best row is {main_df.sort_values('formal_rank').iloc[0]['setting_name']} under the formal gate objective.",
            "This table inherits A0/A1 from the completed HN formal summaries rather than retraining them.",
        ),
    )
    write_caption(
        captions_dir,
        CaptionSpec(
            "table_lite_ablation_delta_vs_hn00",
            "How much each setting moves the formal gate metrics relative to the no-HN anchor.",
            [rel(derived_dir / "ablation_delta_vs_hn00.csv")],
            "All deltas are absolute differences against A0/hn00.",
            "The table isolates the gain attributable to uniform HN or weighted replay rather than backbone changes.",
            "Deltas do not encode statistical significance and should be interpreted as formal point estimates.",
        ),
    )
    write_caption(
        captions_dir,
        CaptionSpec(
            "table_lite_ablation_delta_vs_uniform_hn14",
            "Whether the weighted variants beat the current uniform HN14 anchor under the same replay budget.",
            [rel(derived_dir / "ablation_delta_vs_uniform_hn14.csv")],
            "All deltas are absolute differences against A1/uniform_hn14.",
            "This table directly tests whether pool-internal value heterogeneity matters under a fixed HN14 budget.",
            "The comparison is restricted to the yolo11m mainline and does not revisit backbone choice.",
        ),
    )
    write_caption(
        captions_dir,
        CaptionSpec(
            "table_score_component_stats",
            "How concentrated the score and replay-probability distributions are for the lite scoring variants.",
            [rel(ctx["score_dir"] / "table_score_component_stats.csv")],
            "Scores come from the one-shot hn00 teacher and the fixed top250 pool.",
            "Higher concentration indicates that a small subset of hard negatives receives a larger share of the fixed replay budget.",
            "These statistics operate on the candidate pool only and do not by themselves prove downstream formal gains.",
        ),
    )
    write_caption(
        captions_dir,
        CaptionSpec(
            "fig_lite_ablation_main_barpanel",
            "Whether the weighted variants outperform the two reused anchors on the four formal gate metrics.",
            [rel(derived_dir / "ablation_main_joined.csv")],
            "Settings share the same backbone and formal evaluator; only the replay strategy differs.",
            "The bar panel exposes the relative ordering across A0-A4 under the fixed-budget replay regime.",
            "PTR remains a lower-is-better metric and should be interpreted accordingly.",
        ),
    )
    write_caption(
        captions_dir,
        CaptionSpec(
            "fig_lite_ablation_epoch_dynamics_panel",
            "Whether weighted replay changes the training trajectory rather than only the final best checkpoint.",
            [
                rel(summary_dirs["A0"] / "epoch_gate_summary.csv"),
                rel(summary_dirs["A1"] / "epoch_gate_summary.csv"),
                rel(summary_dirs["A4"] / "epoch_gate_summary.csv"),
            ],
            "Curves are plotted for hn00, uniform_hn14, and the full lite setting A4.",
            "The panel shows how the formal gate metrics evolve under the baseline, uniform HN, and weighted replay regimes.",
            "This figure focuses on representative settings rather than every lite variant to keep the main comparison readable.",
        ),
    )
    write_caption(
        captions_dir,
        CaptionSpec(
            "fig_budget_concentration_tripanel",
            "Whether the weighted variants are redistributing the replay budget smoothly or collapsing it onto a very small subset of the hard pool.",
            [rel(ctx["score_dir"] / "table_score_component_stats.csv")],
            "The tripanel compares the fixed top250 pool under the same one-shot hn00 teacher and budget.",
            "A3 and A4 concentrate much more heavily than A2, with substantially lower effective counts and much higher top10 cumulative replay mass.",
            "These are pool-level concentration diagnostics and do not by themselves prove which weighting policy should win downstream.",
        ),
    )
    write_caption(
        captions_dir,
        CaptionSpec(
            "fig_score_distribution_panel",
            "How the score distributions and replay probabilities differ across A2/A3/A4.",
            [
                rel(ctx["score_dir"] / "A2_candidate_pool_scores.csv"),
                rel(ctx["score_dir"] / "A3_candidate_pool_scores.csv"),
                rel(ctx["score_dir"] / "A4_candidate_pool_scores.csv"),
            ],
            "Scores are computed once from the hn00 teacher over the fixed top250 pool.",
            "The panel shows whether adding consistency and density increases or dampens replay concentration.",
            "This is a static one-shot view and does not cover dynamic score refresh.",
        ),
    )
    write_caption(
        captions_dir,
        CaptionSpec(
            "fig_pool_width_threshold_panel",
            "Whether the top250 hard-normal pool is a narrow extreme tail or a relatively wide high-risk region under the hn00 teacher.",
            [
                rel(ctx["score_dir"] / "pool_source_manifest.json"),
                rel(Path(load_json(ctx["score_dir"] / "pool_source_manifest.json")["pool_scores_csv"])),
            ],
            "The figure uses the hn00 teacher thresholds tau_r995 and tau_r990 together with the fixed top250 cutoff.",
            "The selected top250 region extends far below the formal threshold anchors, which supports the view that the hard-normal pool is wide rather than near-homogeneous.",
            "This figure summarizes the pool scores only and does not by itself identify which subregion is most learnable.",
        ),
    )
    write_caption(
        captions_dir,
        CaptionSpec(
            "fig_top_bottom_gallery_panel",
            "What the top-scored and bottom-scored hard normals look like under each lite scoring variant.",
            [
                rel(ctx["score_dir"] / "A2_candidate_pool_scores.csv"),
                rel(ctx["score_dir"] / "A3_candidate_pool_scores.csv"),
                rel(ctx["score_dir"] / "A4_candidate_pool_scores.csv"),
            ],
            "Each panel shows the top and bottom ranked candidates for one lite variant.",
            "The figure helps audit whether the weighted variants prioritize stable hard negatives rather than obviously anomalous outliers.",
            "Images are rendered directly from source paths at figure-build time and are not copied into the formal materials tree.",
        ),
    )
    write_caption(
        captions_dir,
        CaptionSpec(
            "fig_uniform_vs_weighted_overlap",
            "How much the weighted replay selections overlap with the existing uniform_hn14 replay set.",
            [
                rel(ctx["score_dir"] / "uniform_hn14_reference.csv"),
                rel(ctx["score_dir"] / "A2_candidate_pool_scores.csv"),
                rel(ctx["score_dir"] / "A3_candidate_pool_scores.csv"),
                rel(ctx["score_dir"] / "A4_candidate_pool_scores.csv"),
            ],
            "Uniform HN14 is compared against the selected unique samples from each weighted variant.",
            "The overlap view indicates whether weighted replay primarily reorders the same pool or surfaces a different subset of hard normals.",
            "The figure summarizes unique-sample overlap rather than replay-count overlap.",
        ),
    )
    write_caption(
        captions_dir,
        CaptionSpec(
            "fig_a1_vs_a4_spec995_delta",
            "Whether the best weighted variant reaches the same primary-metric platform as the uniform_hn14 anchor across training epochs.",
            [
                rel(summary_dirs["A1"] / "epoch_gate_summary.csv"),
                rel(summary_dirs["A4"] / "epoch_gate_summary.csv"),
            ],
            "A1 and A4 are compared on the same backbone, metric, and epoch axis; the lower panel shows A4 minus A1.",
            "A4 peaks earlier but stays below the A1 platform on the primary metric for most of training, which matches the final formal ranking gap.",
            "The comparison isolates Spec@R99.5 only and should still be read together with the full four-metric formal rule.",
        ),
    )

    build_summary_docs(
        output_root=output_root,
        main_df=main_df,
        delta_vs_hn00=delta_vs_hn00,
        delta_vs_uniform=delta_vs_uniform,
        score_stats=score_stats,
        overlap_df=overlap_df,
        ctx=ctx,
        cfg=cfg,
    )

    manifest = {
        "tables": sorted(path.name for path in tables_dir.glob("*")),
        "figures": sorted(path.name for path in figures_dir.glob("*")),
        "appendix": sorted(path.name for path in appendix_dir.glob("*")),
        "captions": sorted(path.name for path in captions_dir.glob("*")),
        "sources": {
            "hn_summary": rel(ctx["gate_hn_results"]),
            "teacher_summary_dir": rel(ctx["teacher_summary_dir"]),
            "uniform_summary_dir": rel(ctx["uniform_summary_dir"]),
            "score_dir": rel(ctx["score_dir"]),
            "suite_context_json": rel(ctx["suite_context_json"]),
        },
    }
    (manifests_dir / "paper_assets_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (manifests_dir / "paper_assets_manifest.md").write_text(
        "\n".join(
            [
                "# paper_assets_manifest",
                "",
                "## Sources",
                *[f"- {key}: `{value}`" for key, value in manifest["sources"].items()],
                "",
                "## Tables",
                *[f"- `{name}`" for name in manifest["tables"]],
                "",
                "## Figures",
                *[f"- `{name}`" for name in manifest["figures"]],
                "",
                "## Appendix",
                *[f"- `{name}`" for name in manifest["appendix"]],
                "",
                "## Captions",
                *[f"- `{name}`" for name in manifest["captions"]],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
