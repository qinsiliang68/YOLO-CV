"""Atomic tables-first report for performance-frontier analysis."""

from __future__ import annotations

import html
import json
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from .util import sha256_file


_CORE_TABLES = (
    "baseline_frontier_val_op",
    "baseline_frontier_test",
    "all_run_baseline_dominance",
    "paired_control_frontier_deltas",
    "designed_method_double_gates",
    "method_repeatability_ranking",
    "hypothesis_registry",
)


def _require_tables(tables: dict[str, pd.DataFrame]) -> None:
    missing = sorted(set(_CORE_TABLES).difference(tables))
    if missing:
        raise ValueError(f"Performance report missing tables: {missing}")


def _save_baseline_frontiers(tables: dict[str, pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, (name, title) in zip(
        axes,
        (
            ("baseline_frontier_val_op", "Zero-replay baseline: val_op"),
            ("baseline_frontier_test", "Zero-replay baseline: development benchmark"),
        ),
    ):
        frame = tables[name]
        zoom = frame.loc[frame["fn_budget"] <= min(200, frame["fn_budget"].max())]
        axis.plot(zoom["fn_budget"], zoom["TN"], color="#155EEF", linewidth=2)
        axis.set_title(title)
        axis.set_xlabel("FN budget")
        axis.set_ylabel("Maximum TN [Zoomed y-axis]")
        axis.grid(alpha=0.25)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _save_class_counts(frame: pd.DataFrame, output: Path) -> None:
    counts = (
        frame.groupby(["experiment_family", "performance_class"])["run_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    counts.plot(kind="bar", stacked=True, ax=axis, colormap="tab20")
    axis.set_title("Absolute frontier classes versus zero-replay baseline")
    axis.set_xlabel("Experiment family")
    axis.set_ylabel("Run count")
    axis.legend(title="Performance class", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _save_double_gate(frame: pd.DataFrame, output: Path) -> None:
    counts = (
        frame.groupby(["experiment_family", "outcome_cohort"])["run_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    counts.plot(kind="bar", stacked=True, ax=axis, colormap="Set2")
    axis.set_title("Outcome-first cohorts after baseline + random-control gates")
    axis.set_xlabel("Experiment family")
    axis.set_ylabel("Designed Treatment count")
    axis.legend(title="Outcome cohort", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _save_delta_scatter(frame: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for cohort, group in frame.groupby("outcome_cohort", sort=True):
        axis.scatter(
            group["safe_min_delta_TN"],
            group["delta_TN_at_baseline_fn"],
            label=str(cohort),
            alpha=0.75,
            s=38,
        )
    axis.axhline(0, color="black", linewidth=0.9)
    axis.axvline(0, color="black", linewidth=0.9)
    axis.set_title("Designed methods: safe-range worst case vs baseline operating FN")
    axis.set_xlabel("Minimum delta TN across baseline-safe FN range")
    axis.set_ylabel("Delta TN at baseline FN")
    axis.legend(title="Outcome cohort", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.grid(alpha=0.25)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _table_preview(frame: pd.DataFrame, *, rows: int = 20) -> str:
    preview = frame.head(rows).copy()
    return preview.to_html(index=False, border=0, classes="data-table", escape=True)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    clean = frame.fillna("").astype(str).map(lambda value: value.replace("|", "\\|"))
    header = "| " + " | ".join(clean.columns) + " |"
    separator = "| " + " | ".join("---" for _ in clean.columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in clean.to_numpy().tolist()]
    return "\n".join([header, separator, *rows])


def _build_markdown(tables: dict[str, pd.DataFrame], metadata: dict[str, Any]) -> str:
    all_runs = tables["all_run_baseline_dominance"]
    designed = tables["designed_method_double_gates"]
    class_counts = all_runs["performance_class"].value_counts().to_dict()
    cohort_counts = designed["outcome_cohort"].value_counts().to_dict()
    strong = designed.loc[
        designed["outcome_cohort"].isin(
            ["ROBUST_SAFE_DOUBLE_GATE", "LOCAL_PARETO_DOUBLE_GATE"]
        )
    ].copy()
    strong = strong.sort_values("delta_TN_at_baseline_fn", ascending=False)
    harmful = designed.loc[designed["outcome_cohort"] == "JOINTLY_HARMFUL"].copy()
    hypotheses = tables["hypothesis_registry"]
    lines = [
        "# Stage1 全量性能前沿与强训练方式分析",
        "",
        "## 结论口径",
        "",
        "所有候选都与零操作 `yolo11l best.pt` 在相同 FN 上限下比较。",
        "240-run 的设计方法还必须分别胜过同 seed 的 R1 与 R2；固定阈值平移不计为性能提升。",
        "",
        "## 数据完整性",
        "",
        f"- 候选 run：{metadata.get('candidate_runs', len(all_runs))}",
        f"- 分实验族：{metadata.get('run_counts', {})}",
        f"- 绝对性能分档：{class_counts}",
        f"- 双门结果分档：{cohort_counts}",
        "",
        "## 双门真实增益（区分稳健安全前沿与局部Pareto点）",
        "",
    ]
    if strong.empty:
        lines.append("当前没有训练方式同时通过零操作基线和随机对照双门。")
    else:
        keep = [
            column
            for column in (
                "experiment_family",
                "run_id",
                "condition_id",
                "training_seed",
                "performance_class",
                "delta_TN_at_baseline_fn",
                "safe_min_delta_TN",
                "safe_positive_budget_share",
            )
            if column in strong.columns
        ]
        lines.append(_markdown_table(strong[keep]))
    lines.extend(
        [
            "",
            "## 明确有害结果",
            "",
            (
                "没有同时被零操作基线和随机对照压制的Treatment。"
                if harmful.empty
                else harmful[
                    [
                        column
                        for column in (
                            "experiment_family",
                            "run_id",
                            "condition_id",
                            "performance_class",
                            "delta_TN_at_baseline_fn",
                            "safe_min_delta_TN",
                        )
                        if column in harmful.columns
                    ]
                ].pipe(_markdown_table)
            ),
            "",
            "## 假设判定",
            "",
            _markdown_table(hypotheses),
            "",
            "## 解释边界",
            "",
        ]
    )
    for boundary in metadata.get("scientific_boundaries", []):
        lines.append(f"- {boundary}")
    lines.extend(
        [
            "",
            "强结果表示当前数据内部的真实性能前沿改善，不等于blind/external泛化确认。",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_html(tables: dict[str, pd.DataFrame], metadata: dict[str, Any]) -> str:
    all_runs = tables["all_run_baseline_dominance"]
    designed = tables["designed_method_double_gates"]
    strong = designed.loc[
        designed["outcome_cohort"].isin(
            ["ROBUST_SAFE_DOUBLE_GATE", "LOCAL_PARETO_DOUBLE_GATE"]
        )
    ].sort_values("delta_TN_at_baseline_fn", ascending=False)
    harmful = designed.loc[designed["outcome_cohort"] == "JOINTLY_HARMFUL"].sort_values(
        "delta_TN_at_baseline_fn"
    )
    hypotheses = tables["hypothesis_registry"]
    boundaries = "".join(
        f"<li>{html.escape(str(value))}</li>"
        for value in metadata.get("scientific_boundaries", [])
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Stage1性能前沿分析</title>
<style>
body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:0;background:#f5f7fb;color:#17202a}}
main{{max-width:1440px;margin:auto;padding:28px}} section{{background:white;padding:22px;margin:18px 0;border-radius:10px;box-shadow:0 2px 10px #00000010}}
h1,h2{{color:#173f7a}} .notice{{border-left:5px solid #155eef;background:#eef4ff;padding:14px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}} img{{width:100%;height:auto}}
.data-table{{border-collapse:collapse;width:100%;font-size:12px}} .data-table th,.data-table td{{border:1px solid #d8dee9;padding:5px;text-align:right}} .data-table th{{background:#edf2f7;position:sticky;top:0}}
.scroll{{overflow:auto;max-height:620px}} code{{background:#eef1f5;padding:2px 5px}}
</style></head><body><main>
<h1>Stage1 全量性能前沿与强训练方式分析</h1>
<div class="notice">所有比较都在相同 FN 上限下进行。零操作 yolo11l 基线与随机对照双门同时通过，才称为性能偏强；单纯移动置信度阈值不会进入强组。</div>
<section><h2>完整性</h2><p>候选run：{metadata.get('candidate_runs', len(all_runs))}；分实验族：{html.escape(str(metadata.get('run_counts', {})))}</p></section>
<section><h2>结果总览</h2><div class="grid"><img src="charts/absolute_class_counts.png" alt="absolute classes"><img src="charts/double_gate_cohorts.png" alt="double gate cohorts"><img src="charts/designed_delta_scatter.png" alt="designed delta scatter"><img src="charts/baseline_frontiers_zoomed.png" alt="baseline frontiers"></div></section>
<section><h2>双门真实增益：稳健安全前沿与局部Pareto点分开</h2><div class="scroll">{_table_preview(strong, rows=80)}</div><p><a href="tables/strong_secondary_harmful_cohorts.csv">下载完整双门分档CSV</a></p></section>
<section><h2>明确有害结果</h2><div class="scroll">{_table_preview(harmful, rows=80)}</div></section>
<section><h2>假设判定</h2><div class="scroll">{_table_preview(hypotheses, rows=30)}</div><p><a href="tables/hypothesis_registry.csv">下载假设判定 CSV</a></p></section>
<section><h2>全部run相对基线</h2><div class="scroll">{_table_preview(all_runs.sort_values(['experiment_family','delta_TN_at_baseline_fn'], ascending=[True,False]), rows=120)}</div><p><a href="tables/all_run_baseline_dominance.csv">下载全部run CSV</a></p></section>
<section><h2>科学边界</h2><ul>{boundaries}</ul></section>
</main></body></html>"""


def build_performance_report(
    output_dir: str | Path,
    *,
    tables: dict[str, pd.DataFrame],
    metadata: dict[str, Any],
) -> Path:
    """Write a non-overwriting report via a sibling .inprogress directory."""

    _require_tables(tables)
    output = Path(output_dir).resolve()
    staging = output.with_name(output.name + ".inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"Refusing to overwrite report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        table_dir = staging / "tables"
        chart_dir = staging / "charts"
        audit_dir = staging / "audit"
        table_dir.mkdir()
        chart_dir.mkdir()
        audit_dir.mkdir()
        for name, frame in sorted(tables.items()):
            frame.to_csv(table_dir / f"{name}.csv", index=False)

        _save_baseline_frontiers(tables, chart_dir / "baseline_frontiers_zoomed.png")
        _save_class_counts(
            tables["all_run_baseline_dominance"], chart_dir / "absolute_class_counts.png"
        )
        _save_double_gate(
            tables["designed_method_double_gates"], chart_dir / "double_gate_cohorts.png"
        )
        designed = tables["designed_method_double_gates"]
        if {"safe_min_delta_TN", "delta_TN_at_baseline_fn", "outcome_cohort"}.issubset(designed.columns):
            _save_delta_scatter(designed, chart_dir / "designed_delta_scatter.png")
        else:
            # Small synthetic reporting tests may omit the full science columns.
            fig, axis = plt.subplots(figsize=(6, 3), constrained_layout=True)
            axis.text(0.5, 0.5, "No delta scatter fields", ha="center", va="center")
            axis.set_axis_off()
            fig.savefig(chart_dir / "designed_delta_scatter.png", dpi=120)
            plt.close(fig)

        markdown = _build_markdown(tables, metadata)
        (staging / "FINAL_REPORT_CN.md").write_text(markdown, encoding="utf-8")
        (staging / "README.md").write_text(
            "# Performance frontier report\n\n"
            "Open `index.html`. Tables are the source of truth; charts are views.\n",
            encoding="utf-8",
        )
        (staging / "index.html").write_text(
            _build_html(tables, metadata), encoding="utf-8"
        )
        contract = {
            **metadata,
            "threshold_rule": "predict defect when score >= threshold",
            "frontier_rule": "maximum TN at the same FN budget; whole tie groups",
            "overwrite_policy": "forbidden",
            "source_policy": "read-only",
        }
        (staging / "analysis_contract.yaml").write_text(
            yaml.safe_dump(contract, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        (audit_dir / "report_validation.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "table_count": len(tables),
                    "core_tables": list(_CORE_TABLES),
                    "charts": sorted(path.name for path in chart_dir.glob("*.png")),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        files = []
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                files.append(
                    {
                        "relative_path": path.relative_to(staging).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        manifest = {
            "status": "PASS",
            "analysis_id": metadata.get("analysis_id", "unknown"),
            "file_count_excluding_manifest": len(files),
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return output
