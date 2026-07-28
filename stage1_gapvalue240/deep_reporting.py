from __future__ import annotations

import html
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .util import atomic_write_json, atomic_write_text, atomic_write_yaml, sha256_file


REPORT_SCHEMA_VERSION = "stage1_gapvalue240_deep_report_v1"
_SAFE_TABLE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class ChartSpec:
    filename: str
    title: str
    source_tables: tuple[str, ...]
    note_cn: str


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def _load_table(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    path = Path(value)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table input: {path}")


def _table_stem(name: str) -> str:
    if not name or not _SAFE_TABLE_NAME.fullmatch(name):
        raise ValueError(f"Unsafe table name: {name!r}")
    stem = Path(name).stem
    if stem in {"", ".", ".."}:
        raise ValueError(f"Unsafe table name: {name!r}")
    return stem


def _expanded_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return -1.0, 1.0
    low, high = float(finite.min()), float(finite.max())
    span = high - low
    padding = span * 0.12 if span else max(abs(low) * 0.12, 1.0)
    return low - padding, high + padding


def _save_hypothesis_chart(hypotheses: pd.DataFrame, output: Path) -> None:
    statuses = [
        "SUPPORTED",
        "NOT_SUPPORTED",
        "INCONCLUSIVE",
        "NOT_TESTABLE",
    ]
    counts = Counter(str(value).upper() for value in hypotheses.get("status", []))
    values = [counts.get(status, 0) for status in statuses]
    colors = ["#238636", "#cf222e", "#bf8700", "#6e7781"]

    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    bars = ax.bar(statuses, values, color=colors)
    ax.set_title("Hypothesis evidence status")
    ax.set_ylabel("Hypothesis count")
    ax.set_xlabel("Evidence classification")
    ax.set_ylim(0, max(values + [1]) * 1.22)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _save_condition_effect_chart(summary: pd.DataFrame, output: Path) -> bool:
    required = {"condition_id", "mean_delta_TN", "mean_delta_FN"}
    if not required.issubset(summary.columns) or summary.empty:
        return False
    plot = summary.copy()
    control = plot["control"].astype(str) if "control" in plot else pd.Series([""] * len(plot))
    plot["_label"] = plot["condition_id"].astype(str) + np.where(control.ne(""), "/" + control, "")
    tn = pd.to_numeric(plot["mean_delta_TN"], errors="coerce").to_numpy(dtype=float)
    fn = pd.to_numeric(plot["mean_delta_FN"], errors="coerce").to_numpy(dtype=float)
    x = np.arange(len(plot))

    fig, axes = plt.subplots(2, 1, figsize=(max(10.0, len(plot) * 0.72), 8.2), sharex=True)
    for ax, values, title, ylabel, color in [
        (axes[0], tn, "Treatment effect on TN at FN<=95", "Mean delta TN (higher is better)", "#0969da"),
        (axes[1], fn, "Treatment effect on FN at TN>=68253", "Mean delta FN (lower is better)", "#cf222e"),
    ]:
        ax.axhline(0, color="#57606a", linewidth=1, linestyle="--")
        ax.scatter(x, values, s=48, color=color, zorder=3)
        ax.set_title(f"{title} [Zoomed y-axis]")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*_expanded_limits(values))
        ax.grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x, plot["_label"], rotation=45, ha="right")
    axes[1].set_xlabel("Condition/control")
    fig.suptitle("Condition-level paired effects (zoomed independently; see CSV for raw values)")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return True


def _save_a02_seed_forest(frame: pd.DataFrame, output: Path) -> bool:
    required = {"training_seed", "control", "delta_TN", "delta_FN"}
    if not required.issubset(frame.columns) or frame.empty:
        return False
    plot = frame.copy()
    condition_column = next(
        (column for column in ("condition_slot", "condition_id") if column in plot),
        None,
    )
    if condition_column is not None:
        plot = plot.loc[plot[condition_column].astype(str).eq("A02")].copy()
    if plot.empty:
        return False
    stage_column = next(
        (column for column in ("analysis_cohort", "stage", "discovery_or_confirmation") if column in plot),
        None,
    )
    stage = (
        plot[stage_column].astype(str)
        if stage_column is not None
        else pd.Series([""] * len(plot), index=plot.index)
    )
    plot["_label"] = (
        stage.str.slice(0, 4)
        + "/s"
        + plot["training_seed"].astype(str)
        + "/"
        + plot["control"].astype(str)
    )
    y = np.arange(len(plot))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, max(5.5, len(plot) * 0.38)), sharey=True)
    for ax, column, title, color in [
        (axes[0], "delta_TN", "A02 seed effects: delta TN", "#0969da"),
        (axes[1], "delta_FN", "A02 seed effects: delta FN", "#cf222e"),
    ]:
        values = pd.to_numeric(plot[column], errors="coerce").to_numpy(dtype=float)
        ax.axvline(0, color="#57606a", linewidth=1, linestyle="--")
        ax.scatter(values, y, color=color, s=38)
        ax.set_xlim(*_expanded_limits(np.append(values, 0.0)))
        ax.set_title(f"{title} [Zoomed x-axis]")
        ax.set_xlabel(f"{column} ({'higher' if column == 'delta_TN' else 'lower'} is better)")
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y, plot["_label"])
    axes[0].invert_yaxis()
    fig.suptitle("A02 paired seed forest (points are seed/control effects; no pooled CI shown)")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return True


def _save_condition_pareto(frame: pd.DataFrame, output: Path) -> bool:
    required = {"condition_id", "mean_delta_TN", "mean_delta_FN"}
    if not required.issubset(frame.columns) or frame.empty:
        return False
    x = pd.to_numeric(frame["mean_delta_FN"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame["mean_delta_TN"], errors="coerce").to_numpy(dtype=float)
    labels = frame["condition_id"].astype(str)
    if "control" in frame:
        labels = labels + "/" + frame["control"].astype(str)
    fig, ax = plt.subplots(figsize=(9.4, 6.4), constrained_layout=True)
    ax.axvline(0, color="#57606a", linewidth=1, linestyle="--")
    ax.axhline(0, color="#57606a", linewidth=1, linestyle="--")
    ax.scatter(x, y, color="#8250df", s=50)
    for x_value, y_value, label in zip(x, y, labels):
        ax.annotate(label, (x_value, y_value), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlim(*_expanded_limits(np.append(x, 0.0)))
    ax.set_ylim(*_expanded_limits(np.append(y, 0.0)))
    ax.set_title("Condition safety/utility plane [Zoomed axes]")
    ax.set_xlabel("Mean delta FN (left is safer)")
    ax.set_ylabel("Mean delta TN (higher is better)")
    ax.grid(alpha=0.22)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return True


def _save_r2_contrast(frame: pd.DataFrame, output: Path) -> bool:
    value_column = next(
        (
            column
            for column in (
                "effective_unique_contrast_rate",
                "unique_contrast_rate",
                "effective_unique_rate",
            )
            if column in frame.columns
        ),
        None,
    )
    label_column = next(
        (column for column in ("condition_id", "condition_slot") if column in frame),
        None,
    )
    if value_column is None or label_column is None or frame.empty:
        return False
    plot = (
        frame.assign(
            _value=pd.to_numeric(frame[value_column], errors="coerce"),
            _label=frame[label_column].astype(str),
        )
        .groupby("_label", sort=True)["_value"]
        .mean()
        .reset_index()
    )
    values = plot["_value"].to_numpy(dtype=float)
    labels = plot["_label"]
    fig, ax = plt.subplots(
        figsize=(max(8.5, len(plot) * 0.55), 5.5),
        constrained_layout=True,
    )
    bars = ax.bar(np.arange(len(plot)), values * 100, color="#bf8700")
    ax.set_xticks(np.arange(len(plot)), labels, rotation=45, ha="right")
    ax.set_ylabel("Effective unique contrast (%)")
    ax.set_xlabel("Condition")
    ax.set_title("R2 effective unique contrast (zero-based axis)")
    ax.set_ylim(0, max(float(np.nanmax(values * 100)) * 1.22, 1.0))
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 100, f"{value:.1%}", ha="center", va="bottom", fontsize=8)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return True


def _save_two_metric_response(
    frame: pd.DataFrame,
    output: Path,
    *,
    x_column: str,
    title: str,
) -> bool:
    tn_column = next(
        (column for column in ("mean_delta_TN", "mean_diff_delta_TN") if column in frame),
        None,
    )
    fn_column = next(
        (column for column in ("mean_delta_FN", "mean_diff_delta_FN") if column in frame),
        None,
    )
    if x_column not in frame or tn_column is None or fn_column is None or frame.empty:
        return False
    numeric_x = pd.to_numeric(frame[x_column], errors="coerce")
    if numeric_x.notna().all():
        plot = frame.assign(_x=numeric_x).sort_values("_x")
        x = plot["_x"].to_numpy(dtype=float)
        x_labels = None
    else:
        plot = frame.reset_index(drop=True)
        x = np.arange(len(plot), dtype=float)
        x_labels = plot[x_column].astype(str)
        if "reference_condition" in plot:
            x_labels = plot["reference_condition"].astype(str) + " to " + x_labels
        if "control" in plot:
            x_labels = x_labels + "/" + plot["control"].astype(str)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.8), sharex=True)
    for ax, column, color, direction in [
        (axes[0], tn_column, "#0969da", "higher is better"),
        (axes[1], fn_column, "#cf222e", "lower is better"),
    ]:
        values = pd.to_numeric(plot[column], errors="coerce").to_numpy(dtype=float)
        ax.axhline(0, color="#57606a", linewidth=1, linestyle="--")
        ax.plot(x, values, color=color, marker="o")
        ax.set_ylim(*_expanded_limits(np.append(values, 0.0)))
        ax.set_ylabel(f"{column}\n({direction})")
        ax.set_title(f"{column} [Zoomed y-axis]")
        ax.grid(alpha=0.25)
    axes[1].set_xlabel(x_column)
    if x_labels is not None:
        axes[1].set_xticks(x, x_labels, rotation=35, ha="right")
    fig.suptitle(f"{title} (exact values in source CSV)")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return True


def _save_tail_shift(frame: pd.DataFrame, output: Path) -> bool:
    required = {"condition_id", "delta_normal_tail_score", "delta_defect_tail_score"}
    if not required.issubset(frame.columns) or frame.empty:
        return False
    x = np.arange(len(frame))
    width = 0.38
    normal = pd.to_numeric(frame["delta_normal_tail_score"], errors="coerce").to_numpy(dtype=float)
    defect = pd.to_numeric(frame["delta_defect_tail_score"], errors="coerce").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(9.0, len(frame) * 0.65), 5.8), constrained_layout=True)
    ax.axhline(0, color="#57606a", linewidth=1, linestyle="--")
    ax.bar(x - width / 2, normal, width, label="Normal tail score shift", color="#0969da")
    ax.bar(x + width / 2, defect, width, label="Defect tail score shift", color="#cf222e")
    ax.set_xticks(x, frame["condition_id"].astype(str), rotation=45, ha="right")
    ax.set_ylim(*_expanded_limits(np.append(np.concatenate([normal, defect]), 0.0)))
    ax.set_ylabel("Mean raw-score shift")
    ax.set_xlabel("Condition")
    ax.set_title("Fixed-tail raw-score movement [Zoomed y-axis]")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return True


def _save_threshold_frontier(frame: pd.DataFrame, output: Path) -> bool:
    required = {"arm", "FN", "TN"}
    if not required.issubset(frame.columns) or frame.empty:
        return False
    fig, ax = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    series_column = "series" if "series" in frame.columns else "arm"
    for arm, group in frame.groupby(series_column, sort=True):
        ordered = group.sort_values("FN")
        ax.plot(ordered["FN"], ordered["TN"], marker="o", label=str(arm))
    fn_values = pd.to_numeric(frame["FN"], errors="coerce").to_numpy(dtype=float)
    tn_values = pd.to_numeric(frame["TN"], errors="coerce").to_numpy(dtype=float)
    ax.set_xlim(*_expanded_limits(fn_values))
    ax.set_ylim(*_expanded_limits(tn_values))
    ax.set_title("A02 operational threshold frontier [Zoomed axes]")
    ax.set_xlabel("False negatives")
    ax.set_ylabel("True negatives")
    ax.legend(title="Cohort/arm" if series_column == "series" else "Arm")
    ax.grid(alpha=0.25)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return True


def _save_training_curves(frame: pd.DataFrame, output: Path) -> bool:
    required = {"arm", "epoch", "top1", "val_loss"}
    if not required.issubset(frame.columns) or frame.empty:
        return False
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 8.0), sharex=True)
    series_column = "series" if "series" in frame.columns else "arm"
    for arm, group in frame.groupby(series_column, sort=True):
        ordered = group.sort_values("epoch")
        axes[0].plot(ordered["epoch"], ordered["top1"], label=str(arm))
        axes[1].plot(ordered["epoch"], ordered["val_loss"], label=str(arm))
    for ax, column, direction in [
        (axes[0], "top1", "higher is better"),
        (axes[1], "val_loss", "lower is better"),
    ]:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        ax.set_ylim(*_expanded_limits(values))
        ax.set_ylabel(f"{column} ({direction})")
        ax.set_title(f"A02 {column} [Zoomed y-axis]")
        ax.grid(alpha=0.25)
        ax.legend(title="Cohort/arm" if series_column == "series" else "Arm")
    axes[1].set_xlabel("Epoch")
    fig.suptitle("A02 200-epoch training dynamics")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return True


def _save_reliability(frame: pd.DataFrame, output: Path) -> bool:
    required = {"machine_id", "resume_count"}
    if not required.issubset(frame.columns) or frame.empty:
        return False
    machines = frame["machine_id"].astype(str).value_counts().sort_index()
    resumes = pd.to_numeric(frame["resume_count"], errors="coerce").fillna(0).astype(int).value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    axes[0].bar(machines.index, machines.values, color="#0969da")
    axes[0].set_title("Validated runs by machine")
    axes[0].set_ylabel("Run count")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(resumes.index.astype(str), resumes.values, color="#8250df")
    axes[1].set_title("Runs by native resume count")
    axes[1].set_ylabel("Run count")
    axes[1].set_xlabel("Resume count")
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Run reliability overview (zero-based count axes)")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return True


def _table_preview(frame: pd.DataFrame, max_rows: int = 20) -> str:
    preview = frame.head(max_rows)
    note = ""
    if len(frame) > max_rows:
        note = f"<p class=\"muted\">Previewing {max_rows:,} of {len(frame):,} rows.</p>"
    return note + preview.to_html(index=False, border=0, classes=["data-table"])


def _load_contract(value: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    if value is None:
        return {
            "schema_version": "stage1_gapvalue240_analysis_contract_v1",
            "conclusion_scope": "val_op",
            "blind_external_claim_allowed": False,
        }
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"analysis_contract root must be a mapping: {path}")
    return data


def _normalize_narrative(value: Mapping[str, str] | str | None) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, str):
        return {"分析说明": value}
    return {str(key): str(text) for key, text in value.items()}


def _assert_no_suspicious_question_runs(value: Any, *, context: str) -> None:
    """Reject likely encoding-loss placeholders before any report files are written."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_suspicious_question_runs(key, context=context)
            _assert_no_suspicious_question_runs(item, context=context)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_no_suspicious_question_runs(item, context=context)
        return
    if isinstance(value, str) and re.search(r"\?{3,}", value):
        raise UnicodeError(
            f"Suspicious question-mark run found in report {context}; "
            "this usually indicates encoding loss"
        )


def _write_audits(
    audit_dir: Path,
    audits: Mapping[str, pd.DataFrame | Mapping[str, Any] | Sequence[Any] | str | Path],
) -> list[str]:
    written: list[str] = []
    for name, value in audits.items():
        stem = _table_stem(str(name))
        if isinstance(value, pd.DataFrame):
            path = audit_dir / f"{stem}.csv"
            value.to_csv(path, index=False, lineterminator="\n")
        elif isinstance(value, Mapping) or (
            isinstance(value, Sequence) and not isinstance(value, (str, bytes, Path))
        ):
            path = audit_dir / f"{stem}.json"
            atomic_write_json(path, value)
        elif isinstance(value, Path):
            path = audit_dir / f"{stem}{value.suffix.lower() or '.txt'}"
            atomic_write_text(path, value.read_text(encoding="utf-8"))
        else:
            path = audit_dir / f"{stem}.txt"
            atomic_write_text(path, str(value))
        written.append(path.name)
    return written


def _assert_no_replacement_characters(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".html", ".md", ".json", ".yaml", ".yml", ".txt", ".csv"}:
            text = path.read_text(encoding="utf-8")
            if "\ufffd" in text:
                raise UnicodeError(f"UTF-8 replacement character found in generated report: {path}")


def _write_markdown(
    output: Path,
    *,
    title: str,
    metadata: Mapping[str, Any],
    hypotheses: pd.DataFrame,
    table_names: list[str],
    chart_specs: list[ChartSpec],
    narrative: Mapping[str, str],
    audit_names: list[str],
) -> None:
    status_counts = Counter(str(value).upper() for value in hypotheses.get("status", []))
    lines = [
        f"# {title}",
        "",
        "本报告仅支持 val_op 内部发现与复现；没有 blind/external test，不能宣称外部泛化已经确认。",
        "",
        "## 分析边界",
        "",
        "- 图中的 treatment effect 使用配对差值；R1 与 R2 必须分别解释。",
        "- 所有标记为 zoomed 的纵轴都不从零开始，原始数值以 CSV 表为准。",
        "- 机器、resume、snapshot 等混杂必须结合上游敏感性表判断。",
        "",
        "## 数据概览",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in metadata.items())
    for heading, paragraph in narrative.items():
        lines.extend(["", f"## {heading}", "", paragraph])
    lines.extend(["", "## 假设证据状态", ""])
    lines.extend(
        f"- `{status}`: {status_counts.get(status, 0)}"
        for status in ["SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE", "NOT_TESTABLE"]
    )
    lines.extend(["", "## 产物", ""])
    lines.append("- 分析合同：`analysis_contract.yaml`")
    lines.extend(f"- 审计：`audit/{name}`" for name in audit_names)
    lines.extend(f"- 表：`tables/{name}.csv`" for name in table_names)
    for chart in chart_specs:
        sources = "、".join(f"`tables/{name}.csv`" for name in chart.source_tables)
        lines.append(f"- 图：`charts/{chart.filename}`；来源表：{sources}；{chart.note_cn}")
    lines.append("")
    atomic_write_text(output, "\n".join(lines))


def _write_readme(
    output: Path,
    *,
    table_names: list[str],
    chart_specs: list[ChartSpec],
    audit_names: list[str],
) -> None:
    text = f"""# Stage1 GapValue 深度分析静态报告

本目录是可再生成的 experiment output，不是训练、selection 或人工真相源。

- `index.html`：主阅读入口。
- `FINAL_REPORT_CN.md`：中文文本摘要与结论边界。
- `tables/`：报告使用的完整 CSV，共 {len(table_names)} 份。
- `charts/`：由表格生成的 PNG，共 {len(chart_specs)} 张。
- `audit/`：上游提供的只读审计副本，共 {len(audit_names)} 份。
- `analysis_contract.yaml`：本报告使用的分析边界和判定合同。
- `manifest.json`：除自身外所有文件的大小和 SHA-256。

生成规则：

1. 生成器先写同级 `.inprogress` 目录；
2. 所有文件和清单完成后才将目录原子改名；
3. 已存在的正式目录或 `.inprogress` 目录都不会被覆盖；
4. 图中 zoomed axis 均明确标注，精确值以 CSV 为准。

生命周期：报告可由冻结表格重新生成；不得反向修改训练产物或 selection CSV。
"""
    atomic_write_text(output, text)


def _write_html(
    output: Path,
    *,
    title: str,
    metadata: Mapping[str, Any],
    hypotheses: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    chart_specs: list[ChartSpec],
    narrative: Mapping[str, str],
    audit_names: list[str],
) -> None:
    metadata_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in metadata.items()
    )
    hypothesis_html = _table_preview(hypotheses, max_rows=100)
    narrative_html = "".join(
        f"<section><h2>{html.escape(heading)}</h2><p>{html.escape(paragraph)}</p></section>"
        for heading, paragraph in narrative.items()
    )
    chart_parts = []
    for chart in chart_specs:
        links = " ".join(
            f'<a href="tables/{html.escape(name)}.csv">{html.escape(name)}.csv</a>'
            for name in chart.source_tables
        )
        chart_parts.append(
            f'<figure><h3>{html.escape(chart.title)}</h3>'
            f'<img src="charts/{html.escape(chart.filename)}" alt="{html.escape(chart.filename)}">'
            f"<figcaption>{html.escape(chart.note_cn)} Zoomed axes are explicitly labelled where used. "
            f"Source table(s): {links}</figcaption></figure>"
        )
    chart_html = "".join(chart_parts)
    table_html = "".join(
        f'<section><h3>{html.escape(name)}</h3><p><a href="tables/{html.escape(name)}.csv">'
        f"Download complete CSV ({len(frame):,} rows)</a></p>{_table_preview(frame)}</section>"
        for name, frame in tables.items()
    )
    audit_links = " ".join(
        f'<a href="audit/{html.escape(name)}">{html.escape(name)}</a>'
        for name in audit_names
    ) or "No additional audit files supplied."
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body{{font-family:Arial,"Microsoft YaHei",sans-serif;margin:0;color:#1f2328;background:#f6f8fa}}
main{{max-width:1440px;margin:auto;background:white;padding:28px 42px}}
h1,h2,h3{{color:#24292f}} .warning{{border-left:5px solid #bf8700;background:#fff8c5;padding:12px}}
.muted,figcaption{{color:#57606a;font-size:13px}} table{{border-collapse:collapse;width:100%;font-size:12px}}
th,td{{border:1px solid #d0d7de;padding:5px;text-align:left}} th{{background:#f6f8fa}}
figure{{margin:24px 0}} img{{max-width:100%;height:auto;border:1px solid #d8dee4}}
a{{color:#0969da}} section{{margin:28px 0;overflow-x:auto}}
</style>
</head>
<body><main>
<h1>{html.escape(title)}</h1>
<p class="warning">结论边界：本报告仅支持 val_op 内部发现与复现；没有 blind/external test，
不能宣称外部泛化已经确认。R1、R2 分别解释，机器混杂不得被统计汇总掩盖。</p>
<h2>分析元数据</h2><table>{metadata_rows}</table>
<p><a href="analysis_contract.yaml">Download analysis contract</a></p>
<p>Audit evidence: {audit_links}</p>
{narrative_html}
<h2>假设证据登记</h2>{hypothesis_html}
<h2>图表</h2>
<p class="muted">Zoomed y-axes are labelled in every affected panel. Exact values are in the linked CSV files.</p>
{chart_html}
<h2>完整分析表</h2>{table_html}
</main></body></html>
"""
    atomic_write_text(output, document)


def _build_manifest(
    root: Path,
    *,
    metadata: Mapping[str, Any],
    table_count: int,
    chart_count: int,
    audit_count: int,
) -> dict[str, Any]:
    files = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.name == "manifest.json":
            continue
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "lifecycle": "regenerate",
        "metadata": dict(metadata),
        "counts": {
            "tables": table_count,
            "charts": chart_count,
            "audits": audit_count,
            "files_excluding_manifest": len(files),
        },
        "files": files,
        "manifest_self_excluded": True,
    }


def build_deep_report(
    output_dir: str | Path,
    *,
    tables: Mapping[str, pd.DataFrame | str | Path],
    metadata: Mapping[str, Any],
    hypothesis_registry: pd.DataFrame | str | Path,
    title: str = "Stage1 GapValue 240-Run 全面深度分析",
    analysis_contract: Mapping[str, Any] | str | Path | None = None,
    audits: Mapping[
        str, pd.DataFrame | Mapping[str, Any] | Sequence[Any] | str | Path
    ] | None = None,
    narrative: Mapping[str, str] | str | None = None,
) -> Path:
    """Build a non-overwriting, self-contained static analysis report.

    The output is assembled under ``<output>.inprogress`` and made visible only
    after every table, chart, document, and file hash has been written.
    """

    output = Path(output_dir)
    partial = output.with_name(output.name + ".inprogress")
    narrative_sections = _normalize_narrative(narrative)
    contract = _load_contract(analysis_contract)
    _assert_no_suspicious_question_runs(title, context="title")
    _assert_no_suspicious_question_runs(metadata, context="metadata")
    _assert_no_suspicious_question_runs(narrative_sections, context="narrative")
    _assert_no_suspicious_question_runs(contract, context="analysis contract")
    for candidate in (output, partial):
        if candidate.exists():
            raise FileExistsError(f"Refusing to overwrite: {candidate}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    (partial / "tables").mkdir()
    (partial / "charts").mkdir()
    (partial / "audit").mkdir()

    materialized: dict[str, pd.DataFrame] = {}
    for name, source in tables.items():
        stem = _table_stem(str(name))
        if stem == "hypothesis_registry" or stem in materialized:
            raise ValueError(f"Duplicate or reserved table name: {stem}")
        materialized[stem] = _load_table(source)
    hypotheses = _load_table(hypothesis_registry)
    if "status" not in hypotheses.columns:
        raise ValueError("hypothesis_registry must contain a status column")
    materialized["hypothesis_registry"] = hypotheses

    for name, frame in materialized.items():
        frame.to_csv(partial / "tables" / f"{name}.csv", index=False, lineterminator="\n")

    chart_specs = [
        ChartSpec(
            "hypothesis_status.png",
            "Hypothesis evidence status",
            ("hypothesis_registry",),
            "四级证据判定数量，计数轴从零开始。",
        )
    ]
    _save_hypothesis_chart(hypotheses, partial / "charts" / chart_specs[0].filename)

    def add_chart(
        filename: str,
        title_text: str,
        source_tables: tuple[str, ...],
        note_cn: str,
        renderer,
    ) -> None:
        frames = [materialized.get(name) for name in source_tables]
        if any(frame is None for frame in frames):
            return
        if renderer(*frames, partial / "charts" / filename):
            chart_specs.append(ChartSpec(filename, title_text, source_tables, note_cn))

    condition_summary = materialized.get("condition_control_summaries")
    if condition_summary is not None and _save_condition_effect_chart(
        condition_summary, partial / "charts" / "condition_effects_zoomed.png"
    ):
        chart_specs.append(
            ChartSpec(
                "condition_effects_zoomed.png",
                "Condition-level paired effects",
                ("condition_control_summaries",),
                "TN 与 FN 使用独立缩放纵轴，精确值见来源表。",
            )
        )
    add_chart(
        "a02_seed_forest_zoomed.png",
        "A02 discovery/confirmation seed forest",
        ("triad_control_deltas",),
        "逐 seed、逐对照展示 A02 配对效应，横轴为缩放轴。",
        _save_a02_seed_forest,
    )
    add_chart(
        "condition_pareto_zoomed.png",
        "Condition safety/utility plane",
        ("condition_control_summaries",),
        "左上方向更优；横纵轴均按观察范围缩放。",
        _save_condition_pareto,
    )
    add_chart(
        "r2_effective_unique_contrast.png",
        "R2 effective unique contrast",
        ("r2_overlap_power_audit",),
        "R2 与 treatment 高重合时，独特对比比例决定机制检验功效。",
        _save_r2_contrast,
    )
    budget = materialized.get("budget_response")
    if budget is not None and _save_two_metric_response(
        budget,
        partial / "charts" / "budget_response_zoomed.png",
        x_column="budget" if "budget" in budget else "comparator_condition",
        title="Replay budget response",
    ):
        chart_specs.append(
            ChartSpec(
                "budget_response_zoomed.png",
                "Replay budget response",
                ("budget_response",),
                "600/3000/6000 的响应曲线使用缩放纵轴。",
            )
        )
    guard = materialized.get("guard_policy_contrasts")
    if guard is not None and _save_two_metric_response(
        guard,
        partial / "charts" / "guard_response_zoomed.png",
        x_column="guard_ratio" if "guard_ratio" in guard else "comparator_condition",
        title="Defect guard response",
    ):
        chart_specs.append(
            ChartSpec(
                "guard_response_zoomed.png",
                "Defect guard response",
                ("guard_policy_contrasts",),
                "guard 比例响应使用缩放纵轴，机器混杂需结合审计解释。",
            )
        )
    add_chart(
        "tail_shift_zoomed.png",
        "Fixed-tail raw-score movement",
        ("prediction_tail_summary",),
        "normal 与 defect 固定尾部的原始分数移动，纵轴缩放。",
        _save_tail_shift,
    )
    add_chart(
        "a02_threshold_frontier_zoomed.png",
        "A02 threshold frontier",
        ("a02_threshold_frontier",),
        "展示阈值扫描的 FN/TN 前沿，双轴按观察范围缩放。",
        _save_threshold_frontier,
    )
    add_chart(
        "a02_training_curves_zoomed.png",
        "A02 training curves",
        ("a02_training_curves",),
        "top1 与 val loss 使用独立缩放纵轴，不能替代逐 epoch operational 指标。",
        _save_training_curves,
    )
    add_chart(
        "resume_machine_reliability.png",
        "Resume and machine reliability",
        ("canonical_run_metrics",),
        "机器运行量和 native resume 次数均使用从零开始的计数轴。",
        _save_reliability,
    )

    table_names = list(materialized)
    atomic_write_yaml(partial / "analysis_contract.yaml", contract)
    audit_names = _write_audits(partial / "audit", audits or {})
    _write_markdown(
        partial / "FINAL_REPORT_CN.md",
        title=title,
        metadata=metadata,
        hypotheses=hypotheses,
        table_names=table_names,
        chart_specs=chart_specs,
        narrative=narrative_sections,
        audit_names=audit_names,
    )
    _write_readme(
        partial / "README.md",
        table_names=table_names,
        chart_specs=chart_specs,
        audit_names=audit_names,
    )
    _write_html(
        partial / "index.html",
        title=title,
        metadata=metadata,
        hypotheses=hypotheses,
        tables=materialized,
        chart_specs=chart_specs,
        narrative=narrative_sections,
        audit_names=audit_names,
    )
    _assert_no_replacement_characters(partial)
    manifest = _build_manifest(
        partial,
        metadata=metadata,
        table_count=len(materialized),
        chart_count=len(chart_specs),
        audit_count=len(audit_names),
    )
    atomic_write_json(partial / "manifest.json", manifest)

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    os.rename(partial, output)
    return output
