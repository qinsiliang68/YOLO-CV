"""Focused, non-overwriting report builder for GapValue extreme cohorts.

The report consumes already-derived tables.  It does not discover runs, choose
cohorts, or mutate frozen experiment evidence.  Every output is first written
under a sibling ``.inprogress`` directory and published with one directory
rename only after the package and its file manifest are complete.
"""

from __future__ import annotations

import html
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .util import atomic_write_json, atomic_write_text, atomic_write_yaml, sha256_file


REPORT_SCHEMA_VERSION = "stage1_gapvalue240_extreme_report_v1"
_SAFE_TABLE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_TIER_ORDER = ("S", "A", "B", "M", "H")
_TIER_COLORS = {
    "S": "#1a7f37",
    "A": "#57ab5a",
    "B": "#bf8700",
    "M": "#6e7781",
    "H": "#cf222e",
}
_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "triad_performance_tiers": (
        "triad_id",
        "cohort_code",
        "cohort_label",
        "delta_TN_R1",
        "delta_FN_R1",
        "delta_TN_R2",
        "delta_FN_R2",
    ),
    "training_window_features": (
        "triad_id",
        "control",
        "cohort_code",
        "train_loss_extra_drop_epoch121_to_200",
        "train_loss_robust_drop_121_130_to_191_200",
        "train_loss_slope_121_200",
    ),
    "selection_set_outcomes": (
        "sample_set_digest",
        "cohort_codes",
        "triad_count",
        "exceptional_count",
        "harmful_count",
        "spans_exceptional_and_harmful",
    ),
    "prediction_tail_extreme_contrasts": (
        "label",
        "tail_scope",
        "score_type",
        "analysis_scope",
        "control",
        "feature",
        "exceptional_mean",
        "harmful_mean",
    ),
}
_CHARTS = (
    (
        "triad_performance_quadrants_zoomed.png",
        "Triad operational effect quadrants",
        "triad_performance_tiers",
        "R1/R2 are separate panels. Both axes show the observed range plus zero.",
    ),
    (
        "training_late_loss_s_h_zoomed.png",
        "Late training-loss contrast: S versus H",
        "training_window_features",
        "Endpoint, robust-window, and OLS-slope views use separate zoomed y-axes.",
    ),
    (
        "selection_set_tier_flips.png",
        "Identical selection sets spanning S and H",
        "selection_set_outcomes",
        "Counts start at zero and describe repeated triad outcomes, not independent selections.",
    ),
    (
        "prediction_tail_mechanism_zoomed.png",
        "Operational-tail score shifts: S versus H",
        "prediction_tail_extreme_contrasts",
        "Raw T-control shifts on fixed control-defined tails; lower is better for normal, higher is better for defect.",
    ),
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(current) for key, current in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(current) for current in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is pd.NA:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _table_name(value: str) -> str:
    if not value or not _SAFE_TABLE_NAME.fullmatch(value):
        raise ValueError(f"Unsafe table name: {value!r}")
    stem = Path(value).stem
    if stem in {"", ".", ".."}:
        raise ValueError(f"Unsafe table name: {value!r}")
    return stem


def _require_finite(frame: pd.DataFrame, columns: Sequence[str], *, table: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{table} column {column} must contain only finite numbers")


def _validate_inputs(
    tables: Mapping[str, pd.DataFrame], findings: Sequence[Mapping[str, Any]]
) -> dict[str, pd.DataFrame]:
    if not isinstance(tables, Mapping):
        raise TypeError("tables must be a mapping of names to pandas DataFrames")
    materialized: dict[str, pd.DataFrame] = {}
    for name, frame in tables.items():
        stem = _table_name(str(name))
        if stem in materialized:
            raise ValueError(f"Duplicate table name after normalization: {stem}")
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"Table {stem} must be a pandas DataFrame")
        materialized[stem] = frame.copy()

    for name, columns in _REQUIRED_COLUMNS.items():
        if name not in materialized:
            raise ValueError(f"Missing required table: {name}")
        frame = materialized[name]
        if frame.empty:
            raise ValueError(f"Required table {name} must not be empty")
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise ValueError(f"Required table {name} missing columns: {missing}")

    tiers = materialized["triad_performance_tiers"]
    if tiers["triad_id"].astype(str).duplicated().any():
        raise ValueError("triad_performance_tiers triad_id values must be unique")
    invalid_tiers = sorted(set(tiers["cohort_code"].astype(str)).difference(_TIER_ORDER))
    if invalid_tiers:
        raise ValueError(f"triad_performance_tiers contains invalid cohorts: {invalid_tiers}")
    _require_finite(
        tiers,
        ("delta_TN_R1", "delta_FN_R1", "delta_TN_R2", "delta_FN_R2"),
        table="triad_performance_tiers",
    )

    training = materialized["training_window_features"]
    if not set(training["control"].astype(str)).issubset({"R1", "R2"}):
        raise ValueError("training_window_features control must be R1 or R2")
    if not {"S", "H"}.issubset(set(training["cohort_code"].astype(str))):
        raise ValueError("training_window_features must contain both S and H cohorts")
    _require_finite(
        training,
        (
            "train_loss_extra_drop_epoch121_to_200",
            "train_loss_robust_drop_121_130_to_191_200",
            "train_loss_slope_121_200",
        ),
        table="training_window_features",
    )

    selection_sets = materialized["selection_set_outcomes"]
    if selection_sets["sample_set_digest"].astype(str).duplicated().any():
        raise ValueError("selection_set_outcomes sample_set_digest values must be unique")
    _require_finite(
        selection_sets,
        ("triad_count", "exceptional_count", "harmful_count"),
        table="selection_set_outcomes",
    )
    tail = materialized["prediction_tail_extreme_contrasts"]
    _require_finite(
        tail,
        ("exceptional_mean", "harmful_mean"),
        table="prediction_tail_extreme_contrasts",
    )
    if not all(isinstance(finding, Mapping) for finding in findings):
        raise TypeError("Every finding must be a mapping")
    return materialized


def _expanded_limits(values: np.ndarray, *, include_zero: bool = True) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if include_zero:
        finite = np.append(finite, 0.0)
    if not len(finite):
        return -1.0, 1.0
    low, high = float(finite.min()), float(finite.max())
    span = high - low
    padding = span * 0.10 if span else max(abs(low) * 0.10, 1e-6)
    return low - padding, high + padding


def _save_triad_performance_tiers(frame: pd.DataFrame, output: Path) -> None:
    x_values = np.concatenate(
        [frame["delta_TN_R1"].to_numpy(float), frame["delta_TN_R2"].to_numpy(float)]
    )
    y_values = np.concatenate(
        [frame["delta_FN_R1"].to_numpy(float), frame["delta_FN_R2"].to_numpy(float)]
    )
    x_limits = _expanded_limits(x_values)
    y_limits = _expanded_limits(y_values)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), sharex=True, sharey=True)
    for ax, control in zip(axes, ("R1", "R2")):
        ax.axvline(0, color="#57606a", linestyle="--", linewidth=1)
        ax.axhline(0, color="#57606a", linestyle="--", linewidth=1)
        for cohort in _TIER_ORDER:
            rows = frame[frame["cohort_code"].astype(str) == cohort]
            if rows.empty:
                continue
            ax.scatter(
                rows[f"delta_TN_{control}"],
                rows[f"delta_FN_{control}"],
                color=_TIER_COLORS[cohort],
                label=cohort,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.45,
                s=46,
            )
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_title(f"{control} [Zoomed axes; zero shown]")
        ax.set_xlabel("Delta TN (higher is better) [Zoomed]")
        ax.grid(alpha=0.20)
        ax.text(
            0.98,
            0.03,
            "Favourable: Delta TN > 0, Delta FN <= 0",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#57606a",
        )
    axes[0].set_ylabel("Delta FN (lower is better) [Zoomed]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Triad operational effects by extreme-performance cohort", y=0.985)
    fig.legend(
        handles,
        labels,
        title="Cohort",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_training_window_features(frame: pd.DataFrame, output: Path) -> None:
    plot = frame[frame["cohort_code"].astype(str).isin(["S", "H"])].copy()
    metrics = (
        (
            "train_loss_extra_drop_epoch121_to_200",
            "Endpoint: epoch 121 minus 200",
        ),
        (
            "train_loss_robust_drop_121_130_to_191_200",
            "Robust: mean 121-130 minus 191-200",
        ),
        ("train_loss_slope_121_200", "OLS slope: epochs 121-200"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.5))
    control_offsets = {"R1": -0.09, "R2": 0.09}
    control_markers = {"R1": "o", "R2": "^"}
    for ax, (column, title) in zip(axes, metrics):
        values = plot[column].to_numpy(dtype=float)
        for cohort_index, cohort in enumerate(("S", "H")):
            cohort_rows = plot[plot["cohort_code"].astype(str) == cohort]
            for control in ("R1", "R2"):
                rows = cohort_rows[cohort_rows["control"].astype(str) == control]
                if rows.empty:
                    continue
                x = np.full(len(rows), cohort_index + control_offsets[control])
                y = rows[column].to_numpy(dtype=float)
                ax.scatter(
                    x,
                    y,
                    color=_TIER_COLORS[cohort],
                    marker=control_markers[control],
                    s=48,
                    alpha=0.82,
                    edgecolor="white",
                    linewidth=0.4,
                    label=f"{cohort}/{control}",
                )
                median = float(np.median(y))
                ax.plot(
                    [cohort_index + control_offsets[control] - 0.06, cohort_index + control_offsets[control] + 0.06],
                    [median, median],
                    color="#24292f",
                    linewidth=2,
                )
        ax.axhline(0, color="#57606a", linestyle="--", linewidth=1)
        ax.set_ylim(*_expanded_limits(values))
        ax.set_xticks([0, 1], ["S", "H"])
        ax.set_xlabel("Performance cohort")
        ax.set_ylabel(f"{column} [Zoomed]")
        ax.set_title(f"{title}\n[Zoomed y-axis; zero shown]")
        ax.grid(axis="y", alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "Late training-loss behaviour: exceptional versus harmful triads", y=0.985
    )
    fig.legend(
        handles,
        labels,
        title="Cohort/control",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _save_selection_set_outcomes(frame: pd.DataFrame, output: Path) -> None:
    flags = frame["spans_exceptional_and_harmful"].map(_truthy)
    plot = frame[flags].copy()
    if plot.empty:
        plot = frame.copy()
    plot = plot.sort_values(
        ["exceptional_count", "harmful_count", "sample_set_digest"],
        ascending=[False, False, True],
    )
    labels = plot["sample_set_digest"].astype(str).map(
        lambda value: value if len(value) <= 28 else value[:25] + "..."
    )
    y = np.arange(len(plot))
    height = max(4.8, 0.48 * len(plot) + 2.0)
    fig, ax = plt.subplots(figsize=(11.5, height), constrained_layout=True)
    bar_height = 0.36
    exceptional = plot["exceptional_count"].to_numpy(dtype=float)
    harmful = plot["harmful_count"].to_numpy(dtype=float)
    ax.barh(
        y - bar_height / 2,
        exceptional,
        height=bar_height,
        color=_TIER_COLORS["S"],
        label="S triads",
    )
    ax.barh(
        y + bar_height / 2,
        harmful,
        height=bar_height,
        color=_TIER_COLORS["H"],
        label="H triads",
    )
    for index, (s_count, h_count, codes) in enumerate(
        zip(exceptional, harmful, plot["cohort_codes"].astype(str))
    ):
        ax.text(s_count, index - bar_height / 2, f" {int(s_count)}", va="center", fontsize=8)
        ax.text(h_count, index + bar_height / 2, f" {int(h_count)}", va="center", fontsize=8)
        ax.text(
            1.0,
            index,
            f" cohorts={codes}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7.5,
            color="#57606a",
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("Triad count [Zero-based count axis]")
    ax.set_ylabel("Frozen treatment sample-set digest")
    ax.set_title("Identical treatment selections spanning exceptional and harmful outcomes")
    ax.grid(axis="x", alpha=0.22)
    ax.legend()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_prediction_tail_mechanism(frame: pd.DataFrame, output: Path) -> None:
    plot = frame.loc[
        (frame["tail_scope"] == "operational")
        & (frame["score_type"] == "raw")
        & (frame["analysis_scope"] == "all")
        & (frame["feature"] == "mean_shift")
    ].copy()
    expected = {
        (label, control)
        for label in ("normal", "defect")
        for control in ("R1", "R2")
    }
    actual = set(zip(plot["label"].astype(str), plot["control"].astype(str)))
    if actual != expected or plot.duplicated(["label", "control"]).any():
        raise ValueError(
            "prediction_tail_extreme_contrasts must contain one raw/all/"
            "operational mean_shift row per label and control"
        )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharex=False)
    width = 0.34
    positions = np.arange(2)
    for ax, label in zip(axes, ("normal", "defect"), strict=True):
        subset = plot.loc[plot["label"] == label].set_index("control")
        exceptional = subset.loc[["R1", "R2"], "exceptional_mean"].to_numpy(
            dtype=float
        )
        harmful = subset.loc[["R1", "R2"], "harmful_mean"].to_numpy(dtype=float)
        ax.bar(
            positions - width / 2,
            exceptional,
            width,
            color=_TIER_COLORS["S"],
            label="S triads",
        )
        ax.bar(
            positions + width / 2,
            harmful,
            width,
            color=_TIER_COLORS["H"],
            label="H triads",
        )
        for x, value in zip(positions - width / 2, exceptional, strict=True):
            ax.text(x, value, f"{value:+.6f}", ha="center", va="bottom", fontsize=8)
        for x, value in zip(positions + width / 2, harmful, strict=True):
            ax.text(x, value, f"{value:+.6f}", ha="center", va="bottom", fontsize=8)
        ax.axhline(0, color="#57606a", linestyle="--", linewidth=1)
        ax.set_xticks(positions, ["R1", "R2"])
        ax.set_ylim(*_expanded_limits(np.concatenate([exceptional, harmful])))
        ax.set_xlabel("Control arm")
        ax.set_ylabel("Mean raw-score shift: T - control [Zoomed]")
        direction = "lower is favourable" if label == "normal" else "higher is favourable"
        ax.set_title(
            f"{label.capitalize()} operational tail\n[{direction}; zero shown]"
        )
        ax.grid(axis="y", alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "Fixed operational-tail mechanism: exceptional versus harmful triads",
        y=0.99,
    )
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _finding_text(finding: Mapping[str, Any], key: str, default: str = "") -> str:
    return str(finding.get(key, default))


def _write_markdown(
    path: Path,
    *,
    metadata: Mapping[str, Any],
    findings: Sequence[Mapping[str, Any]],
    table_names: Sequence[str],
) -> None:
    lines = [
        "# Stage1 GapValue 极端条件集对照分析",
        "",
        "本报告集中比较表现特别好的 S 级 triad 与明确有害的 H 级 triad。",
        "R1 与 R2 始终分开；图中缩放坐标均明确标注，精确数值以 CSV 为准。",
        "",
        "## 分析边界",
        "",
        str(metadata.get("conclusion_boundary", "仅限内部 val_op 描述性分析。")),
        "",
        "## 重点发现",
        "",
    ]
    if findings:
        for finding in findings:
            status = _finding_text(finding, "status", "UNSPECIFIED")
            title = _finding_text(finding, "title", _finding_text(finding, "finding", "未命名发现"))
            lines.append(f"### [{status}] {title}")
            lines.append("")
            if finding.get("evidence"):
                lines.append(f"- 证据：{_finding_text(finding, 'evidence')}")
            if finding.get("boundary"):
                lines.append(f"- 边界：{_finding_text(finding, 'boundary')}")
            lines.append("")
    else:
        lines.extend(["暂无登记发现。", ""])
    lines.extend(
        [
            "## 核心图表",
            "",
            "- ![Triad effects](charts/triad_performance_quadrants_zoomed.png)",
            "- ![Late loss](charts/training_late_loss_s_h_zoomed.png)",
            "- ![Selection flips](charts/selection_set_tier_flips.png)",
            "",
            "## 原始结果表",
            "",
        ]
    )
    lines.extend(f"- [{name}](tables/{name}.csv)" for name in table_names)
    lines.append("")
    atomic_write_text(path, "\n".join(lines))


def _write_readme(path: Path, *, table_names: Sequence[str]) -> None:
    lines = [
        "# Extreme-cohort report package",
        "",
        "这是由冻结分析表重新生成的只读报告包；生命周期为 regenerate。",
        "精确数值位于 `tables/`，图表位于 `charts/`，输入结构审计位于 `audit/`。",
        "`manifest.json`覆盖除自身外的全部永久文件。",
        "",
        "## Tables",
        "",
    ]
    lines.extend(f"- `tables/{name}.csv`" for name in table_names)
    lines.append("")
    atomic_write_text(path, "\n".join(lines))


def _write_html(
    path: Path,
    *,
    metadata: Mapping[str, Any],
    findings: Sequence[Mapping[str, Any]],
    table_names: Sequence[str],
) -> None:
    metadata_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td><code>{html.escape(str(value))}</code></td></tr>"
        for key, value in metadata.items()
    )
    finding_cards = []
    for finding in findings:
        status = html.escape(_finding_text(finding, "status", "UNSPECIFIED"))
        title = html.escape(
            _finding_text(finding, "title", _finding_text(finding, "finding", "未命名发现"))
        )
        evidence = html.escape(_finding_text(finding, "evidence"))
        boundary = html.escape(_finding_text(finding, "boundary"))
        finding_cards.append(
            f'<article class="finding"><h3>[{status}] {title}</h3>'
            f"<p>{evidence}</p><p class=\"boundary\">边界：{boundary}</p></article>"
        )
    charts = "".join(
        f'<article class="chart"><h3>{html.escape(title)}</h3>'
        f'<img src="charts/{filename}" alt="{html.escape(title)}">'
        f'<p>{html.escape(note)}</p><p>Source: '
        f'<a href="tables/{source}.csv">{html.escape(source)}.csv</a></p></article>'
        for filename, title, source, note in _CHARTS
    )
    table_links = "".join(
        f'<li><a href="tables/{html.escape(name)}.csv">{html.escape(name)}.csv</a></li>'
        for name in table_names
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage1 GapValue 极端条件集对照分析</title>
<style>
body{{font-family:Arial,"Microsoft YaHei",sans-serif;max-width:1280px;margin:auto;padding:24px;color:#24292f;line-height:1.55}}
h1,h2{{border-bottom:1px solid #d0d7de;padding-bottom:.35rem}} table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #d0d7de;padding:7px;text-align:left;vertical-align:top}} th{{background:#f6f8fa;width:30%}}
.finding,.chart{{border:1px solid #d8dee4;border-radius:8px;padding:14px;margin:14px 0;background:#fff}}
.boundary{{color:#57606a}} img{{display:block;max-width:100%;height:auto;margin:auto}} code{{white-space:pre-wrap}}
</style></head><body>
<h1>Stage1 GapValue 极端条件集对照分析</h1>
<p>重点比较 S 级与 H 级完整 triad 集合。R1/R2分面展示；所有 Zoomed axes 均包含零参考线，原始数值见CSV。</p>
<h2>分析身份与边界</h2><table>{metadata_rows}</table>
<h2>重点发现</h2>{''.join(finding_cards) if finding_cards else '<p>暂无登记发现。</p>'}
<h2>核心图表</h2>{charts}
<h2>原始结果表</h2><ul>{table_links}</ul>
<h2>审计</h2><ul><li><a href="analysis_contract.yaml">analysis_contract.yaml</a></li>
<li><a href="audit/report_inputs.json">report_inputs.json</a></li><li><a href="manifest.json">manifest.json</a></li></ul>
</body></html>"""
    atomic_write_text(path, document)


def _build_manifest(
    root: Path,
    *,
    metadata: Mapping[str, Any],
    table_count: int,
    finding_count: int,
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
        "metadata": _plain(metadata),
        "counts": {
            "tables": table_count,
            "charts": len(_CHARTS),
            "audits": 1,
            "findings": finding_count,
            "files_excluding_manifest": len(files),
        },
        "files": files,
        "manifest_self_excluded": True,
    }


def build_extreme_report(
    output_dir: str | Path,
    *,
    tables: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
    findings: Sequence[Mapping[str, Any]],
) -> Path:
    """Build and atomically publish the focused S/H extreme-cohort report."""

    materialized = _validate_inputs(tables, findings)
    plain_metadata = _plain(dict(metadata))
    plain_findings = [_plain(dict(finding)) for finding in findings]
    output = Path(output_dir)
    partial = output.with_name(output.name + ".inprogress")
    for candidate in (output, partial):
        if candidate.exists():
            raise FileExistsError(f"Refusing to overwrite: {candidate}")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    tables_dir = partial / "tables"
    charts_dir = partial / "charts"
    audit_dir = partial / "audit"
    tables_dir.mkdir()
    charts_dir.mkdir()
    audit_dir.mkdir()

    for name, frame in materialized.items():
        frame.to_csv(tables_dir / f"{name}.csv", index=False, lineterminator="\n")

    _save_triad_performance_tiers(
        materialized["triad_performance_tiers"],
        charts_dir / "triad_performance_quadrants_zoomed.png",
    )
    _save_training_window_features(
        materialized["training_window_features"],
        charts_dir / "training_late_loss_s_h_zoomed.png",
    )
    _save_selection_set_outcomes(
        materialized["selection_set_outcomes"],
        charts_dir / "selection_set_tier_flips.png",
    )
    _save_prediction_tail_mechanism(
        materialized["prediction_tail_extreme_contrasts"],
        charts_dir / "prediction_tail_mechanism_zoomed.png",
    )

    table_names = list(materialized)
    contract = {
        "report_id": REPORT_SCHEMA_VERSION,
        "analysis_generation": "v3 extreme-cohort focused report",
        "required_tables": list(_REQUIRED_COLUMNS),
        "required_columns": {key: list(value) for key, value in _REQUIRED_COLUMNS.items()},
        "controls": "R1 and R2 are displayed separately and never pooled",
        "axis_policy": "zoomed observed ranges explicitly labelled; zero reference shown",
        "selection_unit": "sample_set_digest, not repeated triad rows",
        "output_policy": "sibling .inprogress directory followed by atomic rename; no overwrite",
        "conclusion_boundary": plain_metadata.get(
            "conclusion_boundary", "val_op internal descriptive analysis"
        ),
    }
    atomic_write_yaml(partial / "analysis_contract.yaml", contract)
    atomic_write_json(
        audit_dir / "report_inputs.json",
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "table_rows": {name: len(frame) for name, frame in materialized.items()},
            "table_columns": {name: list(frame.columns) for name, frame in materialized.items()},
            "finding_count": len(plain_findings),
            "required_input_validation": "PASS",
        },
    )
    _write_markdown(
        partial / "FINAL_REPORT_CN.md",
        metadata=plain_metadata,
        findings=plain_findings,
        table_names=table_names,
    )
    _write_readme(partial / "README.md", table_names=table_names)
    _write_html(
        partial / "index.html",
        metadata=plain_metadata,
        findings=plain_findings,
        table_names=table_names,
    )
    manifest = _build_manifest(
        partial,
        metadata=plain_metadata,
        table_count=len(materialized),
        finding_count=len(plain_findings),
    )
    atomic_write_json(partial / "manifest.json", manifest)

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    os.rename(partial, output)
    return output
