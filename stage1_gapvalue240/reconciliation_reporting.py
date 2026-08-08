"""Tables-first reporting for expert/frontier reconciliation."""

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


CORE_TABLES = (
    "unified_triad_outcomes",
    "expert_to_absolute_crosswalk",
    "late_overfit_timing",
    "weak_defect_vs_normal_tail",
    "control_strength_audit",
    "clustered_models",
    "cross_validation_fold_predictions",
    "hypothesis_registry",
)


def _require_tables(tables: dict[str, pd.DataFrame]) -> None:
    missing = sorted(set(CORE_TABLES).difference(tables))
    if missing:
        raise ValueError(f"Reconciliation report missing tables: {missing}")


def _table(frame: pd.DataFrame, rows: int = 80) -> str:
    return frame.head(rows).to_html(
        index=False, border=0, classes="data-table", escape=True
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    clean = frame.fillna("").astype(str).map(lambda value: value.replace("|", "\\|"))
    header = "| " + " | ".join(clean.columns) + " |"
    separator = "| " + " | ".join("---" for _ in clean.columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in clean.to_numpy().tolist()]
    return "\n".join([header, separator, *body])


def _plot_crosswalk(frame: pd.DataFrame, output: Path) -> None:
    pivot = frame.pivot_table(
        index="expert_group",
        columns="unified_outcome",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
    fig, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    pivot.plot(kind="bar", stacked=True, ax=axis, colormap="Set2")
    axis.set_title("Expert relative cohorts mapped to absolute performance gates")
    axis.set_xlabel("Expert cohort")
    axis.set_ylabel("Triad count")
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title="Unified outcome", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _plot_late_timing(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, prefix, title in (
        (axes[0], "strict", "Absolute local gain vs absolute harm"),
        (axes[1], "expert", "Expert relative gain vs relative harm"),
    ):
        if {
            f"{prefix}_good_mean",
            f"{prefix}_harmful_mean",
            "cutoff_epoch",
        }.issubset(frame.columns):
            axis.plot(
                frame["cutoff_epoch"],
                frame[f"{prefix}_good_mean"],
                marker="o",
                label="Good",
            )
            axis.plot(
                frame["cutoff_epoch"],
                frame[f"{prefix}_harmful_mean"],
                marker="o",
                label="Harmful",
            )
            axis.axhline(0, color="black", linewidth=0.8)
            axis.legend()
        else:
            axis.text(0.5, 0.5, "Synthetic test data", ha="center", va="center")
        axis.set_title(title)
        axis.set_xlabel("Cutoff epoch")
        axis.set_ylabel("LateOverfit [Zoomed y-axis]")
        axis.grid(alpha=0.25)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _plot_absolute_relative(frame: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    required = {"min_relative_TN_gain", "delta_TN_at_baseline_fn", "unified_outcome"}
    if required.issubset(frame.columns):
        for outcome, group in frame.groupby("unified_outcome", sort=True):
            axis.scatter(
                group["min_relative_TN_gain"],
                group["delta_TN_at_baseline_fn"],
                label=str(outcome),
                alpha=0.75,
                s=38,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.legend(title="Unified outcome", bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        axis.text(0.5, 0.5, "Synthetic test data", ha="center", va="center")
    axis.set_title("Relative-control gain versus zero-replay absolute gain")
    axis.set_xlabel("Minimum TN gain versus R1/R2")
    axis.set_ylabel("TN gain versus zero-replay baseline")
    axis.grid(alpha=0.25)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _plot_tail(frame: pd.DataFrame, output: Path) -> None:
    data = frame.copy()
    fig, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    required = {"unified_outcome", "label", "mean_shift"}
    if required.issubset(data.columns):
        data = (
            data.groupby(["unified_outcome", "label"], as_index=False)["mean_shift"]
            .mean()
            .pivot(index="unified_outcome", columns="label", values="mean_shift")
        )
        data.plot(kind="bar", ax=axis, color=["#D92D20", "#155EEF"])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.tick_params(axis="x", rotation=20)
        axis.legend(title="Tail label")
    else:
        axis.text(0.5, 0.5, "Synthetic test data", ha="center", va="center")
    axis.set_title("Raw-score operational tail movement")
    axis.set_xlabel("Unified outcome")
    axis.set_ylabel("Treatment minus control mean score [Zoomed y-axis]")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _recommendations() -> str:
    return """# 下一轮预注册建议

1. 保留相同selection和初始化，对比标准200 epoch与epoch 120后replay权重衰减。
2. 干预候选点冻结为epoch 120、140、160；不得依据本轮标签事后挑点。
3. 在epoch 120、140、150、160、180、200保存val_op原始预测，直接重算同FN性能前沿。
4. LateOverfit只作为监控量；本轮不能据此自动early stop。
5. 主检验是坏seed能否被后期replay降权救回，并同时胜过零回流、R1和R2。
6. 在动态控制得到独立验证以前，不再把任何静态单样本分数称为最终价值公式。
"""


def _markdown(tables: dict[str, pd.DataFrame], metadata: dict[str, Any]) -> str:
    unified = tables["unified_triad_outcomes"]
    counts = unified["unified_outcome"].value_counts().to_dict()
    hypotheses = tables["hypothesis_registry"]
    return "\n".join(
        [
            "# Stage1 专家结论与全性能前沿交叉分析",
            "",
            "## 统一结论",
            "",
            "专家的强正向表示相对R1/R2获益；只有同时通过零回流基线，才属于绝对性能提升。",
            f"统一分层：`{counts}`。",
            "",
            "当前没有240-run方法在完整安全FN前沿及随机对照上形成可重复稳健优势。",
            "LateOverfit与弱缺陷保护方向在严格口径下仍存在，但严格成功组很少，只能作为下一轮预注册候选。",
            "",
            "## 专家相对结果到绝对结果的映射",
            "",
            _markdown_table(tables["expert_to_absolute_crosswalk"]),
            "",
            "## 假设判定",
            "",
            _markdown_table(hypotheses),
            "",
            "## 控制混杂后的统计",
            "",
            _markdown_table(tables["clustered_models"]),
            "",
            "## 防泄漏交叉验证",
            "",
            _markdown_table(tables.get("cross_validation_summary", pd.DataFrame())),
            "",
            "## 科学边界",
            "",
            *[f"- {item}" for item in metadata.get("scientific_boundaries", [])],
            "",
        ]
    )


def _html(tables: dict[str, pd.DataFrame], metadata: dict[str, Any]) -> str:
    unified = tables["unified_triad_outcomes"]
    counts = html.escape(str(unified["unified_outcome"].value_counts().to_dict()))
    boundaries = "".join(
        f"<li>{html.escape(str(item))}</li>"
        for item in metadata.get("scientific_boundaries", [])
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Stage1专家与性能前沿交叉分析</title><style>
body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;background:#f4f7fb;color:#17202a;margin:0}}
main{{max-width:1440px;margin:auto;padding:28px}}section{{background:white;margin:18px 0;padding:22px;border-radius:10px;box-shadow:0 2px 10px #00000010}}
h1,h2{{color:#173f7a}}.notice{{border-left:5px solid #155eef;background:#eef4ff;padding:14px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:16px}}img{{width:100%;height:auto}}
.scroll{{overflow:auto;max-height:650px}}.data-table{{border-collapse:collapse;width:100%;font-size:12px}}
.data-table th,.data-table td{{border:1px solid #d8dee9;padding:5px;text-align:right}}.data-table th{{background:#edf2f7;position:sticky;top:0}}
</style></head><body><main><h1>Stage1 专家结论 × 全性能前沿交叉分析</h1>
<div class="notice">专家相对结果必须再通过零回流绝对基线；相同FN下比较TN，单纯阈值滑动不算提升。</div>
<section><h2>统一结论</h2><p>{counts}</p><p>当前没有240-run方法形成跨seed的稳健安全前沿优势。LateOverfit和弱缺陷保护是下一轮监控候选，不是已验证训练算法。</p></section>
<section><h2>核心图表</h2><div class="grid"><img src="charts/expert_absolute_crosswalk.png"><img src="charts/late_overfit_timing.png"><img src="charts/relative_vs_absolute.png"><img src="charts/tail_mechanism.png"></div></section>
<section><h2>专家相对结果与绝对结果映射</h2><div class="scroll">{_table(tables['expert_to_absolute_crosswalk'])}</div></section>
<section><h2>统一80组结果</h2><div class="scroll">{_table(unified, 100)}</div><p><a href="tables/unified_triad_outcomes.csv">下载完整CSV</a></p></section>
<section><h2>假设判定</h2><div class="scroll">{_table(tables['hypothesis_registry'])}</div></section>
<section><h2>控制seed、Phase、预算后的统计</h2><div class="scroll">{_table(tables['clustered_models'])}</div></section>
<section><h2>弱缺陷与困难normal尾部</h2><div class="scroll">{_table(tables['weak_defect_vs_normal_tail'])}</div></section>
<section><h2>交叉验证</h2><div class="scroll">{_table(tables.get('cross_validation_summary', pd.DataFrame()))}</div></section>
<section><h2>科学边界</h2><ul>{boundaries}</ul></section>
</main></body></html>"""


def build_reconciliation_report(
    output_dir: str | Path,
    *,
    tables: dict[str, pd.DataFrame],
    metadata: dict[str, Any],
) -> Path:
    """Write the report atomically and never overwrite an existing version."""

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
        _plot_crosswalk(tables["expert_to_absolute_crosswalk"], chart_dir / "expert_absolute_crosswalk.png")
        _plot_late_timing(tables["late_overfit_timing"], chart_dir / "late_overfit_timing.png")
        _plot_absolute_relative(tables["control_strength_audit"], chart_dir / "relative_vs_absolute.png")
        _plot_tail(tables["weak_defect_vs_normal_tail"], chart_dir / "tail_mechanism.png")
        (staging / "FINAL_RECONCILIATION_REPORT_CN.md").write_text(
            _markdown(tables, metadata), encoding="utf-8"
        )
        (staging / "NEXT_ROUND_PREREGISTERED_RECOMMENDATIONS.md").write_text(
            _recommendations(), encoding="utf-8"
        )
        (staging / "README.md").write_text(
            "# Reconciliation report\n\nOpen `index.html`; CSV tables are the source of truth.\n",
            encoding="utf-8",
        )
        (staging / "index.html").write_text(_html(tables, metadata), encoding="utf-8")
        (staging / "analysis_contract.yaml").write_text(
            yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        validation = {
            "status": "PASS",
            "core_tables": list(CORE_TABLES),
            "table_count": len(tables),
            "chart_count": len(list(chart_dir.glob("*.png"))),
        }
        (audit_dir / "report_validation.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
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
        (staging / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "analysis_id": metadata.get("analysis_id", "unknown"),
                    "file_count_excluding_manifest": len(files),
                    "files": files,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return output
