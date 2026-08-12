"""Clean Chinese reporting helpers for the final 240-run Goal analysis."""

from __future__ import annotations

import html
import os
from pathlib import Path
import re
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_executive_findings(facts: Mapping[str, object]) -> pd.DataFrame:
    """Turn audited facts into claim-bounded executive findings."""

    total = int(facts["total_triads"])
    records = [
        {
            "finding_id": "F01_FIXED_POINT_OUTCOMES",
            "evidence_level": "EXACT_AUDIT",
            "finding_cn": (
                f"{total}个triad中有{int(facts['dual_improvement_triads'])}个在两个固定工作点同时胜过R1/R2；"
                f"{int(facts['high_value_triads'])}个满足高性价比门槛，其中"
                f"{int(facts['dual_high_value_overlap'])}个重合、{int(facts['high_value_only_triads'])}个仅满足高性价比；"
                f"另有{int(facts['dual_harm_triads'])}个双端恶化和{int(facts['mixed_triads'])}个混合结果。"
                f"互斥计数为{int(facts['dual_improvement_triads'])}+{int(facts['high_value_only_triads'])}+"
                f"{int(facts['dual_harm_triads'])}+{int(facts['mixed_triads'])}={total}。"
            ),
            "claim_boundary": "固定工作点结果，不等同于完整安全前沿支配。",
        },
        {
            "finding_id": "F02_RAW_FRONTIER",
            "evidence_level": "EXACT_AUDIT",
            "finding_cn": (
                f"只有{int(facts['raw_dual_safe_triads'])}/{total}个triad在raw score的FN=0–95完整安全前沿上同时支配R1/R2，"
                f"全FN范围为{int(facts['raw_full_frontier_dual_triads'])}/{total}。"
            ),
            "claim_boundary": "这是排除单点阈值滑动后的最严格内部证据。",
        },
        {
            "finding_id": "F03_LATE_FIT",
            "evidence_level": "DESCRIPTIVE",
            "finding_cn": "总体好坏组在150–160轮后出现后期额外拟合差异，但固定同一selection后该差异消失。",
            "claim_boundary": "可作为总体风险画像，不能作为独立的未见seed预测定律。",
        },
        {
            "finding_id": "F04_SELECTION_REVERSAL",
            "evidence_level": "MECHANISM",
            "finding_cn": (
                f"{int(facts['same_selection_reversal_groups'])}个完全相同的Treatment集合覆盖"
                f"{int(facts['same_selection_reversal_triads'])}个跨seed反转triad；"
                f"{int(facts['reversal_static_features_constant'])}/{int(facts['reversal_static_features_checked'])}个静态Treatment特征在digest内恒定。"
            ),
            "claim_boundary": "静态选样特征无法单独解释反转，seed与机器仍可能混杂。",
        },
        {
            "finding_id": "F05_UNSEEN_SEED",
            "evidence_level": "HELD_OUT_TEST",
            "finding_cn": (
                f"固定A02在Phase C新seed上为{int(facts['phase_c_successes'])}/{int(facts['phase_c_total'])}；"
                f"联合规则为{int(facts['joint_rule_phase_c_successes'])}/{int(facts['joint_rule_phase_c_total'])}，"
                "尚未找到80%可靠规则。"
            ),
            "claim_boundary": "AUC仅表示排序区分度，不是成功概率；五个新seed也不足以证明80%下界。",
        },
        {
            "finding_id": "F06_R2_POWER",
            "evidence_level": "LIMITATION",
            "finding_cn": f"R2平均有效独特对比比例仅{float(facts['r2_unique_rate']):.2%}，是高重合、低功效的近Treatment对照。",
            "claim_boundary": "R1与R2必须分开报告，不能合并提升样本量。",
        },
        {
            "finding_id": "F07_RELIABILITY",
            "evidence_level": "LIMITATION",
            "finding_cn": (
                f"共有{int(facts['resumed_runs'])}个续跑run、{int(facts['cross_machine_triads'])}个跨机triad、"
                f"{int(facts['cross_snapshot_triads'])}个跨snapshot triad，均进入敏感性分析。"
            ),
            "claim_boundary": "尤其Phase C和多数Phase B存在arm与机器混杂。",
        },
        {
            "finding_id": "F08_FIELD_COVERAGE",
            "evidence_level": "EXACT_AUDIT",
            "finding_cn": (
                f"字段账本共{int(facts['field_ledger_rows'])}行，UNREVIEWED={int(facts['unreviewed'])}、"
                f"UNCLASSIFIED={int(facts['unclassified'])}、SILENTLY_DROPPED={int(facts['silently_dropped'])}。"
            ),
            "claim_boundary": "未采集字段被显式标记，不从其他字段猜测。",
        },
    ]
    return pd.DataFrame(records)


def build_next_experiment_preregistration() -> pd.DataFrame:
    """Return the minimum next experiment needed to test the discovered mechanism."""

    shared = {
        "independent_confirmation_seeds": 14,
        "frozen_checkpoints": "120,140,150,160,180,200",
        "raw_frontier_required": True,
        "blind_holdout_after_freeze": True,
        "machine_assignment": "within-seed randomized balanced block",
        "success_definition": "beats R1 and R2 with delta_TN>0 and delta_FN<=0 plus raw FN0-95 safe-frontier noninferiority",
    }
    arms = [
        (
            "T_DYNAMIC_DECAY",
            "Same frozen treatment selection; full replay to epoch140, preregistered decay during141-160, then zero replay.",
        ),
        (
            "T_CONTINUOUS",
            "Same treatment selection and total base training, with replay continuing through epoch200.",
        ),
        (
            "T_LEARNABLE_GUARD",
            "Dynamic-decay normal replay plus weak-defect guard filtered for learnability rather than persistent failure.",
        ),
        (
            "R1_GLOBAL_RANDOM",
            "Disjoint global-random replay with exactly the same exposure schedule as each treatment arm.",
        ),
        (
            "R2_MATCHED_RANDOM",
            "Hardness/dynamics-matched random replay with overlap and effective unique contrast audited.",
        ),
        (
            "NR_NO_REPLAY",
            "No replay; identical base data, initialization, optimizer steps policy and checkpoint schedule.",
        ),
    ]
    return pd.DataFrame(
        [{"arm_id": arm_id, "arm_definition": definition, **shared} for arm_id, definition in arms]
    )


def _hypothesis_markdown(hypotheses: pd.DataFrame) -> list[str]:
    required = {"hypothesis_id", "hypothesis", "status"}
    missing = required - set(hypotheses.columns)
    if missing:
        raise ValueError(f"hypotheses missing columns: {sorted(missing)}")
    lines = ["| 假设 | 状态 | 命题 |", "|---|---|---|"]
    for row in hypotheses.to_dict(orient="records"):
        lines.append(
            f"| `{row['hypothesis_id']}` | {row['status']} | {str(row['hypothesis']).replace('|', '/')} |"
        )
    return lines


def render_final_markdown(
    facts: Mapping[str, object], hypotheses: pd.DataFrame
) -> str:
    """Render the central Chinese report narrative without mojibake-prone templates."""

    findings = build_executive_findings(facts)
    lines = [
        "# Stage1 GapValue 240-Run 全面分析最终报告",
        "",
        "本报告严格只使用本次240个VALIDATED canonical run、80个T/R1/R2 triad和48,000条逐epoch记录；不混入旧40/120-run、debug、canary、失败或非canonical attempt。",
        "",
        "## 一句话结论",
        "",
        "本次实验找到了少量真实双端收益和一个可重复观察的总体后期拟合画像，但尚未找到80%可靠规则；静态样本价值在跨seed时会反转，真正稳健的方向应转向可验证的训练阶段控制与弱缺陷尾部保护。",
        "",
        "## 核心证据",
        "",
    ]
    for row in findings.to_dict(orient="records"):
        lines.extend(
            [
                f"### {row['finding_id']}",
                "",
                str(row["finding_cn"]),
                "",
                f"边界：{row['claim_boundary']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 150–160轮的正确解释",
            "",
            "学习率在120轮后平滑下降，没有150–160轮的调度器突变。分叉发生时仍有大量optimizer step和重复replay暴露，因此总体差异更符合累计拟合压力；但是在完全相同selection的反转组内，后期额外train-loss下降不再区分好坏，所以它只能作为群体风险画像，不能作为单独的因果定律或可靠停止器。",
            "",
            "## 真正的性能而非阈值滑动",
            "",
            "固定FN/TN工作点用于筛选候选，但最终判断必须回到raw score的完整安全前沿与固定弱缺陷/困难正常尾部。坏组往往把高风险normal压得更低，却同时伤害弱defect，使阈值被迫下降；真正的收益来自相对排序改善和缺陷尾部保护，而不是把所有normal概率整体下压。",
            "",
            "## 80%目标判定",
            "",
            "尚未找到80%可靠规则。全部3,790个可用预测字段经过fold内筛选、留seed、留selection digest和双重排除验证后，没有形成可确认的候选；Phase C五个新seed也没有成功正例。单侧95% Clopper–Pearson下界超过80%，至少需要14/14、21/22或28/30的独立确认结果。",
            "",
            "## 假设登记",
            "",
            *_hypothesis_markdown(hypotheses),
            "",
            "## 不能回答的问题",
            "",
            "- 没有no-replay arm，不能判断replay是否优于完全不replay。",
            "- 没有blind/external test，不能宣称外部泛化已确认。",
            "- 五个梯度字段均未采集，不能检验真实GraNd、Influence、TracIn或Grad-Align。",
            "- 没有epoch150 checkpoint和逐epoch val_op预测，不能重建该时刻的精确参数或TN/FN曲线。",
            "",
            "## 下一轮",
            "",
            "预注册同一selection的持续replay、140轮后衰减/停止replay、可学习弱缺陷guard、R1、R2和no-replay六臂；在120、140、150、160、180、200轮保存checkpoint及raw val_op预测，并用至少14个完全未见seed做独立确认。",
            "",
        ]
    )
    return "\n".join(lines)


def _html_table(frame: pd.DataFrame, *, max_rows: int = 100) -> str:
    preview = frame.head(max_rows)
    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in preview.columns)
    body = []
    for row in preview.itertuples(index=False, name=None):
        body.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
            + "</tr>"
        )
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _safe_relative(value: str) -> str:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Report link must be safe and relative: {value}")
    return candidate.as_posix()


def render_final_html(
    facts: Mapping[str, object],
    hypotheses: pd.DataFrame,
    findings: pd.DataFrame,
    *,
    table_inventory: pd.DataFrame,
    chart_inventory: pd.DataFrame,
) -> str:
    """Render a self-contained index with explicit links to every derived asset."""

    for required, frame in [
        ({"relative_path", "row_count"}, table_inventory),
        ({"relative_path", "title", "source_tables"}, chart_inventory),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Report inventory missing columns: {sorted(missing)}")
    table_links = []
    for row in table_inventory.to_dict(orient="records"):
        relative = _safe_relative(str(row["relative_path"]))
        display_name = relative
        if relative == "tables/HYPOTHESIS_REGISTRY.csv":
            display_name += " [INITIAL, SUPERSEDED]"
        elif relative == "tables/HYPOTHESIS_REGISTRY_FINAL.csv":
            display_name += " [FINAL TRUTH]"
        table_links.append(
            f'<li><a href="{html.escape(relative)}">{html.escape(display_name)}</a> '
            f'({int(row["row_count"]):,} rows)</li>'
        )
    figures = []
    for row in chart_inventory.to_dict(orient="records"):
        relative = _safe_relative(str(row["relative_path"]))
        source_links = []
        for source in str(row["source_tables"]).split(";"):
            source = source.strip()
            if source:
                safe_source = _safe_relative(source)
                source_links.append(
                    f'<a href="{html.escape(safe_source)}">{html.escape(safe_source)}</a>'
                )
        figures.append(
            f'<figure><h3>{html.escape(str(row["title"]))}</h3>'
            f'<img src="{html.escape(relative)}" alt="{html.escape(str(row["title"]))}">'
            f'<figcaption>Source: {"; ".join(source_links)}</figcaption></figure>'
        )
    cards = "".join(
        f'<article class="card"><h3>{html.escape(str(row.finding_id))}</h3>'
        f'<p>{html.escape(str(row.finding_cn))}</p>'
        f'<p class="boundary">边界：{html.escape(str(row.claim_boundary))}</p></article>'
        for row in findings.itertuples(index=False)
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage1 GapValue 240-Run 全面分析</title>
<style>
body{{margin:0;background:#f3f6fa;color:#1f2937;font-family:Arial,"Microsoft YaHei",sans-serif}}
main{{max-width:1440px;margin:auto;background:#fff;padding:32px 44px}}
h1{{font-size:32px}} h2{{margin-top:36px;border-bottom:1px solid #d0d7de;padding-bottom:8px}}
.hero{{background:#eaf3ff;border-left:6px solid #0969da;padding:18px 22px;font-size:18px}}
.warning{{background:#fff8c5;border-left:6px solid #bf8700;padding:14px 20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:16px}}
.card{{border:1px solid #d0d7de;border-radius:10px;padding:14px;background:#fff}}
.boundary,figcaption{{font-size:13px;color:#57606a}} table{{border-collapse:collapse;width:100%;font-size:12px}}
th,td{{border:1px solid #d0d7de;padding:6px;text-align:left}} th{{background:#f6f8fa}}
figure{{margin:24px 0;padding:12px;border:1px solid #d8dee4}} img{{max-width:100%;height:auto}}
a{{color:#0969da}} .scroll{{overflow:auto}} code{{background:#f6f8fa;padding:2px 4px}}
</style></head><body><main>
<h1>Stage1 GapValue 240-Run 全面分析</h1>
<p class="hero">结论：尚未找到80%可靠规则。固定工作点有少量双端收益，但只有
{int(facts['raw_dual_safe_triads'])}/{int(facts['total_triads'])} 个triad在raw FN=0–95完整安全前沿上同时支配R1和R2。</p>
<p class="warning">本报告严格排除阈值滑动式“优化”：性能判断同时查看raw score、固定FN/TN、完整安全前沿和固定弱缺陷/困难正常尾部。没有no-replay arm，也没有blind/external test。</p>
<h2>核心发现</h2><section class="grid">{cards}</section>
<h2>证据状态</h2><div class="scroll">{_html_table(hypotheses, max_rows=100)}</div>
<h2>图表</h2>{''.join(figures)}
<h2>完整派生表</h2><ul>{''.join(table_links)}</ul>
<h2>阅读边界</h2><ul>
<li>AUC只表示排序区分度，不是成功概率。</li>
<li>总体后期拟合差异在固定selection反转组内消失，不能升级为因果规律。</li>
<li>梯度、epoch150权重、逐epoch val_op预测均未采集，不作猜测。</li>
</ul>
</main></body></html>"""
    if "�" in document or "锛" in document:
        raise UnicodeError("Generated HTML contains a mojibake marker")
    return document


_LINK_PATTERN = re.compile(r"(?:href|src)=\"([^\"]+)\"")


def validate_report_links(
    index_path: str | Path, *, published_root: str | Path | None = None
) -> pd.DataFrame:
    """Validate local links while recording their stable published locations.

    Reports are assembled under an ``.inprogress`` directory and atomically
    renamed only after validation.  ``published_root`` lets the audit record
    the post-rename path while existence is still checked against the active
    directory.
    """

    index = Path(index_path)
    published = Path(published_root) if published_root is not None else index.parent
    text = index.read_text(encoding="utf-8")
    records = []
    for link in sorted(set(_LINK_PATTERN.findall(text))):
        if link.startswith(("http://", "https://", "#", "data:")):
            continue
        safe = _safe_relative(link)
        target = index.parent / safe
        records.append(
            {
                "link": safe,
                "resolved_path": str(published / safe),
                "exists": target.is_file(),
            }
        )
    result = pd.DataFrame(records, columns=["link", "resolved_path", "exists"])
    if not result.empty and not result["exists"].all():
        missing = result.loc[~result["exists"], "link"].tolist()
        raise FileNotFoundError(f"Missing report links: {missing}")
    return result


def _atomic_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite chart: {path}")
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        fig.savefig(temporary, dpi=170, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(fig)


def _zoom_limits(values: np.ndarray, *, include_zero: bool = False) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if include_zero:
        finite = np.append(finite, 0.0)
    if not len(finite):
        return -1.0, 1.0
    low, high = float(finite.min()), float(finite.max())
    span = high - low
    padding = 0.12 * span if span else max(abs(low) * 0.12, 1e-6)
    return low - padding, high + padding


def publish_goal_charts(report_root: str | Path) -> pd.DataFrame:
    """Publish focused charts from already-audited analysis tables."""

    root = Path(report_root)
    if not root.name.endswith(".inprogress"):
        raise ValueError("Charts may only be published into the active .inprogress report")
    tables = root / "tables"
    charts = root / "charts"
    charts.mkdir(exist_ok=True)
    inventory: list[dict[str, object]] = []

    def register(filename: str, title: str, sources: list[str], fig: plt.Figure) -> None:
        path = charts / filename
        _atomic_figure(fig, path)
        inventory.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "title": title,
                "source_tables": ";".join(f"tables/{source}" for source in sources),
            }
        )

    outcomes = pd.read_csv(tables / "triad_outcomes_80.csv")
    counts = [
        int(outcomes["dual_improvement"].astype(bool).sum()),
        int(outcomes["high_value"].astype(bool).sum()),
        int(outcomes["dual_harm"].astype(bool).sum()),
        int(outcomes["exclusive_cohort"].eq("MIXED_OR_REVERSAL").sum()),
    ]
    labels = ["Dual improvement", "High value*", "Dual harm", "Mixed/reversal"]
    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    bars = ax.bar(labels, counts, color=["#238636", "#2da44e", "#cf222e", "#6e7781"])
    ax.set_ylabel("Triad count")
    ax.set_title("Outcome cohorts across all 80 triads")
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, str(value), ha="center")
    ax.text(0.01, -0.19, "*High value overlaps dual improvement; mixed is exclusive.", transform=ax.transAxes, fontsize=9)
    register("01_outcome_cohorts.png", "Outcome cohorts", ["triad_outcomes_80.csv"], fig)

    raw_summary = __import__("json").loads(
        (tables / "raw_frontier_analysis_summary.json").read_text(encoding="utf-8")
    )
    funnel_values = [
        counts[0],
        int(raw_summary["raw_dual_control_safe_frontier_dominant_triads"]),
        int(raw_summary["raw_dual_control_full_frontier_dominant_triads"]),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    bars = ax.bar(
        ["Fixed-point dual gain", "Raw FN0-95 dual dominance", "Raw full-range dual dominance"],
        funnel_values,
        color=["#2da44e", "#bf8700", "#cf222e"],
    )
    ax.set_ylabel("Triad count")
    ax.set_title("Robustness funnel: fixed operating points to full raw frontier")
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, funnel_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, str(value), ha="center")
    register(
        "02_raw_frontier_funnel.png",
        "Raw-frontier robustness funnel",
        ["triad_outcomes_80.csv", "raw_frontier_analysis_summary.json"],
        fig,
    )

    late = pd.read_csv(tables / "targeted_late_dynamics_contrasts.csv")
    late = late.loc[
        late["feature"].astype(str).str.startswith("extra_train_loss_decline__at_")
    ].copy()
    late["cutoff"] = late["feature"].astype(str).str.rsplit("_", n=1).str[-1].astype(int)
    late_plot = late.groupby("cutoff")[["positive_mean", "negative_mean"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9.4, 5.7), constrained_layout=True)
    ax.plot(late_plot["cutoff"], late_plot["positive_mean"], marker="o", label="Dual improvement")
    ax.plot(late_plot["cutoff"], late_plot["negative_mean"], marker="o", label="Dual harm")
    values = late_plot[["positive_mean", "negative_mean"]].to_numpy(dtype=float).ravel()
    ax.set_ylim(*_zoom_limits(values))
    ax.set_xlabel("Observed epoch cutoff")
    ax.set_ylabel("Extra train-loss decline vs controls")
    ax.set_title("Late-fit cohort association [Zoomed y-axis]")
    ax.legend()
    ax.grid(alpha=0.22)
    ax.text(0.01, -0.18, "Descriptive across cohorts; not reproduced within exact-selection reversal blocks.", transform=ax.transAxes, fontsize=9)
    register(
        "03_late_fit_association.png",
        "Late-fit association",
        ["targeted_late_dynamics_contrasts.csv", "reversal_analysis_summary.json"],
        fig,
    )

    validation = pd.read_csv(tables / "joint_prediction_summaries.csv")
    loso = validation.loc[validation["validation_scheme"].eq("DISCOVERY_LOSO_SEED")].sort_values("cutoff")
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 7.2), sharex=True, constrained_layout=True)
    for column, label, color in [
        ("roc_auc", "ROC AUC", "#0969da"),
        ("average_precision", "Average precision", "#8250df"),
    ]:
        values = pd.to_numeric(loso[column], errors="coerce").to_numpy(dtype=float)
        axes[0].plot(loso["cutoff"], values, marker="o", label=label, color=color)
    axes[0].set_ylim(*_zoom_limits(loso[["roc_auc", "average_precision"]].to_numpy(dtype=float).ravel()))
    axes[0].set_ylabel("Discrimination metric")
    axes[0].set_title("LOSO diagnostic discrimination [Zoomed y-axis; not success probability]")
    axes[0].legend(); axes[0].grid(alpha=0.22)
    numeric_cutoffs = pd.to_numeric(loso["cutoff"], errors="raise").to_numpy(dtype=float)
    bars = axes[1].bar(
        numeric_cutoffs,
        loso["selected_n"],
        width=6.0,
        color="#bf8700",
    )
    axes[1].set_xticks(numeric_cutoffs)
    for bar, value in zip(bars, loso["selected_n"]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            float(value) + 0.03,
            str(int(value)),
            ha="center",
        )
    axes[1].set_ylabel("Strictly selected candidates")
    axes[1].set_xlabel("Feature availability cutoff (epoch)")
    axes[1].set_title("Candidates passing the frozen 80% precision gate")
    axes[1].grid(axis="y", alpha=0.22)
    register(
        "04_unseen_seed_validation.png",
        "Leakage-safe unseen-seed validation",
        ["joint_prediction_summaries.csv"],
        fig,
    )

    overlap = pd.read_csv(tables / "selection_triad_overlap_audit.csv")
    r2 = overlap.loc[overlap["right_arm"].astype(str).eq("R2")].merge(
        outcomes[["triad_id", "phase"]], on="triad_id", how="left", validate="many_to_one"
    )
    phase_values = r2.groupby("phase")["effective_unique_contrast_rate"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    bars = ax.bar(phase_values.index.astype(str), phase_values.values * 100, color="#8250df")
    ax.set_ylabel("Effective unique contrast (%)")
    ax.set_xlabel("Phase")
    ax.set_title("R2 treatment-distinguishing power by phase")
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, phase_values.values * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.3, f"{value:.2f}%", ha="center")
    register(
        "05_r2_unique_contrast.png",
        "R2 effective unique contrast",
        ["selection_triad_overlap_audit.csv", "triad_outcomes_80.csv"],
        fig,
    )

    reversal = pd.read_csv(tables / "reversal_raw_mechanism_contrasts.csv")
    focus_names = [
        "raw_tail__mean_shift__normal__operational",
        "raw_tail__mean_shift__defect__operational",
        "raw_tail__beneficial_rate__defect__operational",
        "raw_frontier__worst_delta_TN_at_FN95",
    ]
    focus = reversal.loc[reversal["feature"].isin(focus_names)].copy()
    focus["short"] = focus["feature"].map(
        {
            "raw_tail__mean_shift__normal__operational": "Normal tail mean shift",
            "raw_tail__mean_shift__defect__operational": "Weak-defect mean shift",
            "raw_tail__beneficial_rate__defect__operational": "Weak-defect beneficial rate",
            "raw_frontier__worst_delta_TN_at_FN95": "Worst delta TN at FN95",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0), constrained_layout=True)
    mechanism_panels = [
        (
            "raw_tail__mean_shift__normal__operational",
            "Operational high-risk normal tail",
            "Mean raw-score shift",
        ),
        (
            "raw_tail__mean_shift__defect__operational",
            "Operational weakest 95 defects",
            "Mean raw-score shift",
        ),
        (
            "raw_tail__beneficial_rate__defect__operational",
            "Weak-defect beneficial share",
            "Share with higher raw score",
        ),
        (
            "raw_frontier__worst_delta_TN_at_FN95",
            "Worst paired delta TN at FN95",
            "Normal images",
        ),
    ]
    for ax, (feature, title, ylabel) in zip(axes.ravel(), mechanism_panels):
        row = focus.loc[focus["feature"].eq(feature)]
        if row.empty:
            ax.set_visible(False)
            continue
        values = row[["good_mean", "harm_mean"]].iloc[0].to_numpy(dtype=float)
        bars = ax.bar(["Good", "Harm"], values, color=["#238636", "#cf222e"])
        ax.axhline(0, color="#57606a", linestyle="--", linewidth=1)
        if feature == "raw_tail__beneficial_rate__defect__operational":
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(*_zoom_limits(values, include_zero=True))
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
        for bar, value in zip(bars, values):
            label = f"{value:.4g}" if abs(value) < 10 else f"{value:,.0f}"
            offset = 0.025 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (offset if value >= 0 else -offset),
                label,
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
            )
    register(
        "06_same_selection_raw_mechanism.png",
        "Same-selection reversal raw mechanism",
        ["reversal_raw_mechanism_contrasts.csv"],
        fig,
    )

    checkpoint = pd.read_csv(tables / "checkpoint_cohort_contrasts.csv")
    ck = checkpoint.loc[checkpoint["feature"].astype(str).str.endswith("__ALL")].copy()
    ck["short"] = ck["feature"].astype(str).str.replace("ckpt__", "", regex=False).str.replace("__ALL", "", regex=False)
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.8), constrained_layout=True)
    checkpoint_panels = [
        ("cosine_similarity", "Cosine similarity delta", "Cosine units"),
        ("relative_l2", "Relative L2 delta", "Relative L2 units"),
        ("delta_l2", "Absolute L2 delta", "L2 norm units"),
    ]
    for ax, (token, title, xlabel) in zip(axes, checkpoint_panels):
        subset = ck.loc[ck["feature"].astype(str).str.contains(token, regex=False)].copy()
        if token == "delta_l2":
            subset = subset.loc[
                ~subset["feature"].astype(str).str.contains("relative_l2", regex=False)
            ]
        if subset.empty:
            ax.set_visible(False)
            continue
        y = np.arange(len(subset))
        means = subset["mean_difference"].to_numpy(dtype=float)
        lows = subset["seed_bootstrap_ci_low"].to_numpy(dtype=float)
        highs = subset["seed_bootstrap_ci_high"].to_numpy(dtype=float)
        ax.errorbar(
            means,
            y,
            xerr=np.vstack([means - lows, highs - means]),
            fmt="o",
            color="#0969da",
            capsize=3,
        )
        ax.axvline(0, color="#57606a", linestyle="--", linewidth=1)
        labels = (
            subset["short"]
            .astype(str)
            .str.replace(f"delta_{token}__", "", regex=False)
            .str.replace("_", " ", regex=False)
        )
        ax.set_yticks(y, labels)
        ax.set_xlim(*_zoom_limits(np.concatenate([lows, highs]), include_zero=True))
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.22)
    fig.suptitle("Checkpoint drift contrasts (good minus harm; seed bootstrap)")
    register(
        "07_checkpoint_drift.png",
        "Checkpoint drift contrasts",
        ["checkpoint_cohort_contrasts.csv"],
        fig,
    )

    condition = pd.read_csv(tables / "condition_performance_summary.csv")
    condition = condition.sort_values(["dual_improvement", "dual_harm"], ascending=[False, True])
    x = np.arange(len(condition))
    fig, ax = plt.subplots(figsize=(15.2, 6.2), constrained_layout=True)
    ax.bar(x - 0.18, condition["dual_improvement"], width=0.36, label="Dual improvement", color="#238636")
    ax.bar(x + 0.18, condition["dual_harm"], width=0.36, label="Dual harm", color="#cf222e")
    condition_labels = (
        condition["condition_slot"].astype(str)
        + "/"
        + condition["budget"].astype(str)
        + "\n"
        + condition["discovery_or_confirmation"].astype(str)
    )
    ax.set_xticks(x, condition_labels, rotation=58, ha="right")
    ax.set_ylabel("Triad count within condition")
    ax.set_title("Condition outcomes (mostly 3 seeds; exploratory)")
    ax.legend(); ax.grid(axis="y", alpha=0.22)
    register(
        "08_condition_outcomes.png",
        "Condition-level good/harm counts",
        ["condition_performance_summary.csv"],
        fig,
    )

    literature = pd.read_csv(tables / "literature_evidence_matrix.csv")
    testability = literature["testability_status"].value_counts()
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    bars = ax.bar(testability.index.astype(str), testability.values, color="#0969da")
    ax.set_ylabel("Primary-source hypotheses")
    ax.set_title("Literature hypothesis testability in the 240-run evidence")
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, testability.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.15, str(value), ha="center")
    register(
        "09_literature_testability.png",
        "Literature hypothesis testability",
        ["literature_evidence_matrix.csv"],
        fig,
    )

    return pd.DataFrame(inventory)
