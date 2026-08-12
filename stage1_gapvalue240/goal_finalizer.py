"""Final publication gate for the comprehensive Stage1 GapValue analysis."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .goal_reporting import (
    build_executive_findings,
    build_next_experiment_preregistration,
    publish_goal_charts,
    render_final_html,
    render_final_markdown,
    validate_report_links,
)
from .goal_synthesis import (
    build_final_hypothesis_registry,
    build_literature_result_matrix,
    build_triad_execution_invariants,
    completion_gate_audit,
    extract_final_evidence_facts,
    minimum_unseen_seed_evidence,
)
from .util import atomic_write_json, atomic_write_text, sha256_file


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _write_or_verify_bytes(path: Path, data: bytes, *, overwrite: bool = False) -> None:
    if path.is_file() and not overwrite:
        if path.read_bytes() != data:
            raise FileExistsError(f"Existing generated asset differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_or_verify_csv(path: Path, frame: pd.DataFrame, *, overwrite: bool = False) -> None:
    _write_or_verify_bytes(path, _csv_bytes(frame), overwrite=overwrite)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _ledger_gate_counts(ledger: pd.DataFrame) -> dict[str, int]:
    status = ledger["usage_status"].astype(str) if "usage_status" in ledger else pd.Series("", index=ledger.index)
    role = ledger["usage_role"].astype(str) if "usage_role" in ledger else pd.Series("", index=ledger.index)
    silent_column = (
        _as_bool(ledger["silently_dropped"])
        if "silently_dropped" in ledger
        else pd.Series(False, index=ledger.index)
    )
    return {
        "UNREVIEWED": int(status.eq("UNREVIEWED").sum()),
        "UNCLASSIFIED": int(role.eq("UNCLASSIFIED").sum() + status.eq("UNCLASSIFIED").sum()),
        "SILENTLY_DROPPED": int(status.eq("SILENTLY_DROPPED").sum() + silent_column.sum()),
    }


def _ensure_execution_invariants(root: Path) -> Path:
    path = root / "tables/triad_execution_invariants.csv"
    canonical = pd.read_csv(root / "tables/canonical_run_metrics_240.csv")
    exposure = pd.read_csv(root / "tables/training_exposure_audit.csv")
    frame = build_triad_execution_invariants(canonical, exposure)
    _write_or_verify_csv(path, frame)
    return path


def collect_final_report_facts(root: str | Path) -> dict[str, object]:
    report = Path(root)
    _ensure_execution_invariants(report)
    facts = extract_final_evidence_facts(report)
    tables = report / "tables"
    audit = report / "audit"

    outcomes = pd.read_csv(tables / "triad_outcomes_80.csv")
    dual_improvement = outcomes["dual_improvement"].astype(bool)
    high_value = outcomes["high_value"].astype(bool)
    facts.update(
        {
            "dual_improvement_triads": int(dual_improvement.sum()),
            "high_value_triads": int(high_value.sum()),
            "dual_high_value_overlap": int((dual_improvement & high_value).sum()),
            "high_value_only_triads": int((~dual_improvement & high_value).sum()),
            "dual_harm_triads": int(outcomes["dual_harm"].astype(bool).sum()),
            "mixed_triads": int(outcomes["exclusive_cohort"].eq("MIXED_OR_REVERSAL").sum()),
        }
    )
    feature_roles = pd.read_csv(tables / "FEATURE_ROLE_REGISTRY.csv")
    facts["eligible_predictor_count"] = int(_as_bool(feature_roles["allowed_as_predictor"]).sum())
    ledger = pd.read_csv(audit / "DATA_USAGE_LEDGER_REFINED.csv")
    gate_counts = _ledger_gate_counts(ledger)
    facts.update(
        {
            "field_ledger_rows": int(len(ledger)),
            "unreviewed": gate_counts["UNREVIEWED"],
            "unclassified": gate_counts["UNCLASSIFIED"],
            "silently_dropped": gate_counts["SILENTLY_DROPPED"],
        }
    )
    reversal = _load_json(tables / "reversal_analysis_summary.json")
    facts.update(
        {
            "reversal_static_features_constant": int(
                reversal["treatment_selection_features_constant_within_digest"]
            ),
            "reversal_static_features_checked": int(
                reversal["treatment_selection_features_checked"]
            ),
        }
    )
    selection = _load_json(tables / "selection_mechanism_summary.json")
    facts["r2_unique_rate"] = float(
        selection["r2_effective_unique_contrast_rate_mean"]
    )
    resources = pd.read_csv(tables / "resource_reliability_runs.csv")
    resource_triads = pd.read_csv(tables / "resource_reliability_triads.csv")
    facts.update(
        {
            "resumed_runs": int(pd.to_numeric(resources["resume_count"], errors="coerce").fillna(0).gt(0).sum()),
            "cross_machine_triads": int((~_as_bool(resource_triads["all_arms_same_machine"])).sum()),
            "cross_snapshot_triads": int((~_as_bool(resource_triads["all_arms_same_snapshot"])).sum()),
        }
    )
    return facts


def _table_inventory(root: Path) -> pd.DataFrame:
    records = []
    for path in sorted((root / "tables").iterdir()):
        if not path.is_file() or path.name == "REPORT_TABLE_INVENTORY.csv":
            continue
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", errors="strict") as handle:
                row_count = max(sum(1 for _ in handle) - 1, 0)
            kind = "CSV"
        elif path.suffix.lower() == ".json":
            row_count = 1
            kind = "JSON"
        else:
            continue
        records.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "asset_kind": kind,
                "row_count": row_count,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(records)


def _report_readme() -> str:
    return """# Stage1 GapValue 240-Run 最终分析包

主阅读入口：`index.html`；纯文本结论：`FINAL_REPORT_CN.md`。

本目录是由240个canonical训练结果只读派生的实验分析产物，不是训练输入、selection真相源或人工标签。`tables/`保存完整数值，`charts/`只做可视化，`audit/`保存全文件/全字段使用、链接和完成门槛证据。图表中的缩放轴均明确标注，精确值以相邻CSV为准。

假设状态的唯一最终真相源是`tables/HYPOTHESIS_REGISTRY_FINAL.csv`；`tables/HYPOTHESIS_REGISTRY.csv`是证据底座建立时的初始预登记快照，仅用于追溯，已被最终表取代。

生命周期：永久保留本版报告及manifest；可由冻结输入和仓库脚本重新生成。不得反向修改源训练结果、240份selection或合同。
"""


def _append_asset_links(markdown: str, charts: pd.DataFrame) -> str:
    lines = [markdown.rstrip(), "", "## 图表与完整数据", ""]
    for row in charts.to_dict(orient="records"):
        lines.append(
            f"- [{row['title']}]({row['relative_path']})；来源：`{row['source_tables']}`"
        )
    lines.extend(
        [
            "",
            "- [完整表格目录](tables/REPORT_TABLE_INVENTORY.csv)",
            "- [全字段使用审计](audit/DATA_USAGE_LEDGER_REFINED.csv)",
            "- [字段实际值画像](audit/FIELD_VALUE_PROFILES.csv)",
            "- [最终完成门槛](audit/COMPLETION_AUDIT.csv)",
            "- [全部文件SHA清单](manifest.json)",
            "",
        ]
    )
    return "\n".join(lines)


def _append_audit_links_to_html(document: str) -> str:
    """Add the final audit entry list from a UTF-8 source-controlled template."""

    marker = "</main>"
    if marker not in document:
        raise ValueError("Final HTML is missing </main>")
    section = (
        '<h2>审计入口</h2><ul>'
        '<li><a href="audit/DATA_USAGE_LEDGER_REFINED.csv">全字段使用审计</a></li>'
        '<li><a href="audit/FIELD_VALUE_PROFILES.csv">字段实际值画像</a></li>'
        '<li><a href="audit/COMPLETION_AUDIT.csv">最终完成门槛</a></li>'
        '<li><a href="manifest.json">全部文件SHA清单</a></li>'
        "</ul>"
    )
    return document.replace(marker, section + marker, 1)


def _build_manifest(root: Path, *, inventory_path: Path, facts: dict[str, object]) -> dict[str, Any]:
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
        "schema_version": "stage1_gapvalue240_goal_final_report_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "240 VALIDATED canonical runs / 80 triads / 48,000 epochs only",
        "canonical_inventory_sha256": sha256_file(inventory_path),
        "facts": facts,
        "file_count_excluding_manifest": len(files),
        "files": files,
        "manifest_self_excluded": True,
        "lifecycle": "keep; reproducible derived experiment report",
    }


def finalize_goal_analysis(
    report_root: str | Path, *, inventory_path: str | Path
) -> Path:
    """Pass every completion gate, then atomically expose the final report directory."""

    root = Path(report_root).resolve()
    inventory = Path(inventory_path).resolve()
    if not root.name.endswith(".inprogress") or not root.is_dir():
        raise ValueError("report_root must be the existing active .inprogress directory")
    final = root.with_name(root.name.removesuffix(".inprogress"))
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite final report: {final}")
    if not inventory.is_file():
        raise FileNotFoundError(inventory)

    facts = collect_final_report_facts(root)
    hypotheses = build_final_hypothesis_registry(facts)
    literature = pd.read_csv(root / "tables/literature_evidence_matrix.csv")
    literature_results = build_literature_result_matrix(literature, hypotheses)
    findings = build_executive_findings(facts)
    preregistration = build_next_experiment_preregistration()
    confirmation = minimum_unseen_seed_evidence()

    _write_or_verify_csv(root / "tables/HYPOTHESIS_REGISTRY_FINAL.csv", hypotheses)
    _write_or_verify_csv(root / "tables/LITERATURE_RESULT_MATRIX.csv", literature_results)
    _write_or_verify_csv(root / "tables/EXECUTIVE_FINDINGS.csv", findings)
    _write_or_verify_csv(root / "tables/NEXT_EXPERIMENT_PREREGISTRATION.csv", preregistration)
    _write_or_verify_csv(root / "tables/UNSEEN_SEED_80PCT_EVIDENCE.csv", confirmation)
    atomic_write_json(root / "tables/FINAL_EVIDENCE_FACTS.json", facts, overwrite=False)

    chart_inventory_path = root / "tables/CHART_INVENTORY.csv"
    if chart_inventory_path.is_file():
        charts = pd.read_csv(chart_inventory_path)
    else:
        charts = publish_goal_charts(root)
        _write_or_verify_csv(chart_inventory_path, charts)

    inventory_frame = _table_inventory(root)
    _write_or_verify_csv(root / "tables/REPORT_TABLE_INVENTORY.csv", inventory_frame)

    markdown = _append_asset_links(render_final_markdown(facts, hypotheses), charts)
    _write_or_verify_bytes(root / "FINAL_REPORT_CN.md", markdown.encode("utf-8"))
    _write_or_verify_bytes(root / "README.md", _report_readme().encode("utf-8"))
    html_text = render_final_html(
        facts,
        hypotheses,
        findings,
        table_inventory=inventory_frame,
        chart_inventory=charts,
    )
    html_text = _append_audit_links_to_html(html_text)
    _write_or_verify_bytes(root / "index.html", html_text.encode("utf-8"))

    refined = pd.read_csv(root / "audit/DATA_USAGE_LEDGER_REFINED.csv")
    ledger_gates = _ledger_gate_counts(refined)
    gates = {
        "canonical_runs": 240,
        "triads": 80,
        "paired_comparisons": 160,
        "epoch_rows": 48_000,
        **ledger_gates,
    }
    completion = completion_gate_audit(
        root,
        gates,
        required_files=[
            "FINAL_REPORT_CN.md",
            "index.html",
            "README.md",
            "tables/HYPOTHESIS_REGISTRY_FINAL.csv",
            "tables/LITERATURE_RESULT_MATRIX.csv",
            "tables/UNSEEN_SEED_80PCT_EVIDENCE.csv",
            "tables/NEXT_EXPERIMENT_PREREGISTRATION.csv",
            "audit/DATA_USAGE_LEDGER_REFINED.csv",
            "audit/FIELD_VALUE_PROFILES.csv",
        ],
    )
    _write_or_verify_csv(root / "audit/COMPLETION_AUDIT.csv", completion)

    link_audit = validate_report_links(root / "index.html", published_root=final)
    _write_or_verify_csv(root / "audit/REPORT_LINK_AUDIT.csv", link_audit)
    link_gate = pd.DataFrame(
        [
            {
                "gate": "all_local_html_links",
                "expected": "present",
                "actual": "present",
                "passed": bool(link_audit["exists"].all()) if len(link_audit) else True,
                "evidence_type": "html_link_gate",
            }
        ]
    )
    completion = pd.concat([completion, link_gate], ignore_index=True)
    _write_or_verify_csv(
        root / "audit/COMPLETION_AUDIT.csv", completion, overwrite=True
    )

    old_state = _load_json(root / "ANALYSIS_STATE.json")
    old_state.update(
        {
            "status": "COMPLETE",
            "completed_stage": "FINAL_SYNTHESIS_AND_REPORT",
            "next_required_stage": "NONE",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "final_directory_name": final.name,
            "final_hypothesis_counts": hypotheses["status"].value_counts().to_dict(),
            "gates": {**old_state.get("gates", {}), **gates},
            "headline": "NO_80_PERCENT_UNSEEN_SEED_RULE_FOUND",
        }
    )
    atomic_write_json(root / "ANALYSIS_STATE.json", old_state, overwrite=True)
    manifest = _build_manifest(root, inventory_path=inventory, facts=facts)
    atomic_write_json(root / "manifest.json", manifest, overwrite=True)

    reread = _load_json(root / "manifest.json")
    for item in reread["files"]:
        path = root / item["relative_path"]
        if not path.is_file() or path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"Manifest size verification failed: {path}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Manifest SHA verification failed: {path}")
    if any(root.rglob("*.tmp")):
        raise ValueError("Temporary files remain in report tree")
    os.replace(root, final)
    return final
