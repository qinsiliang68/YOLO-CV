from __future__ import annotations

import pandas as pd

from stage1_gapvalue240.goal_finalizer import _append_audit_links_to_html
from stage1_gapvalue240.goal_reporting import (
    build_executive_findings,
    build_next_experiment_preregistration,
    publish_goal_charts,
    render_final_markdown,
    render_final_html,
    validate_report_links,
)


def _facts() -> dict[str, object]:
    return {
        "total_triads": 80,
        "dual_improvement_triads": 15,
        "high_value_triads": 13,
        "dual_high_value_overlap": 12,
        "high_value_only_triads": 1,
        "dual_harm_triads": 23,
        "mixed_triads": 41,
        "raw_dual_safe_triads": 1,
        "raw_full_frontier_dual_triads": 0,
        "same_selection_reversal_groups": 6,
        "same_selection_reversal_triads": 23,
        "phase_c_successes": 0,
        "phase_c_total": 5,
        "joint_rule_phase_c_successes": 0,
        "joint_rule_phase_c_total": 5,
        "eligible_predictor_count": 3790,
        "field_ledger_rows": 1351,
        "unreviewed": 0,
        "unclassified": 0,
        "silently_dropped": 0,
        "reversal_static_features_constant": 780,
        "reversal_static_features_checked": 780,
        "r2_unique_rate": 0.1947208333,
        "resumed_runs": 15,
        "cross_machine_triads": 24,
        "cross_snapshot_triads": 11,
    }


def test_executive_findings_do_not_call_auc_success_probability() -> None:
    findings = build_executive_findings(_facts())

    assert findings["finding_id"].is_unique
    assert "80%" in " ".join(findings["finding_cn"])
    assert not findings["finding_cn"].str.contains("AUC就是成功概率", regex=False).any()
    assert set(findings["evidence_level"]).issubset(
        {"EXACT_AUDIT", "DESCRIPTIVE", "HELD_OUT_TEST", "MECHANISM", "LIMITATION"}
    )
    cohort_text = findings.loc[
        findings["finding_id"].eq("F01_FIXED_POINT_OUTCOMES"), "finding_cn"
    ].item()
    assert "12个重合" in cohort_text
    assert "互斥计数为15+1+23+41=80" in cohort_text


def test_next_experiment_requires_unseen_seed_count_and_no_replay_arm() -> None:
    plan = build_next_experiment_preregistration()

    assert {"NR_NO_REPLAY", "R1_GLOBAL_RANDOM", "R2_MATCHED_RANDOM"}.issubset(
        set(plan["arm_id"])
    )
    assert plan["independent_confirmation_seeds"].min() >= 14
    assert plan["raw_frontier_required"].all()
    assert plan["frozen_checkpoints"].str.contains("120,140,150,160,180,200").all()


def test_markdown_is_clean_chinese_and_keeps_claim_boundaries() -> None:
    hypotheses = build_executive_findings(_facts()).rename(
        columns={"finding_id": "hypothesis_id", "finding_cn": "hypothesis"}
    )
    hypotheses["status"] = "INCONCLUSIVE"
    text = render_final_markdown(_facts(), hypotheses)

    assert "本报告严格只使用本次240个VALIDATED canonical run" in text
    assert "没有no-replay arm" in text
    assert "没有blind/external test" in text
    assert "尚未找到80%可靠规则" in text
    assert "�" not in text
    assert "锛" not in text


def test_html_links_are_relative_and_validated(tmp_path) -> None:
    facts = _facts()
    findings = build_executive_findings(facts)
    hypotheses = findings.rename(
        columns={"finding_id": "hypothesis_id", "finding_cn": "hypothesis"}
    )
    hypotheses["status"] = "INCONCLUSIVE"
    (tmp_path / "tables").mkdir()
    (tmp_path / "charts").mkdir()
    (tmp_path / "tables" / "a.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "charts" / "a.png").write_bytes(b"png")
    html = render_final_html(
        facts,
        hypotheses,
        findings,
        table_inventory=pd.DataFrame(
            {"relative_path": ["tables/a.csv"], "row_count": [1]}
        ),
        chart_inventory=pd.DataFrame(
            {
                "relative_path": ["charts/a.png"],
                "title": ["A chart"],
                "source_tables": ["tables/a.csv"],
            }
        ),
    )
    (tmp_path / "index.html").write_text(html, encoding="utf-8")

    audit = validate_report_links(tmp_path / "index.html")
    assert audit["exists"].all()
    assert "阈值滑动" in html
    assert "�" not in html


def test_html_marks_the_initial_hypothesis_registry_as_superseded(tmp_path) -> None:
    facts = _facts()
    findings = build_executive_findings(facts)
    hypotheses = findings.rename(
        columns={"finding_id": "hypothesis_id", "finding_cn": "hypothesis"}
    )
    hypotheses["status"] = "INCONCLUSIVE"
    html = render_final_html(
        facts,
        hypotheses,
        findings,
        table_inventory=pd.DataFrame(
            {
                "relative_path": [
                    "tables/HYPOTHESIS_REGISTRY.csv",
                    "tables/HYPOTHESIS_REGISTRY_FINAL.csv",
                ],
                "row_count": [7, 14],
            }
        ),
        chart_inventory=pd.DataFrame(
            columns=["relative_path", "title", "source_tables"]
        ),
    )

    assert "HYPOTHESIS_REGISTRY.csv [INITIAL, SUPERSEDED]" in html
    assert "HYPOTHESIS_REGISTRY_FINAL.csv [FINAL TRUTH]" in html


def test_final_html_audit_links_are_clean_utf8() -> None:
    html = _append_audit_links_to_html("<main><p>正文</p></main>")

    assert "审计入口" in html
    assert "全字段使用审计" in html
    assert "????" not in html
    assert html.endswith("</main>")


def test_link_audit_can_record_the_final_publish_root(tmp_path) -> None:
    active = tmp_path / "report.inprogress"
    final = tmp_path / "report"
    (active / "tables").mkdir(parents=True)
    (active / "tables" / "a.csv").write_text("x\n1\n", encoding="utf-8")
    (active / "index.html").write_text(
        '<a href="tables/a.csv">table</a>', encoding="utf-8"
    )

    audit = validate_report_links(
        active / "index.html", published_root=final
    )

    assert audit["exists"].all()
    assert audit.loc[0, "resolved_path"] == str(final / "tables" / "a.csv")


def test_focused_charts_keep_units_separate_and_condition_labels_unique(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "report.inprogress"
    tables = root / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame(
        {
            "triad_id": ["T1", "T2"],
            "dual_improvement": [True, False],
            "high_value": [True, False],
            "dual_harm": [False, True],
            "exclusive_cohort": ["DUAL_IMPROVEMENT", "DUAL_HARM"],
            "phase": ["A", "B"],
        }
    ).to_csv(tables / "triad_outcomes_80.csv", index=False)
    (tables / "raw_frontier_analysis_summary.json").write_text(
        '{"raw_dual_control_safe_frontier_dominant_triads":1,'
        '"raw_dual_control_full_frontier_dominant_triads":0}',
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "feature": ["extra_train_loss_decline__at_150"] * 2,
            "cutoff": [150, 150],
            "positive_mean": [0.0, 0.0],
            "negative_mean": [0.1, 0.1],
        }
    ).to_csv(tables / "targeted_late_dynamics_contrasts.csv", index=False)
    pd.DataFrame(
        {
            "validation_scheme": ["DISCOVERY_LOSO_SEED"] * 2,
            "cutoff": [120, 200],
            "roc_auc": [0.55, 0.60],
            "average_precision": [0.25, 0.30],
            "selected_n": [0, 1],
        }
    ).to_csv(tables / "joint_prediction_summaries.csv", index=False)
    pd.DataFrame(
        {
            "triad_id": ["T1", "T2"],
            "right_arm": ["R2", "R2"],
            "effective_unique_contrast_rate": [0.2, 0.3],
        }
    ).to_csv(tables / "selection_triad_overlap_audit.csv", index=False)
    pd.DataFrame(
        {
            "feature": [
                "raw_tail__mean_shift__normal__operational",
                "raw_tail__mean_shift__defect__operational",
                "raw_tail__beneficial_rate__defect__operational",
                "raw_frontier__worst_delta_TN_at_FN95",
            ],
            "good_mean": [-0.001, 0.0002, 0.6, 800.0],
            "harm_mean": [-0.006, -0.0001, 0.4, -2100.0],
        }
    ).to_csv(tables / "reversal_raw_mechanism_contrasts.csv", index=False)
    pd.DataFrame(
        {
            "feature": [
                "ckpt__delta_cosine_similarity__best_to_last_ema__ALL",
                "ckpt__delta_relative_l2__best_to_last_ema__ALL",
                "ckpt__delta_delta_l2__best_to_last_ema__ALL",
            ],
            "mean_difference": [0.0001, 0.002, 2.0],
            "seed_bootstrap_ci_low": [-0.0001, -0.001, -1.0],
            "seed_bootstrap_ci_high": [0.0002, 0.003, 3.0],
        }
    ).to_csv(tables / "checkpoint_cohort_contrasts.csv", index=False)
    pd.DataFrame(
        {
            "discovery_or_confirmation": ["discovery", "confirmation"],
            "condition_slot": ["A02", "A02"],
            "budget": [3000, 3000],
            "dual_improvement": [1, 0],
            "dual_harm": [1, 3],
        }
    ).to_csv(tables / "condition_performance_summary.csv", index=False)
    pd.DataFrame(
        {
            "testability_status": ["DIRECTLY_TESTABLE", "NOT_TESTABLE"]
        }
    ).to_csv(tables / "literature_evidence_matrix.csv", index=False)

    captured = {}

    def capture(fig, path):
        captured[path.name] = fig

    monkeypatch.setattr("stage1_gapvalue240.goal_reporting._atomic_figure", capture)
    inventory = publish_goal_charts(root)

    unseen = captured["04_unseen_seed_validation.png"]
    assert unseen.axes[0].get_xlim() == unseen.axes[1].get_xlim()
    assert max(patch.get_x() for patch in unseen.axes[1].patches) > 100
    assert len(captured["06_same_selection_raw_mechanism.png"].axes) == 4
    assert len(captured["07_checkpoint_drift.png"].axes) == 3
    condition_labels = [
        tick.get_text() for tick in captured["08_condition_outcomes.png"].axes[0].get_xticklabels()
    ]
    assert len(condition_labels) == len(set(condition_labels))
    assert any("discovery" in label for label in condition_labels)
    assert any("confirmation" in label for label in condition_labels)
    assert len(inventory) == 9
