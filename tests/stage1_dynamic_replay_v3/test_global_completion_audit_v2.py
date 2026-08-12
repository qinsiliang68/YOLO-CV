from __future__ import annotations

from pathlib import Path

from stage1_dynamic_replay_v3.global_completion_audit_v2 import (
    build_global_completion_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "stage1_sample_value_experiments"
    / "experiments"
    / "dynamic_replay_budget_efficiency_20260807"
)
DESKTOP_LITERATURE_MIRROR = (
    Path(r"C:\Users\28898\Desktop") / "YOLO\u7b14\u8bb0"
    / "01_Stage1_\u6709\u9650\u9884\u7b97\u52a8\u6001\u56de\u6d41_\u6587\u732e\u8bc1\u636e_20260810"
)


def test_current_global_audit_passes_all_controllable_gates_and_blocks_on_source() -> None:
    report = build_global_completion_audit(
        repo_root=REPO_ROOT,
        experiment_root=EXPERIMENT_ROOT,
        desktop_literature_mirror=DESKTOP_LITERATURE_MIRROR,
    )

    assert report["status"] == "INCOMPLETE_SOURCE_MISSING"
    assert report["completion_allowed"] is False
    assert report["candidate_effectiveness_claim"] == "NOT_EVALUATED"
    assert report["blocking_conditions"] == [
        {
            "code": "REPORT_ONLY_SOURCE_MISSING",
            "artifacts": [
                "Stage1_BudgetedReplay_Learnability_20260809_v1.0.0.tar.gz",
                "Stage1_BudgetedReplay_Learnability_20260809_v1.0.0.zip",
                "stage1_budgeted_replay-1.0.0-py3-none-any.whl",
            ],
        }
    ]
    checks = report["checks"]
    for check_id in (
        "expert_assets_accounted_and_present_hashes_verified",
        "tripartite_46_rows_hash_bound",
        "literature_user_scope_decision_registered",
        "literature_desktop_mirror_integrity",
        "preregistration_v3_contract",
        "v3_test_suite",
        "formal_training_not_started",
        "active_v3_gate_and_assignments_not_generated",
        "blind_holdout_not_opened",
        "candidate_effectiveness_not_claimed",
    ):
        assert checks[check_id]["status"] == "PASS", (check_id, checks[check_id])
    assert checks["expert_source_level_audit"]["status"] == (
        "NOT_TESTABLE_SOURCE_MISSING"
    )
    assert checks["legacy_formal_literature_audit"]["status"] == (
        "SUPERSEDED_BY_USER_SCOPE_DECISION"
    )
    assert report["prohibited_actions"]["formal_training_started"] is False
    assert report["prohibited_actions"]["active_v3_engineering_gate_generated"] is False
    assert report["prohibited_actions"]["active_v3_assignments_generated"] is False
    assert report["prohibited_actions"]["blind_holdout_opened"] is False
    assert report["prohibited_actions"]["historical_gate_assets_detected"] is True
