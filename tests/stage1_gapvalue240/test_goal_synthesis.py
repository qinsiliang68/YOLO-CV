from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.goal_synthesis import (
    build_triad_execution_invariants,
    build_final_hypothesis_registry,
    build_literature_result_matrix,
    completion_gate_audit,
    extract_final_evidence_facts,
    minimum_unseen_seed_evidence,
)


def _facts() -> dict[str, object]:
    return {
        "late_loss_full_q_below_005_both_controls": True,
        "late_loss_discovery_q_below_005_both_controls": False,
        "late_loss_same_selection_supported": False,
        "late_loss_unseen_seed_rule_supported": False,
        "lr_schedule_has_150_160_kink": False,
        "within_triad_lr_and_exposure_equal": True,
        "checkpoint_seed_ci_excludes_zero": False,
        "raw_dual_safe_triads": 1,
        "raw_full_frontier_dual_triads": 0,
        "total_triads": 80,
        "same_selection_reversal_groups": 6,
        "same_selection_reversal_triads": 23,
        "phase_c_successes": 0,
        "phase_c_total": 5,
        "joint_rule_phase_c_successes": 0,
        "joint_rule_phase_c_total": 5,
        "gradients_collected": False,
        "epoch150_checkpoint_collected": False,
        "no_replay_arm_collected": False,
        "blind_external_collected": False,
        "learnability_seed_robust": False,
        "guard_seed_robust": False,
        "budget_response_seed_robust": False,
    }


def test_final_hypotheses_separate_descriptive_signal_from_unseen_seed_claim() -> None:
    registry = build_final_hypothesis_registry(_facts()).set_index("hypothesis_id")

    assert registry.loc["H01_LATE_EXTRA_FIT_ASSOCIATION", "status"] == "SUPPORTED"
    assert registry.loc["H02_LATE_EXTRA_FIT_EARLY_WARNING", "status"] == "NOT_SUPPORTED"
    assert registry.loc["H03_LR_SCHEDULE_KINK", "status"] == "NOT_SUPPORTED"
    assert registry.loc["H06_STATIC_SELECTION_STABILITY", "status"] == "NOT_SUPPORTED"
    assert registry.loc["H10_UNSEEN_SEED_80PCT", "status"] == "NOT_SUPPORTED"
    assert registry.loc["H11_TRUE_GRADIENT_ALIGNMENT", "status"] == "NOT_TESTABLE"
    assert set(registry["status"]).issubset(
        {"SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE", "NOT_TESTABLE"}
    )
    assert not registry["causal_claim_allowed"].any()


def test_literature_result_matrix_maps_every_source_without_inventing_testability() -> None:
    literature = pd.DataFrame(
        {
            "evidence_id": [
                "MEMORIZATION_DYNAMICS",
                "GRAND_MAGNITUDE",
                "OPERATIONAL_NEYMAN_PEARSON",
            ],
            "topic": ["memorization", "grand_el2n", "neyman_pearson"],
            "testability_status": [
                "DIRECTLY_TESTABLE",
                "NOT_TESTABLE",
                "DIRECTLY_TESTABLE",
            ],
            "missing_required_capabilities": ["", "per_sample_gradient", ""],
        }
    )
    result = build_literature_result_matrix(
        literature, build_final_hypothesis_registry(_facts())
    )

    assert len(result) == 3
    assert result["mapped_hypothesis_ids"].str.len().gt(0).all()
    grand = result.loc[result["evidence_id"].eq("GRAND_MAGNITUDE")].iloc[0]
    assert grand["study_result_status"] == "NOT_TESTABLE"
    assert "gradient" in grand["study_result_boundary"].lower()


def test_minimum_unseen_seed_evidence_matches_registered_80_percent_gate() -> None:
    evidence = minimum_unseen_seed_evidence(target_rate=0.8, confidence=0.95)

    assert evidence.loc[evidence["allowed_failures"].eq(0), "minimum_total"].item() == 14
    assert evidence.loc[evidence["allowed_failures"].eq(1), "minimum_total"].item() == 22
    assert evidence.loc[evidence["allowed_failures"].eq(2), "minimum_total"].item() == 30
    assert evidence["one_sided_lower_bound"].gt(0.8).all()


def test_completion_gate_requires_all_counts_and_zero_silent_drop(tmp_path: Path) -> None:
    required = ["FINAL_REPORT_CN.md", "index.html", "README.md"]
    for name in required:
        (tmp_path / name).write_text("ok", encoding="utf-8")
    gates = {
        "canonical_runs": 240,
        "triads": 80,
        "paired_comparisons": 160,
        "epoch_rows": 48_000,
        "UNREVIEWED": 0,
        "UNCLASSIFIED": 0,
        "SILENTLY_DROPPED": 0,
    }

    audit = completion_gate_audit(tmp_path, gates, required_files=required)
    assert audit["passed"].all()

    broken = dict(gates, epoch_rows=47_999)
    with pytest.raises(ValueError, match="epoch_rows"):
        completion_gate_audit(tmp_path, broken, required_files=required)


def test_extract_final_facts_uses_tables_instead_of_narrative_constants(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    audit = tmp_path / "audit"
    tables.mkdir()
    audit.mkdir()
    late = pd.DataFrame(
        {
            "control": ["R1", "R2"],
            "feature": [
                "extra_train_loss_decline__at_160",
                "extra_train_loss_decline__at_160",
            ],
            "q_value_bh": [0.04, 0.03],
        }
    )
    late.to_csv(tables / "targeted_late_dynamics_permutation_fdr.csv", index=False)
    late.assign(q_value_bh=[0.08, 0.09]).to_csv(
        tables / "targeted_late_dynamics_discovery_permutation_fdr.csv", index=False
    )
    pd.DataFrame(
        {
            "validation_scheme": ["PHASE_C_EXTERNAL_FALSIFICATION"],
            "n": [5],
            "selected_n": [0],
            "selected_successes": [0],
            "confirmed_above_target": [False],
        }
    ).to_csv(tables / "joint_prediction_summaries.csv", index=False)
    pd.DataFrame(
        {
            "phase": ["C"] * 5,
            "dual_improvement": [False] * 5,
            "triad_id": [f"T{i}" for i in range(5)],
        }
    ).to_csv(tables / "triad_outcomes_80.csv", index=False)
    (tables / "raw_frontier_analysis_summary.json").write_text(
        '{"canonical_triads":80,"raw_dual_control_safe_frontier_dominant_triads":1,'
        '"raw_dual_control_full_frontier_dominant_triads":0}',
        encoding="utf-8",
    )
    (tables / "selection_mechanism_summary.json").write_text(
        '{"same_selection_reversal_digests":6,"same_selection_reversal_triads":23,'
        '"gradient_fields_not_collected":5}',
        encoding="utf-8",
    )
    (tables / "reversal_analysis_summary.json").write_text(
        '{"focus_epoch_evidence":{"extra_train_loss_decline_at_200":'
        '{"permutation_p_two_sided":0.45,"fdr_q_global":1.0}}}',
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "max_abs_second_difference_after_120": [1e-16],
            "active_all_runs": [True],
        }
    ).to_csv(tables / "learning_rate_active_group_audit.csv", index=False)
    pd.DataFrame(
        {
            "seed_bootstrap_ci_low": [1.0],
            "seed_bootstrap_ci_high": [2.0],
            "condition_bootstrap_ci_low": [-1.0],
            "condition_bootstrap_ci_high": [2.0],
        }
    ).to_csv(tables / "checkpoint_cohort_contrasts.csv", index=False)
    pd.DataFrame(
        {
            "triad_id": ["T0", "T0", "T0"],
            "steps_per_epoch": [943, 943, 943],
            "replay_total_rows": [600, 600, 600],
            "lr_step_integral_pg3_to_200": [1.0, 1.0, 1.0],
        }
    ).to_csv(tables / "triad_execution_invariants.csv", index=False)
    pd.DataFrame(
        {
            "normalized_path": ["<NOT_COLLECTED>/evaluation"] * 3,
            "field_path": ["no_replay_arm", "blind_or_external_test", "epoch_150_checkpoint"],
            "usage_status": ["NOT_TESTABLE"] * 3,
        }
    ).to_csv(audit / "DATA_USAGE_LEDGER_REFINED.csv", index=False)

    facts = extract_final_evidence_facts(tmp_path)

    assert facts["late_loss_full_q_below_005_both_controls"] is True
    assert facts["late_loss_discovery_q_below_005_both_controls"] is False
    assert facts["late_loss_same_selection_supported"] is False
    assert facts["lr_schedule_has_150_160_kink"] is False
    assert facts["checkpoint_seed_ci_excludes_zero"] is False
    assert facts["phase_c_successes"] == 0
    assert facts["joint_rule_phase_c_total"] == 5


def test_triad_execution_invariants_preserve_arm_rows_and_reject_missing_arms() -> None:
    canonical = pd.DataFrame(
        {
            "run_slot": ["RUN_001", "RUN_002", "RUN_003"],
            "triad_id": ["TRIAD_001"] * 3,
            "arm": ["T", "R1", "R2"],
        }
    )
    exposure = pd.DataFrame(
        {
            "run_slot": ["RUN_001", "RUN_002", "RUN_003"],
            "steps_per_epoch": [943, 943, 943],
            "replay_total_rows": [600, 600, 600],
            "lr_step_integral_pg3_to_200": [1.0, 1.0, 1.0],
            "resume_count": [0, 0, 0],
        }
    )
    result = build_triad_execution_invariants(canonical, exposure)

    assert result.shape == (3, 6)
    assert set(result["arm"]) == {"T", "R1", "R2"}
    assert result.groupby("triad_id")["steps_per_epoch"].nunique().eq(1).all()

    with pytest.raises(ValueError, match="T/R1/R2"):
        build_triad_execution_invariants(canonical.iloc[:2], exposure.iloc[:2])
