from __future__ import annotations

import inspect
import json

import pandas as pd
import pytest

from stage1_gapvalue240.deep_analysis import CanonicalInputError
from stage1_gapvalue240.deep_pipeline import (
    _a02_training_curves,
    _reliability_tables,
    _tail_plot_summary,
    _training_contract_audit,
    _validate_prediction_identity,
    build_hypothesis_registry,
    build_threshold_frontier,
    run_deep_analysis,
    validate_completeness_audit,
)


def test_completeness_audit_requires_canonical_counts_and_semantic_snapshot_proof(
    tmp_path,
):
    audit = {
        "material_completeness": "COMPLETE",
        "unique_validated_run_count": 240,
        "triad_count": 80,
        "paired_comparison_count": 160,
        "missing_runs": [],
        "extra_runs": [],
        "duplicate_validated_runs": [],
        "input_snapshot_ids": ["a", "b"],
        "input_snapshot_semantic_audit": {
            "snapshot_count": 2,
            "newline_normalized_all_frozen_csv_equal": True,
        },
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit), encoding="utf-8")
    result = validate_completeness_audit(path)
    assert result["semantic_snapshot_exception_authorized"]

    audit["input_snapshot_semantic_audit"][
        "newline_normalized_all_frozen_csv_equal"
    ] = False
    path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(CanonicalInputError, match="semantic"):
        validate_completeness_audit(path)


def test_hypothesis_registry_is_conservative_about_failed_a02_and_guard_confound():
    a02 = pd.DataFrame(
        [
            {
                "analysis_cohort": cohort,
                "control": control,
                "n": n,
                "safety_noninferior": False,
                "confirmed_TN_improvement": False,
                "mean_delta_FN": 5.0,
                "mean_delta_TN": -500.0,
                "machine_confounded": cohort != "discovery",
            }
            for cohort, n in (("discovery", 3), ("confirmation", 5), ("pooled", 8))
            for control in ("R1", "R2")
        ]
    )
    conditions = pd.DataFrame(
        [
            {
                "phase": "A",
                "condition_slot": "A01",
                "control": control,
                "mean_delta_FN": -5.0,
                "mean_delta_TN": 700.0,
                "safety_noninferior": False,
            }
            for control in ("R1", "R2")
        ]
        + [
            {
                "phase": "A",
                "condition_slot": "A13",
                "control": control,
                "mean_delta_FN": 6.0,
                "mean_delta_TN": -700.0,
                "safety_noninferior": False,
            }
            for control in ("R1", "R2")
        ]
    )
    registry = build_hypothesis_registry(
        a02_summaries=a02,
        condition_summaries=conditions,
        budget_comparisons=pd.DataFrame(),
        guard_comparisons=pd.DataFrame(),
        r2_overlap=pd.DataFrame(
            {"condition_slot": ["A02"], "effective_unique_contrast": [0.08]}
        ),
    ).set_index("hypothesis_id")
    assert registry.loc["H1_A02_vs_R1", "status"] == "NOT_SUPPORTED"
    assert registry.loc["H2_A02_vs_R2", "status"] == "NOT_SUPPORTED"
    assert registry.loc["H4_directional_negative_control", "status"] == "NOT_SUPPORTED"
    assert registry.loc["H7_defect_guard", "status"] == "INCONCLUSIVE"


def test_hypothesis_registry_uses_contract_pooled_eight_not_only_discovery_three():
    a02 = pd.DataFrame(
        [
            {
                "analysis_cohort": cohort,
                "control": control,
                "n": n,
                "safety_noninferior": cohort == "discovery",
                "confirmed_TN_improvement": cohort == "discovery",
                "mean_delta_FN": -1.0 if cohort == "discovery" else 5.0,
                "mean_delta_TN": 100.0 if cohort == "discovery" else -500.0,
                "machine_confounded": cohort != "discovery",
            }
            for cohort, n in (("discovery", 3), ("confirmation", 5), ("pooled", 8))
            for control in ("R1", "R2")
        ]
    )
    registry = build_hypothesis_registry(
        a02_summaries=a02,
        condition_summaries=pd.DataFrame(),
        budget_comparisons=pd.DataFrame(),
        guard_comparisons=pd.DataFrame(),
        r2_overlap=pd.DataFrame(),
    ).set_index("hypothesis_id")
    assert registry.loc["H1_A02_vs_R1", "status"] == "NOT_SUPPORTED"
    assert registry.loc["H2_A02_vs_R2", "status"] == "NOT_SUPPORTED"
    assert "pooled-8" in registry.loc["H1_A02_vs_R1", "rationale"]


def test_threshold_frontier_preserves_tie_safe_highest_threshold_rule():
    runs = pd.DataFrame(
        [
            {
                "run_slot": "RUN_001",
                "condition_slot": "A02",
                "arm": "T",
                "phase": "A",
                "training_seed": 1,
                "attempt_dir": "unused",
            }
        ]
    )
    sweep = pd.DataFrame(
        [
            {"threshold": float("inf"), "TP": 0, "FP": 0, "TN": 3, "FN": 2, "tie_group_size": 0},
            {"threshold": 0.9, "TP": 1, "FP": 0, "TN": 3, "FN": 1, "tie_group_size": 1},
            {"threshold": 0.5, "TP": 2, "FP": 1, "TN": 2, "FN": 0, "tie_group_size": 2},
        ]
    )
    frontier = build_threshold_frontier(
        runs,
        fn_budgets=[0, 1, 2],
        sweep_loader=lambda _: sweep,
    )
    assert frontier.set_index("fn_budget").loc[0, "actual_TN"] == 2
    assert frontier.set_index("fn_budget").loc[1, "actual_TN"] == 3
    assert frontier.set_index("fn_budget").loc[1, "threshold"] == 0.9


def test_prediction_identity_requires_full_frozen_sample_and_label_match():
    normal_ids = [f"normal/{index:06d}.png" for index in range(100_000)]
    defect_ids = [f"defect/{index:06d}.png" for index in range(20_000)]
    expected = pd.DataFrame(
        {
            "sample_id": sorted(normal_ids + defect_ids),
        }
    )
    expected["y_true"] = expected["sample_id"].str.startswith("defect/").astype(int)
    predictions = expected.sample(frac=1.0, random_state=7).reset_index(drop=True)
    predictions["score"] = 0.5
    predictions["score_raw"] = 0.0

    audit = _validate_prediction_identity(
        predictions,
        expected,
        run_slot="RUN_001",
        split="val_op",
    )
    assert audit["row_count"] == 120_000
    assert audit["normal_count"] == 100_000
    assert audit["defect_count"] == 20_000
    assert audit["sample_ids_exact"] and audit["labels_exact"]

    normal_index = predictions.index[predictions["y_true"] == 0][0]
    defect_index = predictions.index[predictions["y_true"] == 1][0]
    predictions.loc[[normal_index, defect_index], "y_true"] = [1, 0]
    with pytest.raises(CanonicalInputError, match="labels differ"):
        _validate_prediction_identity(
            predictions,
            expected,
            run_slot="RUN_001",
            split="val_op",
        )


def test_plot_tables_expose_reporting_schema_and_optional_inputs_are_really_optional():
    epochs = pd.DataFrame(
        {
            "condition_slot": ["A02", "A02", "A02", "A02"],
            "run_slot": ["T1", "R1", "T1", "R1"],
            "arm": ["T", "R1", "T", "R1"],
            "epoch": [1, 1, 2, 2],
            "metrics/accuracy_top1": [0.7, 0.6, 0.8, 0.7],
            "val/loss": [0.4, 0.5, 0.3, 0.4],
            "train/loss": [0.3, 0.4, 0.2, 0.3],
        }
    )
    curves = _a02_training_curves(epochs)
    assert {"arm", "epoch", "top1", "val_loss"}.issubset(curves.columns)

    detail = pd.DataFrame(
        {
            "scope": ["operational"] * 4,
            "score_type": ["raw"] * 4,
            "condition_id": ["A02"] * 4,
            "condition_slot": ["A02"] * 4,
            "control": ["R1", "R1", "R2", "R2"],
            "label": ["normal", "defect", "normal", "defect"],
            "mean_shift": [-0.1, 0.2, -0.05, 0.1],
            "triad_id": ["T1", "T1", "T2", "T2"],
        }
    )
    tail = _tail_plot_summary(detail)
    assert {
        "condition_id",
        "delta_normal_tail_score",
        "delta_defect_tail_score",
    }.issubset(tail.columns)

    signature = inspect.signature(run_deep_analysis)
    assert signature.parameters["selection_root"].default is None
    assert signature.parameters["value_table"].default is None


def test_training_contract_reports_completed_model_partial_step_reconciliation(
    tmp_path,
):
    attempt = tmp_path / "attempt"
    audit_path = attempt / "02_logs" / "training_execution_audit.json"
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(
        json.dumps(
            {
                "audit_repair": {
                    "repair_schema": (
                        "stage1_gapvalue240."
                        "audit_partial_step_reconciliation.v1"
                    )
                },
                "discarded_partial_optimizer_steps_total": 211,
            }
        ),
        encoding="utf-8",
    )
    runs = pd.DataFrame(
        {
            "run_slot": ["RUN_001"],
            "budget": [600],
            "attempt_dir": [str(attempt)],
            "resume_count": [1],
        }
    )
    training = pd.DataFrame(
        {
            "run_slot": ["RUN_001"],
            "completed_epochs": [200],
            "expected_steps_per_epoch": [943],
            "optimizer_steps_total": [188_600],
        }
    )
    result = _training_contract_audit(runs, training).iloc[0]
    assert result["partial_step_reconciled"]
    assert result["discarded_partial_optimizer_steps"] == 211
    assert result["optimizer_steps_exact"]


def test_reliability_tables_use_one_to_one_validation_not_invalid_merge_mode(
    tmp_path,
):
    attempts = []
    for index in range(2):
        attempt = tmp_path / f"attempt_{index}"
        audit_path = attempt / "02_logs" / "training_execution_audit.json"
        audit_path.parent.mkdir(parents=True)
        audit_path.write_text(
            json.dumps(
                {
                    "resume_segments": [
                        {
                            "started_at": 100.0,
                            "ended_at": 160.0 + index,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        attempts.append(str(attempt))
    runs = pd.DataFrame(
        {
            "run_slot": ["RUN_001", "RUN_002"],
            "attempt_dir": attempts,
            "machine_id": ["M1", "M1"],
            "input_snapshot_id": ["S1", "S1"],
            "resume_count": [0, 1],
            "TN_at_FN95": [68_300, 68_310],
            "FN_at_TN68253": [90, 89],
            "gap_q68_q050": [0.1, 0.2],
        }
    )
    training = pd.DataFrame(
        {
            "run_slot": ["RUN_001", "RUN_002"],
            "optimizer_steps_total": [188_600, 188_600],
            "completed_epochs": [200, 200],
            "final_top1": [0.9, 0.91],
            "final_val_loss": [0.2, 0.19],
        }
    )
    result = _reliability_tables(
        runs,
        training,
        {"nonvalidated_evidence": [{"run_slot": "RUN_044"}]},
    )
    assert len(result["run_execution_reliability"]) == 2
    assert result["reliability_by_machine"]["run_count"].item() == 2
    assert result["historical_failed_attempts"]["run_slot"].tolist() == ["RUN_044"]
