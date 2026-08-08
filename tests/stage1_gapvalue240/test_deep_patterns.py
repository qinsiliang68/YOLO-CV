from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage1_gapvalue240.deep_patterns import (
    build_analysis_capabilities,
    build_condition_value_effects,
    build_pattern_narrative_sections,
    build_pattern_evidence_registry,
    build_paired_epoch_differences,
    build_raw_calibrated_operational_sensitivity,
    build_selection_value_summary,
    summarize_paired_epoch_differences,
    build_value_effect_associations,
)


def _write_selection(
    root: Path,
    *,
    run_slot: str,
    triad_id: str,
    arm: str,
    sample_ids: list[str],
) -> None:
    target = root / run_slot
    target.mkdir(parents=True)
    pd.DataFrame(
        {
            "run_slot": run_slot,
            "triad_id": triad_id,
            "condition_id": "A02_test",
            "arm": arm,
            "training_seed": 11,
            "selection_seed": 22,
            "rank": range(1, len(sample_ids) + 1),
            "sample_id": sample_ids,
            "y_true": [0] * len(sample_ids),
            "oof_fold": ["00"] * len(sample_ids),
            "dynamic_bucket": ["learnable_hard"] * len(sample_ids),
            "mean_p_defect": [0.5] * len(sample_ids),
            "correct_rate": [0.5] * len(sample_ids),
            "std_p_defect": [0.2] * len(sample_ids),
            "replay_role": ["normal_replay"] * len(sample_ids),
            "source_method": ["GapCritical-Strict"] * len(sample_ids),
        }
    ).to_csv(target / "selection_manifest.csv", index=False)


def test_selection_value_summary_and_condition_effect_association(tmp_path: Path) -> None:
    matrix_rows = []
    selection_root = tmp_path / "selections"
    value_rows = []
    for index in range(12):
        value_rows.append(
            {
                "sample_id": f"s{index}",
                "y_true": 0,
                "dynamic_bucket": "learnable_hard",
                "mean_p_defect": 0.2 + index / 100,
                "correct_rate": 0.8 - index / 100,
                "std_p_defect": 0.1 + index / 200,
                "gap_critical_score": index / 10,
                "gap_guard_score": np.nan,
            }
        )
    value_path = tmp_path / "values.csv"
    pd.DataFrame(value_rows).to_csv(value_path, index=False)

    arms = {
        "T": ["s9", "s10", "s11"],
        "R1": ["s0", "s1", "s2"],
        "R2": ["s7", "s8", "s9"],
    }
    for offset, (arm, sample_ids) in enumerate(arms.items(), start=1):
        run_slot = f"RUN_{offset:03d}"
        matrix_rows.append(
            {
                "run_slot": run_slot,
                "triad_id": "TRIAD_001",
                "condition_slot": "A02",
                "condition_id": "A02_test",
                "phase": "A",
                "method": "GapCritical-Strict",
                "budget": 3,
                "guard_ratio": 0.0,
                "arm": arm,
                "training_seed": 11,
            }
        )
        _write_selection(
            selection_root,
            run_slot=run_slot,
            triad_id="TRIAD_001",
            arm=arm,
            sample_ids=sample_ids,
        )

    matrix = pd.DataFrame(matrix_rows)
    summary = build_selection_value_summary(matrix, selection_root, value_path)
    assert set(summary["scope"]) == {"all", "normal"}
    treatment = summary.query("run_slot == 'RUN_001' and scope == 'normal'").iloc[0]
    assert treatment["mean_gap_critical_score"] == 1.0
    assert treatment["positive_gap_critical_rate"] == 1.0

    deltas = pd.DataFrame(
        [
            {
                "triad_id": "TRIAD_001",
                "condition_slot": "A02",
                "condition_id": "A02_test",
                "phase": "A",
                "control": control,
                "t_run_slot": "RUN_001",
                "control_run_slot": run_slot,
                "delta_FN": delta_fn,
                "delta_TN": delta_tn,
                "delta_gap_q68_q050": delta_gap,
                "machine_pair": "same_machine",
                "any_resumed": False,
            }
            for control, run_slot, delta_fn, delta_tn, delta_gap in [
                ("R1", "RUN_002", -2.0, 100.0, 0.01),
                ("R2", "RUN_003", -1.0, 50.0, 0.005),
            ]
        ]
    )
    effects = build_condition_value_effects(summary, deltas)
    r1 = effects.query("control == 'R1' and scope == 'normal'").iloc[0]
    assert r1["treatment_mean_gap_critical_score"] == 1.0
    assert r1["selection_delta_mean_gap_critical_score"] == 0.9
    assert r1["delta_TN"] == 100.0

    associations = build_value_effect_associations(effects)
    assert {
        "analysis_scope",
        "control",
        "predictor",
        "outcome",
        "n_conditions",
        "spearman_rho",
        "p_value",
        "interpretation",
    }.issubset(associations.columns)


def test_paired_epoch_differences_preserve_seed_and_machine_context() -> None:
    epochs = pd.DataFrame(
        [
            {
                "run_slot": run_slot,
                "epoch": epoch,
                "metrics/accuracy_top1": top1,
                "val/loss": val_loss,
                "train/loss": train_loss,
            }
            for run_slot, values in {
                "T": [(1, 0.8, 0.4, 0.3), (2, 0.9, 0.3, 0.2)],
                "R1": [(1, 0.7, 0.5, 0.4), (2, 0.8, 0.4, 0.3)],
            }.items()
            for epoch, top1, val_loss, train_loss in values
        ]
    )
    deltas = pd.DataFrame(
        [
            {
                "triad_id": "TRIAD_001",
                "condition_slot": "A02",
                "condition_id": "A02_test",
                "phase": "A",
                "training_seed": 11,
                "control": "R1",
                "t_run_slot": "T",
                "control_run_slot": "R1",
                "machine_pair": "cross_machine",
                "any_resumed": True,
            }
        ]
    )
    result = build_paired_epoch_differences(epochs, deltas)
    assert len(result) == 2
    assert np.allclose(result["delta_top1"], 0.1)
    assert np.allclose(result["delta_val_loss"], -0.1)
    assert set(result["machine_pair"]) == {"cross_machine"}
    assert result["any_resumed"].all()
    summary = summarize_paired_epoch_differences(result)
    assert len(summary) == 1
    np.testing.assert_allclose(summary.iloc[0]["final_delta_top1"], 0.1)
    np.testing.assert_allclose(summary.iloc[0]["last20_delta_top1_mean"], 0.1)


def test_raw_calibrated_operational_sensitivity_requires_identical_pairs() -> None:
    calibrated = pd.DataFrame(
        {
            "triad_id": ["T1", "T1"],
            "condition_slot": ["A02", "A02"],
            "control": ["R1", "R2"],
            "delta_TN": [10.0, -5.0],
            "delta_FN": [-1.0, 2.0],
            "delta_gap_q68_q050": [0.02, -0.03],
            "delta_tail_gap_q90_q05": [0.01, -0.02],
        }
    )
    raw = pd.DataFrame(
        {
            "triad_id": ["T1", "T1"],
            "condition_slot": ["A02", "A02"],
            "control": ["R1", "R2"],
            "delta_raw_TN_at_FN95": [10.0, -5.0],
            "delta_raw_FN_at_TN68253": [-1.0, 2.0],
            "delta_raw_gap_q68_q050": [0.03, -0.01],
            "delta_raw_tail_gap_q90_q05": [0.04, -0.01],
        }
    )
    result = build_raw_calibrated_operational_sensitivity(calibrated, raw)
    assert len(result) == 2
    assert result["integer_effects_equal"].all()
    assert result["delta_TN_raw_minus_calibrated"].eq(0).all()
    assert result["delta_FN_raw_minus_calibrated"].eq(0).all()
    assert result["gap_direction_equal"].all()
    assert result["tail_gap_direction_equal"].all()


def test_analysis_capabilities_register_replacements_and_unavailable_evidence(
    tmp_path: Path,
) -> None:
    overlap = tmp_path / "overlap_decisions.json"
    overlap.write_text(
        json.dumps(
            [
                {
                    "candidate": "TailGap-Strict",
                    "replaced": True,
                    "retained_as": "GapResidual-Strict",
                    "max_overlap": 0.9637,
                },
                {
                    "candidate": "Exclude178-GapStrict",
                    "replaced": True,
                    "retained_as": "GapCritical-Strict-TimeMatched",
                    "max_overlap": 1.0,
                },
            ]
        ),
        encoding="utf-8",
    )
    values = tmp_path / "values.csv"
    pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "grad_mag_score": [np.nan, np.nan],
            "grad_align_score": [np.nan, np.nan],
            "grad_mag_align_score": [np.nan, np.nan],
            "diverse_grad_align_score": [np.nan, np.nan],
            "grad_align_guard_score": [np.nan, np.nan],
        }
    ).to_csv(values, index=False)
    matrix = pd.DataFrame({"arm": ["T", "R1", "R2"]})

    capabilities = build_analysis_capabilities(
        matrix,
        overlap_decisions_path=overlap,
        value_table_path=values,
    ).set_index("capability_id")
    assert capabilities.loc["ranking_TailGap-Strict", "status"] == "NOT_TESTABLE"
    assert (
        capabilities.loc["ranking_TailGap-Strict", "replacement"]
        == "GapResidual-Strict"
    )
    assert capabilities.loc["gradient_evidence", "status"] == "NOT_TESTABLE"
    assert capabilities.loc["no_replay_baseline", "status"] == "NOT_TESTABLE"
    assert capabilities.loc["per_epoch_val_op", "status"] == "NOT_TESTABLE"
    assert capabilities.loc["epoch178_provenance", "status"] == "LIMITATION"


def test_pattern_registry_is_data_driven_and_exposes_four_evidence_layers() -> None:
    a02 = pd.DataFrame(
        [
            {
                "analysis_cohort": cohort,
                "control": control,
                "n": n,
                "mean_delta_FN": mean_fn,
                "FN_one_sided_95_upper": mean_fn + 2,
                "worst_delta_FN": mean_fn + 3,
                "safety_noninferior": safe,
                "mean_delta_TN": mean_tn,
                "TN_one_sided_95_lower": mean_tn - 10,
                "confirmed_TN_improvement": safe and mean_tn > 10,
                "machine_confounded": cohort != "discovery",
            }
            for cohort, n, mean_fn, mean_tn, safe in [
                ("discovery", 3, -1.0, 50.0, True),
                ("confirmation", 5, 5.0, -100.0, False),
                ("pooled", 8, 3.0, -40.0, False),
            ]
            for control in ("R1", "R2")
        ]
    )
    conditions = pd.DataFrame(
        [
            {
                "phase": "A",
                "condition_slot": slot,
                "control": control,
                "mean_delta_FN": mean_fn,
                "mean_delta_TN": mean_tn,
                "safety_noninferior": False,
                "confirmed_TN_improvement": False,
            }
            for slot, mean_fn, mean_tn in [
                ("A01", -5.0, 500.0),
                ("A02", 2.0, -200.0),
                ("A03", 8.0, -800.0),
                ("A13", 6.0, -600.0),
                ("A16", 4.0, -300.0),
                ("A17", 3.0, -250.0),
                ("A19", 7.0, -700.0),
            ]
            for control in ("R1", "R2")
        ]
        + [
            {
                "phase": "B",
                "condition_slot": "B05",
                "control": control,
                "mean_delta_FN": -2.0,
                "mean_delta_TN": 300.0,
                "safety_noninferior": False,
                "confirmed_TN_improvement": False,
            }
            for control in ("R1", "R2")
        ]
    )
    tail = pd.DataFrame(
        [
            {
                "condition_slot": slot,
                "control": control,
                "scope": "operational",
                "score_type": "raw",
                "label": label,
                "mean_shift": value,
            }
            for slot, values in {
                "A02": {"normal": -0.01, "defect": -0.001},
                "B05": {"normal": -0.01, "defect": 0.001},
            }.items()
            for control in ("R1", "R2")
            for label, value in values.items()
        ]
    )
    capabilities = pd.DataFrame(
        [
            {"capability_id": "gradient_evidence", "status": "NOT_TESTABLE"},
            {"capability_id": "no_replay_baseline", "status": "NOT_TESTABLE"},
            {
                "capability_id": "ranking_TailGap-Strict",
                "status": "NOT_TESTABLE",
            },
            {"capability_id": "blind_external_test", "status": "NOT_TESTABLE"},
        ]
    )
    registry = build_pattern_evidence_registry(
        a02_summaries=a02,
        condition_summaries=conditions,
        cross_method_comparisons=pd.DataFrame(),
        budget_comparisons=pd.DataFrame(),
        guard_comparisons=pd.DataFrame(),
        sensitivity_summaries=pd.DataFrame(),
        tail_detail=tail,
        capability_registry=capabilities,
        r2_overlap=pd.DataFrame(
            {"effective_unique_contrast_rate": [0.08, 0.09]}
        ),
    ).set_index("hypothesis_id")

    required = {
        "numerically_better",
        "contract_success",
        "mechanism_supported",
        "causal_claim_allowed",
        "status",
        "rationale",
    }
    assert required.issubset(registry.columns)
    assert registry.loc["H1_A02_vs_R1", "status"] == "NOT_SUPPORTED"
    assert registry.loc["H2_A02_vs_R2", "status"] == "NOT_SUPPORTED"
    assert registry.loc["H5_small_budget_advantage", "status"] == "INCONCLUSIVE"
    assert registry.loc["H7_defect_guard", "status"] == "INCONCLUSIVE"
    assert registry.loc["H9_no_replay", "status"] == "NOT_TESTABLE"
    assert not registry["causal_claim_allowed"].any()


def test_pattern_narrative_sections_are_data_driven() -> None:
    conditions = pd.DataFrame(
        [
            {
                "phase": phase,
                "condition_slot": slot,
                "control": control,
                "mean_delta_FN": mean_fn,
                "mean_delta_TN": mean_tn,
                "safety_noninferior": False,
                "confirmed_TN_improvement": False,
            }
            for phase, slot, mean_fn, mean_tn in (
                ("A", "A01", -5.0, 700.0),
                ("A", "A02", 2.0, -600.0),
                ("A", "A03", 9.0, -1000.0),
                ("A", "A05", -3.0, 250.0),
                ("B", "B05", -2.0, 800.0),
            )
            for control in ("R1", "R2")
        ]
    )
    cross_method = pd.DataFrame(
        [
            {
                "reference_condition": "A02",
                "comparator_condition": comparator,
                "control": control,
                "mean_diff_delta_FN": -1.0 if comparator == "A08" else 1.0,
                "mean_diff_delta_TN": 100.0 if comparator == "A08" else -50.0,
                "safety_noninferior": False,
                "confirmed_TN_improvement": False,
            }
            for comparator in ("A06", "A08", "A10", "A12")
            for control in ("R1", "R2")
        ]
    )
    associations = pd.DataFrame(
        [
            {
                "analysis_scope": "phase_a_normal_gapcritical",
                "control": control,
                "outcome": outcome,
                "spearman_rho": rho,
                "p_value": 0.2,
            }
            for control, outcome, rho in (
                ("R1", "delta_FN", -0.2),
                ("R1", "delta_TN", 0.3),
                ("R2", "delta_FN", -0.4),
                ("R2", "delta_TN", 0.5),
            )
        ]
    )
    tail = pd.DataFrame(
        [
            {
                "condition_slot": slot,
                "control": control,
                "scope": "operational",
                "score_type": "raw",
                "label": label,
                "mean_shift": value,
            }
            for slot, control, label, value in (
                ("A02", "R1", "normal", -0.01),
                ("A02", "R1", "defect", -0.001),
                ("A02", "R2", "normal", -0.02),
                ("A02", "R2", "defect", -0.002),
                ("B06", "R1", "normal", -0.003),
                ("B06", "R1", "defect", 0.001),
                ("B06", "R2", "normal", -0.004),
                ("B06", "R2", "defect", 0.002),
            )
        ]
    )
    raw_calibrated = pd.DataFrame(
        {
            "integer_effects_equal": [True, True],
            "gap_direction_equal": [True, False],
            "tail_gap_direction_equal": [True, True],
        }
    )
    training = pd.DataFrame(
        {
            "best_top1_epoch": [190, 200],
            "best_val_loss_epoch": [120, 140],
            "overfit_flag": [False, False],
            "oscillation_flag": [False, False],
            "best_final_top1_gap": [0.001, 0.002],
            "last_window_val_loss_slope": [0.001, 0.002],
        }
    )
    paired_epoch = pd.DataFrame(
        {
            "condition_slot": ["A02", "A02"],
            "final_delta_top1": [-0.001, 0.001],
            "final_delta_val_loss": [0.01, -0.002],
        }
    )
    contract = pd.DataFrame(
        {
            "resume_count": [0, 1],
            "partial_step_reconciled": [False, True],
        }
    )
    selection_values = pd.DataFrame(
        [
            {
                "phase": "A",
                "condition_slot": slot,
                "arm": "T",
                "scope": "normal",
                "mean_gap_critical_score": score,
                "correct_rate_mean": correct,
                "std_p_defect_mean": std,
            }
            for slot, score, correct, std in (
                ("A01", 0.67, 0.45, 0.29),
                ("A02", 0.47, 0.68, 0.22),
                ("A03", 0.35, 0.80, 0.17),
            )
        ]
    )
    selection_composition = pd.DataFrame(
        [
            {
                "phase": "A",
                "condition_slot": slot,
                "arm": "T",
                "y_true": 0,
                "dynamic_bucket": "learnable_hard",
                "count": count,
            }
            for slot, count in (("A01", 600), ("A02", 3000), ("A03", 6000))
        ]
    )

    sections = build_pattern_narrative_sections(
        condition_summaries=conditions,
        cross_method_comparisons=cross_method,
        selection_value_associations=associations,
        tail_detail=tail,
        raw_calibrated_sensitivity=raw_calibrated,
        training_summaries=training,
        paired_epoch_summary=paired_epoch,
        training_contract=contract,
        selection_value_summary=selection_values,
        selection_composition=selection_composition,
    )

    assert "0 个" in sections["全矩阵合同结果"]
    assert "B600" in sections["预算与传统方法"]
    assert "learnable_hard" in sections["预算与传统方法"]
    assert "A02" in sections["固定尾部机制"]
    assert "100.00%" in sections["Raw 与 Platt 敏感性"]
    assert "resume" in sections["200 epoch 与运行可靠性"]
