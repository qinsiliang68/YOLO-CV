from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.deep_analysis import CanonicalInputError
from stage1_gapvalue240.extreme_pipeline import (
    _build_leaveout_suite,
    _build_findings,
    build_controlled_tradeoff_grid,
    build_fixed_selection_seed_flips,
    build_prediction_tail_extreme_contrasts,
    build_raw_calibrated_operational_audit,
    build_tier_composition_audit,
    verify_manifested_inputs,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_verify_manifested_inputs_rejects_one_byte_change(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    target = root / "table.csv"
    target.write_text("a\n1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "relative_path": "table.csv",
                        "size_bytes": target.stat().st_size,
                        "sha256": _sha(target),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    verified = verify_manifested_inputs(root, manifest, ["table.csv"])
    assert verified["table.csv"] == target.resolve()

    target.write_text("a\n2\n", encoding="utf-8")
    with pytest.raises(CanonicalInputError, match="SHA-256 mismatch"):
        verify_manifested_inputs(root, manifest, ["table.csv"])


def test_tier_composition_audit_preserves_phase_seed_and_budget() -> None:
    tiers = pd.DataFrame(
        {
            "triad_id": ["T1", "T2", "T3"],
            "cohort_code": ["S", "H", "M"],
            "phase": ["A", "A", "B"],
            "training_seed": [1, 2, 1],
            "budget": [600, 3000, 3000],
            "all_same_machine": [True, True, False],
            "any_arm_resumed": [False, True, False],
        }
    )

    result = build_tier_composition_audit(tiers)

    assert set(result["dimension"]) == {
        "phase",
        "training_seed",
        "budget",
        "all_same_machine",
        "any_arm_resumed",
    }
    phase_a_s = result.loc[
        (result["dimension"] == "phase")
        & (result["value"] == "A")
        & (result["cohort_code"] == "S"),
        "triad_count",
    ].iloc[0]
    assert phase_a_s == 1


def test_fixed_selection_seed_flips_only_keeps_crossing_sets() -> None:
    rows = pd.DataFrame(
        {
            "triad_id": ["T1", "T2", "T3", "T4"],
            "sample_set_digest": ["x", "x", "x", "y"],
            "cohort_code": ["S", "M", "H", "M"],
            "condition_slot": ["A01", "A01", "A01", "A02"],
            "phase": ["A"] * 4,
            "training_seed": [1, 2, 3, 1],
            "selected_count": [600, 600, 600, 3000],
        }
    )

    result = build_fixed_selection_seed_flips(rows)

    assert result["sample_set_digest"].nunique() == 1
    assert set(result["triad_id"]) == {"T1", "T2", "T3"}
    assert result["spans_exceptional_and_harmful"].all()


def test_findings_match_report_schema_and_do_not_overclaim_late_loss() -> None:
    tiers = pd.DataFrame(
        {
            "triad_id": ["T1", "T2"],
            "cohort_code": ["S", "H"],
            "discovery_or_confirmation": ["discovery", "confirmation"],
        }
    )
    selection_outcomes = pd.DataFrame(
        {
            "sample_set_digest": ["x"],
            "spans_exceptional_and_harmful": [True],
        }
    )
    training_contrasts = pd.DataFrame(
        {
            "analysis_scope": ["all"] * 4,
            "control": ["R1", "R2", "R1", "R2"],
            "feature": [
                "train_loss_extra_drop_epoch121_to_200",
                "train_loss_extra_drop_epoch121_to_200",
                "train_loss_robust_drop_121_130_to_191_200",
                "train_loss_robust_drop_121_130_to_191_200",
            ],
            "mean_difference_S_minus_H": [-0.2, -0.1, -0.1, -0.1],
            "bootstrap_95_low": [-0.4, -0.3, -0.3, -0.3],
            "bootstrap_95_high": [-0.1, -0.05, 0.02, 0.03],
        }
    )
    outcome_contrasts = pd.DataFrame(
        {
            "analysis_scope": ["all", "all"],
            "control": ["R1", "R2"],
            "feature": ["delta_threshold", "delta_threshold"],
            "mean_difference_S_minus_H": [0.02, 0.03],
            "bootstrap_95_low": [0.01, 0.01],
            "bootstrap_95_high": [0.04, 0.05],
        }
    )

    findings = _build_findings(
        tiers, selection_outcomes, training_contrasts, outcome_contrasts
    )

    assert all({"title", "evidence", "boundary"}.issubset(row) for row in findings)
    late_loss = next(row for row in findings if row["finding_id"] == "F03")
    assert late_loss["status"] == "PARTIAL_PATTERN"


def test_prediction_tail_contrasts_keep_label_scope_score_and_control_separate() -> None:
    tiers = pd.DataFrame(
        {
            "triad_id": ["T1", "T2"],
            "cohort_code": ["S", "H"],
            "budget": [600, 600],
        }
    )
    rows = []
    for triad_id, shift in (("T1", -0.2), ("T2", 0.3)):
        for control in ("R1", "R2"):
            rows.append(
                {
                    "triad_id": triad_id,
                    "control": control,
                    "label": "normal",
                    "scope": "operational",
                    "score_type": "raw",
                    "phase": "A",
                    "training_seed": 1,
                    "machine_pair": "same_machine",
                    "any_resumed": False,
                    "mean_shift": shift,
                    "median_shift": shift,
                    "beneficial_rate": 0.8 if shift < 0 else 0.2,
                    "harmed_rate": 0.2 if shift < 0 else 0.8,
                }
            )

    result = build_prediction_tail_extreme_contrasts(
        pd.DataFrame(rows), tiers, bootstrap_samples=100
    )

    assert set(result["control"]) == {"R1", "R2"}
    assert set(result["label"]) == {"normal"}
    assert set(result["tail_scope"]) == {"operational"}
    assert set(result["score_type"]) == {"raw"}
    mean_shift = result.loc[
        (result["analysis_scope"] == "all") & (result["feature"] == "mean_shift")
    ]
    assert set(mean_shift["mean_difference_S_minus_H"]) == {-0.5}


def test_leaveout_suite_reports_phase_a_same_machine_separately() -> None:
    rows = []
    for phase, machine_pair in (("A", "same_machine"), ("B", "cross_machine")):
        for seed in (1, 2):
            for cohort, value in (("S", 1.0), ("H", 3.0)):
                for control in ("R1", "R2"):
                    rows.append(
                        {
                            "triad_id": f"{phase}_{seed}_{cohort}_{control}",
                            "control": control,
                            "cohort_code": cohort,
                            "phase": phase,
                            "machine_pair": machine_pair,
                            "training_seed": seed,
                            "feature": value,
                        }
                    )

    result = _build_leaveout_suite(
        pd.DataFrame(rows), ("feature",), ("training_seed",)
    )

    assert set(result["analysis_scope"]) == {"all", "phase_A_same_machine"}


def test_controlled_tradeoff_grid_uses_worse_of_both_controls() -> None:
    tiers = pd.DataFrame(
        {
            "triad_id": ["strict", "fn2", "one_control_only"],
            "delta_TN_R1": [500, 400, 500],
            "delta_TN_R2": [450, 350, -1],
            "delta_FN_R1": [0, 2, 0],
            "delta_FN_R2": [-1, 1, 0],
        }
    )

    detail, summary = build_controlled_tradeoff_grid(
        tiers, fn_margins=(0, 2), tn_minimums=(1, 300)
    )

    strict = summary.query("fn_margin == 0 and tn_minimum == 300").iloc[0]
    relaxed = summary.query("fn_margin == 2 and tn_minimum == 300").iloc[0]
    assert strict["qualifying_triads"] == 1
    assert relaxed["qualifying_triads"] == 2
    assert not detail.loc[
        (detail["triad_id"] == "one_control_only")
        & (detail["fn_margin"] == 2)
        & (detail["tn_minimum"] == 1),
        "qualifies_both_controls",
    ].iloc[0]


def test_raw_calibrated_operational_audit_detects_exact_frontier_agreement() -> None:
    pairs = pd.DataFrame(
        {
            "triad_id": ["T1", "T1"],
            "control": ["R1", "R2"],
            "t_run_slot": ["T", "T"],
            "control_run_slot": ["C1", "C2"],
            "delta_TN": [10, 20],
            "delta_FN": [-1, 0],
        }
    )
    runs = pd.DataFrame(
        {
            "run_slot": ["T", "C1", "C2"],
            "raw_TN_at_FN95": [100, 90, 80],
            "raw_FN_at_TN68253": [5, 6, 5],
        }
    )

    pair_audit, triad_audit = build_raw_calibrated_operational_audit(pairs, runs)

    assert pair_audit["raw_matches_calibrated_effect"].all()
    assert triad_audit.loc[0, "strict_two_sided_benefit_raw"]
    assert triad_audit.loc[0, "strict_two_sided_benefit_calibrated"]
    assert triad_audit.loc[0, "raw_calibrated_strict_class_agree"]
