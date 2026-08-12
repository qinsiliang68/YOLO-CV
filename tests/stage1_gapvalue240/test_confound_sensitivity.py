from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.confound_sensitivity import (
    ConfoundSensitivityError,
    analyze_confound_sensitivity,
    publish_confound_sensitivity,
)


def _synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes: list[dict[str, object]] = []
    resources: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    canonical: list[dict[str, object]] = []
    for index in range(8):
        triad_id = f"TRIAD_{index + 1:03d}"
        seed = 11 + index % 2 if index < 6 else 101 + index
        same_machine = index < 4
        same_snapshot = index not in {2, 6}
        any_resume = index in {1, 5}
        budget = (600, 3000, 6000, 3000)[index % 4]
        discovery = "discovery" if index < 6 else "confirmation"
        condition_id = f"A{index // 2 + 1:02d}_Synthetic"
        if index in {0, 4}:
            deltas = {"R1": (500.0, -2.0), "R2": (350.0, 0.0)}
        elif index in {1, 5}:
            deltas = {"R1": (400.0, 2.0), "R2": (300.0, 1.0)}
        elif index in {2, 6}:
            deltas = {"R1": (-100.0, 3.0), "R2": (-50.0, 1.0)}
        else:
            deltas = {"R1": (100.0, -1.0), "R2": (-20.0, 2.0)}

        g_tn = min(value[0] for value in deltas.values())
        g_fn = max(value[1] for value in deltas.values())
        harm_tn = max(value[0] for value in deltas.values())
        harm_fn = min(value[1] for value in deltas.values())
        dual_improvement = g_tn > 0 and g_fn <= 0
        high_value = g_tn >= 300 and g_fn <= 2
        dual_harm = harm_tn < 0 and harm_fn > 0
        exclusive = (
            "DUAL_IMPROVEMENT"
            if dual_improvement
            else "HIGH_VALUE"
            if high_value
            else "DUAL_HARM"
            if dual_harm
            else "MIXED_OR_REVERSAL"
        )
        outcomes.append(
            {
                "triad_id": triad_id,
                "phase": "A" if discovery == "discovery" else "C",
                "condition_id": condition_id,
                "condition_slot": condition_id.split("_")[0],
                "method": "Synthetic",
                "budget": budget,
                "guard_ratio": 0.0,
                "training_seed": seed,
                "discovery_or_confirmation": discovery,
                "delta_TN_R1": deltas["R1"][0],
                "delta_FN_R1": deltas["R1"][1],
                "delta_TN_R2": deltas["R2"][0],
                "delta_FN_R2": deltas["R2"][1],
                "G_TN": g_tn,
                "G_FN": g_fn,
                "HARM_TN": harm_tn,
                "HARM_FN": harm_fn,
                "dual_improvement": dual_improvement,
                "high_value": high_value,
                "dual_harm": dual_harm,
                "exclusive_cohort": exclusive,
            }
        )

        t_machine = "M1" if same_machine else "M2"
        r1_machine = t_machine if same_machine else "M3"
        r2_machine = t_machine if same_machine else "M4"
        t_snapshot = "S1"
        r1_snapshot = t_snapshot if same_snapshot else "S2"
        r2_snapshot = t_snapshot if same_snapshot else "S2"
        resources.append(
            {
                "triad_id": triad_id,
                "phase": "A" if discovery == "discovery" else "C",
                "condition_id": condition_id,
                "method": "Synthetic",
                "budget": budget,
                "training_seed": seed,
                "t_machine_id": t_machine,
                "r1_machine_id": r1_machine,
                "r2_machine_id": r2_machine,
                "all_arms_same_machine": same_machine,
                "t_r1_same_machine": same_machine,
                "t_r2_same_machine": same_machine,
                "all_arms_same_snapshot": same_snapshot,
                "resumed_arm_count": int(any_resume),
                "resume_event_count": int(any_resume),
            }
        )

        raw_dual = index == 0
        for control in ("R1", "R2"):
            for score_type in ("raw", "calibrated"):
                raw_rows.append(
                    {
                        "triad_id": triad_id,
                        "control_arm": control,
                        "score_type": score_type,
                        "safe_frontier_dominant": bool(
                            raw_dual if score_type == "raw" else False
                        ),
                    }
                )

        t_tn, t_fn = 68_500 + index, 90 + index
        for arm, machine, snapshot, resume_count in (
            ("T", t_machine, t_snapshot, int(any_resume)),
            ("R1", r1_machine, r1_snapshot, 0),
            ("R2", r2_machine, r2_snapshot, 0),
        ):
            delta_tn, delta_fn = (0.0, 0.0) if arm == "T" else deltas[arm]
            canonical.append(
                {
                    "run_slot": f"RUN_{len(canonical) + 1:03d}",
                    "triad_id": triad_id,
                    "arm": arm,
                    "phase": "A" if discovery == "discovery" else "C",
                    "condition_id": condition_id,
                    "method": "Synthetic",
                    "budget": budget,
                    "training_seed": seed,
                    "discovery_or_confirmation": discovery,
                    "machine_id": machine,
                    "input_snapshot_id": snapshot,
                    "resume_count": resume_count,
                    "TN_at_FN95": t_tn if arm == "T" else t_tn - delta_tn,
                    "FN_at_TN68253": t_fn if arm == "T" else t_fn - delta_fn,
                }
            )
    return tuple(map(pd.DataFrame, (outcomes, resources, raw_rows, canonical)))


def test_confound_sensitivity_preserves_triad_unit_and_raw_dual_safe_definition() -> None:
    result = analyze_confound_sensitivity(
        *_synthetic_inputs(),
        expected_triads=8,
        min_group_n=2,
        bootstrap_resamples=200,
        permutation_resamples=500,
        random_state=7,
    )

    assert len(result.triads) == 8
    assert len(result.within_condition_seed) == 16
    assert result.triads["raw_dual_safe"].sum() == 1
    machine = result.strata_summary.loc[
        result.strata_summary["stratum_family"].eq("machine")
    ]
    assert set(machine["stratum_level"]) == {"SAME_MACHINE", "CROSS_MACHINE"}
    assert machine["n_triads"].sum() == 8
    assert {
        "exclusive_dual_improvement_rate",
        "exclusive_high_value_rate",
        "exclusive_dual_harm_rate",
        "exclusive_mixed_or_reversal_rate",
        "G_TN_mean",
        "G_FN_mean",
        "raw_dual_safe_rate",
    }.issubset(result.strata_summary.columns)
    assert set(result.within_condition_seed["control_arm"]) == {"R1", "R2"}
    assert result.summary["machine_adjustment_interpretation"] == (
        "DESCRIPTIVE_SENSITIVITY_NOT_CAUSAL_CORRECTION"
    )


def test_confound_sensitivity_runs_seed_stratified_tests_only_when_estimable() -> None:
    result = analyze_confound_sensitivity(
        *_synthetic_inputs(),
        expected_triads=8,
        min_group_n=2,
        bootstrap_resamples=200,
        permutation_resamples=500,
        random_state=13,
    )
    contrasts = result.binary_contrasts
    machine_gtn = contrasts.loc[
        contrasts["contrast_id"].eq("machine:SAME_MACHINE-vs-CROSS_MACHINE")
        & contrasts["metric"].eq("G_TN")
    ].iloc[0]
    assert machine_gtn["analysis_status"] == "ESTIMATED"
    assert machine_gtn["bootstrap_cluster"] == "training_seed"
    assert machine_gtn["permutation_stratification"] == "training_seed"
    assert machine_gtn["permutation_swappable_seed_count"] == 2
    stage_gtn = contrasts.loc[
        contrasts["contrast_id"].eq(
            "discovery_confirmation:DISCOVERY-vs-CONFIRMATION"
        )
        & contrasts["metric"].eq("G_TN")
    ].iloc[0]
    assert stage_gtn["analysis_status"] == "NO_WITHIN_SEED_EXCHANGEABILITY"
    assert pd.isna(stage_gtn["permutation_p_value"])
    assert contrasts["q_value_bh"].notna().any()


def test_confound_sensitivity_rejects_resource_or_canonical_drift() -> None:
    outcomes, resources, raw, canonical = _synthetic_inputs()
    resources.loc[0, "all_arms_same_machine"] = False
    with pytest.raises(ConfoundSensitivityError, match="same-machine drift"):
        analyze_confound_sensitivity(
            outcomes,
            resources,
            raw,
            canonical,
            expected_triads=8,
            min_group_n=2,
            bootstrap_resamples=20,
            permutation_resamples=20,
        )


def test_confound_sensitivity_publisher_is_atomic_and_does_not_touch_state(
    tmp_path: Path,
) -> None:
    result = analyze_confound_sensitivity(
        *_synthetic_inputs(),
        expected_triads=8,
        min_group_n=2,
        bootstrap_resamples=20,
        permutation_resamples=20,
    )
    output = tmp_path / "analysis.inprogress"
    output.mkdir()
    state = output / "ANALYSIS_STATE.json"
    state.write_text('{"completed_stage":"UNCHANGED"}', encoding="utf-8")

    published = publish_confound_sensitivity(result, output)

    assert published["output_files"] == 7
    assert json.loads(state.read_text(encoding="utf-8")) == {
        "completed_stage": "UNCHANGED"
    }
    tables = output / "tables"
    expected = {
        "confound_sensitivity_triad_strata.csv",
        "confound_sensitivity_strata_summary.csv",
        "confound_sensitivity_binary_contrasts.csv",
        "confound_sensitivity_within_condition_seed.csv",
        "confound_sensitivity_within_pair_summary.csv",
        "confound_sensitivity_summary.json",
        "confound_sensitivity_output_manifest.csv",
    }
    assert {path.name for path in tables.glob("confound_sensitivity_*")} == expected
    assert not list(tables.glob("*.tmp"))
    manifest = pd.read_csv(tables / "confound_sensitivity_output_manifest.csv")
    assert len(manifest) == 6
    with pytest.raises(FileExistsError):
        publish_confound_sensitivity(result, output)
