from __future__ import annotations

import pandas as pd
import pytest

from stage1_gapvalue240.deep_statistics import (
    build_a02_summaries,
    build_budget_comparisons,
    build_condition_summaries,
    build_cross_method_comparisons,
    build_guard_comparisons,
    build_sensitivity_summaries,
    build_triad_deltas,
)


METRICS = {
    "TN_at_FN95": 68000.0,
    "FN_at_TN68253": 100.0,
    "gap_q68_q050": 0.20,
    "tail_gap_q90_q05": 0.10,
    "normal_q68": 0.30,
    "normal_q90": 0.50,
    "defect_q50": 0.50,
    "defect_q05": 0.60,
}


def _triad(
    triad_id: str,
    condition_slot: str,
    seed: int,
    *,
    phase: str = "A",
    cohort: str = "discovery",
    method: str = "GapCritical-Strict",
    budget: int = 3000,
    guard_ratio: float = 0.0,
    treatment_shift: float = 1.0,
    machine_ids: tuple[str, str, str] = ("m1", "m1", "m1"),
    resume_counts: tuple[int, int, int] = (0, 0, 0),
) -> list[dict]:
    condition_id = f"{phase}_{condition_slot}_{method}_B{budget}"
    rows = []
    for index, arm in enumerate(("T", "R1", "R2")):
        row = {
            "run_slot": f"{triad_id}_{arm}",
            "triad_id": triad_id,
            "phase": phase,
            "condition_slot": condition_slot,
            "condition_id": condition_id,
            "method": method,
            "budget": budget,
            "guard_ratio": guard_ratio,
            "arm": arm,
            "training_seed": seed,
            "selection_seed": seed * 10 + index,
            "discovery_or_confirmation": cohort,
            "machine_id": machine_ids[index],
            "resume_count": resume_counts[index],
            "input_snapshot_id": "snapshot-a",
            **METRICS,
        }
        if arm == "T":
            row.update(
                {
                    "TN_at_FN95": METRICS["TN_at_FN95"] + 10 * treatment_shift,
                    "FN_at_TN68253": METRICS["FN_at_TN68253"] - treatment_shift,
                    "gap_q68_q050": METRICS["gap_q68_q050"] + 0.02 * treatment_shift,
                    "tail_gap_q90_q05": METRICS["tail_gap_q90_q05"] + 0.01 * treatment_shift,
                    "normal_q68": METRICS["normal_q68"] - 0.01 * treatment_shift,
                    "normal_q90": METRICS["normal_q90"] - 0.01 * treatment_shift,
                    "defect_q50": METRICS["defect_q50"] + 0.01 * treatment_shift,
                    "defect_q05": METRICS["defect_q05"] + 0.01 * treatment_shift,
                }
            )
        rows.append(row)
    return rows


def test_build_triad_deltas_keeps_controls_metrics_and_execution_flags():
    frame = pd.DataFrame(
        _triad(
            "TRIAD_001",
            "A02",
            11,
            machine_ids=("m1", "m1", "m2"),
            resume_counts=(1, 0, 0),
        )
    )

    deltas = build_triad_deltas(frame)

    assert deltas["control"].tolist() == ["R1", "R2"]
    assert deltas["delta_TN"].tolist() == [10.0, 10.0]
    assert deltas["delta_FN"].tolist() == [-1.0, -1.0]
    assert deltas["delta_gap_q68_q050"].tolist() == pytest.approx([0.02, 0.02])
    assert deltas["delta_normal_q68"].tolist() == pytest.approx([-0.01, -0.01])
    assert deltas["delta_defect_q05"].tolist() == pytest.approx([0.01, 0.01])
    assert deltas["machine_pair"].tolist() == ["same_machine", "cross_machine"]
    assert deltas["same_machine"].tolist() == [True, False]
    assert deltas["t_resume_count"].tolist() == [1, 1]
    assert deltas["control_resume_count"].tolist() == [0, 0]
    assert deltas["any_resumed"].tolist() == [True, True]


def test_build_triad_deltas_rejects_duplicate_or_incomplete_triads():
    complete = pd.DataFrame(_triad("TRIAD_001", "A02", 11))
    with pytest.raises(ValueError, match="duplicate"):
        build_triad_deltas(pd.concat([complete, complete.iloc[[0]]], ignore_index=True))
    with pytest.raises(ValueError, match="exactly T, R1, and R2"):
        build_triad_deltas(complete[complete.arm != "R2"])


def test_condition_and_a02_summaries_keep_discovery_confirmation_and_pooled_distinct():
    rows: list[dict] = []
    for index in range(3):
        rows += _triad(f"D{index}", "A02", 100 + index, treatment_shift=index + 1)
    for index in range(5):
        rows += _triad(
            f"C{index}",
            "A02",
            200 + index,
            phase="C",
            cohort="confirmation",
            treatment_shift=index + 1,
            machine_ids=("t-machine", "control-machine", "control-machine"),
        )
    deltas = build_triad_deltas(pd.DataFrame(rows))

    conditions = build_condition_summaries(deltas)
    assert set(conditions["control"]) == {"R1", "R2"}
    assert set(conditions["condition_id"]) == {
        "A_A02_GapCritical-Strict_B3000",
        "C_A02_GapCritical-Strict_B3000",
    }
    a02 = build_a02_summaries(deltas)

    assert set(a02["analysis_cohort"]) == {"discovery", "confirmation", "pooled"}
    assert set(a02[a02.analysis_cohort == "discovery"]["n"]) == {3}
    assert set(a02[a02.analysis_cohort == "confirmation"]["n"]) == {5}
    assert set(a02[a02.analysis_cohort == "pooled"]["n"]) == {8}
    assert a02[a02.analysis_cohort == "confirmation"]["machine_confounded"].all()
    assert not a02[a02.analysis_cohort == "discovery"]["machine_confounded"].any()


def test_sensitivity_summaries_separate_same_machine_and_no_resume():
    rows = _triad("SAME", "A02", 1)
    rows += _triad(
        "CROSS",
        "A02",
        2,
        machine_ids=("m1", "m2", "m2"),
        resume_counts=(1, 0, 0),
    )
    deltas = build_triad_deltas(pd.DataFrame(rows))

    summaries = build_sensitivity_summaries(deltas)

    r1 = summaries[summaries.control == "R1"].set_index("analysis_set")
    assert r1.loc["all", "n"] == 2
    assert r1.loc["same_machine", "n"] == 1
    assert r1.loc["cross_machine", "n"] == 1
    assert r1.loc["no_resume", "n"] == 1


def test_preregistered_method_budget_and_guard_comparisons_are_seed_paired():
    rows: list[dict] = []
    for seed in (1, 2, 3):
        rows += _triad(f"A01_{seed}", "A01", seed, budget=600, treatment_shift=1)
        rows += _triad(f"A02_{seed}", "A02", seed, budget=3000, treatment_shift=3)
        rows += _triad(
            f"A05_{seed}",
            "A05",
            seed,
            method="Confidence-Clean",
            budget=600,
            treatment_shift=0,
        )
        rows += _triad(
            f"B03_{seed}",
            "B03",
            seed,
            phase="B",
            method="GapGuard-Raw",
            guard_ratio=0.05,
            treatment_shift=1,
        )
        rows += _triad(
            f"B04_{seed}",
            "B04",
            seed,
            phase="B",
            method="GapGuard-Raw",
            guard_ratio=0.10,
            treatment_shift=2,
        )
    deltas = build_triad_deltas(pd.DataFrame(rows))

    method = build_cross_method_comparisons(deltas)
    comparison = method[
        (method.reference_condition == "A01")
        & (method.comparator_condition == "A05")
        & (method.control == "R1")
    ]
    assert comparison["n"].item() == 3
    assert comparison["mean_diff_delta_TN"].item() == 10.0

    budget = build_budget_comparisons(deltas)
    comparison = budget[
        (budget.reference_condition == "A02")
        & (budget.comparator_condition == "A01")
        & (budget.control == "R1")
    ]
    assert comparison["n"].item() == 3
    assert comparison["mean_diff_delta_TN"].item() == 20.0

    guard = build_guard_comparisons(deltas)
    comparison = guard[
        (guard.reference_condition == "B04")
        & (guard.comparator_condition == "B03")
        & (guard.control == "R2")
    ]
    assert comparison["n"].item() == 3
    assert comparison["mean_diff_delta_FN"].item() == -1.0
