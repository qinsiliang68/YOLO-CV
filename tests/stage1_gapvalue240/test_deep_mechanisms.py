from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stage1_gapvalue240.deep_mechanisms import (
    audit_selections,
    build_control_consensus,
    calibration_diagnostics,
    define_reference_tails,
    pair_prediction_tail_shifts,
    summarize_sample_shift_consistency,
    summarize_selected_score_signs,
)
from stage1_gapvalue240.errors import ValidationError


def _write_selection(
    root: Path,
    run_slot: str,
    *,
    arm: str,
    triad_id: str,
    condition_id: str,
    rows: list[tuple[str, int, str, str]],
    audit: dict | None = None,
) -> None:
    run_root = root / run_slot
    run_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "run_slot": run_slot,
                "triad_id": triad_id,
                "condition_id": condition_id,
                "arm": arm,
                "training_seed": 7,
                "selection_seed": 11,
                "rank": rank,
                "sample_id": sample_id,
                "y_true": y_true,
                "oof_fold": "00",
                "dynamic_bucket": bucket,
                "mean_p_defect": 0.3,
                "correct_rate": 0.8,
                "std_p_defect": 0.1,
                "replay_role": role,
                "source_method": "fixture",
            }
            for rank, (sample_id, y_true, role, bucket) in enumerate(rows, 1)
        ]
    ).to_csv(run_root / "selection_manifest.csv", index=False)
    payload = {
        "arm": arm,
        "rows": len(rows),
        "unique_samples": len({row[0] for row in rows}),
        **(audit or {}),
    }
    (run_root / "selection_audit.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _matrix(rows: list[tuple[str, str, str, str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_slot": slot,
                "triad_id": triad,
                "condition_slot": condition_slot,
                "condition_id": condition_id,
                "phase": phase,
                "method": condition_id,
                "budget": budget,
                "guard_ratio": 0.0 if phase == "A" else 0.5,
                "arm": arm,
                "training_seed": 7,
                "selection_seed": 11,
                "discovery_or_confirmation": "discovery",
            }
            for slot, triad, condition_slot, phase, arm, budget in rows
            for condition_id in [f"{condition_slot}_fixture"]
        ]
    )


def test_selection_audit_preserves_phase_a_and_phase_b_overlap_semantics(tmp_path):
    selection_root = tmp_path / "selections"
    matrix = _matrix(
        [
            ("RUN_001", "TRIAD_001", "A01", "A", "T", 2),
            ("RUN_002", "TRIAD_001", "A01", "A", "R1", 2),
            ("RUN_003", "TRIAD_001", "A01", "A", "R2", 2),
            ("RUN_004", "TRIAD_002", "B01", "B", "T", 4),
            ("RUN_005", "TRIAD_002", "B01", "B", "R1", 4),
            ("RUN_006", "TRIAD_002", "B01", "B", "R2", 4),
        ]
    )
    normal_t = [("n1", 0, "normal_replay", "learnable_hard"),
                ("n2", 0, "normal_replay", "learnable_hard")]
    _write_selection(selection_root, "RUN_001", arm="T", triad_id="TRIAD_001",
                     condition_id="A01_fixture", rows=normal_t)
    _write_selection(
        selection_root,
        "RUN_002",
        arm="R1",
        triad_id="TRIAD_001",
        condition_id="A01_fixture",
        rows=[("n3", 0, "normal_replay", "easy_clean"),
              ("n4", 0, "normal_replay", "ordinary")],
    )
    _write_selection(
        selection_root,
        "RUN_003",
        arm="R2",
        triad_id="TRIAD_001",
        condition_id="A01_fixture",
        rows=[("n1", 0, "normal_replay", "learnable_hard"),
              ("n5", 0, "normal_replay", "ordinary")],
        audit={
            "overlap_count": 1,
            "overlap_rate": 0.5,
            "jaccard": 1 / 3,
            "forced_overlap_count": 1,
            "effective_unique_contrast": 0.5,
            "max_abs_smd": 0.09,
            "fallback_counts": {"L0": 1, "L4_FORCED_OVERLAP": 1},
        },
    )

    phase_b_normal = [
        ("n1", 0, "normal_replay", "learnable_hard"),
        ("n2", 0, "normal_replay", "learnable_hard"),
    ]
    _write_selection(
        selection_root,
        "RUN_004",
        arm="T",
        triad_id="TRIAD_002",
        condition_id="B01_fixture",
        rows=phase_b_normal
        + [("d1", 1, "defect_guard", "learnable_hard"),
           ("d2", 1, "defect_guard", "learnable_hard")],
    )
    _write_selection(
        selection_root,
        "RUN_005",
        arm="R1",
        triad_id="TRIAD_002",
        condition_id="B01_fixture",
        rows=phase_b_normal
        + [("d3", 1, "defect_guard", "ordinary"),
           ("d4", 1, "defect_guard", "ordinary")],
    )
    _write_selection(
        selection_root,
        "RUN_006",
        arm="R2",
        triad_id="TRIAD_002",
        condition_id="B01_fixture",
        rows=phase_b_normal
        + [("d1", 1, "defect_guard", "learnable_hard"),
           ("d5", 1, "defect_guard", "ordinary")],
        audit={
            "overlap_count": 3,
            "overlap_rate": 0.75,
            "jaccard": 0.6,
            "effective_unique_contrast": 0.25,
            "forced_overlap_count": 1,
            "max_abs_smd": 0.05,
            "fallback_counts": {"L0": 1, "L4_FORCED_OVERLAP": 1},
        },
    )

    tables = audit_selections(matrix, selection_root)
    overlap = tables["triad_overlap"]
    phase_a_r1 = overlap.query(
        "triad_id == 'TRIAD_001' and control == 'R1' and scope == 'all'"
    ).iloc[0]
    assert phase_a_r1.overlap_count == 0

    phase_b_normal_r1 = overlap.query(
        "triad_id == 'TRIAD_002' and control == 'R1' and scope == 'normal'"
    ).iloc[0]
    phase_b_defect_r1 = overlap.query(
        "triad_id == 'TRIAD_002' and control == 'R1' and scope == 'defect'"
    ).iloc[0]
    assert phase_b_normal_r1.overlap_rate == 1.0
    assert phase_b_defect_r1.overlap_count == 0

    r2 = tables["run_audit"].query("run_slot == 'RUN_003'").iloc[0]
    assert r2.effective_unique_contrast == 0.5
    assert r2.forced_overlap_count == 1
    assert tables["composition"]["count"].sum() == 18


def test_selection_audit_rejects_duplicate_ids_and_wrong_budget(tmp_path):
    root = tmp_path / "selections"
    matrix = _matrix([("RUN_001", "TRIAD_001", "A01", "A", "T", 2)])
    _write_selection(
        root,
        "RUN_001",
        arm="T",
        triad_id="TRIAD_001",
        condition_id="A01_fixture",
        rows=[
            ("n1", 0, "normal_replay", "ordinary"),
            ("n1", 0, "normal_replay", "ordinary"),
        ],
    )
    with pytest.raises(ValidationError, match="duplicate"):
        audit_selections(matrix, root)


def test_reference_tails_and_pair_shifts_use_control_consensus_direction():
    consensus = pd.DataFrame(
        {
            "sample_id": ["n1", "n2", "n3", "d1", "d2", "d3"],
            "y_true": [0, 0, 0, 1, 1, 1],
            "control_median_score_raw": [0.9, 0.5, 0.1, 0.05, 0.4, 0.8],
        }
    )
    reference = define_reference_tails(
        consensus,
        tn_target=2,
        fn_limit=1,
        normal_tail_fraction=1 / 3,
        defect_tail_fraction=1 / 3,
    )
    assert reference.set_index("sample_id").loc["n1", "operational_tail"]
    assert reference.set_index("sample_id").loc["d1", "operational_tail"]

    control = pd.DataFrame(
        {
            "sample_id": consensus.sample_id,
            "y_true": consensus.y_true,
            "score_raw": [0.9, 0.5, 0.1, 0.05, 0.4, 0.8],
            "score": [0.9, 0.5, 0.1, 0.05, 0.4, 0.8],
        }
    )
    treatment = control.copy()
    treatment.loc[treatment.y_true == 0, ["score_raw", "score"]] -= 0.1
    treatment.loc[treatment.y_true == 1, ["score_raw", "score"]] += 0.1
    shifts, sample_shifts = pair_prediction_tail_shifts(
        treatment, control, reference
    )
    all_normal = shifts.query(
        "label == 'normal' and scope == 'all' and score_type == 'raw'"
    ).iloc[0]
    all_defect = shifts.query(
        "label == 'defect' and scope == 'all' and score_type == 'raw'"
    ).iloc[0]
    assert all_normal.mean_shift == pytest.approx(-0.1)
    assert all_normal.beneficial_rate == 1.0
    assert all_defect.mean_shift == pytest.approx(0.1)
    assert all_defect.beneficial_rate == 1.0
    assert len(sample_shifts) == 6


def test_control_consensus_and_sample_shift_consistency(tmp_path):
    first = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "y_true": [0, 1],
            "score": [0.1, 0.9],
            "score_raw": [0.2, 0.8],
        }
    )
    second = first.copy()
    second["score_raw"] = [0.4, 0.6]
    paths = [tmp_path / "first.csv", tmp_path / "second.csv"]
    first.to_csv(paths[0], index=False)
    second.iloc[::-1].to_csv(paths[1], index=False)
    consensus = build_control_consensus(paths, cache_path=tmp_path / "scores.mmap")
    assert consensus.set_index("sample_id").loc[
        "a", "control_median_score_raw"
    ] == pytest.approx(0.3)
    assert consensus.set_index("sample_id").loc[
        "b", "control_median_score_raw"
    ] == pytest.approx(0.7)

    shift_one = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "y_true": [0, 1],
            "raw_shift": [-0.2, 0.1],
            "calibrated_shift": [-0.1, 0.05],
        }
    )
    shift_two = shift_one.copy()
    shift_two["raw_shift"] = [0.1, 0.2]
    shift_two["calibrated_shift"] = [0.1, 0.1]
    summary = summarize_sample_shift_consistency(
        [("R1", 1, shift_one), ("R1", 2, shift_two)]
    )
    normal = summary.set_index(["control", "sample_id"]).loc[("R1", "a")]
    assert normal.raw_benefit_rate == 0.5
    assert normal.raw_mean_shift == pytest.approx(-0.05)
    defect = summary.set_index(["control", "sample_id"]).loc[("R1", "b")]
    assert defect.raw_benefit_rate == 1.0


def test_calibration_diagnostics_and_bottom_gap_sign_summary(tmp_path):
    prediction = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "y_true": [0, 0, 1, 1],
            "score": [0.1, 0.2, 0.8, 0.9],
            "score_raw": [0.2, 0.3, 0.7, 0.8],
        }
    )
    diagnostics = calibration_diagnostics(prediction, bins=2)
    assert diagnostics["auroc"] == 1.0
    assert diagnostics["auprc"] == 1.0
    assert diagnostics["brier"] < 0.05
    assert diagnostics["ece"] >= 0.0

    selection_root = tmp_path / "selections"
    _write_selection(
        selection_root,
        "RUN_001",
        arm="T",
        triad_id="TRIAD_001",
        condition_id="A13_fixture",
        rows=[
            ("n1", 0, "normal_replay", "ordinary"),
            ("n2", 0, "normal_replay", "ordinary"),
            ("n3", 0, "normal_replay", "ordinary"),
        ],
    )
    matrix = _matrix([("RUN_001", "TRIAD_001", "A13", "A", "T", 3)])
    values = tmp_path / "values.csv"
    pd.DataFrame(
        {
            "sample_id": ["n1", "n2", "n3"],
            "gap_critical_score": [-0.2, -0.1, 0.01],
        }
    ).to_csv(values, index=False)
    summary = summarize_selected_score_signs(
        matrix,
        selection_root,
        values,
        condition_slot="A13",
        score_column="gap_critical_score",
    )
    assert summary["negative_count"] == 2
    assert summary["nonnegative_count"] == 1
    assert summary["min_score"] == pytest.approx(-0.2)
