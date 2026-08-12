from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stage1_gapvalue240.selection_mechanisms import (
    GRADIENT_FIELDS,
    SelectionMechanismError,
    build_budget_nesting,
    build_field_usage_registry,
    build_method_overlaps,
    build_same_selection_reversals,
    build_selection_summaries,
    build_triad_overlap_audit,
    load_frozen_sample_features,
    verify_and_load_selections,
)


VALUE_COLUMNS = [
    "sample_id",
    "y_true",
    "oof_fold",
    "dynamic_bucket",
    "epoch_count",
    "mean_p_defect",
    "std_p_defect",
    "p_defect_start",
    "p_defect_end",
    "p_defect_trend",
    "correct_rate",
    "forgetting_count",
    "mean_loss",
    "mean_abs_margin",
    "first_learned_epoch",
    "last_wrong_epoch",
    "mean_p_good_gap_epochs",
    "mean_p_bad_gap_epochs",
    "gap_delta_raw",
    "confidence_fp_score",
    "boundary_fp_score",
    "persistent_fp_score",
    "learnable_hard_fp_score",
    "gap_critical_score",
    "gap_guard_score",
    "gap_critical_guard_score",
    *GRADIENT_FIELDS,
]

DYNAMICS_COLUMNS = [
    "sample_id",
    "y_true",
    "oof_fold",
    "epoch_count",
    "first_epoch",
    "last_epoch",
    "p_defect_start",
    "p_defect_end",
    "p_defect_trend",
    "mean_p_defect",
    "std_p_defect",
    "min_p_defect",
    "max_p_defect",
    "mean_loss",
    "loss_auc_mean",
    "max_loss",
    "mean_margin_signed",
    "mean_abs_margin",
    "min_margin_signed",
    "correct_count",
    "correct_rate",
    "final_correct",
    "first_learned_epoch",
    "last_wrong_epoch",
    "forgetting_count",
    "dynamic_bucket",
]

MASTER_COLUMNS = [
    "sample_id",
    "canonical_image_relpath",
    "y_true",
    "oof_fold",
    "oof_group_id",
    "oof_group_source",
    "train_primary_class",
    "source_manifest",
    "source_csv_path",
    "source_csv_row_number",
    "filename",
]

SELECTION_COLUMNS = [
    "run_slot",
    "triad_id",
    "condition_id",
    "arm",
    "training_seed",
    "selection_seed",
    "rank",
    "sample_id",
    "y_true",
    "oof_fold",
    "dynamic_bucket",
    "mean_p_defect",
    "correct_rate",
    "std_p_defect",
    "replay_role",
    "source_method",
]


def _feature_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    rows = []
    dynamics = []
    master = []
    for index, sample_id in enumerate(["n1", "n2", "d1", "d2", "n3", "n4"]):
        label = int(sample_id.startswith("d"))
        common = {
            "sample_id": sample_id,
            "y_true": label,
            "oof_fold": f"{index % 2:02d}",
            "dynamic_bucket": "learnable_hard" if index % 2 == 0 else "persistent_hard",
            "epoch_count": 200,
            "mean_p_defect": 0.2 + index / 20,
            "std_p_defect": 0.1 + index / 100,
            "p_defect_start": 0.4,
            "p_defect_end": 0.1 + index / 100,
            "p_defect_trend": -0.3 + index / 100,
            "correct_rate": 0.8 - index / 20,
            "forgetting_count": index,
            "mean_loss": 0.1 + index / 100,
            "mean_abs_margin": 0.3,
            "first_learned_epoch": 2,
            "last_wrong_epoch": [20, 180, 199, np.nan, 160, 161][index],
        }
        value = {
            **common,
            "mean_p_good_gap_epochs": 0.1,
            "mean_p_bad_gap_epochs": 0.4,
            "gap_delta_raw": 0.3,
            "confidence_fp_score": 0.2,
            "boundary_fp_score": 0.4,
            "persistent_fp_score": 1 - common["correct_rate"],
            "learnable_hard_fp_score": 0.5,
            "gap_critical_score": 0.3 if label == 0 else np.nan,
            "gap_guard_score": 0.2 if label == 1 else np.nan,
            "gap_critical_guard_score": 0.2,
            **{field: np.nan for field in GRADIENT_FIELDS},
        }
        rows.append(value)
        dynamics.append(
            {
                **common,
                "first_epoch": 1,
                "last_epoch": 200,
                "min_p_defect": 0.01,
                "max_p_defect": 0.9,
                "loss_auc_mean": common["mean_loss"],
                "max_loss": 0.8,
                "mean_margin_signed": 0.2,
                "min_margin_signed": -0.4,
                "correct_count": round(common["correct_rate"] * 200),
                "final_correct": int(index not in {1, 2}),
            }
        )
        master.append(
            {
                "sample_id": sample_id,
                "canonical_image_relpath": sample_id,
                "y_true": label,
                "oof_fold": f"{index % 2:02d}",
                "oof_group_id": f"group_{index // 2}",
                "oof_group_source": "fixture",
                "train_primary_class": "Normal" if label == 0 else "PF",
                "source_manifest": "fixture.csv",
                "source_csv_path": "fixture.csv",
                "source_csv_row_number": index + 1,
                "filename": f"{sample_id}.png",
            }
        )
    value_path = tmp_path / "sample_value_table.csv"
    dynamics_path = tmp_path / "sample_dynamics_summary.csv"
    master_path = tmp_path / "master_sample_index.csv"
    pd.DataFrame(rows, columns=VALUE_COLUMNS).to_csv(value_path, index=False)
    pd.DataFrame(dynamics, columns=DYNAMICS_COLUMNS).to_csv(dynamics_path, index=False)
    pd.DataFrame(master, columns=MASTER_COLUMNS).to_csv(master_path, index=False)
    return value_path, dynamics_path, master_path


def _selection_row(run: dict, rank: int, sample_id: str, role: str, features: pd.DataFrame) -> dict:
    sample = features.set_index("sample_id").loc[sample_id]
    return {
        "run_slot": run["run_slot"],
        "triad_id": run["triad_id"],
        "condition_id": run["condition_id"],
        "arm": run["arm"],
        "training_seed": run["training_seed"],
        "selection_seed": run["selection_seed"],
        "rank": rank,
        "sample_id": sample_id,
        "y_true": int(sample["y_true"]),
        "oof_fold": sample["oof_fold"],
        "dynamic_bucket": sample["dynamic_bucket"],
        "mean_p_defect": sample["mean_p_defect"],
        "correct_rate": sample["correct_rate"],
        "std_p_defect": sample["std_p_defect"],
        "replay_role": role,
        "source_method": run["method"] if run["arm"] == "T" else run["arm"],
    }


def _write_selections(
    tmp_path: Path, runs: pd.DataFrame, features: pd.DataFrame, sample_sets: dict[str, list[tuple[str, str]]]
) -> tuple[Path, Path]:
    root = tmp_path / "selections"
    records = []
    for run in runs.to_dict("records"):
        role_ranks: dict[str, int] = {}
        rows = []
        for sample_id, role in sample_sets[run["run_slot"]]:
            role_ranks[role] = role_ranks.get(role, 0) + 1
            rows.append(_selection_row(run, role_ranks[role], sample_id, role, features))
        path = root / run["run_slot"] / "selection_manifest.csv"
        path.parent.mkdir(parents=True)
        pd.DataFrame(rows, columns=SELECTION_COLUMNS).to_csv(path, index=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        records.append(
            {
                "run_slot": run["run_slot"],
                "selection_manifest": str(path.relative_to(tmp_path)),
                "sha256": digest,
            }
        )
    index = tmp_path / "selection_index.csv"
    pd.DataFrame(records).to_csv(index, index=False)
    return root, index


def _runs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_slot": "RUN_001",
                "triad_id": "TRIAD_001",
                "phase": "B",
                "condition_slot": "B01",
                "condition_id": "B01_Guard",
                "method": "GapGuard-Raw",
                "budget": 4,
                "guard_ratio": 0.5,
                "arm": arm,
                "training_seed": 11,
                "selection_seed": 100 + i,
            }
            for i, arm in enumerate(["T", "R1", "R2"])
        ]
    ).assign(run_slot=["RUN_001", "RUN_002", "RUN_003"])


def test_feature_catalog_classifies_every_field_and_gradients_are_not_collected(tmp_path: Path) -> None:
    value_path, dynamics_path, master_path = _feature_files(tmp_path)
    features, registry = load_frozen_sample_features(value_path, dynamics_path, master_path)

    assert len(features) == 6
    assert not registry[["source_table", "field_name"]].duplicated().any()
    assert set(registry.loc[registry["availability"] == "NOT_COLLECTED", "field_name"]) == set(
        GRADIENT_FIELDS
    )
    assert registry["analysis_role"].ne("").all()
    expected = (
        len(VALUE_COLUMNS)
        + len(DYNAMICS_COLUMNS)
        + len(MASTER_COLUMNS)
        + len(SELECTION_COLUMNS)
    )
    assert len(registry) == expected

    values = pd.read_csv(value_path)
    values.loc[0, GRADIENT_FIELDS[0]] = 1.0
    values.to_csv(value_path, index=False)
    with pytest.raises(SelectionMechanismError, match="gradient.*not collected"):
        load_frozen_sample_features(value_path, dynamics_path, master_path)


def test_phase_b_rank_is_unique_by_role_and_sha_is_enforced(tmp_path: Path) -> None:
    value_path, dynamics_path, master_path = _feature_files(tmp_path)
    features, _ = load_frozen_sample_features(value_path, dynamics_path, master_path)
    runs = _runs()
    sets = {
        "RUN_001": [("n1", "normal_replay"), ("n2", "normal_replay"), ("d1", "defect_guard"), ("d2", "defect_guard")],
        "RUN_002": [("n3", "normal_replay"), ("n4", "normal_replay"), ("d1", "defect_guard"), ("d2", "defect_guard")],
        "RUN_003": [("n1", "normal_replay"), ("n3", "normal_replay"), ("d1", "defect_guard"), ("d2", "defect_guard")],
    }
    root, index = _write_selections(tmp_path, runs, features, sets)
    indexed = pd.read_csv(index)
    runs = runs.merge(indexed[["run_slot", "sha256"]], on="run_slot").rename(
        columns={"sha256": "selection_sha256"}
    )

    selected, audit = verify_and_load_selections(runs, index, root, features)
    assert len(selected) == 12
    assert len(audit) == 3
    assert audit["phase_b_role_rank_key_valid"].all()
    assert selected.duplicated(["run_slot", "rank"]).any()
    assert not selected.duplicated(["run_slot", "replay_role", "rank"]).any()

    path = root / "RUN_001" / "selection_manifest.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(SelectionMechanismError, match="SHA-256 mismatch"):
        verify_and_load_selections(runs, index, root, features)


def test_summaries_quantify_composition_and_late_persistence_proxies(tmp_path: Path) -> None:
    value_path, dynamics_path, master_path = _feature_files(tmp_path)
    features, _ = load_frozen_sample_features(value_path, dynamics_path, master_path)
    runs = _runs()
    sets = {
        "RUN_001": [("n1", "normal_replay"), ("n2", "normal_replay"), ("d1", "defect_guard"), ("d2", "defect_guard")],
        "RUN_002": [("n3", "normal_replay"), ("n4", "normal_replay"), ("d1", "defect_guard"), ("d2", "defect_guard")],
        "RUN_003": [("n1", "normal_replay"), ("n3", "normal_replay"), ("d1", "defect_guard"), ("d2", "defect_guard")],
    }
    root, index = _write_selections(tmp_path, runs, features, sets)
    indexed = pd.read_csv(index)
    runs = runs.merge(indexed[["run_slot", "sha256"]], on="run_slot").rename(columns={"sha256": "selection_sha256"})
    selected, _ = verify_and_load_selections(runs, index, root, features)

    categorical, numeric, late = build_selection_summaries(selected)
    assert {"dynamic_bucket", "oof_fold", "oof_group_id", "train_primary_class", "y_true"}.issubset(
        set(categorical["dimension"])
    )
    assert {"mean_p_defect", "std_p_defect", "correct_rate", "forgetting_count"}.issubset(
        set(numeric["feature"])
    )
    row = late.loc[(late["run_slot"] == "RUN_001") & (late["replay_role"] == "normal_replay")].iloc[0]
    assert row["late_wrong_after_epoch160_rate"] == pytest.approx(0.5)
    assert row["final_wrong_rate"] == pytest.approx(0.5)
    assert row["late_persistence_semantics"] == "summary_indicator_not_late40_frequency"


def test_triad_overlap_reports_r2_effective_unique_contrast() -> None:
    selected = pd.DataFrame(
        [
            ("TRIAD_001", "RUN_T", "T", "normal_replay", sample)
            for sample in ["a", "b", "c", "d"]
        ]
        + [
            ("TRIAD_001", "RUN_R1", "R1", "normal_replay", sample)
            for sample in ["e", "f", "g", "h"]
        ]
        + [
            ("TRIAD_001", "RUN_R2", "R2", "normal_replay", sample)
            for sample in ["a", "b", "c", "x"]
        ],
        columns=["triad_id", "run_slot", "arm", "replay_role", "sample_id"],
    )
    audit = build_triad_overlap_audit(selected)
    t_r1 = audit.loc[(audit["left_arm"] == "T") & (audit["right_arm"] == "R1")].iloc[0]
    t_r2 = audit.loc[(audit["left_arm"] == "T") & (audit["right_arm"] == "R2")].iloc[0]
    assert t_r1["intersection_count"] == 0
    assert t_r1["effective_unique_contrast_rate"] == 1.0
    assert t_r2["intersection_count"] == 3
    assert t_r2["jaccard"] == pytest.approx(3 / 5)
    assert t_r2["effective_unique_contrast_rate"] == pytest.approx(0.25)


def test_method_overlap_and_budget_nesting_are_role_aware() -> None:
    rows = []
    specifications = [
        ("A", "m1", 2, "r1", ["a", "b"]),
        ("A", "m2", 2, "r2", ["b", "c"]),
        ("A", "m1", 4, "r3", ["a", "b", "c", "d"]),
    ]
    for phase, method, budget, slot, samples in specifications:
        for sample in samples:
            rows.append(
                {
                    "phase": phase,
                    "method": method,
                    "budget": budget,
                    "training_seed": 7,
                    "run_slot": slot,
                    "arm": "T",
                    "replay_role": "normal_replay",
                    "sample_id": sample,
                }
            )
    selected = pd.DataFrame(rows)
    overlaps = build_method_overlaps(selected)
    assert len(overlaps) == 1
    assert overlaps.iloc[0]["intersection_count"] == 1
    nesting = build_budget_nesting(selected)
    assert len(nesting) == 1
    assert bool(nesting.iloc[0]["lower_is_subset_of_higher"])
    assert nesting.iloc[0]["lower_retained_rate"] == 1.0


def test_same_selection_digest_exposes_seed_reversal() -> None:
    selected = pd.DataFrame(
        [
            {
                "run_slot": run,
                "triad_id": triad,
                "arm": "T",
                "training_seed": seed,
                "replay_role": "normal_replay",
                "sample_id": sample,
            }
            for run, triad, seed in [("RUN_1", "TRIAD_1", 1), ("RUN_2", "TRIAD_2", 2)]
            for sample in ["a", "b"]
        ]
    )
    outcomes = pd.DataFrame(
        [
            {"triad_id": "TRIAD_1", "exclusive_cohort": "DUAL_IMPROVEMENT", "dual_improvement": True, "high_value": True, "dual_harm": False},
            {"triad_id": "TRIAD_2", "exclusive_cohort": "DUAL_HARM", "dual_improvement": False, "high_value": False, "dual_harm": True},
        ]
    )
    sets, grouped, reversals = build_same_selection_reversals(selected, outcomes)
    assert sets["sample_set_digest"].nunique() == 1
    assert len(grouped) == 1
    assert bool(grouped.iloc[0]["spans_dual_improvement_and_dual_harm"])
    assert set(reversals["triad_id"]) == {"TRIAD_1", "TRIAD_2"}


def test_registry_rejects_unknown_fields() -> None:
    with pytest.raises(SelectionMechanismError, match="Unclassified fields"):
        build_field_usage_registry(
            value_columns=[*VALUE_COLUMNS, "mystery"],
            dynamics_columns=DYNAMICS_COLUMNS,
            master_columns=MASTER_COLUMNS,
            selection_columns=SELECTION_COLUMNS,
        )
