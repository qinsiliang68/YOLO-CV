from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stage1_gapvalue240.unified_feature_matrix import (
    UnifiedFeatureMatrixError,
    build_unified_feature_matrix,
    publish_unified_feature_matrix,
)


def _inputs() -> dict[str, pd.DataFrame]:
    outcomes = pd.DataFrame(
        [
            {
                "triad_id": "TRIAD_A",
                "phase": "A",
                "condition_slot": "A01",
                "condition_id": "A01_method_B600",
                "method": "method",
                "budget": 600,
                "guard_ratio": 0.0,
                "training_seed": 101,
                "discovery_or_confirmation": "discovery",
                "machine_id": "machine_1",
                "input_snapshot_id": "snapshot_1",
                "resume_count": 0,
                "delta_TN_R1": 10.0,
                "delta_FN_R1": -1.0,
                "delta_TN_R2": 5.0,
                "delta_FN_R2": 0.0,
                "G_TN": 5.0,
                "G_FN": 0.0,
                "dual_improvement": True,
                "high_value": False,
                "dual_harm": False,
                "exclusive_cohort": "DUAL_IMPROVEMENT",
            },
            {
                "triad_id": "TRIAD_B",
                "phase": "B",
                "condition_slot": "B04",
                "condition_id": "B04_guard_B3000",
                "method": "guard",
                "budget": 3000,
                "guard_ratio": 0.1,
                "training_seed": 102,
                "discovery_or_confirmation": "discovery",
                "machine_id": "machine_2",
                "input_snapshot_id": "snapshot_2",
                "resume_count": 1,
                "delta_TN_R1": -10.0,
                "delta_FN_R1": 2.0,
                "delta_TN_R2": -5.0,
                "delta_FN_R2": 1.0,
                "G_TN": -10.0,
                "G_FN": 2.0,
                "dual_improvement": False,
                "high_value": False,
                "dual_harm": True,
                "exclusive_cohort": "DUAL_HARM",
            },
        ]
    )
    telemetry_rows = []
    for triad_index, triad_id in enumerate(("TRIAD_A", "TRIAD_B")):
        for control_index, control in enumerate(("R1", "R2")):
            telemetry_rows.append(
                {
                    "triad_id": triad_id,
                    "control": control,
                    "delta__train_loss__at_120": 0.1 + triad_index + control_index,
                    "delta__train_loss__at_200": 0.2 + triad_index + control_index,
                    "effective_optimizer": "MuSGD",
                    # This outcome is present in a source table but not a predictor.
                    "delta_TN": 999.0,
                }
            )
    telemetry = pd.DataFrame(telemetry_rows)
    telemetry_registry = pd.DataFrame(
        [
            {
                "feature": "train_loss__at_120",
                "feature_family": "TRAINING_TELEMETRY",
                "available_epoch": 120,
                "allowed_as_predictor": True,
                "use": "fold-local candidate feature",
            },
            {
                "feature": "train_loss__at_200",
                "feature_family": "TRAINING_TELEMETRY",
                "available_epoch": 200,
                "allowed_as_predictor": True,
                "use": "fold-local candidate feature",
            },
            {
                "feature": "effective_optimizer",
                "feature_family": "EFFECTIVE_OPTIMIZER",
                "available_epoch": 0,
                "allowed_as_predictor": True,
                "use": "constant training configuration",
            },
        ]
    )

    role_specs = {
        "TRIAD_A": {"phase": "A", "budget": 600, "guard_ratio": 0.0, "roles": {"normal_replay": 600}},
        "TRIAD_B": {
            "phase": "B",
            "budget": 3000,
            "guard_ratio": 0.1,
            "roles": {"normal_replay": 2700, "defect_guard": 300},
        },
    }
    numeric_rows: list[dict[str, object]] = []
    categorical_rows: list[dict[str, object]] = []
    late_rows: list[dict[str, object]] = []
    for triad_id, specification in role_specs.items():
        for arm_index, arm in enumerate(("T", "R1", "R2")):
            for role, count in specification["roles"].items():
                common = {
                    "run_slot": f"{triad_id}_{arm}",
                    "triad_id": triad_id,
                    "phase": specification["phase"],
                    "condition_id": f"{triad_id}_condition",
                    "method": "synthetic",
                    "budget": specification["budget"],
                    "guard_ratio": specification["guard_ratio"],
                    "arm": arm,
                    "training_seed": 101 if triad_id == "TRIAD_A" else 102,
                    "selection_seed": 200 + arm_index,
                    "replay_role": role,
                }
                for feature_index, feature in enumerate(("mean_p_defect", "forgetting_count")):
                    numeric_rows.append(
                        {
                            **common,
                            "feature": feature,
                            "selected_count": count,
                            "non_null_count": count,
                            "mean": arm_index + feature_index + (10 if role == "defect_guard" else 0),
                            "std": 0.5,
                            "min": 0.0,
                            "q05": 0.1,
                            "q25": 0.2,
                            "median": 0.3,
                            "q75": 0.4,
                            "q95": 0.5,
                            "max": 1.0,
                            "positive_rate_among_non_null": 0.8,
                        }
                    )
                # Low-cardinality dynamic_bucket shares.
                for value, share in (("learnable_hard", 0.75), ("ordinary", 0.25)):
                    categorical_rows.append(
                        {
                            **common,
                            "dimension": "dynamic_bucket",
                            "value": value,
                            "count": int(count * share),
                            "share": share,
                            "selected_count": count,
                        }
                    )
                # High-cardinality group distribution when threshold=2.
                for group_index, share in enumerate((0.5, 0.3, 0.2)):
                    categorical_rows.append(
                        {
                            **common,
                            "dimension": "oof_group_id",
                            "value": f"group_{group_index}",
                            "count": int(count * share),
                            "share": share,
                            "selected_count": count,
                        }
                    )
                late_rows.append(
                    {
                        **common,
                        "selected_count": count,
                        "late_wrong_after_epoch160_count": int(count * 0.1),
                        "late_wrong_after_epoch160_rate": 0.1,
                        "last_wrong_epoch_non_null_count": count,
                        "last_wrong_epoch_mean_non_null": 100.0,
                        "final_wrong_rate": 0.05,
                        "persistent_0p5_error_rate": 0.2,
                        "forgetting_count_mean": 1.5,
                        "late_persistence_semantics": "summary_indicator_not_late40_frequency",
                        "blank_last_wrong_semantics": "no_recorded_wrong_epoch",
                    }
                )

    treatment_sets = pd.DataFrame(
        [
            {
                "triad_id": "TRIAD_A",
                "run_slot": "TRIAD_A_T",
                "sample_set_digest": "DIGEST_A",
                "selected_count": 600,
            },
            {
                "triad_id": "TRIAD_B",
                "run_slot": "TRIAD_B_T",
                "sample_set_digest": "DIGEST_B",
                "selected_count": 3000,
            },
        ]
    )
    checkpoints = pd.DataFrame(
        [
            {
                "triad_id": "TRIAD_A",
                "ckpt__delta_relative_l2__best_to_last_ema__ALL": 0.01,
                "delta_TN_R1": 10.0,
            },
            {
                "triad_id": "TRIAD_B",
                "ckpt__delta_relative_l2__best_to_last_ema__ALL": 0.02,
                "delta_TN_R1": -10.0,
            },
        ]
    )
    resources = pd.DataFrame(
        [
            {
                "triad_id": "TRIAD_A",
                "t_machine_id": "machine_1",
                "all_arms_same_machine": True,
                "resumed_arm_count": 0,
                "gpu_total_hours": 40.0,
            },
            {
                "triad_id": "TRIAD_B",
                "t_machine_id": "machine_2",
                "all_arms_same_machine": False,
                "resumed_arm_count": 1,
                "gpu_total_hours": 45.0,
            },
        ]
    )
    return {
        "triad_outcomes": outcomes,
        "paired_telemetry": telemetry,
        "selection_numeric": pd.DataFrame(numeric_rows),
        "selection_categorical": pd.DataFrame(categorical_rows),
        "selection_late": pd.DataFrame(late_rows),
        "treatment_selection_sets": treatment_sets,
        "checkpoint_triads": checkpoints,
        "resource_triads": resources,
        "telemetry_registry": telemetry_registry,
    }


def _build(**overrides):
    inputs = _inputs()
    inputs.update(overrides)
    return build_unified_feature_matrix(
        **inputs,
        expected_triads=2,
        low_cardinality_max_levels=2,
    )


def test_unified_matrix_preserves_controls_arms_roles_and_all_selection_statistics() -> None:
    result = _build()
    matrix = result.matrix.set_index("triad_id")

    assert len(matrix) == 2
    assert matrix.columns.is_unique
    assert matrix.loc["TRIAD_A", "telemetry__R1__delta__train_loss__at_120"] == 0.1
    assert matrix.loc["TRIAD_A", "telemetry__R2__delta__train_loss__at_120"] == 1.1
    numeric_prefix = "selection_numeric__T__normal_replay__mean_p_defect__"
    for statistic in (
        "selected_count",
        "non_null_count",
        "mean",
        "std",
        "min",
        "q05",
        "q25",
        "median",
        "q75",
        "q95",
        "max",
        "positive_rate_among_non_null",
    ):
        assert f"{numeric_prefix}{statistic}" in matrix.columns
    assert (
        matrix.loc[
            "TRIAD_B",
            "selection_numeric__T__defect_guard__mean_p_defect__mean",
        ]
        == 10
    )
    assert np.isnan(
        matrix.loc[
            "TRIAD_A",
            "selection_numeric__T__defect_guard__mean_p_defect__mean",
        ]
    )
    assert (
        matrix.loc[
            "TRIAD_A",
            "selection_categorical__T__normal_replay__dynamic_bucket__learnable_hard__share",
        ]
        == 0.75
    )
    high_prefix = "selection_categorical__T__normal_replay__oof_group_id__"
    assert matrix.loc["TRIAD_A", f"{high_prefix}level_count"] == 3
    assert matrix.loc["TRIAD_A", f"{high_prefix}hhi"] == pytest.approx(0.38)
    assert matrix.loc["TRIAD_A", f"{high_prefix}max_share"] == 0.5
    assert matrix.loc["TRIAD_A", f"{high_prefix}entropy"] == pytest.approx(
        -(0.5 * np.log(0.5) + 0.3 * np.log(0.3) + 0.2 * np.log(0.2))
    )
    assert matrix.loc["TRIAD_A", "treatment_sample_set_digest"] == "DIGEST_A"
    assert matrix.loc[
        "TRIAD_B", "ckpt__delta_relative_l2__best_to_last_ema__ALL"
    ] == 0.02
    assert matrix.loc["TRIAD_A", "confound__resource__gpu_total_hours"] == 40.0


def test_registry_covers_every_column_and_enforces_availability_and_roles() -> None:
    result = _build()
    registry = result.feature_registry.set_index("feature")
    roles = result.role_registry.set_index("feature")

    assert set(registry.index) == set(result.matrix.columns)
    assert set(roles.index) == set(result.matrix.columns)
    assert registry.loc[
        "telemetry__R1__delta__train_loss__at_120", "available_epoch"
    ] == 120
    assert registry.loc[
        "selection_numeric__T__normal_replay__mean_p_defect__mean",
        "available_epoch",
    ] == 0
    assert registry.loc[
        "ckpt__delta_relative_l2__best_to_last_ema__ALL", "available_epoch"
    ] == 200
    assert bool(
        registry.loc[
            "confound__resource__gpu_total_hours", "allowed_as_predictor"
        ]
    ) is False
    assert bool(registry.loc["delta_TN_R1", "allowed_as_predictor"]) is False
    assert bool(
        registry.loc[
            "training_config__effective_optimizer", "allowed_as_predictor"
        ]
    ) is False
    assert roles.loc[
        "training_config__effective_optimizer", "analysis_role"
    ] == "CONSTANT_TRAINING_CONFIG"
    assert roles.loc["delta_TN_R1", "analysis_role"] == "OUTCOME"
    assert roles.loc["machine_id", "analysis_role"] == "EXECUTION_CONFOUND"
    assert roles.loc[
        "treatment_sample_set_digest", "analysis_role"
    ] == "SELECTION_DIGEST"
    assert "delta_TN_R1_checkpoint" not in result.matrix


def test_phase_b_requires_both_replay_roles_with_contract_dose() -> None:
    inputs = _inputs()
    broken = inputs["selection_late"].loc[
        ~(
            inputs["selection_late"]["triad_id"].eq("TRIAD_B")
            & inputs["selection_late"]["arm"].eq("R2")
            & inputs["selection_late"]["replay_role"].eq("defect_guard")
        )
    ]
    inputs["selection_late"] = broken

    with pytest.raises(UnifiedFeatureMatrixError, match="Phase B.*defect_guard"):
        _build(**inputs)


def test_duplicate_and_leaky_predictor_sources_fail_closed() -> None:
    inputs = _inputs()
    duplicate_outcomes = pd.concat(
        [inputs["triad_outcomes"], inputs["triad_outcomes"][["phase"]]], axis=1
    )
    with pytest.raises(UnifiedFeatureMatrixError, match="duplicate columns"):
        _build(triad_outcomes=duplicate_outcomes)

    registry = pd.concat(
        [
            inputs["telemetry_registry"],
            pd.DataFrame(
                [
                    {
                        "feature": "TN_at_FN95",
                        "feature_family": "TRAINING_TELEMETRY",
                        "available_epoch": 120,
                        "allowed_as_predictor": True,
                        "use": "invalid leakage",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    telemetry = inputs["paired_telemetry"].copy()
    telemetry["delta__TN_at_FN95"] = 1.0
    with pytest.raises(UnifiedFeatureMatrixError, match="leaky predictor"):
        _build(paired_telemetry=telemetry, telemetry_registry=registry)


def test_publisher_writes_three_tables_atomically_only_when_explicitly_called(
    tmp_path: Path,
) -> None:
    result = _build()
    output = tmp_path / "analysis.inprogress"
    output.mkdir()

    summary = publish_unified_feature_matrix(result, output)

    assert summary["triads"] == 2
    assert summary["matrix_columns"] == len(result.matrix.columns)
    assert (output / "tables/unified_triad_feature_matrix.csv").is_file()
    assert (output / "tables/EXTENDED_FEATURE_TIME_REGISTRY.csv").is_file()
    assert (output / "tables/FEATURE_ROLE_REGISTRY.csv").is_file()
    assert not list(output.rglob("*.tmp"))
    with pytest.raises(FileExistsError):
        publish_unified_feature_matrix(result, output)
    with pytest.raises(ValueError, match="inprogress"):
        publish_unified_feature_matrix(result, tmp_path / "final")
