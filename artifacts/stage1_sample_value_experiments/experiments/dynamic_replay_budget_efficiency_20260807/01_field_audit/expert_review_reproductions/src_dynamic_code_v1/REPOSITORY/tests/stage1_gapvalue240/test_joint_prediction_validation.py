from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.joint_prediction_validation import (
    JointPredictionValidationError,
    publish_joint_prediction_validation,
    run_joint_prediction_validation,
)


def _matrix_and_registries() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for seed_index, seed in enumerate((101, 102, 103)):
        for condition_index in range(25):
            success = condition_index % 5 == 0
            harmful = condition_index % 5 == 1
            mixed = not success and not harmful
            delta_tn = 500.0 if success else -20.0 if harmful else 10.0
            delta_fn = -1.0 if success else 2.0 if harmful else 1.0
            rows.append(
                {
                    "triad_id": f"D_{seed}_{condition_index:02d}",
                    "phase": "A" if condition_index < 19 else "B",
                    "condition_slot": f"A{condition_index + 1:02d}",
                    "discovery_or_confirmation": "discovery",
                    "training_seed": seed,
                    "treatment_sample_set_digest": f"digest_{condition_index:02d}",
                    "delta_TN_R1": delta_tn,
                    "delta_FN_R1": delta_fn,
                    "delta_TN_R2": delta_tn + (5 if success else 0),
                    "delta_FN_R2": delta_fn,
                    "dual_improvement": success,
                    "high_value": success,
                    "dual_harm": harmful,
                    "exclusive_cohort": (
                        "DUAL_IMPROVEMENT"
                        if success
                        else "DUAL_HARM"
                        if harmful
                        else "MIXED_OR_REVERSAL"
                    ),
                    "feature_120": float(success) + seed_index * 0.01,
                    "feature_200": float(success),
                }
            )
    # External Phase C deliberately has zero successes, matching the real A02
    # falsification structure while retaining all five observations.
    for index, seed in enumerate((201, 202, 203, 204, 205)):
        rows.append(
            {
                "triad_id": f"C_{seed}",
                "phase": "C",
                "condition_slot": "A02",
                "discovery_or_confirmation": "confirmation",
                "training_seed": seed,
                "treatment_sample_set_digest": "digest_01",
                "delta_TN_R1": -10.0,
                "delta_FN_R1": 2.0,
                "delta_TN_R2": -20.0,
                "delta_FN_R2": 1.0,
                "dual_improvement": False,
                "high_value": False,
                "dual_harm": True,
                "exclusive_cohort": "DUAL_HARM",
                "feature_120": 0.1 + index * 0.01,
                "feature_200": 0.0,
            }
        )
    matrix = pd.DataFrame(rows)
    registry_rows = []
    role_rows = []
    for column in matrix.columns:
        allowed = column in {"feature_120", "feature_200"}
        epoch = 120 if column == "feature_120" else 200 if column == "feature_200" else 0
        registry_rows.append(
            {
                "feature": column,
                "feature_family": "SYNTHETIC",
                "available_epoch": epoch,
                "allowed_as_predictor": allowed,
                "use": "candidate" if allowed else "label or identity",
                "base_feature": column,
                "source_table": "synthetic",
            }
        )
        role_rows.append(
            {
                **registry_rows[-1],
                "source_field": column,
                "analysis_role": "PREDICTOR" if allowed else "OUTCOME_OR_IDENTITY",
                "control": "",
                "arm": "",
                "replay_role": "",
            }
        )
    return matrix, pd.DataFrame(registry_rows), pd.DataFrame(role_rows)


def test_all_cutoffs_use_all_80_and_keep_mixed_in_honest_validation() -> None:
    matrix, registry, roles = _matrix_and_registries()
    result = run_joint_prediction_validation(
        matrix,
        registry,
        roles,
        cutoffs=(120, 200),
        max_features=2,
        include_digest_diagnostics=True,
        random_state=7,
    )

    assert set(result.summaries["cutoff"]) == {120, 200}
    assert {
        "balanced_accuracy",
        "precision",
        "coverage",
        "selected_success_rate",
        "precision_lower_bound_one_sided",
    }.issubset(result.summaries.columns)
    for cutoff in (120, 200):
        main = result.predictions.loc[
            (result.predictions["cutoff"] == cutoff)
            & result.predictions["validation_scheme"].isin(
                ["DISCOVERY_LOSO_SEED", "PHASE_C_EXTERNAL_FALSIFICATION"]
            )
        ]
        assert len(main) == 80
        assert main["triad_id"].nunique() == 80
        assert main["exclusive_cohort"].eq("MIXED_OR_REVERSAL").sum() == 45
    assert result.contract["analysis_population"] == "ALL_80_TRIADS_INCLUDING_MIXED"
    assert result.contract["baseline_counts"] == {
        "all_triads": 80,
        "dual_improvement": 15,
        "dual_harm": 20,
        "mixed_or_reversal": 45,
    }
    cutoff_features = result.cutoff_feature_counts.set_index("cutoff")
    assert cutoff_features.loc[120, "eligible_predictor_count"] == 1
    assert cutoff_features.loc[200, "eligible_predictor_count"] == 2
    assert result.fold_details["selected_feature_count"].le(2).all()


def test_joint_label_is_recomputed_from_both_controls_and_leakage_fails_closed() -> None:
    matrix, registry, roles = _matrix_and_registries()
    broken = matrix.copy()
    broken.loc[0, "dual_improvement"] = False
    with pytest.raises(JointPredictionValidationError, match="joint label"):
        run_joint_prediction_validation(
            broken,
            registry,
            roles,
            cutoffs=(120,),
            max_features=1,
            include_digest_diagnostics=False,
        )

    leaky_registry = registry.copy()
    leaky_registry.loc[
        leaky_registry["feature"].eq("delta_TN_R1"), "allowed_as_predictor"
    ] = True
    with pytest.raises(JointPredictionValidationError, match="outcome/confound"):
        run_joint_prediction_validation(
            matrix,
            leaky_registry,
            roles,
            cutoffs=(120,),
            max_features=1,
            include_digest_diagnostics=False,
        )
    with pytest.raises(ValueError, match="max_features"):
        run_joint_prediction_validation(
            matrix,
            registry,
            roles,
            cutoffs=(120,),
            max_features=17,
            include_digest_diagnostics=False,
        )


def test_atomic_publisher_emits_sha_row_manifest_without_touching_state(
    tmp_path: Path,
) -> None:
    matrix, registry, roles = _matrix_and_registries()
    result = run_joint_prediction_validation(
        matrix,
        registry,
        roles,
        cutoffs=(120,),
        max_features=1,
        include_digest_diagnostics=False,
        random_state=5,
    )
    output = tmp_path / "report.inprogress"
    output.mkdir()
    state = output / "ANALYSIS_STATE.json"
    state.write_text('{"status":"UNCHANGED"}', encoding="utf-8")

    summary = publish_joint_prediction_validation(
        result,
        output,
        source_paths={
            "matrix": tmp_path / "source_matrix.csv",
            "feature_registry": tmp_path / "source_registry.csv",
            "role_registry": tmp_path / "source_roles.csv",
        },
        source_frames={
            "matrix": matrix,
            "feature_registry": registry,
            "role_registry": roles,
        },
    )

    assert summary["output_files"] == 8
    assert state.read_text(encoding="utf-8") == '{"status":"UNCHANGED"}'
    manifest_path = output / "tables/joint_prediction_output_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == 7
    for row in manifest.itertuples(index=False):
        path = output / "tables" / row.filename
        assert path.is_file()
        assert sha256(path.read_bytes()).hexdigest().upper() == row.sha256
    contract = json.loads(
        (output / "tables/joint_prediction_validation_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["max_features"] == 1
    assert contract["source_sha256"]
    assert not list(output.rglob("*.tmp"))
    with pytest.raises(FileExistsError):
        publish_joint_prediction_validation(result, output)
