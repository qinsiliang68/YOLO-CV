from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.campaign_field_audit import (
    CampaignAuditError,
    assert_campaign_field_inventory_complete,
    build_campaign_field_inventory,
    build_data_volume_forecast,
    discover_oof_field_inventory,
    publish_campaign_field_audit,
)


def _refined_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "normalized_path": "02_logs/epoch_training_metrics.csv",
                "field_path": "epoch",
                "source_kind": "CSV",
                "files_observed": 240,
                "runs_observed": 240,
                "files_expected": 240,
                "runs_expected": 240,
                "schema_variant_count": 1,
                "usage_role": "PAIRING_KEY",
                "usage_status": "AUDITED",
                "rationale": "Within-run epoch identity.",
                "files_profiled": 240,
                "runs_profiled": 240,
                "values_observed": 48_000,
                "non_null_count": 48_000,
                "null_count": 0,
                "unique_count": 200,
                "constant_state": "NON_CONSTANT",
                "derived_consumer": "",
                "silently_dropped": False,
            },
            {
                "normalized_path": "02_logs/epoch_training_metrics.csv",
                "field_path": "train/loss",
                "source_kind": "CSV",
                "files_observed": 240,
                "runs_observed": 240,
                "files_expected": 240,
                "runs_expected": 240,
                "schema_variant_count": 1,
                "usage_role": "SCIENTIFIC_FEATURE",
                "usage_status": "ANALYZED",
                "rationale": "Aggregate training loss by epoch.",
                "files_profiled": 240,
                "runs_profiled": 240,
                "values_observed": 48_000,
                "non_null_count": 48_000,
                "null_count": 0,
                "unique_count": 47_000,
                "constant_state": "NON_CONSTANT",
                "derived_consumer": "training_telemetry",
                "silently_dropped": False,
            },
            {
                "normalized_path": "<NOT_COLLECTED>/evaluation",
                "field_path": "no_replay_arm",
                "source_kind": "NOT_COLLECTED",
                "files_observed": 0,
                "runs_observed": 0,
                "files_expected": 0,
                "runs_expected": 0,
                "schema_variant_count": 0,
                "usage_role": "NOT_COLLECTED_FIELD",
                "usage_status": "NOT_TESTABLE",
                "rationale": "No no-replay arm was collected.",
                "files_profiled": 0,
                "runs_profiled": 0,
                "values_observed": 0,
                "non_null_count": 0,
                "null_count": 0,
                "unique_count": 0,
                "constant_state": "NO_VALUES",
                "derived_consumer": "",
                "silently_dropped": False,
            },
        ]
    )


def _source_file_ledger() -> pd.DataFrame:
    rows = []
    for run in range(1, 241):
        rows.extend(
            [
                {
                    "run_slot": f"RUN_{run:03d}",
                    "normalized_path": "02_logs/epoch_training_metrics.csv",
                    "relative_path": "02_logs/epoch_training_metrics.csv",
                    "size_bytes": 20_000,
                    "canonical_attempt": True,
                    "artifact_manifest_listed": True,
                    "artifact_manifest_size_match": True,
                    "artifact_manifest_sha256": "A" * 64,
                },
                {
                    "run_slot": f"RUN_{run:03d}",
                    "normalized_path": (
                        "01_manifests/frozen_inputs/base_train_normal_manifest.csv"
                    ),
                    "relative_path": (
                        "01_manifests/frozen_inputs/base_train_normal_manifest.csv"
                    ),
                    "size_bytes": 1_000_000,
                    "canonical_attempt": True,
                    "artifact_manifest_listed": True,
                    "artifact_manifest_size_match": True,
                    "artifact_manifest_sha256": "B" * 64,
                },
                {
                    "run_slot": f"RUN_{run:03d}",
                    "normalized_path": "03_checkpoints/last.pt",
                    "relative_path": "03_checkpoints/last.pt",
                    "size_bytes": 70_000_000,
                    "canonical_attempt": True,
                    "artifact_manifest_listed": True,
                    "artifact_manifest_size_match": True,
                    "artifact_manifest_sha256": "C" * 64,
                },
                {
                    "run_slot": f"RUN_{run:03d}",
                    "normalized_path": "04_predictions/val_op_predictions.csv",
                    "relative_path": "04_predictions/val_op_predictions.csv",
                    "size_bytes": 9_000_000,
                    "canonical_attempt": True,
                    "artifact_manifest_listed": True,
                    "artifact_manifest_size_match": True,
                    "artifact_manifest_sha256": "D" * 64,
                },
            ]
        )
    return pd.DataFrame(rows)


def _write_minimal_oof_tree(root: Path) -> None:
    jobs = root / "01_oof_dynamics/jobs/node/fold_00"
    summary = root / "01_oof_dynamics/summary"
    value_dir = root / "02_sample_value_tables"
    jobs.mkdir(parents=True)
    summary.mkdir(parents=True)
    value_dir.mkdir(parents=True)

    raw = pd.DataFrame(
        [
            {
                "sample_id": "sample-a",
                "checkpoint_id": "fold_00_epoch_001",
                "oof_fold": 0,
                "epoch": 1,
                "y_true": 0,
                "p_defect_raw": 0.2,
                "raw_cross_entropy": 0.22,
            },
            {
                "sample_id": "sample-b",
                "checkpoint_id": "fold_00_epoch_001",
                "oof_fold": 0,
                "epoch": 1,
                "y_true": 1,
                "p_defect_raw": 0.8,
                "raw_cross_entropy": 0.22,
            },
        ]
    )
    raw_path = jobs / "epoch_001_predictions.csv"
    raw.to_csv(raw_path, index=False)
    (raw_path.with_name(raw_path.name + ".manifest.json")).write_text(
        json.dumps({"status": "complete", "row_count": 2}), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "relative_path": raw_path.relative_to(
                    root / "01_oof_dynamics"
                ).as_posix(),
                "row_count": 2,
            }
        ]
    ).to_csv(summary / "summary_input_manifest.csv", index=False)
    pd.DataFrame(
        [
            {
                "sample_id": "sample-a",
                "y_true": 0,
                "oof_fold": 0,
                "epoch_count": 1,
                "mean_p_defect": 0.2,
                "forgetting_count": 0,
                "dynamic_bucket": "easy_clean",
            },
            {
                "sample_id": "sample-b",
                "y_true": 1,
                "oof_fold": 0,
                "epoch_count": 1,
                "mean_p_defect": 0.8,
                "forgetting_count": 0,
                "dynamic_bucket": "easy_clean",
            },
        ]
    ).to_csv(summary / "sample_dynamics_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "sample_id": "sample-a",
                "y_true": 0,
                "gap_critical_score": 0.1,
                "grad_mag_score": pd.NA,
            },
            {
                "sample_id": "sample-b",
                "y_true": 1,
                "gap_critical_score": pd.NA,
                "grad_mag_score": pd.NA,
            },
        ]
    ).to_csv(value_dir / "sample_value_table.csv", index=False)
    (summary / "summary_validation.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "input_file_count": 1,
                "expected_folds": ["00"],
                "expected_epochs": [1],
                "missing_pairs": [],
                "duplicate_pairs": [],
                "sample_count": 2,
                "total_prediction_rows": 2,
            }
        ),
        encoding="utf-8",
    )


def test_discovers_complete_oof_raw_summary_and_value_fields(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    _write_minimal_oof_tree(root)

    fields = discover_oof_field_inventory(root)

    assert {
        "oof_raw_epoch_predictions",
        "oof_sample_dynamics_summary",
        "oof_sample_value_table",
    }.issubset(set(fields["logical_source"]))
    raw = fields[fields["logical_source"] == "oof_raw_epoch_predictions"]
    assert set(raw["field_path"]) >= {
        "sample_id",
        "epoch",
        "p_defect_raw",
        "raw_cross_entropy",
    }
    assert raw["files_observed"].eq(1).all()
    assert raw["values_observed"].eq(2).all()
    assert raw["path_total_bytes"].gt(0).all()


def test_oof_discovery_rejects_incomplete_validation(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    _write_minimal_oof_tree(root)
    validation = root / "01_oof_dynamics/summary/summary_validation.json"
    payload = json.loads(validation.read_text(encoding="utf-8"))
    payload["missing_pairs"] = ["fold_00/epoch_001"]
    validation.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CampaignAuditError, match="OOF validation is incomplete"):
        discover_oof_field_inventory(root)


def test_inventory_has_decision_metadata_and_exposes_p0_gaps(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    _write_minimal_oof_tree(root)
    oof = discover_oof_field_inventory(root)

    inventory = build_campaign_field_inventory(
        _refined_ledger(), _source_file_ledger(), supplementary_fields=oof
    )
    gates = assert_campaign_field_inventory_complete(inventory)

    assert gates["duplicate_field_identities"] == 0
    assert gates["missing_decision_metadata"] == 0
    assert set(
        [
            "canonical_field_id",
            "definition",
            "raw_or_derived",
            "granularity",
            "time_scope",
            "collection_frequency",
            "coverage_rate",
            "value_null_rate",
            "unit",
            "reproducibility",
            "leakage_risk",
            "hypothesis_ids",
            "collection_cost_class",
            "priority",
            "priority_reason",
            "gap_status",
        ]
    ).issubset(inventory.columns)
    p0_gaps = inventory[
        (inventory["priority"] == "P0")
        & inventory["gap_status"].isin(["MISSING_REQUIRED", "NOT_COLLECTED_REQUIRED"])
    ]
    assert {
        "no_replay_arm",
        "blind_or_external_test",
        "per_sample_replay_exposure_count",
        "key_epoch_val_op_raw_predictions",
        "base_vs_replay_loss_split",
        "weak_defect_tail_score_trajectory",
    }.issubset(set(p0_gaps["field_path"]))


def test_volume_forecast_counts_paths_once_and_flags_gradient_unknown() -> None:
    forecast = build_data_volume_forecast(
        _source_file_ledger(),
        seed_counts=(14, 22, 30),
        arm_count=6,
        key_checkpoint_epochs=(120, 140, 150, 160, 180, 200),
    )

    assert forecast["run_count"].tolist() == [84, 132, 180]
    assert forecast["gradient_payload_status"].eq(
        "UNMEASURED_PILOT_REQUIRED"
    ).all()
    assert forecast["estimate_class"].eq(
        "LOWER_BOUND_EXCLUDES_GRADIENT_PAYLOAD"
    ).all()
    assert forecast["shared_immutable_bytes_once"].eq(1_000_000).all()
    assert forecast["shared_immutable_bytes_avoided"].tolist() == [
        83_000_000,
        131_000_000,
        179_000_000,
    ]
    expected_per_run = 20_000 + (6 * 70_000_000) + (6 * 9_000_000)
    assert forecast["projected_run_specific_bytes_per_run"].eq(expected_per_run).all()


def test_publish_writes_required_artifacts_and_hash_lineage(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    _write_minimal_oof_tree(root)
    refined_path = tmp_path / "DATA_USAGE_LEDGER_REFINED.csv"
    source_path = tmp_path / "source_file_ledger.csv"
    _refined_ledger().to_csv(refined_path, index=False)
    _source_file_ledger().to_csv(source_path, index=False)
    output = tmp_path / "campaign/01_field_audit"

    result = publish_campaign_field_audit(
        refined_ledger_path=refined_path,
        source_file_ledger_path=source_path,
        oof_experiment_root=root,
        output_dir=output,
        campaign_id="dynamic_replay_budget_efficiency_20260807",
        seed_counts=(14,),
        arm_count=6,
    )

    expected = {
        "FIELD_INVENTORY.csv",
        "FIELD_GAP_REPORT.md",
        "DATA_LINEAGE.json",
        "DATA_VOLUME_FORECAST.csv",
        "RETENTION_POLICY.md",
        "AUDIT_VALIDATION.json",
    }
    assert expected == {path.name for path in output.iterdir()}
    assert result["status"] == "complete"
    lineage = json.loads((output / "DATA_LINEAGE.json").read_text(encoding="utf-8"))
    assert lineage["campaign_id"] == "dynamic_replay_budget_efficiency_20260807"
    assert all(item["sha256"] for item in lineage["inputs"])
    assert all(item["sha256"] for item in lineage["outputs"])
    report = (output / "FIELD_GAP_REPORT.md").read_text(encoding="utf-8")
    assert "P0" in report
    assert "no_replay_arm" in report
    policy = (output / "RETENTION_POLICY.md").read_text(encoding="utf-8")
    assert "120, 140, 150, 160, 180, 200" in policy
    assert "shared immutable" in policy.lower()
