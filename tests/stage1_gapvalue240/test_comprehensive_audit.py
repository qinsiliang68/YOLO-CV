from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from stage1_gapvalue240.comprehensive_audit import (
    DataCoverageError,
    UsageRule,
    apply_usage_rules,
    audit_epoch_grid,
    assert_complete_usage_ledger,
    build_field_coverage,
    build_canonical_file_inventory,
    default_usage_rules,
    extract_structured_field_inventory,
    normalize_canonical_relative_path,
)
from stage1_gapvalue240.goal_pipeline import publish_evidence_foundation


def test_normalizes_tokenized_and_manual_recovery_artifacts() -> None:
    cases = {
        "02_logs/train_20260712T115558_c05653f5.log": (
            "02_logs/train_{TIMESTAMP}_{TOKEN}.log"
        ),
        "02_logs/train_20260712T115558_c05653f5.log.result.json": (
            "02_logs/train_{TIMESTAMP}_{TOKEN}.log.result.json"
        ),
        "02_logs/gpu_usage_20260712T115558_c05653f5.csv": (
            "02_logs/gpu_usage_{TIMESTAMP}_{TOKEN}.csv"
        ),
        "02_logs/segment_20260713T021730_b2a6ce1a_val_op.log": (
            "02_logs/segment_{TIMESTAMP}_{TOKEN}_{SPLIT}.log"
        ),
        "07_validation/checkpoint_probe_best_0b8d2f0f.json": (
            "07_validation/checkpoint_probe_{CHECKPOINT}_{TOKEN}.json"
        ),
        "02_logs/checkpoint_probe_manual_recovery_20260716T2139.stderr.log": (
            "02_logs/checkpoint_probe_manual_recovery_{TIMESTAMP}.{STREAM}.log"
        ),
        "07_validation/training_execution_audit.pre_repair_20260717T1040.json": (
            "07_validation/training_execution_audit.pre_repair_{TIMESTAMP}.json"
        ),
        "02_logs/.train_20260715T141944_5068acb4.log.result.json.pua_s1nt.tmp": (
            "02_logs/.train_{TIMESTAMP}_{TOKEN}.log.result.json.{TOKEN}.tmp"
        ),
    }
    for relative_path, expected in cases.items():
        assert normalize_canonical_relative_path(relative_path) == expected


def test_inventory_uses_only_the_attempt_named_by_canonical_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "extracted"
    canonical = (
        root
        / "stage1_gapvalue240_P35_upload"
        / "runs"
        / "RUN_001"
        / "attempt_canonical"
    )
    alternate = canonical.parent / "attempt_debug_newer"
    (canonical / "02_logs").mkdir(parents=True)
    (alternate / "02_logs").mkdir(parents=True)
    (canonical / "02_logs/args.yaml").write_text("epochs: 200\n", encoding="utf-8")
    (alternate / "02_logs/should_not_be_seen.json").write_text("{}", encoding="utf-8")
    inventory_path = tmp_path / "inventory.csv"
    pd.DataFrame(
        [
            {
                "run_slot": "RUN_001",
                "package": "P35",
                "attempt_id": "attempt_canonical",
            }
        ]
    ).to_csv(inventory_path, index=False)

    files = build_canonical_file_inventory(root, inventory_path)

    assert files["run_slot"].tolist() == ["RUN_001"]
    assert files["relative_path"].tolist() == ["02_logs/args.yaml"]
    assert files["canonical_attempt"].all()
    assert not files["relative_path"].str.contains("should_not_be_seen").any()


def test_extracts_csv_json_yaml_and_unstructured_content_units(tmp_path: Path) -> None:
    csv_path = tmp_path / "table.csv"
    pd.DataFrame([{"sample_id": "x", "score": 0.5}]).to_csv(csv_path, index=False)
    json_path = tmp_path / "nested.json"
    json_path.write_text(
        json.dumps(
            {
                "optimizer": {"name": "MuSGD", "groups": [{"lr": 0.1}]},
                "issues": [],
            }
        ),
        encoding="utf-8",
    )
    yaml_path = tmp_path / "args.yaml"
    yaml_path.write_text("epochs: 200\naugment:\n  mixup: 0.0\n", encoding="utf-8")
    log_path = tmp_path / "train.log"
    log_path.write_text("optimizer: MuSGD\n", encoding="utf-8")

    csv_fields = extract_structured_field_inventory(csv_path, "table.csv")
    json_fields = extract_structured_field_inventory(json_path, "nested.json")
    yaml_fields = extract_structured_field_inventory(yaml_path, "args.yaml")
    log_fields = extract_structured_field_inventory(log_path, "train.log")

    assert set(csv_fields["field_path"]) == {"sample_id", "score"}
    assert set(json_fields["field_path"]) == {
        "optimizer.name",
        "optimizer.groups[].lr",
        "issues[]",
    }
    assert set(yaml_fields["field_path"]) == {"epochs", "augment.mixup"}
    assert log_fields["field_path"].tolist() == ["<UNSTRUCTURED_TEXT>"]


def test_usage_ledger_gate_rejects_unknown_or_silently_dropped_fields() -> None:
    valid = pd.DataFrame(
        [
            {
                "normalized_path": "02_logs/epoch_training_metrics.csv",
                "field_path": "lr/pg0",
                "usage_role": "SCIENTIFIC_FEATURE",
                "usage_status": "ANALYZED",
            },
            {
                "normalized_path": "00_identity/run_identity.json",
                "field_path": "release_commit",
                "usage_role": "LINEAGE_VALIDATION",
                "usage_status": "AUDITED",
            },
            {
                "normalized_path": "03_checkpoints/best.pt",
                "field_path": "optimizer",
                "usage_role": "MISSING_FIELD",
                "usage_status": "DOCUMENTED_MISSING",
            },
        ]
    )
    summary = assert_complete_usage_ledger(valid)
    assert summary == {
        "UNREVIEWED": 0,
        "UNCLASSIFIED": 0,
        "SILENTLY_DROPPED": 0,
    }

    invalid = pd.concat(
        [
            valid,
            pd.DataFrame(
                [
                    {
                        "normalized_path": "02_logs/epoch_training_metrics.csv",
                        "field_path": "metrics/accuracy_top5",
                        "usage_role": "UNREVIEWED",
                        "usage_status": "SILENTLY_DROPPED",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(DataCoverageError, match="UNREVIEWED=1.*SILENTLY_DROPPED=1"):
        assert_complete_usage_ledger(invalid)


def test_usage_ledger_gate_rejects_duplicate_file_field_identity() -> None:
    ledger = pd.DataFrame(
        [
            {
                "normalized_path": "05_metrics/threshold_sweep.csv",
                "field_path": "threshold",
                "usage_role": "OUTCOME_METRIC",
                "usage_status": "ANALYZED",
            },
            {
                "normalized_path": "05_metrics/threshold_sweep.csv",
                "field_path": "threshold",
                "usage_role": "RECOMPUTED_FIELD",
                "usage_status": "ANALYZED",
            },
        ]
    )
    with pytest.raises(DataCoverageError, match="duplicate ledger identities"):
        assert_complete_usage_ledger(ledger)


def test_usage_rules_are_specific_first_and_leave_unknown_visible() -> None:
    fields = pd.DataFrame(
        [
            {
                "normalized_path": "02_logs/epoch_training_metrics.csv",
                "field_path": "lr/pg0",
                "source_kind": "CSV",
            },
            {
                "normalized_path": "02_logs/epoch_training_metrics.csv",
                "field_path": "epoch",
                "source_kind": "CSV",
            },
            {
                "normalized_path": "new/evidence.json",
                "field_path": "surprise",
                "source_kind": "JSON",
            },
        ]
    )
    rules = [
        UsageRule(
            rule_id="epoch-key",
            file_pattern=r"^02_logs/epoch_training_metrics\.csv$",
            field_pattern=r"^epoch$",
            usage_role="PAIRING_KEY",
            usage_status="AUDITED",
            rationale="Within-run epoch identity.",
        ),
        UsageRule(
            rule_id="epoch-science",
            file_pattern=r"^02_logs/epoch_training_metrics\.csv$",
            field_pattern=r".*",
            usage_role="SCIENTIFIC_FEATURE",
            usage_status="ANALYZED",
            rationale="Full training telemetry.",
        ),
    ]

    ledger = apply_usage_rules(fields, rules)

    by_field = ledger.set_index("field_path")
    assert by_field.loc["epoch", "matched_rule_id"] == "epoch-key"
    assert by_field.loc["lr/pg0", "matched_rule_id"] == "epoch-science"
    assert by_field.loc["surprise", "usage_role"] == "UNREVIEWED"
    assert by_field.loc["surprise", "usage_status"] == "SILENTLY_DROPPED"
    with pytest.raises(DataCoverageError, match="UNREVIEWED=1"):
        assert_complete_usage_ledger(ledger)


def test_default_rules_classify_science_audit_and_orphan_evidence() -> None:
    fields = pd.DataFrame(
        [
            {
                "normalized_path": "02_logs/epoch_training_metrics.csv",
                "field_path": "metrics/accuracy_top5",
                "source_kind": "CSV",
            },
            {
                "normalized_path": "01_manifests/selection_manifest.csv",
                "field_path": "training_seed",
                "source_kind": "CSV",
            },
            {
                "normalized_path": "01_manifests/selection_manifest.csv",
                "field_path": "dynamic_bucket",
                "source_kind": "CSV",
            },
            {
                "normalized_path": "04_predictions/val_op_predictions.csv",
                "field_path": "score_raw",
                "source_kind": "CSV",
            },
            {
                "normalized_path": (
                    "07_validation/training_execution_audit.pre_repair_"
                    "{TIMESTAMP}.json"
                ),
                "field_path": "repair_reason",
                "source_kind": "JSON",
            },
            {
                "normalized_path": (
                    "02_logs/.train_{TIMESTAMP}_{TOKEN}.log.result.json."
                    "{TOKEN}.tmp"
                ),
                "field_path": "<ORPHAN_TEMP_PAYLOAD>",
                "source_kind": "TEMPORARY",
            },
        ]
    )

    ledger = apply_usage_rules(fields, default_usage_rules()).set_index(
        ["normalized_path", "field_path"]
    )

    assert ledger.loc[
        ("02_logs/epoch_training_metrics.csv", "metrics/accuracy_top5"),
        "usage_role",
    ] == "CONSTANT_PARAMETER"
    assert ledger.loc[
        ("01_manifests/selection_manifest.csv", "training_seed"),
        "usage_role",
    ] == "PAIRING_KEY"
    assert ledger.loc[
        ("01_manifests/selection_manifest.csv", "dynamic_bucket"),
        "usage_role",
    ] == "SCIENTIFIC_FEATURE"
    assert ledger.loc[
        ("04_predictions/val_op_predictions.csv", "score_raw"),
        "usage_role",
    ] == "OUTCOME_METRIC"
    assert ledger.loc[
        (
            "07_validation/training_execution_audit.pre_repair_{TIMESTAMP}.json",
            "repair_reason",
        ),
        "usage_role",
    ] == "LINEAGE_VALIDATION"
    assert ledger.loc[
        (
            "02_logs/.train_{TIMESTAMP}_{TOKEN}.log.result.json.{TOKEN}.tmp",
            "<ORPHAN_TEMP_PAYLOAD>",
        ),
        "usage_status",
    ] == "EXCLUDED_WITH_REASON"
    assert_complete_usage_ledger(ledger.reset_index())


def test_field_coverage_preserves_schema_variants_and_partial_fields(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt"
    first = attempt / "07_validation/probe_a.json"
    second = attempt / "07_validation/probe_b.json"
    first.parent.mkdir(parents=True)
    first.write_text('{"status":"PASS","epoch":199}', encoding="utf-8")
    second.write_text(
        '{"status":"FAIL","error":{"type":"RuntimeError"}}',
        encoding="utf-8",
    )
    files = pd.DataFrame(
        [
            {
                "run_slot": "RUN_001",
                "attempt_dir": str(attempt),
                "relative_path": "07_validation/probe_a.json",
                "normalized_path": "07_validation/probe_{TOKEN}.json",
            },
            {
                "run_slot": "RUN_002",
                "attempt_dir": str(attempt),
                "relative_path": "07_validation/probe_b.json",
                "normalized_path": "07_validation/probe_{TOKEN}.json",
            },
        ]
    )

    result = build_field_coverage(files, max_workers=1)

    assert len(result.file_schemas) == 2
    assert result.file_schemas["schema_signature"].nunique() == 2
    coverage = result.field_coverage.set_index("field_path")
    assert coverage.loc["status", "files_observed"] == 2
    assert coverage.loc["status", "files_expected"] == 2
    assert coverage.loc["epoch", "files_observed"] == 1
    assert coverage.loc["error.type", "files_observed"] == 1
    assert set(coverage["schema_variant_count"]) == {2}


def test_real_240_run_file_and_field_ledger_has_zero_silent_omissions() -> None:
    root = Path(r"C:\baidunetdiskdownload\stage1_gapvalue240_all_uploads_extracted")
    inventory = Path(
        r"C:\baidunetdiskdownload\stage1_gapvalue240_extract_audit_20260728"
        r"\GLOBAL_VALIDATED_RUN_INVENTORY.csv"
    )
    if not root.is_dir() or not inventory.is_file():
        pytest.skip("Local canonical 240-run evidence is not mounted")

    files = build_canonical_file_inventory(root, inventory)
    result = build_field_coverage(files, max_workers=32)
    ledger = apply_usage_rules(result.field_coverage, default_usage_rules())

    assert files["run_slot"].nunique() == 240
    assert len(files) == 12_150
    assert int(files["size_bytes"].sum()) == 84_837_690_736
    assert files["normalized_path"].nunique() == 55
    assert len(result.file_schemas) == 12_150
    assert len(result.field_coverage) == 1_328
    assert result.file_schemas["schema_signature"].nunique() == 43
    assert_complete_usage_ledger(ledger)


def test_epoch_grid_requires_every_saved_telemetry_column(tmp_path: Path) -> None:
    attempts = []
    inventory_rows = []
    for slot, arm in (("RUN_001", "T"), ("RUN_002", "R1")):
        attempt = tmp_path / slot
        path = attempt / "02_logs/epoch_training_metrics.csv"
        path.parent.mkdir(parents=True)
        frame = pd.DataFrame(
            {
                "epoch": [1, 2],
                "time": [1.0, 2.0],
                "train/loss": [0.2, 0.1],
                "metrics/accuracy_top1": [0.8, 0.9],
                "metrics/accuracy_top5": [1.0, 1.0],
                "val/loss": [0.3, 0.2],
                **{f"lr/pg{i}": [0.01, 0.001] for i in range(8)},
            }
        )
        frame.to_csv(path, index=False)
        attempts.append(
            {
                "run_slot": slot,
                "attempt_dir": str(attempt),
                "relative_path": "02_logs/epoch_training_metrics.csv",
                "normalized_path": "02_logs/epoch_training_metrics.csv",
            }
        )
        inventory_rows.append(
            {
                "run_slot": slot,
                "triad_id": "TRIAD_001",
                "arm": arm,
                "training_seed": 11,
                "condition_id": "A01_Test",
                "method": "Test",
                "budget": 2,
            }
        )

    curves = audit_epoch_grid(
        pd.DataFrame(attempts),
        pd.DataFrame(inventory_rows),
        expected_runs=2,
        expected_epochs=2,
    )
    assert len(curves) == 4
    assert set(curves["arm"]) == {"T", "R1"}
    assert set(f"lr/pg{i}" for i in range(8)).issubset(curves.columns)

    broken = Path(attempts[0]["attempt_dir"]) / attempts[0]["relative_path"]
    broken_frame = pd.read_csv(broken).drop(columns=["lr/pg7"])
    broken_frame.to_csv(broken, index=False)
    with pytest.raises(DataCoverageError, match="missing telemetry columns.*lr/pg7"):
        audit_epoch_grid(
            pd.DataFrame(attempts),
            pd.DataFrame(inventory_rows),
            expected_runs=2,
            expected_epochs=2,
        )


def test_evidence_foundation_publishes_atomic_recovery_state(tmp_path: Path) -> None:
    root = tmp_path / "extracted"
    rows = []
    for index, arm in enumerate(("T", "R1", "R2"), start=1):
        slot = f"RUN_{index:03d}"
        attempt = (
            root
            / "stage1_gapvalue240_P35_upload"
            / "runs"
            / slot
            / f"attempt_{arm}"
        )
        logs = attempt / "02_logs"
        logs.mkdir(parents=True)
        pd.DataFrame(
            {
                "epoch": [1, 2],
                "time": [1.0, 2.0],
                "train/loss": [0.2, 0.1],
                "metrics/accuracy_top1": [0.8, 0.9],
                "metrics/accuracy_top5": [1.0, 1.0],
                "val/loss": [0.3, 0.2],
                **{f"lr/pg{i}": [0.01, 0.001] for i in range(8)},
            }
        ).to_csv(logs / "epoch_training_metrics.csv", index=False)
        (logs / "args.yaml").write_text("epochs: 2\n", encoding="utf-8")
        rows.append(
            {
                "run_slot": slot,
                "package": "P35",
                "attempt_id": f"attempt_{arm}",
                "triad_id": "TRIAD_001",
                "arm": arm,
                "training_seed": 11,
                "selection_seed": 22 + index,
                "condition_id": "A01_Test",
                "method": "Test",
                "budget": 2,
            }
        )
    inventory = tmp_path / "inventory.csv"
    pd.DataFrame(rows).to_csv(inventory, index=False)
    output = tmp_path / "goal_report_v1.inprogress"

    state = publish_evidence_foundation(
        extracted_root=root,
        inventory_path=inventory,
        output_dir=output,
        expected_runs=3,
        expected_triads=1,
        expected_epochs=2,
        max_workers=2,
    )

    assert state["status"] == "IN_PROGRESS"
    assert state["gates"]["canonical_runs"] == 3
    assert state["gates"]["triads"] == 1
    assert state["gates"]["epoch_rows"] == 6
    assert state["gates"]["UNREVIEWED"] == 0
    assert (output / "ANALYSIS_STATE.json").is_file()
    assert (output / "audit/source_file_ledger.csv").is_file()
    assert (output / "audit/DATA_USAGE_LEDGER.csv").is_file()
    assert (output / "tables/epoch_training_curves_full.csv").is_file()
    assert (output / "tables/HYPOTHESIS_REGISTRY.csv").is_file()
    contract = yaml.safe_load((output / "analysis_contract.yaml").read_text(encoding="utf-8"))
    assert "same-selection" not in contract["cohorts"]["mixed_or_reversal"]
    assert contract["orthogonal_attributes"]["same_selection_cross_seed_reversal"]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["file_count"] == len(manifest["files"])
    assert all(item["sha256"] for item in manifest["files"])
