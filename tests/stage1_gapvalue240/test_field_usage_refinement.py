from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import stage1_gapvalue240.field_usage_refinement as refinement_module
from stage1_gapvalue240.field_usage_refinement import (
    FieldUsageRefinementError,
    assert_refined_ledger_complete,
    build_field_value_profiles,
    publish_refined_field_usage,
    refine_usage_ledger,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def _source_row(run_slot: str, attempt: Path, relative: str, kind: str) -> dict:
    path = attempt / relative
    return {
        "run_slot": run_slot,
        "attempt_dir": str(attempt),
        "relative_path": relative,
        "normalized_path": relative,
        "source_kind": kind,
        "artifact_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
    }


def _usage_rows(normalized_path: str, kind: str, fields: list[str], runs: int = 2) -> list[dict]:
    return [
        {
            "normalized_path": normalized_path,
            "field_path": field,
            "source_kind": kind,
            "files_observed": runs,
            "runs_observed": runs,
            "files_expected": runs,
            "runs_expected": runs,
            "schema_variant_count": 1,
            "usage_role": "CONFOUND_CONTROL",
            "usage_status": "ANALYZED",
            "matched_rule_id": "fixture",
            "rationale": "fixture",
        }
        for field in fields
    ]


def _fixture_ledgers(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source: list[dict] = []
    usage: list[dict] = []
    for index, run_slot in enumerate(("RUN_001", "RUN_002"), start=1):
        attempt = tmp_path / run_slot
        controller = {
            "pid": 1000 + index,
            "platform": "Windows-11",
            "python": "3.11.9",
        }
        _write_json(attempt / "00_identity/environment_controller.json", controller)
        source.append(_source_row(run_slot, attempt, "00_identity/environment_controller.json", "JSON"))

        training_env = {
            "actual": {"pytorch": "2.7.1", "cuda_build": "12.8"},
            "status": "PASS",
        }
        _write_json(attempt / "00_identity/environment_training.json", training_env)
        source.append(_source_row(run_slot, attempt, "00_identity/environment_training.json", "JSON"))

        args = {
            "epochs": 200,
            "batch": 128,
            "seed": index,
            "data": f"C:/private/node{index}/dataset.yaml",
            "resume": bool(index == 2),
            "warmup_bias_lr": 0.0 if index == 1 else 0.1,
        }
        _write_yaml(attempt / "02_logs/args.yaml", args)
        source.append(_source_row(run_slot, attempt, "02_logs/args.yaml", "YAML"))

        epoch = pd.DataFrame(
            {
                "epoch": [1, 2],
                "metrics/accuracy_top5": [1.0, 1.0],
                "train/loss": [0.3 - index / 100, 0.2 - index / 100],
                "time": [1.0, 2.0],
            }
        )
        epoch_path = attempt / "02_logs/epoch_training_metrics.csv"
        epoch_path.parent.mkdir(parents=True, exist_ok=True)
        epoch.to_csv(epoch_path, index=False)
        source.append(_source_row(run_slot, attempt, "02_logs/epoch_training_metrics.csv", "CSV"))

        gpu = pd.DataFrame(
            {"gpu_util_pct": [80 + index], "memory_used_mb": [10000 + index]}
        )
        gpu_path = attempt / "02_logs/gpu_usage_{TIMESTAMP}_{TOKEN}.csv"
        gpu.to_csv(gpu_path, index=False)
        source.append(_source_row(run_slot, attempt, "02_logs/gpu_usage_{TIMESTAMP}_{TOKEN}.csv", "CSV"))

        selection = pd.DataFrame(
            {
                "run_slot": [run_slot],
                "sample_id": [f"Det/images/private_{index}.png"],
                "dynamic_bucket": ["learnable_hard"],
                "mean_p_defect": [0.2 + index / 10],
            }
        )
        selection_path = attempt / "01_manifests/selection_manifest.csv"
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        selection.to_csv(selection_path, index=False)
        source.append(_source_row(run_slot, attempt, "01_manifests/selection_manifest.csv", "CSV"))

        predictions = pd.DataFrame(
            {
                "sample_id": [f"val/private_{index}.png", f"val/private_{index + 2}.png"],
                "y_true": [0, 1],
                "score": [0.1, 0.9],
                "score_raw": [0.2, 0.8],
            }
        )
        prediction_path = attempt / "04_predictions/val_op_predictions.csv"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(prediction_path, index=False)
        source.append(_source_row(run_slot, attempt, "04_predictions/val_op_predictions.csv", "CSV"))

        postflight = {"recomputed": {"TN_at_FN95": {"actual_TN": 68000 + index}}, "status": "PASS"}
        _write_json(attempt / "07_validation/postflight_report.json", postflight)
        source.append(_source_row(run_slot, attempt, "07_validation/postflight_report.json", "JSON"))

        checkpoint = attempt / "03_checkpoints/last.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint" + bytes([index]))
        source.append(_source_row(run_slot, attempt, "03_checkpoints/last.pt", "CHECKPOINT"))

    usage.extend(_usage_rows("00_identity/environment_controller.json", "JSON", ["pid", "platform", "python"]))
    usage.extend(_usage_rows("00_identity/environment_training.json", "JSON", ["actual.pytorch", "actual.cuda_build", "status"]))
    usage.extend(_usage_rows("02_logs/args.yaml", "YAML", ["epochs", "batch", "seed", "data", "resume", "warmup_bias_lr"]))
    usage.extend(_usage_rows("02_logs/epoch_training_metrics.csv", "CSV", ["epoch", "metrics/accuracy_top5", "train/loss", "time"]))
    usage.extend(_usage_rows("02_logs/gpu_usage_{TIMESTAMP}_{TOKEN}.csv", "CSV", ["gpu_util_pct", "memory_used_mb"]))
    usage.extend(_usage_rows("01_manifests/selection_manifest.csv", "CSV", ["run_slot", "sample_id", "dynamic_bucket", "mean_p_defect"]))
    usage.extend(_usage_rows("04_predictions/val_op_predictions.csv", "CSV", ["sample_id", "y_true", "score", "score_raw"]))
    usage.extend(_usage_rows("07_validation/postflight_report.json", "JSON", ["recomputed.TN_at_FN95.actual_TN", "status"]))
    usage.extend(_usage_rows("03_checkpoints/last.pt", "CHECKPOINT", ["<BINARY_CHECKPOINT_PAYLOAD>"]))
    usage.extend(
        [
            {
                **_usage_rows("<NOT_COLLECTED>/gradients", "NOT_COLLECTED", ["grad_align_score"], runs=0)[0],
                "usage_role": "NOT_COLLECTED_FIELD",
                "usage_status": "NOT_TESTABLE",
            },
            {
                **_usage_rows("<MISSING_FROM_ATTEMPT_SELECTION>/selection_manifest.csv", "MISSING_FROM_ATTEMPT", ["method_raw_score"], runs=0)[0],
                "usage_role": "MISSING_FIELD",
                "usage_status": "DOCUMENTED_MISSING",
            },
        ]
    )
    return pd.DataFrame(source), pd.DataFrame(usage)


def test_value_profiles_prove_constants_ranges_and_redact_paths(tmp_path: Path) -> None:
    source, usage = _fixture_ledgers(tmp_path)
    profiles = build_field_value_profiles(source, usage, csv_chunksize=1, max_workers=1)
    by_key = profiles.set_index(["normalized_path", "field_path"])

    top5 = by_key.loc[("02_logs/epoch_training_metrics.csv", "metrics/accuracy_top5")]
    assert top5["values_observed"] == 4
    assert top5["non_null_count"] == 4
    assert top5["unique_count"] == 1
    assert bool(top5["unique_count_exact"])
    assert top5["constant_state"] == "CONSTANT"
    assert top5["numeric_min"] == 1.0
    assert top5["numeric_max"] == 1.0

    loss = by_key.loc[("02_logs/epoch_training_metrics.csv", "train/loss")]
    assert loss["constant_state"] == "NON_CONSTANT"
    assert loss["numeric_min"] == pytest.approx(0.18)
    assert loss["numeric_max"] == pytest.approx(0.29)

    data = by_key.loc[("02_logs/args.yaml", "data")]
    assert "C:/private" not in data["redacted_examples_json"]
    assert "<PATH:" in data["redacted_examples_json"]
    assert len(data["value_digest_sha256"]) == 64

    missing = by_key.loc[("<NOT_COLLECTED>/gradients", "grad_align_score")]
    assert missing["scan_status"] == "DOCUMENTED_ABSENT"
    assert missing["values_observed"] == 0


def test_refinement_distinguishes_constant_parameters_and_varying_semantics(tmp_path: Path) -> None:
    source, usage = _fixture_ledgers(tmp_path)
    profiles = build_field_value_profiles(source, usage, max_workers=1)
    refined = refine_usage_ledger(
        usage,
        profiles,
        verified_derived_outputs={
            "checkpoint_drift",
            "selection_mechanisms",
            "raw_frontier",
            "val_cal",
            "resource_reliability",
            "training_telemetry",
        },
    ).set_index(["normalized_path", "field_path"])

    assert refined.loc[("00_identity/environment_controller.json", "platform"), "usage_role"] == "CONSTANT_PARAMETER"
    assert refined.loc[("00_identity/environment_controller.json", "pid"), "usage_role"] == "DESCRIPTIVE_FIELD"
    assert refined.loc[("00_identity/environment_training.json", "actual.pytorch"), "usage_role"] == "CONSTANT_PARAMETER"
    assert refined.loc[("02_logs/args.yaml", "epochs"), "usage_role"] == "CONSTANT_PARAMETER"
    assert refined.loc[("02_logs/args.yaml", "seed"), "usage_role"] == "PAIRING_KEY"
    assert refined.loc[("02_logs/args.yaml", "data"), "usage_role"] == "LINEAGE_VALIDATION"
    assert refined.loc[("02_logs/args.yaml", "resume"), "usage_role"] == "CONFOUND_CONTROL"
    assert refined.loc[("02_logs/args.yaml", "warmup_bias_lr"), "usage_role"] == "SCIENTIFIC_FEATURE"
    assert refined.loc[("02_logs/epoch_training_metrics.csv", "metrics/accuracy_top5"), "usage_role"] == "CONSTANT_PARAMETER"
    assert refined.loc[("02_logs/epoch_training_metrics.csv", "train/loss"), "usage_role"] == "SCIENTIFIC_FEATURE"
    assert refined.loc[("02_logs/gpu_usage_{TIMESTAMP}_{TOKEN}.csv", "gpu_util_pct"), "usage_role"] == "CONFOUND_CONTROL"
    assert refined.loc[("02_logs/gpu_usage_{TIMESTAMP}_{TOKEN}.csv", "gpu_util_pct"), "derived_consumer"] == "resource_reliability"
    assert refined.loc[("01_manifests/selection_manifest.csv", "sample_id"), "usage_role"] == "PAIRING_KEY"
    assert refined.loc[("01_manifests/selection_manifest.csv", "dynamic_bucket"), "usage_role"] == "SCIENTIFIC_FEATURE"
    assert refined.loc[("04_predictions/val_op_predictions.csv", "score_raw"), "usage_role"] == "OUTCOME_METRIC"
    assert refined.loc[("04_predictions/val_op_predictions.csv", "score_raw"), "derived_consumer"] == "raw_frontier"
    assert refined.loc[("07_validation/postflight_report.json", "recomputed.TN_at_FN95.actual_TN"), "usage_role"] == "RECOMPUTED_FIELD"
    checkpoint = refined.loc[("03_checkpoints/last.pt", "<BINARY_CHECKPOINT_PAYLOAD>")]
    assert checkpoint["usage_role"] == "SCIENTIFIC_FEATURE"
    assert checkpoint["usage_status"] == "ANALYZED"
    assert checkpoint["derived_consumer"] == "checkpoint_drift"
    assert refined.loc[("<NOT_COLLECTED>/gradients", "grad_align_score"), "usage_role"] == "NOT_COLLECTED_FIELD"
    assert refined.loc[("<MISSING_FROM_ATTEMPT_SELECTION>/selection_manifest.csv", "method_raw_score"), "usage_role"] == "MISSING_FIELD"
    assert assert_refined_ledger_complete(refined.reset_index()) == {
        "UNREVIEWED": 0,
        "UNCLASSIFIED": 0,
        "SILENTLY_DROPPED": 0,
    }


def test_profile_fails_if_structured_field_has_no_values(tmp_path: Path) -> None:
    source, usage = _fixture_ledgers(tmp_path)
    usage = pd.concat(
        [usage, pd.DataFrame(_usage_rows("02_logs/args.yaml", "YAML", ["unexpected_field"]))],
        ignore_index=True,
    )
    with pytest.raises(FieldUsageRefinementError, match="structured fields were not profiled"):
        build_field_value_profiles(source, usage, max_workers=1)


def test_unique_count_can_be_bounded_without_weakening_constant_proof(tmp_path: Path) -> None:
    source, usage = _fixture_ledgers(tmp_path)
    profiles = build_field_value_profiles(
        source, usage, max_workers=1, max_exact_unique=2
    ).set_index(["normalized_path", "field_path"])
    sample = profiles.loc[("04_predictions/val_op_predictions.csv", "sample_id")]
    assert not bool(sample["unique_count_exact"])
    assert pd.isna(sample["unique_count"])
    assert sample["unique_count_lower_bound"] == 3
    assert sample["constant_state"] == "NON_CONSTANT"


def test_csv_profiling_redacts_only_distinct_examples_not_every_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "RUN_001"
    path = attempt / "large.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"label": ["normal"] * 50_000 + ["defect"] * 50_000}).to_csv(
        path, index=False
    )
    source = pd.DataFrame(
        [
            {
                "run_slot": "RUN_001",
                "attempt_dir": str(attempt),
                "relative_path": "large.csv",
                "normalized_path": "large.csv",
                "source_kind": "CSV",
                "artifact_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
            }
        ]
    )
    usage = pd.DataFrame(_usage_rows("large.csv", "CSV", ["label"], runs=1))
    calls = 0
    original = refinement_module._redact_example

    def counting_redaction(value: object, field_path: str) -> str:
        nonlocal calls
        calls += 1
        return original(value, field_path)

    monkeypatch.setattr(refinement_module, "_redact_example", counting_redaction)
    profile = build_field_value_profiles(
        source, usage, max_workers=1, csv_chunksize=100_000
    ).iloc[0]
    assert calls <= 2
    assert profile["values_observed"] == 100_000
    assert profile["unique_count"] == 2
    assert profile["constant_state"] == "NON_CONSTANT"


def test_publisher_is_atomic_requires_inprogress_and_does_not_touch_state(tmp_path: Path) -> None:
    source, usage = _fixture_ledgers(tmp_path)
    source_path = tmp_path / "source.csv"
    usage_path = tmp_path / "usage.csv"
    source.to_csv(source_path, index=False)
    usage.to_csv(usage_path, index=False)
    output = tmp_path / "analysis.inprogress"

    result = publish_refined_field_usage(
        source_path,
        usage_path,
        output,
        max_workers=1,
        verified_derived_outputs={"checkpoint_drift"},
    )
    assert result["status"] == "PASS"
    assert (output / "audit/FIELD_VALUE_PROFILES.csv").is_file()
    assert (output / "audit/DATA_USAGE_LEDGER_REFINED.csv").is_file()
    assert not (output / "ANALYSIS_STATE.json").exists()
    assert not (output / "manifest.json").exists()
    assert not list(output.rglob("*.tmp"))

    with pytest.raises(FileExistsError):
        publish_refined_field_usage(source_path, usage_path, output, max_workers=1)
    with pytest.raises(FieldUsageRefinementError, match="inprogress"):
        publish_refined_field_usage(
            source_path, usage_path, tmp_path / "final", max_workers=1
        )
