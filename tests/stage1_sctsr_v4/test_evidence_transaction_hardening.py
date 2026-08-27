from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from stage1_sctsr_v4 import recovery
from stage1_sctsr_v4.columnar import (
    parquet_engine_available,
    validate_columnar_file,
    write_zstd_parquet,
)
from stage1_sctsr_v4.epoch_transaction import EpochTransaction
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.exposure_ledger import write_exposure_partition
from stage1_sctsr_v4.occurrence_ledger import validate_occurrence_rows, write_occurrence_partition
from stage1_sctsr_v4.selection_ledger import validate_selection_rows
from stage1_sctsr_v4.step_ledger import validate_step_rows
from stage1_sctsr_v4.telemetry import sample_telemetry, validate_telemetry_for_closeout


SHA = "A" * 64


def occurrence_row(*, role: str = "BASE") -> dict[str, object]:
    replay = role == "REPLAY"
    return {
        "run_id": "run-1",
        "parent_id": "parent-1",
        "arm_id": "T_U" if replay else "NR",
        "training_seed": 17,
        "epoch": 121,
        "base_batch_index": 0,
        "global_step_before": 112560,
        "occurrence_role": role,
        "occurrence_index_in_step": 0,
        "sample_id": "sample-1",
        "y_true": 0,
        "replay_role": "T_STRESS" if replay else "NOT_APPLICABLE_BASE",
        "identity_pool_id": "T" if replay else "NOT_APPLICABLE_BASE",
        "identity_group": "G0" if replay else "NOT_APPLICABLE_BASE",
        "selection_policy": "T_STRESS" if replay else "BASE_CANONICAL",
        "selection_reason_code": "PLANNED_REPLAY_STEP_SLOT" if replay else "CANONICAL_BASE_OCCURRENCE",
        "oof_fold": 1,
        "oof_group_id": "bucket-1",
        "oof_group_semantic": "FILENAME_BUCKET_SURROGATE_NOT_TRUE_VIDEO_ID",
        "historical_dynamic_bucket": "stable_high",
        "augmentation_seed": 123,
        "augmentation_trace_digest": SHA,
        "replay_count_before": 0,
        "replay_count_after": 1 if replay else 0,
        "last_replay_epoch": None,
        "last_replay_epoch_reason": "NEVER_REPLAYED" if replay else "NOT_APPLICABLE_BASE",
        "epochs_since_last_replay": None,
        "epochs_since_last_replay_reason": "NEVER_REPLAYED" if replay else "NOT_APPLICABLE_BASE",
        "logit_normal": 1.0,
        "logit_defect": 0.0,
        "p_defect_raw": 0.26894143,
        "ce_unreduced": 0.3132617,
        "margin_defect_minus_normal": -1.0,
        "predicted_label_argmax": 0,
        "correct_argmax": True,
        "oof_reference_probability": None,
        "oof_reference_reason": "REGISTERED_NOT_AVAILABLE",
        "rho_candidate_signal": None,
        "rho_reason": "REGISTERED_NOT_REPORTED",
        "row_generation": 1,
        "planned_replay_epoch": 121 if replay else None,
        "planned_replay_epoch_reason": "PRESENT" if replay else "NOT_APPLICABLE_BASE",
        "planned_step_slot": 0 if replay else None,
        "planned_step_slot_reason": "PRESENT" if replay else "NOT_APPLICABLE_BASE",
        "cumulative_replay_count_before": 0,
        "cumulative_replay_count_after": 1 if replay else 0,
        "pool_multiplicity_target": 16 if replay else 0,
        "schedule_family": "U" if replay else "BASE_CANONICAL",
        "fallback_state": "TARGETED" if replay else "NOT_APPLICABLE_BASE",
    }


def step_row() -> dict[str, object]:
    return {
        "run_id": "run-1",
        "parent_id": "parent-1",
        "arm_id": "T_U",
        "training_seed": 17,
        "epoch": 121,
        "base_batch_index": 0,
        "global_step_before": 112560,
        "global_step_after": 112561,
        "base_batch_size": 128,
        "replay_microbatch_size": 6,
        "replay_rate_numerator": 5,
        "replay_rate_denominator": 1000,
        "base_loss": 0.3,
        "replay_loss": 0.02,
        "combined_loss_for_reporting": 0.32,
        "base_loss_items": {"classification_loss": 0.3},
        "parameter_grad_norm_before_clip": 2.0,
        "parameter_grad_norm_after_clip": 2.0,
        "clip_max_norm": 10.0,
        "clip_reason": "PRESENT",
        "optimizer_step_count_delta": 1,
        "learning_rates": [0.001],
        "optimizer_hyperparameters": [
            {"group_index": 0, "lr": 0.001, "momentum": 0.9, "beta1": None, "beta2": None, "weight_decay": 0.0005}
        ],
        "amp_scale_before": 65536.0,
        "amp_scale_after": 65536.0,
        "amp_reason": "PRESENT",
        "overflow_or_step_skipped": False,
        "ema_updates_before": 112560,
        "ema_updates_after": 112561,
        "scheduler_state_digest": SHA,
        "warmup_progress": 1.0,
        "bn_digest_before_replay": SHA,
        "bn_digest_after_replay_restore": SHA,
        "bn_reason": "PRESENT",
        "rng_digest_before_base": SHA,
        "rng_digest_before_replay": SHA,
        "rng_digest_after_replay_restore": SHA,
        "rng_reason": "PRESENT",
        "replay_rng_fork_digest": SHA,
        "replay_rng_fork_reason": "PRESENT",
        "base_augmentation_digest": SHA,
        "replay_augmentation_digest": SHA,
        "replay_augmentation_reason": "PRESENT",
        "dataloader_wait_seconds": 0.001,
        "base_forward_seconds": 0.01,
        "replay_forward_seconds": 0.005,
        "backward_seconds": 0.02,
        "optimizer_seconds": 0.003,
        "write_buffer_bytes": 0,
        "status": "PASS",
        "row_generation": 1,
    }


def exposure_row() -> dict[str, object]:
    return {
        "run_id": "run-1",
        "parent_id": "parent-1",
        "arm_id": "T_U",
        "training_seed": 17,
        "epoch": 121,
        "denominator_role": "CANONICAL_BASE_OPTIMIZER_VISIBLE_EXPOSURE",
        "base_denominator_planned": 120000,
        "base_denominator_actual": 120000,
        "rate_numerator": 5,
        "rate_denominator": 1000,
        "replay_numerator_planned": 600,
        "replay_numerator_actual": 600,
        "unique_replay_ids": 600,
        "repeat_occurrences": 0,
        "cumulative_occurrences": 600,
        "multiplicity_min": 1,
        "multiplicity_max": 1,
        "multiplicity_mean": 1.0,
        "multiplicity_q0": 1.0,
        "multiplicity_q25": 1.0,
        "multiplicity_q50": 1.0,
        "multiplicity_q75": 1.0,
        "multiplicity_q100": 1.0,
        "base_optimizer_steps_planned": 938,
        "base_optimizer_steps_actual": 938,
        "ema_updates_delta": 938,
        "scheduler_epoch_transitions_delta": 1,
        "base_order_digest": SHA,
        "base_augmentation_digest": SHA,
        "replay_schedule_digest": SHA,
        "identity_pool_digest": SHA,
        "occurrence_partition_sha256": SHA,
        "step_partition_sha256": SHA,
        "telemetry_partition_sha256": SHA,
        "checkpoint_sha256": SHA,
        "write_seconds": 1.0,
        "dataloader_wait_seconds": 2.0,
        "training_seconds": 3.0,
        "evaluation_seconds": 0.0,
        "disk_bytes_written": 100,
        "transaction_generation": 1,
        "validation_status": "PASS",
    }


def selection_row() -> dict[str, object]:
    return {
        "candidate_sample_id": "sample-1",
        "eligibility": True,
        "exclusion_reason": "ELIGIBLE",
        "allowed_strata": "0|stable_high|1|bucket-1",
        "stratum_quota_required": 1,
        "stratum_quota_available": 2,
        "selection_counter_hash": SHA,
        "selected": True,
        "selected_pool": "R2",
        "terminal_field_guard_digest": SHA,
        "terminal_field_status": "TERMINAL_FIELDS_NOT_LOADED",
        "source_row_asset_sha256": SHA,
        "duplicate_overlap_status": "ZERO_OVERLAP",
        "row_generation": 1,
    }


@pytest.mark.parametrize(
    "field",
    [
        "replay_count_before",
        "last_replay_epoch_reason",
        "epochs_since_last_replay_reason",
        "oof_reference_reason",
        "rho_reason",
        "planned_replay_epoch_reason",
        "cumulative_replay_count_after",
        "fallback_state",
    ],
)
def test_sa_130_to_135_occurrence_schema_rejects_every_missing_required_field(field: str):
    row = occurrence_row(role="REPLAY")
    del row[field]
    with pytest.raises(SctsrError) as captured:
        validate_occurrence_rows([row])
    assert captured.value.code is ErrorCode.SCHEMA_VALIDATION_FAILED


def test_sa_134_replay_count_and_last_exposure_are_conserved():
    row = occurrence_row(role="REPLAY")
    row["replay_count_after"] = 2
    with pytest.raises(SctsrError):
        validate_occurrence_rows([row])


def test_sa_142_nullable_occurrence_value_requires_registered_reason():
    row = occurrence_row(role="REPLAY")
    row["last_replay_epoch"] = None
    row["last_replay_epoch_reason"] = "PRESENT"
    with pytest.raises(SctsrError):
        validate_occurrence_rows([row])


@pytest.mark.parametrize(
    "field",
    [
        "replay_rate_numerator",
        "base_loss_items",
        "parameter_grad_norm_before_clip",
        "learning_rates",
        "optimizer_hyperparameters",
        "amp_reason",
        "bn_reason",
        "rng_reason",
        "base_augmentation_digest",
        "dataloader_wait_seconds",
        "row_generation",
    ],
)
def test_sa_131_and_136_step_schema_rejects_missing_required_field(field: str):
    row = step_row()
    del row[field]
    with pytest.raises(SctsrError) as captured:
        validate_step_rows([row])
    assert captured.value.code is ErrorCode.SCHEMA_VALIDATION_FAILED


def test_sa_136_step_rejects_amp_skip_even_when_delta_claims_one():
    row = step_row()
    row["overflow_or_step_skipped"] = True
    with pytest.raises(SctsrError) as captured:
        validate_step_rows([row])
    assert captured.value.code is ErrorCode.OPTIMIZER_STEP_SKIPPED


def test_sa_137_exposure_schema_rejects_missing_cumulative_or_telemetry_sha(tmp_path: Path):
    for field in ("cumulative_occurrences", "multiplicity_q50", "telemetry_partition_sha256"):
        row = exposure_row()
        del row[field]
        with pytest.raises(SctsrError):
            write_exposure_partition([row], tmp_path / f"{field}.parquet")


def test_sa_144_selection_schema_requires_guard_quota_and_overlap():
    for field in ("allowed_strata", "stratum_quota_required", "terminal_field_guard_digest", "duplicate_overlap_status"):
        row = selection_row()
        del row[field]
        with pytest.raises(SctsrError):
            validate_selection_rows([row], r2=True)


def test_sa_138_pyarrow_is_a_locked_runtime_dependency():
    assert parquet_engine_available(), "canonical SCTSR evidence requires real PyArrow/Zstd"


def test_sa_138_to_140_real_parquet_has_strict_schema_zstd_and_partition_manifest(tmp_path: Path):
    path = tmp_path / "occurrence" / "run_id=run-1" / "epoch=0121" / "part-00000.parquet"
    manifest = write_occurrence_partition([occurrence_row()], path)
    report = validate_columnar_file(
        path,
        expected_rows=1,
        expected_schema_version="stage1.sctsr.occurrence_ledger.v1",
        expected_sha256=manifest.sha256,
    )
    assert manifest.canonical_parquet
    assert manifest.storage_format == "PARQUET_ZSTD"
    assert manifest.compression == "ZSTD"
    assert manifest.run_id == "run-1"
    assert manifest.epoch == 121
    assert len(manifest.schema_digest) == 64
    assert report["compression"] == "ZSTD"


def test_sa_139_ledger_writer_rejects_unpartitioned_large_table(tmp_path: Path):
    with pytest.raises(SctsrError):
        write_occurrence_partition([occurrence_row()], tmp_path / "unpartitioned.parquet")


def test_sa_143_empty_and_unknown_sentinels_are_rejected():
    for invalid in ("", "unknown", "UNKNOWN", "同上"):
        row = occurrence_row()
        row["selection_reason_code"] = invalid
        with pytest.raises(SctsrError):
            validate_occurrence_rows([row])


def test_sa_150_cadence_jitter_is_nonfatal_but_ordering_remains_strict(tmp_path: Path):
    row = sample_telemetry(run_id="run-1", arm_id="NR", training_seed=1, epoch=121, run_path=tmp_path, artifact_path=tmp_path)
    validate_telemetry_for_closeout([row, replace(row, monotonic_seconds=row.monotonic_seconds + 3.0)])
    with pytest.raises(SctsrError):
        validate_telemetry_for_closeout([replace(row, monotonic_seconds=row.monotonic_seconds + 3.0), row])


def test_sa_154_to_157_unavailable_hardware_fields_have_reason_not_fake_zero(tmp_path: Path):
    row = sample_telemetry(run_id="run-1", arm_id="NR", training_seed=1, epoch=121, run_path=tmp_path, artifact_path=tmp_path)
    values = asdict(row)
    for provider in ("process", "system", "gpu", "cuda", "disk"):
        assert f"{provider}_provider_status" in values
        assert f"{provider}_provider_reason" in values
        if values[f"{provider}_provider_status"] != "PASS":
            assert values[f"{provider}_provider_reason"] not in (None, "", "unknown", "UNKNOWN")
    gpu_fields = ("gpu_index", "gpu_uuid", "gpu_name", "gpu_utilization", "gpu_memory_used", "gpu_memory_total", "gpu_temperature", "gpu_power")
    if values["gpu_provider_status"] == "NOT_AVAILABLE":
        assert all(values[field] is None for field in gpu_fields)
    else:
        assert all(values[field] is not None for field in gpu_fields)


def test_sa_158_all_critical_providers_unavailable_rejects_closeout(tmp_path: Path):
    row = sample_telemetry(run_id="run-1", arm_id="NR", training_seed=1, epoch=121, run_path=tmp_path, artifact_path=tmp_path)
    bad = replace(
        row,
        process_rss=None,
        system_memory_total=None,
        run_volume_free=None,
        artifact_volume_free=None,
        telemetry_provider_status="FAILED",
    )
    with pytest.raises(SctsrError):
        validate_telemetry_for_closeout([bad])


def test_sa_221_commit_detects_half_written_json_without_optional_validator(tmp_path: Path):
    tx = EpochTransaction(tmp_path / "04_ledgers", "run-1", 121, 1).begin()
    (tx.inprogress / "broken.json").write_bytes(b'{"status":')
    with pytest.raises(SctsrError):
        tx.commit()
    assert not tx.complete.exists()


def test_sa_221_commit_detects_half_written_parquet_without_optional_validator(tmp_path: Path):
    tx = EpochTransaction(tmp_path / "04_ledgers", "run-1", 121, 1).begin()
    (tx.inprogress / "broken.parquet").write_bytes(b"PAR1\x15\x00TRUNCATED")
    with pytest.raises(SctsrError):
        tx.commit()
    assert not tx.complete.exists()


def test_sa_221_generation_manifest_binds_file_sha_and_telemetry_partition(tmp_path: Path):
    tx = EpochTransaction(tmp_path / "04_ledgers", "run-1", 121, 1).begin()
    path = tx.inprogress / "04_ledgers" / "telemetry" / "run_id=run-1" / "epoch=0121" / "part-00000.parquet"
    manifest = write_zstd_parquet(
        [{"run_id": "run-1", "epoch": 121, "value": 1}],
        path,
        schema_version="test.telemetry.v1",
    )
    generation = tx.commit()
    assert any(row["sha256"] == manifest.sha256 for row in generation["files"])
    assert generation["telemetry_partition_sha256"] == manifest.sha256
    pointer = json.loads((tmp_path / "ROLLING_RECOVERY_POINTER.json").read_text(encoding="utf-8"))
    assert pointer["generation_manifest_sha256"] == generation["generation_manifest_sha256"]


def test_sa_227_tampered_generation_manifest_is_rejected_by_recovery_pointer(tmp_path: Path):
    root = tmp_path / "04_ledgers"
    tx = EpochTransaction(root, "run-1", 121, 1).begin()
    tx.write_json("value.json", {"value": 1})
    tx.commit()
    manifest_path = tx.complete / "GENERATION_MANIFEST.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["generation"] = 2
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SctsrError):
        recovery.validate_recovery_pointer(tmp_path / "ROLLING_RECOVERY_POINTER.json")


def test_sa_230_corrupt_latest_complete_generation_is_not_resumable(tmp_path: Path):
    root = tmp_path / "04_ledgers"
    first = EpochTransaction(root, "run-1", 121, 1).begin()
    first.write_json("value.json", {"value": 1})
    first.commit()
    second = EpochTransaction(root, "run-1", 122, 1).begin()
    second.write_json("value.json", {"value": 2})
    second.commit()
    (second.complete / "value.json").write_text("corrupt", encoding="utf-8")
    with pytest.raises(SctsrError):
        recovery.find_last_complete_epoch(root, fail_on_corrupt_latest=True)


def test_sa_229_resume_identity_binds_rng_and_receipt_chain():
    fields = recovery.ResumeIdentity.__dataclass_fields__
    assert "rng_state_digest" in fields
    assert "receipt_chain_digest" in fields


def test_sa_231_repeated_quarantine_never_overwrites_previous_evidence(tmp_path: Path):
    root = tmp_path / "04_ledgers"
    first = EpochTransaction(root, "run-1", 121, 1).begin()
    first.write_json("marker.json", {"attempt": 1})
    first_path = first.abort("OOM")
    second = EpochTransaction(root, "run-1", 121, 1).begin()
    second.write_json("marker.json", {"attempt": 2})
    second_path = second.abort("OOM")
    assert first_path != second_path
    assert json.loads((first_path / "marker.json").read_text(encoding="utf-8"))["attempt"] == 1
    assert json.loads((second_path / "marker.json").read_text(encoding="utf-8"))["attempt"] == 2


def test_sa_232_and_233_checkpoint_retention_keeps_anchors_and_last_two():
    function = getattr(recovery, "retained_checkpoint_epochs", None)
    assert callable(function)
    assert function(range(120, 201)) == {120, 140, 150, 160, 180, 199, 200}
