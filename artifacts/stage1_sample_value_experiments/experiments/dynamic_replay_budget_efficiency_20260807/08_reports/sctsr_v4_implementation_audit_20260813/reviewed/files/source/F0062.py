from __future__ import annotations

"""Strict Arrow schemas for the SCTSR v4 evidence tables.

The schemas are code, not inferred from the first row.  This prevents a null in
an early synthetic row from silently changing a formal column type and makes
the schema digest reproducible across Windows/Python 3.11 and 3.12.
"""

import pyarrow as pa


def _f(name: str, data_type: pa.DataType, nullable: bool = False) -> pa.Field:
    return pa.field(name, data_type, nullable=nullable)


OCCURRENCE_SCHEMA = pa.schema(
    [
        _f("run_id", pa.string()),
        _f("parent_id", pa.string()),
        _f("arm_id", pa.string()),
        _f("training_seed", pa.int64()),
        _f("epoch", pa.int16()),
        _f("base_batch_index", pa.int32()),
        _f("global_step_before", pa.int64()),
        _f("occurrence_role", pa.string()),
        _f("occurrence_index_in_step", pa.int16()),
        _f("sample_id", pa.string()),
        _f("y_true", pa.int8()),
        _f("replay_role", pa.string()),
        _f("identity_pool_id", pa.string()),
        _f("identity_group", pa.string()),
        _f("selection_policy", pa.string()),
        _f("selection_reason_code", pa.string()),
        _f("oof_fold", pa.int8()),
        _f("oof_group_id", pa.string()),
        _f("oof_group_semantic", pa.string()),
        _f("historical_dynamic_bucket", pa.string()),
        _f("augmentation_seed", pa.uint64()),
        _f("augmentation_trace_digest", pa.string()),
        _f("replay_count_before", pa.int32()),
        _f("replay_count_after", pa.int32()),
        _f("last_replay_epoch", pa.int16(), nullable=True),
        _f("last_replay_epoch_reason", pa.string()),
        _f("epochs_since_last_replay", pa.int16(), nullable=True),
        _f("epochs_since_last_replay_reason", pa.string()),
        _f("logit_normal", pa.float32()),
        _f("logit_defect", pa.float32()),
        _f("p_defect_raw", pa.float32()),
        _f("ce_unreduced", pa.float32()),
        _f("margin_defect_minus_normal", pa.float32()),
        _f("predicted_label_argmax", pa.int8()),
        _f("correct_argmax", pa.bool_()),
        _f("oof_reference_probability", pa.float32(), nullable=True),
        _f("oof_reference_reason", pa.string()),
        _f("rho_candidate_signal", pa.float32(), nullable=True),
        _f("rho_reason", pa.string()),
        _f("row_generation", pa.int32()),
        _f("planned_replay_epoch", pa.int16(), nullable=True),
        _f("planned_replay_epoch_reason", pa.string()),
        _f("planned_step_slot", pa.int32(), nullable=True),
        _f("planned_step_slot_reason", pa.string()),
        _f("cumulative_replay_count_before", pa.int64()),
        _f("cumulative_replay_count_after", pa.int64()),
        _f("pool_multiplicity_target", pa.int32()),
        _f("schedule_family", pa.string()),
        _f("fallback_state", pa.string()),
    ]
)


OPTIMIZER_GROUP_TYPE = pa.struct(
    [
        _f("group_index", pa.int16()),
        _f("lr", pa.float64()),
        _f("initial_lr", pa.float64(), nullable=True),
        _f("momentum", pa.float64(), nullable=True),
        _f("beta1", pa.float64(), nullable=True),
        _f("beta2", pa.float64(), nullable=True),
        _f("weight_decay", pa.float64()),
        _f("dampening", pa.float64(), nullable=True),
        _f("nesterov", pa.bool_(), nullable=True),
    ]
)


STEP_SCHEMA = pa.schema(
    [
        _f("run_id", pa.string()),
        _f("parent_id", pa.string()),
        _f("arm_id", pa.string()),
        _f("training_seed", pa.int64()),
        _f("epoch", pa.int16()),
        _f("base_batch_index", pa.int32()),
        _f("global_step_before", pa.int64()),
        _f("global_step_after", pa.int64()),
        _f("base_batch_size", pa.int16()),
        _f("replay_microbatch_size", pa.int16()),
        _f("replay_rate_numerator", pa.int32()),
        _f("replay_rate_denominator", pa.int32()),
        _f("base_loss", pa.float64()),
        _f("replay_loss", pa.float64()),
        _f("combined_loss_for_reporting", pa.float64()),
        _f("base_loss_items", pa.map_(pa.string(), pa.float64())),
        _f("parameter_grad_norm_before_clip", pa.float64()),
        _f("parameter_grad_norm_after_clip", pa.float64()),
        _f("clip_max_norm", pa.float64(), nullable=True),
        _f("clip_reason", pa.string()),
        _f("optimizer_step_count_delta", pa.int8()),
        _f("learning_rates", pa.list_(pa.float64())),
        _f("optimizer_hyperparameters", pa.list_(OPTIMIZER_GROUP_TYPE)),
        _f("amp_scale_before", pa.float64(), nullable=True),
        _f("amp_scale_after", pa.float64(), nullable=True),
        _f("amp_reason", pa.string()),
        _f("overflow_or_step_skipped", pa.bool_()),
        _f("ema_updates_before", pa.int64()),
        _f("ema_updates_after", pa.int64()),
        _f("scheduler_state_digest", pa.string()),
        _f("warmup_progress", pa.float64()),
        _f("bn_digest_before_replay", pa.string(), nullable=True),
        _f("bn_digest_after_replay_restore", pa.string(), nullable=True),
        _f("bn_reason", pa.string()),
        _f("rng_digest_before_base", pa.string()),
        _f("rng_digest_before_replay", pa.string(), nullable=True),
        _f("rng_digest_after_replay_restore", pa.string(), nullable=True),
        _f("rng_reason", pa.string()),
        _f("replay_rng_fork_digest", pa.string(), nullable=True),
        _f("replay_rng_fork_reason", pa.string()),
        _f("base_augmentation_digest", pa.string()),
        _f("replay_augmentation_digest", pa.string(), nullable=True),
        _f("replay_augmentation_reason", pa.string()),
        _f("dataloader_wait_seconds", pa.float64()),
        _f("base_forward_seconds", pa.float64()),
        _f("replay_forward_seconds", pa.float64()),
        _f("backward_seconds", pa.float64()),
        _f("optimizer_seconds", pa.float64()),
        _f("write_buffer_bytes", pa.int64()),
        _f("status", pa.string()),
        _f("row_generation", pa.int32()),
    ]
)


EXPOSURE_SCHEMA = pa.schema(
    [
        _f("run_id", pa.string()),
        _f("parent_id", pa.string()),
        _f("arm_id", pa.string()),
        _f("training_seed", pa.int64()),
        _f("epoch", pa.int16()),
        _f("denominator_role", pa.string()),
        _f("base_denominator_planned", pa.int32()),
        _f("base_denominator_actual", pa.int32()),
        _f("rate_numerator", pa.int32()),
        _f("rate_denominator", pa.int32()),
        _f("replay_numerator_planned", pa.int32()),
        _f("replay_numerator_actual", pa.int32()),
        _f("unique_replay_ids", pa.int32()),
        _f("repeat_occurrences", pa.int32()),
        _f("cumulative_occurrences", pa.int64()),
        _f("multiplicity_min", pa.int32()),
        _f("multiplicity_max", pa.int32()),
        _f("multiplicity_mean", pa.float64()),
        _f("multiplicity_q0", pa.float64()),
        _f("multiplicity_q25", pa.float64()),
        _f("multiplicity_q50", pa.float64()),
        _f("multiplicity_q75", pa.float64()),
        _f("multiplicity_q100", pa.float64()),
        _f("base_optimizer_steps_planned", pa.int32()),
        _f("base_optimizer_steps_actual", pa.int32()),
        _f("ema_updates_delta", pa.int32()),
        _f("scheduler_epoch_transitions_delta", pa.int16()),
        _f("base_order_digest", pa.string()),
        _f("base_augmentation_digest", pa.string()),
        _f("replay_schedule_digest", pa.string()),
        _f("identity_pool_digest", pa.string()),
        _f("occurrence_partition_sha256", pa.string()),
        _f("step_partition_sha256", pa.string()),
        _f("telemetry_partition_sha256", pa.string()),
        _f("checkpoint_sha256", pa.string()),
        _f("write_seconds", pa.float64()),
        _f("dataloader_wait_seconds", pa.float64()),
        _f("training_seconds", pa.float64()),
        _f("evaluation_seconds", pa.float64()),
        _f("disk_bytes_written", pa.int64()),
        _f("transaction_generation", pa.int32()),
        _f("validation_status", pa.string()),
    ]
)


SELECTION_SCHEMA = pa.schema(
    [
        _f("candidate_sample_id", pa.string()),
        _f("eligibility", pa.bool_()),
        _f("exclusion_reason", pa.string()),
        _f("allowed_strata", pa.string()),
        _f("stratum_quota_required", pa.int32()),
        _f("stratum_quota_available", pa.int32()),
        _f("selection_counter_hash", pa.string()),
        _f("selected", pa.bool_()),
        _f("selected_pool", pa.string()),
        _f("terminal_field_guard_digest", pa.string()),
        _f("terminal_field_status", pa.string()),
        _f("source_row_asset_sha256", pa.string()),
        _f("duplicate_overlap_status", pa.string()),
        _f("row_generation", pa.int32()),
    ]
)


TELEMETRY_SCHEMA = pa.schema(
    [
        _f("timestamp_utc", pa.string()),
        _f("monotonic_seconds", pa.float64()),
        _f("run_id", pa.string()),
        _f("arm_id", pa.string()),
        _f("training_seed", pa.int64()),
        _f("epoch", pa.int16()),
        _f("process_pid", pa.int32()),
        _f("process_cpu_percent", pa.float64(), nullable=True),
        _f("process_rss", pa.int64(), nullable=True),
        _f("process_vms", pa.int64(), nullable=True),
        _f("process_read_bytes", pa.int64(), nullable=True),
        _f("process_write_bytes", pa.int64(), nullable=True),
        _f("process_read_count", pa.int64(), nullable=True),
        _f("process_write_count", pa.int64(), nullable=True),
        _f("system_cpu_percent", pa.float64(), nullable=True),
        _f("system_memory_total", pa.int64(), nullable=True),
        _f("system_memory_available", pa.int64(), nullable=True),
        _f("system_memory_used", pa.int64(), nullable=True),
        _f("system_memory_percent", pa.float64(), nullable=True),
        _f("gpu_index", pa.int16(), nullable=True),
        _f("gpu_uuid", pa.string(), nullable=True),
        _f("gpu_name", pa.string(), nullable=True),
        _f("gpu_utilization", pa.float64(), nullable=True),
        _f("gpu_memory_used", pa.int64(), nullable=True),
        _f("gpu_memory_total", pa.int64(), nullable=True),
        _f("gpu_temperature", pa.float64(), nullable=True),
        _f("gpu_power", pa.float64(), nullable=True),
        _f("cuda_allocated", pa.int64(), nullable=True),
        _f("cuda_reserved", pa.int64(), nullable=True),
        _f("cuda_max_allocated", pa.int64(), nullable=True),
        _f("cuda_max_reserved", pa.int64(), nullable=True),
        _f("run_volume_total", pa.int64(), nullable=True),
        _f("run_volume_free", pa.int64(), nullable=True),
        _f("run_volume_used", pa.int64(), nullable=True),
        _f("artifact_volume_total", pa.int64(), nullable=True),
        _f("artifact_volume_free", pa.int64(), nullable=True),
        _f("artifact_volume_used", pa.int64(), nullable=True),
        _f("process_provider_status", pa.string()),
        _f("process_provider_reason", pa.string()),
        _f("system_provider_status", pa.string()),
        _f("system_provider_reason", pa.string()),
        _f("gpu_provider_status", pa.string()),
        _f("gpu_provider_reason", pa.string()),
        _f("cuda_provider_status", pa.string()),
        _f("cuda_provider_reason", pa.string()),
        _f("disk_provider_status", pa.string()),
        _f("disk_provider_reason", pa.string()),
        _f("telemetry_provider_status", pa.string()),
        _f("provider_error_code", pa.string(), nullable=True),
        _f("row_generation", pa.int32()),
    ]
)


PREDICTION_SCHEMA = pa.schema(
    [
        _f("run_id", pa.string()),
        _f("arm_id", pa.string()),
        _f("training_seed", pa.int64()),
        _f("split_role", pa.string()),
        _f("split_manifest_path", pa.string()),
        _f("split_manifest_sha256", pa.string()),
        _f("sample_id", pa.string()),
        _f("y_true", pa.int8()),
        _f("logit_normal", pa.float64()),
        _f("logit_defect", pa.float64()),
        _f("p_defect_raw", pa.float64()),
        _f("checkpoint_epoch", pa.int16()),
        _f("checkpoint_sha256", pa.string()),
        _f("model_variant", pa.string()),
        _f("source_tree_digest", pa.string()),
        _f("prediction_generation", pa.int32()),
        _f("sample_label_identity_digest", pa.string()),
        _f("artifact_row_count", pa.int64()),
    ]
)


FRONTIER_SCHEMA = pa.schema(
    [
        _f("fn_budget", pa.int16()),
        _f("actual_fn", pa.int64()),
        _f("tn", pa.int64()),
        _f("fp", pa.int64()),
        _f("tp", pa.int64()),
        _f("threshold", pa.float64()),
        _f("threshold_rule", pa.string()),
        _f("tie_size", pa.int64()),
        _f("reachable", pa.bool_()),
        _f("defect_count", pa.int64()),
        _f("normal_count", pa.int64()),
        _f("normalized_tn", pa.float64()),
        _f("checkpoint_sha256", pa.string()),
        _f("prediction_artifact_sha256", pa.string()),
    ]
)
