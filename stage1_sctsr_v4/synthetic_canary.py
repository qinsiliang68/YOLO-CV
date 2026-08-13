from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .baseline_reference import SOURCE_TREE_INCLUDE_PATHS, source_external_references
from .arm_spec import ArmId, default_phase1_arms
from .branch_lineage import BranchLineage
from .checkpointing import (
    build_checkpoint_payload,
    checkpoint_payload_digest,
    load_checkpoint,
    save_checkpoint_atomic,
    validate_checkpoint_payload,
)
from .columnar import write_zstd_parquet
from .epoch_transaction import EpochTransaction
from .evaluation import compute_tie_safe_frontier, write_frontier_artifacts
from .errors import SctsrError
from .fault_injection import FaultKind, inject_fault
from .exposure_ledger import build_exposure_row, write_exposure_partition
from .fixed_step_runtime import (
    ExponentialMovingAverage,
    OccurrenceEvent,
    run_fixed_step_epoch,
)
from .identity_pool import OOF_GROUP_SEMANTIC, partition_five_groups
from .logical_artifact_index import LogicalArtifactEntry, LogicalArtifactIndex
from .occurrence_ledger import write_occurrence_partition
from .prediction_artifact import PredictionArtifactBinding, PredictionRow, write_prediction_artifact
from .random_controls import counter_hash
from .recovery import ResumeIdentity, find_last_complete_epoch, quarantine_inprogress, validate_recovery_pointer
from .replay_step_plan import build_replay_step_plan
from .rng_isolation import derive_counter_seed, restore_global_rng
from .schedule import (
    SchedulePlan,
    build_schedule,
    schedule_to_dict,
    validate_common_prefix,
    validate_u_f_parity,
)
from .selection_ledger import write_selection_partition
from .serialization import atomic_write_json, sha256_file, stable_digest
from .source_identity import build_source_tree_manifest
from .step_ledger import write_step_partition
from .synthetic_fixture import (
    SYNTHETIC_BASE_DENOMINATOR,
    TinyClassifier,
    SyntheticFixture,
    build_synthetic_fixture,
    make_base_loader,
    make_replay_provider,
    synthetic_split,
)
from .telemetry import TelemetrySampler, validate_telemetry_for_closeout


SYNTHETIC_MARKER = "SYNTHETIC_NOT_SCIENTIFIC_RESULT"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _new_training_stack(seed: int):
    _seed_everything(seed)
    model = TinyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=5e-4)
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0 - 0.5 * min(epoch, 10) / 10)
    scaler = torch.amp.GradScaler("cpu")
    ema = ExponentialMovingAverage.from_model(model, decay=0.99)
    return model, optimizer, scheduler, scaler, ema


def _restore_training_stack(payload: Mapping[str, Any]):
    model, optimizer, scheduler, scaler, ema = _new_training_stack(int(payload["training_seed"]))
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    scaler.load_state_dict(payload["scaler_state"])
    ema.load_state_dict(payload["ema_state"])
    # ema_updates is also top-level for explicit auditability.
    ema.updates = int(payload["ema_updates"])
    restore_global_rng(payload["rng_state"])
    return model, optimizer, scheduler, scaler, ema


def _build_all_schedules(fixture: SyntheticFixture) -> dict[ArmId, SchedulePlan]:
    t = fixture.groups_by_pool["T"]
    r1 = fixture.groups_by_pool["R1"]
    r2 = fixture.groups_by_pool["R2"]
    schedules = {
        ArmId.NR: build_schedule(ArmId.NR, primary_groups=None, primary_digest="NONE", base_denominator=fixture.base_denominator),
        ArmId.R1_U: build_schedule(ArmId.R1_U, primary_groups=r1, primary_digest=fixture.r1_result.pool.spec.identity_digest, base_denominator=fixture.base_denominator),
        ArmId.R2_U: build_schedule(ArmId.R2_U, primary_groups=r2, primary_digest=fixture.r2_result.pool.spec.identity_digest, base_denominator=fixture.base_denominator),
        ArmId.T_U: build_schedule(ArmId.T_U, primary_groups=t, primary_digest=fixture.t_pool.spec.identity_digest, base_denominator=fixture.base_denominator),
        ArmId.R2_F: build_schedule(ArmId.R2_F, primary_groups=r2, primary_digest=fixture.r2_result.pool.spec.identity_digest, base_denominator=fixture.base_denominator),
        ArmId.T_F: build_schedule(ArmId.T_F, primary_groups=t, primary_digest=fixture.t_pool.spec.identity_digest, base_denominator=fixture.base_denominator),
        ArmId.T_TO_R2_AT_160: build_schedule(
            ArmId.T_TO_R2_AT_160,
            primary_groups=t,
            primary_digest=fixture.t_pool.spec.identity_digest,
            fallback_groups=r2,
            fallback_digest=fixture.r2_result.pool.spec.identity_digest,
            base_denominator=fixture.base_denominator,
        ),
        ArmId.T_TO_NR_AT_160: build_schedule(
            ArmId.T_TO_NR_AT_160,
            primary_groups=t,
            primary_digest=fixture.t_pool.spec.identity_digest,
            base_denominator=fixture.base_denominator,
        ),
    }
    validate_u_f_parity(schedules[ArmId.T_U], schedules[ArmId.T_F])
    validate_u_f_parity(schedules[ArmId.R2_U], schedules[ArmId.R2_F])
    validate_common_prefix(schedules[ArmId.T_U], schedules[ArmId.T_TO_R2_AT_160])
    validate_common_prefix(schedules[ArmId.T_U], schedules[ArmId.T_TO_NR_AT_160])
    return schedules


def _pool_for_arm(fixture: SyntheticFixture, arm: ArmId, epoch: int = 121):
    if arm is ArmId.R1_U:
        return fixture.r1_result.pool
    if arm in {ArmId.R2_U, ArmId.R2_F}:
        return fixture.r2_result.pool
    if arm is ArmId.T_TO_R2_AT_160 and epoch > 160:
        return fixture.r2_result.pool
    if arm is ArmId.NR:
        return None
    return fixture.t_pool


def _identity_group_lookup(groups: Mapping[str, Sequence[Any]]) -> dict[str, str]:
    return {record.sample_id: group for group, records in groups.items() for record in records}


def _occurrence_rows(
    events: Sequence[OccurrenceEvent],
    *,
    fixture: SyntheticFixture,
    run_id: str,
    parent_id: str,
    arm_id: ArmId,
    epoch: int,
    epoch_plan: Any,
    parent_global_step: int,
    pool: Any,
    groups: Mapping[str, Sequence[Any]] | None,
) -> list[dict[str, Any]]:
    records = fixture.record_by_id
    group_by_id = _identity_group_lookup(groups or {})
    replay_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    cumulative = 0
    for event in events:
        probabilities = torch.softmax(event.logits.float(), dim=1)
        for index, sample_id in enumerate(event.sample_ids):
            record = records[sample_id]
            is_replay = event.occurrence_role == "REPLAY"
            before = replay_counts[sample_id]
            after = before + (1 if is_replay else 0)
            if is_replay:
                replay_counts[sample_id] = after
                cumulative_before = cumulative
                cumulative += 1
                cumulative_after = cumulative
            else:
                cumulative_before = cumulative_after = cumulative
            logits = event.logits[index]
            p_defect = float(probabilities[index, 1])
            predicted = int(torch.argmax(logits).item())
            rows.append(
                {
                    "run_id": run_id,
                    "parent_id": parent_id,
                    "arm_id": arm_id.value,
                    "training_seed": fixture.training_seed,
                    "epoch": epoch,
                    "base_batch_index": event.base_step_index,
                    "global_step_before": parent_global_step + event.base_step_index,
                    "occurrence_role": event.occurrence_role,
                    "occurrence_index_in_step": index,
                    "sample_id": sample_id,
                    "y_true": record.y_true,
                    "replay_role": record.replay_role if is_replay else "NOT_APPLICABLE_BASE",
                    "identity_pool_id": pool.spec.pool_id if is_replay and pool is not None else "NOT_APPLICABLE_BASE",
                    "identity_group": group_by_id.get(sample_id, "MISSING_REPLAY_GROUP") if is_replay else "NOT_APPLICABLE_BASE",
                    "selection_policy": epoch_plan.identity_policy if is_replay else "BASE_CANONICAL",
                    "selection_reason_code": "PLANNED_REPLAY_STEP_SLOT" if is_replay else "CANONICAL_BASE_OCCURRENCE",
                    "oof_fold": record.oof_fold,
                    "oof_group_id": record.oof_group_id,
                    "oof_group_semantic": OOF_GROUP_SEMANTIC,
                    "historical_dynamic_bucket": record.historical_dynamic_bucket,
                    "augmentation_seed": derive_counter_seed(
                        "replay_augmentation" if is_replay else "base_augmentation",
                        fixture.training_seed,
                        epoch,
                        f"{event.base_step_index}:{sample_id}" if is_replay else sample_id,
                    ) % (2**63),
                    "augmentation_trace_digest": event.augmentation_digests[index],
                    "replay_count_before": before if is_replay else 0,
                    "replay_count_after": after if is_replay else 0,
                    "last_replay_epoch": None,
                    "last_replay_epoch_reason": "NEVER_REPLAYED" if is_replay else "NOT_APPLICABLE_BASE",
                    "epochs_since_last_replay": None,
                    "epochs_since_last_replay_reason": "NEVER_REPLAYED" if is_replay else "NOT_APPLICABLE_BASE",
                    "logit_normal": float(logits[0]),
                    "logit_defect": float(logits[1]),
                    "p_defect_raw": p_defect,
                    "ce_unreduced": float(event.per_sample_ce[index]),
                    "margin_defect_minus_normal": float(logits[1] - logits[0]),
                    "predicted_label_argmax": predicted,
                    "correct_argmax": bool(predicted == record.y_true),
                    "oof_reference_probability": None,
                    "oof_reference_reason": "REGISTERED_NOT_AVAILABLE_SYNTHETIC",
                    "rho_candidate_signal": None,
                    "rho_reason": "REGISTERED_NOT_REPORTED",
                    "row_generation": 1,
                    "planned_replay_epoch": epoch if is_replay else None,
                    "planned_replay_epoch_reason": "PRESENT" if is_replay else "NOT_APPLICABLE_BASE",
                    "planned_step_slot": event.base_step_index if is_replay else None,
                    "planned_step_slot_reason": "PRESENT" if is_replay else "NOT_APPLICABLE_BASE",
                    "cumulative_replay_count_before": cumulative_before,
                    "cumulative_replay_count_after": cumulative_after,
                    "pool_multiplicity_target": 16 if is_replay else 0,
                    "schedule_family": epoch_plan.schedule_family if is_replay else "BASE_CANONICAL",
                    "fallback_state": epoch_plan.fallback_state if is_replay else "NOT_APPLICABLE_BASE",
                }
            )
    return rows


def _step_rows(
    result: Any,
    *,
    run_id: str,
    parent_id: str,
    arm_id: ArmId,
    training_seed: int,
    epoch: int,
    parent_global_step: int,
    epoch_plan: Any,
) -> list[dict[str, Any]]:
    rows = []
    for record in result.records:
        has_replay = record.replay_microbatch_size > 0
        rows.append(
            {
                "run_id": run_id,
                "parent_id": parent_id,
                "arm_id": arm_id.value,
                "training_seed": training_seed,
                "epoch": epoch,
                "base_batch_index": record.base_step_index,
                "global_step_before": parent_global_step + record.base_step_index,
                "global_step_after": parent_global_step + record.base_step_index + 1,
                "base_batch_size": record.base_batch_size,
                "replay_microbatch_size": record.replay_microbatch_size,
                "replay_rate_numerator": epoch_plan.rate.numerator,
                "replay_rate_denominator": epoch_plan.rate.denominator,
                "base_loss": record.base_loss,
                "replay_loss": record.replay_loss,
                # The ledger contract defines this reporting-only field as the
                # arithmetic sum of the two recorded components.  Recompute it
                # from those serialized values so a float32 training scalar
                # cannot introduce a third, slightly different rounding path.
                "combined_loss_for_reporting": float(record.base_loss) + float(record.replay_loss),
                "base_loss_items": {"classification_loss": record.base_loss},
                "parameter_grad_norm_before_clip": record.parameter_grad_norm_before_clip,
                "parameter_grad_norm_after_clip": record.parameter_grad_norm_after_clip,
                "clip_max_norm": record.clip_max_norm,
                "clip_reason": "PRESENT" if record.clip_max_norm is not None else "DISABLED_SYNTHETIC",
                "optimizer_step_count_delta": record.optimizer_step_count_delta,
                "learning_rates": list(record.learning_rates),
                "optimizer_hyperparameters": list(record.optimizer_hyperparameters),
                "amp_scale_before": record.amp_scale_before,
                "amp_scale_after": record.amp_scale_after,
                "amp_reason": "PRESENT" if record.amp_scale_before is not None else "AMP_DISABLED",
                "overflow_or_step_skipped": record.overflow_or_step_skipped,
                "ema_updates_before": record.ema_updates_before,
                "ema_updates_after": record.ema_updates_after,
                "scheduler_state_digest": record.scheduler_state_digest,
                "warmup_progress": record.warmup_progress,
                "bn_digest_before_replay": record.bn_digest_before_replay,
                "bn_digest_after_replay_restore": record.bn_digest_after_replay_restore,
                "bn_reason": "PRESENT" if has_replay else "NO_REPLAY_IN_STEP",
                "rng_digest_before_base": record.rng_digest_before_base,
                "rng_digest_before_replay": record.rng_digest_before_replay,
                "rng_digest_after_replay_restore": record.rng_digest_after_replay_restore,
                "rng_reason": "PRESENT" if has_replay else "NO_REPLAY_IN_STEP",
                "replay_rng_fork_digest": record.replay_rng_fork_digest,
                "replay_rng_fork_reason": "PRESENT" if has_replay else "NO_REPLAY_IN_STEP",
                "base_augmentation_digest": record.base_augmentation_digest,
                "replay_augmentation_digest": record.replay_augmentation_digest,
                "replay_augmentation_reason": "PRESENT" if has_replay else "NO_REPLAY_IN_STEP",
                "dataloader_wait_seconds": record.dataloader_wait_seconds,
                "base_forward_seconds": record.base_forward_seconds,
                "replay_forward_seconds": record.replay_forward_seconds,
                "backward_seconds": record.backward_seconds,
                "optimizer_seconds": record.optimizer_seconds,
                "write_buffer_bytes": record.write_buffer_bytes,
                "status": "PASS",
                "row_generation": 1,
            }
        )
    return rows


def _selection_rows(fixture: SyntheticFixture, policy: str) -> list[dict[str, Any]]:
    if policy == "T_STRESS":
        pool = fixture.t_pool
        seed = 0
        guard_digest = "NOT_APPLICABLE_CANONICAL_T"
        selected_ids = {record.sample_id for record in pool.records}
        terminal_status = "TERMINAL_FIELDS_NOT_USED_CANONICAL_T"
        overlap = "CANONICAL_T"
    elif policy == "R1_GLOBAL_RANDOM":
        pool = fixture.r1_result.pool
        seed = fixture.r1_result.audit.selection_seed
        guard_digest = "NOT_APPLICABLE_GLOBAL_RANDOM"
        selected_ids = {record.sample_id for record in pool.records}
        terminal_status = "TERMINAL_FIELDS_NOT_USED_GLOBAL_RANDOM"
        overlap = "NATURAL_OVERLAP_REPORTED"
    else:
        pool = fixture.r2_result.pool
        seed = fixture.r2_result.audit.selection_seed
        guard_digest = fixture.r2_result.audit.terminal_field_guard_digest or "MISSING"
        selected_ids = {record.sample_id for record in pool.records}
        terminal_status = "TERMINAL_FIELDS_NOT_LOADED"
        overlap = "ZERO_OVERLAP"
    required = Counter("|".join(map(str, record.stratum())) for record in fixture.t_pool.records)
    available = Counter("|".join(map(str, record.stratum())) for record in fixture.base_records if record.sample_id not in {r.sample_id for r in fixture.t_pool.records})
    rows = []
    for record in fixture.base_records:
        stratum = "|".join(map(str, record.stratum()))
        eligible = record.base_manifest_membership and not (policy == "R2_MATCHED_RANDOM" and record.sample_id in {r.sample_id for r in fixture.t_pool.records})
        rows.append(
            {
                "candidate_sample_id": record.sample_id,
                "eligibility": eligible,
                "exclusion_reason": "ELIGIBLE" if eligible else "EXCLUDED_T_IDENTITY_FOR_R2",
                "allowed_strata": stratum,
                "stratum_quota_required": int(required.get(stratum, 0)),
                "stratum_quota_available": int(available.get(stratum, 0)) if policy == "R2_MATCHED_RANDOM" else SYNTHETIC_BASE_DENOMINATOR,
                "selection_counter_hash": counter_hash(policy, seed, stratum, record.sample_id),
                "selected": record.sample_id in selected_ids,
                "selected_pool": pool.spec.pool_id if record.sample_id in selected_ids else "NOT_SELECTED",
                "terminal_field_guard_digest": guard_digest,
                "source_row_asset_sha256": "2" * 64,
                "duplicate_overlap_status": overlap,
                "terminal_field_status": terminal_status,
                "row_generation": 1,
            }
        )
    return rows


def _write_selection_evidence(root: Path, fixture: SyntheticFixture) -> dict[str, Any]:
    output = {}
    for policy in ("T_STRESS", "R1_GLOBAL_RANDOM", "R2_MATCHED_RANDOM"):
        path = root / "04_ledgers" / "selection" / "run_id=SYNTHETIC_SELECTION" / "epoch=0000" / f"{policy}.parquet"
        manifest = write_selection_partition(_selection_rows(fixture, policy), path, policy=policy)
        output[policy] = asdict(manifest)
    return output


def _predict(
    model: torch.nn.Module,
    fixture: SyntheticFixture,
    *,
    run_id: str,
    arm_id: ArmId,
    checkpoint_sha: str,
    source_tree_digest: str,
) -> tuple[PredictionRow, ...]:
    split = synthetic_split(fixture)
    split_digest = stable_digest([(record.sample_id, record.y_true) for record in split])
    model.eval()
    with torch.no_grad():
        features = torch.stack([fixture.features[record.sample_id] for record in split]).reshape(len(split), 1, 2, 2)
        logits = model(features).float()
        probabilities = torch.softmax(logits, dim=1)
    rows = []
    for index, record in enumerate(split):
        rows.append(
            PredictionRow(
                run_id=run_id,
                arm_id=arm_id.value,
                training_seed=fixture.training_seed,
                split_role="synthetic",
                split_manifest_path="SYNTHETIC_IN_MEMORY_SPLIT",
                split_manifest_sha256=split_digest,
                sample_id=record.sample_id,
                y_true=record.y_true,
                logit_normal=float(logits[index, 0]),
                logit_defect=float(logits[index, 1]),
                p_defect_raw=float(probabilities[index, 1]),
                checkpoint_epoch=200,
                checkpoint_sha256=checkpoint_sha,
                model_variant="MODEL_SYNTHETIC_CANARY",
                source_tree_digest=source_tree_digest,
                prediction_generation=1,
            )
        )
    return tuple(rows)


def _artifact_index(root: Path) -> dict[str, Any]:
    files = []
    for path in root.rglob("*"):
        if path.is_file() and path.name != "ARTIFACT_INDEX.json":
            files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    files.sort(key=lambda row: row["path"])
    index = {"schema_version": "stage1.sctsr.synthetic_artifact_index.v1", "semantic": SYNTHETIC_MARKER, "files": files}
    index["artifact_index_digest"] = stable_digest(files)
    return index


def _exercise_checkpoint_resume(
    root: Path,
    *,
    parent_checkpoint: Path,
    parent_sha256: str,
    fixture: SyntheticFixture,
    schedule: SchedulePlan,
    training_seed: int,
    source_tree_digest: str,
    contract_digest: str,
    asset_registry_digest: str,
) -> dict[str, Any]:
    """Prove that restoring E121 reproduces an uninterrupted E122 state.

    This is a compressed mechanism canary, not a formal training resume. It
    nevertheless performs real forward/backward/optimizer/scheduler/EMA work
    on both paths and compares the full typed checkpoint payload digest,
    including RNG and optimizer state.
    """

    arm = ArmId.T_U
    run_id = f"SYNTH_RESUME_{arm.value}_{training_seed}"
    parent_payload = load_checkpoint(
        parent_checkpoint,
        expected_sha256=parent_sha256,
        expected_epoch=120,
    )
    model, optimizer, scheduler, scaler, ema = _restore_training_stack(parent_payload)

    def run_epoch(epoch: int, stack: tuple[Any, Any, Any, Any, Any]):
        current_model, current_optimizer, current_scheduler, current_scaler, current_ema = stack
        epoch_plan = schedule.epoch(epoch)
        base_loader = make_base_loader(fixture, epoch=epoch)
        replay_plan = build_replay_step_plan(
            run_id=run_id,
            arm_id=arm.value,
            training_seed=training_seed,
            epoch=epoch,
            schedule_family=epoch_plan.schedule_family,
            sample_ids=epoch_plan.sample_ids,
            base_batch_sizes=[int(batch["labels"].shape[0]) for batch in base_loader],
        )
        result = run_fixed_step_epoch(
            model=current_model,
            optimizer=current_optimizer,
            base_loader=base_loader,
            replay_plan=replay_plan,
            replay_batch_provider=make_replay_provider(fixture),
            training_seed=training_seed,
            epoch=epoch,
            ema=current_ema,
            scheduler=current_scheduler,
            scaler=current_scaler,
            clip_max_norm=10.0,
        )
        return result

    first_result = run_epoch(121, (model, optimizer, scheduler, scaler, ema))
    checkpoint_121_payload = build_checkpoint_payload(
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=121,
        global_step=int(parent_payload["global_step"]) + first_result.optimizer_steps,
        base_sampler_generation=121,
        canonical_training_lock_sha256=str(parent_payload["canonical_training_lock_sha256"]),
        initial_checkpoint_sha256=str(parent_payload["initial_checkpoint_sha256"]),
        base_manifest_sha256=str(parent_payload["base_manifest_sha256"]),
        training_seed=training_seed,
        source_tree_digest=source_tree_digest,
        runtime_config_digest=contract_digest,
        asset_registry_digest=asset_registry_digest,
    )
    checkpoint_121 = root / "05_checkpoints" / "synthetic_resume_t_u_e121.pt"
    checkpoint_121_sha_before = save_checkpoint_atomic(checkpoint_121, checkpoint_121_payload)

    uninterrupted_result = run_epoch(122, (model, optimizer, scheduler, scaler, ema))
    uninterrupted_payload = build_checkpoint_payload(
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=122,
        global_step=int(checkpoint_121_payload["global_step"]) + uninterrupted_result.optimizer_steps,
        base_sampler_generation=122,
        canonical_training_lock_sha256=str(parent_payload["canonical_training_lock_sha256"]),
        initial_checkpoint_sha256=str(parent_payload["initial_checkpoint_sha256"]),
        base_manifest_sha256=str(parent_payload["base_manifest_sha256"]),
        training_seed=training_seed,
        source_tree_digest=source_tree_digest,
        runtime_config_digest=contract_digest,
        asset_registry_digest=asset_registry_digest,
    )

    restored_payload = load_checkpoint(
        checkpoint_121,
        expected_sha256=checkpoint_121_sha_before,
        expected_epoch=121,
    )
    resumed_stack = _restore_training_stack(restored_payload)
    resumed_result = run_epoch(122, resumed_stack)
    resumed_model, resumed_optimizer, resumed_scheduler, resumed_scaler, resumed_ema = resumed_stack
    resumed_payload = build_checkpoint_payload(
        model=resumed_model,
        ema=resumed_ema,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=resumed_scaler,
        epoch=122,
        global_step=int(restored_payload["global_step"]) + resumed_result.optimizer_steps,
        base_sampler_generation=122,
        canonical_training_lock_sha256=str(restored_payload["canonical_training_lock_sha256"]),
        initial_checkpoint_sha256=str(restored_payload["initial_checkpoint_sha256"]),
        base_manifest_sha256=str(restored_payload["base_manifest_sha256"]),
        training_seed=training_seed,
        source_tree_digest=source_tree_digest,
        runtime_config_digest=contract_digest,
        asset_registry_digest=asset_registry_digest,
    )

    uninterrupted_digest = checkpoint_payload_digest(uninterrupted_payload)
    resumed_digest = checkpoint_payload_digest(resumed_payload)
    if uninterrupted_digest != resumed_digest:
        raise RuntimeError(
            "Synthetic checkpoint resume changed the full E122 training state: "
            f"{uninterrupted_digest} != {resumed_digest}"
        )
    if uninterrupted_result.optimizer_steps != resumed_result.optimizer_steps:
        raise RuntimeError("Synthetic checkpoint resume changed optimizer-step count")
    checkpoint_121_sha_after = sha256_file(checkpoint_121)
    if checkpoint_121_sha_after != checkpoint_121_sha_before:
        raise RuntimeError("Synthetic checkpoint resume modified the immutable E121 checkpoint")

    checkpoint_122 = root / "05_checkpoints" / "synthetic_resume_t_u_e122.pt"
    checkpoint_122_sha = save_checkpoint_atomic(checkpoint_122, resumed_payload)
    receipt = {
        "schema_version": "stage1.sctsr.synthetic_checkpoint_resume_receipt.v1",
        "status": "PASS",
        "semantic": SYNTHETIC_MARKER,
        "scientific_result": False,
        "arm_id": arm.value,
        "training_seed": training_seed,
        "checkpoint_epoch": 121,
        "resumed_epoch": 122,
        "resume_checkpoint_path": checkpoint_121.relative_to(root).as_posix(),
        "resume_checkpoint_sha256_before": checkpoint_121_sha_before,
        "resume_checkpoint_sha256_after": checkpoint_121_sha_after,
        "resumed_checkpoint_path": checkpoint_122.relative_to(root).as_posix(),
        "resumed_checkpoint_sha256": checkpoint_122_sha,
        "uninterrupted_checkpoint_payload_digest": uninterrupted_digest,
        "resumed_checkpoint_payload_digest": resumed_digest,
        "uninterrupted_optimizer_steps": uninterrupted_result.optimizer_steps,
        "resumed_optimizer_steps": resumed_result.optimizer_steps,
        "uninterrupted_global_step": int(uninterrupted_payload["global_step"]),
        "resumed_global_step": int(resumed_payload["global_step"]),
        "uninterrupted_ema_updates": int(uninterrupted_payload["ema_updates"]),
        "resumed_ema_updates": int(resumed_payload["ema_updates"]),
        "old_generation_overwritten": False,
        "formal_training_started": False,
    }
    atomic_write_json(root / "08_receipts" / "CHECKPOINT_RESUME_RECEIPT.json", receipt)
    return receipt


def run_synthetic_canary(
    output_root: str | Path,
    *,
    repository_root: str | Path | None = None,
    training_seed: int = 20260812,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    if overwrite:
        raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Synthetic evidence is immutable; choose a new output root instead of overwriting")
    if root.exists():
        raise FileExistsError(f"Synthetic output already exists: {root}")
    for relative in (
        "00_contract", "01_assets", "02_parent", "03_branch", "04_ledgers/selection",
        "05_checkpoints", "06_predictions", "07_evaluation", "08_receipts", "09_quarantine",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "00_contract" / "SYNTHETIC_MARKER.json", {"semantic": SYNTHETIC_MARKER, "scientific_result": False})

    source_root = Path(repository_root or Path(__file__).resolve().parents[1])
    source_manifest = build_source_tree_manifest(
        source_root,
        SOURCE_TREE_INCLUDE_PATHS,
        external_references=source_external_references(),
    )
    atomic_write_json(root / "00_contract" / "SOURCE_TREE_MANIFEST.json", source_manifest)
    source_digest = source_manifest["source_tree_digest"]
    contract_digest = stable_digest({"synthetic": True, "base_denominator": SYNTHETIC_BASE_DENOMINATOR, "fixed_anchors": [120, 160, 200]})
    asset_digest = stable_digest({"synthetic_fixture": "v1", "base_denominator": SYNTHETIC_BASE_DENOMINATOR})

    fixture = build_synthetic_fixture(training_seed=training_seed)
    schedules = _build_all_schedules(fixture)
    atomic_write_json(
        root / "01_assets" / "SYNTHETIC_ASSET_REGISTRY.json",
        {
            "schema_version": "stage1.sctsr.synthetic_asset_registry.v1",
            "semantic": SYNTHETIC_MARKER,
            "base_denominator": SYNTHETIC_BASE_DENOMINATOR,
            "base_identity_digest": stable_digest(fixture.base_ids),
            "t_identity_digest": fixture.t_pool.spec.identity_digest,
            "r1_identity_digest": fixture.r1_result.pool.spec.identity_digest,
            "r2_identity_digest": fixture.r2_result.pool.spec.identity_digest,
            "t_r2_overlap": len({r.sample_id for r in fixture.t_pool.records} & {r.sample_id for r in fixture.r2_result.pool.records}),
            "val_target_available": False,
        },
    )
    selection_evidence = _write_selection_evidence(root, fixture)

    # One real mechanism epoch is persisted as the compressed synthetic E120 parent.
    parent_id = f"SYNTH_PARENT_{training_seed}"
    parent_model, parent_optimizer, parent_scheduler, parent_scaler, parent_ema = _new_training_stack(training_seed)
    parent_loader = make_base_loader(fixture, epoch=120)
    parent_plan = build_replay_step_plan(
        run_id=parent_id,
        arm_id="COMMON_PARENT_NR",
        training_seed=training_seed,
        epoch=120,
        schedule_family="NR",
        sample_ids=(),
        base_batch_sizes=[int(batch["labels"].shape[0]) for batch in parent_loader],
    )
    parent_result = run_fixed_step_epoch(
        model=parent_model,
        optimizer=parent_optimizer,
        base_loader=parent_loader,
        replay_plan=parent_plan,
        replay_batch_provider=make_replay_provider(fixture),
        training_seed=training_seed,
        epoch=120,
        ema=parent_ema,
        scheduler=parent_scheduler,
        scaler=parent_scaler,
        clip_max_norm=10.0,
    )
    parent_payload = build_checkpoint_payload(
        model=parent_model,
        ema=parent_ema,
        optimizer=parent_optimizer,
        scheduler=parent_scheduler,
        scaler=parent_scaler,
        epoch=120,
        global_step=parent_result.optimizer_steps,
        base_sampler_generation=120,
        canonical_training_lock_sha256="A" * 64,
        initial_checkpoint_sha256="B" * 64,
        base_manifest_sha256="C" * 64,
        training_seed=training_seed,
        source_tree_digest=source_digest,
        runtime_config_digest=contract_digest,
        asset_registry_digest=asset_digest,
    )
    parent_checkpoint = root / "05_checkpoints" / "synthetic_parent_e120.pt"
    parent_sha = save_checkpoint_atomic(parent_checkpoint, parent_payload)
    parent_checkpoint.chmod(0o444)
    atomic_write_json(
        root / "02_parent" / "PARENT_RECEIPT.json",
        {
            "schema_version": "stage1.sctsr.synthetic_parent_receipt.v1",
            "semantic": SYNTHETIC_MARKER,
            "compressed_timeline": True,
            "logical_epoch": 120,
            "mechanism_epochs_executed": 1,
            "parent_id": parent_id,
            "checkpoint_path": parent_checkpoint.relative_to(root).as_posix(),
            "checkpoint_sha256": parent_sha,
            "optimizer_steps": parent_result.optimizer_steps,
            "base_occurrences": parent_result.base_occurrences,
            "replay_occurrences": parent_result.replay_occurrences,
        },
    )

    branch_summaries: dict[str, Any] = {}
    logical_index = LogicalArtifactIndex()
    for logical_epoch in range(1, 121):
        logical_index.add(
            LogicalArtifactEntry(
                logical_run_id="SYNTHETIC_ALL_ARMS",
                logical_epoch=logical_epoch,
                physical_owner_type="PARENT",
                physical_run_id=parent_id,
                artifact_relative_path="02_parent/PARENT_RECEIPT.json",
                artifact_sha256=sha256_file(root / "02_parent" / "PARENT_RECEIPT.json"),
                checkpoint_sha256=parent_sha,
                source_tree_digest=source_digest,
                lineage_digest="NOT_APPLICABLE_PARENT",
            )
        )

    for arm_spec in default_phase1_arms():
        arm = arm_spec.arm_id
        run_id = f"SYNTH_{arm.value}_{training_seed}"
        branch_root = root / "03_branch" / arm.value
        branch_root.mkdir(parents=True, exist_ok=True)
        payload = load_checkpoint(parent_checkpoint, expected_sha256=parent_sha, expected_epoch=120)
        model, optimizer, scheduler, scaler, ema = _restore_training_stack(payload)
        epoch_plan = schedules[arm].epoch(121)
        base_loader = make_base_loader(fixture, epoch=121)
        replay_plan = build_replay_step_plan(
            run_id=run_id,
            arm_id=arm.value,
            training_seed=training_seed,
            epoch=121,
            schedule_family=epoch_plan.schedule_family,
            sample_ids=epoch_plan.sample_ids,
            base_batch_sizes=[int(batch["labels"].shape[0]) for batch in base_loader],
        )
        events: list[OccurrenceEvent] = []
        telemetry_sampler = TelemetrySampler(
            run_id=run_id,
            arm_id=arm.value,
            training_seed=training_seed,
            epoch=121,
            run_path=branch_root,
            artifact_path=root,
            row_generation=1,
        ).start()
        training_start = time.monotonic()
        result = run_fixed_step_epoch(
            model=model,
            optimizer=optimizer,
            base_loader=base_loader,
            replay_plan=replay_plan,
            replay_batch_provider=make_replay_provider(fixture),
            training_seed=training_seed,
            epoch=121,
            ema=ema,
            scheduler=scheduler,
            scaler=scaler,
            clip_max_norm=10.0,
            occurrence_event_sink=events.append,
        )
        training_seconds = time.monotonic() - training_start
        telemetry_rows = telemetry_sampler.stop()
        validate_telemetry_for_closeout(telemetry_rows)
        pool = _pool_for_arm(fixture, arm)
        if arm is ArmId.R1_U:
            groups = fixture.groups_by_pool["R1"]
        elif arm in {ArmId.R2_U, ArmId.R2_F}:
            groups = fixture.groups_by_pool["R2"]
        elif arm is ArmId.NR:
            groups = None
        else:
            groups = fixture.groups_by_pool["T"]
        occurrence_rows = _occurrence_rows(
            events,
            fixture=fixture,
            run_id=run_id,
            parent_id=parent_id,
            arm_id=arm,
            epoch=121,
            epoch_plan=epoch_plan,
            parent_global_step=int(payload["global_step"]),
            pool=pool,
            groups=groups,
        )
        step_rows = _step_rows(
            result,
            run_id=run_id,
            parent_id=parent_id,
            arm_id=arm,
            training_seed=training_seed,
            epoch=121,
            parent_global_step=int(payload["global_step"]),
            epoch_plan=epoch_plan,
        )
        occurrence_path = root / "04_ledgers" / "occurrence" / f"run_id={run_id}" / "epoch=0121" / "part-00000.parquet"
        step_path = root / "04_ledgers" / "optimizer_step" / f"run_id={run_id}" / "epoch=0121" / "part-00000.parquet"
        telemetry_path = root / "04_ledgers" / "telemetry" / f"run_id={run_id}" / "epoch=0121" / "part-00000.parquet"
        occurrence_manifest = write_occurrence_partition(occurrence_rows, occurrence_path)
        step_manifest = write_step_partition(step_rows, step_path)
        telemetry_manifest = write_zstd_parquet(
            [asdict(row) for row in telemetry_rows],
            telemetry_path,
            schema_version="stage1.sctsr.resource_telemetry.v1",
        )

        branch_payload = build_checkpoint_payload(
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=200,
            global_step=int(payload["global_step"]) + result.optimizer_steps,
            base_sampler_generation=200,
            canonical_training_lock_sha256="A" * 64,
            initial_checkpoint_sha256="B" * 64,
            base_manifest_sha256="C" * 64,
            training_seed=training_seed,
            source_tree_digest=source_digest,
            runtime_config_digest=contract_digest,
            asset_registry_digest=asset_digest,
        )
        branch_checkpoint = root / "05_checkpoints" / f"{run_id}_e200.pt"
        branch_sha = save_checkpoint_atomic(branch_checkpoint, branch_payload)
        lineage = BranchLineage.create(
            logical_run_id=run_id,
            parent_id=parent_id,
            parent_checkpoint_path=parent_checkpoint.relative_to(root).as_posix(),
            parent_checkpoint_sha256=parent_sha,
            training_seed=training_seed,
            arm_id=arm.value,
            child_source_tree_digest=source_digest,
            child_contract_digest=contract_digest,
        )
        lineage.validate(
            parent_sha=parent_sha,
            training_seed=training_seed,
            arm_id=arm.value,
            source_digest=source_digest,
            contract_digest=contract_digest,
        )
        atomic_write_json(branch_root / "BRANCH_LINEAGE.json", asdict(lineage))
        atomic_write_json(branch_root / "SCHEDULE.json", schedule_to_dict(schedules[arm]))
        atomic_write_json(branch_root / "REPLAY_STEP_PLAN.json", asdict(replay_plan))

        predictions = _predict(model, fixture, run_id=run_id, arm_id=arm, checkpoint_sha=branch_sha, source_tree_digest=source_digest)
        probability_counts = Counter(row.p_defect_raw for row in predictions)
        maximum_probability_tie_size = max(probability_counts.values())
        if maximum_probability_tie_size < 2:
            raise RuntimeError("Synthetic prediction fixture did not produce a real probability tie")
        prediction_path = root / "06_predictions" / f"run_id={run_id}" / "epoch=0200" / "predictions.parquet"
        prediction_binding = PredictionArtifactBinding(
            checkpoint_path=branch_checkpoint.as_posix(),
            checkpoint_sha256=branch_sha,
            checkpoint_epoch=200,
            model_variant="MODEL_SYNTHETIC_CANARY",
            source_tree_digest=source_digest,
            training_seed=training_seed,
            split_role="synthetic",
            split_manifest_path=predictions[0].split_manifest_path,
            split_manifest_sha256=predictions[0].split_manifest_sha256,
            evaluation_mode="synthetic",
            selection_semantic=SYNTHETIC_MARKER,
        )
        prediction_manifest, prediction_summary = write_prediction_artifact(
            predictions,
            prediction_path,
            formal_endpoint=False,
            binding=prediction_binding,
            expected_sample_labels={row.sample_id: row.y_true for row in predictions},
        )
        eval_start = time.monotonic()
        frontier, frontier_summary = compute_tie_safe_frontier(
            predictions,
            max_fn=95,
            target_tn=68,
            checkpoint_sha256=branch_sha,
            prediction_artifact_sha256=prediction_manifest.sha256,
        )
        _, unreachable_summary = compute_tie_safe_frontier(
            predictions,
            max_fn=95,
            target_tn=10**9,
            checkpoint_sha256=branch_sha,
            prediction_artifact_sha256=prediction_manifest.sha256,
        )
        evaluation_seconds = time.monotonic() - eval_start
        frontier_path = root / "07_evaluation" / f"run_id={run_id}" / "epoch=0200" / "frontier.parquet"
        frontier_manifest, frontier_receipt = write_frontier_artifacts(
            frontier,
            frontier_summary,
            frontier_path=frontier_path,
            summary_path=frontier_path.with_name("frontier_summary.json"),
            evaluation_mode="synthetic",
            split_role="synthetic",
            checkpoint_epoch=200,
        )
        atomic_write_json(
            frontier_path.with_name("unreachable_target_diagnostic.json"),
            {
                "semantic": SYNTHETIC_MARKER,
                "scientific_result": False,
                "synthetic_target_tn": 68,
                "frontier_summary": asdict(frontier_summary),
                "unreachable_fixture_target_tn": 10**9,
                "unreachable_fixture_reachable": unreachable_summary.target_tn_reachable,
                "prediction_summary": prediction_summary,
                "frontier_partition_sha256": frontier_manifest.sha256,
                "frontier_receipt": frontier_receipt,
            },
        )

        exposure_row = build_exposure_row(
            run_id=run_id,
            parent_id=parent_id,
            arm_id=arm.value,
            training_seed=training_seed,
            epoch=121,
            base_denominator=SYNTHETIC_BASE_DENOMINATOR,
            replay_rate_numerator=epoch_plan.rate.numerator,
            replay_rate_denominator=epoch_plan.rate.denominator,
            replay_sample_ids=epoch_plan.sample_ids,
            optimizer_steps=result.optimizer_steps,
            expected_optimizer_steps=len(base_loader),
            base_order_digest=result.base_order_digest,
            base_augmentation_digest=result.base_augmentation_digest,
            schedule_digest=schedules[arm].plan_digest,
            identity_pool_digest=schedules[arm].identity_pool_digest,
            occurrence_partition_sha=occurrence_manifest.sha256,
            step_partition_sha=step_manifest.sha256,
            telemetry_partition_sha=telemetry_manifest.sha256,
            checkpoint_sha=branch_sha,
            cumulative_occurrences=result.replay_occurrences,
            ema_updates_delta=result.ema_updates_delta,
            scheduler_epoch_transitions_delta=result.scheduler_epoch_transitions_delta,
            write_seconds=0.0,
            dataloader_wait_seconds=sum(record.dataloader_wait_seconds for record in result.records),
            training_seconds=training_seconds,
            evaluation_seconds=evaluation_seconds,
            disk_bytes_written=occurrence_manifest.bytes + step_manifest.bytes + telemetry_manifest.bytes + prediction_manifest.bytes + frontier_manifest.bytes,
            transaction_generation=1,
            actual_base_denominator=result.base_occurrences,
        )
        exposure_path = root / "04_ledgers" / "exposure" / f"run_id={run_id}" / "epoch=0121" / "part-00000.parquet"
        exposure_manifest = write_exposure_partition([exposure_row], exposure_path)

        # Exercise atomic epoch publish and recovery pointer.
        transaction_root = branch_root / "epoch_transactions"
        tx = EpochTransaction(transaction_root, run_id, 121, 1, root / "09_quarantine").begin()
        tx.write_json("transaction_summary.json", {"semantic": SYNTHETIC_MARKER, "occurrence_sha256": occurrence_manifest.sha256, "step_sha256": step_manifest.sha256})
        generation_manifest = tx.commit()
        validate_recovery_pointer(transaction_root.parent / "ROLLING_RECOVERY_POINTER.json")
        if find_last_complete_epoch(transaction_root) is None:
            raise RuntimeError("Synthetic transaction did not publish a complete generation")

        branch_receipt = {
            "schema_version": "stage1.sctsr.synthetic_branch_receipt.v1",
            "semantic": SYNTHETIC_MARKER,
            "scientific_result": False,
            "run_id": run_id,
            "arm_id": arm.value,
            "parent_checkpoint_sha256": parent_sha,
            "branch_checkpoint_sha256": branch_sha,
            "lineage_digest": lineage.lineage_digest,
            "optimizer_steps": result.optimizer_steps,
            "base_occurrences": result.base_occurrences,
            "replay_occurrences": result.replay_occurrences,
            "ema_updates_delta": result.ema_updates_delta,
            "scheduler_transitions_delta": result.scheduler_epoch_transitions_delta,
            "occurrence_partition": asdict(occurrence_manifest),
            "step_partition": asdict(step_manifest),
            "exposure_partition": asdict(exposure_manifest),
            "telemetry_partition": asdict(telemetry_manifest),
            "prediction_partition": asdict(prediction_manifest),
            "frontier_partition": asdict(frontier_manifest),
            "frontier_point_count": len(frontier),
            "maximum_probability_tie_size": maximum_probability_tie_size,
            "unreachable_fixture_pass": unreachable_summary.target_tn_reachable is False,
            "generation_digest": generation_manifest["generation_digest"],
        }
        atomic_write_json(branch_root / "BRANCH_RECEIPT.json", branch_receipt)
        branch_summaries[arm.value] = branch_receipt
        for logical_epoch in range(121, 201):
            logical_index.add(
                LogicalArtifactEntry(
                    logical_run_id=run_id,
                    logical_epoch=logical_epoch,
                    physical_owner_type="CHILD",
                    physical_run_id=run_id,
                    artifact_relative_path=f"03_branch/{arm.value}/BRANCH_RECEIPT.json",
                    artifact_sha256=sha256_file(branch_root / "BRANCH_RECEIPT.json"),
                    checkpoint_sha256=branch_sha,
                    source_tree_digest=source_digest,
                    lineage_digest=lineage.lineage_digest,
                )
            )

    checkpoint_resume_receipt = _exercise_checkpoint_resume(
        root,
        parent_checkpoint=parent_checkpoint,
        parent_sha256=parent_sha,
        fixture=fixture,
        schedule=schedules[ArmId.T_U],
        training_seed=training_seed,
        source_tree_digest=source_digest,
        contract_digest=contract_digest,
        asset_registry_digest=asset_digest,
    )

    # Exercise every failure mode required by the taskbook. Each transaction receives a
    # distinct logical epoch so its quarantine path cannot overwrite another fault.
    expected_failure_codes = {
        FaultKind.KILL: "SYNTHETIC_KILL_INJECTION",
        FaultKind.OOM: "OOM_FIXED_CONTRACT_ABORT",
        FaultKind.DISK_FULL: "DISK_SPACE_PRECHECK_FAILED",
        FaultKind.CORRUPT_RECEIPT: "RESUME_GENERATION_MISMATCH",
        FaultKind.HALF_WRITTEN_JSON: "ATOMIC_TRANSACTION_INCOMPLETE",
        FaultKind.HALF_WRITTEN_PARQUET: "ATOMIC_TRANSACTION_INCOMPLETE",
    }
    failure_injections: list[dict[str, Any]] = []
    failure_root = root / "03_branch" / "FAILURE_INJECTION" / "epoch_transactions"
    for offset, fault in enumerate(FaultKind):
        epoch = 122 + offset
        transaction = EpochTransaction(
            failure_root,
            f"SYNTH_FAILURE_{fault.value}",
            epoch,
            1,
            root / "09_quarantine",
        ).begin()
        partial_relative_path = f"partial_{fault.value.lower()}"
        if fault is FaultKind.HALF_WRITTEN_JSON:
            partial_relative_path += ".json"
            partial_path = transaction.inprogress / partial_relative_path
            partial_path.write_bytes(b'{"semantic":"SYNTHETIC_NOT_SCIENTIFIC_RESULT","status":')
        elif fault is FaultKind.HALF_WRITTEN_PARQUET:
            partial_relative_path += ".parquet"
            partial_path = transaction.inprogress / partial_relative_path
            partial_path.write_bytes(b"PAR1\x15\x00SYNTHETIC_TRUNCATED_PAGE")
        elif fault is FaultKind.CORRUPT_RECEIPT:
            partial_relative_path += ".receipt.json"
            partial_path = transaction.inprogress / partial_relative_path
            partial_path.write_bytes(b'{"status":"COMPLETE","sha256":')
        else:
            partial_relative_path += ".json"
            partial_path = transaction.write_json(
                partial_relative_path,
                {"semantic": SYNTHETIC_MARKER, "fault": fault.value, "status": "INPROGRESS"},
            )

        observed_code = None
        exception_type = None
        try:
            inject_fault(fault)
        except SctsrError as exc:
            observed_code = exc.code.value
            exception_type = type(exc).__name__
        except RuntimeError as exc:
            observed_code = str(exc)
            exception_type = type(exc).__name__
        else:  # pragma: no cover - a registered fault must always raise
            raise RuntimeError(f"Fault injection unexpectedly returned: {fault.value}")

        expected_code = expected_failure_codes[fault]
        if observed_code != expected_code:
            raise RuntimeError(
                f"Failure injection code mismatch for {fault.value}: {observed_code!r} != {expected_code!r}"
            )
        quarantine_path = transaction.abort(f"{fault.value}:{observed_code}")
        quarantined_partial = quarantine_path / partial_relative_path
        if transaction.complete.exists() or not quarantine_path.exists() or not quarantined_partial.exists():
            raise RuntimeError(f"Synthetic {fault.value} injection did not quarantine the partial generation")
        if fault in {FaultKind.HALF_WRITTEN_JSON, FaultKind.CORRUPT_RECEIPT}:
            try:
                json.loads(quarantined_partial.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            else:
                raise RuntimeError(f"Synthetic {fault.value} fixture is not actually corrupt")
        if fault is FaultKind.HALF_WRITTEN_PARQUET:
            raw = quarantined_partial.read_bytes()
            if raw.endswith(b"PAR1"):
                raise RuntimeError("Synthetic half-written Parquet fixture unexpectedly has a canonical footer")
        failure_injections.append(
            {
                "fault": fault.value,
                "status": "PASS",
                "expected_error_code": expected_code,
                "observed_error_code": observed_code,
                "exception_type": exception_type,
                "logical_epoch": epoch,
                "quarantine_path": quarantine_path.relative_to(root).as_posix(),
                "partial_artifact": quarantined_partial.relative_to(root).as_posix(),
                "partial_artifact_bytes": quarantined_partial.stat().st_size,
                "partial_artifact_sha256": sha256_file(quarantined_partial),
            }
        )

    failure_summary = {
        "schema_version": "stage1.sctsr.synthetic_failure_injection_summary.v1",
        "semantic": SYNTHETIC_MARKER,
        "scientific_result": False,
        "status": "PASS",
        "faults": failure_injections,
    }
    failure_summary["failure_injection_digest"] = stable_digest(failure_injections)
    atomic_write_json(root / "08_receipts" / "FAILURE_INJECTION_SUMMARY.json", failure_summary)

    logical_index.validate()
    logical_index.write(root / "ARTIFACT_INDEX_LOGICAL.json")
    resume_identity = ResumeIdentity(
        logical_run_id="SYNTHETIC_ALL_ARMS",
        parent_sha256=parent_sha,
        arm_id="MULTI_ARM_CANARY",
        training_seed=training_seed,
        source_tree_digest=source_digest,
        contract_digest=contract_digest,
        asset_registry_digest=asset_digest,
        generation_chain_digest=logical_index.digest,
    )
    resume_identity.validate(resume_identity)
    atomic_write_json(root / "08_receipts" / "RESUME_IDENTITY.json", asdict(resume_identity))

    # This is deliberately not a CompletionAudit.  A compressed synthetic run
    # cannot prove the full-repository v3 regression, formal asset identity, or
    # final self-audit gates required by canonical implementation closeout.
    synthetic_mechanism_audit = {
        "schema_version": "stage1.sctsr.synthetic_mechanism_audit.v1",
        "status": "PASS_SYNTHETIC_MECHANISMS_ONLY",
        "semantic": SYNTHETIC_MARKER,
        "source_tree_digest": source_digest,
        "checks": {
            "contract": "PASS",
            "identity_pools": "PASS",
            "r2_zero_overlap_exact_quota": "PASS",
            "schedule_parity": "PASS",
            "common_parent": "PASS",
            "fixed_base_step": "PASS",
            "replay_gradient": "PASS",
            "rng_isolation": "PASS",
            "bn_isolation": "PASS",
            "occurrence_ledger": "PASS",
            "step_ledger": "PASS",
            "exposure_ledger": "PASS",
            "telemetry": "PASS",
            "prediction_identity": "PASS",
            "tie_safe_frontier": "PASS",
            "checkpoint_resume": "PASS",
            "recovery_and_quarantine": "PASS",
            "formal_seed_registry": "NOT_ASSESSED_SYNTHETIC",
            "val_target": "NOT_ASSESSED_SYNTHETIC",
            "formal_training_release": "NOT_ASSESSED_SYNTHETIC",
            "v3_regression": "NOT_ASSESSED_SYNTHETIC",
            "full_implementation_completion": "NOT_ASSESSED_SYNTHETIC",
        },
        "not_assessed_reasons": {
            "formal_seed_registry": "Formal discovery/confirmation seeds are frozen only by a future release authority.",
            "val_target": "No independent identity- and group-disjoint val_target is registered.",
            "formal_training_release": "No signed SCTSR v4 formal training release exists.",
            "v3_regression": "A synthetic mechanism canary is not the full v3 regression command.",
            "full_implementation_completion": "Canonical implementation completion is produced only by the full repository self-audit.",
        },
        "formal_training_started": False,
        "method_effectiveness_claimed": False,
    }
    atomic_write_json(root / "08_receipts" / "SYNTHETIC_MECHANISM_AUDIT.json", synthetic_mechanism_audit)

    run_manifest = {
        "schema_version": "stage1.sctsr.synthetic_run_manifest.v1",
        "semantic": SYNTHETIC_MARKER,
        "scientific_result": False,
        "compressed_timeline": True,
        "mechanism_parent_epochs_executed": 1,
        "mechanism_branch_epochs_executed_per_arm": 1,
        "logical_anchors_exercised": [120, 121, 160, 200],
        "training_seed": training_seed,
        "parent_checkpoint_sha256": parent_sha,
        "source_tree_digest": source_digest,
        "contract_digest": contract_digest,
        "asset_registry_digest": asset_digest,
        "eight_arms": [arm.arm_id.value for arm in default_phase1_arms()],
        "branch_summaries": branch_summaries,
        "selection_evidence": selection_evidence,
        "failure_injections": failure_injections,
        "failure_injection_summary": "08_receipts/FAILURE_INJECTION_SUMMARY.json",
        "checkpoint_resume_receipt": "08_receipts/CHECKPOINT_RESUME_RECEIPT.json",
        "checkpoint_resume_status": checkpoint_resume_receipt["status"],
        "formal_training_started": False,
        "engineering_gate_generated": False,
        "assignments_generated": False,
        "pilot_release_generated": False,
        "blind_holdout_opened": False,
        "selector_trained": False,
        "method_effectiveness_claimed": False,
    }
    atomic_write_json(root / "RUN_MANIFEST.json", run_manifest)
    # Write the terminal receipt before indexing so the artifact index is complete.
    pre_index_count = sum(1 for path in root.rglob("*") if path.is_file() and path.name != "ARTIFACT_INDEX.json")
    overall = {
        "schema_version": "stage1.sctsr.synthetic_canary_receipt.v1",
        "status": "PASS",
        "semantic": SYNTHETIC_MARKER,
        "scientific_result": False,
        "output_root": root.as_posix(),
        "source_tree_digest": source_digest,
        "parent_checkpoint_sha256": parent_sha,
        "arms_completed": sorted(branch_summaries),
        "artifact_count_excluding_index": pre_index_count + 1,
        "artifact_index_binding": "SEE_ROOT_ARTIFACT_INDEX_JSON",
        "failure_injection_count": len(failure_injections),
        "formal_training_started": False,
        "engineering_gate_generated": False,
        "assignments_generated": False,
        "pilot_release_generated": False,
        "blind_holdout_opened": False,
        "selector_trained": False,
        "method_effectiveness_claimed": False,
    }
    atomic_write_json(root / "08_receipts" / "SYNTHETIC_CANARY_RECEIPT.json", overall)
    artifact_index = _artifact_index(root)
    atomic_write_json(root / "ARTIFACT_INDEX.json", artifact_index)
    overall["artifact_count"] = len(artifact_index["files"])
    overall["artifact_index_digest"] = artifact_index["artifact_index_digest"]
    return overall
