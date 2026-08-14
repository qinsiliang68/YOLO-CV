from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .columnar import StreamingZstdParquetWriter
from .epoch_transaction import EpochTransaction
from .errors import ErrorCode, SctsrError
from .exposure_ledger import write_exposure_partition
from .fixed_step_runtime import OccurrenceEvent
from .ledger_schema import OCCURRENCE_SCHEMA, STEP_SCHEMA
from .occurrence_ledger import OOF_GROUP_SEMANTIC, validate_occurrence_rows
from .rng_isolation import capture_global_rng
from .serialization import atomic_write_json, sha256_file
from .step_ledger import validate_step_rows
from .telemetry import TelemetrySampler, write_telemetry_partition
from .ultralytics_overlay import UpstreamStepReceipt


@dataclass(frozen=True, slots=True)
class SampleEvidence:
    sample_id: str
    y_true: int
    replay_role: str
    oof_fold: int
    oof_group_id: str
    historical_dynamic_bucket: str
    identity_group: str
    oof_reference_probability: float | None = None
    oof_reference_reason: str = "REGISTERED_NOT_AVAILABLE"
    rho_candidate_signal: float | None = None
    rho_reason: str = "REGISTERED_NOT_REPORTED"

    def validate(self) -> None:
        if not self.sample_id or self.y_true not in {0, 1} or not 0 <= self.oof_fold <= 9:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Sample evidence identity/label/fold is invalid", observed=asdict(self))
        for name in ("replay_role", "oof_group_id", "historical_dynamic_bucket"):
            if str(getattr(self, name)).strip() in {"", "UNREGISTERED", "unknown", "UNKNOWN"}:
                raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Sample evidence contains an unregistered field", failing_field=name)
        if self.rho_candidate_signal is not None and self.rho_reason != "CANDIDATE_SIGNAL_NOT_UTILITY":
            raise SctsrError(ErrorCode.CANDIDATE_SIGNAL_AS_UTILITY_FORBIDDEN, "RHO must remain a non-utility candidate signal")


@dataclass(slots=True)
class ReplayHistoryState:
    counts: dict[str, int] = field(default_factory=dict)
    last_epoch: dict[str, int] = field(default_factory=dict)
    cumulative_occurrences: int = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "counts": dict(sorted(self.counts.items())),
            "last_epoch": dict(sorted(self.last_epoch.items())),
            "cumulative_occurrences": self.cumulative_occurrences,
        }


def sample_evidence_from_trainer(trainer: Any) -> dict[str, SampleEvidence]:
    dataset = getattr(trainer.train_loader, "dataset", None)
    identities = getattr(dataset, "identities", None)
    if identities is None:
        raise SctsrError(
            ErrorCode.UPSTREAM_BINDING_FAILED,
            "Formal evidence requires IdentityAugmentingDataset.identities on the frozen base DataLoader",
        )
    rows: dict[str, SampleEvidence] = {}
    for identity in identities:
        row = SampleEvidence(
            sample_id=str(identity.sample_id),
            y_true=int(identity.y_true),
            replay_role=str(identity.replay_role),
            oof_fold=int(identity.oof_fold),
            oof_group_id=str(identity.oof_group_id),
            historical_dynamic_bucket=str(identity.historical_dynamic_bucket),
            identity_group=str(identity.identity_group),
            oof_reference_probability=getattr(identity, "oof_reference_probability", None),
            oof_reference_reason=str(getattr(identity, "oof_reference_reason", "REGISTERED_NOT_AVAILABLE")),
            rho_candidate_signal=getattr(identity, "rho_candidate_signal", None),
            rho_reason=str(getattr(identity, "rho_reason", "REGISTERED_NOT_REPORTED")),
        )
        row.validate()
        if row.sample_id in rows:
            raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "Formal sample evidence contains a duplicate ID", observed=row.sample_id)
        rows[row.sample_id] = row
    return rows


def _pool_id(identity_policy: str) -> str:
    return {
        "T_STRESS": "T_STRESS_POOL",
        "R1_GLOBAL_RANDOM": "R1_GLOBAL_RANDOM_POOL",
        "R2_MATCHED_RANDOM": "R2_MATCHED_RANDOM_POOL",
    }.get(identity_policy, "NOT_APPLICABLE_NO_REPLAY")


class EpochEvidenceRecorder:
    """Stream one real epoch's complete evidence into one generation transaction."""

    def __init__(
        self,
        *,
        transaction: EpochTransaction,
        parent_id: str,
        arm_id: str,
        training_seed: int,
        sample_evidence: Mapping[str, SampleEvidence],
        identity_policy: str,
        schedule_family: str,
        fallback_state: str,
        rate_numerator: int,
        rate_denominator: int,
        schedule_digest: str,
        identity_pool_digest: str,
        pool_multiplicity_targets: Mapping[str, int],
        expected_base_denominator: int,
        expected_optimizer_steps: int,
        global_step_start: int,
        history: ReplayHistoryState,
        artifact_root: str | Path,
    ) -> None:
        if not transaction.inprogress.is_dir():
            raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Evidence recorder requires a begun epoch transaction")
        self.transaction = transaction
        self.run_id = transaction.run_id
        self.epoch = transaction.epoch
        self.generation = transaction.generation
        self.parent_id = parent_id
        self.arm_id = arm_id
        self.training_seed = training_seed
        self.sample_evidence = dict(sample_evidence)
        for row in self.sample_evidence.values():
            row.validate()
        self.identity_policy = identity_policy
        self.schedule_family = schedule_family
        self.fallback_state = fallback_state
        self.rate_numerator = int(rate_numerator)
        self.rate_denominator = int(rate_denominator)
        self.schedule_digest = schedule_digest
        self.identity_pool_digest = identity_pool_digest
        self.pool_multiplicity_targets = {str(key): int(value) for key, value in pool_multiplicity_targets.items()}
        self.expected_base_denominator = int(expected_base_denominator)
        self.expected_optimizer_steps = int(expected_optimizer_steps)
        self.global_step_start = int(global_step_start)
        self.history = history
        self.replay_ids: list[str] = []
        self.base_rows = 0
        self.replay_rows = 0
        self.step_rows = 0
        self.dataloader_wait_seconds = 0.0
        self.started_monotonic = time.monotonic()
        self.epoch_rng_digest_start = capture_global_rng().digest()
        self.occurrence_relative = f"04_ledgers/occurrence/run_id={self.run_id}/epoch={self.epoch:04d}/part-00000.parquet"
        self.step_relative = f"04_ledgers/optimizer_step/run_id={self.run_id}/epoch={self.epoch:04d}/part-00000.parquet"
        self.telemetry_relative = f"04_ledgers/telemetry/run_id={self.run_id}/epoch={self.epoch:04d}/part-00000.parquet"
        self.exposure_relative = f"04_ledgers/exposure/run_id={self.run_id}/epoch={self.epoch:04d}/part-00000.parquet"
        self.summary_relative = "EPOCH_EVIDENCE_SUMMARY.json"
        self.checkpoint_relative = f"05_checkpoints/rolling_epoch_{self.epoch:04d}.generation_{self.generation}.pt"
        transaction.required_relative_paths = (
            self.occurrence_relative,
            self.step_relative,
            self.telemetry_relative,
            self.exposure_relative,
            self.summary_relative,
            self.checkpoint_relative,
        )
        self._closed = False
        self.occurrence_writer: StreamingZstdParquetWriter | None = None
        self.step_writer: StreamingZstdParquetWriter | None = None
        self.telemetry: TelemetrySampler | None = None
        try:
            self.occurrence_writer = StreamingZstdParquetWriter(
                transaction.path_for(self.occurrence_relative),
                schema_version="stage1.sctsr.occurrence_ledger.v1",
                schema=OCCURRENCE_SCHEMA,
            )
            self.step_writer = StreamingZstdParquetWriter(
                transaction.path_for(self.step_relative),
                schema_version="stage1.sctsr.optimizer_step_ledger.v1",
                schema=STEP_SCHEMA,
            )
            self.telemetry = TelemetrySampler(
                run_id=self.run_id,
                arm_id=self.arm_id,
                training_seed=self.training_seed,
                epoch=self.epoch,
                run_path=transaction.inprogress,
                artifact_path=artifact_root,
                row_generation=self.generation,
            ).start()
        except BaseException as primary:
            try:
                self.abort()
            except BaseException as cleanup:
                primary.add_note(f"SCTSR recorder-construction cleanup failure: {type(cleanup).__name__}: {cleanup}")
            raise

    @property
    def checkpoint_path(self) -> Path:
        return self.transaction.path_for(self.checkpoint_relative)

    def occurrence_sink(self, event: OccurrenceEvent) -> None:
        if event.augmentation_seeds is None or len(event.augmentation_seeds) != len(event.sample_ids):
            raise SctsrError(
                ErrorCode.BASE_AUGMENTATION_MISMATCH,
                "Canonical occurrence evidence requires the actual counter-domain augmentation seed for every row",
            )
        probabilities = torch.softmax(event.logits.float(), dim=1)
        rows = []
        for index, sample_id in enumerate(event.sample_ids):
            try:
                evidence = self.sample_evidence[sample_id]
            except KeyError as exc:
                raise SctsrError(ErrorCode.IDENTITY_NOT_IN_BASE, "Occurrence has no registered sample evidence", observed=sample_id) from exc
            is_replay = event.occurrence_role == "REPLAY"
            before = int(self.history.counts.get(sample_id, 0))
            last_epoch = self.history.last_epoch.get(sample_id)
            cumulative_before = self.history.cumulative_occurrences
            if is_replay:
                after = before + 1
                self.history.counts[sample_id] = after
                self.history.last_epoch[sample_id] = self.epoch
                self.history.cumulative_occurrences += 1
                self.replay_ids.append(sample_id)
                self.replay_rows += 1
            else:
                after = 0
                self.base_rows += 1
            cumulative_after = self.history.cumulative_occurrences
            logits = event.logits[index]
            predicted = int(torch.argmax(logits).item())
            if is_replay and last_epoch is not None:
                last_reason, since, since_reason = "PRESENT", self.epoch - last_epoch, "PRESENT"
            elif is_replay:
                last_reason, since, since_reason = "NEVER_REPLAYED", None, "NEVER_REPLAYED"
            else:
                last_epoch, last_reason, since, since_reason = None, "NOT_APPLICABLE_BASE", None, "NOT_APPLICABLE_BASE"
            row = {
                "run_id": self.run_id,
                "parent_id": self.parent_id,
                "arm_id": self.arm_id,
                "training_seed": self.training_seed,
                "epoch": self.epoch,
                "base_batch_index": event.base_step_index,
                "global_step_before": self.global_step_start + event.base_step_index,
                "occurrence_role": event.occurrence_role,
                "occurrence_index_in_step": index,
                "sample_id": sample_id,
                "y_true": evidence.y_true,
                "replay_role": evidence.replay_role if is_replay else "NOT_APPLICABLE_BASE",
                "identity_pool_id": _pool_id(self.identity_policy) if is_replay else "NOT_APPLICABLE_BASE",
                "identity_group": evidence.identity_group if is_replay else "NOT_APPLICABLE_BASE",
                "selection_policy": self.identity_policy if is_replay else "BASE_CANONICAL",
                "selection_reason_code": "PLANNED_REPLAY_STEP_SLOT" if is_replay else "CANONICAL_BASE_OCCURRENCE",
                "oof_fold": evidence.oof_fold,
                "oof_group_id": evidence.oof_group_id,
                "oof_group_semantic": OOF_GROUP_SEMANTIC,
                "historical_dynamic_bucket": evidence.historical_dynamic_bucket,
                "augmentation_seed": int(event.augmentation_seeds[index]),
                "augmentation_trace_digest": event.augmentation_digests[index],
                "replay_count_before": before if is_replay else 0,
                "replay_count_after": after if is_replay else 0,
                "last_replay_epoch": last_epoch,
                "last_replay_epoch_reason": last_reason,
                "epochs_since_last_replay": since,
                "epochs_since_last_replay_reason": since_reason,
                "logit_normal": float(logits[0]),
                "logit_defect": float(logits[1]),
                "p_defect_raw": float(probabilities[index, 1]),
                "ce_unreduced": float(event.per_sample_ce[index]),
                "margin_defect_minus_normal": float(logits[1] - logits[0]),
                "predicted_label_argmax": predicted,
                "correct_argmax": bool(predicted == evidence.y_true),
                "oof_reference_probability": evidence.oof_reference_probability,
                "oof_reference_reason": evidence.oof_reference_reason,
                "rho_candidate_signal": evidence.rho_candidate_signal,
                "rho_reason": evidence.rho_reason,
                "row_generation": self.generation,
                "planned_replay_epoch": self.epoch if is_replay else None,
                "planned_replay_epoch_reason": "PRESENT" if is_replay else "NOT_APPLICABLE_BASE",
                "planned_step_slot": event.base_step_index if is_replay else None,
                "planned_step_slot_reason": "PRESENT" if is_replay else "NOT_APPLICABLE_BASE",
                "cumulative_replay_count_before": cumulative_before,
                "cumulative_replay_count_after": cumulative_after,
                "pool_multiplicity_target": self.pool_multiplicity_targets.get(sample_id, 0) if is_replay else 0,
                "schedule_family": self.schedule_family if is_replay else "BASE_CANONICAL",
                "fallback_state": self.fallback_state if is_replay else "NOT_APPLICABLE_BASE",
            }
            rows.append(row)
        validate_occurrence_rows(rows)
        self.occurrence_writer.append(rows)

    def step_sink(self, receipt: UpstreamStepReceipt) -> None:
        has_replay = receipt.replay_microbatch_size > 0
        row = {
            "run_id": self.run_id,
            "parent_id": self.parent_id,
            "arm_id": self.arm_id,
            "training_seed": self.training_seed,
            "epoch": self.epoch,
            "base_batch_index": receipt.base_step_index,
            "global_step_before": receipt.global_step_before,
            "global_step_after": receipt.global_step_after,
            "base_batch_size": receipt.base_batch_size,
            "replay_microbatch_size": receipt.replay_microbatch_size,
            "replay_rate_numerator": receipt.replay_rate_numerator,
            "replay_rate_denominator": receipt.replay_rate_denominator,
            "base_loss": receipt.base_loss,
            "replay_loss": receipt.replay_loss,
            "combined_loss_for_reporting": receipt.combined_loss_for_reporting,
            "base_loss_items": dict(receipt.base_loss_items),
            "parameter_grad_norm_before_clip": receipt.parameter_grad_norm_before_clip,
            "parameter_grad_norm_after_clip": receipt.parameter_grad_norm_after_clip,
            "clip_max_norm": receipt.clip_max_norm,
            "clip_reason": "PRESENT",
            "optimizer_step_count_delta": receipt.optimizer_step_delta,
            "learning_rates": list(receipt.learning_rates),
            "optimizer_hyperparameters": list(receipt.optimizer_hyperparameters),
            "amp_scale_before": receipt.amp_scale_before,
            "amp_scale_after": receipt.amp_scale_after,
            "amp_reason": "PRESENT",
            "overflow_or_step_skipped": receipt.overflow_or_step_skipped,
            "ema_updates_before": receipt.ema_updates_before,
            "ema_updates_after": receipt.ema_updates_after,
            "scheduler_state_digest": receipt.scheduler_state_digest,
            "warmup_progress": receipt.warmup_progress,
            "bn_digest_before_replay": receipt.bn_before_replay,
            "bn_digest_after_replay_restore": receipt.bn_after_replay,
            "bn_reason": "PRESENT" if has_replay else "NO_REPLAY_IN_STEP",
            "rng_digest_before_base": receipt.rng_before_base,
            "rng_digest_before_replay": receipt.rng_before_replay,
            "rng_digest_after_replay_restore": receipt.rng_after_replay,
            "rng_reason": "PRESENT" if has_replay else "NO_REPLAY_IN_STEP",
            "replay_rng_fork_digest": receipt.replay_rng_fork_digest,
            "replay_rng_fork_reason": "PRESENT" if has_replay else "NO_REPLAY_IN_STEP",
            "base_augmentation_digest": receipt.base_augmentation_digest,
            "replay_augmentation_digest": receipt.replay_augmentation_digest,
            "replay_augmentation_reason": "PRESENT" if has_replay else "NO_REPLAY_IN_STEP",
            "dataloader_wait_seconds": receipt.dataloader_wait_seconds,
            "base_forward_seconds": receipt.base_forward_seconds,
            "replay_forward_seconds": receipt.replay_forward_seconds,
            "backward_seconds": receipt.backward_seconds,
            "optimizer_seconds": receipt.optimizer_seconds,
            "write_buffer_bytes": receipt.write_buffer_bytes,
            "status": "PASS",
            "row_generation": receipt.row_generation,
        }
        validate_step_rows([row])
        # Parquet nested optimizer structs are normalized by the ledger writer;
        # perform the same deterministic normalization before streaming.
        from .step_ledger import _normalize_optimizer_groups

        row["base_loss_items"] = {str(key): float(value) for key, value in row["base_loss_items"].items()}
        row["optimizer_hyperparameters"] = _normalize_optimizer_groups(row["optimizer_hyperparameters"])
        self.step_writer.append([row])
        self.step_rows += 1
        self.dataloader_wait_seconds += float(receipt.dataloader_wait_seconds)

    def finalize(
        self,
        *,
        runtime_result: Mapping[str, Any],
        checkpoint_sha256: str,
        evaluation_seconds: float = 0.0,
    ) -> dict[str, Any]:
        if self._closed:
            raise SctsrError(ErrorCode.ATOMIC_TRANSACTION_INCOMPLETE, "Epoch evidence recorder is already closed")
        telemetry_rows = self.telemetry.stop()
        occurrence_manifest = self.occurrence_writer.close()
        step_manifest = self.step_writer.close()
        self._closed = True
        if self.base_rows != int(runtime_result["base_occurrences"]) or self.replay_rows != int(runtime_result["replay_occurrences"]):
            raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "Occurrence evidence does not conserve runtime exposure")
        if self.step_rows != int(runtime_result["optimizer_steps"]):
            raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "Step evidence does not conserve optimizer steps")
        telemetry_manifest = write_telemetry_partition(telemetry_rows, self.transaction.path_for(self.telemetry_relative))
        if sha256_file(self.checkpoint_path) != checkpoint_sha256:
            raise SctsrError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Epoch checkpoint SHA changed before evidence publication")
        write_start = time.monotonic()
        exposure = {
            "run_id": self.run_id,
            "parent_id": self.parent_id,
            "arm_id": self.arm_id,
            "training_seed": self.training_seed,
            "epoch": self.epoch,
            "denominator_role": "CANONICAL_BASE_OPTIMIZER_VISIBLE_EXPOSURE",
            "base_denominator_planned": self.expected_base_denominator,
            "base_denominator_actual": self.base_rows,
            "rate_numerator": self.rate_numerator,
            "rate_denominator": self.rate_denominator,
            "replay_numerator_planned": self.expected_base_denominator * self.rate_numerator // self.rate_denominator,
            "replay_numerator_actual": self.replay_rows,
            "unique_replay_ids": len(set(self.replay_ids)),
            "repeat_occurrences": self.replay_rows - len(set(self.replay_ids)),
            "cumulative_occurrences": self.history.cumulative_occurrences,
            "multiplicity_min": min(Counter(self.replay_ids).values(), default=0),
            "multiplicity_max": max(Counter(self.replay_ids).values(), default=0),
            "multiplicity_mean": (self.replay_rows / len(set(self.replay_ids))) if self.replay_ids else 0.0,
            "multiplicity_q0": float(min(Counter(self.replay_ids).values(), default=0)),
            "multiplicity_q25": 1.0 if self.replay_ids else 0.0,
            "multiplicity_q50": 1.0 if self.replay_ids else 0.0,
            "multiplicity_q75": 1.0 if self.replay_ids else 0.0,
            "multiplicity_q100": float(max(Counter(self.replay_ids).values(), default=0)),
            "base_optimizer_steps_planned": self.expected_optimizer_steps,
            "base_optimizer_steps_actual": int(runtime_result["optimizer_steps"]),
            "ema_updates_delta": int(runtime_result["ema_updates_delta"]),
            "scheduler_epoch_transitions_delta": int(runtime_result.get("scheduler_epoch_transitions_delta", 1)),
            "base_order_digest": str(runtime_result["base_order_digest"]),
            "base_augmentation_digest": str(runtime_result["base_augmentation_digest"]),
            "replay_schedule_digest": self.schedule_digest,
            "identity_pool_digest": self.identity_pool_digest,
            "occurrence_partition_sha256": occurrence_manifest.sha256,
            "step_partition_sha256": step_manifest.sha256,
            "telemetry_partition_sha256": telemetry_manifest.sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "write_seconds": 0.0,
            "dataloader_wait_seconds": self.dataloader_wait_seconds,
            "training_seconds": time.monotonic() - self.started_monotonic,
            "evaluation_seconds": float(evaluation_seconds),
            "disk_bytes_written": occurrence_manifest.bytes + step_manifest.bytes + telemetry_manifest.bytes + self.checkpoint_path.stat().st_size,
            "transaction_generation": self.generation,
            "validation_status": "PASS",
        }
        exposure["write_seconds"] = time.monotonic() - write_start
        exposure_manifest = write_exposure_partition([exposure], self.transaction.path_for(self.exposure_relative))
        occurrence_summary = occurrence_manifest.as_dict()
        occurrence_summary["path"] = self.occurrence_relative
        step_summary = step_manifest.as_dict()
        step_summary["path"] = self.step_relative
        telemetry_summary = telemetry_manifest.as_dict()
        telemetry_summary["path"] = self.telemetry_relative
        exposure_summary = exposure_manifest.as_dict()
        exposure_summary["path"] = self.exposure_relative
        summary = {
            "schema_version": "stage1.sctsr.epoch_evidence_summary.v1",
            "status": "VALIDATED_READY_FOR_GENERATION_COMMIT",
            "scientific_result": False,
            "run_id": self.run_id,
            "epoch": self.epoch,
            "generation": self.generation,
            "occurrence_partition": occurrence_summary,
            "step_partition": step_summary,
            "telemetry_partition": telemetry_summary,
            "exposure_partition": exposure_summary,
            "checkpoint_sha256": checkpoint_sha256,
            "history_after_epoch": self.history.snapshot(),
            "rng_evidence": {
                "recorder_epoch_start_digest": self.epoch_rng_digest_start,
                "runtime_epoch_start_digest": runtime_result.get("epoch_rng_digest_start"),
                "runtime_epoch_end_digest": runtime_result.get("epoch_rng_digest_end"),
                "finalize_entry_digest": capture_global_rng().digest(),
                "base_counter_domains": runtime_result.get("base_rng_domain_receipt"),
            },
        }
        atomic_write_json(self.transaction.path_for(self.summary_relative), summary)
        return summary

    def abort(self) -> None:
        if not self._closed:
            errors: list[tuple[str, BaseException]] = []
            for role, component, method in (
                ("telemetry stop", self.telemetry, "stop"),
                ("occurrence writer abort", self.occurrence_writer, "abort"),
                ("step writer abort", self.step_writer, "abort"),
            ):
                if component is None:
                    continue
                try:
                    getattr(component, method)()
                except BaseException as exc:
                    errors.append((role, exc))
            self._closed = True
            if errors:
                role, primary = errors[0]
                primary.add_note(f"SCTSR cleanup component failed: {role}")
                for later_role, later in errors[1:]:
                    primary.add_note(f"Additional SCTSR cleanup failure during {later_role}: {type(later).__name__}: {later}")
                raise primary
