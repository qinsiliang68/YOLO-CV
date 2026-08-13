from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from .branch_lineage import BranchLineage
from .base_rng import BaseEpochRngReceipt, prepare_counter_domain_base_loader
from .checkpointing import build_checkpoint_payload, load_checkpoint, save_checkpoint_atomic
from .contracts import require_synthetic_or_authorized
from .epoch_transaction import EpochTransaction, GenerationIdentity
from .errors import ErrorCode, SctsrError
from .evidence_runtime import EpochEvidenceRecorder, ReplayHistoryState, SampleEvidence, sample_evidence_from_trainer
from .filesystem import windows_safe_resolved_path
from .logical_artifact_index import LogicalArtifactEntry, LogicalArtifactIndex
from .replay_step_plan import build_replay_step_plan
from .rng_isolation import capture_global_rng
from .schedule import SchedulePlan
from .serialization import atomic_write_json, sha256_file, stable_digest
from .ultralytics_overlay import run_ultralytics_classification_epoch


CANONICAL_BATCH_SIZE = 128
CANONICAL_BASE_DENOMINATOR = 120_000
CANONICAL_BASE_STEPS = 938
KEEP_CHECKPOINTS = {120, 140, 150, 160, 180, 200}


@dataclass(frozen=True, slots=True)
class FormalIdentity:
    training_seed: int
    canonical_training_lock_sha256: str
    initial_checkpoint_sha256: str
    base_manifest_sha256: str
    source_tree_digest: str
    runtime_config_digest: str
    asset_registry_digest: str


def _base_batch_sizes(trainer: Any) -> tuple[int, ...]:
    dataset_size = len(trainer.train_loader.dataset)
    if dataset_size != CANONICAL_BASE_DENOMINATOR:
        raise SctsrError(
            ErrorCode.DENOMINATOR_IDENTITY_MISMATCH,
            "Formal base dataset must contain exactly the frozen 120,000 optimizer-visible occurrences",
            observed=dataset_size, expected=CANONICAL_BASE_DENOMINATOR,
        )
    full, tail = divmod(dataset_size, CANONICAL_BATCH_SIZE)
    sizes = [CANONICAL_BATCH_SIZE] * full
    if tail:
        sizes.append(tail)
    if len(sizes) != CANONICAL_BASE_STEPS:
        raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "Formal base process must have 938 steps", observed=len(sizes), expected=CANONICAL_BASE_STEPS)
    if len(trainer.train_loader) != CANONICAL_BASE_STEPS:
        raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "Upstream DataLoader does not have 938 base batches")
    return tuple(sizes)


def _empty_replay_provider(*_args: Any, **_kwargs: Any) -> Mapping[str, Any]:
    raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "No-replay parent requested a replay batch")


def _restore_trainer(payload: Mapping[str, Any], trainer: Any) -> None:
    trainer.model.load_state_dict(payload["model_state"])
    trainer.optimizer.load_state_dict(payload["optimizer_state"])
    trainer.scheduler.load_state_dict(payload["scheduler_state"])
    trainer.scaler.load_state_dict(payload["scaler_state"])
    ema_state = payload.get("ema_state", {})
    if getattr(trainer, "ema", None) is not None:
        if hasattr(trainer.ema, "load_state_dict") and ema_state:
            trainer.ema.load_state_dict(ema_state)
        elif hasattr(trainer.ema, "ema") and isinstance(ema_state, Mapping):
            model_state = ema_state.get("ema_model_state", ema_state.get("shadow", {}))
            if model_state:
                trainer.ema.ema.load_state_dict(model_state)
        trainer.ema.updates = int(payload.get("ema_updates", 0))
    from .rng_isolation import restore_global_rng
    restore_global_rng(payload["rng_state"])


def _checkpoint_payload(trainer: Any, identity: FormalIdentity, *, epoch: int, global_step: int) -> dict[str, Any]:
    return build_checkpoint_payload(
        model=trainer.model, ema=getattr(trainer, "ema", None), optimizer=trainer.optimizer,
        scheduler=trainer.scheduler, scaler=trainer.scaler, epoch=epoch, global_step=global_step,
        base_sampler_generation=epoch, canonical_training_lock_sha256=identity.canonical_training_lock_sha256,
        initial_checkpoint_sha256=identity.initial_checkpoint_sha256, base_manifest_sha256=identity.base_manifest_sha256,
        training_seed=identity.training_seed, source_tree_digest=identity.source_tree_digest,
        runtime_config_digest=identity.runtime_config_digest, asset_registry_digest=identity.asset_registry_digest,
    )


def _evidence_enabled(execution_mode: str, requested: bool | None) -> bool:
    enabled = execution_mode == "formal" if requested is None else bool(requested)
    if execution_mode == "formal" and not enabled:
        raise SctsrError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Formal SCTSR execution may not disable the taskbook evidence chain",
        )
    return enabled


def _run_transactional_epoch(
    *,
    trainer: Any,
    identity: FormalIdentity,
    root: Path,
    run_id: str,
    parent_id: str,
    arm_id: str,
    epoch: int,
    replay_plan: Any,
    replay_batch_provider: Callable[[Sequence[str], int, int, int], Mapping[str, Any]],
    global_step_start: int,
    identity_policy: str,
    schedule_family: str,
    fallback_state: str,
    rate_numerator: int,
    rate_denominator: int,
    schedule_digest: str,
    identity_pool_digest: str,
    pool_multiplicity_targets: Mapping[str, int],
    sample_evidence: Mapping[str, SampleEvidence],
    history: ReplayHistoryState,
    previous_checkpoint_sha256: str,
    previous_generation_digest: str,
    base_rng_receipt: BaseEpochRngReceipt,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run, checkpoint, validate and publish exactly one epoch generation."""

    generation = 1
    transaction = EpochTransaction(
        root / "03_epoch_transactions",
        run_id,
        epoch,
        generation,
        quarantine_root=root / "09_quarantine",
        identity=GenerationIdentity(
            parent_sha256=previous_checkpoint_sha256,
            arm_id=arm_id,
            training_seed=identity.training_seed,
            source_tree_digest=identity.source_tree_digest,
            contract_digest=identity.runtime_config_digest,
            asset_registry_digest=identity.asset_registry_digest,
            rng_state_digest=capture_global_rng().digest(),
            previous_generation_digest=previous_generation_digest,
        ),
    ).begin()
    recorder = EpochEvidenceRecorder(
        transaction=transaction,
        parent_id=parent_id,
        arm_id=arm_id,
        training_seed=identity.training_seed,
        sample_evidence=sample_evidence,
        identity_policy=identity_policy,
        schedule_family=schedule_family,
        fallback_state=fallback_state,
        rate_numerator=rate_numerator,
        rate_denominator=rate_denominator,
        schedule_digest=schedule_digest,
        identity_pool_digest=identity_pool_digest,
        pool_multiplicity_targets=pool_multiplicity_targets,
        expected_base_denominator=CANONICAL_BASE_DENOMINATOR,
        expected_optimizer_steps=CANONICAL_BASE_STEPS,
        global_step_start=global_step_start,
        history=history,
        artifact_root=root,
    )
    try:
        result = run_ultralytics_classification_epoch(
            trainer=trainer,
            replay_plan=replay_plan,
            replay_batch_provider=replay_batch_provider,
            training_seed=identity.training_seed,
            epoch=epoch,
            global_step_start=global_step_start,
            step_receipt_sink=recorder.step_sink,
            occurrence_event_sink=recorder.occurrence_sink,
            replay_rate_numerator=rate_numerator,
            replay_rate_denominator=rate_denominator,
            row_generation=generation,
        )
        result["base_rng_domain_receipt"] = base_rng_receipt.as_dict()
        checkpoint_sha = save_checkpoint_atomic(
            recorder.checkpoint_path,
            _checkpoint_payload(
                trainer,
                identity,
                epoch=epoch,
                global_step=int(result["global_step_end"]),
            ),
        )
        evidence_summary = recorder.finalize(runtime_result=result, checkpoint_sha256=checkpoint_sha)
        generation_manifest = transaction.commit()
    except BaseException:
        recorder.abort()
        transaction.abort("EPOCH_RUNTIME_OR_EVIDENCE_FAILURE")
        raise
    published_checkpoint = transaction.complete / recorder.checkpoint_relative
    if sha256_file(published_checkpoint) != checkpoint_sha:
        raise SctsrError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Published epoch checkpoint differs from the transaction-bound checkpoint",
            artifact_path=str(published_checkpoint),
        )
    evidence_receipt = {
        "status": "EPOCH_GENERATION_COMPLETE",
        "generation_digest": generation_manifest["generation_digest"],
        "generation_manifest_sha256": generation_manifest["generation_manifest_sha256"],
        "receipt_chain_digest": generation_manifest["receipt_chain_digest"],
        "checkpoint_path": published_checkpoint.as_posix(),
        "checkpoint_sha256": checkpoint_sha,
        "evidence_summary_digest": stable_digest(evidence_summary),
        "transaction_complete_path": transaction.complete.as_posix(),
    }
    return result, evidence_receipt


def _assert_expected_checkpoint_payload(payload: Mapping[str, Any], identity: FormalIdentity) -> None:
    expected = {
        "training_seed": identity.training_seed,
        "canonical_training_lock_sha256": identity.canonical_training_lock_sha256,
        "initial_checkpoint_sha256": identity.initial_checkpoint_sha256,
        "base_manifest_sha256": identity.base_manifest_sha256,
        "source_tree_digest": identity.source_tree_digest,
        "runtime_config_digest": identity.runtime_config_digest,
        "asset_registry_digest": identity.asset_registry_digest,
    }
    for field, value in expected.items():
        observed = payload.get(field)
        if observed != value:
            raise SctsrError(
                ErrorCode.BRANCH_LINEAGE_MISMATCH,
                "Parent checkpoint content differs from the prepared branch identity",
                failing_field=field,
                observed=observed,
                expected=value,
            )


def _validate_parent_completion_receipt(parent_checkpoint: str | Path, parent_sha: str, training_seed: int) -> dict[str, Any]:
    checkpoint = Path(parent_checkpoint)
    candidates = tuple(parent / "PARENT_RECEIPT.json" for parent in checkpoint.parents[:7])
    receipt_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if receipt_path is None:
        raise SctsrError(
            ErrorCode.BRANCH_LINEAGE_MISMATCH,
            "Child startup requires the canonical completed parent receipt, not a bare checkpoint path",
            artifact_path=str(checkpoint),
        )
    from .serialization import load_json

    receipt = load_json(receipt_path)
    expected = {
        "parent_id": f"PARENT_{training_seed}",
        "training_seed": training_seed,
        "checkpoint_sha256": parent_sha,
        "epoch_end": 120,
        "best_pt_used": False,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise SctsrError(
                ErrorCode.BRANCH_LINEAGE_MISMATCH,
                "Parent completion receipt differs from the selected checkpoint",
                failing_field=field,
                observed=receipt.get(field),
                expected=value,
                artifact_path=str(receipt_path),
            )
    if receipt.get("status") not in {"FORMAL_PARENT_COMPLETE", "IMPLEMENTED_NOT_FORMALLY_RUN"}:
        raise SctsrError(ErrorCode.BRANCH_LINEAGE_MISMATCH, "Parent receipt is not complete", artifact_path=str(receipt_path))
    return {**receipt, "_receipt_path": receipt_path.as_posix()}


def _manifest_entry(
    *,
    physical_root: Path,
    logical_run_id: str,
    logical_epoch: int,
    owner: str,
    physical_run_id: str,
    source_tree_digest: str,
    lineage_digest: str,
) -> LogicalArtifactEntry:
    from .serialization import load_json

    manifest_path = physical_root / "03_epoch_transactions" / f"epoch_{logical_epoch:04d}.generation_1.complete" / "GENERATION_MANIFEST.json"
    if not manifest_path.is_file():
        raise SctsrError(
            ErrorCode.LOGICAL_ARTIFACT_IDENTITY_MISMATCH,
            "Logical timeline references a missing completed epoch generation",
            epoch=logical_epoch,
            artifact_path=str(manifest_path),
        )
    manifest = load_json(manifest_path)
    checkpoint_rows = [row for row in manifest.get("files", []) if str(row.get("path", "")).endswith(".pt")]
    if len(checkpoint_rows) != 1:
        raise SctsrError(
            ErrorCode.LOGICAL_ARTIFACT_IDENTITY_MISMATCH,
            "Epoch generation must contain exactly one transaction-bound checkpoint",
            epoch=logical_epoch,
            observed=len(checkpoint_rows),
        )
    return LogicalArtifactEntry(
        logical_run_id=logical_run_id,
        logical_epoch=logical_epoch,
        physical_owner_type=owner,
        physical_run_id=physical_run_id,
        artifact_relative_path=manifest_path.relative_to(physical_root).as_posix(),
        artifact_sha256=sha256_file(manifest_path),
        checkpoint_sha256=str(checkpoint_rows[0]["sha256"]),
        source_tree_digest=source_tree_digest,
        lineage_digest=lineage_digest,
    )


def _publish_complete_logical_timeline(
    *,
    child_root: Path,
    parent_root: Path,
    lineage: BranchLineage,
    identity: FormalIdentity,
) -> dict[str, Any]:
    from .serialization import load_json

    entries = [
        _manifest_entry(
            physical_root=parent_root,
            logical_run_id=lineage.logical_run_id,
            logical_epoch=epoch,
            owner="PARENT",
            physical_run_id=lineage.parent_id,
            source_tree_digest=identity.source_tree_digest,
            lineage_digest="NOT_APPLICABLE_PARENT",
        )
        for epoch in range(1, 121)
    ]
    entries.extend(
        _manifest_entry(
            physical_root=child_root,
            logical_run_id=lineage.logical_run_id,
            logical_epoch=epoch,
            owner="CHILD",
            physical_run_id=lineage.logical_run_id,
            source_tree_digest=identity.source_tree_digest,
            lineage_digest=lineage.lineage_digest,
        )
        for epoch in range(121, 201)
    )
    logical = LogicalArtifactIndex(entries)
    logical.validate(require_complete_timeline=True, logical_run_id=lineage.logical_run_id)
    index_path = child_root / "ARTIFACT_INDEX.json"
    generation_index = load_json(index_path)
    combined = {
        "schema_version": "stage1.sctsr.combined_artifact_index.v1",
        "epoch_generations": generation_index["epoch_generations"],
        "epoch_generation_index_digest": generation_index["epoch_generation_index_digest"],
        "logical_run_id": lineage.logical_run_id,
        "logical_timeline": [asdict(entry) for entry in sorted(entries, key=lambda item: item.logical_epoch)],
        "logical_timeline_digest": logical.digest,
        "physical_parent_root": parent_root.as_posix(),
        "physical_child_root": child_root.as_posix(),
    }
    atomic_write_json(index_path, combined)
    return {
        "path": index_path.as_posix(),
        "sha256": sha256_file(index_path),
        "logical_timeline_digest": logical.digest,
        "logical_epoch_count": len(entries),
    }


def run_prepared_common_parent(
    *, trainer: Any, identity: FormalIdentity, output_root: str | Path,
    release_authorization: str | Path | None, execution_mode: str = "formal",
    collect_epoch_evidence: bool | None = None,
) -> dict[str, Any]:
    """Run the formal E1-E120 no-replay common parent on a prepared trainer.

    A prepared trainer has its model, optimizer, scheduler, scaler, EMA and
    canonical base DataLoader initialized, but has not entered the upstream
    training loop. The function deliberately bypasses upstream final_eval so
    ``best.pt`` can never influence SCTSR.
    """

    require_synthetic_or_authorized(execution_mode, release_authorization)
    sizes = _base_batch_sizes(trainer)
    evidence_enabled = _evidence_enabled(execution_mode, collect_epoch_evidence)
    sample_evidence = sample_evidence_from_trainer(trainer) if evidence_enabled else {}
    root = windows_safe_resolved_path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    global_step = 0
    epoch_receipts = []
    history = ReplayHistoryState()
    previous_checkpoint_sha = identity.initial_checkpoint_sha256
    previous_generation_digest = stable_digest(
        {"role": "COMMON_PARENT_START", "initial_checkpoint_sha256": identity.initial_checkpoint_sha256}
    )
    final_evidence: dict[str, Any] | None = None
    no_replay_schedule_digest = stable_digest({"role": "COMMON_PARENT_NR", "epochs": [1, 120]})
    no_replay_pool_digest = stable_digest({"role": "NO_REPLAY_IDENTITY_POOL"})
    for epoch in range(1, 121):
        sampler = getattr(trainer.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch - 1)
        base_rng_receipt = prepare_counter_domain_base_loader(
            trainer,
            training_seed=identity.training_seed,
            epoch=epoch,
        ) if evidence_enabled else None
        plan = build_replay_step_plan(
            run_id=f"PARENT_{identity.training_seed}", arm_id="COMMON_PARENT_NR",
            training_seed=identity.training_seed, epoch=epoch, schedule_family="NR",
            sample_ids=(), base_batch_sizes=sizes,
        )
        if evidence_enabled:
            result, final_evidence = _run_transactional_epoch(
                trainer=trainer,
                identity=identity,
                root=root,
                run_id=f"PARENT_{identity.training_seed}",
                parent_id=f"PARENT_{identity.training_seed}",
                arm_id="COMMON_PARENT_NR",
                epoch=epoch,
                replay_plan=plan,
                replay_batch_provider=_empty_replay_provider,
                global_step_start=global_step,
                identity_policy="NONE",
                schedule_family="NR",
                fallback_state="NOT_APPLICABLE",
                rate_numerator=0,
                rate_denominator=1000,
                schedule_digest=no_replay_schedule_digest,
                identity_pool_digest=no_replay_pool_digest,
                pool_multiplicity_targets={},
                sample_evidence=sample_evidence,
                history=history,
                previous_checkpoint_sha256=previous_checkpoint_sha,
                previous_generation_digest=previous_generation_digest,
                base_rng_receipt=base_rng_receipt,
            )
            previous_checkpoint_sha = final_evidence["checkpoint_sha256"]
            previous_generation_digest = final_evidence["generation_digest"]
        else:
            result = run_ultralytics_classification_epoch(
                trainer=trainer, replay_plan=plan, replay_batch_provider=_empty_replay_provider,
                training_seed=identity.training_seed, epoch=epoch, global_step_start=global_step,
            )
        global_step = int(result["global_step_end"])
        if result["optimizer_steps"] != CANONICAL_BASE_STEPS or result["replay_occurrences"] != 0:
            raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "Common parent violated fixed base process", observed=result)
        epoch_receipts.append(
            {**result, "epoch_evidence": final_evidence if evidence_enabled else "NOT_COLLECTED_SYNTHETIC_ONLY"}
        )
    if evidence_enabled:
        assert final_evidence is not None
        checkpoint = Path(final_evidence["checkpoint_path"])
        checkpoint_sha = str(final_evidence["checkpoint_sha256"])
    else:
        checkpoint = root / "epoch_0120.pt"
        checkpoint_sha = save_checkpoint_atomic(checkpoint, _checkpoint_payload(trainer, identity, epoch=120, global_step=global_step))
    try:
        os.chmod(checkpoint, 0o444)
    except OSError:
        pass
    receipt = {
        "schema_version": "stage1.sctsr.formal_parent_receipt.v1",
        "status": "IMPLEMENTED_NOT_FORMALLY_RUN" if execution_mode != "formal" else "FORMAL_PARENT_COMPLETE",
        "parent_id": f"PARENT_{identity.training_seed}", "training_seed": identity.training_seed,
        "epoch_start": 1, "epoch_end": 120, "global_step": global_step,
        "checkpoint_path": checkpoint.as_posix(), "checkpoint_sha256": checkpoint_sha,
        "epoch_receipt_digest": stable_digest(epoch_receipts), "best_pt_used": False,
        "epoch_evidence_enabled": evidence_enabled,
        "recovery_pointer_path": (root / "ROLLING_RECOVERY_POINTER.json").as_posix() if evidence_enabled else None,
    }
    atomic_write_json(root / "PARENT_RECEIPT.json", receipt)
    return receipt


def run_prepared_branch(
    *, trainer: Any, identity: FormalIdentity, parent_checkpoint: str | Path,
    lineage: BranchLineage, schedule: SchedulePlan,
    replay_batch_provider: Callable[[Sequence[str], int, int, int], Mapping[str, Any]],
    output_root: str | Path, release_authorization: str | Path,
    execution_mode: str = "formal", collect_epoch_evidence: bool | None = None,
) -> dict[str, Any]:
    require_synthetic_or_authorized(execution_mode, release_authorization)
    parent_sha = sha256_file(parent_checkpoint)
    parent_receipt = _validate_parent_completion_receipt(parent_checkpoint, parent_sha, identity.training_seed)
    lineage.validate(
        parent_sha=parent_sha, training_seed=identity.training_seed, arm_id=schedule.arm_id.value,
        source_digest=identity.source_tree_digest, contract_digest=lineage.child_contract_digest,
    )
    payload = load_checkpoint(parent_checkpoint, expected_sha256=parent_sha, expected_epoch=120)
    _assert_expected_checkpoint_payload(payload, identity)
    _restore_trainer(payload, trainer)
    sizes = _base_batch_sizes(trainer)
    evidence_enabled = _evidence_enabled(execution_mode, collect_epoch_evidence)
    sample_evidence = sample_evidence_from_trainer(trainer) if evidence_enabled else {}
    root = windows_safe_resolved_path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    global_step = int(payload["global_step"])
    parent_sha_before = sha256_file(parent_checkpoint)
    epoch_receipts = []
    checkpoints: dict[int, dict[str, str]] = {}
    history = ReplayHistoryState()
    previous_checkpoint_sha = parent_sha
    previous_generation_digest = stable_digest(
        {"role": "BRANCH_START", "parent_checkpoint_sha256": parent_sha, "lineage_digest": lineage.lineage_digest}
    )
    multiplicity = schedule.multiplicity()
    for epoch in range(121, 201):
        sampler = getattr(trainer.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch - 1)
        base_rng_receipt = prepare_counter_domain_base_loader(
            trainer,
            training_seed=identity.training_seed,
            epoch=epoch,
        ) if evidence_enabled else None
        epoch_plan = schedule.epoch(epoch)
        plan = build_replay_step_plan(
            run_id=lineage.logical_run_id, arm_id=schedule.arm_id.value,
            training_seed=identity.training_seed, epoch=epoch,
            schedule_family=epoch_plan.schedule_family, sample_ids=epoch_plan.sample_ids,
            base_batch_sizes=sizes,
        )
        if evidence_enabled:
            result, evidence = _run_transactional_epoch(
                trainer=trainer,
                identity=identity,
                root=root,
                run_id=lineage.logical_run_id,
                parent_id=lineage.parent_id,
                arm_id=schedule.arm_id.value,
                epoch=epoch,
                replay_plan=plan,
                replay_batch_provider=replay_batch_provider,
                global_step_start=global_step,
                identity_policy=epoch_plan.identity_policy,
                schedule_family=epoch_plan.schedule_family,
                fallback_state=epoch_plan.fallback_state,
                rate_numerator=epoch_plan.rate.numerator,
                rate_denominator=epoch_plan.rate.denominator,
                schedule_digest=schedule.plan_digest,
                identity_pool_digest=schedule.identity_pool_digest,
                pool_multiplicity_targets=multiplicity,
                sample_evidence=sample_evidence,
                history=history,
                previous_checkpoint_sha256=previous_checkpoint_sha,
                previous_generation_digest=previous_generation_digest,
                base_rng_receipt=base_rng_receipt,
            )
            previous_checkpoint_sha = evidence["checkpoint_sha256"]
            previous_generation_digest = evidence["generation_digest"]
        else:
            result = run_ultralytics_classification_epoch(
                trainer=trainer, replay_plan=plan, replay_batch_provider=replay_batch_provider,
                training_seed=identity.training_seed, epoch=epoch, global_step_start=global_step,
            )
        global_step = int(result["global_step_end"])
        epoch_receipts.append({**result, "epoch_evidence": evidence if evidence_enabled else "NOT_COLLECTED_SYNTHETIC_ONLY"})
        if epoch in KEEP_CHECKPOINTS:
            if evidence_enabled:
                path = Path(evidence["checkpoint_path"])
                sha = str(evidence["checkpoint_sha256"])
            else:
                path = root / f"epoch_{epoch:04d}.pt"
                sha = save_checkpoint_atomic(path, _checkpoint_payload(trainer, identity, epoch=epoch, global_step=global_step))
            checkpoints[epoch] = {"path": path.as_posix(), "sha256": sha}
    if sha256_file(parent_checkpoint) != parent_sha_before:
        raise SctsrError(ErrorCode.CHILD_MUTATED_PARENT, "Branch mutated the read-only parent checkpoint")
    if execution_mode == "formal" and parent_receipt.get("epoch_evidence_enabled") is not True:
        raise SctsrError(
            ErrorCode.BRANCH_LINEAGE_MISMATCH,
            "Formal branch requires a parent with complete E1-E120 transaction evidence",
        )
    logical_index = None
    if evidence_enabled:
        logical_index = _publish_complete_logical_timeline(
            child_root=root,
            parent_root=Path(parent_receipt["_receipt_path"]).parent,
            lineage=lineage,
            identity=identity,
        )
    receipt = {
        "schema_version": "stage1.sctsr.formal_branch_receipt.v1",
        "status": "FORMAL_BRANCH_COMPLETE" if execution_mode == "formal" else "IMPLEMENTED_NOT_FORMALLY_RUN", "logical_run_id": lineage.logical_run_id,
        "arm_id": schedule.arm_id.value, "training_seed": identity.training_seed,
        "parent_checkpoint_sha256": parent_sha, "lineage_digest": lineage.lineage_digest,
        "epoch_start": 121, "epoch_end": 200, "global_step": global_step,
        "checkpoints": checkpoints, "fixed_formal_endpoint": checkpoints[200],
        "epoch_receipt_digest": stable_digest(epoch_receipts), "best_pt_used": False,
        "epoch_evidence_enabled": evidence_enabled,
        "recovery_pointer_path": (root / "ROLLING_RECOVERY_POINTER.json").as_posix() if evidence_enabled else None,
        "logical_artifact_index": logical_index,
    }
    atomic_write_json(root / "BRANCH_RECEIPT.json", receipt)
    return receipt
