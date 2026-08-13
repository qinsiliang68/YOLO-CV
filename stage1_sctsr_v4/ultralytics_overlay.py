from __future__ import annotations

"""Narrow Ultralytics classification overlay for SCTSR v4.

The module intentionally imports no YOLO-CV upstream files at import time.  It is
loaded inside the full YOLO-CV main checkout, where the frozen
``YOLOv11/ultralytics`` package is already importable.  The overlay does not
modify upstream source files and does not extend the base DataLoader length.
"""

import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .bn_isolation import capture_batchnorm_buffers, preserve_batchnorm_buffers
from .errors import ErrorCode, SctsrError
from .replay_step_plan import ReplayStepPlan
from .rng_isolation import capture_global_rng, replay_rng_domain
from .serialization import stable_digest


@dataclass(frozen=True, slots=True)
class UpstreamStepReceipt:
    epoch: int
    base_step_index: int
    global_step_before: int
    global_step_after: int
    base_batch_size: int
    replay_microbatch_size: int
    base_loss: float
    replay_loss: float
    optimizer_step_delta: int
    ema_update_delta: int
    rng_before_replay: str | None
    rng_after_replay: str | None
    bn_before_replay: str | None
    bn_after_replay: str | None


def _trainer_world_size(trainer: Any) -> int:
    return int(getattr(trainer, "world_size", 1) or 1)


def validate_upstream_trainer(trainer: Any, replay_plan: ReplayStepPlan) -> None:
    if _trainer_world_size(trainer) != 1:
        raise SctsrError(ErrorCode.DISTRIBUTED_MODE_NOT_SUPPORTED_IN_V4_PHASE1, "Formal v4 phase 1 is single-process/single-GPU only")
    if int(getattr(trainer, "batch_size", 128)) != 128:
        raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "Canonical batch size is frozen at 128")
    if len(trainer.train_loader) != replay_plan.base_step_count:
        raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "Upstream base DataLoader length differs from materialized replay plan")
    if getattr(trainer, "accumulate", 1) not in {None, 1}:
        raise SctsrError(ErrorCode.REPLAY_ADDED_OPTIMIZER_STEP, "Implicit gradient accumulation is forbidden")
    if str(getattr(trainer.args, "resume", "")).lower().endswith("best.pt"):
        raise SctsrError(ErrorCode.BEST_PT_FORBIDDEN, "best.pt is forbidden")
    if bool(getattr(trainer.args, "compile", False)):
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Compiled model execution is not supported by the v4 replay hook")
    if not callable(getattr(trainer, "optimizer_step", None)):
        raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Frozen upstream optimizer_step method is unavailable")


def _identity_trace(batch: Mapping[str, Any], *, occurrence_role: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    labels = batch.get("cls")
    size = int(labels.shape[0]) if isinstance(labels, torch.Tensor) else -1
    raw_ids = batch.get("sample_ids")
    raw_aug = batch.get("augmentation_digests")
    if isinstance(raw_ids, str) or not isinstance(raw_ids, Sequence):
        raise SctsrError(
            ErrorCode.BASE_ORDER_MISMATCH if occurrence_role == "BASE" else ErrorCode.SCHEDULE_EXPOSURE_MISMATCH,
            f"{occurrence_role} batch lacks a materialized sample identity vector",
        )
    if isinstance(raw_aug, str) or not isinstance(raw_aug, Sequence):
        raise SctsrError(
            ErrorCode.BASE_AUGMENTATION_MISMATCH if occurrence_role == "BASE" else ErrorCode.SCHEDULE_EXPOSURE_MISMATCH,
            f"{occurrence_role} batch lacks a materialized augmentation trace vector",
        )
    sample_ids = tuple(str(value) for value in raw_ids)
    augmentation_digests = tuple(str(value) for value in raw_aug)
    if size < 0 or len(sample_ids) != size or len(augmentation_digests) != size or len(set(sample_ids)) != size:
        raise SctsrError(
            ErrorCode.BASE_ORDER_MISMATCH if occurrence_role == "BASE" else ErrorCode.SCHEDULE_EXPOSURE_MISMATCH,
            f"{occurrence_role} identity/augmentation trace does not match batch cardinality",
            observed={"labels": size, "sample_ids": len(sample_ids), "augmentation_digests": len(augmentation_digests)},
            expected="equal non-duplicate vectors",
        )
    return sample_ids, augmentation_digests


def _preprocess_replay(trainer: Any, batch: Mapping[str, Any]) -> Mapping[str, Any]:
    # ClassificationTrainer.preprocess_batch mutates the mapping.  Copy the top
    # level so caller-owned metadata is not mutated.
    return trainer.preprocess_batch(dict(batch))


def run_ultralytics_classification_epoch(
    *,
    trainer: Any,
    replay_plan: ReplayStepPlan,
    replay_batch_provider: Callable[[Sequence[str], int, int, int], Mapping[str, Any]],
    training_seed: int,
    epoch: int,
    global_step_start: int,
    step_receipt_sink: Callable[[UpstreamStepReceipt], None] | None = None,
) -> dict[str, Any]:
    """Run one fixed-base-step classification epoch through a frozen trainer.

    Base loss is obtained from ``trainer.model(batch)`` exactly as the frozen
    upstream loop does.  Replay contributes ``sum(CE)/128`` to the same
    gradients.  Each base batch calls scaler.step/update, zero_grad and EMA
    exactly once; replay never creates an additional optimizer step.
    """

    validate_upstream_trainer(trainer, replay_plan)
    replay_plan.validate_structure()
    if replay_plan.epoch != epoch or replay_plan.training_seed != training_seed:
        raise SctsrError(
            ErrorCode.CONFIGURATION_MISMATCH,
            "Runtime epoch/seed differs from the materialized replay plan",
            observed={"epoch": epoch, "training_seed": training_seed},
            expected={"epoch": replay_plan.epoch, "training_seed": replay_plan.training_seed},
        )
    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    global_step = int(global_step_start)
    optimizer_steps = 0
    ema_start = int(getattr(trainer.ema, "updates", 0)) if getattr(trainer, "ema", None) is not None else 0
    base_order: list[str] = []
    base_aug: list[str] = []
    replay_occurrences = 0
    base_occurrences = 0
    nb = len(trainer.train_loader)
    warmup_epochs = float(getattr(trainer.args, "warmup_epochs", 0.0) or 0.0)
    warmup_iterations = max(round(warmup_epochs * nb), 100) if warmup_epochs > 0 else -1

    # Upstream trainer advances its scheduler at the epoch boundary.  No replay
    # forward or occurrence may advance it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        trainer.scheduler.step()

    for base_step_index, raw_batch in enumerate(trainer.train_loader):
        base_batch_size = int(raw_batch["cls"].shape[0])
        replay_plan.validate_step(base_step_index, base_batch_size)
        base_occurrences += base_batch_size

        # Reproduce the frozen upstream warmup on base-step coordinates only.
        ni = base_step_index + nb * (epoch - 1)
        if warmup_iterations >= 0 and ni <= warmup_iterations:
            accumulate = max(1, int(np.interp(ni, [0, warmup_iterations], [1, getattr(trainer.args, "nbs", 64) / 128]).round()))
            if accumulate != 1:
                raise SctsrError(ErrorCode.REPLAY_ADDED_OPTIMIZER_STEP, "Frozen SCTSR phase 1 requires one optimizer step per base batch during warmup")
            for group in trainer.optimizer.param_groups:
                initial_lr = float(group.get("initial_lr", group.get("lr", 0.0)))
                target_lr = initial_lr * float(trainer.lf(epoch - 1)) if hasattr(trainer, "lf") else initial_lr
                start_lr = float(getattr(trainer.args, "warmup_bias_lr", 0.1)) if group.get("param_group") == "bias" else 0.0
                group["lr"] = float(np.interp(ni, [0, warmup_iterations], [start_lr, target_lr]))
                if "momentum" in group:
                    group["momentum"] = float(np.interp(ni, [0, warmup_iterations], [float(getattr(trainer.args, "warmup_momentum", 0.8)), float(getattr(trainer.args, "momentum", group["momentum"]))]))

        batch = trainer.preprocess_batch(raw_batch)
        sample_ids, augmentation_digests = _identity_trace(batch, occurrence_role="BASE")
        base_order.extend(sample_ids)
        base_aug.extend(augmentation_digests)

        try:
            with torch.autocast(device_type=trainer.device.type, enabled=bool(trainer.amp)):
                base_loss_raw, trainer.loss_items = trainer.model(batch)
                base_loss = base_loss_raw.sum()
            trainer.scaler.scale(base_loss).backward()
        except RuntimeError as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower():
                raise SctsrError(ErrorCode.OOM_FIXED_CONTRACT_ABORT, "OOM in base path; contract may not auto-reduce batch", epoch=epoch, recoverable=True) from exc
            raise

        replay_ids = replay_plan.per_step_identity_slices[base_step_index]
        replay_loss = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)
        rng_before = rng_after = bn_before = bn_after = None
        if replay_ids:
            cap = int(batch["cls"].shape[0]) // 4
            if len(replay_ids) > cap:
                raise SctsrError(ErrorCode.REPLAY_MICROBATCH_CAP_EXCEEDED, "Replay microbatch exceeds 25% cap", observed=len(replay_ids), expected=cap)
            rng_snapshot = capture_global_rng()
            bn_snapshot = capture_batchnorm_buffers(trainer.model)
            rng_before, bn_before = rng_snapshot.digest(), bn_snapshot.digest()
            try:
                with replay_rng_domain("replay_augmentation", training_seed, epoch, base_step_index):
                    with preserve_batchnorm_buffers(trainer.model):
                        replay_batch = _preprocess_replay(trainer, replay_batch_provider(replay_ids, epoch, base_step_index, training_seed))
                        replay_sample_ids, replay_aug = _identity_trace(replay_batch, occurrence_role="REPLAY")
                        if replay_sample_ids != tuple(replay_ids):
                            raise SctsrError(
                                ErrorCode.SCHEDULE_EXPOSURE_MISMATCH,
                                "Replay provider changed the materialized identity order",
                                observed=replay_sample_ids,
                                expected=tuple(replay_ids),
                            )
                        with torch.autocast(device_type=trainer.device.type, enabled=bool(trainer.amp)):
                            replay_logits = trainer.model(replay_batch["img"])
                            per_sample = F.cross_entropy(replay_logits, replay_batch["cls"].view(-1).long(), reduction="none")
                            replay_loss = per_sample.sum() / 128
                        trainer.scaler.scale(replay_loss).backward()
            except RuntimeError as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower():
                    raise SctsrError(ErrorCode.OOM_FIXED_CONTRACT_ABORT, "OOM in replay path; fixed contract aborts", epoch=epoch, recoverable=True) from exc
                raise
            rng_after = capture_global_rng().digest()
            bn_after = capture_batchnorm_buffers(trainer.model).digest()
            if rng_before != rng_after:
                raise SctsrError(ErrorCode.RNG_NOT_RESTORED, "Replay changed global RNG trajectory")
            if bn_before != bn_after:
                raise SctsrError(ErrorCode.BN_BUFFER_NOT_RESTORED, "Replay changed BatchNorm running buffers")
            replay_occurrences += len(replay_ids)

        ema_before = int(getattr(trainer.ema, "updates", 0)) if getattr(trainer, "ema", None) is not None else 0
        scale_before = float(trainer.scaler.get_scale())
        trainer.optimizer_step()
        scale_after = float(trainer.scaler.get_scale())
        if scale_after < scale_before:
            raise SctsrError(
                ErrorCode.OPTIMIZER_STEP_SKIPPED,
                "AMP overflow skipped a frozen base optimizer step; the epoch must abort",
                epoch=epoch,
                details={"base_step_index": base_step_index, "scale_before": scale_before, "scale_after": scale_after},
                recoverable=True,
                required_action="Quarantine the partial epoch and resume from the last complete generation without changing the contract",
            )
        ema_after = int(getattr(trainer.ema, "updates", 0)) if getattr(trainer, "ema", None) is not None else 0
        global_step += 1
        optimizer_steps += 1
        if step_receipt_sink is not None:
            step_receipt_sink(UpstreamStepReceipt(epoch, base_step_index, global_step - 1, global_step, int(batch["cls"].shape[0]), len(replay_ids), float(base_loss.detach().cpu()), float(replay_loss.detach().cpu()), 1, ema_after - ema_before, rng_before, rng_after, bn_before, bn_after))

    if optimizer_steps != len(trainer.train_loader):
        raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "Formal epoch did not preserve one optimizer step per base batch")
    if replay_occurrences != replay_plan.planned_replay_occurrences:
        raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "Formal epoch replay occurrence count mismatch")
    return {
        "status": "IMPLEMENTED_NOT_FORMALLY_RUN",
        "epoch": epoch,
        "optimizer_steps": optimizer_steps,
        "base_occurrences": base_occurrences,
        "replay_occurrences": replay_occurrences,
        "ema_updates_delta": (int(getattr(trainer.ema, "updates", 0)) - ema_start) if getattr(trainer, "ema", None) is not None else 0,
        "base_order_digest": stable_digest(base_order),
        "base_augmentation_digest": stable_digest(base_aug),
        "global_step_end": global_step,
    }
