from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from .bn_isolation import capture_batchnorm_buffers, preserve_batchnorm_buffers
from .errors import ErrorCode, SctsrError
from .replay_step_plan import ReplayStepPlan
from .rng_isolation import capture_global_rng, replay_rng_domain
from .rng_isolation import derive_counter_seed
from .serialization import stable_digest


@dataclass
class ExponentialMovingAverage:
    decay: float = 0.999
    updates: int = 0
    shadow: dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def from_model(cls, model: torch.nn.Module, decay: float = 0.999) -> "ExponentialMovingAverage":
        return cls(decay=decay, shadow={k: v.detach().clone() for k, v in model.state_dict().items()})

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        current = model.state_dict()
        for key, value in current.items():
            if key not in self.shadow or not torch.is_floating_point(value):
                self.shadow[key] = value.detach().clone()
            else:
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "updates": self.updates, "shadow": self.shadow}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.updates = int(state["updates"])
        self.shadow = {k: v.detach().clone() for k, v in state["shadow"].items()}


@dataclass(frozen=True, slots=True)
class OccurrenceEvent:
    occurrence_role: str
    base_step_index: int
    sample_ids: tuple[str, ...]
    labels: torch.Tensor
    logits: torch.Tensor
    per_sample_ce: torch.Tensor
    augmentation_digests: tuple[str, ...]
    augmentation_seeds: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class StepRuntimeRecord:
    base_step_index: int
    base_batch_size: int
    replay_microbatch_size: int
    base_loss: float
    replay_loss: float
    combined_loss_for_reporting: float
    parameter_grad_norm_before_clip: float
    parameter_grad_norm_after_clip: float
    clip_max_norm: float | None
    optimizer_step_count_delta: int
    ema_updates_before: int
    ema_updates_after: int
    amp_scale_before: float | None
    amp_scale_after: float | None
    rng_digest_before_replay: str | None
    rng_digest_after_replay_restore: str | None
    rng_digest_before_base: str
    replay_rng_fork_digest: str | None
    bn_digest_before_replay: str | None
    bn_digest_after_replay_restore: str | None
    overflow_or_step_skipped: bool
    learning_rates: tuple[float, ...]
    optimizer_hyperparameters: tuple[Mapping[str, Any], ...]
    scheduler_state_digest: str
    warmup_progress: float
    base_augmentation_digest: str
    replay_augmentation_digest: str | None
    write_buffer_bytes: int
    dataloader_wait_seconds: float
    base_forward_seconds: float
    replay_forward_seconds: float
    backward_seconds: float
    optimizer_seconds: float
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class EpochRuntimeResult:
    optimizer_steps: int
    base_occurrences: int
    replay_occurrences: int
    ema_updates_delta: int
    scheduler_epoch_transitions_delta: int
    base_order_digest: str
    base_augmentation_digest: str
    records: tuple[StepRuntimeRecord, ...]


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...], tuple[str, ...]]:
    if isinstance(batch, Mapping):
        x = batch["images"]
        y = batch["labels"]
        if "sample_ids" not in batch:
            raise SctsrError(ErrorCode.BASE_ORDER_MISMATCH, "Runtime batch lacks canonical sample identities")
        if "augmentation_digests" not in batch:
            raise SctsrError(ErrorCode.BASE_AUGMENTATION_MISMATCH, "Runtime batch lacks augmentation trace digests")
        sample_ids = tuple(str(v) for v in batch["sample_ids"])
        aug = tuple(str(v) for v in batch["augmentation_digests"])
        if len(sample_ids) != len(y) or len(aug) != len(y) or len(set(sample_ids)) != len(sample_ids):
            raise SctsrError(ErrorCode.BASE_ORDER_MISMATCH, "Runtime batch trace cardinality or uniqueness is invalid")
        return x, y, sample_ids, aug
    if isinstance(batch, (tuple, list)) and len(batch) >= 4:
        x, y = batch[0], batch[1]
        sample_ids = tuple(str(v) for v in batch[2])
        aug = tuple(str(v) for v in batch[3])
        if len(sample_ids) != len(y) or len(aug) != len(y) or len(set(sample_ids)) != len(sample_ids):
            raise SctsrError(ErrorCode.BASE_ORDER_MISMATCH, "Runtime tuple trace cardinality or uniqueness is invalid")
        return x, y, sample_ids, aug
    raise SctsrError(
        ErrorCode.BASE_ORDER_MISMATCH,
        "Batch must materialize images, labels, sample IDs, and augmentation digests",
    )


def _backward(loss: torch.Tensor, scaler: Any | None) -> None:
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()


def _gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    norms = [parameter.grad.detach().float().norm(2) for parameter in parameters if parameter.grad is not None]
    if not norms:
        return 0.0
    return float(torch.stack(norms).norm(2).cpu())


def _is_oom_exception(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _fixed_contract_oom(*, epoch: int, stage: str, cause: BaseException) -> SctsrError:
    return SctsrError(
        ErrorCode.OOM_FIXED_CONTRACT_ABORT,
        f"OOM during {stage} aborted the fixed contract; no automatic batch, precision, size, or replay change is allowed",
        epoch=epoch,
        recoverable=True,
        required_action="Quarantine the partial epoch and resume from the last complete epoch without changing the contract",
        details={"stage": stage, "cause_type": type(cause).__name__},
    )


def run_fixed_step_epoch(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    base_loader: Iterable[Any],
    replay_plan: ReplayStepPlan,
    replay_batch_provider: Callable[[Sequence[str], int, int, int], Any],
    training_seed: int,
    epoch: int,
    ema: ExponentialMovingAverage | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    base_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    clip_max_norm: float | None = None,
    canonical_base_batch_size: int = 128,
    world_size: int = 1,
    gradient_accumulation: int = 1,
    occurrence_event_sink: Callable[[OccurrenceEvent], None] | None = None,
    warmup_progress_fn: Callable[[int], float] | None = None,
    scheduler_step_timing: str = "END_OF_EPOCH",
) -> EpochRuntimeResult:
    if world_size != 1:
        raise SctsrError(
            ErrorCode.DISTRIBUTED_MODE_NOT_SUPPORTED_IN_V4_PHASE1,
            "Phase 1 fixed-step runtime supports world_size=1 only",
        )
    if gradient_accumulation != 1:
        raise SctsrError(ErrorCode.REPLAY_ADDED_OPTIMIZER_STEP, "Implicit gradient accumulation is forbidden")
    if canonical_base_batch_size != 128:
        raise SctsrError(
            ErrorCode.SCHEMA_VALIDATION_FAILED,
            "Replay CE denominator is frozen at canonical base batch size 128",
            observed=canonical_base_batch_size,
            expected=128,
        )

    if scheduler_step_timing not in {"START_OF_EPOCH", "END_OF_EPOCH"}:
        raise SctsrError(
            ErrorCode.SCHEMA_VALIDATION_FAILED,
            "scheduler_step_timing must be START_OF_EPOCH or END_OF_EPOCH",
            observed=scheduler_step_timing,
        )
    scheduler_delta = 0
    if scheduler is not None and scheduler_step_timing == "START_OF_EPOCH":
        # Frozen Ultralytics performs this boundary transition before the batch
        # loop and suppresses PyTorch's generic ordering warning. The restored
        # branch optimizer is not a fresh optimizer in formal use.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            scheduler.step()
        scheduler_delta = 1

    model.train()
    replay_plan.validate_structure()
    if replay_plan.epoch != epoch or replay_plan.training_seed != training_seed:
        raise SctsrError(
            ErrorCode.CONFIGURATION_MISMATCH,
            "Runtime epoch/seed differs from the materialized replay plan",
            observed={"epoch": epoch, "training_seed": training_seed},
            expected={"epoch": replay_plan.epoch, "training_seed": replay_plan.training_seed},
        )
    try:
        loader_length = len(base_loader)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        loader_length = None
    if loader_length is not None and loader_length != replay_plan.base_step_count:
        raise SctsrError(
            ErrorCode.BASE_STEP_COUNT_MISMATCH,
            "Base loader length differs from replay plan",
            observed=loader_length,
            expected=replay_plan.base_step_count,
        )
    base_iterator = iter(base_loader)

    step_records: list[StepRuntimeRecord] = []
    base_ids_all: list[str] = []
    base_aug_all: list[str] = []
    start_ema = ema.updates if ema is not None else 0
    replay_total = 0
    base_total = 0
    optimizer_steps = 0
    base_loss_fn = base_loss_fn or (lambda logits, labels: F.cross_entropy(logits, labels, reduction="mean"))

    for step_index in range(replay_plan.base_step_count):
        rng_before_base = capture_global_rng().digest()
        wait_start = time.monotonic()
        try:
            batch = next(base_iterator)
        except StopIteration as exc:
            raise SctsrError(
                ErrorCode.BASE_STEP_COUNT_MISMATCH,
                "Base loader ended before the frozen replay plan",
                observed=step_index,
                expected=replay_plan.base_step_count,
            ) from exc
        dataloader_wait_seconds = time.monotonic() - wait_start
        step_start = time.monotonic()
        images, labels, sample_ids, augmentation_digests = _unpack_batch(batch)
        replay_plan.validate_step(step_index, int(labels.shape[0]))
        base_total += int(labels.shape[0])
        base_ids_all.extend(sample_ids)
        base_aug_all.extend(augmentation_digests)
        optimizer.zero_grad(set_to_none=True)

        try:
            base_forward_start = time.monotonic()
            logits = model(images)
            base_loss = base_loss_fn(logits, labels)
            base_forward_seconds = time.monotonic() - base_forward_start
            if occurrence_event_sink is not None:
                occurrence_event_sink(
                    OccurrenceEvent(
                        occurrence_role="BASE",
                        base_step_index=step_index,
                        sample_ids=sample_ids,
                        labels=labels.detach().cpu().clone(),
                        logits=logits.detach().cpu().clone(),
                        per_sample_ce=F.cross_entropy(logits, labels, reduction="none").detach().cpu().clone(),
                        augmentation_digests=augmentation_digests,
                        augmentation_seeds=tuple(
                            derive_counter_seed("base_augmentation", training_seed, epoch, sample_id)
                            for sample_id in sample_ids
                        ),
                    )
                )

            backward_seconds = 0.0
            backward_start = time.monotonic()
            _backward(base_loss, scaler)
            backward_seconds += time.monotonic() - backward_start
        except RuntimeError as exc:
            if _is_oom_exception(exc):
                raise _fixed_contract_oom(epoch=epoch, stage="base_forward_or_backward", cause=exc) from exc
            raise

        replay_ids = replay_plan.per_step_identity_slices[step_index]
        replay_loss = torch.zeros((), dtype=base_loss.dtype, device=base_loss.device)
        replay_forward_seconds = 0.0
        replay_augmentation_digest = None
        rng_before = rng_after = bn_before = bn_after = None
        replay_rng_fork_digest = None

        if replay_ids:
            cap = (
                int(labels.shape[0])
                * replay_plan.max_fraction_numerator
                // replay_plan.max_fraction_denominator
            )
            if len(replay_ids) > cap:
                raise SctsrError(
                    ErrorCode.REPLAY_MICROBATCH_CAP_EXCEEDED,
                    "Replay microbatch exceeds fixed cap",
                    observed=len(replay_ids),
                    expected=cap,
                    epoch=epoch,
                )
            global_rng = capture_global_rng()
            bn_snapshot = capture_batchnorm_buffers(model)
            rng_before = global_rng.digest()
            bn_before = bn_snapshot.digest()
            try:
                with replay_rng_domain("replay_augmentation", training_seed, epoch, step_index) as replay_rng_receipt:
                    replay_rng_fork_digest = str(replay_rng_receipt["fork_digest"])
                    with preserve_batchnorm_buffers(model):
                        replay_forward_start = time.monotonic()
                        replay_batch = replay_batch_provider(replay_ids, epoch, step_index, training_seed)
                        replay_images, replay_labels, replay_sample_ids, replay_aug = _unpack_batch(replay_batch)
                        if int(replay_labels.shape[0]) != len(replay_ids):
                            raise SctsrError(
                                ErrorCode.SCHEDULE_EXPOSURE_MISMATCH,
                                "Replay provider returned wrong occurrence count",
                            )
                        if tuple(replay_sample_ids) != tuple(replay_ids):
                            raise SctsrError(
                                ErrorCode.SCHEDULE_EXPOSURE_MISMATCH,
                                "Replay provider changed the materialized identity order",
                                observed=replay_sample_ids,
                                expected=tuple(replay_ids),
                            )
                        replay_logits = model(replay_images)
                        replay_augmentation_digest = stable_digest(replay_aug)
                        replay_per_sample_ce = F.cross_entropy(replay_logits, replay_labels, reduction="none")
                        replay_forward_seconds += time.monotonic() - replay_forward_start
                        if occurrence_event_sink is not None:
                            occurrence_event_sink(
                                OccurrenceEvent(
                                    occurrence_role="REPLAY",
                                    base_step_index=step_index,
                                    sample_ids=tuple(replay_ids),
                                    labels=replay_labels.detach().cpu().clone(),
                                    logits=replay_logits.detach().cpu().clone(),
                                    per_sample_ce=replay_per_sample_ce.detach().cpu().clone(),
                                    augmentation_digests=replay_aug,
                                    augmentation_seeds=tuple(
                                        derive_counter_seed(
                                            "replay_augmentation",
                                            training_seed,
                                            epoch,
                                            f"{step_index}:{sample_id}",
                                        )
                                        for sample_id in replay_ids
                                    ),
                                )
                            )
                        replay_loss = replay_per_sample_ce.sum() / 128
                        replay_backward_start = time.monotonic()
                        _backward(replay_loss, scaler)
                        backward_seconds += time.monotonic() - replay_backward_start
            except RuntimeError as exc:
                if _is_oom_exception(exc):
                    raise _fixed_contract_oom(epoch=epoch, stage="replay_forward_or_backward", cause=exc) from exc
                raise
            rng_after = capture_global_rng().digest()
            bn_after = capture_batchnorm_buffers(model).digest()
            if rng_after != rng_before:
                raise SctsrError(
                    ErrorCode.RNG_NOT_RESTORED,
                    "Replay changed the base RNG trajectory",
                    observed=rng_after,
                    expected=rng_before,
                )
            if bn_after != bn_before:
                raise SctsrError(
                    ErrorCode.BN_BUFFER_NOT_RESTORED,
                    "Replay changed BatchNorm running buffers",
                    observed=bn_after,
                    expected=bn_before,
                )
            replay_total += len(replay_ids)

        amp_scale_before = float(scaler.get_scale()) if scaler is not None else None
        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_before = _gradient_norm(model.parameters())
        if clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        grad_after = _gradient_norm(model.parameters())

        learning_rates = tuple(float(group.get("lr", 0.0)) for group in optimizer.param_groups)
        optimizer_hyperparameters = tuple(
            {
                key: value
                for key, value in group.items()
                if key in {"lr", "initial_lr", "momentum", "betas", "weight_decay", "dampening", "nesterov"}
            }
            for group in optimizer.param_groups
        )
        scheduler_state_digest = stable_digest(scheduler.state_dict()) if scheduler is not None else "NOT_APPLICABLE_NO_SCHEDULER"
        warmup_progress = float(warmup_progress_fn(step_index) if warmup_progress_fn is not None else 1.0)
        before_ema = ema.updates if ema is not None else 0
        optimizer_start = time.monotonic()
        try:
            if scaler is None:
                optimizer.step()
                skipped = False
                amp_scale_after = None
            else:
                scaler.step(optimizer)
                scaler.update()
                amp_scale_after = float(scaler.get_scale())
                skipped = amp_scale_after < float(amp_scale_before)
        except RuntimeError as exc:
            if _is_oom_exception(exc):
                raise _fixed_contract_oom(epoch=epoch, stage="optimizer_step", cause=exc) from exc
            raise
        optimizer_seconds = time.monotonic() - optimizer_start
        if skipped:
            raise SctsrError(
                ErrorCode.OPTIMIZER_STEP_SKIPPED,
                "AMP overflow skipped a frozen base optimizer step; the epoch must abort",
                epoch=epoch,
                details={"base_step_index": step_index, "scale_before": amp_scale_before, "scale_after": amp_scale_after},
                recoverable=True,
                required_action="Quarantine the partial epoch and resume from the last complete generation without changing the contract",
            )
        optimizer_steps += 1
        if ema is not None:
            ema.update(model)
        after_ema = ema.updates if ema is not None else 0

        # Persist the exact scalar operands used by the evidence validator.
        # Adding the float32 tensors first can differ from adding their
        # serialized float64 values by more than the ledger's 1e-8 guard.
        base_loss_for_reporting = float(base_loss.detach().cpu())
        replay_loss_for_reporting = float(replay_loss.detach().cpu())
        step_records.append(
            StepRuntimeRecord(
                base_step_index=step_index,
                base_batch_size=int(labels.shape[0]),
                replay_microbatch_size=len(replay_ids),
                base_loss=base_loss_for_reporting,
                replay_loss=replay_loss_for_reporting,
                combined_loss_for_reporting=base_loss_for_reporting + replay_loss_for_reporting,
                parameter_grad_norm_before_clip=grad_before,
                parameter_grad_norm_after_clip=grad_after,
                clip_max_norm=clip_max_norm,
                optimizer_step_count_delta=1,
                ema_updates_before=before_ema,
                ema_updates_after=after_ema,
                amp_scale_before=amp_scale_before,
                amp_scale_after=amp_scale_after,
                rng_digest_before_replay=rng_before,
                rng_digest_after_replay_restore=rng_after,
                rng_digest_before_base=rng_before_base,
                replay_rng_fork_digest=replay_rng_fork_digest,
                bn_digest_before_replay=bn_before,
                bn_digest_after_replay_restore=bn_after,
                overflow_or_step_skipped=skipped,
                learning_rates=learning_rates,
                optimizer_hyperparameters=optimizer_hyperparameters,
                scheduler_state_digest=scheduler_state_digest,
                warmup_progress=warmup_progress,
                base_augmentation_digest=stable_digest(augmentation_digests),
                replay_augmentation_digest=replay_augmentation_digest,
                write_buffer_bytes=0,
                dataloader_wait_seconds=dataloader_wait_seconds,
                base_forward_seconds=base_forward_seconds,
                replay_forward_seconds=replay_forward_seconds,
                backward_seconds=backward_seconds,
                optimizer_seconds=optimizer_seconds,
                elapsed_seconds=time.monotonic() - step_start,
            )
        )

    try:
        extra_batch = next(base_iterator)
    except StopIteration:
        extra_batch = None
    if extra_batch is not None:
        raise SctsrError(
            ErrorCode.BASE_STEP_COUNT_MISMATCH,
            "Base loader yielded more batches than the frozen replay plan",
            observed=f"> {replay_plan.base_step_count}",
            expected=replay_plan.base_step_count,
        )
    if scheduler is not None and scheduler_step_timing == "END_OF_EPOCH":
        scheduler.step()
        scheduler_delta = 1
    if optimizer_steps != replay_plan.base_step_count:
        raise SctsrError(
            ErrorCode.REPLAY_ADDED_OPTIMIZER_STEP,
            "Optimizer step count differs from base batch count",
            observed=optimizer_steps,
            expected=replay_plan.base_step_count,
        )
    return EpochRuntimeResult(
        optimizer_steps=optimizer_steps,
        base_occurrences=base_total,
        replay_occurrences=replay_total,
        ema_updates_delta=(ema.updates - start_ema) if ema is not None else 0,
        scheduler_epoch_transitions_delta=scheduler_delta,
        base_order_digest=stable_digest(base_ids_all),
        base_augmentation_digest=stable_digest(base_aug_all),
        records=tuple(step_records),
    )
