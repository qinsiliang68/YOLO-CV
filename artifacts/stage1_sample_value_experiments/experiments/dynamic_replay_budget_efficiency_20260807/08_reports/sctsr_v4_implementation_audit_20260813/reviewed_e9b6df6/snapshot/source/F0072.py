from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Sequence

from .errors import ErrorCode, SctsrError
from .serialization import stable_digest
from .rng_isolation import derive_counter_seed


def _rank(counter_seed: int, token: object) -> tuple[str, str]:
    payload = f"{counter_seed}\0{token}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), str(token)


@dataclass(frozen=True, slots=True)
class ReplayStepPlan:
    run_id: str
    arm_id: str
    training_seed: int
    epoch: int
    base_step_count: int
    planned_replay_occurrences: int
    step_slot_seed: int
    identity_order_seed: int
    per_step_replay_counts: tuple[int, ...]
    per_step_identity_slices: tuple[tuple[str, ...], ...]
    max_fraction_numerator: int = 1
    max_fraction_denominator: int = 4
    plan_digest: str = ""

    def content_payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "arm_id": self.arm_id,
            "training_seed": self.training_seed,
            "epoch": self.epoch,
            "base_step_count": self.base_step_count,
            "planned_replay_occurrences": self.planned_replay_occurrences,
            "step_slot_seed": self.step_slot_seed,
            "identity_order_seed": self.identity_order_seed,
            "per_step_replay_counts": self.per_step_replay_counts,
            "per_step_identity_slices": self.per_step_identity_slices,
            "max_fraction_numerator": self.max_fraction_numerator,
            "max_fraction_denominator": self.max_fraction_denominator,
        }

    def validate_structure(self) -> None:
        """Validate the materialized replay plan without touching the base loader."""

        if self.base_step_count <= 0:
            raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "Replay plan base step count must be positive")
        if not self.run_id.strip() or not self.arm_id.strip() or not 1 <= self.epoch <= 200 or self.training_seed < 0:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Replay plan run/arm/seed/epoch identity is invalid")
        if len(self.per_step_replay_counts) != self.base_step_count or len(self.per_step_identity_slices) != self.base_step_count:
            raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "Replay plan length differs from base step count")
        if sum(self.per_step_replay_counts) != self.planned_replay_occurrences:
            raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "Replay step plan does not conserve occurrences")
        if tuple(len(values) for values in self.per_step_identity_slices) != self.per_step_replay_counts:
            raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "Replay identity slices do not match per-step counts")
        if self.max_fraction_numerator < 0 or self.max_fraction_denominator <= 0:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Replay microbatch cap fraction is invalid")
        if (self.max_fraction_numerator, self.max_fraction_denominator) != (1, 4):
            raise SctsrError(
                ErrorCode.REPLAY_MICROBATCH_CAP_EXCEEDED,
                "Replay microbatch cap is frozen at exactly 1/4",
                observed=f"{self.max_fraction_numerator}/{self.max_fraction_denominator}",
                expected="1/4",
            )
        materialized_ids = tuple(sample_id for values in self.per_step_identity_slices for sample_id in values)
        if len(set(materialized_ids)) != len(materialized_ids):
            raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "Replay step plan repeats an identity within one epoch")
        expected_step_seed = derive_counter_seed("replay_step_slots", self.training_seed, self.epoch)
        expected_identity_seed = derive_counter_seed("replay_identity_order", self.training_seed, self.epoch)
        if self.step_slot_seed != expected_step_seed or self.identity_order_seed != expected_identity_seed:
            raise SctsrError(
                ErrorCode.IDENTITY_DIGEST_MISMATCH,
                "Replay plan counter-domain seed does not match run/seed/epoch identity",
                observed={"step_slot_seed": self.step_slot_seed, "identity_order_seed": self.identity_order_seed},
                expected={"step_slot_seed": expected_step_seed, "identity_order_seed": expected_identity_seed},
            )
        expected_digest = stable_digest(self.content_payload())
        if self.plan_digest != expected_digest:
            raise SctsrError(
                ErrorCode.IDENTITY_DIGEST_MISMATCH,
                "Replay step-plan digest does not match its materialized content",
                failing_field="plan_digest",
                observed=self.plan_digest,
                expected=expected_digest,
            )

    def validate_step(self, index: int, base_batch_size: int) -> None:
        """Validate one streamed base step against its frozen replay slice."""

        self.validate_structure()
        if not 0 <= index < self.base_step_count:
            raise SctsrError(
                ErrorCode.BASE_STEP_COUNT_MISMATCH,
                "Base loader yielded a step outside the replay plan",
                observed=index,
                expected=f"0..{self.base_step_count - 1}",
            )
        if base_batch_size <= 0:
            raise SctsrError(
                ErrorCode.BASE_STEP_COUNT_MISMATCH,
                "Base loader yielded an empty batch",
                observed=base_batch_size,
                expected="> 0",
                details={"base_step": index},
            )
        count = self.per_step_replay_counts[index]
        cap = (base_batch_size * self.max_fraction_numerator) // self.max_fraction_denominator
        if count > cap:
            raise SctsrError(
                ErrorCode.REPLAY_MICROBATCH_CAP_EXCEEDED,
                "Replay microbatch exceeds 25% of actual base batch size",
                observed=count,
                expected=cap,
                epoch=self.epoch,
                details={"base_step": index, "base_batch_size": base_batch_size},
            )

    def validate(self, base_batch_sizes: Sequence[int]) -> None:
        self.validate_structure()
        if len(base_batch_sizes) != self.base_step_count:
            raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "Base batch-size vector differs from base step count")
        for index, base_size in enumerate(base_batch_sizes):
            self.validate_step(index, int(base_size))


def build_replay_step_plan(
    *,
    run_id: str,
    arm_id: str,
    training_seed: int,
    epoch: int,
    schedule_family: str,
    sample_ids: Sequence[str],
    base_batch_sizes: Sequence[int],
) -> ReplayStepPlan:
    base_steps = len(base_batch_sizes)
    if base_steps <= 0:
        raise SctsrError(ErrorCode.BASE_STEP_COUNT_MISMATCH, "Base step count must be positive")
    n = len(sample_ids)
    q, r = divmod(n, base_steps)
    counts = [q] * base_steps
    step_slot_seed = derive_counter_seed("replay_step_slots", training_seed, epoch)
    identity_order_seed = derive_counter_seed("replay_identity_order", training_seed, epoch)
    ranked_steps = sorted(range(base_steps), key=lambda index: _rank(step_slot_seed, index))
    for index in ranked_steps[:r]:
        counts[index] += 1
    ordered_ids = sorted(sample_ids, key=lambda sample_id: _rank(identity_order_seed, sample_id))
    slices: list[tuple[str, ...]] = []
    cursor = 0
    for count in counts:
        slices.append(tuple(ordered_ids[cursor : cursor + count]))
        cursor += count
    plan = ReplayStepPlan(
        run_id=run_id,
        arm_id=arm_id,
        training_seed=training_seed,
        epoch=epoch,
        base_step_count=base_steps,
        planned_replay_occurrences=n,
        step_slot_seed=step_slot_seed,
        identity_order_seed=identity_order_seed,
        per_step_replay_counts=tuple(counts),
        per_step_identity_slices=tuple(slices),
    )
    # Frozen slot dataclasses have no ``__dict__``; replace is the explicit
    # publication step that binds the digest to all materialized content.
    plan = replace(plan, plan_digest=stable_digest(plan.content_payload()))
    plan.validate(base_batch_sizes)
    return plan
