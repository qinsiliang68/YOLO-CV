from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

from .arm_spec import ArmId
from .errors import ErrorCode, SctsrError
from .identity_pool import IdentityRecord
from .rate_spec import (
    F_ACTIVE_RATE,
    U_RATE,
    ZERO_RATE,
    DenominatorRole,
    RateSemantic,
    ReplayRateSpec,
)
from .serialization import stable_digest


@dataclass(frozen=True, slots=True)
class EpochReplayPlan:
    epoch: int
    arm_id: ArmId
    identity_policy: str
    schedule_family: str
    rate: ReplayRateSpec
    group_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    fallback_state: str

    @property
    def replay_occurrences(self) -> int:
        return len(self.sample_ids)


@dataclass(frozen=True, slots=True)
class SchedulePlan:
    arm_id: ArmId
    epochs: tuple[EpochReplayPlan, ...]
    base_denominator: int
    identity_pool_digest: str
    plan_digest: str

    def epoch(self, number: int) -> EpochReplayPlan:
        return self.epochs[number - 1]

    def multiplicity(self) -> Counter[str]:
        return Counter(sample_id for epoch in self.epochs for sample_id in epoch.sample_ids)

    @property
    def total_occurrences(self) -> int:
        return sum(len(epoch.sample_ids) for epoch in self.epochs)


def _group_ids(groups: Mapping[str, Sequence[IdentityRecord]], names: Sequence[str]) -> tuple[str, ...]:
    return tuple(record.sample_id for name in names for record in groups[name])


def _epoch_payload(epochs: Sequence[EpochReplayPlan]) -> list[dict[str, object]]:
    return [
        {
            "epoch": item.epoch,
            "arm_id": item.arm_id.value,
            "identity_policy": item.identity_policy,
            "schedule_family": item.schedule_family,
            "rate": item.rate.as_dict(),
            "group_ids": list(item.group_ids),
            "sample_ids": list(item.sample_ids),
            "fallback_state": item.fallback_state,
        }
        for item in epochs
    ]


def _schedule_digest(arm_id: ArmId, base_denominator: int, identity_pool_digest: str, epochs: Sequence[EpochReplayPlan]) -> str:
    return stable_digest(
        {
            "arm_id": arm_id.value,
            "base_denominator": base_denominator,
            "identity_pool_digest": identity_pool_digest,
            "epochs": _epoch_payload(epochs),
        }
    )


def build_schedule(
    arm_id: ArmId,
    *,
    primary_groups: Mapping[str, Sequence[IdentityRecord]] | None,
    primary_digest: str,
    fallback_groups: Mapping[str, Sequence[IdentityRecord]] | None = None,
    fallback_digest: str | None = None,
    base_denominator: int = 120_000,
) -> SchedulePlan:
    if base_denominator <= 0:
        raise SctsrError(ErrorCode.DENOMINATOR_IDENTITY_MISMATCH, "Schedule base denominator must be positive")
    if arm_id is not ArmId.NR and primary_groups is None:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Replay arm requires a registered identity group mapping")
    epochs: list[EpochReplayPlan] = []
    for epoch in range(1, 201):
        rate = ZERO_RATE
        names: tuple[str, ...] = ()
        ids: tuple[str, ...] = ()
        policy = "NONE"
        family = "NR"
        fallback_state = "NOT_APPLICABLE"
        if epoch >= 121:
            j = (epoch - 121) % 5
            if arm_id in {ArmId.R1_U, ArmId.R2_U, ArmId.T_U}:
                rate, names, family = U_RATE, (f"G{j}",), "U"
                ids = _group_ids(primary_groups or {}, names)
                policy = {ArmId.R1_U: "R1_GLOBAL_RANDOM", ArmId.R2_U: "R2_MATCHED_RANDOM", ArmId.T_U: "T_STRESS"}[arm_id]
            elif arm_id in {ArmId.R2_F, ArmId.T_F} and epoch <= 160:
                rate, names, family = F_ACTIVE_RATE, (f"G{j}", f"G{(j + 4) % 5}"), "F"
                ids = _group_ids(primary_groups or {}, names)
                policy = "R2_MATCHED_RANDOM" if arm_id is ArmId.R2_F else "T_STRESS"
            elif arm_id is ArmId.T_TO_R2_AT_160:
                rate, names, family = U_RATE, (f"G{j}",), "U"
                if epoch <= 160:
                    ids = _group_ids(primary_groups or {}, names)
                    policy, fallback_state = "T_STRESS", "TARGETED_PREFIX"
                else:
                    if fallback_groups is None or fallback_digest is None:
                        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "T_TO_R2_AT_160 requires a registered R2 fallback pool")
                    ids = _group_ids(fallback_groups, names)
                    policy, fallback_state = "R2_MATCHED_RANDOM", "FALLBACK_R2"
            elif arm_id is ArmId.T_TO_NR_AT_160 and epoch <= 160:
                rate, names, family = U_RATE, (f"G{j}",), "U"
                ids = _group_ids(primary_groups or {}, names)
                policy, fallback_state = "T_STRESS", "TARGETED_PREFIX"
            elif arm_id is ArmId.NR:
                pass
        epochs.append(EpochReplayPlan(epoch, arm_id, policy, family, rate, names, ids, fallback_state))
    identity_digest = primary_digest if arm_id is not ArmId.T_TO_R2_AT_160 else stable_digest({"primary": primary_digest, "fallback": fallback_digest})
    digest = _schedule_digest(arm_id, base_denominator, identity_digest, epochs)
    plan = SchedulePlan(arm_id, tuple(epochs), base_denominator, identity_digest, digest)
    validate_schedule(plan)
    return plan


def validate_schedule(plan: SchedulePlan) -> None:
    if plan.base_denominator <= 0:
        raise SctsrError(ErrorCode.DENOMINATOR_IDENTITY_MISMATCH, "Schedule base denominator must be positive")
    observed_digest = _schedule_digest(plan.arm_id, plan.base_denominator, plan.identity_pool_digest, plan.epochs)
    if observed_digest != plan.plan_digest:
        raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Schedule plan digest does not match materialized content", failing_field="plan_digest", observed=plan.plan_digest, expected=observed_digest)
    if len(plan.epochs) != 200 or tuple(e.epoch for e in plan.epochs) != tuple(range(1, 201)):
        raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "Schedule must contain exactly E1-E200")
    if any(e.sample_ids for e in plan.epochs[:120]):
        raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "All arms must have zero replay in E1-E120")
    if any(item.arm_id is not plan.arm_id for item in plan.epochs):
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Schedule contains an epoch for another arm")
    for item in plan.epochs:
        expected_occurrences = item.rate.derive_count(plan.base_denominator)
        if item.replay_occurrences != expected_occurrences:
            raise SctsrError(
                ErrorCode.SCHEDULE_EXPOSURE_MISMATCH,
                "Materialized replay occurrences differ from the rational rate",
                arm_id=plan.arm_id.value,
                epoch=item.epoch,
                observed=item.replay_occurrences,
                expected=expected_occurrences,
            )
        if len(item.sample_ids) != len(set(item.sample_ids)):
            raise SctsrError(ErrorCode.SCHEDULE_MULTIPLICITY_MISMATCH, "An epoch repeats an identity within its replay slot", arm_id=plan.arm_id.value, epoch=item.epoch)
        expected_groups = 0 if item.rate.numerator == 0 else (2 if item.rate.canonical_token() == F_ACTIVE_RATE.canonical_token() else 1)
        if len(item.group_ids) != expected_groups:
            raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "Group count differs from the registered rate", arm_id=plan.arm_id.value, epoch=item.epoch, observed=len(item.group_ids), expected=expected_groups)
    expected_pool_count = 25 * plan.base_denominator // 1000
    if 25 * plan.base_denominator % 1000:
        raise SctsrError(ErrorCode.RATE_NOT_INTEGRAL, "Identity pool rate is not integral for schedule denominator")
    if plan.arm_id is ArmId.NR and plan.total_occurrences != 0:
        raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "NR must have zero replay")
    if plan.arm_id is not ArmId.NR and plan.identity_pool_digest in {"", "NONE"}:
        raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Replay schedule is not bound to an identity pool digest")
    if plan.arm_id in {ArmId.R1_U, ArmId.R2_U, ArmId.T_U}:
        counts = plan.multiplicity()
        if len(counts) != expected_pool_count:
            raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "U schedule unique identity count differs from the 2.5% pool", observed=len(counts), expected=expected_pool_count)
        if set(counts.values()) != {16}:
            raise SctsrError(ErrorCode.SCHEDULE_MULTIPLICITY_MISMATCH, "U schedule must expose every identity exactly 16 times", observed=sorted(set(counts.values())), expected=[16])
    if plan.arm_id in {ArmId.R2_F, ArmId.T_F}:
        if any(e.sample_ids for e in plan.epochs[160:]):
            raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "F schedule must have zero replay in E161-E200")
        counts = plan.multiplicity()
        if len(counts) != expected_pool_count:
            raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "F schedule unique identity count differs from the 2.5% pool", observed=len(counts), expected=expected_pool_count)
        if set(counts.values()) != {16}:
            raise SctsrError(ErrorCode.SCHEDULE_MULTIPLICITY_MISMATCH, "F schedule must expose every identity exactly 16 times")
        for e in plan.epochs[120:160]:
            if len(e.sample_ids) != len(set(e.sample_ids)):
                raise SctsrError(ErrorCode.SCHEDULE_MULTIPLICITY_MISMATCH, "F active epoch may not repeat an identity within the epoch")
    if plan.arm_id is ArmId.T_TO_NR_AT_160 and any(e.sample_ids for e in plan.epochs[160:]):
        raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "Stop-all arm must have zero replay after E160")
    if plan.arm_id is ArmId.T_TO_NR_AT_160:
        counts = plan.multiplicity()
        if len(counts) != expected_pool_count or set(counts.values()) != {8}:
            raise SctsrError(ErrorCode.SCHEDULE_MULTIPLICITY_MISMATCH, "Stop-all prefix must expose every T identity exactly eight times")
    if plan.arm_id is ArmId.T_TO_R2_AT_160:
        before = Counter(sample_id for epoch in plan.epochs[120:160] for sample_id in epoch.sample_ids)
        after = Counter(sample_id for epoch in plan.epochs[160:] for sample_id in epoch.sample_ids)
        if len(before) != expected_pool_count or len(after) != expected_pool_count or set(before.values()) != {8} or set(after.values()) != {8}:
            raise SctsrError(ErrorCode.SCHEDULE_MULTIPLICITY_MISMATCH, "Fallback schedule must expose T and R2 pools eight times each")
        if set(before) & set(after):
            raise SctsrError(ErrorCode.R2_OVERLAPS_T, "Fallback R2 identities overlap the T prefix")


def validate_u_f_parity(u: SchedulePlan, f: SchedulePlan) -> None:
    if set(u.multiplicity()) != set(f.multiplicity()):
        raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "U/F identity sets differ")
    if u.total_occurrences != f.total_occurrences:
        raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "U/F total occurrences differ", observed=(u.total_occurrences, f.total_occurrences))
    if u.multiplicity() != f.multiplicity():
        raise SctsrError(ErrorCode.SCHEDULE_MULTIPLICITY_MISMATCH, "U/F per-ID multiplicity vectors differ")


def validate_common_prefix(reference: SchedulePlan, candidate: SchedulePlan, *, through_epoch: int = 160) -> None:
    for epoch in range(121, through_epoch + 1):
        left, right = reference.epoch(epoch), candidate.epoch(epoch)
        if (left.sample_ids, left.rate.canonical_token()) != (right.sample_ids, right.rate.canonical_token()):
            raise SctsrError(ErrorCode.SCHEDULE_EXPOSURE_MISMATCH, "Schedule prefixes differ before the frozen stop point", epoch=epoch)


def validate_phase1_schedule_registry(
    plans: Mapping[ArmId, SchedulePlan],
    *,
    t_pool_digest: str,
    r1_pool_digest: str,
    r2_pool_digest: str,
) -> dict[str, object]:
    required = set(ArmId)
    if set(plans) != required:
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "All eight phase-1 schedules are required for cross-arm validation", observed=sorted(item.value for item in plans), expected=sorted(item.value for item in required))
    denominators = {plan.base_denominator for plan in plans.values()}
    if len(denominators) != 1:
        raise SctsrError(ErrorCode.DENOMINATOR_IDENTITY_MISMATCH, "Phase-1 schedules use different base denominators", observed=sorted(denominators))
    expected_digests = {
        ArmId.NR: "NONE",
        ArmId.R1_U: r1_pool_digest,
        ArmId.R2_U: r2_pool_digest,
        ArmId.T_U: t_pool_digest,
        ArmId.R2_F: r2_pool_digest,
        ArmId.T_F: t_pool_digest,
        ArmId.T_TO_R2_AT_160: stable_digest({"primary": t_pool_digest, "fallback": r2_pool_digest}),
        ArmId.T_TO_NR_AT_160: t_pool_digest,
    }
    for arm, plan in plans.items():
        validate_schedule(plan)
        if plan.identity_pool_digest != expected_digests[arm]:
            raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Schedule identity pool binding differs from the registered pool", arm_id=arm.value, observed=plan.identity_pool_digest, expected=expected_digests[arm])
    validate_u_f_parity(plans[ArmId.T_U], plans[ArmId.T_F])
    validate_u_f_parity(plans[ArmId.R2_U], plans[ArmId.R2_F])
    validate_common_prefix(plans[ArmId.T_U], plans[ArmId.T_TO_R2_AT_160])
    validate_common_prefix(plans[ArmId.T_U], plans[ArmId.T_TO_NR_AT_160])
    return {
        "status": "PASS",
        "validated_arms": [arm.value for arm in ArmId],
        "base_denominator": next(iter(denominators)),
        "schedule_digests": {arm.value: plans[arm].plan_digest for arm in ArmId},
    }


def schedule_to_dict(plan: SchedulePlan) -> dict:
    return {
        "schema_version": "stage1.sctsr.schedule.v1",
        "arm_id": plan.arm_id.value,
        "base_denominator": plan.base_denominator,
        "identity_pool_digest": plan.identity_pool_digest,
        "plan_digest": plan.plan_digest,
        "epochs": [
            {
                "epoch": item.epoch,
                "arm_id": item.arm_id.value,
                "identity_policy": item.identity_policy,
                "schedule_family": item.schedule_family,
                "rate": item.rate.as_dict(),
                "group_ids": list(item.group_ids),
                "sample_ids": list(item.sample_ids),
                "fallback_state": item.fallback_state,
            }
            for item in plan.epochs
        ],
    }


def schedule_from_dict(raw: Mapping[str, object]) -> SchedulePlan:
    epochs = []
    for item in raw["epochs"]:  # type: ignore[index]
        rate_raw = item["rate"]
        rate = ReplayRateSpec(
            int(rate_raw["numerator"]),
            int(rate_raw["denominator"]),
            RateSemantic(str(rate_raw["semantic"])),
            DenominatorRole(str(rate_raw["denominator_role"])),
        )
        if rate.as_dict() != dict(rate_raw):
            raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Serialized rate is not canonical", observed=rate_raw, expected=rate.as_dict())
        epochs.append(
            EpochReplayPlan(
                epoch=int(item["epoch"]),
                arm_id=ArmId(item["arm_id"]),
                identity_policy=str(item["identity_policy"]),
                schedule_family=str(item["schedule_family"]),
                rate=rate,
                group_ids=tuple(str(v) for v in item["group_ids"]),
                sample_ids=tuple(str(v) for v in item["sample_ids"]),
                fallback_state=str(item["fallback_state"]),
            )
        )
    plan = SchedulePlan(
        arm_id=ArmId(raw["arm_id"]),
        epochs=tuple(epochs),
        base_denominator=int(raw["base_denominator"]),
        identity_pool_digest=str(raw["identity_pool_digest"]),
        plan_digest=str(raw["plan_digest"]),
    )
    validate_schedule(plan)
    return plan
