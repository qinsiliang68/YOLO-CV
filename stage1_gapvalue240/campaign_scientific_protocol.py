"""Percentage-based, staged scientific contract for the final replay campaign."""

from __future__ import annotations

from fractions import Fraction
import hashlib
from typing import Any, Iterable

import pandas as pd

from .campaign_layout import CAMPAIGN_ID


BASE_SAMPLE_COUNT = 120_000
TOTAL_EPOCHS = 200
SCHEDULE_BOUNDARIES = ((1, 140), (141, 150), (151, 160), (161, 200))

RATIO_FRACTIONS: dict[str, Fraction] = {
    "RHO_0P5_PERCENT": Fraction(1, 200),
    "RHO_1P0_PERCENT": Fraction(1, 100),
    "RHO_2P5_PERCENT": Fraction(1, 40),
}

POLICIES = {
    "CONTINUOUS",
    "SAME_PEAK_TAPER",
    "DOSE_MATCHED_TAPER",
    "NO_REPLAY",
}


class ProtocolError(RuntimeError):
    """Raised when a scientific schedule cannot satisfy its causal contract."""


def _round_fraction(value: Fraction) -> int:
    if value < 0:
        raise ProtocolError("negative replay slot calculation")
    numerator, denominator = value.numerator, value.denominator
    return (2 * numerator + denominator) // (2 * denominator)


def ratio_to_slots(ratio_id: str, *, base_sample_count: int = BASE_SAMPLE_COUNT) -> int:
    if ratio_id not in RATIO_FRACTIONS:
        raise ProtocolError(f"unknown replay ratio: {ratio_id}")
    if int(base_sample_count) <= 0:
        raise ProtocolError("base sample count must be positive")
    exact = RATIO_FRACTIONS[ratio_id] * int(base_sample_count)
    if exact.denominator != 1:
        raise ProtocolError(
            f"ratio {ratio_id} does not map to an integer slot count for base={base_sample_count}"
        )
    return int(exact)


def _policy_slots(ratio_id: str, policy_id: str) -> tuple[int, int, int, int]:
    if policy_id not in POLICIES:
        raise ProtocolError(f"unknown replay policy: {policy_id}")
    if policy_id == "NO_REPLAY":
        if ratio_id != "NO_REPLAY":
            raise ProtocolError("NO_REPLAY policy requires ratio_id=NO_REPLAY")
        return (0, 0, 0, 0)
    if ratio_id == "NO_REPLAY":
        raise ProtocolError(f"policy {policy_id} requires a registered percentage ratio")
    peak = ratio_to_slots(ratio_id)
    if policy_id == "CONTINUOUS":
        return (peak, peak, peak, peak)
    if peak % 3:
        raise ProtocolError(f"same-peak taper requires peak divisible by three: {peak}")
    if policy_id == "SAME_PEAK_TAPER":
        return (peak, 2 * peak // 3, peak // 3, 0)
    dose_peak = 4 * peak // 3
    if dose_peak * 3 != peak * 4:
        raise ProtocolError(f"dose-matched peak is not integral for ratio {ratio_id}")
    # The two ten-epoch taper levels sum to the plateau. This makes the
    # 150 plateau-equivalent epochs exactly equal 200 epochs at the base peak.
    taper_high = _round_fraction(Fraction(2 * dose_peak, 3))
    taper_low = dose_peak - taper_high
    return (dose_peak, taper_high, taper_low, 0)


def build_epoch_replay_schedule(
    ratio_id: str,
    policy_id: str,
    *,
    guard_fraction: Fraction = Fraction(0, 1),
) -> pd.DataFrame:
    """Return the complete 200-epoch intended exposure schedule."""

    guard_fraction = Fraction(guard_fraction)
    if not Fraction(0, 1) <= guard_fraction <= Fraction(1, 1):
        raise ProtocolError("guard fraction must be within [0, 1]")
    segment_slots = _policy_slots(ratio_id, policy_id)
    rows: list[dict[str, Any]] = []
    cumulative = 0
    cumulative_normal = 0
    cumulative_defect = 0
    ratio = RATIO_FRACTIONS.get(ratio_id, Fraction(0, 1))
    for segment_index, ((start, end), total) in enumerate(
        zip(SCHEDULE_BOUNDARIES, segment_slots), start=1
    ):
        defect = _round_fraction(Fraction(total) * guard_fraction)
        normal = total - defect
        for epoch in range(start, end + 1):
            cumulative += total
            cumulative_normal += normal
            cumulative_defect += defect
            rows.append(
                {
                    "epoch": epoch,
                    "segment_index": segment_index,
                    "ratio_id": ratio_id,
                    "policy_id": policy_id,
                    "scientific_schedule_id": f"{ratio_id}__{policy_id}",
                    "target_ratio_numerator": ratio.numerator,
                    "target_ratio_denominator": ratio.denominator,
                    "target_percent": float(ratio * 100),
                    "guard_fraction_numerator": guard_fraction.numerator,
                    "guard_fraction_denominator": guard_fraction.denominator,
                    "guard_target_percent": float(guard_fraction * 100),
                    "normal_replay_slots": normal,
                    "defect_guard_slots": defect,
                    "total_replay_slots": total,
                    "realized_ratio_of_base": total / BASE_SAMPLE_COUNT,
                    "configured_train_samples": BASE_SAMPLE_COUNT + total,
                    "cumulative_normal_exposure": cumulative_normal,
                    "cumulative_defect_exposure": cumulative_defect,
                    "cumulative_replay_exposure": cumulative,
                }
            )
    frame = pd.DataFrame(rows)
    validate_epoch_schedule(frame)
    return frame


def validate_epoch_schedule(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "epoch",
        "segment_index",
        "normal_replay_slots",
        "defect_guard_slots",
        "total_replay_slots",
        "cumulative_replay_exposure",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ProtocolError(f"epoch schedule missing fields: {sorted(missing)}")
    if len(frame) != TOTAL_EPOCHS or frame.epoch.astype(int).tolist() != list(
        range(1, TOTAL_EPOCHS + 1)
    ):
        raise ProtocolError("epoch schedule must cover epochs 1 through 200 exactly")
    numeric = frame[
        ["normal_replay_slots", "defect_guard_slots", "total_replay_slots"]
    ].apply(pd.to_numeric, errors="raise")
    if (numeric < 0).any().any():
        raise ProtocolError("epoch schedule contains negative replay slots")
    if not (numeric.normal_replay_slots + numeric.defect_guard_slots).equals(
        numeric.total_replay_slots
    ):
        raise ProtocolError("guard samples must replace, not add to, replay slots")
    expected_cumulative = numeric.total_replay_slots.cumsum().astype(int)
    if not expected_cumulative.equals(
        pd.to_numeric(frame.cumulative_replay_exposure, errors="raise").astype(int)
    ):
        raise ProtocolError("cumulative replay exposure is inconsistent")
    return {
        "status": "PASS",
        "epoch_count": TOTAL_EPOCHS,
        "cumulative_replay_exposure": int(expected_cumulative.iloc[-1]),
        "maximum_replay_slots": int(numeric.total_replay_slots.max()),
    }


def collapse_epoch_schedule(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse to fixed restart boundaries shared by every causal arm."""

    validate_epoch_schedule(frame)
    rows: list[dict[str, Any]] = []
    for segment_index, (start, end) in enumerate(SCHEDULE_BOUNDARIES, start=1):
        part = frame.loc[frame.epoch.between(start, end)]
        for field in ("normal_replay_slots", "defect_guard_slots", "total_replay_slots"):
            if part[field].nunique() != 1:
                raise ProtocolError(f"{field} changes inside registered segment {start}-{end}")
        first = part.iloc[0]
        rows.append(
            {
                "segment_index": segment_index,
                "segment_start_epoch": start,
                "segment_end_epoch": end,
                "segment_epoch_count": end - start + 1,
                "ratio_id": first.ratio_id,
                "policy_id": first.policy_id,
                "scientific_schedule_id": first.scientific_schedule_id,
                "guard_target_percent": first.guard_target_percent,
                "normal_replay_slots": int(first.normal_replay_slots),
                "defect_guard_slots": int(first.defect_guard_slots),
                "total_replay_slots": int(first.total_replay_slots),
                "segment_replay_exposure": int(first.total_replay_slots) * (end - start + 1),
                "expected_steps_batch128": (
                    BASE_SAMPLE_COUNT + int(first.total_replay_slots) + 127
                )
                // 128,
            }
        )
    return pd.DataFrame(rows)


def build_cycle_registry(canonical_lock_file_sha256: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    lock_sha = str(canonical_lock_file_sha256).upper()
    if len(lock_sha) != 64:
        raise ProtocolError("canonical lock file SHA must contain 64 hexadecimal characters")
    cycles = pd.DataFrame(
        [
            {
                "cycle_id": "CYCLE_1",
                "question": "Does reducing late replay change the paired tail outcome at high pressure?",
                "seed_scope": "DISCOVERY_TIMING_DOSE",
                "seed_count": 8,
                "release_state": "ENGINEERING_GATE",
                "depends_on": "LOCAL_SMOKE_AND_TEN_MACHINE_CANARY",
                "result_deadline_days": 7,
            },
            {
                "cycle_id": "CYCLE_2",
                "question": "Is any timing effect caused by timing, cumulative dose, or replay ratio?",
                "seed_scope": "DISCOVERY_TIMING_DOSE",
                "seed_count": 8,
                "release_state": "HELD",
                "depends_on": "CYCLE_1_REGISTERED_DECISION",
                "result_deadline_days": 7,
            },
            {
                "cycle_id": "CYCLE_3",
                "question": "Does a learnable weak-defect guard reduce tail harm at fixed total slots?",
                "seed_scope": "DISCOVERY_GUARD",
                "seed_count": 8,
                "release_state": "HELD",
                "depends_on": "CYCLE_2_POLICY_FREEZE",
                "result_deadline_days": 7,
            },
            {
                "cycle_id": "CYCLE_4",
                "question": "Does one frozen combined policy replicate on entirely unseen seeds?",
                "seed_scope": "UNSEEN_CONFIRMATION",
                "seed_count": 14,
                "release_state": "HELD",
                "depends_on": "CYCLE_3_FINAL_POLICY_FREEZE",
                "result_deadline_days": 7,
            },
        ]
    )
    arms: list[dict[str, Any]] = []

    def add(
        cycle: str,
        arm: str,
        ratio: str,
        policy: str,
        purpose: str,
        *,
        guard_rule: str = "NONE",
        guard_percent: str = "0",
        release_state: str,
    ) -> None:
        arms.append(
            {
                "cycle_id": cycle,
                "arm_id": arm,
                "ratio_id": ratio,
                "policy_id": policy,
                "schedule_id": f"{cycle}__{ratio}__{policy}__{guard_rule}__G{guard_percent}PCT",
                "guard_rule": guard_rule,
                "guard_percent": guard_percent,
                "purpose": purpose,
                "release_state": release_state,
                "canonical_lock_file_sha256": lock_sha,
            }
        )

    add(
        "CYCLE_1",
        "C1_T_RHO_2P5_CONTINUOUS",
        "RHO_2P5_PERCENT",
        "CONTINUOUS",
        "High-pressure continuous reference.",
        release_state="ENGINEERING_GATE",
    )
    add(
        "CYCLE_1",
        "C1_T_RHO_2P5_SAME_PEAK_TAPER",
        "RHO_2P5_PERCENT",
        "SAME_PEAK_TAPER",
        "Tests late-exposure reduction with the same initial peak.",
        release_state="ENGINEERING_GATE",
    )
    add(
        "CYCLE_1",
        "C1_NR_NO_REPLAY",
        "NO_REPLAY",
        "NO_REPLAY",
        "Separates any replay effect from the canonical base learner.",
        release_state="ENGINEERING_GATE",
    )
    add(
        "CYCLE_2",
        "C2_T_RHO_2P5_DOSE_MATCHED_TAPER",
        "RHO_2P5_PERCENT",
        "DOSE_MATCHED_TAPER",
        "Separates timing from cumulative exposure at high pressure.",
        release_state="HELD",
    )
    for token in ("0P5", "1P0"):
        ratio = f"RHO_{token}_PERCENT"
        for policy in ("CONTINUOUS", "SAME_PEAK_TAPER", "DOSE_MATCHED_TAPER"):
            add(
                "CYCLE_2",
                f"C2_T_RHO_{token}_{policy}",
                ratio,
                policy,
                "Ratio transfer under a fixed timing/dose definition.",
                release_state="HELD",
            )
    for arm, guard_rule, guard_percent, purpose in (
        ("C3_G0_NO_GUARD", "NONE", "0", "Selected normal policy without defect replacement."),
        ("C3_G_RAW_10PCT", "OOF_GAP_GUARD_RAW", "10", "Historical score-based guard comparator."),
        ("C3_G_LEARNABLE_10PCT", "OOF_LEARNABLE_WEAK_DEFECT", "10", "Temporal weak-defect guard."),
        ("C3_G_LEARNABLE_20PCT", "OOF_LEARNABLE_WEAK_DEFECT", "20", "Guard fraction response."),
        ("C3_G_MATCHED_RANDOM_10PCT", "MATCHED_RANDOM_DEFECT", "10", "Guard-selection control at 10 percent."),
        ("C3_G_MATCHED_RANDOM_20PCT", "MATCHED_RANDOM_DEFECT", "20", "Guard-selection control at 20 percent."),
    ):
        add(
            "CYCLE_3",
            arm,
            "GATE_SELECTED_RATIO",
            "GATE_SELECTED_POLICY",
            purpose,
            guard_rule=guard_rule,
            guard_percent=guard_percent,
            release_state="HELD",
        )
    for arm, purpose in (
        ("C4_FINAL_POLICY", "Frozen complete policy."),
        ("C4_SAME_SELECTION_CONTINUOUS", "Timing component ablation."),
        ("C4_DYNAMIC_NO_GUARD", "Guard component ablation."),
        ("C4_R1_GLOBAL_RANDOM", "Global random replay control."),
        ("C4_R2_DISJOINT_MATCHED_RANDOM", "Disjoint difficulty-matched replay control."),
        ("C4_NR_NO_REPLAY", "Canonical learner without replay."),
    ):
        add(
            "CYCLE_4",
            arm,
            "FINAL_FROZEN_RATIO",
            "FINAL_FROZEN_POLICY",
            purpose,
            guard_rule="FINAL_FROZEN_GUARD",
            guard_percent="FINAL",
            release_state="HELD",
        )
    return cycles, pd.DataFrame(arms)


def _stable_seed(namespace: str, index: int, excluded: set[int]) -> tuple[int, str]:
    nonce = 0
    while True:
        material = f"{CAMPAIGN_ID}|{namespace}|{index:03d}|{nonce}"
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest().upper()
        seed = int(digest[:16], 16) % (2**31 - 1)
        if seed > 0 and seed not in excluded:
            return seed, digest
        nonce += 1


def build_seed_registry(*, prior_training_seeds: Iterable[int]) -> pd.DataFrame:
    excluded = {int(value) for value in prior_training_seeds}
    rows: list[dict[str, Any]] = []
    scopes = (
        ("DISCOVERY_TIMING_DOSE", 8, "CYCLE_1;CYCLE_2"),
        ("DISCOVERY_GUARD", 8, "CYCLE_3"),
        ("UNSEEN_CONFIRMATION", 14, "CYCLE_4"),
    )
    global_index = 1
    for scope, count, cycles in scopes:
        for scope_index in range(1, count + 1):
            seed, digest = _stable_seed(scope, scope_index, excluded)
            excluded.add(seed)
            rows.append(
                {
                    "seed_id": f"S{global_index:03d}",
                    "seed_scope": scope,
                    "scope_index": scope_index,
                    "training_seed": seed,
                    "seed_derivation_sha256": digest,
                    "eligible_cycles": cycles,
                    "prior_seed_collision": False,
                    "paired_within_scope": True,
                }
            )
            global_index += 1
    return pd.DataFrame(rows)
