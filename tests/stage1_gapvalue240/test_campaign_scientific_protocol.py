from __future__ import annotations

from fractions import Fraction
import re

import pandas as pd
import pytest

from stage1_gapvalue240.campaign_scientific_protocol import (
    BASE_SAMPLE_COUNT,
    ProtocolError,
    build_cycle_registry,
    build_epoch_replay_schedule,
    build_seed_registry,
    collapse_epoch_schedule,
    ratio_to_slots,
    validate_epoch_schedule,
)


@pytest.mark.parametrize(
    ("ratio_id", "expected"),
    [
        ("RHO_0P5_PERCENT", 600),
        ("RHO_1P0_PERCENT", 1200),
        ("RHO_2P5_PERCENT", 3000),
    ],
)
def test_percentages_are_derived_from_the_full_120k_pool(
    ratio_id: str, expected: int
) -> None:
    assert BASE_SAMPLE_COUNT == 120_000
    assert ratio_to_slots(ratio_id) == expected


@pytest.mark.parametrize(
    "ratio_id", ["RHO_0P5_PERCENT", "RHO_1P0_PERCENT", "RHO_2P5_PERCENT"]
)
def test_timing_and_dose_schedules_have_exact_exposure_contracts(ratio_id: str) -> None:
    continuous = build_epoch_replay_schedule(ratio_id, "CONTINUOUS")
    same_peak = build_epoch_replay_schedule(ratio_id, "SAME_PEAK_TAPER")
    dose_matched = build_epoch_replay_schedule(ratio_id, "DOSE_MATCHED_TAPER")

    for frame in (continuous, same_peak, dose_matched):
        assert validate_epoch_schedule(frame)["status"] == "PASS"
        assert frame.epoch.tolist() == list(range(1, 201))
        assert (
            frame.normal_replay_slots + frame.defect_guard_slots
            == frame.total_replay_slots
        ).all()

    continuous_total = int(continuous.total_replay_slots.sum())
    assert int(same_peak.total_replay_slots.sum()) * 4 == continuous_total * 3
    assert int(dose_matched.total_replay_slots.sum()) == continuous_total
    assert same_peak.total_replay_slots.max() == continuous.total_replay_slots.max()
    assert dose_matched.total_replay_slots.max() * 3 == continuous.total_replay_slots.max() * 4
    assert same_peak.loc[same_peak.epoch > 160, "total_replay_slots"].eq(0).all()
    assert dose_matched.loc[dose_matched.epoch > 160, "total_replay_slots"].eq(0).all()


def test_collapsed_schedules_use_identical_restart_boundaries() -> None:
    boundaries = None
    for policy in ("CONTINUOUS", "SAME_PEAK_TAPER", "DOSE_MATCHED_TAPER", "NO_REPLAY"):
        ratio = "NO_REPLAY" if policy == "NO_REPLAY" else "RHO_2P5_PERCENT"
        collapsed = collapse_epoch_schedule(build_epoch_replay_schedule(ratio, policy))
        observed = list(zip(collapsed.segment_start_epoch, collapsed.segment_end_epoch))
        assert observed == [(1, 140), (141, 150), (151, 160), (161, 200)]
        boundaries = observed if boundaries is None else boundaries
        assert observed == boundaries


@pytest.mark.parametrize("guard_fraction", [Fraction(1, 10), Fraction(1, 5)])
def test_guard_replaces_slots_without_increasing_total_dose(guard_fraction: Fraction) -> None:
    baseline = build_epoch_replay_schedule("RHO_2P5_PERCENT", "SAME_PEAK_TAPER")
    guarded = build_epoch_replay_schedule(
        "RHO_2P5_PERCENT",
        "SAME_PEAK_TAPER",
        guard_fraction=guard_fraction,
    )

    assert guarded.total_replay_slots.tolist() == baseline.total_replay_slots.tolist()
    assert (guarded.normal_replay_slots + guarded.defect_guard_slots).equals(
        guarded.total_replay_slots
    )
    assert guarded.defect_guard_slots.sum() > 0


def test_scientific_ids_use_percentages_and_never_absolute_budget_names() -> None:
    cycles, arms = build_cycle_registry("A" * 64)
    forbidden = re.compile(r"(?:^|_)(?:600|1200|3000|4000)(?:_|$)")
    assert not any(forbidden.search(value) for value in arms.arm_id.astype(str))
    assert not any(forbidden.search(value) for value in arms.schedule_id.astype(str))
    assert set(cycles.cycle_id) == {"CYCLE_1", "CYCLE_2", "CYCLE_3", "CYCLE_4"}
    assert set(arms.canonical_lock_file_sha256) == {"A" * 64}
    assert cycles.set_index("cycle_id").loc["CYCLE_1", "release_state"] == "ENGINEERING_GATE"
    assert cycles.set_index("cycle_id").loc["CYCLE_4", "release_state"] == "HELD"


def test_seed_scopes_are_disjoint_and_confirmation_is_unseen() -> None:
    prior = {10, 20, 30}
    seeds = build_seed_registry(prior_training_seeds=prior)

    assert len(seeds) == 30
    assert seeds.training_seed.is_unique
    assert not set(seeds.training_seed) & prior
    scope_counts = seeds.groupby("seed_scope").size().to_dict()
    assert scope_counts == {
        "DISCOVERY_GUARD": 8,
        "DISCOVERY_TIMING_DOSE": 8,
        "UNSEEN_CONFIRMATION": 14,
    }
    timing = set(seeds.loc[seeds.seed_scope == "DISCOVERY_TIMING_DOSE", "training_seed"])
    guard = set(seeds.loc[seeds.seed_scope == "DISCOVERY_GUARD", "training_seed"])
    confirm = set(seeds.loc[seeds.seed_scope == "UNSEEN_CONFIRMATION", "training_seed"])
    assert timing.isdisjoint(guard)
    assert timing.isdisjoint(confirm)
    assert guard.isdisjoint(confirm)


def test_unknown_ratio_or_policy_fails_before_queue_generation() -> None:
    with pytest.raises(ProtocolError, match="ratio"):
        ratio_to_slots("RHO_UNKNOWN")
    with pytest.raises(ProtocolError, match="policy"):
        build_epoch_replay_schedule("RHO_0P5_PERCENT", "ADAPT_AFTER_RESULTS")
