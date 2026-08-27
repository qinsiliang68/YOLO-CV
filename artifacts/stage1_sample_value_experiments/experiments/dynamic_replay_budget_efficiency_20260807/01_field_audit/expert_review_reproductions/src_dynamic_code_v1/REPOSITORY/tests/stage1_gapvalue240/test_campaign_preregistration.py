from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd
import pytest

from stage1_gapvalue240.campaign_preregistration import (
    PreregistrationError,
    build_campaign_preregistration,
    validate_preregistration_tables,
)
from stage1_gapvalue240.util import sha256_file


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "configs/stage1_gapvalue240/CANONICAL_TRAINING_LOCK_v1.json"


def _treatment(path: Path, count: int = 6000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "rank": index,
                "sample_id": f"Det/images/normal_train/{index:08d}.png",
                "y_true": 0,
                "oof_fold": (index - 1) % 10,
                "dynamic_bucket": "learnable_hard",
                "mean_p_defect": 0.9 - index / 100_000,
                "correct_rate": 0.2 + index / 100_000,
                "std_p_defect": 0.3 - index / 100_000,
                "replay_role": "normal_replay",
                "source_method": "GapCritical-Strict",
            }
            for index in range(1, count + 1)
        ]
    ).to_csv(path, index=False)
    return path


def _build(tmp_path: Path):
    return build_campaign_preregistration(
        tmp_path / "03_preregistration",
        treatment_ranking_source=_treatment(tmp_path / "sources/treatment.csv"),
        canonical_lock_path=LOCK,
        machine_count=10,
    )


def _read(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(root / name, keep_default_na=False)


def test_builds_four_cycle_protocol_but_releases_only_frozen_cycle_one(tmp_path: Path) -> None:
    result = _build(tmp_path)
    root = Path(result["output_dir"])
    cycles = _read(root, "CYCLE_REGISTRY.csv")
    arms = _read(root, "ARM_REGISTRY.csv")
    matrix = _read(root, "EXPERIMENT_MATRIX.csv")
    jobs = _read(root, "PHYSICAL_JOB_GRAPH.csv")

    assert set(cycles.cycle_id) == {"CYCLE_1", "CYCLE_2", "CYCLE_3", "CYCLE_4"}
    assert len(matrix) == 80
    assert matrix.groupby("cycle_id").size().to_dict() == {"CYCLE_1": 24, "CYCLE_2": 56}
    assert set(matrix.seed_scope) == {"DISCOVERY_TIMING_DOSE"}
    assert matrix.training_seed.nunique() == 8
    assert len(arms[arms.cycle_id == "CYCLE_3"]) == 6
    assert len(arms[arms.cycle_id == "CYCLE_4"]) == 6
    assert set(jobs.loc[jobs.cycle_id == "CYCLE_1", "release_state"]) == {
        "ENGINEERING_GATE"
    }
    assert set(jobs.loc[jobs.cycle_id == "CYCLE_2", "release_state"]) == {"HELD"}
    assert result["logical_run_count"] == 80
    assert result["physical_job_count"] == 296


def test_schedule_uses_percent_ids_exact_dose_and_common_restart_boundaries(tmp_path: Path) -> None:
    result = _build(tmp_path)
    root = Path(result["output_dir"])
    schedule = _read(root, "REPLAY_SCHEDULE.csv")
    epoch = _read(root, "EPOCH_REPLAY_SCHEDULE.csv")
    forbidden = re.compile(r"(?:^|_)(?:600|1200|3000|4000)(?:_|$)")

    assert not any(forbidden.search(value) for value in schedule.schedule_id.astype(str))
    for _, group in schedule.groupby("arm_id"):
        assert list(zip(group.segment_start_epoch, group.segment_end_epoch)) == [
            (1, 140),
            (141, 150),
            (151, 160),
            (161, 200),
        ]
    for ratio in ("RHO_0P5_PERCENT", "RHO_1P0_PERCENT", "RHO_2P5_PERCENT"):
        subset = epoch[epoch.ratio_id == ratio]
        totals = {
            policy: int(group.total_replay_slots.sum())
            for policy, group in subset.groupby("policy_id")
        }
        assert totals["DOSE_MATCHED_TAPER"] == totals["CONTINUOUS"]
        assert totals["SAME_PEAK_TAPER"] * 4 == totals["CONTINUOUS"] * 3


def test_treatment_pool_is_nested_and_sized_for_the_largest_derived_peak(tmp_path: Path) -> None:
    result = _build(tmp_path)
    root = Path(result["output_dir"])
    frozen = _read(root, "frozen_selections/TREATMENT_GAPCRITICAL_NESTED.csv")
    schedule = _read(root, "REPLAY_SCHEDULE.csv")
    bindings = _read(root, "SELECTION_BINDINGS.csv")

    assert len(frozen) == 6000
    assert frozen.selection_rank.tolist() == list(range(1, 6001))
    assert frozen.y_true.eq(0).all()
    assert schedule.total_replay_slots.max() == 4000
    assert schedule.total_replay_slots.max() <= len(frozen)
    treatment = bindings.set_index("selection_id").loc["TREATMENT_GAPCRITICAL_NESTED"]
    assert treatment.row_count == 6000
    assert treatment.selection_digest
    assert treatment.source_sha256 == sha256_file(tmp_path / "sources/treatment.csv")


def test_every_manifest_and_job_carries_the_same_canonical_lock_sha(tmp_path: Path) -> None:
    result = _build(tmp_path)
    root = Path(result["output_dir"])
    expected = sha256_file(LOCK)
    binding = json.loads((root / "CANONICAL_TRAINING_LOCK_BINDING.json").read_text(encoding="utf-8"))
    assert binding["canonical_lock_file_sha256"] == expected
    for filename in ("ARM_REGISTRY.csv", "EXPERIMENT_MATRIX.csv", "PHYSICAL_JOB_GRAPH.csv"):
        frame = _read(root, filename)
        assert set(frame.canonical_lock_file_sha256) == {expected}
    validation = json.loads((root / "PREREGISTRATION_VALIDATION.json").read_text(encoding="utf-8"))
    assert validation["canonical_lock_file_sha256"] == expected
    assert validation["status"] == "PASS"


def test_shared_prefixes_and_all_comparators_have_identical_restart_boundaries(tmp_path: Path) -> None:
    result = _build(tmp_path)
    root = Path(result["output_dir"])
    jobs = _read(root, "PHYSICAL_JOB_GRAPH.csv")
    matrix = _read(root, "EXPERIMENT_MATRIX.csv")

    for row in matrix.itertuples(index=False):
        arm_jobs = jobs[
            (jobs.seed_id == row.seed_id)
            & (
                (jobs.logical_arm_id == row.arm_id)
                | (jobs.job_id == row.parent_job_id)
            )
        ].sort_values("segment_start_epoch")
        assert list(zip(arm_jobs.segment_start_epoch, arm_jobs.segment_end_epoch)) == [
            (1, 140),
            (141, 150),
            (151, 160),
            (161, 200),
        ]
        assert arm_jobs.machine_id.nunique() == 1
    assert not jobs.job_id.duplicated().any()
    assert set(jobs.dependency_job_id) - {""} <= set(jobs.job_id)


def test_seed_scopes_are_disjoint_and_future_cycles_remain_unbound(tmp_path: Path) -> None:
    result = _build(tmp_path)
    root = Path(result["output_dir"])
    seeds = _read(root, "SEED_REGISTRY.csv")
    gates = _read(root, "GATED_STAGE_TEMPLATES.csv")

    assert len(seeds) == 30
    assert seeds.training_seed.is_unique
    by_scope = {
        scope: set(group.training_seed)
        for scope, group in seeds.groupby("seed_scope")
    }
    assert by_scope["DISCOVERY_TIMING_DOSE"].isdisjoint(by_scope["DISCOVERY_GUARD"])
    assert by_scope["DISCOVERY_TIMING_DOSE"].isdisjoint(by_scope["UNSEEN_CONFIRMATION"])
    assert by_scope["DISCOVERY_GUARD"].isdisjoint(by_scope["UNSEEN_CONFIRMATION"])
    assert set(gates.cycle_id) == {"CYCLE_3", "CYCLE_4"}
    assert gates.binding_status.eq("UNBOUND_SCIENTIFIC_GATE").all()


def test_validation_rejects_cumulative_dose_or_lock_drift(tmp_path: Path) -> None:
    result = _build(tmp_path)
    root = Path(result["output_dir"])
    tables = {
        "cycles": _read(root, "CYCLE_REGISTRY.csv"),
        "arms": _read(root, "ARM_REGISTRY.csv"),
        "seeds": _read(root, "SEED_REGISTRY.csv"),
        "matrix": _read(root, "EXPERIMENT_MATRIX.csv"),
        "epoch_schedule": _read(root, "EPOCH_REPLAY_SCHEDULE.csv"),
        "schedule": _read(root, "REPLAY_SCHEDULE.csv"),
        "jobs": _read(root, "PHYSICAL_JOB_GRAPH.csv"),
        "bindings": _read(root, "SELECTION_BINDINGS.csv"),
    }
    row = tables["epoch_schedule"].index[
        (tables["epoch_schedule"].ratio_id == "RHO_2P5_PERCENT")
        & (tables["epoch_schedule"].policy_id == "DOSE_MATCHED_TAPER")
        & (tables["epoch_schedule"].epoch == 1)
    ][0]
    tables["epoch_schedule"].loc[row, "total_replay_slots"] += 1
    with pytest.raises(PreregistrationError, match="role-slot arithmetic|dose"):
        validate_preregistration_tables(tables, canonical_lock_file_sha256=sha256_file(LOCK))

    tables["epoch_schedule"] = _read(root, "EPOCH_REPLAY_SCHEDULE.csv")
    tables["jobs"].loc[0, "canonical_lock_file_sha256"] = "0" * 64
    with pytest.raises(PreregistrationError, match="canonical lock"):
        validate_preregistration_tables(tables, canonical_lock_file_sha256=sha256_file(LOCK))


def test_rejects_short_or_non_normal_treatment_source(tmp_path: Path) -> None:
    with pytest.raises(PreregistrationError, match="at least 4000"):
        build_campaign_preregistration(
            tmp_path / "short",
            treatment_ranking_source=_treatment(tmp_path / "short.csv", count=3999),
            canonical_lock_path=LOCK,
        )

    source = _treatment(tmp_path / "bad.csv")
    frame = pd.read_csv(source)
    frame.loc[0, "y_true"] = 1
    frame.to_csv(source, index=False)
    with pytest.raises(PreregistrationError, match="normal"):
        build_campaign_preregistration(
            tmp_path / "bad",
            treatment_ranking_source=source,
            canonical_lock_path=LOCK,
        )
