"""Build the staged, percentage-based dynamic replay preregistration.

The preregistration has two different kinds of objects:

* frozen Cycle 1/2 jobs that can be executed after engineering release gates;
* unbound Cycle 3/4 templates whose scientific parameters may only be filled by
  the registered preceding-cycle decision.

Every executable object carries the file hash of the canonical 240-run training
lock.  Absolute replay counts are derived implementation fields; scientific IDs
and hypotheses use percentages only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

import pandas as pd

from .campaign_canonical_lock import load_canonical_training_lock
from .campaign_layout import CAMPAIGN_ID
from .campaign_scientific_protocol import (
    BASE_SAMPLE_COUNT,
    SCHEDULE_BOUNDARIES,
    build_cycle_registry,
    build_epoch_replay_schedule,
    build_seed_registry,
    collapse_epoch_schedule,
)
from .util import sha256_file, stable_hash


class PreregistrationError(RuntimeError):
    """Raised when the preregistered design is incomplete or inconsistent."""


KEY_CHECKPOINT_EPOCHS = (120, 140, 150, 160, 180, 200)
TREATMENT_SELECTION_ID = "TREATMENT_GAPCRITICAL_NESTED"
NO_SELECTION_ID = "NR_NO_SELECTION"
DEFAULT_PRIOR_TRAINING_SEEDS = (
    1829910790,
    1385197443,
    1343723075,
    440790218,
    1625332720,
    1024521099,
    216172108,
    53303004,
)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_text(path, frame.to_csv(index=False, lineterminator="\n"))


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _selection_digest(frame: pd.DataFrame) -> str:
    fields = ["selection_rank", "role_rank", "sample_id", "y_true", "replay_role"]
    return stable_hash(frame[fields].to_dict("records"))


def _load_treatment_ranking(path: str | Path) -> tuple[pd.DataFrame, Path, str]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source, keep_default_na=False)
    required = {"sample_id", "y_true"}
    missing = required - set(frame.columns)
    if missing:
        raise PreregistrationError(f"treatment ranking missing columns: {sorted(missing)}")
    if len(frame) < 4000:
        raise PreregistrationError(
            f"treatment ranking requires at least 4000 rows for the largest derived peak; got {len(frame)}"
        )
    labels = pd.to_numeric(frame["y_true"], errors="raise").astype(int)
    if set(labels) != {0}:
        raise PreregistrationError("treatment ranking must contain only normal samples (y_true=0)")
    sample_ids = frame["sample_id"].astype(str)
    if sample_ids.eq("").any() or sample_ids.duplicated().any():
        raise PreregistrationError("treatment ranking sample_id values must be non-empty and unique")
    if "rank" in frame:
        role_rank = pd.to_numeric(frame["rank"], errors="raise").astype(int)
        if role_rank.tolist() != list(range(1, len(frame) + 1)):
            raise PreregistrationError("treatment ranking rank must be contiguous from one")
    else:
        role_rank = pd.Series(range(1, len(frame) + 1), index=frame.index)

    frozen = pd.DataFrame(
        {
            "selection_id": TREATMENT_SELECTION_ID,
            "selection_rank": range(1, len(frame) + 1),
            "role_rank": role_rank,
            "sample_id": sample_ids,
            "y_true": labels,
            "replay_role": "normal_replay",
            "oof_fold": frame["oof_fold"].astype(str) if "oof_fold" in frame else "",
            "dynamic_bucket": (
                frame["dynamic_bucket"].astype(str) if "dynamic_bucket" in frame else ""
            ),
            "mean_p_defect": frame["mean_p_defect"] if "mean_p_defect" in frame else "",
            "correct_rate": frame["correct_rate"] if "correct_rate" in frame else "",
            "std_p_defect": frame["std_p_defect"] if "std_p_defect" in frame else "",
            "source_method": (
                frame["source_method"].astype(str)
                if "source_method" in frame
                else "GapCritical-Strict"
            ),
        }
    )
    return frozen, source, sha256_file(source)


def _prepare_selections(
    treatment_ranking_source: str | Path,
    output: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    frozen, source, source_hash = _load_treatment_ranking(treatment_ranking_source)
    frozen_dir = output / "frozen_selections"
    treatment_path = frozen_dir / f"{TREATMENT_SELECTION_ID}.csv"
    _atomic_csv(treatment_path, frozen)
    treatment_digest = _selection_digest(frozen)

    empty = frozen.iloc[:0].copy()
    empty["selection_id"] = NO_SELECTION_ID
    empty_path = frozen_dir / f"{NO_SELECTION_ID}.csv"
    _atomic_csv(empty_path, empty)
    empty_digest = _selection_digest(empty)

    bindings = pd.DataFrame(
        [
            {
                "selection_id": TREATMENT_SELECTION_ID,
                "selection_role": "TARGET_NORMAL_NESTED_POOL",
                "selection_file": treatment_path.relative_to(output).as_posix(),
                "selection_digest": treatment_digest,
                "row_count": len(frozen),
                "normal_count": len(frozen),
                "defect_guard_count": 0,
                "source_path": str(source),
                "source_sha256": source_hash,
                "subset_rule": "TOP_SELECTION_RANK_PREFIX_DERIVED_FROM_PERCENTAGE",
                "scientific_status": "FROZEN_FOR_CYCLE_1_AND_2_ONLY",
            },
            {
                "selection_id": NO_SELECTION_ID,
                "selection_role": "NO_REPLAY",
                "selection_file": empty_path.relative_to(output).as_posix(),
                "selection_digest": empty_digest,
                "row_count": 0,
                "normal_count": 0,
                "defect_guard_count": 0,
                "source_path": "",
                "source_sha256": "",
                "subset_rule": "NONE",
                "scientific_status": "FROZEN",
            },
        ]
    )
    return bindings, {
        TREATMENT_SELECTION_ID: treatment_digest,
        NO_SELECTION_ID: empty_digest,
    }


def _active_arms(arms: pd.DataFrame) -> pd.DataFrame:
    return arms[arms.cycle_id.isin(["CYCLE_1", "CYCLE_2"])].copy().reset_index(drop=True)


def _build_schedules(active_arms: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    epoch_frames: list[pd.DataFrame] = []
    segment_frames: list[pd.DataFrame] = []
    for arm in active_arms.itertuples(index=False):
        epoch = build_epoch_replay_schedule(str(arm.ratio_id), str(arm.policy_id))
        epoch.insert(0, "cycle_id", str(arm.cycle_id))
        epoch.insert(1, "arm_id", str(arm.arm_id))
        epoch["schedule_id"] = str(arm.schedule_id)
        epoch["canonical_lock_file_sha256"] = str(arm.canonical_lock_file_sha256)
        epoch_frames.append(epoch)

        segment = collapse_epoch_schedule(epoch)
        segment.insert(0, "cycle_id", str(arm.cycle_id))
        segment.insert(1, "arm_id", str(arm.arm_id))
        segment["schedule_id"] = str(arm.schedule_id)
        segment["selection_id"] = (
            NO_SELECTION_ID if str(arm.policy_id) == "NO_REPLAY" else TREATMENT_SELECTION_ID
        )
        segment["configured_train_samples_per_epoch"] = (
            BASE_SAMPLE_COUNT + segment.total_replay_slots.astype(int)
        )
        segment["canonical_lock_file_sha256"] = str(arm.canonical_lock_file_sha256)
        segment_frames.append(segment)
    return (
        pd.concat(epoch_frames, ignore_index=True),
        pd.concat(segment_frames, ignore_index=True),
    )


def _machine_id(bundle_index: int, machine_count: int) -> str:
    return f"M{(bundle_index % machine_count) + 1:02d}"


def _retained_checkpoints(start: int, end: int) -> str:
    return ";".join(str(value) for value in KEY_CHECKPOINT_EPOCHS if start <= value <= end)


def _build_jobs(
    timing_seeds: pd.DataFrame,
    active_arms: pd.DataFrame,
    schedule: pd.DataFrame,
    machine_count: int,
    canonical_lock_file_sha256: str,
) -> tuple[pd.DataFrame, dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    if machine_count <= 0:
        raise PreregistrationError("machine_count must be positive")
    arm_index = active_arms.set_index("arm_id")
    schedule_by_arm = {
        arm_id: part.sort_values("segment_index", kind="stable")
        for arm_id, part in schedule.groupby("arm_id", sort=False)
    }
    rows: list[dict[str, Any]] = []
    parent_by_seed_arm: dict[tuple[str, str], str] = {}
    machine_by_seed_arm: dict[tuple[str, str], str] = {}
    bundle_index = 0

    def append_job(
        *,
        seed_id: str,
        training_seed: int,
        cycle_id: str,
        logical_arm_id: str,
        schedule_id: str,
        ratio_id: str,
        policy_id: str,
        selection_id: str,
        segment_index: int,
        start: int,
        end: int,
        normal: int,
        defect: int,
        dependency: str,
        machine: str,
        job_kind: str,
        release_state: str,
    ) -> str:
        job_id = (
            f"JOB_{seed_id}_{ratio_id}_{policy_id}_{logical_arm_id}_"
            f"E{start:03d}_{end:03d}"
        )
        rows.append(
            {
                "job_id": job_id,
                "cycle_id": cycle_id,
                "release_state": release_state,
                "seed_id": seed_id,
                "training_seed": training_seed,
                "logical_arm_id": logical_arm_id,
                "schedule_id": schedule_id,
                "ratio_id": ratio_id,
                "policy_id": policy_id,
                "selection_id": selection_id,
                "job_kind": job_kind,
                "segment_index": segment_index,
                "segment_start_epoch": start,
                "segment_end_epoch": end,
                "planned_epoch_equivalents": end - start + 1,
                "normal_replay_slots": normal,
                "defect_guard_slots": defect,
                "total_replay_slots": normal + defect,
                "configured_train_samples_per_epoch": BASE_SAMPLE_COUNT + normal + defect,
                "expected_steps_batch128": math.ceil(
                    (BASE_SAMPLE_COUNT + normal + defect) / 128
                ),
                "dependency_job_id": dependency,
                "resume_from_epoch": 0 if start == 1 else start - 1,
                "machine_id": machine,
                "checkpoint_epochs_to_retain": _retained_checkpoints(start, end),
                "branch_checkpoint_required": bool(dependency),
                "canonical_lock_file_sha256": canonical_lock_file_sha256,
            }
        )
        return job_id

    for seed in timing_seeds.itertuples(index=False):
        seed_id = str(seed.seed_id)
        training_seed = int(seed.training_seed)
        for ratio_id in ("RHO_0P5_PERCENT", "RHO_1P0_PERCENT", "RHO_2P5_PERCENT"):
            continuous_id = next(
                value
                for value in active_arms.arm_id.astype(str)
                if str(arm_index.loc[value, "ratio_id"]) == ratio_id
                and str(arm_index.loc[value, "policy_id"]) == "CONTINUOUS"
            )
            taper_id = next(
                value
                for value in active_arms.arm_id.astype(str)
                if str(arm_index.loc[value, "ratio_id"]) == ratio_id
                and str(arm_index.loc[value, "policy_id"]) == "SAME_PEAK_TAPER"
            )
            machine = _machine_id(bundle_index, machine_count)
            bundle_index += 1
            prefix = schedule_by_arm[continuous_id].iloc[0]
            parent_id = append_job(
                seed_id=seed_id,
                training_seed=training_seed,
                cycle_id=str(arm_index.loc[continuous_id, "cycle_id"]),
                logical_arm_id=f"SHARED_{ratio_id}_CONTINUOUS_TAPER_PREFIX",
                schedule_id=f"SHARED_{ratio_id}_CONTINUOUS_TAPER_PREFIX",
                ratio_id=ratio_id,
                policy_id="SHARED_PREFIX",
                selection_id=TREATMENT_SELECTION_ID,
                segment_index=1,
                start=1,
                end=140,
                normal=int(prefix.normal_replay_slots),
                defect=int(prefix.defect_guard_slots),
                dependency="",
                machine=machine,
                job_kind="SHARED_TREATMENT_PREFIX",
                release_state=str(arm_index.loc[continuous_id, "release_state"]),
            )
            for arm_id in (continuous_id, taper_id):
                parent_by_seed_arm[(seed_id, arm_id)] = parent_id
                machine_by_seed_arm[(seed_id, arm_id)] = machine
                dependency = parent_id
                arm = arm_index.loc[arm_id]
                for segment in schedule_by_arm[arm_id].iloc[1:].itertuples(index=False):
                    dependency = append_job(
                        seed_id=seed_id,
                        training_seed=training_seed,
                        cycle_id=str(arm.cycle_id),
                        logical_arm_id=arm_id,
                        schedule_id=str(arm.schedule_id),
                        ratio_id=ratio_id,
                        policy_id=str(arm.policy_id),
                        selection_id=TREATMENT_SELECTION_ID,
                        segment_index=int(segment.segment_index),
                        start=int(segment.segment_start_epoch),
                        end=int(segment.segment_end_epoch),
                        normal=int(segment.normal_replay_slots),
                        defect=int(segment.defect_guard_slots),
                        dependency=dependency,
                        machine=machine,
                        job_kind="ARM_SEGMENT",
                        release_state=str(arm.release_state),
                    )

        for ratio_id in ("RHO_0P5_PERCENT", "RHO_1P0_PERCENT", "RHO_2P5_PERCENT"):
            arm_id = next(
                value
                for value in active_arms.arm_id.astype(str)
                if str(arm_index.loc[value, "ratio_id"]) == ratio_id
                and str(arm_index.loc[value, "policy_id"]) == "DOSE_MATCHED_TAPER"
            )
            machine = _machine_id(bundle_index, machine_count)
            bundle_index += 1
            machine_by_seed_arm[(seed_id, arm_id)] = machine
            parent_by_seed_arm[(seed_id, arm_id)] = ""
            dependency = ""
            arm = arm_index.loc[arm_id]
            for segment in schedule_by_arm[arm_id].itertuples(index=False):
                dependency = append_job(
                    seed_id=seed_id,
                    training_seed=training_seed,
                    cycle_id=str(arm.cycle_id),
                    logical_arm_id=arm_id,
                    schedule_id=str(arm.schedule_id),
                    ratio_id=ratio_id,
                    policy_id=str(arm.policy_id),
                    selection_id=TREATMENT_SELECTION_ID,
                    segment_index=int(segment.segment_index),
                    start=int(segment.segment_start_epoch),
                    end=int(segment.segment_end_epoch),
                    normal=int(segment.normal_replay_slots),
                    defect=int(segment.defect_guard_slots),
                    dependency=dependency,
                    machine=machine,
                    job_kind="ARM_SEGMENT",
                    release_state=str(arm.release_state),
                )

        nr_id = "C1_NR_NO_REPLAY"
        machine = _machine_id(bundle_index, machine_count)
        bundle_index += 1
        machine_by_seed_arm[(seed_id, nr_id)] = machine
        parent_by_seed_arm[(seed_id, nr_id)] = ""
        dependency = ""
        arm = arm_index.loc[nr_id]
        for segment in schedule_by_arm[nr_id].itertuples(index=False):
            dependency = append_job(
                seed_id=seed_id,
                training_seed=training_seed,
                cycle_id=str(arm.cycle_id),
                logical_arm_id=nr_id,
                schedule_id=str(arm.schedule_id),
                ratio_id="NO_REPLAY",
                policy_id="NO_REPLAY",
                selection_id=NO_SELECTION_ID,
                segment_index=int(segment.segment_index),
                start=int(segment.segment_start_epoch),
                end=int(segment.segment_end_epoch),
                normal=0,
                defect=0,
                dependency=dependency,
                machine=machine,
                job_kind="ARM_SEGMENT",
                release_state=str(arm.release_state),
            )
    return pd.DataFrame(rows), parent_by_seed_arm, machine_by_seed_arm


def _build_matrix(
    timing_seeds: pd.DataFrame,
    active_arms: pd.DataFrame,
    epoch_schedule: pd.DataFrame,
    selection_digests: Mapping[str, str],
    parent_by_seed_arm: Mapping[tuple[str, str], str],
    machine_by_seed_arm: Mapping[tuple[str, str], str],
    canonical_lock_file_sha256: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in timing_seeds.itertuples(index=False):
        for arm in active_arms.itertuples(index=False):
            selection_id = (
                NO_SELECTION_ID if str(arm.policy_id) == "NO_REPLAY" else TREATMENT_SELECTION_ID
            )
            epoch = epoch_schedule[epoch_schedule.arm_id == arm.arm_id]
            rows.append(
                {
                    "logical_run_id": f"{arm.cycle_id}_{seed.seed_id}_{arm.arm_id}",
                    "cycle_id": str(arm.cycle_id),
                    "release_state": str(arm.release_state),
                    "seed_scope": str(seed.seed_scope),
                    "seed_id": str(seed.seed_id),
                    "training_seed": int(seed.training_seed),
                    "arm_id": str(arm.arm_id),
                    "ratio_id": str(arm.ratio_id),
                    "policy_id": str(arm.policy_id),
                    "schedule_id": str(arm.schedule_id),
                    "selection_id": selection_id,
                    "selection_digest": selection_digests[selection_id],
                    "parent_job_id": parent_by_seed_arm[(str(seed.seed_id), str(arm.arm_id))],
                    "machine_id": machine_by_seed_arm[(str(seed.seed_id), str(arm.arm_id))],
                    "planned_physical_job_count": 4,
                    "logical_epoch_count": 200,
                    "total_configured_replay_exposures": int(epoch.total_replay_slots.sum()),
                    "key_checkpoint_epochs": ";".join(map(str, KEY_CHECKPOINT_EPOCHS)),
                    "analysis_set": "DISCOVERY_TIMING_DOSE",
                    "blind_holdout_access": "FORBIDDEN_UNTIL_FINAL_PROTOCOL_FREEZE",
                    "canonical_lock_file_sha256": canonical_lock_file_sha256,
                }
            )
    return pd.DataFrame(rows)


def _build_gated_templates(arms: pd.DataFrame) -> pd.DataFrame:
    future = arms[arms.cycle_id.isin(["CYCLE_3", "CYCLE_4"])].copy()
    future.insert(2, "binding_status", "UNBOUND_SCIENTIFIC_GATE")
    future["required_release_artifact"] = future.cycle_id.map(
        {
            "CYCLE_3": "CYCLE_2_REGISTERED_POLICY_DECISION.json",
            "CYCLE_4": "CYCLE_3_FINAL_POLICY_FREEZE.json",
        }
    )
    future["queue_eligible"] = False
    return future


def _contrasts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cycle_id": "CYCLE_1",
                "contrast_id": "C1_LATE_EXPOSURE",
                "left_arm": "C1_T_RHO_2P5_SAME_PEAK_TAPER",
                "right_arm": "C1_T_RHO_2P5_CONTINUOUS",
                "estimand": "paired conditional effect of removing late replay at the same peak",
                "primary_endpoint": "raw_safe_frontier_fn_0_95",
                "safety_endpoint": "FN_at_TN68253",
                "efficacy_endpoint": "TN_at_FN95",
                "analysis_rule": "paired by seed; safety first; no weighted composite score",
            },
            {
                "cycle_id": "CYCLE_1",
                "contrast_id": "C1_REPLAY_VS_NONE",
                "left_arm": "C1_T_RHO_2P5_SAME_PEAK_TAPER",
                "right_arm": "C1_NR_NO_REPLAY",
                "estimand": "conditional effect of replay relative to the canonical learner",
                "primary_endpoint": "raw_safe_frontier_fn_0_95",
                "safety_endpoint": "FN_at_TN68253",
                "efficacy_endpoint": "TN_at_FN95",
                "analysis_rule": "paired descriptive mechanism contrast",
            },
            {
                "cycle_id": "CYCLE_2",
                "contrast_id": "C2_TIMING_VS_DOSE",
                "left_arm": "same-ratio dose-matched taper",
                "right_arm": "same-ratio continuous and same-peak taper",
                "estimand": "separate replay timing from cumulative replay exposure",
                "primary_endpoint": "raw_safe_frontier_fn_0_95",
                "safety_endpoint": "FN_at_TN68253",
                "efficacy_endpoint": "TN_at_FN95",
                "analysis_rule": "paired factorial contrasts within ratio; hierarchical interpretation",
            },
        ]
    )


def _hypotheses() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hypothesis_id": "H1_LATE_REPLAY_HARM",
                "cycle_id": "CYCLE_1",
                "statement": "At high pressure, stopping replay after a taper reduces weak-defect harm.",
                "falsifier": "Same-peak taper is not safer or is worse than continuous replay across paired seeds.",
            },
            {
                "hypothesis_id": "H2_TIMING_NOT_JUST_DOSE",
                "cycle_id": "CYCLE_2",
                "statement": "Replay timing contributes information beyond cumulative exposure.",
                "falsifier": "Dose-matched taper does not improve over continuous replay while same-peak taper does.",
            },
            {
                "hypothesis_id": "H3_RATIO_TRANSFER",
                "cycle_id": "CYCLE_2",
                "statement": "A timing effect has a coherent response across preregistered replay ratios.",
                "falsifier": "Direction changes without a dose-response or tail-risk explanation.",
            },
        ]
    )


def _coverage_is_exact(frame: pd.DataFrame) -> bool:
    rows = frame.sort_values("segment_index", kind="stable")
    return list(zip(rows.segment_start_epoch.astype(int), rows.segment_end_epoch.astype(int))) == list(
        SCHEDULE_BOUNDARIES
    )


def validate_preregistration_tables(
    tables: Mapping[str, pd.DataFrame],
    *,
    canonical_lock_file_sha256: str,
) -> dict[str, Any]:
    """Validate scientific arithmetic, provenance, pairing, and execution graph."""

    aliases = {
        "cycles": "cycles",
        "arms": "arms",
        "seeds": "seeds",
        "matrix": "matrix",
        "epoch_schedule": "epoch_schedule",
        "schedule": "schedule",
        "jobs": "jobs",
        "bindings": "bindings",
    }
    missing = set(aliases) - set(tables)
    if missing:
        raise PreregistrationError(f"missing preregistration tables: {sorted(missing)}")
    cycles = tables["cycles"]
    arms = tables["arms"]
    seeds = tables["seeds"]
    matrix = tables["matrix"]
    epoch = tables["epoch_schedule"]
    schedule = tables["schedule"]
    jobs = tables["jobs"]
    bindings = tables["bindings"]
    expected_lock = str(canonical_lock_file_sha256).upper()

    if set(cycles.cycle_id.astype(str)) != {"CYCLE_1", "CYCLE_2", "CYCLE_3", "CYCLE_4"}:
        raise PreregistrationError("cycle registry must contain exactly four registered cycles")
    if len(seeds) != 30 or seeds.training_seed.nunique() != 30:
        raise PreregistrationError("seed registry must contain 30 unique training seeds")
    scope_sets = {
        scope: set(part.training_seed.astype(int))
        for scope, part in seeds.groupby("seed_scope")
    }
    if set(scope_sets) != {"DISCOVERY_TIMING_DOSE", "DISCOVERY_GUARD", "UNSEEN_CONFIRMATION"}:
        raise PreregistrationError("seed scopes do not match the staged protocol")
    if any(scope_sets[left] & scope_sets[right] for left in scope_sets for right in scope_sets if left < right):
        raise PreregistrationError("seed scopes must be mutually disjoint")

    for name, frame in (("arm", arms), ("matrix", matrix), ("job", jobs)):
        if "canonical_lock_file_sha256" not in frame:
            raise PreregistrationError(f"{name} table is missing canonical lock provenance")
        observed = set(frame.canonical_lock_file_sha256.astype(str).str.upper())
        if observed != {expected_lock}:
            raise PreregistrationError(f"{name} canonical lock drift: {sorted(observed)}")

    if len(matrix) != 80 or matrix.groupby("cycle_id").size().to_dict() != {
        "CYCLE_1": 24,
        "CYCLE_2": 56,
    }:
        raise PreregistrationError("frozen logical matrix must contain 24 Cycle-1 and 56 Cycle-2 runs")
    if set(matrix.seed_scope.astype(str)) != {"DISCOVERY_TIMING_DOSE"}:
        raise PreregistrationError("only timing/dose discovery seeds may enter the frozen matrix")
    if matrix.logical_run_id.duplicated().any():
        raise PreregistrationError("logical run identifiers must be unique")
    if matrix.training_seed.nunique() != 8:
        raise PreregistrationError("frozen matrix must pair eight timing/dose seeds")

    numeric = epoch[["normal_replay_slots", "defect_guard_slots", "total_replay_slots"]].apply(
        pd.to_numeric, errors="raise"
    )
    if not (numeric.normal_replay_slots + numeric.defect_guard_slots).equals(
        numeric.total_replay_slots
    ):
        raise PreregistrationError("epoch role-slot arithmetic mismatch")
    if not epoch.groupby("arm_id").epoch.count().eq(200).all():
        raise PreregistrationError("every frozen arm must contain all 200 epoch records")
    for ratio_id in ("RHO_0P5_PERCENT", "RHO_1P0_PERCENT", "RHO_2P5_PERCENT"):
        part = epoch[epoch.ratio_id == ratio_id]
        totals = {
            policy: int(group.total_replay_slots.sum())
            for policy, group in part.groupby("policy_id")
        }
        if set(totals) != {"CONTINUOUS", "SAME_PEAK_TAPER", "DOSE_MATCHED_TAPER"}:
            raise PreregistrationError(f"dose policies missing for {ratio_id}")
        if totals["DOSE_MATCHED_TAPER"] != totals["CONTINUOUS"]:
            raise PreregistrationError(f"dose-matched taper total differs for {ratio_id}")
        if totals["SAME_PEAK_TAPER"] * 4 != totals["CONTINUOUS"] * 3:
            raise PreregistrationError(f"same-peak taper dose is not 75 percent for {ratio_id}")

    if not schedule.groupby("arm_id").apply(_coverage_is_exact, include_groups=False).all():
        raise PreregistrationError("every arm must use the common four restart boundaries")
    schedule_numeric = schedule[
        ["normal_replay_slots", "defect_guard_slots", "total_replay_slots"]
    ].apply(pd.to_numeric, errors="raise")
    if not (schedule_numeric.normal_replay_slots + schedule_numeric.defect_guard_slots).equals(
        schedule_numeric.total_replay_slots
    ):
        raise PreregistrationError("segment role-slot arithmetic mismatch")

    if len(jobs) != 296 or jobs.job_id.duplicated().any():
        raise PreregistrationError("physical graph must contain 296 unique jobs")
    job_ids = set(jobs.job_id.astype(str))
    dependencies = {value for value in jobs.dependency_job_id.astype(str) if value}
    if not dependencies <= job_ids:
        raise PreregistrationError("physical job graph contains missing dependencies")
    machine_by_job = jobs.set_index("job_id").machine_id.astype(str).to_dict()
    for row in jobs.itertuples(index=False):
        dependency = str(row.dependency_job_id)
        if dependency and machine_by_job[dependency] != str(row.machine_id):
            raise PreregistrationError("dependent jobs must remain on the same machine")
    used_machines = sorted(set(jobs.machine_id.astype(str)))
    expected_machines = [f"M{index:02d}" for index in range(1, len(used_machines) + 1)]
    if used_machines != expected_machines:
        raise PreregistrationError("machine identifiers must be contiguous")

    known_digests = set(bindings.selection_digest.astype(str))
    if not set(matrix.selection_digest.astype(str)) <= known_digests:
        raise PreregistrationError("matrix references an unbound selection digest")
    treatment = bindings.set_index("selection_id").loc[TREATMENT_SELECTION_ID]
    if int(treatment.row_count) < int(schedule.total_replay_slots.max()):
        raise PreregistrationError("treatment pool is smaller than the largest derived replay peak")

    return {
        "status": "PASS",
        "campaign_id": CAMPAIGN_ID,
        "cycle_count": 4,
        "seed_count": 30,
        "logical_run_count": 80,
        "physical_job_count": 296,
        "machine_count": len(used_machines),
        "canonical_lock_file_sha256": expected_lock,
        "maximum_derived_replay_slots": int(schedule.total_replay_slots.max()),
        "weighted_primary_score_registered": False,
    }


def _preregistration_note(lock_sha: str) -> str:
    return f"""# Dynamic replay budget-efficiency preregistration

## Frozen question

Can the same OOF-ranked normal pool produce a more reliable tail benefit when replay is
conditioned on training stage and cumulative exposure, without changing the canonical
240-run learner?

The estimand is conditional: `V(selection | theta_t, schedule, exposure, seed, context)`.
No universal static `V(x)` and no weighted primary score are registered.

## Canonical training lock

Every formal arm uses the exact 240-run canonical configuration. The immutable lock file
SHA256 is `{lock_sha}`. Replay ratio, replay schedule, selection composition, training seed,
device/data/output paths, and valid resume identity are the only registered variations.
OOM is a failed attempt; it never authorizes silent batch, image-size, optimizer, precision,
augmentation, worker, or schedule changes.

## Seven-day staged releases

1. Cycle 1: high-pressure continuous replay, same-peak taper, and no replay.
2. Cycle 2: dose-matched timing separation plus 0.5% and 1.0% transfer.
3. Cycle 3: weak-defect guard templates; parameters remain unbound until Cycle 2 freezes.
4. Cycle 4: six-arm unseen-seed confirmation; parameters remain unbound until Cycle 3 freezes.

Cycle 1/2 contain 80 frozen logical runs over eight paired discovery seeds. Cycle 1 jobs
begin at `ENGINEERING_GATE`; Cycle 2 remains `HELD`. Cycle 3/4 have no executable jobs yet.
All arms use restart boundaries 140, 150, 160, and 200 so checkpoint/resume mechanics are
not confounded with policy. Continuous and same-peak taper share an exact epoch-140 prefix.

## Interpretation

Primary reporting is safety first: the raw FN=0..95 frontier, `TN_at_FN95`, and
`FN_at_TN68253`, paired by training seed. Process trajectories and gradients are mechanism
evidence, not substitutes for endpoint replication. Blind holdout access is forbidden until
the final policy and analysis are frozen.
"""


def _r2_feasibility_note() -> str:
    return """# Matched-random control feasibility

The real 120,000-sample OOF table shows that a disjoint 2.5% control cannot reproduce the
active Treatment tail's trajectory variability under the originally proposed SMD <= 0.10
contract. Reintroducing Treatment rows would destroy the selection contrast.

Therefore Cycle 1/2 do not contain an R2 arm. After ratio and timing are frozen, Cycle 4
will attempt ratio-specific common-support matching. If the support check fails, the arm
must be renamed a descriptive nearest-support control and cannot support a clean causal
claim. This is a preregistered feasibility result, not a missing implementation.
"""


def _rolling_release_note() -> str:
    return """# Rolling seven-day release protocol

- Local real-data smoke and failure injection precede every fleet release.
- Cycle 1 is released only after canonical-lock, all-epoch telemetry, resume, and artifact
  integrity gates pass on the ten-machine canary.
- Every completed seed block is validated and summarized immediately; daily operations
  reports are generated without changing the frozen within-cycle hypothesis.
- Cycle 2 jobs are prepared but held until the registered Cycle 1 decision artifact exists.
- Cycle 3 and Cycle 4 are templates only. They cannot be queued by an operator override.
- A cycle may finish before seven days when its complete preregistered block is done. It is
  never shortened by dropping unfavorable runs or changing endpoints.
"""


def build_campaign_preregistration(
    output_dir: str | Path,
    *,
    treatment_ranking_source: str | Path,
    canonical_lock_path: str | Path,
    machine_count: int = 10,
    prior_training_seeds: Iterable[int] = DEFAULT_PRIOR_TRAINING_SEEDS,
    built_at: str = "2026-08-07",
) -> dict[str, Any]:
    """Build and atomically publish the staged preregistration package."""

    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise PreregistrationError(f"refusing to overwrite non-empty output: {output}")
    lock = load_canonical_training_lock(canonical_lock_path)
    lock_sha = lock.file_sha256
    if int(lock.immutable_args.get("epochs", -1)) != 200:
        raise PreregistrationError("canonical training lock must specify epochs=200")
    if int(lock.immutable_args.get("batch", -1)) != 128:
        raise PreregistrationError("canonical training lock must specify batch=128")
    if int(lock.immutable_args.get("workers", -1)) != 4:
        raise PreregistrationError("canonical training lock must specify workers=4")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.{os.getpid()}.tmpdir"
    if staging.exists():
        raise PreregistrationError(f"staging path already exists: {staging}")
    try:
        staging.mkdir(parents=True)
        bindings, selection_digests = _prepare_selections(treatment_ranking_source, staging)
        cycles, arms = build_cycle_registry(lock_sha)
        seeds = build_seed_registry(prior_training_seeds=prior_training_seeds)
        active_arms = _active_arms(arms)
        epoch_schedule, schedule = _build_schedules(active_arms)
        timing_seeds = seeds[seeds.seed_scope == "DISCOVERY_TIMING_DOSE"].copy()
        jobs, parents, machines = _build_jobs(
            timing_seeds,
            active_arms,
            schedule,
            machine_count,
            lock_sha,
        )
        matrix = _build_matrix(
            timing_seeds,
            active_arms,
            epoch_schedule,
            selection_digests,
            parents,
            machines,
            lock_sha,
        )
        gates = _build_gated_templates(arms)
        contrasts = _contrasts()
        hypotheses = _hypotheses()
        tables = {
            "cycles": cycles,
            "arms": arms,
            "seeds": seeds,
            "matrix": matrix,
            "epoch_schedule": epoch_schedule,
            "schedule": schedule,
            "jobs": jobs,
            "bindings": bindings,
        }
        validation = validate_preregistration_tables(
            tables,
            canonical_lock_file_sha256=lock_sha,
        )

        for filename, frame in (
            ("CYCLE_REGISTRY.csv", cycles),
            ("ARM_REGISTRY.csv", arms),
            ("SEED_REGISTRY.csv", seeds),
            ("SELECTION_BINDINGS.csv", bindings),
            ("EPOCH_REPLAY_SCHEDULE.csv", epoch_schedule),
            ("REPLAY_SCHEDULE.csv", schedule),
            ("EXPERIMENT_MATRIX.csv", matrix),
            ("PHYSICAL_JOB_GRAPH.csv", jobs),
            ("CONTRASTS_AND_ENDPOINTS.csv", contrasts),
            ("HYPOTHESIS_REGISTRY.csv", hypotheses),
            ("GATED_STAGE_TEMPLATES.csv", gates),
        ):
            _atomic_csv(staging / filename, frame)

        _atomic_json(
            staging / "CANONICAL_TRAINING_LOCK_BINDING.json",
            {
                "schema_version": "1.0",
                "campaign_id": CAMPAIGN_ID,
                "canonical_lock_path": str(Path(canonical_lock_path).resolve()),
                "canonical_lock_file_sha256": lock_sha,
                "canonical_lock_payload_sha256": lock.data["lock_payload_sha256"],
                "immutable_field_count": len(lock.immutable_args),
                "runtime_bound_fields": list(lock.runtime_bound_fields),
                "formal_rule": "FAIL_BEFORE_GPU_ON_ANY_UNREGISTERED_DRIFT",
            },
        )
        _atomic_text(staging / "PREREGISTRATION.md", _preregistration_note(lock_sha))
        _atomic_text(staging / "R2_COMMON_SUPPORT_FEASIBILITY.md", _r2_feasibility_note())
        _atomic_text(staging / "ROLLING_7_DAY_RELEASE_PROTOCOL.md", _rolling_release_note())
        _atomic_json(
            staging / "BLIND_HOLDOUT_STATUS.json",
            {
                "status": "UNBOUND",
                "access": "FORBIDDEN_UNTIL_FINAL_PROTOCOL_FREEZE",
                "claim_gate": "EXTERNAL_GENERALIZATION_BLOCKED",
            },
        )

        artifacts = [
            {
                "relative_path": path.relative_to(staging).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(item for item in staging.rglob("*") if item.is_file())
        ]
        validation.update(
            {
                "built_at": built_at,
                "deadline": "2026-09-10",
                "selection_digests": selection_digests,
                "artifacts": artifacts,
                "artifact_hashes_verified": all(
                    sha256_file(staging / item["relative_path"]) == item["sha256"]
                    for item in artifacts
                ),
            }
        )
        _atomic_json(staging / "PREREGISTRATION_VALIDATION.json", validation)
        if output.exists():
            output.rmdir()
        os.replace(staging, output)
        return {
            "status": "complete",
            "output_dir": str(output),
            "logical_run_count": len(matrix),
            "physical_job_count": len(jobs),
            "canonical_lock_file_sha256": lock_sha,
            "selection_digests": selection_digests,
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


__all__ = [
    "NO_SELECTION_ID",
    "PreregistrationError",
    "TREATMENT_SELECTION_ID",
    "build_campaign_preregistration",
    "validate_preregistration_tables",
]
