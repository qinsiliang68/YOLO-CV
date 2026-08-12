"""Preregistered logical run matrix for the four v3 cycles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .seeds import SeedRecord


CAPS = (("RHO_0P5", 0.005), ("RHO_1P0", 0.01), ("RHO_2P5", 0.025))
ENVELOPES = ("CONTINUOUS", "LINEAR_TAPER_E141_E160")


@dataclass(frozen=True)
class LogicalRun:
    job_id: str
    cycle_id: str
    release_state: str
    seed_id: str
    seed_scope: str
    training_seed: int
    policy_kind: str
    cap_id: str
    cap_fraction: float | None
    envelope: str
    trace_dependency_job_id: str | None
    endpoint_epoch: int = 200
    key_prediction_epochs: tuple[int, ...] = (120, 140, 150, 160, 180, 200)
    defect_guard_fraction: float = 0.0
    gradient_collection: bool = False


@dataclass(frozen=True)
class MatrixValidation:
    status: str
    cycle_counts: dict[str, int]
    run_count: int


def _job(cycle: str, seed: SeedRecord, suffix: str) -> str:
    return f"{cycle}_{seed.seed_id}_{suffix}"


def build_preregistered_matrix(seed_registry: Sequence[SeedRecord]) -> tuple[LogicalRun, ...]:
    by_scope: dict[str, list[SeedRecord]] = {}
    for seed in seed_registry:
        by_scope.setdefault(seed.seed_scope, []).append(seed)
    rows: list[LogicalRun] = []
    for seed in by_scope.get("DISCOVERY", []):
        treatments: list[LogicalRun] = []
        for cap_id, cap in CAPS:
            for envelope in ENVELOPES:
                suffix = f"T_{cap_id}_{envelope}"
                treatment = LogicalRun(
                    _job("C1", seed, suffix), "CYCLE_1", "HELD", seed.seed_id, seed.seed_scope,
                    seed.training_seed, "OOF_RHO_ADAPTIVE", cap_id, cap, envelope, None,
                )
                treatments.append(treatment)
                rows.append(treatment)
        rows.append(
            LogicalRun(
                _job("C1", seed, "NR"), "CYCLE_1", "HELD", seed.seed_id, seed.seed_scope,
                seed.training_seed, "NO_REPLAY", "NO_REPLAY", None, "NO_REPLAY", None,
            )
        )
        for treatment in treatments:
            rows.append(
                LogicalRun(
                    treatment.job_id.replace("C1_", "C2_").replace("_T_", "_R_"),
                    "CYCLE_2", "HELD", seed.seed_id, seed.seed_scope, seed.training_seed,
                    "TRACE_MATCHED_RANDOM", treatment.cap_id, treatment.cap_fraction, treatment.envelope,
                    treatment.job_id,
                )
            )

    for seed in by_scope.get("RANKING_ABLATION", []):
        primary = _job("C3", seed, "T_RHO_GATE_SELECTED")
        rows.extend(
            [
                LogicalRun(primary, "CYCLE_3", "HELD", seed.seed_id, seed.seed_scope, seed.training_seed,
                           "OOF_RHO_ADAPTIVE", "GATE_SELECTED_CAP", None, "GATE_SELECTED_ENVELOPE", None),
                LogicalRun(_job("C3", seed, "C_CURRENT_LOSS"), "CYCLE_3", "HELD", seed.seed_id,
                           seed.seed_scope, seed.training_seed, "TRACE_MATCHED_CURRENT_LOSS",
                           "GATE_SELECTED_CAP", None, "GATE_SELECTED_ENVELOPE", primary),
                LogicalRun(_job("C3", seed, "R_RANDOM"), "CYCLE_3", "HELD", seed.seed_id,
                           seed.seed_scope, seed.training_seed, "TRACE_MATCHED_RANDOM",
                           "GATE_SELECTED_CAP", None, "GATE_SELECTED_ENVELOPE", primary),
                LogicalRun(_job("C3", seed, "T_RHO_ALTERNATE_ENVELOPE"), "CYCLE_3", "HELD", seed.seed_id,
                           seed.seed_scope, seed.training_seed, "OOF_RHO_ADAPTIVE",
                           "GATE_SELECTED_CAP", None, "ALTERNATE_ENVELOPE", None),
                LogicalRun(_job("C3", seed, "NR"), "CYCLE_3", "HELD", seed.seed_id, seed.seed_scope,
                           seed.training_seed, "NO_REPLAY", "NO_REPLAY", None, "NO_REPLAY", None),
            ]
        )

    for seed in by_scope.get("UNSEEN_CONFIRMATION", []):
        primary = _job("C4", seed, "T_RHO_FROZEN")
        rows.extend(
            [
                LogicalRun(primary, "CYCLE_4", "HELD", seed.seed_id, seed.seed_scope, seed.training_seed,
                           "OOF_RHO_ADAPTIVE", "FROZEN_CAP", None, "FROZEN_ENVELOPE", None),
                LogicalRun(_job("C4", seed, "C_CURRENT_LOSS"), "CYCLE_4", "HELD", seed.seed_id,
                           seed.seed_scope, seed.training_seed, "TRACE_MATCHED_CURRENT_LOSS",
                           "FROZEN_CAP", None, "FROZEN_ENVELOPE", primary),
                LogicalRun(_job("C4", seed, "R_RANDOM"), "CYCLE_4", "HELD", seed.seed_id,
                           seed.seed_scope, seed.training_seed, "TRACE_MATCHED_RANDOM",
                           "FROZEN_CAP", None, "FROZEN_ENVELOPE", primary),
                LogicalRun(_job("C4", seed, "NR"), "CYCLE_4", "HELD", seed.seed_id, seed.seed_scope,
                           seed.training_seed, "NO_REPLAY", "NO_REPLAY", None, "NO_REPLAY", None),
            ]
        )
    return tuple(rows)


def validate_preregistered_matrix(rows: Iterable[LogicalRun]) -> MatrixValidation:
    runs = tuple(rows)
    identifiers = [row.job_id for row in runs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("logical job IDs are not unique")
    expected_counts = {"CYCLE_1": 70, "CYCLE_2": 60, "CYCLE_3": 50, "CYCLE_4": 56}
    observed = {cycle: sum(row.cycle_id == cycle for row in runs) for cycle in expected_counts}
    if observed != expected_counts or len(runs) != 236:
        raise ValueError(f"v3 matrix shape mismatch: {observed}, total={len(runs)}")
    if any(row.release_state != "HELD" for row in runs):
        raise ValueError("all v3 jobs must remain HELD before owner canary")
    if any(row.defect_guard_fraction != 0 or row.gradient_collection for row in runs):
        raise ValueError("v3 formal matrix must not contain guard or gradient arms")
    known = set(identifiers)
    missing_dependencies = sorted(
        {row.trace_dependency_job_id for row in runs if row.trace_dependency_job_id and row.trace_dependency_job_id not in known}
    )
    if missing_dependencies:
        raise ValueError(f"trace dependencies are absent from the matrix: {missing_dependencies}")
    return MatrixValidation("PASS", observed, len(runs))


__all__ = ["CAPS", "ENVELOPES", "LogicalRun", "MatrixValidation", "build_preregistered_matrix", "validate_preregistered_matrix"]
