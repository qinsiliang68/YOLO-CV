from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .arm_spec import PHASE1_ARM_ORDER
from .errors import ErrorCode, SctsrError
from .serialization import stable_digest


DEPLOYMENT_PLAN_SCHEMA = "stage1.sctsr.deployment_plan.v1"
MACHINE_POLICY = "SEEDED_SHUFFLED_WAVES_12_ACTIVE_PLUS_1_BUFFER"
PHASE_ORDER = (
    "DISCOVERY_PARENT",
    "DISCOVERY_BRANCH",
    "CONFIRMATION_PARENT",
    "CONFIRMATION_BRANCH",
)
ACTIVE_MACHINE_COUNT = 12
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,127}")
_JOB_FIELDS = {
    "job_id",
    "phase",
    "run_role",
    "logical_run_id",
    "arm_id",
    "training_seed",
    "depends_on",
}
_PLACEMENT_FIELDS = _JOB_FIELDS | {"machine_id", "wave"}
_PLAN_FIELDS = {
    "schema_version",
    "state",
    "machine_policy",
    "assignment_seed",
    "active_machine_ids",
    "buffer_machine_id",
    "job_count",
    "placements",
    "deployment_plan_generated",
    "formal_assignments_generated",
    "formal_training_started",
    "release_authorization_required",
    "plan_digest",
}


def _fail(message: str, *, field: str | None = None, observed: Any = None) -> None:
    raise SctsrError(
        ErrorCode.DEPLOYMENT_PLAN_INVALID,
        message,
        failing_field=field,
        observed=observed,
        required_action="Use exactly 12 unique active machine IDs, one separate buffer ID, the frozen 8+14 seeds, and an explicit assignment seed.",
    )


def _validate_seed_groups(
    discovery_seeds: Sequence[int],
    confirmation_seeds: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    discovery = tuple(discovery_seeds)
    confirmation = tuple(confirmation_seeds)
    if len(discovery) != 8 or len(confirmation) != 14:
        _fail(
            "Phase-1 deployment requires exactly 8 discovery and 14 confirmation seeds",
            field="seeds",
            observed={"discovery": len(discovery), "confirmation": len(confirmation)},
        )
    combined = discovery + confirmation
    if any(type(seed) is not int or seed < 0 or seed >= 2**63 for seed in combined):
        _fail("Training seeds must be nonnegative signed 64-bit integers", field="seeds")
    if len(set(combined)) != len(combined):
        _fail("Discovery and confirmation seeds must be globally unique", field="seeds")
    return discovery, confirmation


def build_phase1_logical_jobs(
    *,
    discovery_seeds: Sequence[int],
    confirmation_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """Build the 22 parent and 176 branch identities without starting work."""

    discovery, confirmation = _validate_seed_groups(discovery_seeds, confirmation_seeds)
    jobs: list[dict[str, Any]] = []
    for cohort, seeds in (("DISCOVERY", discovery), ("CONFIRMATION", confirmation)):
        for ordinal, seed in enumerate(seeds, start=1):
            parent_id = f"PARENT_{seed}"
            jobs.append(
                {
                    "job_id": parent_id,
                    "phase": f"{cohort}_PARENT",
                    "run_role": "COMMON_PARENT",
                    "logical_run_id": parent_id,
                    "arm_id": "COMMON_PARENT_NR",
                    "training_seed": seed,
                    "depends_on": [],
                }
            )
            for arm in PHASE1_ARM_ORDER:
                run_id = f"SCTSR_{cohort}_S{ordinal:03d}_{arm.value}"
                jobs.append(
                    {
                        "job_id": run_id,
                        "phase": f"{cohort}_BRANCH",
                        "run_role": "BRANCH",
                        "logical_run_id": run_id,
                        "arm_id": arm.value,
                        "training_seed": seed,
                        "depends_on": [parent_id],
                    }
                )
    _validate_jobs(jobs)
    return jobs


def _validate_machine_ids(
    active_machine_ids: Sequence[str],
    buffer_machine_id: str,
) -> tuple[tuple[str, ...], str]:
    active = tuple(active_machine_ids)
    if len(active) != ACTIVE_MACHINE_COUNT:
        _fail("Deployment requires exactly 12 active machines", field="active_machine_ids", observed=len(active))
    if any(not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None for value in active):
        _fail("Active machine ID is invalid", field="active_machine_ids")
    if len(set(active)) != len(active):
        _fail("Active machine IDs must be unique", field="active_machine_ids")
    if not isinstance(buffer_machine_id, str) or _IDENTIFIER.fullmatch(buffer_machine_id) is None:
        _fail("Buffer machine ID is invalid", field="buffer_machine_id")
    if buffer_machine_id in set(active):
        _fail("Buffer machine must not also be active", field="buffer_machine_id")
    return active, buffer_machine_id


def _validate_jobs(jobs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        if not isinstance(job, Mapping) or set(job) != _JOB_FIELDS:
            _fail("Logical job fields are invalid", field=f"jobs[{index}]")
        row = dict(job)
        if row["phase"] not in PHASE_ORDER:
            _fail("Logical job phase is invalid", field=f"jobs[{index}].phase", observed=row["phase"])
        if row["run_role"] not in {"COMMON_PARENT", "BRANCH"}:
            _fail("Logical job role is invalid", field=f"jobs[{index}].run_role")
        for field in ("job_id", "logical_run_id", "arm_id"):
            if not isinstance(row[field], str) or _IDENTIFIER.fullmatch(row[field]) is None:
                _fail("Logical job identifier is invalid", field=f"jobs[{index}].{field}")
        if row["job_id"] != row["logical_run_id"]:
            _fail("Job ID and logical run ID must be identical", field=f"jobs[{index}].job_id")
        if type(row["training_seed"]) is not int or row["training_seed"] < 0:
            _fail("Logical job training seed is invalid", field=f"jobs[{index}].training_seed")
        if not isinstance(row["depends_on"], list) or any(not isinstance(item, str) for item in row["depends_on"]):
            _fail("Logical job dependency list is invalid", field=f"jobs[{index}].depends_on")
        rows.append(row)
    ids = [row["job_id"] for row in rows]
    if len(ids) != len(set(ids)):
        _fail("Logical job IDs are not unique", field="job_id")
    if len(rows) != 198:
        _fail("Phase-1 logical job count must be 198", field="job_count", observed=len(rows))
    parents = {row["training_seed"]: row for row in rows if row["run_role"] == "COMMON_PARENT"}
    branches = [row for row in rows if row["run_role"] == "BRANCH"]
    if len(parents) != 22 or len(branches) != 176:
        _fail("Phase-1 parent/branch counts are invalid", observed={"parents": len(parents), "branches": len(branches)})
    expected_arms = {arm.value for arm in PHASE1_ARM_ORDER}
    by_seed: dict[int, set[str]] = defaultdict(set)
    for row in branches:
        seed = row["training_seed"]
        parent = parents.get(seed)
        if parent is None or row["depends_on"] != [parent["job_id"]]:
            _fail("Branch does not depend on its own seed's common parent", field=row["job_id"])
        if row["phase"].replace("_BRANCH", "") != parent["phase"].replace("_PARENT", ""):
            _fail("Branch and parent cohorts differ", field=row["job_id"])
        by_seed[seed].add(row["arm_id"])
    if any(arms != expected_arms for arms in by_seed.values()):
        _fail("Every seed must contain exactly the frozen eight arms", field="arm_id")
    return rows


def build_seeded_random_deployment_plan(
    jobs: Sequence[Mapping[str, Any]],
    *,
    active_machine_ids: Sequence[str],
    buffer_machine_id: str,
    assignment_seed: int,
) -> dict[str, Any]:
    """Randomize job-to-machine placement in reproducible, one-job-per-wave slots."""

    rows = _validate_jobs(jobs)
    active, buffer = _validate_machine_ids(active_machine_ids, buffer_machine_id)
    if type(assignment_seed) is not int or assignment_seed < 0 or assignment_seed >= 2**63:
        _fail("Assignment seed must be a nonnegative signed 64-bit integer", field="assignment_seed")
    rng = random.Random(assignment_seed)
    placements: list[dict[str, Any]] = []
    for phase in PHASE_ORDER:
        phase_rows = sorted((row for row in rows if row["phase"] == phase), key=lambda row: row["job_id"])
        rng.shuffle(phase_rows)
        for start in range(0, len(phase_rows), ACTIVE_MACHINE_COUNT):
            wave = start // ACTIVE_MACHINE_COUNT + 1
            machine_order = list(active)
            rng.shuffle(machine_order)
            for offset, row in enumerate(phase_rows[start : start + ACTIVE_MACHINE_COUNT]):
                placements.append({**row, "machine_id": machine_order[offset], "wave": wave})
    core = {
        "schema_version": DEPLOYMENT_PLAN_SCHEMA,
        "state": "PLANNED_NOT_RELEASED",
        "machine_policy": MACHINE_POLICY,
        "assignment_seed": assignment_seed,
        "active_machine_ids": list(active),
        "buffer_machine_id": buffer,
        "job_count": len(placements),
        "placements": placements,
        "deployment_plan_generated": True,
        "formal_assignments_generated": False,
        "formal_training_started": False,
        "release_authorization_required": True,
    }
    plan = {**core, "plan_digest": stable_digest(core)}
    validate_deployment_plan(plan)
    return plan


def validate_deployment_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, Mapping) or set(plan) != _PLAN_FIELDS:
        _fail("Deployment plan fields are invalid")
    core = {key: value for key, value in plan.items() if key != "plan_digest"}
    if (
        plan.get("schema_version") != DEPLOYMENT_PLAN_SCHEMA
        or plan.get("state") != "PLANNED_NOT_RELEASED"
        or plan.get("machine_policy") != MACHINE_POLICY
        or plan.get("plan_digest") != stable_digest(core)
        or plan.get("deployment_plan_generated") is not True
        or plan.get("formal_assignments_generated") is not False
        or plan.get("formal_training_started") is not False
        or plan.get("release_authorization_required") is not True
    ):
        _fail("Deployment plan header, digest, or safety state is invalid")
    active, buffer = _validate_machine_ids(plan["active_machine_ids"], plan["buffer_machine_id"])
    if type(plan.get("assignment_seed")) is not int or not 0 <= plan["assignment_seed"] < 2**63:
        _fail("Deployment assignment seed is invalid", field="assignment_seed")
    placements = plan.get("placements")
    if not isinstance(placements, list) or plan.get("job_count") != len(placements):
        _fail("Deployment placement count is invalid", field="placements")
    jobs: list[dict[str, Any]] = []
    per_wave: dict[tuple[str, int], list[str]] = defaultdict(list)
    for index, placement in enumerate(placements):
        if not isinstance(placement, Mapping) or set(placement) != _PLACEMENT_FIELDS:
            _fail("Placement fields are invalid", field=f"placements[{index}]")
        if placement["machine_id"] not in active or placement["machine_id"] == buffer:
            _fail("Placement uses a non-active or buffer machine", field=f"placements[{index}].machine_id")
        if type(placement["wave"]) is not int or placement["wave"] < 1:
            _fail("Placement wave is invalid", field=f"placements[{index}].wave")
        per_wave[(str(placement["phase"]), int(placement["wave"]))].append(str(placement["machine_id"]))
        jobs.append({field: placement[field] for field in _JOB_FIELDS})
    _validate_jobs(jobs)
    if any(len(values) != len(set(values)) or len(values) > ACTIVE_MACHINE_COUNT for values in per_wave.values()):
        _fail("A machine receives multiple concurrent jobs in the same phase wave", field="wave")
    counts = Counter(row["machine_id"] for row in placements)
    return {
        "status": "PASS",
        "job_count": len(placements),
        "active_machine_count": len(active),
        "buffer_machine_id": buffer,
        "machine_job_counts": dict(sorted(counts.items())),
        "plan_digest": plan["plan_digest"],
    }
