from __future__ import annotations

import json
import subprocess
import sys

import pytest

from stage1_sctsr_v4.arm_spec import ArmId
from stage1_sctsr_v4.deployment_plan import (
    build_phase1_logical_jobs,
    build_seeded_random_deployment_plan,
    validate_deployment_plan,
)
from stage1_sctsr_v4.errors import ErrorCode, SctsrError


DISCOVERY = tuple(range(101, 109))
CONFIRMATION = tuple(range(201, 215))
ACTIVE = tuple(f"RTX3090_{index:02d}" for index in range(1, 13))
BUFFER = "RTX3090_13_BUFFER"


def _plan(seed: int = 20260815):
    jobs = build_phase1_logical_jobs(
        discovery_seeds=DISCOVERY,
        confirmation_seeds=CONFIRMATION,
    )
    return build_seeded_random_deployment_plan(
        jobs,
        active_machine_ids=ACTIVE,
        buffer_machine_id=BUFFER,
        assignment_seed=seed,
    )


def test_phase1_job_inventory_is_exactly_22_parents_plus_176_branches():
    jobs = build_phase1_logical_jobs(
        discovery_seeds=DISCOVERY,
        confirmation_seeds=CONFIRMATION,
    )

    assert len(jobs) == 198
    assert sum(row["run_role"] == "COMMON_PARENT" for row in jobs) == 22
    assert sum(row["run_role"] == "BRANCH" for row in jobs) == 176
    assert {row["arm_id"] for row in jobs if row["run_role"] == "BRANCH"} == {
        arm.value for arm in ArmId
    }
    assert all(
        row["depends_on"] == [f"PARENT_{row['training_seed']}"]
        for row in jobs
        if row["run_role"] == "BRANCH"
    )


def test_seeded_random_plan_uses_12_active_machines_and_never_uses_buffer():
    plan = _plan()
    checked = validate_deployment_plan(plan)

    assert checked["status"] == "PASS"
    assert plan["active_machine_ids"] == list(ACTIVE)
    assert plan["buffer_machine_id"] == BUFFER
    assert {row["machine_id"] for row in plan["placements"]} <= set(ACTIVE)
    assert BUFFER not in {row["machine_id"] for row in plan["placements"]}
    assert plan["formal_training_started"] is False
    assert plan["release_authorization_required"] is True


def test_seeded_random_plan_is_reproducible_but_changes_with_assignment_seed():
    first = _plan(11)
    second = _plan(11)
    third = _plan(12)

    assert first == second
    assert first["plan_digest"] == second["plan_digest"]
    assert [row["machine_id"] for row in first["placements"]] != [
        row["machine_id"] for row in third["placements"]
    ]


def test_each_phase_wave_assigns_at_most_one_job_per_machine_and_is_balanced():
    plan = _plan()
    phase_waves: dict[tuple[str, int], list[dict]] = {}
    for row in plan["placements"]:
        phase_waves.setdefault((row["phase"], row["wave"]), []).append(row)

    for rows in phase_waves.values():
        machine_ids = [row["machine_id"] for row in rows]
        assert len(machine_ids) == len(set(machine_ids))
        assert len(machine_ids) <= 12

    discovery_parent = [
        row for row in plan["placements"] if row["phase"] == "DISCOVERY_PARENT"
    ]
    assert len(discovery_parent) == 8
    assert len({row["machine_id"] for row in discovery_parent}) == 8


@pytest.mark.parametrize(
    ("active", "buffer"),
    [
        (ACTIVE[:-1], BUFFER),
        (ACTIVE + (ACTIVE[0],), BUFFER),
        (ACTIVE, ACTIVE[0]),
    ],
)
def test_machine_inventory_must_be_exactly_12_unique_active_plus_one_buffer(active, buffer):
    jobs = build_phase1_logical_jobs(
        discovery_seeds=DISCOVERY,
        confirmation_seeds=CONFIRMATION,
    )
    with pytest.raises(SctsrError) as caught:
        build_seeded_random_deployment_plan(
            jobs,
            active_machine_ids=active,
            buffer_machine_id=buffer,
            assignment_seed=7,
        )
    assert caught.value.code is ErrorCode.DEPLOYMENT_PLAN_INVALID


def test_plan_digest_or_duplicate_job_tampering_is_rejected():
    plan = _plan()
    plan["placements"][1]["job_id"] = plan["placements"][0]["job_id"]

    with pytest.raises(SctsrError) as caught:
        validate_deployment_plan(plan)
    assert caught.value.code is ErrorCode.DEPLOYMENT_PLAN_INVALID


def test_cli_builds_inactive_random_plan_from_frozen_seed_values(repository_root, tmp_path):
    seed_registry = tmp_path / "SEED_REGISTRY.json"
    seed_registry.write_text(
        json.dumps(
            {
                "schema_version": "stage1.sctsr.seed_registry.v1",
                "state": "FROZEN_BY_RELEASE_AUTHORITY",
                "historical_training_seeds": [],
                "selection_seeds": [],
                "discovery_seeds": list(DISCOVERY),
                "confirmation_seeds": list(CONFIRMATION),
                "required_counts_when_formal": {"discovery": 8, "confirmation": 14},
            }
        ),
        encoding="utf-8",
    )
    plan_output = tmp_path / "DEPLOYMENT_PLAN.json"
    receipt = tmp_path / "receipt.json"
    command = [
        sys.executable,
        str(repository_root / "scripts/stage1_sctsr_v4/build_deployment_plan.py"),
        "--seed-registry",
        str(seed_registry),
        "--buffer-machine",
        BUFFER,
        "--assignment-seed",
        "20260815",
        "--plan-output",
        str(plan_output),
        "--output",
        str(receipt),
    ]
    for machine in ACTIVE:
        command.extend(("--active-machine", machine))

    completed = subprocess.run(command, cwd=repository_root, text=True, capture_output=True)

    assert completed.returncode == 0, completed.stderr
    plan = json.loads(plan_output.read_text(encoding="utf-8"))
    assert validate_deployment_plan(plan)["job_count"] == 198
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "PASS"
