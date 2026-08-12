from __future__ import annotations

from stage1_dynamic_replay_v3.matrix import build_preregistered_matrix, validate_preregistered_matrix
from stage1_dynamic_replay_v3.seeds import HISTORICAL_AND_V2_RESERVED_SEEDS, build_seed_registry


def test_seed_registry_is_disjoint_and_reproducible() -> None:
    left = build_seed_registry()
    right = build_seed_registry()

    assert left == right
    assert len(left) == 34
    assert not ({row.training_seed for row in left} & HISTORICAL_AND_V2_RESERVED_SEEDS)
    assert len({row.training_seed for row in left}) == 34
    assert all(row.training_seed == (int(row.derivation_sha256[:8], 16) >> 1) for row in left)


def test_matrix_contains_exactly_236_guard_free_runs() -> None:
    rows = build_preregistered_matrix(build_seed_registry())
    report = validate_preregistered_matrix(rows)

    assert report.status == "PASS"
    assert len(rows) == 236
    assert report.cycle_counts == {"CYCLE_1": 70, "CYCLE_2": 60, "CYCLE_3": 50, "CYCLE_4": 56}
    assert all(row.defect_guard_fraction == 0 for row in rows)
    assert all(row.gradient_collection is False for row in rows)
    assert all(row.endpoint_epoch == 200 for row in rows)


def test_trace_matched_jobs_depend_on_rho_treatment() -> None:
    rows = build_preregistered_matrix(build_seed_registry())
    controls = [row for row in rows if row.policy_kind in {"TRACE_MATCHED_RANDOM", "TRACE_MATCHED_CURRENT_LOSS"}]
    assert controls
    assert all(row.trace_dependency_job_id for row in controls)
