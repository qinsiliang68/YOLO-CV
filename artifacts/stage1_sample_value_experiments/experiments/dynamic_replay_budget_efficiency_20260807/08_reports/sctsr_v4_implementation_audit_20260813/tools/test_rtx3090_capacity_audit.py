from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_rtx3090_capacity_audit.py")
REPOSITORY_ROOT = next(parent for parent in MODULE_PATH.parents if (parent / ".git").exists())


def _load_module():
    spec = importlib.util.spec_from_file_location("build_rtx3090_capacity_audit", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dependency_aware_phase_schedule_is_exact_and_capacity_optimal():
    module = _load_module()

    discovery = module.simulate_phase(seed_count=8, machine_count=10)
    confirmation = module.simulate_phase(seed_count=14, machine_count=10)

    assert discovery["total_work_epoch_units"] == 6080
    assert discovery["makespan_epoch_units"] == 680
    assert confirmation["total_work_epoch_units"] == 10640
    assert confirmation["makespan_epoch_units"] == 1080
    assert confirmation["granularity_lower_bound_epoch_units"] == 1080


def test_exact_sctsr_workload_includes_replay_calls_and_occurrence_rows():
    module = _load_module()

    workload = module.sctsr_workload(seed_count=22)

    assert workload["base_steps_per_seed"] == 712_880
    assert workload["base_occurrence_rows_per_seed"] == 91_200_000
    assert workload["replay_microbatch_calls_per_seed"] == 291_040
    assert workload["replay_occurrences_per_seed"] == 312_000
    assert workload["aggregate_base_occurrence_rows"] == 2_006_400_000
    assert workload["aggregate_replay_microbatch_calls"] == 6_402_880
    assert workload["aggregate_replay_occurrences"] == 6_864_000


def test_historical_inventory_is_40_same_gpu_full_runs():
    module = _load_module()

    rows = module.load_historical_runs(REPOSITORY_ROOT)

    assert len(rows) == 40
    assert {row["gpu_name"] for row in rows} == {"NVIDIA GeForce RTX 3090"}
    assert {row["model_key"] for row in rows} == {"l"}
    assert {row["epochs"] for row in rows} == {200}
    assert {row["batch"] for row in rows} == {128}
    assert {row["imgsz"] for row in rows} == {224}
    assert {row["status"] for row in rows} == {"ok"}
    closest = [row for row in rows if row["train_image_count"] == 120_600]
    assert len(closest) == 2


def test_report_separates_measured_baseline_from_unmeasured_v4_overhead():
    module = _load_module()

    report = module.build_report(REPOSITORY_ROOT)

    assert report["status"] == "PASS_WITH_EXPLICIT_RUNTIME_UNCERTAINTY"
    assert report["hardware_contract"]["machine_count"] == 10
    assert report["hardware_contract"]["gpu_per_machine"] == 1
    assert report["historical_evidence"]["run_count"] == 40
    assert report["estimate"]["base_only_dependency_aware"]["closest_120600_mean"]["hours"] > 120
    assert report["estimate"]["base_only_dependency_aware"]["closest_120600_mean"]["hours"] < 130
    assert report["estimate"]["recommended_capacity_reservation"]["calendar_days"] == 8
    assert report["estimate"]["unmeasured_overhead"]["status"] == "NOT_DIRECTLY_MEASURED_ON_RTX3090"
    assert report["estimate"]["unmeasured_overhead"]["included_in_recommended_reserve_percent"] == 20
