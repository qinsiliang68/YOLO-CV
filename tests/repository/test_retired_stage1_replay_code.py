from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


# These modules produced literature, expert-audit, or identity evidence and remain
# reproducible.  Everything else with the old campaign_* prefix was an unrun v2
# participant/runtime implementation and must not remain in the active source tree.
ALLOWED_CAMPAIGN_MODULES = {
    "campaign_canonical_lock.py",
    "campaign_documentation_validation.py",
    "campaign_field_audit.py",
    "campaign_literature.py",
    "campaign_literature_core.py",
    "campaign_literature_pipeline.py",
    "campaign_literature_registry.py",
}

RETIRED_V3_PARTICIPANT_MODULES = {
    "assignment.py",
    "completion.py",
    "controller.py",
    "draw_plan.py",
    "dynamic_dataset.py",
    "epoch_transaction.py",
    "recovery.py",
    "resource_monitor.py",
    "runtime_checkpoint.py",
    "runtime_controller.py",
    "smoke.py",
    "training_runtime.py",
    "trajectory.py",
    "ultralytics_overlay.py",
}

RETIRED_V3_PARTICIPANT_TESTS = {
    "test_assignment_completion.py",
    "test_controller.py",
    "test_draw_plan.py",
    "test_dynamic_dataset.py",
    "test_epoch_transaction.py",
    "test_recovery.py",
    "test_resource_monitor.py",
    "test_runtime_checkpoint.py",
    "test_runtime_controller.py",
    "test_smoke_fixture.py",
    "test_training_runtime.py",
    "test_trajectory.py",
    "test_ultralytics_overlay.py",
}

RETIRED_CAMPAIGN_SCRIPTS = {
    "activate_dynamic_campaign_assignment.py",
    "aggregate_coordination_root_canary.py",
    "aggregate_ten_machine_real_data_canary.py",
    "bind_dynamic_campaign_validation_evidence.py",
    "build_coordination_root_canary_commands.py",
    "build_daily_campaign_status.py",
    "build_dynamic_campaign_assignment.py",
    "build_dynamic_campaign_engineering_gate_v2.py",
    "build_dynamic_campaign_gradient_candidates.py",
    "build_dynamic_campaign_release_gates.py",
    "build_dynamic_campaign_run_queue.py",
    "build_dynamic_replay_preregistration.py",
    "build_epoch_resource_log.py",
    "build_ten_machine_real_data_canary_commands.py",
    "dry_generate_dynamic_campaign_v2.py",
    "dynamic_campaign_train_worker.py",
    "initialize_dynamic_replay_campaign.py",
    "local_dynamic_telemetry_benchmark.py",
    "run_coordination_root_canary.py",
    "run_disk_gpu_preflight.py",
    "run_dynamic_campaign_controller.py",
    "run_local_dynamic_failure_injection.py",
    "summarize_crossed_telemetry_benchmarks.py",
    "validate_all_epoch_telemetry.py",
    "validate_campaign_failure_recovery.py",
    "validate_campaign_leases.py",
    "validate_cycle_closeout.py",
    "validate_dynamic_campaign_contracts.py",
    "validate_dynamic_campaign_report_schema.py",
}

RETIRED_CAMPAIGN_SCHEMAS = {
    "all_epoch_telemetry_validation.schema.json",
    "assignment_reassignment_validation.schema.json",
    "coordination_root_canary_aggregate.schema.json",
    "coordination_root_canary_node.schema.json",
    "cycle_closeout_validation.schema.json",
    "daily_campaign_status.schema.json",
    "disk_gpu_preflight_validation.schema.json",
    "documentation_handoff_validation.schema.json",
    "dynamic_campaign_dry_generation_v2.schema.json",
    "engineering_gate_v2.schema.json",
    "failure_injection_validation.schema.json",
    "lease_concurrency_validation.schema.json",
    "lease_fencing_validation.schema.json",
    "resource_log_validation.schema.json",
    "source_tree_immutability_validation.schema.json",
    "source_tree_manifest.schema.json",
    "standalone_entry_validation.schema.json",
    "ten_machine_real_data_canary_aggregate.schema.json",
    "ten_machine_real_data_canary_node.schema.json",
    "validation_evidence_envelope.schema.json",
}


def _existing_names(directory: Path, names: set[str]) -> set[str]:
    return {name for name in names if (directory / name).exists()}


def test_only_executed_evidence_campaign_modules_remain() -> None:
    campaign_modules = {
        path.name for path in (REPO_ROOT / "stage1_gapvalue240").glob("campaign_*.py")
    }
    assert campaign_modules == ALLOWED_CAMPAIGN_MODULES


def test_v3_training_participant_modules_and_smoke_cli_are_retired() -> None:
    package = REPO_ROOT / "stage1_dynamic_replay_v3"
    assert _existing_names(package, RETIRED_V3_PARTICIPANT_MODULES) == set()
    assert not (REPO_ROOT / "scripts/stage1_dynamic_replay_v3/run_local_smoke.py").exists()


def test_old_participant_tests_are_retired() -> None:
    tests = REPO_ROOT / "tests/stage1_dynamic_replay_v3"
    assert _existing_names(tests, RETIRED_V3_PARTICIPANT_TESTS) == set()


def test_old_campaign_entrypoints_configs_and_schemas_are_retired() -> None:
    scripts = REPO_ROOT / "scripts/stage1_gapvalue240"
    assert _existing_names(scripts, RETIRED_CAMPAIGN_SCRIPTS) == set()
    assert not (
        REPO_ROOT / "configs/stage1_gapvalue240/DYNAMIC_MACHINE_SLOT_MAP_v1.csv"
    ).exists()
    schema_dir = REPO_ROOT / "schemas/stage1_gapvalue240"
    assert _existing_names(schema_dir, RETIRED_CAMPAIGN_SCHEMAS) == set()


def test_v2_runtime_hooks_are_absent_from_preserved_240run_modules() -> None:
    forbidden_markers = {
        "stage1_gapvalue240/hardlink_staging.py": "staged_identity_replay_session",
        "stage1_gapvalue240/machine.py": "coordination_root",
        "stage1_gapvalue240/monitor.py": "RESOURCE_COLUMNS",
        "stage1_gapvalue240/prediction_worker.py": "causal_train_probe",
    }
    for relative_path, marker in forbidden_markers.items():
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert marker not in text, f"retired v2 marker remains in {relative_path}"


def test_evidence_and_contract_tools_are_not_removed_by_code_cleanup() -> None:
    required = {
        "stage1_gapvalue240/campaign_canonical_lock.py",
        "stage1_gapvalue240/campaign_field_audit.py",
        "stage1_gapvalue240/campaign_literature_pipeline.py",
        "stage1_dynamic_replay_v3/expert_delivery_audit.py",
        "stage1_dynamic_replay_v3/global_completion_audit_v2.py",
        "stage1_dynamic_replay_v3/literature_evidence_v2.py",
        "stage1_dynamic_replay_v3/review_claim_ledger.py",
        "stage1_dynamic_replay_v3/tripartite_crosswalk.py",
    }
    missing = {path for path in required if not (REPO_ROOT / path).is_file()}
    assert missing == set()
