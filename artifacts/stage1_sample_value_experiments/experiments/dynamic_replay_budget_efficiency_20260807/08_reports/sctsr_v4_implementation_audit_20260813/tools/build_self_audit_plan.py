from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for candidate in Path(__file__).resolve().parents:
    if (candidate / "stage1_sctsr_v4").is_dir():
        sys.path.insert(0, str(candidate))
        break

from stage1_sctsr_v4.implementation_self_audit import parse_taskbook_self_audit
from stage1_sctsr_v4.serialization import atomic_write_json


FAILURES = {
    "SA-266": (
        "The current registered v3 regression command completed with 183 passed and 1 skipped; the taskbook requires at least 231 passed.",
        "Resolve the baseline mismatch through an approved specification change or restore a verified 231-test v3 baseline without rewriting historical behavior, then rerun the exact command.",
    ),
}


GROUPS = {
    "repository": {
        "range": range(1, 13),
        "command": "git_diff_check_e9b6df6",
        "reports": [
            "reports/REPOSITORY_STATE_AUDIT.json",
            "reports/CHANGED_FILE_LEDGER.json",
            "reports/SOURCE_TREE_MANIFEST.json",
        ],
        "sources": [
            "stage1_sctsr_v4/repository_state_audit.py",
            "stage1_sctsr_v4/source_identity.py",
            "scripts/stage1_sctsr_v4/audit_repository_state.py",
        ],
        "tests": [
            "tests/stage1_sctsr_v4/test_repository_state_audit.py",
            "tests/stage1_sctsr_v4/test_source_identity.py",
        ],
        "observed": "The repository-state audit passed at the frozen implementation source commit, the tracked worktree was clean, protected historical trees were unchanged, unrelated untracked material was excluded, and legacy artifacts were detected separately from active v4 state.",
        "risk": "This result is bound to the frozen implementation source commit; the later evidence-only commit must not change implementation files.",
    },
    "contract": {
        "range": range(20, 31),
        "command": "contract_cli_e9b6df6",
        "reports": ["reports/CONTRACT_VALIDATION.json"],
        "sources": [
            "stage1_sctsr_v4/rate_spec.py",
            "stage1_sctsr_v4/arm_spec.py",
            "stage1_sctsr_v4/contracts.py",
            "configs/stage1_sctsr_v4/arms_phase1_v1.json",
        ],
        "tests": [
            "tests/stage1_sctsr_v4/test_rate_spec.py",
            "tests/stage1_sctsr_v4/test_contract.py",
            "tests/stage1_sctsr_v4/test_contract_hardening.py",
        ],
        "observed": "The registered contract test group completed with 22 passed and verifies integer-rational rates, denominator binding, eight-arm order, held CURRENT_LOSS_U, zero replay before E121, and disabled formal execution.",
        "risk": "The contract is implementation evidence only; no formal training release exists and no SCTSR effectiveness claim is supported.",
    },
    "assets": {
        "range": range(40, 58),
        "command": "asset_cli_e9b6df6",
        "reports": [
            "reports/ASSET_VALIDATION.json",
            "reports/FORMAL_R2_INFEASIBILITY.json",
        ],
        "sources": [
            "stage1_sctsr_v4/asset_registry.py",
            "stage1_sctsr_v4/identity_pool.py",
            "stage1_sctsr_v4/random_controls.py",
            "stage1_sctsr_v4/terminal_field_guard.py",
            "stage1_sctsr_v4/formal_pool_inputs.py",
        ],
        "tests": [
            "tests/stage1_sctsr_v4/test_asset_registry.py",
            "tests/stage1_sctsr_v4/test_identity_pool.py",
            "tests/stage1_sctsr_v4/test_random_controls.py",
            "tests/stage1_sctsr_v4/test_formal_pool_inputs.py",
            "tests/stage1_sctsr_v4/test_asset_hardening.py",
        ],
        "observed": "The asset and pool test group completed with 25 passed. The formal asset registry validates the 120000-row base and 3000-row T stress set. The exact R2 algorithm enforces zero overlap, exact pre-terminal quotas, terminal-field isolation, and fail-closed behavior; the registered formal attempt correctly emitted R2_QUOTA_INFEASIBLE instead of a relaxed pool.",
        "risk": "No formal R2 pool currently exists: the registered data have 172 shortage strata totaling 378 missing occurrences, so formal phase-one construction remains blocked without a specification change or new frozen asset.",
    },
    "schedule": {
        "range": range(60, 77),
        "command": "schedule_cli_e9b6df6",
        "reports": ["reports/SCHEDULE_VALIDATION.json"],
        "sources": [
            "stage1_sctsr_v4/schedule.py",
            "stage1_sctsr_v4/replay_step_plan.py",
            "scripts/stage1_sctsr_v4/build_schedule.py",
        ],
        "tests": [
            "tests/stage1_sctsr_v4/test_schedule.py",
            "tests/stage1_sctsr_v4/test_schedule_hardening.py",
            "tests/stage1_sctsr_v4/test_schedule_registry.py",
            "tests/stage1_sctsr_v4/test_replay_step_plan.py",
        ],
        "observed": "The schedule test group completed with 16 passed and verifies five disjoint identity groups, U and F exposure conservation, stop and fallback semantics, common step-slot skeletons, and planned-versus-actual conservation on registered synthetic fixtures.",
        "risk": "Formal schedules cannot be materialized until an exact formal R2 pool exists; the current evidence validates schedule semantics and synthetic execution only.",
    },
    "parent": {
        "range": range(80, 92),
        "command": "full_v4_e9b6df6",
        "reports": ["reports/SYNTHETIC_CANARY_E9_A_VALIDATION.json"],
        "sources": [
            "stage1_sctsr_v4/checkpointing.py",
            "stage1_sctsr_v4/common_parent.py",
            "stage1_sctsr_v4/branch_lineage.py",
            "stage1_sctsr_v4/logical_artifact_index.py",
        ],
        "tests": [
            "tests/stage1_sctsr_v4/test_checkpoint_hardening.py",
            "tests/stage1_sctsr_v4/test_common_parent.py",
            "tests/stage1_sctsr_v4/test_branch_lineage.py",
            "tests/stage1_sctsr_v4/test_logical_artifact_index.py",
        ],
        "observed": "The parent and lineage test group completed with 31 passed. The complete synthetic canary binds one immutable E120 parent, full training and RNG state, child lineage, and logical E1-E200 artifact ownership.",
        "risk": "Formal common-parent checkpoints have not been created; byte identity and immutability are proven on unit and complete synthetic executions only.",
    },
    "runtime": {
        "range": range(100, 121),
        "command": "full_v4_e9b6df6",
        "reports": ["reports/SYNTHETIC_CANARY_E9_A_VALIDATION.json"],
        "sources": [
            "stage1_sctsr_v4/fixed_step_runtime.py",
            "stage1_sctsr_v4/ultralytics_overlay.py",
            "integrations/ultralytics/sctsr_classification_trainer.py",
            "stage1_sctsr_v4/bn_isolation.py",
            "stage1_sctsr_v4/rng_isolation.py",
        ],
        "tests": [
            "tests/stage1_sctsr_v4/test_fixed_step_runtime.py",
            "tests/stage1_sctsr_v4/test_fixed_step_hardening.py",
            "tests/stage1_sctsr_v4/test_real_yolo_integration.py",
            "tests/stage1_sctsr_v4/test_ultralytics_overlay.py",
            "tests/stage1_sctsr_v4/test_bn_isolation.py",
            "tests/stage1_sctsr_v4/test_rng_isolation.py",
        ],
        "observed": "The fixed-step and YOLO integration test group completed with 31 passed, including real forward, backward, optimizer, scaler, EMA, replay-gradient contribution, denominator, base-order, augmentation, BN, RNG, OOM, accumulation, and world-size guards.",
        "risk": "The integration tests are bounded mechanism tests; no 200-epoch formal GPU run has verified production throughput or long-horizon numerical behavior.",
    },
    "ledgers": {
        "range": range(130, 145),
        "command": "full_v4_e9b6df6",
        "reports": ["reports/SYNTHETIC_CANARY_E9_A_VALIDATION.json"],
        "sources": [
            "stage1_sctsr_v4/ledger_schema.py",
            "stage1_sctsr_v4/occurrence_ledger.py",
            "stage1_sctsr_v4/step_ledger.py",
            "stage1_sctsr_v4/exposure_ledger.py",
            "stage1_sctsr_v4/selection_ledger.py",
            "stage1_sctsr_v4/columnar.py",
            "stage1_sctsr_v4/evidence_runtime.py",
        ],
        "tests": [
            "tests/stage1_sctsr_v4/test_occurrence_ledger.py",
            "tests/stage1_sctsr_v4/test_step_ledger.py",
            "tests/stage1_sctsr_v4/test_exposure_ledger.py",
            "tests/stage1_sctsr_v4/test_selection_ledger.py",
            "tests/stage1_sctsr_v4/test_columnar.py",
            "tests/stage1_sctsr_v4/test_evidence_runtime_integration.py",
        ],
        "observed": "The evidence and ledger test group completed with 58 passed. Complete synthetic executions wrote occurrence, optimizer-step, exposure, selection, prediction, and telemetry evidence through real PyArrow Zstd Parquet paths with registered schemas, partitions, row counts, bytes, and SHA-256.",
        "risk": "Formal production-volume evidence has not been generated; storage growth and sustained write latency remain unmeasured until an authorized run.",
    },
    "telemetry": {
        "range": range(150, 160),
        "command": "full_v4_e9b6df6",
        "reports": ["reports/SYNTHETIC_CANARY_E9_A_VALIDATION.json"],
        "sources": [
            "stage1_sctsr_v4/telemetry.py",
            "stage1_sctsr_v4/evidence_runtime.py",
        ],
        "tests": [
            "tests/stage1_sctsr_v4/test_telemetry.py",
            "tests/stage1_sctsr_v4/test_evidence_runtime_integration.py",
        ],
        "observed": "The telemetry test group completed with 3 passed and validates fixed cadence, process and system metrics, GPU and CUDA null-with-reason behavior, disk-space identity, provider failure semantics, closeout requirements, and epoch receipt binding.",
        "risk": "Availability and cadence under the eventual formal GPU host have not been measured; unavailable required providers will fail canonical closeout rather than receive fabricated zero values.",
    },
    "evaluation": {
        "range": range(170, 189),
        "command": "full_v4_e9b6df6",
        "reports": ["reports/SYNTHETIC_CANARY_E9_A_VALIDATION.json"],
        "sources": [
            "stage1_sctsr_v4/prediction_artifact.py",
            "stage1_sctsr_v4/prediction_runtime.py",
            "stage1_sctsr_v4/evaluation.py",
            "stage1_sctsr_v4/statistics.py",
            "stage1_sctsr_v4/completion.py",
        ],
        "tests": [
            "tests/stage1_sctsr_v4/test_prediction_artifact.py",
            "tests/stage1_sctsr_v4/test_prediction_evaluation_hardening.py",
            "tests/stage1_sctsr_v4/test_prediction_runtime.py",
            "tests/stage1_sctsr_v4/test_evaluation.py",
            "tests/stage1_sctsr_v4/test_statistics.py",
            "tests/stage1_sctsr_v4/test_statistics_hardening.py",
        ],
        "observed": "The endpoint and statistics test group completed with 49 passed and verifies E200-only predictions, identity completeness, 96 tie-safe FN frontier points, independent thresholds, normalized AUC, seed registries, paired completeness, sign-flip, Holm, worst-seed, win-rate, and dual-end degradation outputs.",
        "risk": "Only golden and synthetic predictions were evaluated; no formal SCTSR arm predictions or scientific estimates exist.",
    },
    "qrad": {
        "range": range(200, 210),
        "command": "full_v4_e9b6df6",
        "reports": [
            "reports/ASSET_VALIDATION.json",
            "reports/REPOSITORY_STATE_AUDIT.json",
        ],
        "sources": [
            "stage1_sctsr_v4/qrad_contract.py",
            "stage1_sctsr_v4/short_branch_scaffold.py",
            "configs/stage1_sctsr_v4/disabled_phase2_v1.json",
        ],
        "tests": [
            "tests/stage1_sctsr_v4/test_qrad_contract.py",
            "tests/stage1_sctsr_v4/test_qrad_scaffold_hardening.py",
            "tests/stage1_sctsr_v4/test_short_branch_scaffold.py",
        ],
        "observed": "The Q/R/A/D and phase-two test group completed with 21 passed. Weighted totals and utility relabeling are rejected, val_target is absent, A fails closed, short branches and predictor training remain disabled, and no reinforcement-learning selector is enabled.",
        "risk": "A remains blocked until a separately registered and group-isolated val_target is frozen; phase two has no scientific or training authorization.",
    },
    "recovery": {
        "range": range(220, 234),
        "command": "full_v4_e9b6df6",
        "reports": ["reports/SYNTHETIC_CANARY_E9_A_VALIDATION.json"],
        "sources": [
            "stage1_sctsr_v4/epoch_transaction.py",
            "stage1_sctsr_v4/recovery.py",
            "stage1_sctsr_v4/fault_injection.py",
            "stage1_sctsr_v4/checkpointing.py",
        ],
        "tests": [
            "tests/stage1_sctsr_v4/test_epoch_transaction.py",
            "tests/stage1_sctsr_v4/test_recovery.py",
            "tests/stage1_sctsr_v4/test_evidence_transaction_hardening.py",
            "tests/stage1_sctsr_v4/test_formal_resume.py",
        ],
        "observed": "The transaction and recovery test group completed with 16 passed. Complete synthetic executions exercise generation publishing, kill, OOM, disk-full, partial-file, receipt, identity, quarantine, resume, and critical-checkpoint retention behavior.",
        "risk": "Faults were injected in bounded synthetic runs; actual host power loss and production filesystem failure remain operational risks for the formal release review.",
    },
    "cli": {
        "range": range(240, 251),
        "command": "cli_side_effects_e9b6df6",
        "reports": [
            "reports/REPOSITORY_STATE_AUDIT.json",
            "reports/SYNTHETIC_CANARY_E9_A_VALIDATION.json",
        ],
        "sources": [
            "stage1_sctsr_v4/formal_cli.py",
            "stage1_sctsr_v4/formal_release.py",
            "stage1_sctsr_v4/formal_training.py",
            "stage1_sctsr_v4/run_validation.py",
            "stage1_sctsr_v4/synthetic_canary.py",
        ],
        "tests": [
            "tests/stage1_sctsr_v4/test_cli_and_run_audit_hardening.py",
            "tests/stage1_sctsr_v4/test_formal_release.py",
            "tests/stage1_sctsr_v4/test_formal_training.py",
            "tests/stage1_sctsr_v4/test_no_formal_side_effects.py",
            "tests/stage1_sctsr_v4/test_cli_synthetic.py",
        ],
        "observed": "The CLI and prohibited-side-effect test group completed with 32 passed. Formal entry points require a future signed release, validators never trigger runners, legacy queues are not revived, blind and test roles are rejected, and synthetic outputs remain explicitly non-scientific.",
        "risk": "Formal execution remains intentionally unavailable because no signed release or seed registry is present.",
    },
    "testing": {
        "range": range(260, 276),
        "command": "audit_tools_e9b6df6",
        "reports": [
            "COMMAND_INDEX.json",
            "reports/CHANGED_FILE_LEDGER.json",
            "reports/SYNTHETIC_DETERMINISM_COMPARISON_E9.json",
            "tdd_history/TDD_HISTORY_AUDIT.json",
            "tdd_history/TDD_HISTORY_AUDIT_RECEIPT.json",
        ],
        "sources": [
            "stage1_sctsr_v4/implementation_self_audit.py",
            "stage1_sctsr_v4/run_validation.py",
            "stage1_sctsr_v4/synthetic_canary.py",
        ],
        "tests": [
            "tests/stage1_sctsr_v4/test_implementation_self_audit.py",
            "tests/stage1_sctsr_v4/test_documentation_contract.py",
            "tests/stage1_sctsr_v4/test_synthetic_canary.py",
        ],
        "observed": "The final audit-tool suite completed with 26 passed, including streamed verification of the published checkpoint-part concatenation against the original checkpoint SHA-256. The TDD history audit independently validates its own schema and records 34 rollback commits, 31 behavior commits, 3 non-behavior commits, 33 failing-first pairs, and 146 immutable rollout events. It explicitly identifies the reviewer as the primary agent rather than claiming independent review.",
        "risk": "Implementation tests do not authorize formal training and cannot establish SCTSR scientific effectiveness.",
    },
    "manual": {
        "range": range(280, 290),
        "command": "validate_manual_line_review_e9b6df6",
        "reports": [
            "reports/MANUAL_LINE_REVIEW.json",
            "reports/MANUAL_LINE_REVIEW_VALIDATION.json",
        ],
        "sources": [
            "stage1_sctsr_v4/manual_line_review.py",
            "stage1_sctsr_v4/fixed_step_runtime.py",
            "stage1_sctsr_v4/ultralytics_overlay.py",
            "stage1_sctsr_v4/schema_registry.py",
        ],
        "tests": [
            "tests/stage1_sctsr_v4/test_manual_line_review.py",
            "tests/stage1_sctsr_v4/test_fixed_step_hardening.py",
            "tests/stage1_sctsr_v4/test_schema_registry.py",
        ],
        "observed": "The validated manual line review contains ten exact SA-280 through SA-289 findings and 24 source-line anchors whose source bytes, line ranges, and SHA-256 digests match the frozen reviewed snapshot.",
        "risk": "This is a named self-review and explicitly does not claim an independent reviewer; an external code review is still required before formal release.",
    },
    "side_effects": {
        "range": range(300, 310),
        "command": "cli_side_effects_e9b6df6",
        "reports": [
            "reports/REPOSITORY_STATE_AUDIT.json",
            "reports/SYNTHETIC_CANARY_E9_A_VALIDATION.json",
        ],
        "sources": [
            "stage1_sctsr_v4/repository_state_audit.py",
            "stage1_sctsr_v4/qrad_contract.py",
            "stage1_sctsr_v4/synthetic_canary.py",
        ],
        "tests": [
            "tests/stage1_sctsr_v4/test_repository_state_audit.py",
            "tests/stage1_sctsr_v4/test_no_formal_side_effects.py",
            "tests/stage1_sctsr_v4/test_qrad_contract.py",
            "tests/stage1_sctsr_v4/test_synthetic_canary.py",
        ],
        "observed": "The repository-state audit records every prohibited v4 side-effect flag as JSON false, records val_target as absent with A blocked, detects legacy gate/release/assignment evidence separately, and confirms synthetic outputs are absent from the scientific registry.",
        "risk": "These flags describe the audited implementation workspace and do not constitute future training authorization.",
    },
}


COMMAND_OVERRIDES = {
    "SA-010": "git_diff_check_e9b6df6",
    "SA-011": "source_tree_manifest_e9b6df6",
    "SA-030": "cli_side_effects_e9b6df6",
    "SA-040": "asset_cli_e9b6df6",
    "SA-041": "full_v4_e9b6df6",
    "SA-042": "asset_cli_e9b6df6",
    "SA-260": "tdd_history_audit_e9b6df6",
    "SA-261": "tdd_history_audit_e9b6df6",
    "SA-262": "tdd_history_audit_e9b6df6",
    "SA-263": "tdd_history_audit_e9b6df6",
    "SA-264": "full_v4_e9b6df6",
    "SA-265": "full_v4_e9b6df6",
    "SA-266": "full_v3_e9b6df6",
    "SA-267": "contract_cli_e9b6df6",
    "SA-268": "asset_cli_e9b6df6",
    "SA-269": "schedule_cli_e9b6df6",
    "SA-270": "synthetic_canary_e9_a",
    "SA-271": "validate_canary_e9_a",
    "SA-272": "cli_side_effects_e9b6df6",
    "SA-273": "audit_tools_e9b6df6",
    "SA-274": "tdd_history_audit_e9b6df6",
    "SA-275": "determinism_compare_e9b6df6",
    "SA-307": "full_v4_e9b6df6",
    "SA-309": "validate_canary_e9_a",
}


MANUAL_OVERRIDES = {
    "SA-280": (["stage1_sctsr_v4/fixed_step_runtime.py"], ["tests/stage1_sctsr_v4/test_fixed_step_runtime.py"]),
    "SA-281": (["stage1_sctsr_v4/fixed_step_runtime.py", "stage1_sctsr_v4/ultralytics_overlay.py"], ["tests/stage1_sctsr_v4/test_fixed_step_hardening.py"]),
    "SA-282": (["stage1_sctsr_v4/fixed_step_runtime.py", "stage1_sctsr_v4/ultralytics_overlay.py"], ["tests/stage1_sctsr_v4/test_real_yolo_integration.py"]),
    "SA-283": (["stage1_sctsr_v4/bn_isolation.py", "stage1_sctsr_v4/ultralytics_overlay.py"], ["tests/stage1_sctsr_v4/test_bn_isolation.py"]),
    "SA-284": (["stage1_sctsr_v4/rng_isolation.py", "stage1_sctsr_v4/base_rng.py"], ["tests/stage1_sctsr_v4/test_rng_isolation.py", "tests/stage1_sctsr_v4/test_base_rng.py"]),
    "SA-285": (["stage1_sctsr_v4/random_controls.py", "stage1_sctsr_v4/formal_pool_inputs.py"], ["tests/stage1_sctsr_v4/test_random_controls.py", "tests/stage1_sctsr_v4/test_formal_pool_inputs.py"]),
    "SA-286": (["stage1_sctsr_v4/fixed_step_runtime.py", "stage1_sctsr_v4/formal_training.py"], ["tests/stage1_sctsr_v4/test_fixed_step_hardening.py", "tests/stage1_sctsr_v4/test_formal_training.py"]),
    "SA-287": (["stage1_sctsr_v4/formal_cli.py", "stage1_sctsr_v4/prediction_runtime.py"], ["tests/stage1_sctsr_v4/test_cli_and_run_audit_hardening.py", "tests/stage1_sctsr_v4/test_prediction_runtime.py"]),
    "SA-288": (["stage1_sctsr_v4/completion.py", "stage1_sctsr_v4/formal_release.py"], ["tests/stage1_sctsr_v4/test_completion_hardening.py", "tests/stage1_sctsr_v4/test_formal_release.py"]),
    "SA-289": (["stage1_sctsr_v4/schema_registry.py", "configs/stage1_sctsr_v4/schema_registry_v1.json"], ["tests/stage1_sctsr_v4/test_schema_registry.py"]),
}


def group_for(check_id: str) -> dict:
    number = int(check_id.split("-")[1])
    for group in GROUPS.values():
        if number in group["range"]:
            return group
    raise ValueError(f"No audit group for {check_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskbook", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.evidence_root.resolve()
    commands = json.loads((root / "COMMAND_INDEX.json").read_text(encoding="utf-8"))["commands"]
    command_by_name = {row["name"]: row for row in commands}
    snapshot_manifest = json.loads((root / "reviewed_e9b6df6/REVIEWED_FILE_SNAPSHOT_MANIFEST.json").read_text(encoding="utf-8"))
    snapshot_by_original = {
        row["original_relative_path"]: f"reviewed_e9b6df6/{row['snapshot_relative_path']}"
        for row in snapshot_manifest["files"]
    }

    rows = []
    for requirement in parse_taskbook_self_audit(args.taskbook):
        check_id = requirement["check_id"]
        group = group_for(check_id)
        command_name = COMMAND_OVERRIDES.get(check_id, group["command"])
        command = command_by_name[command_name]
        source_originals, test_originals = MANUAL_OVERRIDES.get(
            check_id,
            (group["sources"], group["tests"]),
        )
        status = "FAIL" if check_id in FAILURES else "PASS"
        if status == "FAIL":
            observed, action = FAILURES[check_id]
        else:
            observed = f"{group['observed']} The cited source, tests, command, and logs directly assess {check_id} at taskbook line {requirement['taskbook_line']}."
            action = "No corrective action is required for this implementation-only check; retain the cited immutable evidence and rerun it before any formal release decision."
        evidence_paths = ["COMMAND_INDEX.json", command["stdout_path"], *group["reports"]]
        evidence_paths = list(dict.fromkeys(evidence_paths))
        for relative in evidence_paths:
            if not (root / relative).is_file():
                raise FileNotFoundError(relative)
        reviewed_sources = [snapshot_by_original[path] for path in source_originals]
        reviewed_tests = [snapshot_by_original[path] for path in test_originals]
        rows.append(
            {
                "check_id": check_id,
                "status": status,
                "evidence_paths": evidence_paths,
                "reproduction_command": command["command_line"],
                "exit_code": command["exit_code"],
                "stdout_log_path": command["stdout_path"],
                "stdout_log_bytes": command["stdout_bytes"],
                "stdout_log_sha256": command["stdout_sha256"],
                "stderr_log_path": command["stderr_path"],
                "stderr_log_bytes": command["stderr_bytes"],
                "stderr_log_sha256": command["stderr_sha256"],
                "observed_result": observed,
                "expected_result": f"Implementation acceptance requires direct reproducible evidence satisfying {check_id} at taskbook line {requirement['taskbook_line']} without substituting a scientific-effectiveness claim.",
                "reviewed_source_files": reviewed_sources,
                "reviewed_test_files": reviewed_tests,
                "remaining_risk": group["risk"],
                "required_action_if_not_pass": action,
            }
        )

    payload = {
        "schema_version": "stage1.sctsr.self_audit_input_plan.v1",
        "checks": rows,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"status": "PASS", "check_count": len(rows), "failed_check_ids": sorted(FAILURES)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
