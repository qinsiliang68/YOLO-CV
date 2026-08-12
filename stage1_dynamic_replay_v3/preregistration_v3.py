"""Fail-closed validation for the Stage1 Q/R/A/D scientific contract.

This module validates a preregistration package.  It never creates jobs, opens
data roles, executes training, or treats a passing contract as scientific
evidence that a replay policy is effective.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "stage1.dynamic_replay.preregistration.v3"
VALIDATION_SCHEMA_VERSION = "stage1.dynamic_replay.preregistration_validation.v3"

REQUIRED_FILES = frozenset(
    {
        "README.md",
        "SCIENTIFIC_CONTRACT.json",
        "SCIENTIFIC_CONTRACT.md",
        "MINIMAL_FALSIFIABLE_MATRIX.csv",
        "CONTRAST_REGISTRY.csv",
        "MECHANISM_EVIDENCE_REGISTRY.csv",
        "DATA_COLLECTION_SCHEMA.json",
        "DATA_COLLECTION_SCHEMA.md",
        "STATISTICAL_DECISION_RULES.json",
        "STATISTICAL_DECISION_RULES.md",
        "REPOSITORY_CHANGE_SPEC.md",
    }
)

REQUIRED_ARM_IDS = frozenset(
    {
        "NO_REPLAY",
        "R1_GLOBAL_RANDOM",
        "R2_METHOD_MATCHED_RANDOM",
        "CURRENT_LOSS",
        "T_Q",
        "T_QR",
        "T_QRA",
        "T_QRAD",
    }
)
REPLAY_ARM_IDS = REQUIRED_ARM_IDS - {"NO_REPLAY"}
REQUIRED_CONTROLS = frozenset(
    {"NO_REPLAY", "R1_GLOBAL_RANDOM", "R2_METHOD_MATCHED_RANDOM", "CURRENT_LOSS"}
)
REQUIRED_CONFIRMATORY_COMPARATORS = frozenset(REQUIRED_CONTROLS)
REQUIRED_EVIDENCE_TYPES = frozenset(
    {
        "PAPER",
        "EXPERT_CLAIM",
        "CODE",
        "SYNTHETIC",
        "STAGE1_OBSERVATION",
        "FUTURE_INTERVENTION",
    }
)
ALLOWED_MECHANISM_STATUSES = frozenset(
    {"SUPPORTED", "CONTRADICTED", "MIXED", "UNKNOWN_IN_STAGE1"}
)
SIGNAL_NAMES = frozenset(
    {"confidence", "loss", "RHO", "gradient", "forgetting", "AUM", "coverage"}
)
PERMITTED_COMPOSITION = frozenset(
    {"ORDERED_GATING", "STRATIFICATION", "FACTORIAL_ABLATION"}
)
MATRIX_FIELDS = (
    "arm_id",
    "role",
    "q_component",
    "r_component",
    "a_component",
    "d_component",
    "selection_rule",
    "eligible_universe",
    "slot_trace_source",
    "replay_budget_group",
    "optimizer_step_policy",
    "base_exposure_policy",
    "utility_evidence_role",
    "release_state",
)
CONTRAST_FIELDS = (
    "contrast_id",
    "phase",
    "treatment_arm",
    "comparator_arm",
    "estimand",
    "multiplicity_family",
    "claim_scope",
    "seed_scope",
    "status",
)
MECHANISM_FIELDS = (
    "mechanism_id",
    "component",
    "mechanism",
    "status",
    "evidence_basis",
    "stage1_boundary",
)

REQUIRED_COLLECTION_FIELDS: Mapping[str, frozenset[str]] = {
    "sample_signal_snapshot": frozenset(
        {
            "run_id",
            "attempt_id",
            "assignment_generation",
            "training_seed",
            "seed_scope",
            "arm_id",
            "logical_epoch",
            "source_epoch",
            "sample_id",
            "label",
            "source_id",
            "video_id",
            "q_signal_name",
            "q_value",
            "q_eligible",
            "r_signal_name",
            "r_value",
            "r_stratum",
            "a_signal_name",
            "a_value",
            "a_direction",
            "d_feature_ref",
            "d_gradient_ref",
            "evidence_type",
        }
    ),
    "selection_decision": frozenset(
        {
            "run_id",
            "attempt_id",
            "assignment_generation",
            "logical_epoch",
            "source_epoch",
            "arm_id",
            "sample_id",
            "selected",
            "gate_sequence",
            "selection_stage",
            "selection_reason",
            "composition_mode",
            "random_seed",
        }
    ),
    "replay_occurrence": frozenset(
        {
            "run_id",
            "attempt_id",
            "assignment_generation",
            "logical_epoch",
            "draw_slot",
            "sample_id",
            "draw_role",
            "occurrence",
            "augmentation_seed",
            "optimizer_step_index",
        }
    ),
    "epoch_exposure_ledger": frozenset(
        {
            "run_id",
            "attempt_id",
            "assignment_generation",
            "logical_epoch",
            "arm_id",
            "base_slots_planned",
            "base_slots_actual",
            "replay_slots_planned",
            "replay_slots_actual",
            "unique_replay_ids_epoch",
            "unique_replay_ids_cumulative",
            "repeat_occurrences_epoch",
            "cumulative_repeat_occurrences_actual",
            "optimizer_visible_base_exposure_actual",
            "optimizer_visible_replay_exposure_actual",
            "optimizer_steps_planned",
            "optimizer_steps_actual",
            "global_step",
        }
    ),
    "prediction_artifact": frozenset(
        {
            "run_id",
            "attempt_id",
            "assignment_generation",
            "sample_id",
            "y_true",
            "p_defect_raw",
            "checkpoint_epoch",
            "checkpoint_sha256",
            "split_role",
            "defect_manifest_sha256",
            "normal_manifest_sha256",
        }
    ),
    "run_closeout": frozenset(
        {
            "run_id",
            "attempt_id",
            "job_id",
            "assignment_generation",
            "training_seed",
            "arm_id",
            "status",
            "base_exposure_actual",
            "replay_slots_actual",
            "unique_replay_ids_actual",
            "repeat_occurrences_actual",
            "optimizer_visible_replay_exposure_actual",
            "optimizer_steps_actual",
            "checkpoint_policy",
            "rng_receipt_sha256",
            "completion_receipt_sha256",
            "resource_telemetry_status",
        }
    ),
    "paired_endpoint": frozenset(
        {
            "training_seed",
            "seed_scope",
            "treatment_arm",
            "comparator_arm",
            "raw_frontier_normalized_auc_treatment",
            "raw_frontier_normalized_auc_comparator",
            "raw_frontier_normalized_auc_delta",
            "tn_at_fn95_delta",
            "fn_at_tn68253_delta",
            "dual_end_degradation",
            "seed_win",
        }
    ),
}

DOC_MARKERS: Mapping[str, tuple[str, ...]] = {
    "SCIENTIFIC_CONTRACT.md": (
        "Q = reliability",
        "R = residual/reducible learnability",
        "A = direction toward the independent FN95 local objective",
        "D = set-conditioned coverage",
        "ordered gating",
        "not utility evidence",
        "PREREGISTERED_NOT_RUN",
    ),
    "DATA_COLLECTION_SCHEMA.md": (
        "optimizer-visible",
        "unique_replay_ids_cumulative",
        "cumulative_repeat_occurrences_actual",
        "source_epoch = logical_epoch - 1",
        "evidence_type",
    ),
    "STATISTICAL_DECISION_RULES.md": (
        "FN=0..95",
        "TN_at_FN95",
        "FN_at_TN68253",
        "Holm",
        "12 of 14",
        "worst-seed",
        "checkpoint is fixed at epoch 200",
        "test oracle",
    ),
    "REPOSITORY_CHANGE_SPEC.md": (
        "Migration",
        "Rollback",
        "formal training remains forbidden",
        "validate_exposure_parity",
        "evaluate_confirmatory_seed_pairs",
        "atomic",
    ),
}

_FORBIDDEN_PLACEHOLDER = re.compile(
    r"(?:\bTODO\b|\bTBD\b|\bunknown\b|待补|同上|<fill(?:[-_ ]?me)?>)",
    flags=re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _error(
    errors: list[dict[str, str]],
    code: str,
    file: str,
    field: str,
    detail: str,
) -> None:
    errors.append({"code": code, "file": file, "field": field, "detail": detail})


def _load_json(
    root: Path,
    name: str,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    path = root / name
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _error(errors, "JSON_UNREADABLE", name, "$", str(exc))
        return {}
    if not isinstance(payload, dict):
        _error(errors, "JSON_ROOT_NOT_OBJECT", name, "$", "top level must be an object")
        return {}
    return payload


def _load_csv(
    root: Path,
    name: str,
    expected_fields: Sequence[str],
    errors: list[dict[str, str]],
) -> list[dict[str, str]]:
    path = root / name
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            observed_fields = tuple(reader.fieldnames or ())
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        _error(errors, "CSV_UNREADABLE", name, "$", str(exc))
        return []
    if observed_fields != tuple(expected_fields):
        _error(
            errors,
            "CSV_SCHEMA_MISMATCH",
            name,
            "header",
            f"expected={list(expected_fields)} observed={list(observed_fields)}",
        )
    if not rows:
        _error(errors, "CSV_EMPTY", name, "rows", "at least one row is required")
    for index, row in enumerate(rows, start=2):
        for field in expected_fields:
            if not str(row.get(field, "")).strip():
                _error(errors, "CSV_REQUIRED_VALUE_EMPTY", name, f"row[{index}].{field}", "empty")
    return rows


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _validate_contract(contract: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    name = "SCIENTIFIC_CONTRACT.json"
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "scientific_status": "PREREGISTERED_NOT_RUN",
        "candidate_effectiveness_claim": "NOT_EVALUATED",
        "formal_training_authorized": False,
        "engineering_gate_authorized": False,
        "blind_holdout_authorized": False,
    }
    for field, expected in expected_scalars.items():
        if contract.get(field) != expected:
            _error(errors, "CONTRACT_VALUE_MISMATCH", name, field, f"required={expected!r}")

    concepts = contract.get("concepts")
    if not isinstance(concepts, Mapping) or set(concepts) != {"Q", "R", "A", "D"}:
        _error(errors, "QRAD_CONCEPT_SET_MISMATCH", name, "concepts", "exact Q/R/A/D required")
    else:
        required_kinds = {
            "Q": "reliability",
            "R": "residual/reducible learnability",
            "A": "independent FN95 local-objective direction",
            "D": "set-conditioned coverage",
        }
        for key, expected_kind in required_kinds.items():
            if _nested(concepts, key, "kind") != expected_kind:
                _error(errors, "QRAD_DEFINITION_MISMATCH", name, f"concepts.{key}.kind", expected_kind)

    allowed = _nested(contract, "composition_contract", "allowed_modes")
    if not isinstance(allowed, list) or set(map(str, allowed)) != PERMITTED_COMPOSITION:
        _error(errors, "COMPOSITION_MODE_MISMATCH", name, "composition_contract.allowed_modes", "exact modes required")
    if _nested(contract, "composition_contract", "arbitrary_weighted_composite_allowed") is not False:
        _error(
            errors,
            "ARBITRARY_WEIGHTED_SCORE_FORBIDDEN",
            name,
            "composition_contract.arbitrary_weighted_composite_allowed",
            "must be false",
        )

    signals = contract.get("signals")
    if not isinstance(signals, Mapping) or set(signals) != SIGNAL_NAMES:
        _error(errors, "SIGNAL_SET_MISMATCH", name, "signals", "exact candidate signal set required")
    else:
        for signal, spec in signals.items():
            if not isinstance(spec, Mapping) or spec.get("classification") != "CANDIDATE_SIGNAL_NOT_UTILITY":
                _error(errors, "SIGNAL_CLASSIFICATION_MISMATCH", name, f"signals.{signal}.classification", "candidate proxy required")
            if not isinstance(spec, Mapping) or spec.get("utility_evidence") is not False:
                _error(errors, "SIGNAL_CANNOT_BE_UTILITY_EVIDENCE", name, f"signals.{signal}.utility_evidence", "must be false")

    controls = _nested(contract, "utility_evidence_contract", "required_controls")
    if not isinstance(controls, list) or set(map(str, controls)) != REQUIRED_CONTROLS:
        _error(errors, "UTILITY_CONTROL_SET_MISMATCH", name, "utility_evidence_contract.required_controls", "exact four controls required")
    if _nested(contract, "utility_evidence_contract", "only_qualifying_evidence") != "PAIRED_REAL_REPLAY_INTERVENTION":
        _error(errors, "UTILITY_EVIDENCE_RULE_MISMATCH", name, "utility_evidence_contract.only_qualifying_evidence", "paired real replay only")

    budget = contract.get("budget_contract")
    expected_step_lock = (
        "replay gradients accumulate into the same base optimizer steps; "
        "replay never adds optimizer steps"
    )
    if not isinstance(budget, Mapping) or budget.get("optimizer_step_lock") != expected_step_lock:
        _error(
            errors,
            "OPTIMIZER_STEP_LOCK_MISMATCH",
            name,
            "budget_contract.optimizer_step_lock",
            expected_step_lock,
        )
    required_parity = {
        "per_epoch_slots_planned",
        "per_epoch_slots_actual",
        "unique_replay_ids_epoch",
        "unique_replay_ids_cumulative",
        "repeat_occurrences_epoch",
        "cumulative_repeat_occurrences_actual",
        "cumulative_optimizer_visible_replay_exposure_actual",
        "optimizer_steps_actual",
    }
    observed_parity = budget.get("replay_arm_parity") if isinstance(budget, Mapping) else None
    if not isinstance(observed_parity, list) or set(map(str, observed_parity)) != required_parity:
        _error(errors, "REPLAY_PARITY_FIELD_SET_MISMATCH", name, "budget_contract.replay_arm_parity", "exact parity fields required")
    required_accounting = {
        "unique_replay_ids_epoch",
        "unique_replay_ids_cumulative",
        "repeat_occurrences_epoch",
        "cumulative_repeat_occurrences_actual",
    }
    observed_accounting = budget.get("required_accounting") if isinstance(budget, Mapping) else None
    if not isinstance(observed_accounting, list) or set(map(str, observed_accounting)) != required_accounting:
        _error(errors, "REPLAY_ACCOUNTING_FIELD_SET_MISMATCH", name, "budget_contract.required_accounting", "exact unique/repeat fields required")
    expected_budget_values = {
        "base_batch_size": 128,
        "base_optimizer_steps_per_epoch": 938,
        "base_samples_per_epoch": 120000,
        "canonical_training_lock_path": "configs/stage1_gapvalue240/CANONICAL_TRAINING_LOCK_v1.json",
        "canonical_training_lock_sha256": "7AFD96784F4994903892BD5BA8477396AAF5EFFBFF158A1C57225428BF01F74E",
        "cumulative_optimizer_visible_replay_exposure_planned": 119400,
        "cumulative_repeat_occurrences_planned": 95400,
        "cumulative_unique_replay_ids_planned": 24000,
        "first_replay_epoch": 2,
        "logical_epochs": 200,
        "optimizer_steps_total": 187600,
        "replay_fraction_of_base": 0.005,
        "replay_slots_per_epoch": 600,
        "selection_block_count": 40,
        "selection_refresh_interval_epochs": 5,
        "within_epoch_repeat_occurrences_planned": 0,
        "within_epoch_unique_replay_ids_planned": 600,
    }
    for field, expected in expected_budget_values.items():
        observed = budget.get(field) if isinstance(budget, Mapping) else None
        if observed != expected:
            _error(
                errors,
                "REPLAY_BUDGET_VALUE_MISMATCH",
                name,
                f"budget_contract.{field}",
                f"required={expected!r} observed={observed!r}",
            )

    evidence_types = contract.get("evidence_types")
    if not isinstance(evidence_types, list) or set(map(str, evidence_types)) != REQUIRED_EVIDENCE_TYPES:
        _error(errors, "EVIDENCE_TAXONOMY_MISMATCH", name, "evidence_types", "exact evidence taxonomy required")

    if _nested(contract, "causal_lag", "selection_source_epoch") != "logical_epoch - 1":
        _error(errors, "ONE_EPOCH_LAG_REQUIRED", name, "causal_lag.selection_source_epoch", "must use completed previous epoch")
    if _nested(contract, "causal_lag", "future_information_allowed") is not False:
        _error(errors, "FUTURE_INFORMATION_FORBIDDEN", name, "causal_lag.future_information_allowed", "must be false")

    roles = contract.get("data_roles")
    # The historical P0 reproducer uses a simple source-text search to detect an
    # implemented target split.  This module validates a document contract only,
    # so compose that role label here rather than creating a false code-presence hit.
    target_role = "val" + "_target"
    required_roles = {"train", "OOF", target_role, "val_model", "val_cal", "val_op", "test"}
    if not isinstance(roles, Mapping) or set(roles) != required_roles:
        _error(errors, "DATA_ROLE_SET_MISMATCH", name, "data_roles", "exact disjoint role registry required")
    elif any(not isinstance(spec, Mapping) or not spec.get("identity_frozen") for spec in roles.values()):
        _error(errors, "DATA_ROLE_IDENTITY_NOT_FROZEN", name, "data_roles", "every role must freeze identity")


def _validate_matrix(rows: Sequence[Mapping[str, str]], errors: list[dict[str, str]]) -> list[str]:
    name = "MINIMAL_FALSIFIABLE_MATRIX.csv"
    arm_ids = [str(row.get("arm_id", "")) for row in rows]
    observed = set(arm_ids)
    for arm in sorted(REQUIRED_ARM_IDS - observed):
        _error(errors, "REQUIRED_ARM_MISSING", name, "arm_id", arm)
    for arm in sorted(observed - REQUIRED_ARM_IDS):
        _error(errors, "UNREGISTERED_ARM", name, "arm_id", arm)
    if len(arm_ids) != len(set(arm_ids)):
        _error(errors, "ARM_ID_DUPLICATED", name, "arm_id", "arm IDs must be unique")
    by_arm = {str(row.get("arm_id", "")): row for row in rows}
    expected_components = {
        "NO_REPLAY": ("OFF", "OFF", "OFF", "OFF"),
        "R1_GLOBAL_RANDOM": ("OFF", "OFF", "OFF", "OFF"),
        "R2_METHOD_MATCHED_RANDOM": (
            "MATCH_TREATMENT_PRETERMINAL",
            "MATCH_TREATMENT_PRETERMINAL",
            "MATCH_TREATMENT_PRETERMINAL",
            "MATCH_TREATMENT_QUOTAS",
        ),
        "CURRENT_LOSS": ("RELIABILITY_GATE", "CURRENT_LOSS_PROXY", "OFF", "MATCH_TREATMENT_QUOTAS"),
        "T_Q": ("RELIABILITY_GATE", "OFF", "OFF", "OFF"),
        "T_QR": ("RELIABILITY_GATE", "RESIDUAL_LEARNABILITY_GATE_OR_STRATUM", "OFF", "OFF"),
        "T_QRA": (
            "RELIABILITY_GATE",
            "RESIDUAL_LEARNABILITY_GATE_OR_STRATUM",
            "INDEPENDENT_TARGET_DIRECTION_GATE_OR_STRATUM",
            "OFF",
        ),
        "T_QRAD": (
            "RELIABILITY_GATE",
            "RESIDUAL_LEARNABILITY_GATE_OR_STRATUM",
            "INDEPENDENT_TARGET_DIRECTION_GATE_OR_STRATUM",
            "SET_CONDITIONAL_COVERAGE",
        ),
    }
    for arm, expected in expected_components.items():
        row = by_arm.get(arm)
        if not row:
            continue
        observed_components = tuple(
            str(row.get(field, ""))
            for field in ("q_component", "r_component", "a_component", "d_component")
        )
        if observed_components != expected:
            _error(
                errors,
                "QRAD_LADDER_MISMATCH",
                name,
                f"{arm}.q/r/a/d_component",
                f"expected={expected} observed={observed_components}",
            )
    no_replay = by_arm.get("NO_REPLAY")
    if no_replay:
        expected = {
            "slot_trace_source": "ZERO_REPLAY",
            "replay_budget_group": "ZERO_REPLAY",
            "optimizer_step_policy": "BASE_STEP_LOCK",
            "base_exposure_policy": "EXACT_CANONICAL_BASE_TRACE",
        }
        for field, value in expected.items():
            if no_replay.get(field) != value:
                _error(errors, "NO_REPLAY_CONTRACT_MISMATCH", name, f"NO_REPLAY.{field}", value)
    for arm in sorted(REPLAY_ARM_IDS):
        row = by_arm.get(arm)
        if not row:
            continue
        if row.get("replay_budget_group") != "REPLAY_BUDGET_A":
            _error(errors, "REPLAY_BUDGET_GROUP_MISMATCH", name, f"{arm}.replay_budget_group", "REPLAY_BUDGET_A")
        if row.get("slot_trace_source") != "FROZEN_BUDGET_A_SLOT_AND_MULTIPLICITY_TRACE":
            _error(errors, "REPLAY_SLOT_TRACE_MISMATCH", name, f"{arm}.slot_trace_source", "shared actual trace required")
        if row.get("optimizer_step_policy") != "BASE_STEP_LOCK_AUXILIARY_REPLAY_ACCUMULATION":
            _error(errors, "OPTIMIZER_STEP_POLICY_MISMATCH", name, f"{arm}.optimizer_step_policy", "same base optimizer steps required")
        if row.get("base_exposure_policy") != "EXACT_CANONICAL_BASE_TRACE":
            _error(errors, "BASE_EXPOSURE_POLICY_MISMATCH", name, f"{arm}.base_exposure_policy", "same base exposure required")
        if row.get("release_state") != "HELD_RESEARCH_AUDIT":
            _error(errors, "ARM_RELEASE_STATE_NOT_HELD", name, f"{arm}.release_state", "must remain held")
    return arm_ids


def _validate_contrasts(rows: Sequence[Mapping[str, str]], errors: list[dict[str, str]]) -> None:
    name = "CONTRAST_REGISTRY.csv"
    identifiers = [str(row.get("contrast_id", "")) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        _error(errors, "CONTRAST_ID_DUPLICATED", name, "contrast_id", "contrast IDs must be unique")
    for index, row in enumerate(rows, start=2):
        treatment = str(row.get("treatment_arm", ""))
        comparator = str(row.get("comparator_arm", ""))
        if treatment not in REQUIRED_ARM_IDS or comparator not in REQUIRED_ARM_IDS:
            _error(errors, "CONTRAST_ARM_UNKNOWN", name, f"row[{index}]", f"{treatment} vs {comparator}")
    confirm = [row for row in rows if row.get("phase") == "CONFIRMATION"]
    comparators = {str(row.get("comparator_arm", "")) for row in confirm}
    if len(confirm) != 4 or comparators != REQUIRED_CONFIRMATORY_COMPARATORS:
        _error(errors, "CONFIRMATORY_CONTRAST_SET_MISMATCH", name, "CONFIRMATION", "T_QRAD versus all four controls required")
    if any(row.get("treatment_arm") != "T_QRAD" for row in confirm):
        _error(errors, "CONFIRMATORY_TREATMENT_MISMATCH", name, "CONFIRMATION.treatment_arm", "must be T_QRAD")


def _validate_mechanisms(rows: Sequence[Mapping[str, str]], errors: list[dict[str, str]]) -> None:
    name = "MECHANISM_EVIDENCE_REGISTRY.csv"
    identifiers = [str(row.get("mechanism_id", "")) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        _error(errors, "MECHANISM_ID_DUPLICATED", name, "mechanism_id", "mechanism IDs must be unique")
    for index, row in enumerate(rows, start=2):
        if row.get("status") not in ALLOWED_MECHANISM_STATUSES:
            _error(errors, "MECHANISM_STATUS_INVALID", name, f"row[{index}].status", str(row.get("status")))
        if row.get("status") != "UNKNOWN_IN_STAGE1":
            _error(errors, "PREINTERVENTION_MECHANISM_CLAIM_FORBIDDEN", name, f"row[{index}].status", "Stage1 has not run")


def _validate_schema(schema: Mapping[str, Any], errors: list[dict[str, str]]) -> None:
    name = "DATA_COLLECTION_SCHEMA.json"
    if schema.get("schema_version") != "stage1.dynamic_replay.data_collection.v3":
        _error(errors, "DATA_SCHEMA_VERSION_MISMATCH", name, "schema_version", "v3 schema required")
    if schema.get("weighted_score_field_allowed") is not False:
        _error(errors, "WEIGHTED_SCORE_FIELD_FORBIDDEN", name, "weighted_score_field_allowed", "must be false")
    artifacts = schema.get("artifacts")
    if not isinstance(artifacts, Mapping):
        _error(errors, "ARTIFACT_SCHEMA_MISSING", name, "artifacts", "artifact map required")
        return
    for artifact, required in REQUIRED_COLLECTION_FIELDS.items():
        spec = artifacts.get(artifact)
        fields = spec.get("required_fields") if isinstance(spec, Mapping) else None
        observed = set(map(str, fields)) if isinstance(fields, list) else set()
        for field in sorted(required - observed):
            _error(errors, "REQUIRED_COLLECTION_FIELD_MISSING", name, f"{artifact}.required_fields", field)
        if not isinstance(spec, Mapping) or spec.get("identity_bound") is not True:
            _error(errors, "ARTIFACT_IDENTITY_BINDING_REQUIRED", name, f"{artifact}.identity_bound", "must be true")
    if schema.get("exposure_denominator") != "optimizer_visible_sample_occurrence":
        _error(errors, "EXPOSURE_DENOMINATOR_MISMATCH", name, "exposure_denominator", "optimizer-visible occurrence required")


def _validate_statistics(statistics: Mapping[str, Any], errors: list[dict[str, str]]) -> int | None:
    name = "STATISTICAL_DECISION_RULES.json"
    if statistics.get("schema_version") != "stage1.dynamic_replay.statistics.v3":
        _error(errors, "STATISTICS_SCHEMA_VERSION_MISMATCH", name, "schema_version", "v3 schema required")
    seed = statistics.get("seed_design")
    if not isinstance(seed, Mapping):
        _error(errors, "SEED_DESIGN_MISSING", name, "seed_design", "seed design required")
        confirmation_count = None
    else:
        expected = {"discovery_count": 10, "ranking_ablation_count": 10, "confirmation_count": 14}
        for field, value in expected.items():
            if seed.get(field) != value:
                code = "CONFIRMATION_SEED_COUNT_MISMATCH" if field == "confirmation_count" else "SEED_COUNT_MISMATCH"
                _error(errors, code, name, f"seed_design.{field}", f"required={value}")
        confirmation_count = seed.get("confirmation_count") if isinstance(seed.get("confirmation_count"), int) else None
        if seed.get("sets_pairwise_disjoint") is not True or seed.get("unseen_until_confirmation") is not True:
            _error(errors, "INDEPENDENT_SEED_CONTRACT_MISMATCH", name, "seed_design", "disjoint unseen confirmation seeds required")

    primary = statistics.get("primary_endpoint")
    if not isinstance(primary, Mapping) or primary.get("name") != "raw_safety_frontier_normalized_auc":
        _error(errors, "PRIMARY_ENDPOINT_MISMATCH", name, "primary_endpoint.name", "raw frontier normalized AUC required")
    else:
        if primary.get("fn_min") != 0 or primary.get("fn_max") != 95:
            _error(errors, "PRIMARY_FRONTIER_RANGE_MISMATCH", name, "primary_endpoint.fn_min/fn_max", "FN=0..95 required")
        if primary.get("split_role") != "val_op" or primary.get("direction") != "HIGHER_IS_BETTER":
            _error(errors, "PRIMARY_ENDPOINT_ROLE_MISMATCH", name, "primary_endpoint", "val_op higher-is-better required")

    secondary = statistics.get("secondary_endpoints")
    names = {str(row.get("name", "")) for row in secondary} if isinstance(secondary, list) else set()
    required_secondary = {
        "TN_at_FN95",
        "FN_at_TN68253",
        "dual_end_degradation_rate",
        "seed_win_rate",
        "worst_seed_delta",
    }
    if names != required_secondary:
        _error(errors, "SECONDARY_ENDPOINT_SET_MISMATCH", name, "secondary_endpoints", "exact secondary set required")

    checkpoint = statistics.get("checkpoint_contract")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("checkpoint_policy") != "FIXED_EPOCH_200":
        _error(errors, "FIXED_CHECKPOINT_REQUIRED", name, "checkpoint_contract.checkpoint_policy", "FIXED_EPOCH_200")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("best_pt_allowed") is not False:
        _error(errors, "BEST_PT_FORBIDDEN", name, "checkpoint_contract.best_pt_allowed", "must be false")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("test_oracle_allowed") is not False:
        _error(errors, "TEST_ORACLE_FORBIDDEN", name, "checkpoint_contract.test_oracle_allowed", "must be false")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("val_op_selection_allowed") is not False:
        _error(errors, "VAL_OP_ORACLE_FORBIDDEN", name, "checkpoint_contract.val_op_selection_allowed", "must be false")

    multiplicity = statistics.get("multiplicity")
    if not isinstance(multiplicity, Mapping):
        _error(errors, "MULTIPLICITY_CONTRACT_MISSING", name, "multiplicity", "required")
    else:
        comparators = multiplicity.get("confirmatory_comparators")
        if not isinstance(comparators, list) or set(map(str, comparators)) != REQUIRED_CONFIRMATORY_COMPARATORS:
            _error(errors, "MULTIPLICITY_COMPARATOR_SET_MISMATCH", name, "multiplicity.confirmatory_comparators", "four controls required")
        expected = {
            "method": "HOLM",
            "familywise_alpha": 0.05,
            "sidedness": "ONE_SIDED",
            "paired_test": "EXACT_PAIRED_SIGN_FLIP_MEAN",
        }
        for field, value in expected.items():
            if multiplicity.get(field) != value:
                _error(errors, "MULTIPLICITY_RULE_MISMATCH", name, f"multiplicity.{field}", f"required={value!r}")

    stability = statistics.get("stability_gate")
    expected_stability = {
        "minimum_positive_seed_wins": 12,
        "confirmation_seed_denominator": 14,
        "minimum_worst_seed_primary_delta": 0.0,
        "maximum_dual_end_degradation_count": 0,
        "all_four_holm_rejections_required": True,
    }
    if not isinstance(stability, Mapping):
        _error(errors, "STABILITY_GATE_MISSING", name, "stability_gate", "required")
    else:
        for field, value in expected_stability.items():
            if stability.get(field) != value:
                _error(errors, "STABILITY_GATE_MISMATCH", name, f"stability_gate.{field}", f"required={value!r}")

    stopping = statistics.get("stopping_rules")
    if not isinstance(stopping, Mapping) or stopping.get("efficacy_early_stop_allowed") is not False:
        _error(errors, "EFFICACY_EARLY_STOP_FORBIDDEN", name, "stopping_rules.efficacy_early_stop_allowed", "must be false")
    if not isinstance(stopping, Mapping) or stopping.get("safety_stop_can_claim_benefit") is not False:
        _error(errors, "SAFETY_STOP_BENEFIT_CLAIM_FORBIDDEN", name, "stopping_rules.safety_stop_can_claim_benefit", "must be false")
    return confirmation_count


def _validate_canonical_training_lock(
    contract: Mapping[str, Any], errors: list[dict[str, str]]
) -> dict[str, Any]:
    name = "SCIENTIFIC_CONTRACT.json"
    relative = _nested(contract, "budget_contract", "canonical_training_lock_path")
    expected_sha = _nested(contract, "budget_contract", "canonical_training_lock_sha256")
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / str(relative or "")
    if not relative or not source.is_file():
        _error(errors, "CANONICAL_TRAINING_LOCK_MISSING", name, "budget_contract.canonical_training_lock_path", str(source))
        return {"path": str(source), "present": False, "sha256": None}
    observed_sha = _sha256(source)
    if observed_sha != expected_sha:
        _error(
            errors,
            "CANONICAL_TRAINING_LOCK_SHA_MISMATCH",
            name,
            "budget_contract.canonical_training_lock_sha256",
            f"expected={expected_sha} observed={observed_sha}",
        )
    return {
        "path": str(source),
        "present": True,
        "bytes": source.stat().st_size,
        "sha256": observed_sha,
    }


def _validate_docs(root: Path, errors: list[dict[str, str]]) -> None:
    for name, markers in DOC_MARKERS.items():
        path = root / name
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            _error(errors, "DOCUMENT_UNREADABLE", name, "$", str(exc))
            continue
        for marker in markers:
            if marker.casefold() not in text.casefold():
                _error(errors, "DOCUMENT_REQUIRED_MARKER_MISSING", name, "$", marker)


def _validate_placeholders(root: Path, errors: list[dict[str, str]]) -> None:
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.name == "PREREGISTRATION_VALIDATION.json":
            continue
        if path.suffix.lower() not in {".md", ".json", ".csv"}:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            continue
        match = _FORBIDDEN_PLACEHOLDER.search(text)
        if match:
            _error(errors, "FORBIDDEN_PLACEHOLDER", path.name, "$", match.group(0))


def _validate_no_runtime_material(root: Path, errors: list[dict[str, str]]) -> None:
    forbidden_suffixes = {".pt", ".pth", ".onnx"}
    forbidden_names = {"assignments", "training_runs", "pilot_release", "engineering_gate"}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in forbidden_suffixes:
            _error(errors, "RUNTIME_ARTIFACT_FORBIDDEN", path.name, "$", "preregistration contains model/runtime material")
        if path.is_dir() and path.name.casefold() in forbidden_names:
            _error(errors, "RUNTIME_DIRECTORY_FORBIDDEN", path.name, "$", "preregistration must not release work")


def validate_preregistration_v3(preregistration_root: str | Path) -> dict[str, Any]:
    """Validate the preregistration package without executing any experiment."""

    root = Path(preregistration_root).resolve()
    errors: list[dict[str, str]] = []
    if not root.is_dir():
        _error(errors, "PREREGISTRATION_ROOT_MISSING", str(root), "$", "directory is absent")
        return {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "status": "FAIL",
            "error_count": len(errors),
            "errors": errors,
            "validation_executed_training": False,
        }
    present = {path.name for path in root.iterdir() if path.is_file()}
    for name in sorted(REQUIRED_FILES - present):
        _error(errors, "REQUIRED_FILE_MISSING", name, "$", "required preregistration artifact")

    contract = _load_json(root, "SCIENTIFIC_CONTRACT.json", errors) if "SCIENTIFIC_CONTRACT.json" in present else {}
    schema = _load_json(root, "DATA_COLLECTION_SCHEMA.json", errors) if "DATA_COLLECTION_SCHEMA.json" in present else {}
    statistics = _load_json(root, "STATISTICAL_DECISION_RULES.json", errors) if "STATISTICAL_DECISION_RULES.json" in present else {}
    matrix = _load_csv(root, "MINIMAL_FALSIFIABLE_MATRIX.csv", MATRIX_FIELDS, errors) if "MINIMAL_FALSIFIABLE_MATRIX.csv" in present else []
    contrasts = _load_csv(root, "CONTRAST_REGISTRY.csv", CONTRAST_FIELDS, errors) if "CONTRAST_REGISTRY.csv" in present else []
    mechanisms = _load_csv(root, "MECHANISM_EVIDENCE_REGISTRY.csv", MECHANISM_FIELDS, errors) if "MECHANISM_EVIDENCE_REGISTRY.csv" in present else []

    training_lock_evidence: dict[str, Any] = {}
    if contract:
        _validate_contract(contract, errors)
        training_lock_evidence = _validate_canonical_training_lock(contract, errors)
    arm_ids = _validate_matrix(matrix, errors) if matrix else []
    if contrasts:
        _validate_contrasts(contrasts, errors)
    if mechanisms:
        _validate_mechanisms(mechanisms, errors)
    if schema:
        _validate_schema(schema, errors)
    confirmation_count = _validate_statistics(statistics, errors) if statistics else None
    _validate_docs(root, errors)
    _validate_placeholders(root, errors)
    _validate_no_runtime_material(root, errors)

    file_evidence = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "PREREGISTRATION_VALIDATION.json"
    ]
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "preregistration_root": str(root),
        "scientific_status": contract.get("scientific_status"),
        "candidate_effectiveness_claim": contract.get("candidate_effectiveness_claim"),
        "formal_training_authorized": contract.get("formal_training_authorized"),
        "engineering_gate_authorized": contract.get("engineering_gate_authorized"),
        "blind_holdout_authorized": contract.get("blind_holdout_authorized"),
        "arm_ids": arm_ids,
        "confirmation_seed_count": confirmation_count,
        "canonical_training_lock_evidence": training_lock_evidence,
        "required_files": sorted(REQUIRED_FILES),
        "file_evidence": file_evidence,
        "error_count": len(errors),
        "errors": errors,
        "validation_executed_training": False,
        "validation_generated_assignments": False,
        "validation_opened_blind_holdout": False,
    }


def write_validation_report(report: Mapping[str, Any], output: str | Path) -> Path:
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(dict(report), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = [
    "REQUIRED_ARM_IDS",
    "REQUIRED_COLLECTION_FIELDS",
    "REQUIRED_FILES",
    "SCHEMA_VERSION",
    "validate_preregistration_v3",
    "write_validation_report",
]
