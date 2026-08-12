"""Recoverable orchestration for the comprehensive 240-run analysis goal."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import pandas as pd
import yaml

from .comprehensive_audit import (
    DataCoverageError,
    apply_usage_rules,
    assert_complete_usage_ledger,
    audit_epoch_grid,
    build_canonical_file_inventory,
    build_field_coverage,
    default_usage_rules,
)
from .training_telemetry import (
    build_execution_exposure_audit,
    build_paired_telemetry_deltas,
    build_run_telemetry_features,
    build_triad_outcomes,
    parse_effective_optimizers,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_yaml(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_triads(
    inventory: pd.DataFrame,
    *,
    expected_runs: int,
    expected_triads: int,
) -> None:
    required = {"run_slot", "triad_id", "arm"}
    missing = required - set(inventory.columns)
    if missing:
        raise DataCoverageError(
            f"Canonical inventory missing triad columns: {sorted(missing)}"
        )
    if len(inventory) != expected_runs or inventory["run_slot"].nunique() != expected_runs:
        raise DataCoverageError(
            f"Expected {expected_runs} unique canonical runs, found "
            f"{len(inventory)}/{inventory['run_slot'].nunique()}"
        )
    if inventory["triad_id"].nunique() != expected_triads:
        raise DataCoverageError(
            f"Expected {expected_triads} triads, found {inventory['triad_id'].nunique()}"
        )
    for triad_id, group in inventory.groupby("triad_id", sort=False):
        arms = sorted(group["arm"].astype(str).tolist())
        if arms != ["R1", "R2", "T"]:
            raise DataCoverageError(
                f"Triad {triad_id} must be exactly T/R1/R2, found {arms}"
            )


def _synthetic_capability_rows(columns: list[str]) -> pd.DataFrame:
    capabilities = [
        (
            "<NOT_COLLECTED>/gradients",
            field,
            "Gradient fields exist in the value schema but contain no observations.",
        )
        for field in (
            "grad_mag_score",
            "grad_align_score",
            "grad_align_normal_tail",
            "grad_align_defect_tail",
            "gradient_diversity_embedding",
        )
    ] + [
        (
            "<NOT_COLLECTED>/checkpoints",
            "epoch_150_checkpoint",
            "Only initial, stripped best EMA and final resumable EMA evidence is available.",
        ),
        (
            "<NOT_COLLECTED>/checkpoints",
            "per_epoch_checkpoint_trajectory",
            "No per-epoch checkpoint series was retained for the 240 canonical runs.",
        ),
        (
            "<NOT_COLLECTED>/predictions",
            "last_checkpoint_val_op_predictions",
            "Saved val_cal/val_op predictions were generated from best.pt only.",
        ),
        (
            "<NOT_COLLECTED>/predictions",
            "per_epoch_val_op_predictions",
            "No per-epoch operational predictions were retained.",
        ),
        (
            "<NOT_COLLECTED>/optimizer",
            "per_sample_gradient_or_update",
            "Optimizer state is recoverable in last.pt but per-sample gradients/updates are not.",
        ),
        (
            "<NOT_COLLECTED>/optimizer",
            "per_step_momentum_buffer_and_minibatch_order",
            "No per-step optimizer/momentum trajectory or minibatch order was retained.",
        ),
        (
            "<NOT_COLLECTED>/augmentation",
            "sample_epoch_exposure_rng_realization",
            "Configured augmentation is known; actual per-sample RNG realization is not retained.",
        ),
        (
            "<NOT_COLLECTED>/evaluation",
            "no_replay_arm",
            "The 240-run matrix contains T/R1/R2 replay arms and no no-replay arm.",
        ),
        (
            "<NOT_COLLECTED>/evaluation",
            "blind_or_external_test",
            "No untouched blind/external set is present in the 240-run result package.",
        ),
    ]
    records = []
    missing_from_attempt = [
        "method_raw_score",
        "method_normalized_score",
        "candidate_pool",
        "r2_matching_bin",
        "r2_fallback_level",
        "r2_matching_distance",
        "group_id",
        "explicit_noise_flag",
        "phase_b_global_rank",
    ]
    for path, field, rationale in capabilities:
        record: dict[str, Any] = {column: pd.NA for column in columns}
        record.update(
            {
                "normalized_path": path,
                "field_path": field,
                "source_kind": "NOT_COLLECTED",
                "files_observed": 0,
                "runs_observed": 0,
                "files_expected": 0,
                "runs_expected": 0,
                "schema_variant_count": 0,
                "usage_role": "NOT_COLLECTED_FIELD",
                "usage_status": "NOT_TESTABLE",
                "matched_rule_id": "synthetic-not-collected-capability",
                "rationale": rationale,
            }
        )
        records.append(record)
    for field in missing_from_attempt:
        record = {column: pd.NA for column in columns}
        record.update(
            {
                "normalized_path": "<MISSING_FROM_ATTEMPT_SELECTION>/selection_manifest.csv",
                "field_path": field,
                "source_kind": "MISSING_FROM_ATTEMPT",
                "files_observed": 0,
                "runs_observed": 0,
                "files_expected": 240,
                "runs_expected": 240,
                "schema_variant_count": 0,
                "usage_role": "MISSING_FIELD",
                "usage_status": "DOCUMENTED_MISSING",
                "matched_rule_id": "synthetic-attempt-local-missing",
                "rationale": (
                    "Not stored in each attempt selection manifest; may be recoverable "
                    "only from checksum-matched shared ranking/value assets."
                ),
            }
        )
        records.append(record)
    return pd.DataFrame(records, columns=columns)


def _initial_hypotheses() -> pd.DataFrame:
    records = [
        (
            "H01_LATE_MEMORIZATION",
            "Late extra train-loss reduction with val/tail deterioration predicts dual harm.",
            "INCONCLUSIVE",
            "EPOCH+PRED+OPS",
        ),
        (
            "H02_LEARNABLE_NOT_PERSISTENT",
            "Low-forgetting, finally learned samples outperform persistent-hard composition.",
            "INCONCLUSIVE",
            "DYN+SEL+OPS",
        ),
        (
            "H03_EXPOSURE_LR_INTERACTION",
            "The 150-160 split reflects selection/exposure interacting with non-zero LR, not an LR kink.",
            "INCONCLUSIVE",
            "EPOCH+EXEC+SEL",
        ),
        (
            "H04_LAYERWISE_DRIFT",
            "Dual-good and dual-harm runs have different best-to-last layerwise EMA drift patterns.",
            "INCONCLUSIVE",
            "CKPT+OPS",
        ),
        (
            "H05_RAW_NP_FRONTIER",
            "Real improvement requires raw-score NP frontier and fixed-tail gains, not calibration/threshold sliding.",
            "INCONCLUSIVE",
            "PRED+OPS",
        ),
        (
            "H06_UNSEEN_SEED_STABILITY",
            "A predeclared joint rule can beat both R1/R2 on at least 80% of held-out training seeds.",
            "INCONCLUSIVE",
            "CTX+ALL_PREOUTCOME_FEATURES",
        ),
        (
            "H07_TRUE_GRADIENT_ALIGNMENT",
            "True per-sample gradient alignment predicts sample value.",
            "NOT_TESTABLE",
            "GRAD",
        ),
    ]
    return pd.DataFrame(
        records,
        columns=(
            "hypothesis_id",
            "hypothesis",
            "evidence_status",
            "required_evidence",
        ),
    ).assign(causal_claim_allowed=False)


def _refresh_manifest(output: Path, inventory_file: Path) -> None:
    manifest_files = []
    for path in sorted(
        candidate
        for candidate in output.rglob("*")
        if candidate.is_file() and candidate.name != "manifest.json"
    ):
        manifest_files.append(
            {
                "relative_path": path.relative_to(output).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "schema_version": "stage1_gapvalue240_goal_manifest_v1",
        "inventory_sha256": _sha256(inventory_file),
        "file_count": len(manifest_files),
        "files": manifest_files,
    }
    _atomic_json(manifest, output / "manifest.json")


def publish_evidence_foundation(
    *,
    extracted_root: str | Path,
    inventory_path: str | Path,
    output_dir: str | Path,
    expected_runs: int = 240,
    expected_triads: int = 80,
    expected_epochs: int = 200,
    max_workers: int = 32,
) -> dict[str, Any]:
    """Publish the recoverable all-file/all-field foundation into .inprogress."""

    output = Path(output_dir).resolve()
    if not output.name.endswith(".inprogress"):
        raise DataCoverageError(
            "Evidence foundation must remain in a .inprogress directory until the full Goal passes"
        )
    output.mkdir(parents=True, exist_ok=True)
    inventory_file = Path(inventory_path).resolve()
    inventory = pd.read_csv(inventory_file, keep_default_na=False)
    _validate_triads(
        inventory,
        expected_runs=expected_runs,
        expected_triads=expected_triads,
    )

    files = build_canonical_file_inventory(extracted_root, inventory_file)
    coverage = build_field_coverage(files, max_workers=max_workers)
    ledger = apply_usage_rules(coverage.field_coverage, default_usage_rules())
    synthetic = _synthetic_capability_rows(list(ledger.columns))
    ledger = pd.concat([ledger, synthetic], ignore_index=True).sort_values(
        ["normalized_path", "field_path"], ignore_index=True
    )
    gate = assert_complete_usage_ledger(ledger)
    curves = audit_epoch_grid(
        files,
        inventory,
        expected_runs=expected_runs,
        expected_epochs=expected_epochs,
    )

    role_summary = (
        ledger.groupby("normalized_path", sort=True)
        .agg(
            usage_roles=("usage_role", lambda values: ";".join(sorted(set(map(str, values))))),
            usage_statuses=("usage_status", lambda values: ";".join(sorted(set(map(str, values))))),
        )
        .reset_index()
    )
    source_files = files.merge(
        coverage.file_schemas[
            [
                "run_slot",
                "relative_path",
                "field_count",
                "schema_signature",
                "source_kind",
            ]
        ],
        on=["run_slot", "relative_path"],
        how="left",
        validate="one_to_one",
    ).merge(
        role_summary,
        on="normalized_path",
        how="left",
        validate="many_to_one",
    )

    state = {
        "schema_version": "stage1_gapvalue240_goal_analysis_state_v1",
        "status": "IN_PROGRESS",
        "completed_stage": "EVIDENCE_FOUNDATION",
        "source_scope": "240 VALIDATED canonical runs only",
        "gates": {
            "canonical_runs": int(inventory["run_slot"].nunique()),
            "triads": int(inventory["triad_id"].nunique()),
            "paired_comparisons": int(inventory["triad_id"].nunique() * 2),
            "epoch_rows": int(len(curves)),
            "source_files": int(len(files)),
            "normalized_file_types": int(files["normalized_path"].nunique()),
            "field_ledger_rows": int(len(ledger)),
            **gate,
        },
        "limitations": [
            "No no-replay arm exists; replay versus no replay is not testable.",
            "No blind/external test exists; val_op conclusions are internal discovery evidence.",
            "Per-sample gradients, epoch-150 checkpoint and per-epoch val_op predictions were not collected.",
        ],
        "next_required_stage": "TRAINING_TELEMETRY_AND_CHECKPOINT_DRIFT",
    }
    contract = {
        "scope": "240 VALIDATED canonical runs; 80 T/R1/R2 triads",
        "excluded": ["legacy 40-run", "legacy 120-run", "debug", "canary", "failed attempts", "noncanonical attempts"],
        "primary_deltas": {
            "delta_TN": "TN_at_FN95(T) - TN_at_FN95(control); higher is better",
            "delta_FN": "FN_at_TN68253(T) - FN_at_TN68253(control); lower is better",
        },
        "cohorts": {
            "dual_improvement": "both controls delta_TN>0 and delta_FN<=0",
            "high_value": "both controls delta_TN>=300 and delta_FN<=2",
            "dual_harm": "both controls delta_TN<0 and delta_FN>0",
            "mixed_or_reversal": "all remaining triads after the exclusive outcome assignment",
        },
        "orthogonal_attributes": {
            "same_selection_cross_seed_reversal": (
                "audited separately across treatment-selection digests; may occur inside "
                "dual_improvement, dual_harm, or mixed outcome cohorts"
            ),
        },
        "completion_gate": "UNREVIEWED=UNCLASSIFIED=SILENTLY_DROPPED=0 plus all requested analyses",
    }

    _atomic_csv(source_files, output / "audit/source_file_ledger.csv")
    _atomic_csv(ledger, output / "audit/DATA_USAGE_LEDGER.csv")
    _atomic_csv(coverage.file_schemas, output / "audit/file_schema_signatures.csv")
    _atomic_csv(curves, output / "tables/epoch_training_curves_full.csv")
    _atomic_csv(_initial_hypotheses(), output / "tables/HYPOTHESIS_REGISTRY.csv")
    _atomic_json(state, output / "ANALYSIS_STATE.json")
    _atomic_yaml(contract, output / "analysis_contract.yaml")

    _refresh_manifest(output, inventory_file)
    return state


def _cohort_telemetry_summary(
    paired: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    delta_columns = [column for column in paired.columns if column.startswith("delta__")]
    consensus = (
        paired.groupby("triad_id", sort=True)[delta_columns]
        .mean()
        .reset_index()
        .merge(
            outcomes[
                [
                    "triad_id",
                    "exclusive_cohort",
                    "dual_improvement",
                    "high_value",
                    "dual_harm",
                ]
            ],
            on="triad_id",
            how="left",
            validate="one_to_one",
        )
    )
    records: list[dict[str, Any]] = []
    for cohort, group in consensus.groupby("exclusive_cohort", sort=True):
        for column in delta_columns:
            values = pd.to_numeric(group[column], errors="raise")
            records.append(
                {
                    "exclusive_cohort": cohort,
                    "feature": column,
                    "triad_count": int(len(values)),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                }
            )
    return pd.DataFrame(records)


def _feature_time_registry(run_features: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    identity = {
        "run_slot",
        "triad_id",
        "arm",
        "condition_id",
        "condition_slot",
        "method",
        "phase",
        "training_seed",
        "selection_seed",
        "budget",
        "guard_ratio",
        "machine_id",
        "input_snapshot_id",
        "resume_count",
    }
    for column in run_features.columns:
        if column in identity:
            records.append(
                {
                    "feature": column,
                    "feature_family": "IDENTITY_OR_CONFOUND",
                    "available_epoch": 0,
                    "allowed_as_predictor": False,
                    "use": "grouping/pairing/confound only",
                }
            )
            continue
        cutoff_matches = re.findall(r"(?:_to_|__at_|__unique_to_)(\d+)$", column)
        available = int(cutoff_matches[-1]) if cutoff_matches else 200
        if column.startswith(("effective_", "parameter_group_")):
            available = 0
            family = "EFFECTIVE_OPTIMIZER"
        elif column.startswith(("optimizer_steps_", "replay_", "lr_step_", "lr_replay_")):
            family = "EXPOSURE_OR_LR_INTEGRAL"
        elif "__" in column or column.startswith("time_"):
            family = "TRAINING_TELEMETRY"
        else:
            family = "DESCRIPTIVE_OR_LINEAGE"
        records.append(
            {
                "feature": column,
                "feature_family": family,
                "available_epoch": available,
                "allowed_as_predictor": family
                in {"EFFECTIVE_OPTIMIZER", "EXPOSURE_OR_LR_INTEGRAL", "TRAINING_TELEMETRY"},
                "use": "fold-local candidate feature" if family != "DESCRIPTIVE_OR_LINEAGE" else "audit only",
            }
        )
    return pd.DataFrame(records).sort_values("feature", ignore_index=True)


def publish_training_telemetry_stage(
    *,
    extracted_root: str | Path,
    inventory_path: str | Path,
    output_dir: str | Path,
    canonical_metrics_path: str | Path,
    max_workers: int = 32,
) -> dict[str, Any]:
    """Extend a validated foundation with all-parameter training telemetry."""

    del max_workers  # retained in the public contract for later parallel stages
    output = Path(output_dir).resolve()
    inventory_file = Path(inventory_path).resolve()
    required_existing = [
        output / "ANALYSIS_STATE.json",
        output / "audit/DATA_USAGE_LEDGER.csv",
        output / "tables/HYPOTHESIS_REGISTRY.csv",
        output / "tables/epoch_training_curves_full.csv",
    ]
    for path in required_existing:
        if not path.is_file():
            raise DataCoverageError(f"Missing recoverable Goal state input: {path}")
    state = json.loads((output / "ANALYSIS_STATE.json").read_text(encoding="utf-8"))
    ledger = pd.read_csv(output / "audit/DATA_USAGE_LEDGER.csv")
    assert_complete_usage_ledger(ledger)
    pd.read_csv(output / "tables/HYPOTHESIS_REGISTRY.csv")
    if int(state.get("gates", {}).get("canonical_runs", -1)) != 240:
        raise DataCoverageError("Foundation does not prove 240 canonical runs")

    inventory = pd.read_csv(inventory_file, keep_default_na=False)
    files = build_canonical_file_inventory(extracted_root, inventory_file)
    curves = pd.read_csv(output / "tables/epoch_training_curves_full.csv")
    if len(curves) != 48_000:
        raise DataCoverageError(f"Expected 48,000 epoch rows, found {len(curves)}")
    run_features = build_run_telemetry_features(curves)
    optimizers = parse_effective_optimizers(files)
    exposure = build_execution_exposure_audit(files, curves)

    metrics = pd.read_csv(Path(canonical_metrics_path).resolve())
    if len(metrics) != 240 or set(metrics["run_slot"].astype(str)) != set(
        inventory["run_slot"].astype(str)
    ):
        raise DataCoverageError("Canonical metrics do not match the 240-run inventory")
    outcomes = build_triad_outcomes(metrics)
    if len(outcomes) != 80:
        raise DataCoverageError(f"Expected 80 triad outcomes, found {len(outcomes)}")

    merged = run_features.merge(
        optimizers,
        on="run_slot",
        how="left",
        validate="one_to_one",
    ).merge(
        exposure,
        on="run_slot",
        how="left",
        validate="one_to_one",
        suffixes=("", "_execution"),
    )
    paired = build_paired_telemetry_deltas(merged)
    if len(paired) != 160:
        raise DataCoverageError(f"Expected 160 paired telemetry rows, found {len(paired)}")
    cohort_summary = _cohort_telemetry_summary(paired, outcomes)
    feature_registry = _feature_time_registry(merged)

    _atomic_csv(metrics, output / "tables/canonical_run_metrics_240.csv")
    _atomic_csv(outcomes, output / "tables/triad_outcomes_80.csv")
    _atomic_csv(run_features, output / "tables/run_training_telemetry_features.csv")
    _atomic_csv(optimizers, output / "tables/effective_optimizer_audit.csv")
    _atomic_csv(exposure, output / "tables/training_exposure_audit.csv")
    _atomic_csv(merged, output / "tables/run_training_features_merged.csv")
    _atomic_csv(paired, output / "tables/triad_paired_telemetry_deltas.csv")
    _atomic_csv(cohort_summary, output / "tables/telemetry_cohort_summary.csv")
    _atomic_csv(feature_registry, output / "tables/FEATURE_TIME_REGISTRY.csv")

    state["completed_stage"] = "TRAINING_TELEMETRY"
    state["next_required_stage"] = "CHECKPOINT_AND_PREDICTION_MECHANISMS"
    state["gates"].update(
        {
            "run_training_feature_rows": int(len(run_features)),
            "effective_optimizer_rows": int(len(optimizers)),
            "execution_exposure_rows": int(len(exposure)),
            "triad_outcome_rows": int(len(outcomes)),
            "paired_telemetry_rows": int(len(paired)),
            "dual_improvement_triads": int(outcomes["dual_improvement"].sum()),
            "high_value_triads": int(outcomes["high_value"].sum()),
            "dual_harm_triads": int(outcomes["dual_harm"].sum()),
        }
    )
    _atomic_json(state, output / "ANALYSIS_STATE.json")
    _refresh_manifest(output, inventory_file)
    return state
