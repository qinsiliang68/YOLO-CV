"""Field sufficiency and storage audit for the dynamic replay campaign.

The module treats the existing 240-run field ledger and the 10-fold OOF
trajectory export as immutable evidence.  It does not scan model checkpoints or
rewrite prior reports.  Its purpose is to turn those sources into a decision
ledger for the next campaign: what exists, what is missing, why each field
matters, and what the retention plan will cost.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


class CampaignAuditError(RuntimeError):
    """Raised when evidence or campaign field contracts are incomplete."""


@dataclass(frozen=True)
class PlannedField:
    """One field required or considered for the next replay campaign."""

    field_path: str
    domain: str
    definition: str
    raw_or_derived: str
    granularity: str
    time_scope: str
    collection_frequency: str
    unit: str
    reproducibility: str
    leakage_risk: str
    hypothesis_ids: str
    collection_cost_class: str
    priority: str
    priority_reason: str


DECISION_METADATA_COLUMNS = (
    "canonical_field_id",
    "definition",
    "definition_status",
    "raw_or_derived",
    "granularity",
    "time_scope",
    "collection_frequency",
    "unit",
    "reproducibility",
    "leakage_risk",
    "availability_phase",
    "hypothesis_ids",
    "collection_cost_class",
    "priority",
    "priority_reason",
    "gap_status",
    "storage_attribution",
)


def default_planned_fields() -> tuple[PlannedField, ...]:
    """Return the preregistration field contract, ordered by urgency."""

    p0 = "Required before confirmatory launch because it identifies the causal replay mechanism."
    guard = "Required to test whether normal replay harms the weak-defect tail."
    process = "Required to distinguish configured replay from realized optimization exposure."
    return (
        PlannedField(
            "no_replay_arm",
            "evaluation",
            "Arm with identical base data and initialization but zero replay exposure.",
            "RAW_RECORDED",
            "RUN",
            "FULL_TRAINING",
            "ONCE_PER_RUN",
            "arm_label",
            "HASH_VERIFIABLE",
            "NONE",
            "H1_REPLAY_CAUSAL_EFFECT;H4_SEED_SENSITIVITY",
            "LOW",
            "P0",
            p0,
        ),
        PlannedField(
            "blind_or_external_test",
            "evaluation",
            "Untouched sample split opened only after protocol and model-selection freeze.",
            "RAW_RECORDED",
            "DATASET_SPLIT",
            "FINAL_ONLY",
            "ONCE_PER_CAMPAIGN",
            "split_identity",
            "HASH_VERIFIABLE",
            "CRITICAL_DO_NOT_OPEN_EARLY",
            "H7_BLIND_GENERALIZATION",
            "MEDIUM",
            "P0",
            "Required to separate mechanism discovery from final generalization evidence.",
        ),
        PlannedField(
            "run_arm_schedule_manifest",
            "replay_schedule",
            "Machine-readable replay and guard weights for every epoch of one arm.",
            "RAW_RECORDED",
            "RUN_EPOCH",
            "PRETRAIN_DECLARED",
            "ONCE_PER_RUN",
            "weight",
            "HASH_VERIFIABLE",
            "NONE",
            "H2_DYNAMIC_REPLAY;H3_WEAK_DEFECT_GUARD",
            "LOW",
            "P0",
            process,
        ),
        PlannedField(
            "replay_weight_per_epoch",
            "replay_schedule",
            "Configured normal-replay multiplier applied at each training epoch.",
            "RAW_RECORDED",
            "RUN_EPOCH",
            "DURING_TRAINING",
            "EVERY_EPOCH",
            "weight",
            "DETERMINISTIC_FROM_ARM_CONTRACT",
            "NONE",
            "H2_DYNAMIC_REPLAY",
            "LOW",
            "P0",
            process,
        ),
        PlannedField(
            "guard_weight_per_epoch",
            "replay_schedule",
            "Configured weak-defect guard multiplier applied at each epoch.",
            "RAW_RECORDED",
            "RUN_EPOCH",
            "DURING_TRAINING",
            "EVERY_EPOCH",
            "weight",
            "DETERMINISTIC_FROM_ARM_CONTRACT",
            "NONE",
            "H3_WEAK_DEFECT_GUARD",
            "LOW",
            "P0",
            guard,
        ),
        PlannedField(
            "per_sample_replay_exposure_count",
            "realized_exposure",
            "Realized number of replay draws for each selected sample by epoch and cumulatively.",
            "RAW_RECORDED",
            "RUN_SAMPLE_EPOCH",
            "DURING_TRAINING",
            "EVERY_EPOCH",
            "draw_count",
            "REPLAYABLE_FROM_SAMPLER_TRACE",
            "TRAIN_ONLY",
            "H2_DYNAMIC_REPLAY;H4_SEED_SENSITIVITY",
            "MEDIUM",
            "P0",
            process,
        ),
        PlannedField(
            "base_vs_replay_loss_split",
            "loss_telemetry",
            "Separate aggregate loss for base samples, normal replay, and defect guard samples.",
            "RAW_RECORDED",
            "RUN_EPOCH_ROLE",
            "DURING_TRAINING",
            "EVERY_EPOCH",
            "loss",
            "DETERMINISTIC_FROM_BATCH_LOSS_LOG",
            "TRAIN_ONLY",
            "H2_DYNAMIC_REPLAY;H3_WEAK_DEFECT_GUARD",
            "LOW",
            "P0",
            process,
        ),
        PlannedField(
            "base_normal_loss_per_epoch",
            "loss_telemetry",
            "Mean loss of non-replayed normal base samples per epoch.",
            "DERIVED",
            "RUN_EPOCH_CLASS_ROLE",
            "DURING_TRAINING",
            "EVERY_EPOCH",
            "loss",
            "RECOMPUTABLE_FROM_ROLE_AGGREGATES",
            "TRAIN_ONLY",
            "H2_DYNAMIC_REPLAY",
            "LOW",
            "P0",
            process,
        ),
        PlannedField(
            "base_defect_loss_per_epoch",
            "loss_telemetry",
            "Mean loss of non-guard defect base samples per epoch.",
            "DERIVED",
            "RUN_EPOCH_CLASS_ROLE",
            "DURING_TRAINING",
            "EVERY_EPOCH",
            "loss",
            "RECOMPUTABLE_FROM_ROLE_AGGREGATES",
            "TRAIN_ONLY",
            "H3_WEAK_DEFECT_GUARD",
            "LOW",
            "P0",
            guard,
        ),
        PlannedField(
            "replay_normal_loss_per_epoch",
            "loss_telemetry",
            "Mean loss of selected normal replay draws per epoch.",
            "DERIVED",
            "RUN_EPOCH_CLASS_ROLE",
            "DURING_TRAINING",
            "EVERY_EPOCH",
            "loss",
            "RECOMPUTABLE_FROM_ROLE_AGGREGATES",
            "TRAIN_ONLY",
            "H2_DYNAMIC_REPLAY",
            "LOW",
            "P0",
            process,
        ),
        PlannedField(
            "guard_defect_loss_per_epoch",
            "loss_telemetry",
            "Mean loss of selected weak-defect guard draws per epoch.",
            "DERIVED",
            "RUN_EPOCH_CLASS_ROLE",
            "DURING_TRAINING",
            "EVERY_EPOCH",
            "loss",
            "RECOMPUTABLE_FROM_ROLE_AGGREGATES",
            "TRAIN_ONLY",
            "H3_WEAK_DEFECT_GUARD",
            "LOW",
            "P0",
            guard,
        ),
        PlannedField(
            "tail_probe_membership_manifest",
            "probe_sets",
            "Frozen difficult-normal and weak-defect probe membership with derivation provenance.",
            "RAW_RECORDED",
            "SAMPLE",
            "PRETRAIN_DECLARED",
            "ONCE_PER_CAMPAIGN",
            "membership",
            "HASH_VERIFIABLE",
            "TRAIN_OR_VALIDATION_ONLY",
            "H2_DYNAMIC_REPLAY;H3_WEAK_DEFECT_GUARD",
            "LOW",
            "P0",
            "Probe membership must remain fixed across arms and seeds.",
        ),
        PlannedField(
            "weak_defect_tail_score_trajectory",
            "probe_predictions",
            "Raw defect probabilities for the frozen weak-defect probe set at key epochs.",
            "OBSERVED_MODEL_OUTPUT",
            "RUN_SAMPLE_CHECKPOINT",
            "DURING_TRAINING",
            "KEY_EPOCHS_120_140_150_160_180_200",
            "probability",
            "RECOMPUTABLE_FROM_CHECKPOINT_AND_INPUT_HASH",
            "VALIDATION_ONLY",
            "H3_WEAK_DEFECT_GUARD;H4_SEED_SENSITIVITY",
            "MEDIUM",
            "P0",
            guard,
        ),
        PlannedField(
            "difficult_normal_tail_score_trajectory",
            "probe_predictions",
            "Raw defect probabilities for the frozen difficult-normal probe set at key epochs.",
            "OBSERVED_MODEL_OUTPUT",
            "RUN_SAMPLE_CHECKPOINT",
            "DURING_TRAINING",
            "KEY_EPOCHS_120_140_150_160_180_200",
            "probability",
            "RECOMPUTABLE_FROM_CHECKPOINT_AND_INPUT_HASH",
            "VALIDATION_ONLY",
            "H2_DYNAMIC_REPLAY;H4_SEED_SENSITIVITY",
            "MEDIUM",
            "P0",
            "Required to verify that dynamic replay actually lowers the target normal tail.",
        ),
        PlannedField(
            "key_epoch_val_op_raw_predictions",
            "probe_predictions",
            "Unrounded sample-level operational-validation scores at epochs 120, 140, 150, 160, 180, and 200.",
            "OBSERVED_MODEL_OUTPUT",
            "RUN_SAMPLE_CHECKPOINT",
            "DURING_TRAINING",
            "KEY_EPOCHS_120_140_150_160_180_200",
            "raw_score",
            "RECOMPUTABLE_FROM_CHECKPOINT_AND_INPUT_HASH",
            "VALIDATION_ONLY",
            "H2_DYNAMIC_REPLAY;H3_WEAK_DEFECT_GUARD;H4_SEED_SENSITIVITY",
            "HIGH",
            "P0",
            p0,
        ),
        PlannedField(
            "key_epoch_checkpoint",
            "checkpoints",
            "Full resumable model and optimizer state at epochs 120, 140, 150, 160, 180, and 200.",
            "RAW_MODEL_STATE",
            "RUN_CHECKPOINT",
            "DURING_TRAINING",
            "KEY_EPOCHS_120_140_150_160_180_200",
            "binary_state",
            "CONTENT_HASH_VERIFIABLE",
            "TRAIN_ONLY",
            "H2_DYNAMIC_REPLAY;H4_SEED_SENSITIVITY;H5_GRADIENT_ALIGNMENT",
            "VERY_HIGH",
            "P0",
            "Required for trajectory reconstruction, recovery, and key-epoch gradient probes.",
        ),
        PlannedField(
            "epoch_150_checkpoint",
            "checkpoints",
            "Full checkpoint at epoch 150, the center of the observed late-training divergence window.",
            "RAW_MODEL_STATE",
            "RUN_CHECKPOINT",
            "DURING_TRAINING",
            "ONCE_PER_RUN",
            "binary_state",
            "CONTENT_HASH_VERIFIABLE",
            "TRAIN_ONLY",
            "H2_DYNAMIC_REPLAY;H4_SEED_SENSITIVITY;H5_GRADIENT_ALIGNMENT",
            "VERY_HIGH",
            "P0",
            "The prior campaign did not retain the checkpoint needed to inspect the 150-160 epoch window.",
        ),
        PlannedField(
            "selected_checkpoint_and_threshold_provenance",
            "evaluation",
            "Rule, calibration split, checkpoint identity, and threshold used for every reported outcome.",
            "RAW_RECORDED",
            "RUN_EVALUATION",
            "POST_TRAINING",
            "ONCE_PER_EVALUATION",
            "provenance_record",
            "HASH_VERIFIABLE",
            "HIGH_IF_USED_BEFORE_FREEZE",
            "H7_BLIND_GENERALIZATION",
            "LOW",
            "P0",
            "Prevents implicit checkpoint or threshold cherry-picking.",
        ),
        PlannedField(
            "grad_mag_score",
            "gradients",
            "True last-layer per-sample gradient norm at preregistered key checkpoints.",
            "OBSERVED_GRADIENT",
            "RUN_SAMPLE_CHECKPOINT",
            "DURING_TRAINING",
            "PILOT_THEN_KEY_EPOCHS",
            "gradient_norm",
            "RECOMPUTABLE_FROM_CHECKPOINT_BATCH_AND_LOSS",
            "TRAIN_OR_VALIDATION_ONLY",
            "H5_GRADIENT_ALIGNMENT",
            "PILOT_REQUIRED",
            "P1",
            "High scientific value, but payload and runtime must be measured before broad collection.",
        ),
        PlannedField(
            "grad_align_normal_tail",
            "gradients",
            "Cosine and dot-product alignment to the difficult-normal target gradient.",
            "DERIVED",
            "RUN_SAMPLE_CHECKPOINT",
            "DURING_TRAINING",
            "PILOT_THEN_KEY_EPOCHS",
            "alignment",
            "RECOMPUTABLE_FROM_GRADIENT_EMBEDDINGS",
            "TRAIN_OR_VALIDATION_ONLY",
            "H5_GRADIENT_ALIGNMENT",
            "PILOT_REQUIRED",
            "P1",
            "Tests whether impact direction, rather than magnitude alone, predicts benefit.",
        ),
        PlannedField(
            "grad_align_defect_tail",
            "gradients",
            "Cosine and dot-product alignment to the weak-defect target gradient.",
            "DERIVED",
            "RUN_SAMPLE_CHECKPOINT",
            "DURING_TRAINING",
            "PILOT_THEN_KEY_EPOCHS",
            "alignment",
            "RECOMPUTABLE_FROM_GRADIENT_EMBEDDINGS",
            "TRAIN_OR_VALIDATION_ONLY",
            "H3_WEAK_DEFECT_GUARD;H5_GRADIENT_ALIGNMENT",
            "PILOT_REQUIRED",
            "P1",
            "Separates influential beneficial samples from influential harmful samples.",
        ),
        PlannedField(
            "gradient_diversity_embedding",
            "gradients",
            "Compressed last-layer gradient embedding used for diversity and redundancy analysis.",
            "OBSERVED_GRADIENT",
            "RUN_SAMPLE_CHECKPOINT",
            "DURING_TRAINING",
            "PILOT_THEN_KEY_EPOCHS",
            "embedding_vector",
            "RECOMPUTABLE_FROM_CHECKPOINT_BATCH_AND_LOSS",
            "TRAIN_OR_VALIDATION_ONLY",
            "H5_GRADIENT_ALIGNMENT;H6_DIVERSITY_REDUNDANCY",
            "PILOT_REQUIRED",
            "P1",
            "Needed only after a pilot fixes dimensionality, runtime, and compression.",
        ),
        PlannedField(
            "gradient_outlier_score",
            "gradients",
            "Robust distance of a sample gradient from its label and source-group neighborhood.",
            "DERIVED",
            "RUN_SAMPLE_CHECKPOINT",
            "POST_KEY_CHECKPOINT",
            "PILOT_THEN_KEY_EPOCHS",
            "robust_distance",
            "RECOMPUTABLE_FROM_GRADIENT_EMBEDDINGS",
            "TRAIN_OR_VALIDATION_ONLY",
            "H5_GRADIENT_ALIGNMENT;H6_DIVERSITY_REDUNDANCY",
            "LOW_AFTER_GRADIENT_EXPORT",
            "P1",
            "Candidate discriminator for hard-clean versus mislabeled or pathological samples.",
        ),
        PlannedField(
            "minibatch_order_digest_per_epoch",
            "realized_exposure",
            "Stable digest of the ordered sample and role sequence consumed in each epoch.",
            "RAW_RECORDED",
            "RUN_EPOCH",
            "DURING_TRAINING",
            "EVERY_EPOCH",
            "sha256",
            "HASH_VERIFIABLE",
            "TRAIN_ONLY",
            "H4_SEED_SENSITIVITY",
            "LOW",
            "P1",
            "Provides a compact seed-sensitive exposure trace without storing every batch verbatim.",
        ),
        PlannedField(
            "batch_role_composition_per_epoch",
            "realized_exposure",
            "Counts of base-normal, base-defect, replay-normal, and guard-defect draws per epoch.",
            "RAW_RECORDED",
            "RUN_EPOCH_ROLE",
            "DURING_TRAINING",
            "EVERY_EPOCH",
            "draw_count",
            "REPLAYABLE_FROM_SAMPLER_TRACE",
            "TRAIN_ONLY",
            "H2_DYNAMIC_REPLAY;H3_WEAK_DEFECT_GUARD;H4_SEED_SENSITIVITY",
            "LOW",
            "P1",
            "Makes realized class and replay pressure comparable across arms and seeds.",
        ),
        PlannedField(
            "augmentation_realization_digest_per_epoch",
            "augmentation",
            "Stable digest and aggregate counters for realized stochastic augmentation choices.",
            "RAW_RECORDED",
            "RUN_EPOCH",
            "DURING_TRAINING",
            "EVERY_EPOCH",
            "digest_and_counts",
            "HASH_VERIFIABLE",
            "TRAIN_ONLY",
            "H4_SEED_SENSITIVITY",
            "MEDIUM",
            "P1",
            "Tests whether augmentation realization contributes to same-selection seed reversal.",
        ),
        PlannedField(
            "optimizer_step_summary_per_epoch",
            "optimizer",
            "Optimizer-step count, skipped-step count, gradient-scale state, and norm summaries by epoch.",
            "RAW_RECORDED",
            "RUN_EPOCH",
            "DURING_TRAINING",
            "EVERY_EPOCH",
            "mixed",
            "HASH_VERIFIABLE",
            "TRAIN_ONLY",
            "H4_SEED_SENSITIVITY;H5_GRADIENT_ALIGNMENT",
            "LOW",
            "P1",
            "Rules out unequal update exposure and numerical instability as confounds.",
        ),
        PlannedField(
            "per_step_momentum_buffer_and_minibatch_order",
            "optimizer",
            "Full per-step optimizer state and minibatch order trace.",
            "RAW_RECORDED",
            "RUN_STEP_PARAMETER",
            "DURING_TRAINING",
            "EVERY_STEP",
            "binary_state",
            "REPLAYABLE_IF_COMPLETE",
            "TRAIN_ONLY",
            "H4_SEED_SENSITIVITY",
            "PROHIBITIVE",
            "P3",
            "Documented for completeness; compact digests and epoch summaries are preferred.",
        ),
        PlannedField(
            "sample_epoch_exposure_rng_realization",
            "augmentation",
            "Full per-sample augmentation and sampler random-number realization for every epoch.",
            "RAW_RECORDED",
            "RUN_SAMPLE_EPOCH",
            "DURING_TRAINING",
            "EVERY_DRAW",
            "rng_trace",
            "REPLAYABLE_IF_COMPLETE",
            "TRAIN_ONLY",
            "H4_SEED_SENSITIVITY",
            "PROHIBITIVE",
            "P3",
            "Documented for completeness; retain compact digests unless a focused pilot justifies expansion.",
        ),
        PlannedField(
            "per_epoch_checkpoint_trajectory",
            "checkpoints",
            "Full resumable checkpoint at all 200 epochs.",
            "RAW_MODEL_STATE",
            "RUN_CHECKPOINT",
            "DURING_TRAINING",
            "EVERY_EPOCH",
            "binary_state",
            "CONTENT_HASH_VERIFIABLE",
            "TRAIN_ONLY",
            "H2_DYNAMIC_REPLAY;H4_SEED_SENSITIVITY",
            "PROHIBITIVE",
            "P3",
            "Not retained in the main campaign because six preregistered checkpoints answer the hypothesis.",
        ),
        PlannedField(
            "per_epoch_val_op_predictions",
            "probe_predictions",
            "Full operational-validation prediction export at all 200 epochs.",
            "OBSERVED_MODEL_OUTPUT",
            "RUN_SAMPLE_CHECKPOINT",
            "DURING_TRAINING",
            "EVERY_EPOCH",
            "raw_score",
            "RECOMPUTABLE_FROM_CHECKPOINT_AND_INPUT_HASH",
            "VALIDATION_ONLY",
            "H2_DYNAMIC_REPLAY;H4_SEED_SENSITIVITY",
            "PROHIBITIVE",
            "P3",
            "Six key epochs and lightweight tail summaries are retained instead.",
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(value: Any, path: Path) -> None:
    _atomic_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        path,
    )


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise CampaignAuditError(f"{label} missing columns: {sorted(missing)}")


def _read_csv_header(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(csv.reader(handle), None)
    if not row:
        raise CampaignAuditError(f"CSV has no header: {path}")
    return tuple(row)


def _resolve_oof_manifest_path(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise CampaignAuditError(f"Unsafe OOF manifest path: {relative_path}")
    candidates = (root / "01_oof_dynamics" / relative, root / relative)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_relative_to(root) and resolved.is_file():
            return resolved
    return candidates[0].resolve()


def _table_field_rows(
    path: Path,
    *,
    normalized_path: str,
    logical_source: str,
    runs_observed: int,
    runs_expected: int,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CampaignAuditError(f"Missing OOF table: {path}")
    frame = pd.read_csv(path, low_memory=False)
    rows: list[dict[str, Any]] = []
    size = int(path.stat().st_size)
    for field in frame.columns:
        non_null = int(frame[field].notna().sum())
        rows.append(
            {
                "normalized_path": normalized_path,
                "field_path": str(field),
                "source_kind": "CSV",
                "logical_source": logical_source,
                "files_observed": 1,
                "runs_observed": runs_observed,
                "files_expected": 1,
                "runs_expected": runs_expected,
                "schema_variant_count": 1,
                "usage_role": (
                    "PAIRING_KEY"
                    if field in {"sample_id", "oof_fold", "epoch", "checkpoint_id"}
                    else "SCIENTIFIC_FEATURE"
                ),
                "usage_status": "ANALYZED",
                "rationale": f"Field from validated {logical_source}.",
                "files_profiled": 1,
                "runs_profiled": runs_observed,
                "values_observed": len(frame),
                "non_null_count": non_null,
                "null_count": int(len(frame) - non_null),
                "unique_count": int(frame[field].nunique(dropna=True)),
                "constant_state": (
                    "ALL_NULL"
                    if non_null == 0
                    else "CONSTANT"
                    if frame[field].nunique(dropna=True) <= 1
                    else "NON_CONSTANT"
                ),
                "derived_consumer": "campaign_field_audit",
                "silently_dropped": False,
                "path_total_bytes": size,
                "path_mean_bytes_per_file": size,
                "path_min_bytes": size,
                "path_max_bytes": size,
                "artifact_manifest_listed_rate": 1.0,
                "artifact_manifest_size_match_rate": 1.0,
                "artifact_sha256_present_rate": 1.0,
            }
        )
    return rows


def _json_top_level_rows(
    path: Path,
    *,
    normalized_path: str,
    logical_source: str,
    files_observed: int = 1,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise CampaignAuditError(f"Expected JSON object: {path}")
    size = int(path.stat().st_size)
    return [
        {
            "normalized_path": normalized_path,
            "field_path": str(field),
            "source_kind": "JSON",
            "logical_source": logical_source,
            "files_observed": files_observed,
            "runs_observed": files_observed,
            "files_expected": files_observed,
            "runs_expected": files_observed,
            "schema_variant_count": 1,
            "usage_role": "LINEAGE_VALIDATION",
            "usage_status": "AUDITED",
            "rationale": f"Validation metadata from {logical_source}.",
            "files_profiled": 1,
            "runs_profiled": files_observed,
            "values_observed": files_observed,
            "non_null_count": files_observed if value is not None else 0,
            "null_count": files_observed if value is None else 0,
            "unique_count": 1 if value is not None else 0,
            "constant_state": "CONSTANT" if value is not None else "ALL_NULL",
            "derived_consumer": "campaign_field_audit",
            "silently_dropped": False,
            "path_total_bytes": size,
            "path_mean_bytes_per_file": size,
            "path_min_bytes": size,
            "path_max_bytes": size,
            "artifact_manifest_listed_rate": 1.0,
            "artifact_manifest_size_match_rate": 1.0,
            "artifact_sha256_present_rate": 1.0,
        }
        for field, value in payload.items()
    ]


def discover_oof_field_inventory(oof_experiment_root: str | Path) -> pd.DataFrame:
    """Inventory the complete 10-fold OOF trajectory without rereading 24M rows."""

    root = Path(oof_experiment_root).resolve()
    summary = root / "01_oof_dynamics/summary"
    validation_path = summary / "summary_validation.json"
    input_manifest_path = summary / "summary_input_manifest.csv"
    if not validation_path.is_file() or not input_manifest_path.is_file():
        raise CampaignAuditError(f"Missing OOF validation or input manifest below {root}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    errors = []
    if validation.get("status") != "ok":
        errors.append(f"status={validation.get('status')}")
    for key in ("missing_pairs", "duplicate_pairs", "bad_epoch_count_samples", "bad_fold_samples"):
        if validation.get(key):
            errors.append(f"{key}={len(validation[key])}")
    if errors:
        raise CampaignAuditError("OOF validation is incomplete: " + ", ".join(errors))

    manifest = pd.read_csv(input_manifest_path, keep_default_na=False)
    _require_columns(manifest, {"relative_path", "row_count"}, "OOF input manifest")
    if manifest["relative_path"].duplicated().any():
        raise CampaignAuditError("OOF input manifest contains duplicate paths")
    expected_files = int(validation.get("input_file_count", len(manifest)))
    if len(manifest) != expected_files:
        raise CampaignAuditError(
            f"OOF input file count mismatch: manifest={len(manifest)} expected={expected_files}"
        )
    expected_rows = int(validation.get("total_prediction_rows", manifest["row_count"].sum()))
    if int(manifest["row_count"].sum()) != expected_rows:
        raise CampaignAuditError("OOF input row count differs from summary validation")

    schemas: set[tuple[str, ...]] = set()
    raw_sizes: list[int] = []
    sidecar_sizes: list[int] = []
    sidecar_keys: set[str] = set()
    for row in manifest.itertuples(index=False):
        path = _resolve_oof_manifest_path(root, str(row.relative_path))
        if not path.is_file():
            raise CampaignAuditError(f"Missing OOF prediction file: {path}")
        schemas.add(_read_csv_header(path))
        raw_sizes.append(int(path.stat().st_size))
        sidecar = path.with_name(path.name + ".manifest.json")
        if not sidecar.is_file():
            raise CampaignAuditError(f"Missing OOF prediction sidecar: {sidecar}")
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or int(payload.get("row_count", -1)) != int(
            row.row_count
        ):
            raise CampaignAuditError(f"Invalid OOF prediction sidecar: {sidecar}")
        sidecar_keys.update(map(str, payload.keys()))
        sidecar_sizes.append(int(sidecar.stat().st_size))
    if len(schemas) != 1:
        raise CampaignAuditError(f"OOF raw prediction schema variants: {len(schemas)}")

    fold_count = len(validation.get("expected_folds", [])) or 1
    rows: list[dict[str, Any]] = []
    raw_path = "01_oof_dynamics/jobs/*/fold_*/epoch_*_predictions.csv"
    raw_total = int(sum(raw_sizes))
    for field in next(iter(schemas)):
        rows.append(
            {
                "normalized_path": raw_path,
                "field_path": field,
                "source_kind": "CSV",
                "logical_source": "oof_raw_epoch_predictions",
                "files_observed": len(manifest),
                "runs_observed": fold_count,
                "files_expected": expected_files,
                "runs_expected": fold_count,
                "schema_variant_count": 1,
                "usage_role": (
                    "PAIRING_KEY"
                    if field in {"sample_id", "checkpoint_id", "oof_fold", "epoch"}
                    else "SCIENTIFIC_FEATURE"
                ),
                "usage_status": "ANALYZED",
                "rationale": "Validated OOF sample-by-epoch model output.",
                "files_profiled": 0,
                "runs_profiled": fold_count,
                "values_observed": expected_rows,
                "non_null_count": pd.NA,
                "null_count": pd.NA,
                "unique_count": pd.NA,
                "constant_state": "NOT_VALUE_PROFILED",
                "derived_consumer": "sample_dynamics_summary",
                "silently_dropped": False,
                "path_total_bytes": raw_total,
                "path_mean_bytes_per_file": raw_total / len(raw_sizes),
                "path_min_bytes": min(raw_sizes),
                "path_max_bytes": max(raw_sizes),
                "artifact_manifest_listed_rate": 1.0,
                "artifact_manifest_size_match_rate": 1.0,
                "artifact_sha256_present_rate": 0.0,
            }
        )
    sidecar_total = int(sum(sidecar_sizes))
    for field in sorted(sidecar_keys):
        rows.append(
            {
                "normalized_path": raw_path + ".manifest.json",
                "field_path": field,
                "source_kind": "JSON",
                "logical_source": "oof_raw_epoch_sidecars",
                "files_observed": len(sidecar_sizes),
                "runs_observed": fold_count,
                "files_expected": expected_files,
                "runs_expected": fold_count,
                "schema_variant_count": 1,
                "usage_role": "LINEAGE_VALIDATION",
                "usage_status": "AUDITED",
                "rationale": "Per-checkpoint OOF export completion evidence.",
                "files_profiled": len(sidecar_sizes),
                "runs_profiled": fold_count,
                "values_observed": len(sidecar_sizes),
                "non_null_count": pd.NA,
                "null_count": pd.NA,
                "unique_count": pd.NA,
                "constant_state": "NOT_VALUE_PROFILED",
                "derived_consumer": "oof_validation",
                "silently_dropped": False,
                "path_total_bytes": sidecar_total,
                "path_mean_bytes_per_file": sidecar_total / len(sidecar_sizes),
                "path_min_bytes": min(sidecar_sizes),
                "path_max_bytes": max(sidecar_sizes),
                "artifact_manifest_listed_rate": 1.0,
                "artifact_manifest_size_match_rate": 1.0,
                "artifact_sha256_present_rate": 0.0,
            }
        )

    rows.extend(
        _table_field_rows(
            summary / "sample_dynamics_summary.csv",
            normalized_path="01_oof_dynamics/summary/sample_dynamics_summary.csv",
            logical_source="oof_sample_dynamics_summary",
            runs_observed=fold_count,
            runs_expected=fold_count,
        )
    )
    rows.extend(
        _table_field_rows(
            root / "02_sample_value_tables/sample_value_table.csv",
            normalized_path="02_sample_value_tables/sample_value_table.csv",
            logical_source="oof_sample_value_table",
            runs_observed=fold_count,
            runs_expected=fold_count,
        )
    )
    rows.extend(
        _table_field_rows(
            input_manifest_path,
            normalized_path="01_oof_dynamics/summary/summary_input_manifest.csv",
            logical_source="oof_summary_input_manifest",
            runs_observed=fold_count,
            runs_expected=fold_count,
        )
    )
    rows.extend(
        _json_top_level_rows(
            validation_path,
            normalized_path="01_oof_dynamics/summary/summary_validation.json",
            logical_source="oof_summary_validation",
            files_observed=1,
        )
    )
    return pd.DataFrame.from_records(rows)


def _source_path_profiles(source: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        source,
        {"normalized_path", "size_bytes", "run_slot"},
        "Source file ledger",
    )
    if "canonical_attempt" in source and not source["canonical_attempt"].fillna(False).all():
        raise CampaignAuditError("Source file ledger contains non-canonical attempts")
    if source.duplicated(["run_slot", "normalized_path", "relative_path"]).any():
        raise CampaignAuditError("Source file ledger contains duplicate canonical paths")

    work = source.copy()
    for column in (
        "artifact_manifest_listed",
        "artifact_manifest_size_match",
    ):
        if column not in work:
            work[column] = pd.NA
    if "artifact_manifest_sha256" not in work:
        work["artifact_manifest_sha256"] = ""
    work["sha_present"] = work["artifact_manifest_sha256"].fillna("").astype(str).str.len().gt(0)
    return (
        work.groupby("normalized_path", as_index=False)
        .agg(
            path_total_bytes=("size_bytes", "sum"),
            path_mean_bytes_per_file=("size_bytes", "mean"),
            path_min_bytes=("size_bytes", "min"),
            path_max_bytes=("size_bytes", "max"),
            path_files_observed=("size_bytes", "size"),
            artifact_manifest_listed_rate=("artifact_manifest_listed", "mean"),
            artifact_manifest_size_match_rate=("artifact_manifest_size_match", "mean"),
            artifact_sha256_present_rate=("sha_present", "mean"),
        )
        .reset_index(drop=True)
    )


def _infer_granularity(path: str, field: str, logical_source: str) -> str:
    lower = path.lower()
    if logical_source == "oof_raw_epoch_predictions":
        return "SAMPLE_CHECKPOINT"
    if logical_source in {"oof_sample_dynamics_summary", "oof_sample_value_table"}:
        return "SAMPLE"
    if "epoch_training_metrics" in lower:
        return "RUN_EPOCH"
    if "selection_manifest" in lower:
        return "RUN_SAMPLE"
    if "predictions" in lower:
        return "RUN_SAMPLE_CHECKPOINT"
    if "threshold_sweep" in lower:
        return "RUN_THRESHOLD"
    if "checkpoint" in lower or lower.endswith(".pt"):
        return "RUN_CHECKPOINT"
    if "gpu_usage" in lower:
        return "RUN_TIMEPOINT"
    if "identity" in lower or "validation" in lower or "metrics" in lower:
        return "RUN"
    if field in {"sample_id", "filename", "image_path"}:
        return "SAMPLE"
    return "FILE_OR_RUN"


def _infer_time_scope(path: str, field: str, logical_source: str) -> str:
    lower = f"{path}/{field}".lower()
    if "blind" in lower:
        return "FINAL_ONLY"
    if logical_source == "oof_raw_epoch_predictions" or "epoch_training" in lower:
        return "DURING_TRAINING"
    if any(token in lower for token in ("prediction", "threshold", "metrics")):
        return "POST_CHECKPOINT_EVALUATION"
    if any(token in lower for token in ("manifest", "identity", "args", "environment")):
        return "PRETRAIN_OR_RUNTIME_DECLARED"
    if "checkpoint" in lower:
        return "DURING_TRAINING"
    return "RUN_LIFECYCLE"


def _infer_frequency(granularity: str, logical_source: str) -> str:
    if logical_source == "oof_raw_epoch_predictions":
        return "EVERY_OOF_EPOCH"
    return {
        "RUN_EPOCH": "EVERY_EPOCH",
        "RUN_TIMEPOINT": "PERIODIC",
        "RUN_SAMPLE_CHECKPOINT": "PER_EVALUATED_CHECKPOINT",
        "RUN_CHECKPOINT": "PER_RETAINED_CHECKPOINT",
        "SAMPLE_CHECKPOINT": "PER_EVALUATED_CHECKPOINT",
        "RUN_THRESHOLD": "PER_THRESHOLD_GRID_POINT",
    }.get(granularity, "ONCE_PER_SOURCE_ARTIFACT")


def _infer_unit(field: str) -> str:
    lower = field.lower()
    if any(token in lower for token in ("sha", "digest", "hash")):
        return "sha256_or_digest"
    if any(token in lower for token in ("path", "filename", "sample_id", "run_id", "checkpoint_id")):
        return "identifier"
    if "byte" in lower:
        return "bytes"
    if any(token in lower for token in ("duration", "elapsed", "time")):
        return "seconds_or_timestamp"
    if "lr" in lower or "learning_rate" in lower:
        return "learning_rate"
    if any(token in lower for token in ("loss", "cross_entropy")):
        return "loss"
    if any(token in lower for token in ("p_defect", "p_normal", "probability")):
        return "probability"
    if any(token in lower for token in ("accuracy", "rate", "pct", "correct")):
        return "rate_or_boolean"
    if any(token in lower for token in ("epoch", "count", "rank", "step", "pid")):
        return "count_or_index"
    if any(token in lower for token in ("score", "margin", "uncertainty", "gap")):
        return "model_score"
    if lower in {"y_true", "label", "target", "class"}:
        return "class_label"
    return "unitless_or_categorical"


def _infer_raw_or_derived(path: str, role: str, logical_source: str) -> str:
    lower = path.lower()
    if role == "RECOMPUTED_FIELD" or any(
        token in lower for token in ("summary", "sample_value", "metrics", "threshold_sweep")
    ):
        return "DERIVED"
    if "predictions" in lower or logical_source == "oof_raw_epoch_predictions":
        return "OBSERVED_MODEL_OUTPUT"
    if "checkpoint" in lower or lower.endswith(".pt"):
        return "RAW_MODEL_STATE"
    return "RAW_RECORDED"


def _infer_reproducibility(path: str, raw_or_derived: str) -> str:
    lower = path.lower()
    if raw_or_derived == "DERIVED":
        return "RECOMPUTABLE_FROM_HASHED_INPUTS"
    if raw_or_derived in {"OBSERVED_MODEL_OUTPUT", "RAW_MODEL_STATE"}:
        return "REQUIRES_CHECKPOINT_AND_INPUT_HASH"
    if any(token in lower for token in ("manifest", "identity", "validation")):
        return "HASH_VERIFIABLE"
    return "OBSERVED_NOT_FULLY_REPLAYABLE"


def _infer_leakage_risk(path: str, field: str) -> str:
    lower = f"{path}/{field}".lower()
    if "blind" in lower or "external_test" in lower:
        return "CRITICAL_DO_NOT_USE_FOR_SELECTION"
    if "val_op" in lower or "outcome" in lower or "threshold" in lower:
        return "EVALUATION_ONLY_DO_NOT_SELECT_TRAINING_SAMPLES"
    if "oof" in lower:
        return "TRAIN_OOF_ALLOWED_WITH_FOLD_ISOLATION"
    return "NONE_OR_TRAIN_ONLY"


def _infer_hypotheses(path: str, field: str, role: str) -> str:
    lower = f"{path}/{field}".lower()
    hypotheses: list[str] = []
    if any(token in lower for token in ("replay", "loss", "epoch", "prediction")):
        hypotheses.append("H2_DYNAMIC_REPLAY")
    if any(token in lower for token in ("defect", "fn", "guard", "low_tail")):
        hypotheses.append("H3_WEAK_DEFECT_GUARD")
    if any(token in lower for token in ("seed", "environment", "gpu", "optimizer", "augment")):
        hypotheses.append("H4_SEED_SENSITIVITY")
    if "grad" in lower:
        hypotheses.append("H5_GRADIENT_ALIGNMENT")
    if any(token in lower for token in ("group", "cluster", "divers", "video")):
        hypotheses.append("H6_DIVERSITY_REDUNDANCY")
    if role in {"OUTCOME_METRIC", "PAIRING_KEY"}:
        hypotheses.append("H1_REPLAY_CAUSAL_EFFECT")
    return ";".join(dict.fromkeys(hypotheses)) or "H0_LINEAGE_AND_REPRODUCIBILITY"


def _infer_priority(path: str, field: str, role: str, collection_state: str) -> tuple[str, str]:
    lower = f"{path}/{field}".lower()
    if collection_state in {"NOT_COLLECTED", "MISSING", "PRESENT_ALL_NULL"}:
        if any(token in lower for token in ("gradient", "grad_")):
            return "P1", "Unmeasured mechanism candidate; collect only after a runtime and payload pilot."
        return "P2", "Documented historical gap that is not part of the confirmatory P0 contract."
    if role in {"OUTCOME_METRIC", "PAIRING_KEY"} or any(
        token in lower
        for token in (
            "epoch_training_metrics",
            "oof_raw_epoch_predictions",
            "sample_dynamics_summary",
            "training_seed",
            "data_seed",
            "selection_manifest",
            "val_op_predictions",
        )
    ):
        return "P0", "Required to pair runs, reconstruct trajectories, or compute preregistered outcomes."
    if role in {"CONFOUND_CONTROL", "LINEAGE_VALIDATION", "SCIENTIFIC_FEATURE"}:
        return "P1", "Needed for confound control, mechanism analysis, or reproducibility."
    if role in {"CONSTANT_PARAMETER", "RECOMPUTED_FIELD"}:
        return "P2", "Retain or recompute for auditability; not a primary mechanism field."
    return "P3", "Descriptive or low-leverage field retained only when collection is effectively free."


def _infer_cost(path: str, granularity: str) -> str:
    lower = path.lower()
    if "checkpoint" in lower or lower.endswith(".pt"):
        return "VERY_HIGH"
    if "gradient" in lower or "grad_" in lower:
        return "PILOT_REQUIRED"
    if granularity in {"RUN_SAMPLE_CHECKPOINT", "SAMPLE_CHECKPOINT"}:
        return "HIGH"
    if granularity in {"RUN_SAMPLE", "SAMPLE"}:
        return "MEDIUM"
    return "LOW"


def _availability_phase(time_scope: str) -> str:
    if time_scope == "FINAL_ONLY":
        return "AFTER_PROTOCOL_FREEZE"
    if "PRETRAIN" in time_scope:
        return "BEFORE_TRAINING"
    if "DURING" in time_scope:
        return "DURING_TRAINING"
    if "POST" in time_scope:
        return "AFTER_CHECKPOINT"
    return "RUN_LIFECYCLE"


def _coverage_rate(observed: Any, expected: Any) -> float:
    try:
        expected_value = float(expected)
        observed_value = float(observed)
    except (TypeError, ValueError):
        return 0.0
    if expected_value <= 0:
        return 0.0 if observed_value <= 0 else 1.0
    return max(0.0, min(1.0, observed_value / expected_value))


def _collection_state(row: Mapping[str, Any]) -> str:
    role = str(row.get("usage_role", ""))
    if role == "NOT_COLLECTED_FIELD" or str(row.get("source_kind", "")) == "NOT_COLLECTED":
        return "NOT_COLLECTED"
    if role == "MISSING_FIELD" or str(row.get("source_kind", "")) == "MISSING_FROM_ATTEMPT":
        return "MISSING"
    values = pd.to_numeric(pd.Series([row.get("values_observed")]), errors="coerce").iloc[0]
    non_null = pd.to_numeric(pd.Series([row.get("non_null_count")]), errors="coerce").iloc[0]
    if pd.notna(values) and values > 0 and pd.notna(non_null) and non_null == 0:
        return "PRESENT_ALL_NULL"
    coverage = _coverage_rate(row.get("runs_observed"), row.get("runs_expected"))
    return "COLLECTED" if coverage >= 1.0 else "PARTIAL"


def _metadata_row(row: Mapping[str, Any]) -> dict[str, Any]:
    path = str(row.get("normalized_path", ""))
    field = str(row.get("field_path", ""))
    logical_source = str(row.get("logical_source", "canonical_240run_evidence"))
    role = str(row.get("usage_role", "UNREVIEWED"))
    state = _collection_state(row)
    granularity = _infer_granularity(path, field, logical_source)
    time_scope = _infer_time_scope(path, field, logical_source)
    raw_or_derived = _infer_raw_or_derived(path, role, logical_source)
    priority, priority_reason = _infer_priority(path, field, role, state)
    values = pd.to_numeric(pd.Series([row.get("values_observed")]), errors="coerce").iloc[0]
    nulls = pd.to_numeric(pd.Series([row.get("null_count")]), errors="coerce").iloc[0]
    value_null_rate = (
        float(nulls / values) if pd.notna(values) and values > 0 and pd.notna(nulls) else pd.NA
    )
    coverage = _coverage_rate(row.get("runs_observed"), row.get("runs_expected"))
    if state == "COLLECTED":
        gap_status = "SUFFICIENT_CURRENT"
    elif state == "PARTIAL":
        gap_status = "PARTIAL"
    elif state == "PRESENT_ALL_NULL":
        gap_status = "NO_USABLE_VALUES"
    elif priority == "P0":
        gap_status = "NOT_COLLECTED_REQUIRED" if state == "NOT_COLLECTED" else "MISSING_REQUIRED"
    else:
        gap_status = "OPTIONAL_OR_HISTORICAL_GAP"
    rationale = str(row.get("rationale", "")).strip()
    definition = f"Observed field '{field}' in '{path}'."
    if rationale:
        definition += f" {rationale}"
    result = dict(row)
    result.update(
        {
            "logical_source": logical_source,
            "canonical_field_id": f"{logical_source}::{path}::{field}",
            "definition": definition,
            "definition_status": "SOURCE_RATIONALE",
            "collection_state": state,
            "raw_or_derived": raw_or_derived,
            "granularity": granularity,
            "time_scope": time_scope,
            "collection_frequency": _infer_frequency(granularity, logical_source),
            "coverage_rate": coverage,
            "missing_rate": 1.0 - coverage,
            "value_null_rate": value_null_rate,
            "null_profile_status": (
                "PROFILED" if pd.notna(value_null_rate) else "UNKNOWN_NOT_PROFILED"
            ),
            "unit": _infer_unit(field),
            "reproducibility": _infer_reproducibility(path, raw_or_derived),
            "leakage_risk": _infer_leakage_risk(path, field),
            "availability_phase": _availability_phase(time_scope),
            "hypothesis_ids": _infer_hypotheses(path, field, role),
            "collection_cost_class": _infer_cost(path, granularity),
            "priority": priority,
            "priority_reason": priority_reason,
            "gap_status": gap_status,
            "storage_attribution": (
                "PATH_LEVEL_REPEAT_ON_FIELDS_DO_NOT_SUM"
                if pd.notna(row.get("path_total_bytes", pd.NA))
                else "NO_STORED_ARTIFACT"
            ),
        }
    )
    return result


def _planned_row(spec: PlannedField) -> dict[str, Any]:
    data = asdict(spec)
    path = f"<PLANNED>/{spec.domain}"
    data.update(
        {
            "normalized_path": path,
            "logical_source": "next_campaign_field_contract",
            "source_kind": "PLANNED",
            "canonical_field_id": f"planned::{spec.field_path}",
            "definition_status": "PLANNED_CONTRACT",
            "collection_state": "NOT_COLLECTED",
            "files_observed": 0,
            "runs_observed": 0,
            "files_expected": 0,
            "runs_expected": 0,
            "schema_variant_count": 0,
            "usage_role": "NOT_COLLECTED_FIELD",
            "usage_status": "NOT_TESTABLE",
            "rationale": spec.priority_reason,
            "files_profiled": 0,
            "runs_profiled": 0,
            "values_observed": 0,
            "non_null_count": 0,
            "null_count": 0,
            "unique_count": 0,
            "constant_state": "NO_VALUES",
            "derived_consumer": "next_campaign",
            "silently_dropped": False,
            "coverage_rate": 0.0,
            "missing_rate": 1.0,
            "value_null_rate": pd.NA,
            "null_profile_status": "NOT_COLLECTED",
            "availability_phase": _availability_phase(spec.time_scope),
            "gap_status": (
                "NOT_COLLECTED_REQUIRED"
                if spec.priority == "P0"
                else "OPTIONAL_OR_PILOT_GAP"
            ),
            "path_total_bytes": pd.NA,
            "path_mean_bytes_per_file": pd.NA,
            "path_min_bytes": pd.NA,
            "path_max_bytes": pd.NA,
            "artifact_manifest_listed_rate": pd.NA,
            "artifact_manifest_size_match_rate": pd.NA,
            "artifact_sha256_present_rate": pd.NA,
            "storage_attribution": "FORECAST_SEPARATELY",
        }
    )
    return data


def build_campaign_field_inventory(
    refined_ledger: pd.DataFrame,
    source_file_ledger: pd.DataFrame,
    *,
    supplementary_fields: pd.DataFrame | None = None,
    planned_fields: Sequence[PlannedField] | None = None,
) -> pd.DataFrame:
    """Combine historical fields, OOF fields, and the next-campaign contract."""

    _require_columns(
        refined_ledger,
        {
            "normalized_path",
            "field_path",
            "source_kind",
            "runs_observed",
            "runs_expected",
            "usage_role",
            "usage_status",
            "rationale",
        },
        "Refined field ledger",
    )
    if refined_ledger.duplicated(["normalized_path", "field_path"]).any():
        raise CampaignAuditError("Refined field ledger contains duplicate identities")
    profiles = _source_path_profiles(source_file_ledger)
    historical = refined_ledger.copy()
    historical["logical_source"] = "canonical_240run_evidence"
    historical = historical.merge(profiles, on="normalized_path", how="left")

    records = [_metadata_row(row) for row in historical.to_dict(orient="records")]
    if supplementary_fields is not None:
        _require_columns(
            supplementary_fields,
            {"normalized_path", "field_path", "logical_source", "source_kind"},
            "Supplementary field ledger",
        )
        records.extend(
            _metadata_row(row) for row in supplementary_fields.to_dict(orient="records")
        )
    specs = tuple(planned_fields) if planned_fields is not None else default_planned_fields()
    records.extend(_planned_row(spec) for spec in specs)
    inventory = pd.DataFrame.from_records(records)
    preferred = [
        "canonical_field_id",
        "logical_source",
        "normalized_path",
        "field_path",
        "definition",
        "definition_status",
        "collection_state",
        "raw_or_derived",
        "granularity",
        "time_scope",
        "collection_frequency",
        "files_observed",
        "runs_observed",
        "files_expected",
        "runs_expected",
        "coverage_rate",
        "missing_rate",
        "values_observed",
        "non_null_count",
        "null_count",
        "value_null_rate",
        "null_profile_status",
        "unit",
        "reproducibility",
        "leakage_risk",
        "availability_phase",
        "hypothesis_ids",
        "collection_cost_class",
        "priority",
        "priority_reason",
        "gap_status",
        "path_total_bytes",
        "path_mean_bytes_per_file",
        "path_min_bytes",
        "path_max_bytes",
        "artifact_manifest_listed_rate",
        "artifact_manifest_size_match_rate",
        "artifact_sha256_present_rate",
        "storage_attribution",
        "usage_role",
        "usage_status",
        "rationale",
    ]
    remaining = [column for column in inventory.columns if column not in preferred]
    inventory = inventory[[column for column in preferred if column in inventory] + remaining]
    inventory = inventory.sort_values(
        ["priority", "gap_status", "logical_source", "normalized_path", "field_path"],
        kind="stable",
    ).reset_index(drop=True)
    assert_campaign_field_inventory_complete(inventory)
    return inventory


def assert_campaign_field_inventory_complete(inventory: pd.DataFrame) -> dict[str, int]:
    """Fail if any field lacks the metadata needed for a collection decision."""

    _require_columns(
        inventory,
        {"normalized_path", "field_path", *DECISION_METADATA_COLUMNS},
        "Campaign field inventory",
    )
    duplicate = int(inventory["canonical_field_id"].duplicated().sum())
    required = list(DECISION_METADATA_COLUMNS)
    missing_metadata = int(
        inventory[required]
        .apply(lambda column: column.isna() | column.astype(str).str.strip().eq(""))
        .any(axis=1)
        .sum()
    )
    invalid_priority = int((~inventory["priority"].isin(["P0", "P1", "P2", "P3"])).sum())
    invalid_coverage = int(
        (
            pd.to_numeric(inventory["coverage_rate"], errors="coerce").isna()
            | ~pd.to_numeric(inventory["coverage_rate"], errors="coerce").between(0, 1)
        ).sum()
    )
    gates = {
        "duplicate_field_identities": duplicate,
        "missing_decision_metadata": missing_metadata,
        "invalid_priority": invalid_priority,
        "invalid_coverage_rate": invalid_coverage,
    }
    if any(gates.values()):
        raise CampaignAuditError(
            "Campaign field inventory failed gates: "
            + ", ".join(f"{key}={value}" for key, value in gates.items())
        )
    return gates


def _mean_path_bytes(source: pd.DataFrame, normalized_path: str) -> int:
    values = source.loc[source["normalized_path"] == normalized_path, "size_bytes"]
    if values.empty:
        raise CampaignAuditError(f"Reference artifact missing from source ledger: {normalized_path}")
    return int(round(float(values.mean())))


def build_data_volume_forecast(
    source_file_ledger: pd.DataFrame,
    *,
    seed_counts: Sequence[int] = (14, 22, 30),
    arm_count: int = 6,
    key_checkpoint_epochs: Sequence[int] = (120, 140, 150, 160, 180, 200),
    checkpoint_reference_path: str = "03_checkpoints/last.pt",
    prediction_reference_path: str = "04_predictions/val_op_predictions.csv",
) -> pd.DataFrame:
    """Forecast retained bytes without multiplying shared inputs by field count."""

    _require_columns(
        source_file_ledger,
        {"run_slot", "normalized_path", "size_bytes"},
        "Source file ledger",
    )
    if arm_count <= 0 or not seed_counts or any(int(seed) <= 0 for seed in seed_counts):
        raise CampaignAuditError("Arm and seed counts must be positive")
    source = source_file_ledger.copy()
    source["size_bytes"] = pd.to_numeric(source["size_bytes"], errors="raise")
    run_count_observed = source["run_slot"].nunique()
    if run_count_observed <= 0:
        raise CampaignAuditError("Source ledger has no runs")

    path_mean = source.groupby("normalized_path")["size_bytes"].mean()
    legacy_per_run = int(round(float(source["size_bytes"].sum() / run_count_observed)))
    shared_mask = path_mean.index.to_series().str.startswith("01_manifests/frozen_inputs/")
    shared_once = int(round(float(path_mean[shared_mask].sum())))
    checkpoint_bytes = _mean_path_bytes(source, checkpoint_reference_path)
    prediction_bytes = _mean_path_bytes(source, prediction_reference_path)

    excluded_prefixes = ("01_manifests/frozen_inputs/", "03_checkpoints/", "04_predictions/")
    excluded_exact = {"05_metrics/threshold_sweep.csv"}
    core_paths = [
        path
        for path in path_mean.index
        if not path.startswith(excluded_prefixes) and path not in excluded_exact
    ]
    core_per_run = int(round(float(path_mean.loc[core_paths].sum())))
    key_count = len(tuple(key_checkpoint_epochs))
    projected_per_run = core_per_run + key_count * checkpoint_bytes + key_count * prediction_bytes

    records = []
    for seed_count in seed_counts:
        runs = int(seed_count) * int(arm_count)
        projected = shared_once + runs * projected_per_run
        records.append(
            {
                "scenario_id": f"{arm_count}arms_x_{int(seed_count)}seeds",
                "seed_count": int(seed_count),
                "arm_count": int(arm_count),
                "run_count": runs,
                "key_checkpoint_epochs": ";".join(map(str, key_checkpoint_epochs)),
                "key_checkpoint_count": key_count,
                "raw_prediction_export_count": key_count,
                "observed_legacy_bytes_per_run": legacy_per_run,
                "shared_immutable_bytes_once": shared_once,
                "shared_immutable_bytes_avoided": shared_once * (runs - 1),
                "run_specific_core_bytes_per_run": core_per_run,
                "checkpoint_reference_bytes": checkpoint_bytes,
                "prediction_reference_bytes": prediction_bytes,
                "projected_run_specific_bytes_per_run": projected_per_run,
                "projected_campaign_bytes": projected,
                "projected_campaign_gib": projected / (2**30),
                "legacy_mirror_campaign_bytes": legacy_per_run * runs,
                "legacy_mirror_campaign_gib": legacy_per_run * runs / (2**30),
                "gradient_payload_status": "UNMEASURED_PILOT_REQUIRED",
                "gradient_payload_bytes": pd.NA,
                "estimate_class": "LOWER_BOUND_EXCLUDES_GRADIENT_PAYLOAD",
                "storage_accounting_rule": "PATHS_COUNTED_ONCE_FIELDS_NEVER_SUMMED",
            }
        )
    return pd.DataFrame.from_records(records)


def _render_gap_report(inventory: pd.DataFrame, forecast: pd.DataFrame) -> str:
    p0 = inventory[
        (inventory["priority"] == "P0")
        & inventory["gap_status"].isin(
            ["MISSING_REQUIRED", "NOT_COLLECTED_REQUIRED", "NO_USABLE_VALUES"]
        )
    ].copy()
    p1 = inventory[
        (inventory["priority"] == "P1")
        & inventory["gap_status"].isin(
            ["MISSING_REQUIRED", "NOT_COLLECTED_REQUIRED", "NO_USABLE_VALUES", "OPTIONAL_OR_PILOT_GAP"]
        )
    ].copy()
    collected = int(inventory["collection_state"].eq("COLLECTED").sum())
    logical_sources = inventory["logical_source"].nunique()
    lines = [
        "# Stage1 campaign field-gap report",
        "",
        "## Decision",
        "",
        "The current evidence is broad but not sufficient to explain same-selection seed reversals causally. ",
        "The 10-fold OOF trajectory is complete for static sample dynamics, while the replay runs lack ",
        "realized exposure, role-separated loss, key-epoch raw predictions, a no-replay arm, and blind evaluation.",
        "",
        "## Inventory scope",
        "",
        f"- Field rows: {len(inventory):,}",
        f"- Logical sources: {logical_sources:,}",
        f"- Currently collected rows: {collected:,}",
        f"- P0 unresolved rows: {len(p0):,}",
        f"- P1 pilot or unresolved rows: {len(p1):,}",
        "",
        "A row is a source-field identity, not a unique physical file. Storage must be summed by path only.",
        "",
        "## P0 gaps before confirmatory launch",
        "",
        "| Field | Granularity | Why required |",
        "|---|---|---|",
    ]
    for row in p0.sort_values("field_path").itertuples(index=False):
        reason = str(row.priority_reason).replace("|", "/")
        lines.append(f"| `{row.field_path}` | {row.granularity} | {reason} |")
    lines.extend(
        [
            "",
            "## P1 pilot fields",
            "",
            "P1 fields are scientifically useful but must not delay the P0 six-arm pilot. Gradient payloads remain ",
            "unmeasured and therefore require a one-checkpoint benchmark before fleet-wide collection.",
            "",
            "| Field | Cost class | Hypothesis |",
            "|---|---|---|",
        ]
    )
    for row in p1.sort_values("field_path").itertuples(index=False):
        lines.append(f"| `{row.field_path}` | {row.collection_cost_class} | {row.hypothesis_ids} |")
    lines.extend(
        [
            "",
            "## Storage lower bounds",
            "",
            "The forecast excludes gradient payloads until the pilot measures dimensions, compression, and runtime.",
            "",
            "| Scenario | Runs | Projected GiB | Estimate class |",
            "|---|---:|---:|---|",
        ]
    )
    for row in forecast.itertuples(index=False):
        lines.append(
            f"| {row.scenario_id} | {row.run_count} | {row.projected_campaign_gib:.2f} | {row.estimate_class} |"
        )
    lines.extend(
        [
            "",
            "## Highest-value investigation order",
            "",
            "1. Establish causal replay effect with NR_NO_REPLAY and matched seeds.",
            "2. Test dynamic replay decay while measuring realized replay exposure.",
            "3. Test weak-defect guard protection on fixed tail probes at key epochs.",
            "4. Quantify seed reversal through exposure, loss, prediction, and optimizer trajectories.",
            "5. Pilot true last-layer gradient direction and outlier metrics; scale only if predictive.",
            "6. Freeze the protocol, then open the blind holdout once.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_retention_policy(forecast: pd.DataFrame) -> str:
    largest = forecast.sort_values("run_count").iloc[-1]
    return "\n".join(
        [
            "# Stage1 dynamic replay retention policy",
            "",
            "## Keep for every run",
            "",
            "- Identity, environment, resolved configuration, input hashes, arm schedule, and sampler seed.",
            "- Lightweight per-epoch metrics for all 200 epochs, including role-separated losses and exposure counts.",
            "- Key checkpoints at epochs 120, 140, 150, 160, 180, 200.",
            "- Raw val_op predictions at epochs 120, 140, 150, 160, 180, 200.",
            "- Tail-probe trajectories, optimizer-step summaries, and compact batch/augmentation digests.",
            "- Completion sidecars, row counts, SHA-256 manifests, and failure/retry lineage.",
            "",
            "## Shared immutable assets",
            "",
            "Store each frozen base split, probe manifest, and selection manifest once in a content-addressed shared immutable area. ",
            "Run folders contain only hashes and relative references. Never copy the same 120k-row manifest into every run.",
            "",
            "## Pilot-gated heavy fields",
            "",
            "True per-sample gradients are retained only for the preregistered candidate/probe subset and key checkpoints after ",
            "a pilot measures runtime, vector dimension, compression ratio, and predictive utility. Full-model gradients and ",
            "200-epoch checkpoints are not default campaign outputs.",
            "",
            "## Recomputable outputs",
            "",
            "Threshold sweeps, plots, HTML, and aggregate tables are recomputed from raw predictions and are not duplicated per run.",
            "Text logs are compressed after successful postflight validation.",
            "",
            "## Upload and deletion gate",
            "",
            "A run is deletable from a worker only after: postflight passes, the artifact manifest is complete, remote upload succeeds, ",
            "remote SHA-256 verification succeeds, and the central index records the remote location. Failed or partial runs remain quarantined.",
            "",
            f"Largest current lower-bound scenario: {int(largest.run_count)} runs, {largest.projected_campaign_gib:.2f} GiB, ",
            "excluding the gradient payload pending pilot measurement.",
            "",
        ]
    )


def _lineage_input(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _lineage_output(path: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(output_dir).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def publish_campaign_field_audit(
    *,
    refined_ledger_path: str | Path,
    source_file_ledger_path: str | Path,
    oof_experiment_root: str | Path,
    output_dir: str | Path,
    campaign_id: str,
    seed_counts: Sequence[int] = (14, 22, 30),
    arm_count: int = 6,
) -> dict[str, Any]:
    """Publish the complete field/capacity audit with atomic writes and lineage."""

    refined_path = Path(refined_ledger_path).resolve()
    source_path = Path(source_file_ledger_path).resolve()
    oof_root = Path(oof_experiment_root).resolve()
    output = Path(output_dir).resolve()
    for path in (refined_path, source_path):
        if not path.is_file():
            raise CampaignAuditError(f"Missing audit input: {path}")
    output.mkdir(parents=True, exist_ok=True)
    refined = pd.read_csv(refined_path, low_memory=False)
    source = pd.read_csv(source_path, low_memory=False)
    supplementary = discover_oof_field_inventory(oof_root)
    inventory = build_campaign_field_inventory(
        refined, source, supplementary_fields=supplementary
    )
    gates = assert_campaign_field_inventory_complete(inventory)
    forecast = build_data_volume_forecast(
        source, seed_counts=seed_counts, arm_count=arm_count
    )

    inventory_path = output / "FIELD_INVENTORY.csv"
    gap_path = output / "FIELD_GAP_REPORT.md"
    forecast_path = output / "DATA_VOLUME_FORECAST.csv"
    retention_path = output / "RETENTION_POLICY.md"
    lineage_path = output / "DATA_LINEAGE.json"
    validation_path = output / "AUDIT_VALIDATION.json"
    _atomic_csv(inventory, inventory_path)
    _atomic_text(_render_gap_report(inventory, forecast), gap_path)
    _atomic_csv(forecast, forecast_path)
    _atomic_text(_render_retention_policy(forecast), retention_path)

    oof_summary = oof_root / "01_oof_dynamics/summary"
    input_paths = [
        (refined_path, "canonical_240run_refined_field_ledger"),
        (source_path, "canonical_240run_source_file_ledger"),
        (oof_summary / "summary_validation.json", "oof_completeness_validation"),
        (oof_summary / "summary_input_manifest.csv", "oof_raw_prediction_index"),
        (oof_summary / "sample_dynamics_summary.csv", "oof_sample_trajectory_summary"),
        (
            oof_root / "02_sample_value_tables/sample_value_table.csv",
            "oof_sample_value_features",
        ),
    ]
    core_outputs = [inventory_path, gap_path, forecast_path, retention_path]
    lineage = {
        "schema_version": "1.0",
        "campaign_id": campaign_id,
        "inputs": [_lineage_input(path, role) for path, role in input_paths],
        "bulk_sources": [
            {
                "role": "oof_raw_epoch_predictions",
                "integrity_basis": "validated input manifest plus per-file completion sidecars",
                "content_hash_status": "NOT_RECORDED_IN_LEGACY_EXPORT",
                "field_inventory_rows": int(
                    supplementary["logical_source"].eq("oof_raw_epoch_predictions").sum()
                ),
            }
        ],
        "transforms": [
            "discover_oof_field_inventory",
            "build_campaign_field_inventory",
            "build_data_volume_forecast",
        ],
        "outputs": [_lineage_output(path, output) for path in core_outputs],
    }
    _atomic_json(lineage, lineage_path)
    validation = {
        "status": "complete",
        "campaign_id": campaign_id,
        "field_rows": len(inventory),
        "logical_sources": int(inventory["logical_source"].nunique()),
        "p0_gap_rows": int(
            (
                inventory["priority"].eq("P0")
                & inventory["gap_status"].isin(
                    ["MISSING_REQUIRED", "NOT_COLLECTED_REQUIRED", "NO_USABLE_VALUES"]
                )
            ).sum()
        ),
        "forecast_scenarios": len(forecast),
        "inventory_gates": gates,
        "artifacts": [
            _lineage_output(path, output)
            for path in [*core_outputs, lineage_path]
        ],
    }
    _atomic_json(validation, validation_path)
    return validation
