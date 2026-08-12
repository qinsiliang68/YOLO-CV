"""Leakage-safe joint-success prediction across all 80 canonical triads."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .grouped_statistics import (
    minimum_confirmations,
    run_candidate_validation,
    validate_frozen_split,
)


class JointPredictionValidationError(ValueError):
    """Raised when unified inputs violate the no-leakage validation contract."""


@dataclass(frozen=True)
class JointPredictionValidationResult:
    predictions: pd.DataFrame
    summaries: pd.DataFrame
    fold_details: pd.DataFrame
    feature_selection_frequency: pd.DataFrame
    cutoff_feature_counts: pd.DataFrame
    confirmation_requirements: pd.DataFrame
    contract: dict[str, object]


_FORBIDDEN_ROLES = {
    "OUTCOME",
    "OUTCOME_AUDIT",
    "EXECUTION_CONFOUND",
    "DESIGN_CONFOUND",
    "SELECTION_DIGEST",
    "DESCRIPTIVE_NOT_PREDICTOR",
    "TRIAD_IDENTITY",
}


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _validate_registries(
    matrix: pd.DataFrame,
    feature_registry: pd.DataFrame,
    role_registry: pd.DataFrame,
) -> None:
    for name, frame, required in (
        (
            "feature_registry",
            feature_registry,
            {"feature", "available_epoch", "allowed_as_predictor"},
        ),
        (
            "role_registry",
            role_registry,
            {"feature", "allowed_as_predictor", "analysis_role"},
        ),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise JointPredictionValidationError(f"{name} missing {sorted(missing)}")
        if frame.columns.duplicated().any() or frame["feature"].astype(str).duplicated().any():
            raise JointPredictionValidationError(f"{name} contains duplicate columns/features")
        if set(frame["feature"].astype(str)) != set(matrix.columns.astype(str)):
            raise JointPredictionValidationError(f"{name} does not exactly cover matrix columns")

    feature_allowed = feature_registry.set_index("feature")["allowed_as_predictor"].map(_as_bool)
    role_indexed = role_registry.set_index("feature")
    role_allowed = role_indexed["allowed_as_predictor"].map(_as_bool)
    if not feature_allowed.sort_index().equals(role_allowed.sort_index()):
        raise JointPredictionValidationError(
            "feature and role registries disagree on outcome/confound predictor status"
        )
    invalid = role_indexed.loc[
        role_allowed
        & role_indexed["analysis_role"].astype(str).isin(_FORBIDDEN_ROLES)
    ]
    if len(invalid):
        raise JointPredictionValidationError(
            "outcome/confound fields are marked as predictors: "
            + ",".join(map(str, invalid.index[:5]))
        )


def _validate_joint_target(matrix: pd.DataFrame) -> None:
    required = {
        "delta_TN_R1",
        "delta_FN_R1",
        "delta_TN_R2",
        "delta_FN_R2",
        "dual_improvement",
        "dual_harm",
        "exclusive_cohort",
    }
    missing = required - set(matrix.columns)
    if missing:
        raise JointPredictionValidationError(
            f"Unified matrix missing joint-label fields: {sorted(missing)}"
        )
    recomputed = (
        pd.to_numeric(matrix["delta_TN_R1"], errors="raise").gt(0)
        & pd.to_numeric(matrix["delta_TN_R2"], errors="raise").gt(0)
        & pd.to_numeric(matrix["delta_FN_R1"], errors="raise").le(0)
        & pd.to_numeric(matrix["delta_FN_R2"], errors="raise").le(0)
    )
    recorded = matrix["dual_improvement"].map(_as_bool)
    if not recomputed.equals(recorded):
        bad = matrix.loc[recomputed.ne(recorded), "triad_id"].astype(str).tolist()
        raise JointPredictionValidationError(
            f"Recorded joint label does not mean simultaneous R1/R2 victory: {bad[:5]}"
        )


def _feature_frequency(fold_details: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    keys = ["cutoff", "validation_scheme"]
    for key_values, group in fold_details.groupby(keys, sort=True):
        cutoff, scheme = key_values
        counts: dict[str, int] = {}
        for value in group["selected_features"].fillna("").astype(str):
            for feature in filter(None, value.split(";")):
                counts[feature] = counts.get(feature, 0) + 1
        for feature, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            records.append(
                {
                    "cutoff": int(cutoff),
                    "validation_scheme": str(scheme),
                    "feature": feature,
                    "selected_fold_count": int(count),
                    "fold_count": int(len(group)),
                    "selected_fold_rate": float(count / len(group)),
                }
            )
    return pd.DataFrame(
        records,
        columns=[
            "cutoff",
            "validation_scheme",
            "feature",
            "selected_fold_count",
            "fold_count",
            "selected_fold_rate",
        ],
    )


def run_joint_prediction_validation(
    matrix: pd.DataFrame,
    feature_registry: pd.DataFrame,
    role_registry: pd.DataFrame,
    *,
    cutoffs: Sequence[int] = (120, 140, 150, 160, 180, 200),
    max_features: int = 16,
    target_precision: float = 0.8,
    max_harmful_rate: float = 0.2,
    include_digest_diagnostics: bool = True,
    random_state: int = 20_260_806,
) -> JointPredictionValidationResult:
    """Cross-fit an honest candidate screen at each preregistered cutoff."""

    if not 1 <= int(max_features) <= 16:
        raise ValueError("max_features must be between 1 and 16")
    cutoffs = tuple(int(value) for value in cutoffs)
    if not cutoffs or len(set(cutoffs)) != len(cutoffs) or any(value < 0 for value in cutoffs):
        raise ValueError("cutoffs must be unique non-negative epochs")
    if matrix.columns.duplicated().any() or matrix["triad_id"].astype(str).duplicated().any():
        raise JointPredictionValidationError("Unified matrix has duplicate columns/triads")
    _validate_registries(matrix, feature_registry, role_registry)
    _validate_joint_target(matrix)

    candidate = matrix.copy()
    if "treatment_sample_set_digest" not in candidate:
        raise JointPredictionValidationError("Unified matrix lacks treatment selection digest")
    candidate["selection_digest"] = candidate["treatment_sample_set_digest"].astype(str)
    split = validate_frozen_split(candidate)
    baseline = {
        "all_triads": int(len(candidate)),
        "dual_improvement": int(candidate["dual_improvement"].map(_as_bool).sum()),
        "dual_harm": int(candidate["dual_harm"].map(_as_bool).sum()),
        "mixed_or_reversal": int(candidate["exclusive_cohort"].eq("MIXED_OR_REVERSAL").sum()),
    }
    cohort_map = candidate.set_index("triad_id")["exclusive_cohort"].astype(str)

    prediction_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []
    cutoff_rows: list[dict[str, object]] = []
    registry_allowed = feature_registry["allowed_as_predictor"].map(_as_bool)
    available_epoch = pd.to_numeric(feature_registry["available_epoch"], errors="raise")
    for cutoff_index, cutoff in enumerate(cutoffs):
        eligible = feature_registry.loc[
            registry_allowed & available_epoch.le(cutoff), "feature"
        ].astype(str)
        if len(eligible) == 0:
            raise JointPredictionValidationError(
                f"No eligible predictor is available by cutoff {cutoff}"
            )
        validation = run_candidate_validation(
            candidate,
            feature_registry,
            cutoff=cutoff,
            target_column="dual_improvement",
            harmful_column="dual_harm",
            max_features=max_features,
            target_precision=target_precision,
            max_harmful_rate=max_harmful_rate,
            include_digest_diagnostics=include_digest_diagnostics,
            random_state=random_state + cutoff_index * 1_000_000,
        )
        predictions = validation.predictions.copy()
        predictions.insert(0, "cutoff", cutoff)
        predictions["exclusive_cohort"] = predictions["triad_id"].map(cohort_map)
        if predictions["exclusive_cohort"].isna().any():
            raise JointPredictionValidationError("Prediction rows lost outcome cohort identity")
        prediction_frames.append(predictions)

        summaries = validation.summaries.copy()
        summaries.insert(0, "cutoff", cutoff)
        summaries["eligible_predictor_count"] = len(eligible)
        summaries["max_features_per_fold"] = max_features
        summaries["selected_success_rate"] = summaries["precision"]
        summaries["observed_success_rate"] = summaries["positive_n"] / summaries["n"]
        summaries["all_mixed_included"] = True
        summary_frames.append(summaries)

        details = validation.fold_details.copy()
        details.insert(0, "cutoff", cutoff)
        details["eligible_predictor_count"] = len(eligible)
        if not details["selected_feature_count"].le(max_features).all():
            raise JointPredictionValidationError(
                f"Cutoff {cutoff} exceeded fold-local max_features={max_features}"
            )
        detail_frames.append(details)
        cutoff_rows.append(
            {
                "cutoff": cutoff,
                "eligible_predictor_count": len(eligible),
                "max_features_per_fold": max_features,
                "training_population": 75,
                "external_population": 5,
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    summaries = pd.concat(summary_frames, ignore_index=True)
    fold_details = pd.concat(detail_frames, ignore_index=True)
    main_schemes = {"DISCOVERY_LOSO_SEED", "PHASE_C_EXTERNAL_FALSIFICATION"}
    for cutoff, group in predictions.loc[
        predictions["validation_scheme"].isin(main_schemes)
    ].groupby("cutoff"):
        if len(group) != 80 or group["triad_id"].nunique() != 80:
            raise JointPredictionValidationError(
                f"Cutoff {cutoff} did not evaluate all 80 triads including mixed"
            )

    confirmations = minimum_confirmations(
        target_rate=target_precision,
        alpha=0.05,
        max_failures=2,
    )
    contract: dict[str, object] = {
        "analysis_population": "ALL_80_TRIADS_INCLUDING_MIXED",
        "independent_unit": "TRIAD_T_R1_R2",
        "joint_success_definition": (
            "delta_TN_R1>0 and delta_TN_R2>0 and "
            "delta_FN_R1<=0 and delta_FN_R2<=0"
        ),
        "baseline_counts": baseline,
        "split": split,
        "cutoffs": list(cutoffs),
        "max_features": int(max_features),
        "target_precision": float(target_precision),
        "max_harmful_rate": float(max_harmful_rate),
        "feature_selection": "FOLD_LOCAL_SELECT_K_BEST_ONLY",
        "threshold_selection": "INNER_TRAINING_SEED_OOF_ONLY",
        "discovery_validation": (
            "LOSO_TRAINING_SEED_PLUS_SELECTION_DIGEST_AND_DOUBLE_EXCLUSION_DIAGNOSTICS"
            if include_digest_diagnostics
            else "LOSO_TRAINING_SEED"
        ),
        "phase_c_role": "FIVE_UNSEEN_SEEDS_EXTERNAL_FALSIFICATION_ONLY",
        "auc_interpretation": "DESCRIPTIVE_DISCRIMINATION_NOT_80_PERCENT_SUCCESS",
        "confirmation_rule": (
            "ONE_SIDED_95_PERCENT_CLOPPER_PEARSON_LOWER_BOUND_GREATER_THAN_0.8"
        ),
        "source_sha256": {},
    }
    return JointPredictionValidationResult(
        predictions=predictions.sort_values(
            ["cutoff", "validation_scheme", "triad_id"], ignore_index=True
        ),
        summaries=summaries.sort_values(
            ["cutoff", "validation_scheme"], ignore_index=True
        ),
        fold_details=fold_details.sort_values(
            ["cutoff", "validation_scheme", "outer_group"], ignore_index=True
        ),
        feature_selection_frequency=_feature_frequency(fold_details),
        cutoff_feature_counts=pd.DataFrame(cutoff_rows),
        confirmation_requirements=confirmations,
        contract=contract,
    )


def _file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _frame_sha(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return sha256(payload).hexdigest().upper()


def _stage_bytes(payload: bytes, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def publish_joint_prediction_validation(
    result: JointPredictionValidationResult,
    output_dir: str | Path,
    *,
    source_paths: Mapping[str, str | Path] | None = None,
    source_frames: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, int]:
    """Atomically publish independent validation tables without state mutation."""

    output = Path(output_dir).resolve()
    if not output.name.endswith(".inprogress"):
        raise ValueError("Joint prediction output must remain .inprogress")
    tables = output / "tables"
    names = {
        "joint_prediction_predictions.csv": result.predictions,
        "joint_prediction_summaries.csv": result.summaries,
        "joint_prediction_fold_details.csv": result.fold_details,
        "joint_prediction_feature_selection_frequency.csv": result.feature_selection_frequency,
        "joint_prediction_cutoff_feature_counts.csv": result.cutoff_feature_counts,
        "joint_prediction_80pct_confirmation_requirements.csv": result.confirmation_requirements,
    }
    contract_name = "joint_prediction_validation_contract.json"
    manifest_name = "joint_prediction_output_manifest.csv"
    all_targets = [tables / name for name in (*names, contract_name, manifest_name)]
    existing = [path for path in all_targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite joint prediction outputs: {existing}")

    source_hashes: dict[str, str] = {}
    source_frames = dict(source_frames or {})
    for label, path_value in (source_paths or {}).items():
        path = Path(path_value)
        if path.is_file():
            source_hashes[str(label)] = _file_sha(path)
        elif label in source_frames:
            source_hashes[str(label)] = _frame_sha(source_frames[label])
        else:
            raise FileNotFoundError(path)
    contract = {**result.contract, "source_sha256": source_hashes}

    payloads: dict[str, tuple[bytes, int]] = {
        name: (
            frame.to_csv(index=False, lineterminator="\n").encode("utf-8"),
            len(frame),
        )
        for name, frame in names.items()
    }
    contract_payload = json.dumps(
        contract, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")
    payloads[contract_name] = (contract_payload, 1)
    manifest_rows = [
        {
            "filename": name,
            "sha256": sha256(payload).hexdigest().upper(),
            "size_bytes": len(payload),
            "row_count": row_count,
        }
        for name, (payload, row_count) in payloads.items()
    ]
    manifest = pd.DataFrame(manifest_rows).sort_values("filename", ignore_index=True)
    manifest_payload = manifest.to_csv(index=False, lineterminator="\n").encode("utf-8")
    payloads[manifest_name] = (manifest_payload, len(manifest))

    staged: dict[Path, Path] = {}
    try:
        for name, (payload, _) in payloads.items():
            target = tables / name
            staged[target] = _stage_bytes(payload, target)
        for target, temporary in staged.items():
            os.replace(temporary, target)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
    return {
        "output_files": len(payloads),
        "prediction_rows": len(result.predictions),
        "summary_rows": len(result.summaries),
        "fold_detail_rows": len(result.fold_details),
        "manifest_rows": len(manifest),
    }
