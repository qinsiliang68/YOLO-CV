"""Read-only sample-selection and OOF-dynamics mechanism analysis.

This module treats the frozen selection manifests as intervention identity.  It
never re-ranks or samples candidates.  Every manifest is checked against both
the frozen selection index and the canonical run inventory before its rows are
joined to the frozen sample-value, sample-dynamics, and master-index tables.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class SelectionMechanismError(ValueError):
    """Raised when frozen selection evidence is incomplete or inconsistent."""


GRADIENT_FIELDS = (
    "grad_mag_score",
    "grad_align_score",
    "grad_mag_align_score",
    "diverse_grad_align_score",
    "grad_align_guard_score",
)


_VALUE_FIELD_ROLES = {
    "sample_id": ("IDENTITY", "join_and_selection_identity"),
    "y_true": ("LABEL", "class_composition_and_identity_check"),
    "oof_fold": ("COMPOSITION", "fold_balance"),
    "dynamic_bucket": ("COMPOSITION", "training_dynamics_bucket"),
    "epoch_count": ("COVERAGE", "oof_epoch_coverage"),
    "mean_p_defect": ("DYNAMIC_FEATURE", "numeric_selection_summary"),
    "std_p_defect": ("DYNAMIC_FEATURE", "numeric_selection_summary"),
    "p_defect_start": ("DYNAMIC_FEATURE", "numeric_selection_summary"),
    "p_defect_end": ("DYNAMIC_FEATURE", "numeric_selection_summary"),
    "p_defect_trend": ("DYNAMIC_FEATURE", "numeric_selection_summary"),
    "correct_rate": ("LEARNABILITY", "numeric_and_persistence_summary"),
    "forgetting_count": ("LEARNABILITY", "numeric_and_persistence_summary"),
    "mean_loss": ("DYNAMIC_FEATURE", "numeric_selection_summary"),
    "mean_abs_margin": ("DYNAMIC_FEATURE", "numeric_selection_summary"),
    "first_learned_epoch": ("LEARNABILITY", "numeric_selection_summary"),
    "last_wrong_epoch": ("LEARNABILITY", "late_persistence_indicator"),
    "mean_p_good_gap_epochs": ("GAP_FEATURE", "numeric_selection_summary"),
    "mean_p_bad_gap_epochs": ("GAP_FEATURE", "numeric_selection_summary"),
    "gap_delta_raw": ("GAP_FEATURE", "numeric_selection_summary"),
    "confidence_fp_score": ("RANKING_FEATURE", "numeric_selection_summary"),
    "boundary_fp_score": ("RANKING_FEATURE", "numeric_selection_summary"),
    "persistent_fp_score": ("RANKING_FEATURE", "persistence_summary"),
    "learnable_hard_fp_score": ("RANKING_FEATURE", "numeric_selection_summary"),
    "gap_critical_score": ("RANKING_FEATURE", "numeric_selection_summary"),
    "gap_guard_score": ("RANKING_FEATURE", "numeric_selection_summary"),
    "gap_critical_guard_score": ("RANKING_FEATURE", "numeric_selection_summary"),
    **{
        field: ("NOT_COLLECTED", "gradient_evidence_unavailable")
        for field in GRADIENT_FIELDS
    },
}

_DYNAMICS_FIELD_ROLES = {
    "sample_id": ("IDENTITY", "join_identity"),
    "y_true": ("LABEL", "identity_check"),
    "oof_fold": ("COMPOSITION", "fold_identity_check"),
    "epoch_count": ("COVERAGE", "oof_epoch_coverage"),
    "first_epoch": ("COVERAGE", "oof_epoch_range"),
    "last_epoch": ("COVERAGE", "oof_epoch_range"),
    "p_defect_start": ("DYNAMIC_FEATURE", "numeric_selection_summary"),
    "p_defect_end": ("DYNAMIC_FEATURE", "numeric_selection_summary"),
    "p_defect_trend": ("DYNAMIC_FEATURE", "numeric_selection_summary"),
    "mean_p_defect": ("DYNAMIC_FEATURE", "numeric_selection_summary"),
    "std_p_defect": ("DYNAMIC_FEATURE", "numeric_selection_summary"),
    "min_p_defect": ("DYNAMIC_FEATURE", "numeric_selection_summary"),
    "max_p_defect": ("DYNAMIC_FEATURE", "numeric_selection_summary"),
    "mean_loss": ("DYNAMIC_FEATURE", "numeric_selection_summary"),
    "loss_auc_mean": ("DYNAMIC_FEATURE", "numeric_selection_summary"),
    "max_loss": ("DYNAMIC_FEATURE", "numeric_selection_summary"),
    "mean_margin_signed": ("DYNAMIC_FEATURE", "numeric_selection_summary"),
    "mean_abs_margin": ("DYNAMIC_FEATURE", "numeric_selection_summary"),
    "min_margin_signed": ("DYNAMIC_FEATURE", "numeric_selection_summary"),
    "correct_count": ("LEARNABILITY", "numeric_selection_summary"),
    "correct_rate": ("LEARNABILITY", "numeric_and_persistence_summary"),
    "final_correct": ("LEARNABILITY", "late_persistence_indicator"),
    "first_learned_epoch": ("LEARNABILITY", "numeric_selection_summary"),
    "last_wrong_epoch": ("LEARNABILITY", "late_persistence_indicator"),
    "forgetting_count": ("LEARNABILITY", "numeric_and_persistence_summary"),
    "dynamic_bucket": ("COMPOSITION", "training_dynamics_bucket"),
}

_MASTER_FIELD_ROLES = {
    "sample_id": ("IDENTITY", "join_identity"),
    "canonical_image_relpath": ("IDENTITY", "canonical_path_check"),
    "y_true": ("LABEL", "identity_check"),
    "oof_fold": ("COMPOSITION", "fold_identity_check"),
    "oof_group_id": ("COMPOSITION", "group_diversity"),
    "oof_group_source": ("LINEAGE", "group_provenance"),
    "train_primary_class": ("COMPOSITION", "class_composition"),
    "source_manifest": ("LINEAGE", "sample_provenance"),
    "source_csv_path": ("LINEAGE", "sample_provenance"),
    "source_csv_row_number": ("LINEAGE", "sample_provenance"),
    "filename": ("IDENTITY", "filename_audit"),
}

_SELECTION_FIELD_ROLES = {
    "run_slot": ("RUN_IDENTITY", "canonical_run_join"),
    "triad_id": ("RUN_IDENTITY", "triad_pairing"),
    "condition_id": ("SCIENCE_CONFIG", "condition_identity"),
    "arm": ("SCIENCE_CONFIG", "treatment_control_identity"),
    "training_seed": ("SCIENCE_CONFIG", "seed_stratification"),
    "selection_seed": ("SCIENCE_CONFIG", "selection_randomization_audit"),
    "rank": ("SELECTION_IDENTITY", "role_local_rank_audit"),
    "sample_id": ("SELECTION_IDENTITY", "selected_sample_identity"),
    "y_true": ("LABEL", "class_and_role_check"),
    "oof_fold": ("MIRRORED_FEATURE", "frozen_feature_consistency_check"),
    "dynamic_bucket": ("MIRRORED_FEATURE", "frozen_feature_consistency_check"),
    "mean_p_defect": ("MIRRORED_FEATURE", "frozen_feature_consistency_check"),
    "correct_rate": ("MIRRORED_FEATURE", "frozen_feature_consistency_check"),
    "std_p_defect": ("MIRRORED_FEATURE", "frozen_feature_consistency_check"),
    "replay_role": ("SELECTION_IDENTITY", "normal_vs_defect_replay"),
    "source_method": ("SCIENCE_CONFIG", "selection_method_provenance"),
}

_NUMERIC_FEATURES = tuple(
    dict.fromkeys(
        [
            field
            for field, (role, _) in (*_VALUE_FIELD_ROLES.items(), *_DYNAMICS_FIELD_ROLES.items())
            if role in {"COVERAGE", "DYNAMIC_FEATURE", "LEARNABILITY", "GAP_FEATURE", "RANKING_FEATURE"}
            and field not in GRADIENT_FIELDS
        ]
    )
)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise SelectionMechanismError(f"{label} missing columns: {missing}")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _set_digest(frame: pd.DataFrame) -> str:
    records = sorted(
        (str(role), str(sample_id))
        for role, sample_id in frame[["replay_role", "sample_id"]].itertuples(index=False, name=None)
    )
    payload = "".join(f"{role}\t{sample_id}\n" for role, sample_id in records)
    return sha256(payload.encode("utf-8")).hexdigest().upper()


def build_field_usage_registry(
    *,
    value_columns: Sequence[str],
    dynamics_columns: Sequence[str],
    master_columns: Sequence[str],
    selection_columns: Sequence[str],
) -> pd.DataFrame:
    """Classify every available source field; unknown fields fail closed."""

    sources = (
        ("sample_value_table", value_columns, _VALUE_FIELD_ROLES),
        ("sample_dynamics_summary", dynamics_columns, _DYNAMICS_FIELD_ROLES),
        ("master_sample_index", master_columns, _MASTER_FIELD_ROLES),
        ("selection_manifest", selection_columns, _SELECTION_FIELD_ROLES),
    )
    unknown: list[str] = []
    records: list[dict[str, str]] = []
    for source, columns, mapping in sources:
        for field in columns:
            if field not in mapping:
                unknown.append(f"{source}.{field}")
                continue
            role, consumer = mapping[field]
            records.append(
                {
                    "source_table": source,
                    "field_name": field,
                    "availability": "NOT_COLLECTED" if role == "NOT_COLLECTED" else "COLLECTED",
                    "analysis_role": role,
                    "analysis_consumer": consumer,
                    "silently_dropped": "NO",
                }
            )
    if unknown:
        raise SelectionMechanismError(f"Unclassified fields: {sorted(unknown)}")
    return pd.DataFrame(records)


def _assert_same_values(
    left: pd.Series, right: pd.Series, *, field: str, left_label: str, right_label: str
) -> None:
    numeric_left = pd.to_numeric(left, errors="coerce")
    numeric_right = pd.to_numeric(right, errors="coerce")
    numeric_candidate = bool(numeric_left.notna().any() or numeric_right.notna().any())
    if numeric_candidate:
        equal = np.isclose(
            numeric_left.to_numpy(dtype=float),
            numeric_right.to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-10,
            equal_nan=True,
        )
    else:
        equal = left.fillna("<NA>").astype(str).to_numpy() == right.fillna("<NA>").astype(str).to_numpy()
    if not bool(np.all(equal)):
        raise SelectionMechanismError(
            f"{field} differs between {left_label} and {right_label}"
        )


def load_frozen_sample_features(
    value_table_path: str | Path,
    dynamics_path: str | Path,
    master_index_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and cross-check the three frozen sample-level truth tables."""

    paths = [Path(value_table_path), Path(dynamics_path), Path(master_index_path)]
    for path in paths:
        if not path.is_file():
            raise SelectionMechanismError(f"Missing frozen sample table: {path}")
    value = pd.read_csv(paths[0], dtype={"sample_id": "string", "oof_fold": "string"})
    dynamics = pd.read_csv(paths[1], dtype={"sample_id": "string", "oof_fold": "string"})
    master = pd.read_csv(
        paths[2],
        dtype={"sample_id": "string", "oof_fold": "string", "oof_group_id": "string"},
    )
    registry = build_field_usage_registry(
        value_columns=list(value.columns),
        dynamics_columns=list(dynamics.columns),
        master_columns=list(master.columns),
        selection_columns=list(_SELECTION_FIELD_ROLES),
    )
    for label, frame in (("sample value", value), ("sample dynamics", dynamics), ("master index", master)):
        _require_columns(frame, ["sample_id", "y_true", "oof_fold"], label=label)
        if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise SelectionMechanismError(f"{label} sample IDs must be unique and non-null")
    for field in GRADIENT_FIELDS:
        if field in value and value[field].notna().any():
            raise SelectionMechanismError(
                f"{field}: gradient fields are declared not collected but contain values"
            )
    expected_ids = set(value["sample_id"].astype(str))
    if set(dynamics["sample_id"].astype(str)) != expected_ids or set(master["sample_id"].astype(str)) != expected_ids:
        raise SelectionMechanismError("Frozen sample tables contain different sample ID sets")

    dynamics = dynamics.set_index("sample_id").loc[value["sample_id"]].reset_index()
    master = master.set_index("sample_id").loc[value["sample_id"]].reset_index()
    common_dynamics = sorted((set(value.columns) & set(dynamics.columns)) - {"sample_id"})
    common_master = sorted((set(value.columns) & set(master.columns)) - {"sample_id"})
    for field in common_dynamics:
        _assert_same_values(value[field], dynamics[field], field=field, left_label="value", right_label="dynamics")
    for field in common_master:
        _assert_same_values(value[field], master[field], field=field, left_label="value", right_label="master")

    features = value.copy()
    for field in dynamics.columns:
        if field not in features:
            features[field] = dynamics[field]
    for field in master.columns:
        if field not in features:
            features[field] = master[field]
    return features, registry


def _check_manifest_mirrors(manifest: pd.DataFrame, merged: pd.DataFrame, run_slot: str) -> None:
    for field in ("y_true", "oof_fold", "dynamic_bucket", "mean_p_defect", "correct_rate", "std_p_defect"):
        _assert_same_values(
            merged[f"{field}_selection"],
            merged[f"{field}_feature"],
            field=f"{run_slot}.{field}",
            left_label="selection",
            right_label="sample features",
        )
    labels = pd.to_numeric(manifest["y_true"], errors="raise").astype(int)
    roles = manifest["replay_role"].astype(str)
    if not ((roles.eq("normal_replay") & labels.eq(0)) | (roles.eq("defect_guard") & labels.eq(1))).all():
        raise SelectionMechanismError(f"{run_slot} replay role conflicts with y_true")


def verify_and_load_selections(
    canonical_runs: pd.DataFrame,
    selection_index_path: str | Path,
    selection_root: str | Path,
    sample_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify selection SHA/provenance and return fully enriched selected rows."""

    run_required = {
        "run_slot", "triad_id", "condition_id", "method", "budget", "arm",
        "training_seed", "selection_seed", "phase", "guard_ratio",
    }
    _require_columns(canonical_runs, run_required, label="canonical runs")
    if canonical_runs["run_slot"].astype(str).duplicated().any():
        raise SelectionMechanismError("Canonical run slots must be unique")
    index = pd.read_csv(selection_index_path, dtype={"run_slot": "string", "sha256": "string"})
    _require_columns(index, ["run_slot", "sha256"], label="selection index")
    if index["run_slot"].isna().any() or index["run_slot"].duplicated().any():
        raise SelectionMechanismError("Selection index run slots must be unique and non-null")
    expected_slots = set(canonical_runs["run_slot"].astype(str))
    if set(index["run_slot"].astype(str)) != expected_slots:
        raise SelectionMechanismError("Selection index does not exactly cover canonical runs")
    if "selection_sha256" in canonical_runs:
        canonical_hash = canonical_runs.set_index(canonical_runs["run_slot"].astype(str))["selection_sha256"]
    else:
        canonical_hash = None
    indexed_hash = index.set_index(index["run_slot"].astype(str))["sha256"]
    feature_lookup = sample_features.set_index("sample_id", drop=False)
    if feature_lookup.index.has_duplicates:
        raise SelectionMechanismError("Sample feature IDs must be unique")

    root = Path(selection_root)
    all_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for run in canonical_runs.sort_values("run_slot", kind="stable").to_dict("records"):
        run_slot = str(run["run_slot"])
        path = root / run_slot / "selection_manifest.csv"
        if not path.is_file():
            raise SelectionMechanismError(f"Missing selection manifest: {path}")
        actual_sha = _sha256_file(path)
        expected_sha = str(indexed_hash.loc[run_slot]).upper()
        if actual_sha != expected_sha:
            raise SelectionMechanismError(
                f"SHA-256 mismatch for {run_slot}: {actual_sha} != {expected_sha}"
            )
        if canonical_hash is not None and str(canonical_hash.loc[run_slot]).upper() != actual_sha:
            raise SelectionMechanismError(f"Canonical selection SHA mismatch for {run_slot}")
        manifest = pd.read_csv(path, dtype={"sample_id": "string", "oof_fold": "string"})
        _require_columns(manifest, _SELECTION_FIELD_ROLES, label=f"{run_slot} selection")
        build_field_usage_registry(
            value_columns=[],
            dynamics_columns=[],
            master_columns=[],
            selection_columns=list(manifest.columns),
        )
        if len(manifest) != int(run["budget"]):
            raise SelectionMechanismError(
                f"{run_slot} budget mismatch: expected={int(run['budget'])}, actual={len(manifest)}"
            )
        if manifest["sample_id"].isna().any() or manifest["sample_id"].duplicated().any():
            raise SelectionMechanismError(f"{run_slot} sample IDs must be unique and non-null")
        for field in ("run_slot", "triad_id", "condition_id", "arm", "training_seed", "selection_seed"):
            if not manifest[field].astype(str).eq(str(run[field])).all():
                raise SelectionMechanismError(f"{run_slot} manifest {field} differs from canonical run")
        rank_key_valid = not manifest.duplicated(["replay_role", "rank"]).any()
        if not rank_key_valid:
            raise SelectionMechanismError(
                f"{run_slot} rank is not unique by (run_slot, replay_role, rank)"
            )
        for role, group in manifest.groupby("replay_role", sort=True):
            ranks = sorted(pd.to_numeric(group["rank"], errors="raise").astype(int).tolist())
            if ranks != list(range(1, len(group) + 1)):
                raise SelectionMechanismError(f"{run_slot}/{role} ranks are not contiguous from 1")
        missing_ids = sorted(set(manifest["sample_id"].astype(str)) - set(feature_lookup.index.astype(str)))
        if missing_ids:
            raise SelectionMechanismError(f"{run_slot} selected IDs missing from sample features: {missing_ids[:3]}")
        merged = manifest.merge(
            sample_features,
            on="sample_id",
            how="left",
            validate="one_to_one",
            suffixes=("_selection", "_feature"),
        )
        _check_manifest_mirrors(manifest, merged, run_slot)
        mirrored = {"y_true", "oof_fold", "dynamic_bucket", "mean_p_defect", "correct_rate", "std_p_defect"}
        manifest_metadata = [column for column in manifest.columns if column not in mirrored]
        enriched = manifest[manifest_metadata].merge(
            sample_features, on="sample_id", how="left", validate="one_to_one"
        )
        for field in ("phase", "method", "budget", "guard_ratio"):
            enriched[field] = run[field]
        all_rows.append(enriched)
        audit_rows.append(
            {
                "run_slot": run_slot,
                "expected_selection_sha256": expected_sha,
                "actual_selection_sha256": actual_sha,
                "sha_exact": True,
                "selected_rows": len(manifest),
                "unique_samples": int(manifest["sample_id"].nunique()),
                "replay_role_count": int(manifest["replay_role"].nunique()),
                "phase_b_role_rank_key_valid": rank_key_valid,
                "global_run_rank_has_duplicates": bool(manifest.duplicated(["rank"]).any()),
                "sample_set_digest": _set_digest(manifest),
                "status": "PASS",
            }
        )
    return pd.concat(all_rows, ignore_index=True), pd.DataFrame(audit_rows)


def _group_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "run_slot", "triad_id", "phase", "condition_id", "method", "budget",
        "guard_ratio", "arm", "training_seed", "selection_seed", "replay_role",
    ]
    return [field for field in candidates if field in frame]


def build_selection_summaries(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build categorical, all-numeric, and late-persistence summaries."""

    required = {"run_slot", "sample_id", "replay_role", "dynamic_bucket", "oof_fold", "oof_group_id", "train_primary_class", "y_true"}
    _require_columns(selected, required, label="enriched selections")
    keys = _group_columns(selected)
    categorical_rows: list[dict[str, Any]] = []
    for key_values, group in selected.groupby(keys, dropna=False, sort=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        metadata = dict(zip(keys, key_values, strict=True))
        for dimension in ("dynamic_bucket", "oof_fold", "oof_group_id", "train_primary_class", "y_true"):
            counts = group[dimension].fillna("<NA>").astype(str).value_counts(dropna=False)
            for value, count in counts.items():
                categorical_rows.append(
                    {
                        **metadata,
                        "dimension": dimension,
                        "value": value,
                        "count": int(count),
                        "share": float(count / len(group)),
                        "selected_count": len(group),
                    }
                )

    numeric_rows: list[dict[str, Any]] = []
    numeric_features = [field for field in _NUMERIC_FEATURES if field in selected]
    for key_values, group in selected.groupby(keys, dropna=False, sort=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        metadata = dict(zip(keys, key_values, strict=True))
        for feature in numeric_features:
            values = pd.to_numeric(group[feature], errors="coerce")
            finite = values[np.isfinite(values)]
            numeric_rows.append(
                {
                    **metadata,
                    "feature": feature,
                    "selected_count": len(group),
                    "non_null_count": len(finite),
                    "mean": float(finite.mean()) if len(finite) else np.nan,
                    "std": float(finite.std(ddof=0)) if len(finite) else np.nan,
                    "min": float(finite.min()) if len(finite) else np.nan,
                    "q05": float(finite.quantile(0.05)) if len(finite) else np.nan,
                    "q25": float(finite.quantile(0.25)) if len(finite) else np.nan,
                    "median": float(finite.median()) if len(finite) else np.nan,
                    "q75": float(finite.quantile(0.75)) if len(finite) else np.nan,
                    "q95": float(finite.quantile(0.95)) if len(finite) else np.nan,
                    "max": float(finite.max()) if len(finite) else np.nan,
                    "positive_rate_among_non_null": float((finite > 0).mean()) if len(finite) else np.nan,
                }
            )

    late_rows: list[dict[str, Any]] = []
    _require_columns(selected, ["last_wrong_epoch", "final_correct", "correct_rate", "forgetting_count"], label="late-persistence features")
    for key_values, group in selected.groupby(keys, dropna=False, sort=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        metadata = dict(zip(keys, key_values, strict=True))
        last_wrong = pd.to_numeric(group["last_wrong_epoch"], errors="coerce")
        final_correct = pd.to_numeric(group["final_correct"], errors="coerce")
        correct_rate = pd.to_numeric(group["correct_rate"], errors="coerce")
        forgetting = pd.to_numeric(group["forgetting_count"], errors="coerce")
        late_rows.append(
            {
                **metadata,
                "selected_count": len(group),
                "late_wrong_after_epoch160_count": int((last_wrong.fillna(-np.inf) >= 161).sum()),
                "late_wrong_after_epoch160_rate": float((last_wrong.fillna(-np.inf) >= 161).mean()),
                "last_wrong_epoch_non_null_count": int(last_wrong.notna().sum()),
                "last_wrong_epoch_mean_non_null": float(last_wrong.mean()) if last_wrong.notna().any() else np.nan,
                "final_wrong_rate": float((1 - final_correct).mean()),
                "persistent_0p5_error_rate": float((1 - correct_rate).mean()),
                "forgetting_count_mean": float(forgetting.mean()),
                "late_persistence_semantics": "summary_indicator_not_late40_frequency",
                "blank_last_wrong_semantics": "no_recorded_wrong_epoch",
            }
        )
    return pd.DataFrame(categorical_rows), pd.DataFrame(numeric_rows), pd.DataFrame(late_rows)


def _overlap_record(left: set[str], right: set[str]) -> dict[str, Any]:
    intersection = len(left & right)
    union = len(left | right)
    left_only = len(left - right)
    right_only = len(right - left)
    denominator = min(len(left), len(right))
    return {
        "left_count": len(left),
        "right_count": len(right),
        "intersection_count": intersection,
        "union_count": union,
        "left_only_count": left_only,
        "right_only_count": right_only,
        "symmetric_difference_count": left_only + right_only,
        "jaccard": float(intersection / union) if union else 1.0,
        "overlap_coefficient": float(intersection / denominator) if denominator else np.nan,
        "effective_unique_contrast_count_each_side": min(left_only, right_only),
        "effective_unique_contrast_rate": float(min(left_only, right_only) / denominator) if denominator else np.nan,
    }


def build_triad_overlap_audit(selected: pd.DataFrame) -> pd.DataFrame:
    """Quantify T/R1/R2 overlap separately for each replay role and overall."""

    _require_columns(selected, ["triad_id", "run_slot", "arm", "replay_role", "sample_id"], label="selected rows")
    records: list[dict[str, Any]] = []
    for triad_id, triad in selected.groupby("triad_id", sort=True):
        if set(triad["arm"].astype(str)) != {"T", "R1", "R2"}:
            raise SelectionMechanismError(f"{triad_id} does not contain T/R1/R2")
        scopes = ["all", *sorted(triad["replay_role"].astype(str).unique())]
        for scope in scopes:
            scoped = triad if scope == "all" else triad.loc[triad["replay_role"].astype(str) == scope]
            arm_sets = {
                arm: set(scoped.loc[scoped["arm"].astype(str) == arm, "sample_id"].astype(str))
                for arm in ("T", "R1", "R2")
            }
            run_slots = {
                arm: "|".join(sorted(scoped.loc[scoped["arm"].astype(str) == arm, "run_slot"].astype(str).unique()))
                for arm in ("T", "R1", "R2")
            }
            for left_arm, right_arm in (("T", "R1"), ("T", "R2"), ("R1", "R2")):
                records.append(
                    {
                        "triad_id": triad_id,
                        "scope": scope,
                        "left_arm": left_arm,
                        "right_arm": right_arm,
                        "left_run_slot": run_slots[left_arm],
                        "right_run_slot": run_slots[right_arm],
                        **_overlap_record(arm_sets[left_arm], arm_sets[right_arm]),
                    }
                )
    return pd.DataFrame(records)


def build_method_overlaps(selected: pd.DataFrame) -> pd.DataFrame:
    """Pair Treatment methods within the same seed, phase, budget, and role."""

    required = ["phase", "method", "budget", "training_seed", "run_slot", "arm", "replay_role", "sample_id"]
    _require_columns(selected, required, label="selected rows")
    treatment = selected.loc[selected["arm"].astype(str) == "T"]
    records: list[dict[str, Any]] = []
    group_keys = ["phase", "training_seed", "budget", "replay_role"]
    for values, group in treatment.groupby(group_keys, sort=True, dropna=False):
        sets = []
        for run_slot, run in group.groupby("run_slot", sort=True):
            sets.append(
                {
                    "run_slot": str(run_slot),
                    "method": str(run["method"].iloc[0]),
                    "condition_id": str(run["condition_id"].iloc[0]) if "condition_id" in run else "",
                    "samples": set(run["sample_id"].astype(str)),
                }
            )
        for left, right in combinations(sets, 2):
            records.append(
                {
                    **dict(zip(group_keys, values if isinstance(values, tuple) else (values,), strict=True)),
                    "left_run_slot": left["run_slot"],
                    "right_run_slot": right["run_slot"],
                    "left_method": left["method"],
                    "right_method": right["method"],
                    "left_condition_id": left["condition_id"],
                    "right_condition_id": right["condition_id"],
                    **_overlap_record(left["samples"], right["samples"]),
                }
            )
    return pd.DataFrame(records)


def build_budget_nesting(selected: pd.DataFrame) -> pd.DataFrame:
    """Audit whether smaller Treatment budgets are exact prefixes/subsets."""

    required = ["phase", "method", "budget", "training_seed", "run_slot", "arm", "replay_role", "sample_id"]
    _require_columns(selected, required, label="selected rows")
    treatment = selected.loc[selected["arm"].astype(str) == "T"]
    records: list[dict[str, Any]] = []
    group_keys = ["phase", "method", "training_seed", "replay_role"]
    for values, group in treatment.groupby(group_keys, sort=True, dropna=False):
        entries = []
        for run_slot, run in group.groupby("run_slot", sort=True):
            entries.append(
                {
                    "run_slot": str(run_slot),
                    "budget": int(run["budget"].iloc[0]),
                    "samples": set(run["sample_id"].astype(str)),
                }
            )
        for a, b in combinations(entries, 2):
            if a["budget"] == b["budget"]:
                continue
            lower, higher = (a, b) if a["budget"] < b["budget"] else (b, a)
            overlap = _overlap_record(lower["samples"], higher["samples"])
            records.append(
                {
                    **dict(zip(group_keys, values if isinstance(values, tuple) else (values,), strict=True)),
                    "lower_run_slot": lower["run_slot"],
                    "higher_run_slot": higher["run_slot"],
                    "lower_budget": lower["budget"],
                    "higher_budget": higher["budget"],
                    "intersection_count": overlap["intersection_count"],
                    "lower_is_subset_of_higher": lower["samples"].issubset(higher["samples"]),
                    "lower_retained_rate": float(overlap["intersection_count"] / len(lower["samples"])) if lower["samples"] else np.nan,
                    "higher_increment_count": len(higher["samples"] - lower["samples"]),
                }
            )
    return pd.DataFrame(records)


def build_same_selection_reversals(
    selected: pd.DataFrame, triad_outcomes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Link identical Treatment sample sets across seeds to outcome reversals."""

    _require_columns(selected, ["run_slot", "triad_id", "arm", "training_seed", "replay_role", "sample_id"], label="selected rows")
    _require_columns(
        triad_outcomes,
        ["triad_id", "exclusive_cohort", "dual_improvement", "high_value", "dual_harm"],
        label="triad outcomes",
    )
    treatment = selected.loc[selected["arm"].astype(str) == "T"]
    set_rows: list[dict[str, Any]] = []
    for run_slot, run in treatment.groupby("run_slot", sort=True):
        record: dict[str, Any] = {
            "run_slot": str(run_slot),
            "triad_id": str(run["triad_id"].iloc[0]),
            "training_seed": int(run["training_seed"].iloc[0]),
            "selected_count": len(run),
            "sample_set_digest": _set_digest(run),
        }
        for field in ("phase", "condition_id", "method", "budget", "guard_ratio"):
            if field in run:
                record[field] = run[field].iloc[0]
        set_rows.append(record)
    sets = pd.DataFrame(set_rows).merge(
        triad_outcomes[
            ["triad_id", "exclusive_cohort", "dual_improvement", "high_value", "dual_harm"]
        ],
        on="triad_id",
        how="left",
        validate="one_to_one",
    )
    if sets["exclusive_cohort"].isna().any():
        raise SelectionMechanismError("Treatment selection sets lack triad outcomes")
    group_rows: list[dict[str, Any]] = []
    for digest, group in sets.groupby("sample_set_digest", sort=True):
        group_rows.append(
            {
                "sample_set_digest": digest,
                "selected_count": int(group["selected_count"].iloc[0]),
                "triad_count": len(group),
                "seed_count": int(group["training_seed"].nunique()),
                "training_seeds": "|".join(sorted(group["training_seed"].astype(str).unique())),
                "condition_ids": "|".join(sorted(group.get("condition_id", pd.Series(dtype=str)).astype(str).unique())),
                "cohorts": "|".join(sorted(group["exclusive_cohort"].astype(str).unique())),
                "dual_improvement_count": int(group["dual_improvement"].astype(bool).sum()),
                "high_value_count": int(group["high_value"].astype(bool).sum()),
                "dual_harm_count": int(group["dual_harm"].astype(bool).sum()),
                "spans_dual_improvement_and_dual_harm": bool(
                    group["dual_improvement"].astype(bool).any()
                    and group["dual_harm"].astype(bool).any()
                ),
            }
        )
    grouped = pd.DataFrame(group_rows)
    reversal_digests = set(
        grouped.loc[grouped["spans_dual_improvement_and_dual_harm"], "sample_set_digest"]
    )
    reversals = sets.loc[sets["sample_set_digest"].isin(reversal_digests)].sort_values(
        ["sample_set_digest", "training_seed", "triad_id"], kind="stable"
    )
    return sets, grouped, reversals.reset_index(drop=True)
