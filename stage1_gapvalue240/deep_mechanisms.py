from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .errors import ValidationError


SELECTION_REQUIRED_COLUMNS = {
    "run_slot",
    "triad_id",
    "condition_id",
    "arm",
    "sample_id",
    "y_true",
    "dynamic_bucket",
    "mean_p_defect",
    "correct_rate",
    "std_p_defect",
    "replay_role",
}


def _selection_path(selection_root: Path, run_slot: str) -> Path:
    return selection_root / run_slot / "selection_manifest.csv"


def _audit_path(selection_root: Path, run_slot: str) -> Path:
    return selection_root / run_slot / "selection_audit.json"


def _load_selection(selection_root: Path, row: pd.Series) -> pd.DataFrame:
    run_slot = str(row.run_slot)
    path = _selection_path(selection_root, run_slot)
    if not path.is_file():
        raise ValidationError(f"Missing selection manifest for {run_slot}: {path}")
    frame = pd.read_csv(
        path,
        dtype={
            "run_slot": "string",
            "triad_id": "string",
            "condition_id": "string",
            "arm": "string",
            "sample_id": "string",
            "oof_fold": "string",
            "dynamic_bucket": "string",
            "replay_role": "string",
        },
    )
    missing = SELECTION_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValidationError(
            f"Selection manifest {run_slot} missing columns: {sorted(missing)}"
        )
    expected_budget = int(row.budget)
    if len(frame) != expected_budget:
        raise ValidationError(
            f"Selection manifest {run_slot} budget mismatch: "
            f"expected={expected_budget}, actual={len(frame)}"
        )
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValidationError(f"Selection manifest {run_slot} contains duplicate sample IDs")
    for column, expected in (
        ("run_slot", run_slot),
        ("triad_id", str(row.triad_id)),
        ("arm", str(row.arm)),
    ):
        actual = set(frame[column].dropna().astype(str).unique())
        if actual != {expected}:
            raise ValidationError(
                f"Selection manifest {run_slot} {column} mismatch: {sorted(actual)}"
            )
    labels = set(pd.to_numeric(frame.y_true, errors="raise").astype(int).unique())
    if not labels.issubset({0, 1}):
        raise ValidationError(f"Selection manifest {run_slot} has invalid labels")
    return frame


def _scope_set(frame: pd.DataFrame, scope: str) -> set[str]:
    if scope == "all":
        subset = frame
    elif scope == "normal":
        subset = frame[pd.to_numeric(frame.y_true).astype(int) == 0]
    elif scope == "defect":
        subset = frame[pd.to_numeric(frame.y_true).astype(int) == 1]
    else:
        raise ValueError(scope)
    return set(subset.sample_id.astype(str))


def _overlap_record(
    *,
    triad_id: str,
    condition_slot: str,
    phase: str,
    budget: int,
    guard_ratio: float,
    control: str,
    scope: str,
    treatment: set[str],
    comparison: set[str],
) -> dict[str, Any]:
    intersection = treatment & comparison
    union = treatment | comparison
    denominator = len(treatment)
    return {
        "triad_id": triad_id,
        "condition_slot": condition_slot,
        "phase": phase,
        "budget": budget,
        "guard_ratio": guard_ratio,
        "control": control,
        "scope": scope,
        "treatment_count": len(treatment),
        "control_count": len(comparison),
        "overlap_count": len(intersection),
        "overlap_rate": len(intersection) / denominator if denominator else np.nan,
        "jaccard": len(intersection) / len(union) if union else np.nan,
        "unique_to_treatment": len(treatment - comparison),
        "unique_to_control": len(comparison - treatment),
        "effective_unique_contrast": (
            len(treatment - comparison) / denominator if denominator else np.nan
        ),
    }


def audit_selections(
    matrix: pd.DataFrame,
    selection_root: str | Path,
) -> dict[str, pd.DataFrame]:
    """Audit frozen selection manifests without modifying them.

    The returned tables deliberately keep all-sample and role-specific overlap
    separate.  In Phase B, shared normal replay is expected; the scientific
    contrast is carried by the defect guard portion.
    """

    selection_root = Path(selection_root).resolve()
    required_matrix = {
        "run_slot",
        "triad_id",
        "condition_slot",
        "phase",
        "method",
        "budget",
        "guard_ratio",
        "arm",
        "training_seed",
    }
    missing = required_matrix - set(matrix.columns)
    if missing:
        raise ValidationError(f"Matrix missing columns: {sorted(missing)}")
    if matrix.run_slot.astype(str).duplicated().any():
        raise ValidationError("Matrix run_slot values must be unique")

    frames: dict[str, pd.DataFrame] = {}
    run_audits: list[dict[str, Any]] = []
    compositions: list[dict[str, Any]] = []
    matrix_index = matrix.copy()
    matrix_index["run_slot"] = matrix_index.run_slot.astype(str)

    for _, row in matrix_index.iterrows():
        slot = str(row.run_slot)
        frame = _load_selection(selection_root, row)
        frames[slot] = frame
        audit_file = _audit_path(selection_root, slot)
        if not audit_file.is_file():
            raise ValidationError(f"Missing selection audit for {slot}: {audit_file}")
        try:
            audit = json.loads(audit_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise ValidationError(f"Invalid selection audit for {slot}: {exc}") from exc
        if int(audit.get("rows", -1)) != len(frame):
            raise ValidationError(f"Selection audit row mismatch for {slot}")
        if int(audit.get("unique_samples", -1)) != frame.sample_id.nunique():
            raise ValidationError(f"Selection audit unique-sample mismatch for {slot}")

        fallback = audit.get("fallback_counts", {})
        run_audits.append(
            {
                "run_slot": slot,
                "triad_id": str(row.triad_id),
                "condition_slot": str(row.condition_slot),
                "phase": str(row.phase),
                "method": str(row.method),
                "budget": int(row.budget),
                "guard_ratio": float(row.guard_ratio),
                "arm": str(row.arm),
                "training_seed": int(row.training_seed),
                "rows": len(frame),
                "unique_samples": int(frame.sample_id.nunique()),
                "overlap_count": audit.get("overlap_count"),
                "overlap_rate": audit.get("overlap_rate"),
                "jaccard": audit.get("jaccard"),
                "forced_overlap_count": audit.get("forced_overlap_count"),
                "effective_unique_contrast": audit.get(
                    "effective_unique_contrast"
                ),
                "max_abs_smd": audit.get("max_abs_smd"),
                "fallback_L0": fallback.get("L0", 0),
                "fallback_L1": fallback.get("L1", 0),
                "fallback_L2": fallback.get("L2", 0),
                "fallback_L3": fallback.get("L3", 0),
                "fallback_L4_forced_overlap": fallback.get(
                    "L4_FORCED_OVERLAP", 0
                ),
                "fallback_counts_json": json.dumps(
                    fallback, sort_keys=True, ensure_ascii=False
                ),
                "fold_total_variation": (
                    audit.get("fold_balance", {}) or {}
                ).get("total_variation"),
                "bucket_total_variation": (
                    audit.get("dynamic_bucket_balance", {}) or {}
                ).get("total_variation"),
            }
        )
        grouped = (
            frame.assign(y_true=pd.to_numeric(frame.y_true).astype(int))
            .groupby(["y_true", "replay_role", "dynamic_bucket"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        for record in grouped.to_dict("records"):
            compositions.append(
                {
                    "run_slot": slot,
                    "triad_id": str(row.triad_id),
                    "condition_slot": str(row.condition_slot),
                    "phase": str(row.phase),
                    "method": str(row.method),
                    "budget": int(row.budget),
                    "guard_ratio": float(row.guard_ratio),
                    "arm": str(row.arm),
                    **record,
                    "share": int(record["count"]) / len(frame),
                }
            )

    triad_overlaps: list[dict[str, Any]] = []
    for triad_id, group in matrix_index.groupby("triad_id", sort=True):
        arms = {str(row.arm): row for _, row in group.iterrows()}
        if set(arms) != {"T", "R1", "R2"}:
            raise ValidationError(
                f"Triad {triad_id} must contain exactly T/R1/R2: {sorted(arms)}"
            )
        treatment_frame = frames[str(arms["T"].run_slot)]
        for control in ("R1", "R2"):
            comparison_frame = frames[str(arms[control].run_slot)]
            for scope in ("all", "normal", "defect"):
                record = _overlap_record(
                    triad_id=str(triad_id),
                    condition_slot=str(arms["T"].condition_slot),
                    phase=str(arms["T"].phase),
                    budget=int(arms["T"].budget),
                    guard_ratio=float(arms["T"].guard_ratio),
                    control=control,
                    scope=scope,
                    treatment=_scope_set(treatment_frame, scope),
                    comparison=_scope_set(comparison_frame, scope),
                )
                triad_overlaps.append(record)

    treatment_rows = matrix_index[matrix_index.arm.astype(str) == "T"].copy()
    method_overlaps: list[dict[str, Any]] = []
    for (_, budget, seed), group in treatment_rows.groupby(
        ["phase", "budget", "training_seed"], sort=True
    ):
        for (_, left), (_, right) in combinations(group.iterrows(), 2):
            left_set = _scope_set(frames[str(left.run_slot)], "all")
            right_set = _scope_set(frames[str(right.run_slot)], "all")
            intersection = left_set & right_set
            union = left_set | right_set
            method_overlaps.append(
                {
                    "phase": str(left.phase),
                    "budget": int(budget),
                    "training_seed": int(seed),
                    "left_condition": str(left.condition_slot),
                    "right_condition": str(right.condition_slot),
                    "left_method": str(left.method),
                    "right_method": str(right.method),
                    "overlap_count": len(intersection),
                    "jaccard": len(intersection) / len(union) if union else np.nan,
                    "left_overlap_rate": (
                        len(intersection) / len(left_set) if left_set else np.nan
                    ),
                    "right_overlap_rate": (
                        len(intersection) / len(right_set) if right_set else np.nan
                    ),
                }
            )

    run_audit_frame = pd.DataFrame(run_audits)
    triad_overlap_frame = pd.DataFrame(triad_overlaps)
    # Cross-check R2 audit metadata against the direct all-sample set operation.
    direct_r2 = triad_overlap_frame.query(
        "control == 'R2' and scope == 'all'"
    )[["triad_id", "overlap_count", "jaccard"]]
    r2_audit = run_audit_frame.query("arm == 'R2'")
    merged = r2_audit.merge(
        direct_r2,
        on="triad_id",
        how="left",
        suffixes=("_audit", "_direct"),
        validate="one_to_one",
    )
    for _, row in merged.iterrows():
        if pd.notna(row.overlap_count_audit) and int(row.overlap_count_audit) != int(
            row.overlap_count_direct
        ):
            raise ValidationError(
                f"R2 overlap audit mismatch for {row.run_slot}: "
                f"audit={row.overlap_count_audit}, direct={row.overlap_count_direct}"
            )
        if pd.notna(row.jaccard_audit) and not np.isclose(
            float(row.jaccard_audit),
            float(row.jaccard_direct),
            rtol=0,
            atol=1e-12,
        ):
            raise ValidationError(f"R2 Jaccard audit mismatch for {row.run_slot}")

    return {
        "run_audit": run_audit_frame,
        "composition": pd.DataFrame(compositions),
        "triad_overlap": triad_overlap_frame,
        "method_overlap": pd.DataFrame(method_overlaps),
    }


def calibration_diagnostics(
    predictions: pd.DataFrame,
    *,
    bins: int = 15,
) -> dict[str, float | int]:
    required = {"sample_id", "y_true", "score", "score_raw"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValidationError(f"Prediction table missing columns: {sorted(missing)}")
    if predictions.sample_id.duplicated().any():
        raise ValidationError("Prediction sample IDs must be unique")
    y = pd.to_numeric(predictions.y_true, errors="raise").to_numpy(dtype=np.int8)
    score = pd.to_numeric(predictions.score, errors="raise").to_numpy(dtype=float)
    raw = pd.to_numeric(predictions.score_raw, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(score).all() or not np.isfinite(raw).all():
        raise ValidationError("Prediction scores contain NaN/Inf")
    if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) != 2:
        raise ValidationError("Calibration diagnostics require both 0 and 1 labels")
    clipped = np.clip(score, 1e-12, 1 - 1e-12)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.minimum(np.digitize(clipped, edges[1:-1], right=False), bins - 1)
    ece = 0.0
    for index in range(bins):
        mask = bin_ids == index
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(clipped[mask].mean()) - float(y[mask].mean())
            )
    return {
        "row_count": int(len(y)),
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "auroc_raw": float(roc_auc_score(y, raw)),
        "auprc_raw": float(average_precision_score(y, raw)),
        "brier": float(brier_score_loss(y, score)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "ece": float(ece),
        "ece_bins": int(bins),
    }


def _stable_tail_indices(
    frame: pd.DataFrame,
    *,
    count: int,
    highest: bool,
) -> pd.Index:
    if count <= 0:
        return pd.Index([], dtype=frame.index.dtype)
    ordered = frame.sort_values(
        ["control_median_score_raw", "sample_id"],
        ascending=[not highest, True],
        kind="mergesort",
    )
    return ordered.index[: min(count, len(ordered))]


def define_reference_tails(
    consensus: pd.DataFrame,
    *,
    tn_target: int = 68253,
    fn_limit: int = 95,
    normal_tail_fraction: float = 0.10,
    defect_tail_fraction: float = 0.05,
) -> pd.DataFrame:
    required = {"sample_id", "y_true", "control_median_score_raw"}
    missing = required - set(consensus.columns)
    if missing:
        raise ValidationError(f"Consensus table missing columns: {sorted(missing)}")
    result = consensus.copy()
    if result.sample_id.duplicated().any():
        raise ValidationError("Consensus sample IDs must be unique")
    result["y_true"] = pd.to_numeric(result.y_true, errors="raise").astype(int)
    if not set(result.y_true.unique()).issubset({0, 1}):
        raise ValidationError("Consensus y_true must be 0/1")
    if not np.isfinite(
        pd.to_numeric(result.control_median_score_raw, errors="raise")
    ).all():
        raise ValidationError("Consensus scores contain NaN/Inf")
    normal = result[result.y_true == 0]
    defect = result[result.y_true == 1]
    normal_operational_count = len(normal) - int(tn_target)
    if normal_operational_count < 0:
        raise ValidationError(
            f"TN target {tn_target} exceeds normal count {len(normal)}"
        )
    if fn_limit > len(defect):
        raise ValidationError(
            f"FN limit {fn_limit} exceeds defect count {len(defect)}"
        )
    normal_tail_count = max(1, int(round(len(normal) * normal_tail_fraction)))
    defect_tail_count = max(1, int(round(len(defect) * defect_tail_fraction)))
    result["operational_tail"] = False
    result["distribution_tail"] = False
    result.loc[
        _stable_tail_indices(
            normal, count=normal_operational_count, highest=True
        ),
        "operational_tail",
    ] = True
    result.loc[
        _stable_tail_indices(defect, count=int(fn_limit), highest=False),
        "operational_tail",
    ] = True
    result.loc[
        _stable_tail_indices(normal, count=normal_tail_count, highest=True),
        "distribution_tail",
    ] = True
    result.loc[
        _stable_tail_indices(defect, count=defect_tail_count, highest=False),
        "distribution_tail",
    ] = True
    result["reference_rank_within_class"] = (
        result.groupby("y_true")["control_median_score_raw"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return result


def build_control_consensus(
    prediction_paths: list[str | Path],
    *,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build a treatment-independent sample risk reference from control runs.

    Scores are stored as float32 because the matrix is only an intermediate
    ranking asset.  The returned median is float64.  When ``cache_path`` is
    supplied the score matrix is a local memmap that callers may delete after
    the report is finalized.
    """

    if not prediction_paths:
        raise ValidationError("At least one control prediction file is required")
    paths = [Path(path).resolve() for path in prediction_paths]
    required = ["sample_id", "y_true", "score_raw"]
    first = pd.read_csv(
        paths[0],
        usecols=required,
        dtype={"sample_id": "string"},
    ).sort_values("sample_id", kind="mergesort")
    if first.sample_id.duplicated().any():
        raise ValidationError(f"Duplicate prediction IDs in {paths[0]}")
    if cache_path is None:
        scores: np.ndarray = np.empty(
            (len(first), len(paths)), dtype=np.float32
        )
    else:
        cache = Path(cache_path).resolve()
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            raise ValidationError(f"Consensus cache already exists: {cache}")
        scores = np.memmap(
            cache,
            mode="w+",
            dtype=np.float32,
            shape=(len(first), len(paths)),
        )
    canonical_ids = first.sample_id.astype(str).to_numpy()
    canonical_labels = pd.to_numeric(first.y_true, errors="raise").to_numpy(
        dtype=np.int8
    )
    for column, path in enumerate(paths):
        frame = (
            first
            if column == 0
            else pd.read_csv(
                path,
                usecols=required,
                dtype={"sample_id": "string"},
            ).sort_values("sample_id", kind="mergesort")
        )
        if len(frame) != len(first) or frame.sample_id.duplicated().any():
            raise ValidationError(f"Prediction identity mismatch in {path}")
        ids = frame.sample_id.astype(str).to_numpy()
        labels = pd.to_numeric(frame.y_true, errors="raise").to_numpy(dtype=np.int8)
        if not np.array_equal(ids, canonical_ids):
            raise ValidationError(f"Prediction sample ID set differs in {path}")
        if not np.array_equal(labels, canonical_labels):
            raise ValidationError(f"Prediction labels differ in {path}")
        values = pd.to_numeric(frame.score_raw, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValidationError(f"Prediction score_raw contains NaN/Inf in {path}")
        scores[:, column] = values.astype(np.float32)
    if isinstance(scores, np.memmap):
        scores.flush()
    medians = np.empty(len(first), dtype=np.float64)
    block = 4096
    for start in range(0, len(first), block):
        stop = min(start + block, len(first))
        medians[start:stop] = np.median(
            np.asarray(scores[start:stop], dtype=np.float32), axis=1
        )
    return pd.DataFrame(
        {
            "sample_id": canonical_ids,
            "y_true": canonical_labels,
            "control_median_score_raw": medians,
            "control_run_count": len(paths),
        }
    )


def _aligned_pair(
    treatment: pd.DataFrame,
    control: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["sample_id", "y_true", "score", "score_raw"]
    for label, frame in (("treatment", treatment), ("control", control)):
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValidationError(f"{label} prediction table missing {sorted(missing)}")
        if frame.sample_id.duplicated().any():
            raise ValidationError(f"{label} prediction sample IDs must be unique")
    merged = (
        treatment[columns]
        .rename(
            columns={
                "y_true": "y_true_t",
                "score": "score_t",
                "score_raw": "score_raw_t",
            }
        )
        .merge(
            control[columns].rename(
                columns={
                    "y_true": "y_true_c",
                    "score": "score_c",
                    "score_raw": "score_raw_c",
                }
            ),
            on="sample_id",
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
    )
    if not (merged["_merge"] == "both").all():
        raise ValidationError("Treatment/control prediction ID sets differ")
    if not (merged.y_true_t.astype(int) == merged.y_true_c.astype(int)).all():
        raise ValidationError("Treatment/control prediction labels differ")
    ref_columns = {
        "sample_id",
        "y_true",
        "operational_tail",
        "distribution_tail",
    }
    missing_ref = ref_columns - set(reference.columns)
    if missing_ref:
        raise ValidationError(f"Reference table missing {sorted(missing_ref)}")
    merged = merged.drop(columns="_merge").merge(
        reference[list(ref_columns)],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if merged.operational_tail.isna().any():
        raise ValidationError("Reference sample ID set differs from predictions")
    if not (merged.y_true_t.astype(int) == merged.y_true.astype(int)).all():
        raise ValidationError("Reference labels differ from predictions")
    return merged


def pair_prediction_tail_shifts(
    treatment: pd.DataFrame,
    control: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = _aligned_pair(treatment, control, reference)
    merged["raw_shift"] = merged.score_raw_t.astype(float) - merged.score_raw_c.astype(
        float
    )
    merged["calibrated_shift"] = merged.score_t.astype(float) - merged.score_c.astype(
        float
    )
    merged["raw_beneficial"] = np.where(
        merged.y_true.astype(int) == 0, merged.raw_shift < 0, merged.raw_shift > 0
    )
    merged["calibrated_beneficial"] = np.where(
        merged.y_true.astype(int) == 0,
        merged.calibrated_shift < 0,
        merged.calibrated_shift > 0,
    )
    sample_shifts = merged[
        [
            "sample_id",
            "y_true",
            "operational_tail",
            "distribution_tail",
            "raw_shift",
            "calibrated_shift",
            "raw_beneficial",
            "calibrated_beneficial",
        ]
    ].copy()

    summaries: list[dict[str, Any]] = []
    for y_true, label in ((0, "normal"), (1, "defect")):
        label_frame = sample_shifts[sample_shifts.y_true.astype(int) == y_true]
        for scope, mask in (
            ("all", np.ones(len(label_frame), dtype=bool)),
            (
                "operational",
                label_frame.operational_tail.to_numpy(dtype=bool),
            ),
            (
                "tail_gap",
                label_frame.distribution_tail.to_numpy(dtype=bool),
            ),
        ):
            subset = label_frame.loc[mask]
            for score_type, shift_column in (
                ("raw", "raw_shift"),
                ("calibrated", "calibrated_shift"),
            ):
                values = subset[shift_column].to_numpy(dtype=float)
                beneficial = values < 0 if y_true == 0 else values > 0
                harmed = values > 0 if y_true == 0 else values < 0
                summaries.append(
                    {
                        "label": label,
                        "scope": scope,
                        "score_type": score_type,
                        "n": int(len(values)),
                        "mean_shift": float(np.mean(values)) if len(values) else np.nan,
                        "median_shift": (
                            float(np.median(values)) if len(values) else np.nan
                        ),
                        "std_shift": (
                            float(np.std(values, ddof=1))
                            if len(values) > 1
                            else 0.0
                        ),
                        "beneficial_rate": (
                            float(np.mean(beneficial)) if len(values) else np.nan
                        ),
                        "harmed_rate": (
                            float(np.mean(harmed)) if len(values) else np.nan
                        ),
                        "neutral_rate": (
                            float(np.mean(values == 0)) if len(values) else np.nan
                        ),
                    }
                )
    return pd.DataFrame(summaries), sample_shifts


def summarize_sample_shift_consistency(
    shift_frames: list[tuple[str, int, pd.DataFrame]],
) -> pd.DataFrame:
    """Aggregate per-sample T-control directions across seeds.

    ``shift_frames`` entries are ``(control, training_seed, sample_shift_frame)``.
    This is descriptive mechanism evidence and is not a per-training-sample
    causal estimate.
    """

    if not shift_frames:
        return pd.DataFrame()
    records: list[pd.DataFrame] = []
    required = {
        "sample_id",
        "y_true",
        "raw_shift",
        "calibrated_shift",
    }
    for control, seed, frame in shift_frames:
        missing = required - set(frame.columns)
        if missing:
            raise ValidationError(f"Sample-shift frame missing {sorted(missing)}")
        if frame.sample_id.duplicated().any():
            raise ValidationError("Sample-shift frame IDs must be unique")
        current = frame[list(required)].copy()
        current["control"] = str(control)
        current["training_seed"] = int(seed)
        records.append(current)
    combined = pd.concat(records, ignore_index=True)
    combined["y_true"] = pd.to_numeric(combined.y_true, errors="raise").astype(int)
    combined["raw_beneficial"] = np.where(
        combined.y_true == 0, combined.raw_shift < 0, combined.raw_shift > 0
    )
    combined["calibrated_beneficial"] = np.where(
        combined.y_true == 0,
        combined.calibrated_shift < 0,
        combined.calibrated_shift > 0,
    )
    grouped = combined.groupby(["control", "sample_id"], sort=True)
    result = grouped.agg(
        y_true=("y_true", "first"),
        seed_count=("training_seed", "nunique"),
        raw_mean_shift=("raw_shift", "mean"),
        raw_median_shift=("raw_shift", "median"),
        raw_benefit_rate=("raw_beneficial", "mean"),
        calibrated_mean_shift=("calibrated_shift", "mean"),
        calibrated_median_shift=("calibrated_shift", "median"),
        calibrated_benefit_rate=("calibrated_beneficial", "mean"),
    ).reset_index()
    expected = combined.groupby("control").training_seed.nunique().to_dict()
    bad = result[
        result.apply(
            lambda row: int(row.seed_count) != int(expected[str(row.control)]),
            axis=1,
        )
    ]
    if not bad.empty:
        raise ValidationError(
            "Sample-shift ID sets are incomplete across seeds for at least one control"
        )
    return result


def summarize_selected_score_signs(
    matrix: pd.DataFrame,
    selection_root: str | Path,
    value_table: str | Path,
    *,
    condition_slot: str,
    score_column: str,
) -> dict[str, Any]:
    selection_root = Path(selection_root)
    candidates = matrix[
        (matrix.condition_slot.astype(str) == str(condition_slot))
        & (matrix.arm.astype(str) == "T")
    ]
    if candidates.empty:
        raise ValidationError(f"No T runs for condition {condition_slot}")
    selected_sets: list[set[str]] = []
    for _, row in candidates.iterrows():
        selected_sets.append(
            set(_load_selection(selection_root, row).sample_id.astype(str))
        )
    if any(current != selected_sets[0] for current in selected_sets[1:]):
        raise ValidationError(
            f"Treatment selections differ across seeds for {condition_slot}"
        )
    values = pd.read_csv(
        value_table,
        usecols=["sample_id", score_column],
        dtype={"sample_id": "string"},
    )
    if values.sample_id.duplicated().any():
        raise ValidationError("Value table sample IDs must be unique")
    selected = values[values.sample_id.astype(str).isin(selected_sets[0])].copy()
    if len(selected) != len(selected_sets[0]):
        raise ValidationError(
            f"Value table does not cover every selected sample for {condition_slot}"
        )
    score = pd.to_numeric(selected[score_column], errors="raise")
    if not np.isfinite(score).all():
        raise ValidationError(f"Selected {score_column} contains NaN/Inf")
    quantiles = score.quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "condition_slot": str(condition_slot),
        "score_column": score_column,
        "selected_count": int(len(score)),
        "negative_count": int((score < 0).sum()),
        "nonnegative_count": int((score >= 0).sum()),
        "min_score": float(score.min()),
        "max_score": float(score.max()),
        "mean_score": float(score.mean()),
        "q00": float(quantiles.loc[0.0]),
        "q10": float(quantiles.loc[0.1]),
        "q25": float(quantiles.loc[0.25]),
        "q50": float(quantiles.loc[0.5]),
        "q75": float(quantiles.loc[0.75]),
        "q90": float(quantiles.loc[0.9]),
        "q100": float(quantiles.loc[1.0]),
    }
