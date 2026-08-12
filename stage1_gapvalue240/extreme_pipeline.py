"""Read-only pipeline for the focused GapValue extreme-cohort report."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .deep_analysis import CanonicalInputError
from .extreme_cohorts import (
    build_outcome_mechanism_pairs,
    build_stratified_extreme_contrasts,
    build_training_window_features,
    classify_triad_cohorts,
    compute_operational_sample_dynamics,
    pair_selection_feature_deltas,
    summarize_extreme_feature_contrasts,
    summarize_leave_one_group_out,
    summarize_selection_operational_features,
    summarize_selection_set_outcomes,
)
from .util import sha256_file


REQUIRED_V2_TABLES: tuple[str, ...] = (
    "tables/triad_control_deltas.csv",
    "tables/canonical_run_metrics.csv",
    "tables/calibration_diagnostics.csv",
    "tables/paired_epoch_differences.csv",
    "tables/training_curve_summary.csv",
    "tables/r2_overlap_power_audit.csv",
    "tables/prediction_tail_detail.csv",
    "tables/prediction_tail_summary.csv",
    "tables/selection_composition.csv",
    "tables/selection_value_effects.csv",
)

REQUIRED_EXPERT_INPUTS: tuple[str, ...] = (
    "inputs/oof_probabilities_float64.mmap",
    "inputs/oof_probabilities_metadata.json",
    "inputs/sample_ids.csv",
    "inputs/epoch_gap_metrics.csv",
)


def _manifest_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise CanonicalInputError(f"Missing file manifest: {path}")
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CanonicalInputError(f"Invalid JSON manifest {path}: {exc}") from exc
        rows = payload.get("files") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise CanonicalInputError(f"JSON manifest has no files list: {path}")
    elif path.suffix.lower() == ".csv":
        rows = pd.read_csv(path).to_dict("records")
    else:
        raise CanonicalInputError(f"Unsupported manifest type: {path}")
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        relative = str(row.get("relative_path", "")).replace("\\", "/")
        if not relative or relative in records:
            raise CanonicalInputError(f"Invalid/duplicate manifest path: {relative!r}")
        records[relative] = dict(row)
    return records


def verify_manifested_inputs(
    root: str | Path,
    manifest_path: str | Path,
    required_paths: Iterable[str],
) -> dict[str, Path]:
    """Verify size and SHA for the requested immutable source files."""

    source_root = Path(root).resolve()
    manifest = Path(manifest_path).resolve()
    records = _manifest_records(manifest)
    verified: dict[str, Path] = {}
    for requested in required_paths:
        relative = str(requested).replace("\\", "/")
        row = records.get(relative)
        if row is None:
            raise CanonicalInputError(
                f"Manifest {manifest} does not cover required input {relative}"
            )
        path = (source_root / Path(relative)).resolve()
        try:
            path.relative_to(source_root)
        except ValueError as exc:
            raise CanonicalInputError(f"Unsafe manifested path: {relative}") from exc
        if not path.is_file():
            raise CanonicalInputError(f"Missing manifested input: {path}")
        expected_size = int(row["size_bytes"])
        if path.stat().st_size != expected_size:
            raise CanonicalInputError(
                f"Size mismatch for {path}: {path.stat().st_size} != {expected_size}"
            )
        expected_sha = str(row["sha256"]).upper()
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise CanonicalInputError(
                f"SHA-256 mismatch for {path}: {actual_sha} != {expected_sha}"
            )
        verified[relative] = path
    return verified


def build_tier_composition_audit(tiers: pd.DataFrame) -> pd.DataFrame:
    required = {"triad_id", "cohort_code", "phase", "training_seed", "budget"}
    missing = sorted(required.difference(tiers.columns))
    if missing:
        raise CanonicalInputError(f"Tier table missing columns: {missing}")
    dimensions = ["phase", "training_seed", "budget"]
    for optional in ("all_same_machine", "any_arm_resumed"):
        if optional in tiers.columns:
            dimensions.append(optional)
    cohort_sizes = tiers.groupby("cohort_code")["triad_id"].nunique()
    rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        grouped = (
            tiers.groupby(["cohort_code", dimension], dropna=False)["triad_id"]
            .nunique()
            .rename("triad_count")
            .reset_index()
        )
        for row in grouped.itertuples(index=False):
            value = getattr(row, dimension)
            count = int(row.triad_count)
            rows.append(
                {
                    "dimension": dimension,
                    "value": str(value),
                    "cohort_code": row.cohort_code,
                    "triad_count": count,
                    "share_within_cohort": count / int(cohort_sizes[row.cohort_code]),
                }
            )
    return pd.DataFrame(rows)


def build_fixed_selection_seed_flips(treatment_sets: pd.DataFrame) -> pd.DataFrame:
    outcomes = summarize_selection_set_outcomes(treatment_sets)
    crossing = outcomes.loc[outcomes["spans_exceptional_and_harmful"]].copy()
    if crossing.empty:
        return treatment_sets.iloc[0:0].assign(
            cohort_codes=pd.Series(dtype=str),
            spans_exceptional_and_harmful=pd.Series(dtype=bool),
        )
    return (
        treatment_sets.merge(
            crossing[
                [
                    "sample_set_digest",
                    "cohort_codes",
                    "exceptional_count",
                    "harmful_count",
                    "spans_exceptional_and_harmful",
                ]
            ],
            on="sample_set_digest",
            validate="many_to_one",
        )
        .sort_values(["sample_set_digest", "training_seed", "triad_id"])
        .reset_index(drop=True)
    )


def build_controlled_tradeoff_grid(
    tiers: pd.DataFrame,
    *,
    fn_margins: Sequence[int] = (0, 1, 2, 5),
    tn_minimums: Sequence[int] = (1, 100, 300, 600),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit transparent FN/TN gates using the worse of R1 and R2.

    This helper preserves the historical fixed-operating-point sensitivity
    table.  It does not treat a relaxed FN margin as a frontier improvement;
    the performance-frontier analysis implements that stricter comparison in
    a separate module.
    """

    required = {
        "triad_id",
        "delta_TN_R1",
        "delta_TN_R2",
        "delta_FN_R1",
        "delta_FN_R2",
    }
    missing = sorted(required.difference(tiers.columns))
    if missing:
        raise CanonicalInputError(f"Tier table missing columns: {missing}")
    margins = tuple(int(value) for value in fn_margins)
    minimums = tuple(int(value) for value in tn_minimums)
    if any(value < 0 for value in margins) or any(value <= 0 for value in minimums):
        raise ValueError("FN margins must be nonnegative and TN minimums positive")

    rows: list[dict[str, Any]] = []
    for row in tiers.itertuples(index=False):
        worst_tn = min(float(row.delta_TN_R1), float(row.delta_TN_R2))
        worst_fn = max(float(row.delta_FN_R1), float(row.delta_FN_R2))
        for margin in margins:
            for minimum in minimums:
                rows.append(
                    {
                        "triad_id": str(row.triad_id),
                        "fn_margin": margin,
                        "tn_minimum": minimum,
                        "min_delta_TN_across_controls": worst_tn,
                        "max_delta_FN_across_controls": worst_fn,
                        "qualifies_both_controls": bool(
                            worst_tn >= minimum and worst_fn <= margin
                        ),
                    }
                )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["fn_margin", "tn_minimum"], as_index=False)
        .agg(
            qualifying_triads=("qualifies_both_controls", "sum"),
            total_triads=("triad_id", "nunique"),
        )
        .sort_values(["fn_margin", "tn_minimum"], kind="stable")
        .reset_index(drop=True)
    )
    summary["qualifying_share"] = (
        summary["qualifying_triads"] / summary["total_triads"]
    )
    return detail, summary


def build_raw_calibrated_operational_audit(
    pairs: pd.DataFrame,
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify raw-score and calibrated fixed-point effects agree exactly."""

    pair_required = {
        "triad_id",
        "control",
        "t_run_slot",
        "control_run_slot",
        "delta_TN",
        "delta_FN",
    }
    run_required = {"run_slot", "raw_TN_at_FN95", "raw_FN_at_TN68253"}
    missing_pair = sorted(pair_required.difference(pairs.columns))
    missing_run = sorted(run_required.difference(runs.columns))
    if missing_pair or missing_run:
        raise CanonicalInputError(
            f"Raw/calibrated audit missing columns: pairs={missing_pair}, runs={missing_run}"
        )
    if runs["run_slot"].astype(str).duplicated().any():
        raise CanonicalInputError("Run metrics contain duplicate run_slot values")
    lookup = runs.set_index(runs["run_slot"].astype(str))

    records: list[dict[str, Any]] = []
    for pair in pairs.itertuples(index=False):
        treatment = str(pair.t_run_slot)
        control = str(pair.control_run_slot)
        if treatment not in lookup.index or control not in lookup.index:
            raise CanonicalInputError(
                f"Raw/calibrated audit cannot resolve {treatment}/{control}"
            )
        raw_tn = int(lookup.loc[treatment, "raw_TN_at_FN95"]) - int(
            lookup.loc[control, "raw_TN_at_FN95"]
        )
        raw_fn = int(lookup.loc[treatment, "raw_FN_at_TN68253"]) - int(
            lookup.loc[control, "raw_FN_at_TN68253"]
        )
        calibrated_tn = int(pair.delta_TN)
        calibrated_fn = int(pair.delta_FN)
        records.append(
            {
                "triad_id": str(pair.triad_id),
                "control": str(pair.control),
                "t_run_slot": treatment,
                "control_run_slot": control,
                "delta_TN_calibrated": calibrated_tn,
                "delta_FN_calibrated": calibrated_fn,
                "delta_TN_raw": raw_tn,
                "delta_FN_raw": raw_fn,
                "raw_matches_calibrated_effect": bool(
                    raw_tn == calibrated_tn and raw_fn == calibrated_fn
                ),
            }
        )
    pair_audit = pd.DataFrame(records)
    triad_rows: list[dict[str, Any]] = []
    for triad_id, group in pair_audit.groupby("triad_id", sort=True):
        if set(group["control"].astype(str)) != {"R1", "R2"} or len(group) != 2:
            raise CanonicalInputError(
                f"{triad_id} must contain exactly one R1 and one R2 audit row"
            )
        strict_calibrated = bool(
            (group["delta_TN_calibrated"] > 0).all()
            and (group["delta_FN_calibrated"] <= 0).all()
        )
        strict_raw = bool(
            (group["delta_TN_raw"] > 0).all()
            and (group["delta_FN_raw"] <= 0).all()
        )
        triad_rows.append(
            {
                "triad_id": str(triad_id),
                "strict_two_sided_benefit_raw": strict_raw,
                "strict_two_sided_benefit_calibrated": strict_calibrated,
                "raw_calibrated_strict_class_agree": bool(
                    strict_raw == strict_calibrated
                ),
                "all_pair_effects_exact": bool(
                    group["raw_matches_calibrated_effect"].all()
                ),
            }
        )
    return pair_audit, pd.DataFrame(triad_rows)


def _verify_selection_index(
    selection_root: Path, canonical_runs: pd.DataFrame
) -> dict[str, str]:
    index_path = selection_root.parent / "selection_index.csv"
    if not index_path.is_file():
        raise CanonicalInputError(f"Missing frozen selection index: {index_path}")
    index = pd.read_csv(index_path, dtype={"run_slot": str, "sha256": str})
    required = {"run_slot", "sha256"}
    if not required.issubset(index.columns) or len(index) != len(canonical_runs):
        raise CanonicalInputError("Frozen selection index does not cover canonical runs")
    if index["run_slot"].duplicated().any():
        raise CanonicalInputError("Frozen selection index contains duplicate run slots")
    expected_runs = set(canonical_runs["run_slot"].astype(str))
    if set(index["run_slot"].astype(str)) != expected_runs:
        raise CanonicalInputError("Selection index run slots differ from canonical runs")
    by_run = index.set_index("run_slot")
    hashes: dict[str, str] = {}
    for run_slot in sorted(expected_runs):
        path = selection_root / run_slot / "selection_manifest.csv"
        actual = sha256_file(path)
        expected = str(by_run.loc[run_slot, "sha256"]).upper()
        if actual != expected:
            raise CanonicalInputError(
                f"Selection SHA mismatch for {run_slot}: {actual} != {expected}"
            )
        hashes[run_slot] = actual
    if "selection_sha256" in canonical_runs.columns:
        canonical_hashes = canonical_runs.set_index("run_slot")["selection_sha256"]
        for run_slot, digest in hashes.items():
            if str(canonical_hashes.loc[run_slot]).upper() != digest:
                raise CanonicalInputError(
                    f"Canonical selection SHA mismatch for {run_slot}"
                )
    return hashes


def _selection_set_feature_summary(
    run_summary: pd.DataFrame,
    treatment_sets: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    treatment_all = run_summary.loc[
        (run_summary["arm"] == "T") & (run_summary["scope"] == "all")
    ].copy()
    numeric = [
        column
        for column in treatment_all.columns
        if column.startswith("mean_") or column.startswith("share_")
    ]
    rows: list[dict[str, Any]] = []
    for digest, group in treatment_all.groupby("sample_set_digest", sort=True):
        record: dict[str, Any] = {
            "sample_set_digest": digest,
            "representative_condition_slot": "|".join(
                sorted(set(group["condition_slot"].astype(str)))
            ),
            "selected_count": int(group["selected_count"].iloc[0]),
        }
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce")
            if values.max() - values.min() > 1e-12:
                raise CanonicalInputError(
                    f"Identical sample set {digest} has inconsistent {column}"
                )
            record[column] = float(values.iloc[0])
        rows.append(record)
    result = pd.DataFrame(rows).merge(
        outcomes, on="sample_set_digest", how="left", validate="one_to_one"
    )
    if len(result) != treatment_sets["sample_set_digest"].nunique():
        raise AssertionError("Selection-set feature summary lost Treatment sets")
    result["statistical_unit"] = "unique_treatment_sample_set"
    return result


def _condition_seed_tier_matrix(tiers: pd.DataFrame) -> pd.DataFrame:
    matrix = tiers.pivot_table(
        index=["phase", "condition_slot", "condition_id", "method", "budget"],
        columns="training_seed",
        values="cohort_code",
        aggfunc="first",
        fill_value="NOT_RUN",
    ).reset_index()
    matrix.columns = [str(column) for column in matrix.columns]
    return matrix


def _group_numeric_summary(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    numeric_columns: Sequence[str],
    statistical_unit: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(group_columns), dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metadata = dict(zip(group_columns, keys, strict=True))
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    **metadata,
                    "feature": column,
                    "n": int(len(values)),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "statistical_unit": statistical_unit,
                }
            )
    return pd.DataFrame(rows)


def build_prediction_tail_extreme_contrasts(
    prediction_tail_detail: pd.DataFrame,
    tiers: pd.DataFrame,
    *,
    bootstrap_samples: int = 2000,
) -> pd.DataFrame:
    """Compare S/H prediction shifts without pooling labels, scopes, or scores."""

    features = ("mean_shift", "median_shift", "beneficial_rate", "harmed_rate")
    required = {
        "triad_id",
        "control",
        "label",
        "scope",
        "score_type",
        "phase",
        "training_seed",
        "machine_pair",
        "any_resumed",
        *features,
    }
    missing = sorted(required.difference(prediction_tail_detail.columns))
    if missing:
        raise CanonicalInputError(
            f"Prediction-tail detail missing columns: {missing}"
        )
    tier_columns = ["triad_id", "cohort_code", "budget"]
    missing_tier = sorted(set(tier_columns).difference(tiers.columns))
    if missing_tier:
        raise CanonicalInputError(f"Tier table missing columns: {missing_tier}")
    detail = prediction_tail_detail.merge(
        tiers[tier_columns], on="triad_id", how="left", validate="many_to_one"
    )
    if detail[["cohort_code", "budget"]].isna().any().any():
        raise CanonicalInputError("Prediction-tail detail contains unknown triads")

    tables: list[pd.DataFrame] = []
    grouping = ["label", "scope", "score_type"]
    for keys, group in detail.groupby(grouping, dropna=False, sort=True):
        contrast = summarize_extreme_feature_contrasts(
            group,
            features,
            random_seed=20260801,
            bootstrap_samples=bootstrap_samples,
        )
        if contrast.empty:
            continue
        label, tail_scope, score_type = keys
        contrast.insert(0, "score_type", score_type)
        contrast.insert(0, "tail_scope", tail_scope)
        contrast.insert(0, "label", label)
        tables.append(contrast)
    if not tables:
        raise CanonicalInputError("Prediction-tail detail has no S/H contrasts")
    return pd.concat(tables, ignore_index=True)


def _build_leaveout_suite(
    features: pd.DataFrame,
    feature_columns: Sequence[str],
    group_columns: Sequence[str],
) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("all", features)]
    if {"phase", "machine_pair"}.issubset(features.columns):
        scopes.append(
            (
                "phase_A_same_machine",
                features.loc[
                    (features["phase"] == "A")
                    & (features["machine_pair"] == "same_machine")
                ],
            )
        )
    for analysis_scope, scoped in scopes:
        if scoped.empty:
            continue
        for group_column in group_columns:
            if group_column not in scoped.columns:
                continue
            table = summarize_leave_one_group_out(
                scoped, feature_columns, group_column=group_column
            )
            if table.empty:
                continue
            source_column = f"omitted_{group_column}"
            table.insert(0, "omitted_value", table[source_column].astype(str))
            table.insert(0, "leaveout_dimension", group_column)
            table.insert(0, "analysis_scope", analysis_scope)
            tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _build_findings(
    tiers: pd.DataFrame,
    selection_outcomes: pd.DataFrame,
    training_contrasts: pd.DataFrame,
    outcome_contrasts: pd.DataFrame,
    *,
    selection_contrasts: pd.DataFrame | None = None,
    prediction_tail_contrasts: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    def pair_values(frame: pd.DataFrame, column: str, digits: int = 6) -> str:
        if frame.empty or column not in frame.columns:
            return "无可用数值"
        parts = []
        for row in frame.sort_values("control").itertuples(index=False):
            value = float(getattr(row, column))
            parts.append(f"{row.control}={value:+.{digits}f}")
        return "、".join(parts)

    counts = tiers["cohort_code"].value_counts().to_dict()
    crossing = selection_outcomes.loc[
        selection_outcomes["spans_exceptional_and_harmful"]
    ]
    findings: list[dict[str, Any]] = [
        {
            "finding_id": "F01",
            "status": "REPEATED_PATTERN",
            "title": "极好与有害triad数量足够形成集合对照",
            "evidence": (
                f"80个triad互斥分为S={counts.get('S', 0)}、A={counts.get('A', 0)}、"
                f"B={counts.get('B', 0)}、M={counts.get('M', 0)}、H={counts.get('H', 0)}。"
            ),
            "boundary": "档位由训练后结果定义，只用于集合画像，不能作为训练前预测标签。",
        },
        {
            "finding_id": "F02",
            "status": "REPEATED_PATTERN" if len(crossing) else "INSUFFICIENT_EVIDENCE",
            "title": "相同Treatment图片可随seed从S翻转到H",
            "evidence": (
                f"共有{len(crossing)}套完全相同的Treatment样本集合同时产生过S级和H级结果；"
                "固定图片组成不足以单独解释成功。"
            ),
            "boundary": "选样特征的独立统计单位是25套唯一集合，不是80个重复seed的triad。",
        },
    ]

    loss = training_contrasts.loc[
        (training_contrasts["analysis_scope"] == "all")
        & (training_contrasts["feature"] == "train_loss_extra_drop_epoch121_to_200")
    ]
    robust = training_contrasts.loc[
        (training_contrasts["analysis_scope"] == "all")
        & (
            training_contrasts["feature"]
            == "train_loss_robust_drop_121_130_to_191_200"
        )
    ]
    loss_both = (
        len(loss) == 2
        and (loss["mean_difference_S_minus_H"] < 0).all()
        and (loss["bootstrap_95_high"] < 0).all()
    )
    robust_both = (
        len(robust) == 2
        and (robust["mean_difference_S_minus_H"] < 0).all()
        and (robust["bootstrap_95_high"] < 0).all()
    )
    findings.append(
        {
            "finding_id": "F03",
            "status": "REPEATED_PATTERN" if loss_both and robust_both else "PARTIAL_PATTERN",
            "title": "S级后期继续压训练损失较少，但尚非跨seed定律",
            "evidence": (
                "epoch121→200端点S−H为"
                f"{pair_values(loss, 'mean_difference_S_minus_H')}；"
                + (
                    "首末10轮稳健指标的95% bootstrap区间也均保持同向。"
                    if robust_both
                    else (
                        "首末10轮稳健S−H为"
                        f"{pair_values(robust, 'mean_difference_S_minus_H')}，"
                        "但两个对照的95%区间均未排除零。"
                    )
                )
            ),
            "boundary": "这是训练后候选预警信号；不能从整体loss曲线反推出选样价值。",
        }
    )
    threshold = outcome_contrasts.loc[
        (outcome_contrasts["analysis_scope"] == "all")
        & (outcome_contrasts["feature"] == "delta_threshold")
    ]
    threshold_both = (
        len(threshold) == 2
        and (threshold["mean_difference_S_minus_H"] > 0).all()
        and (threshold["bootstrap_95_low"] > 0).all()
    )
    findings.append(
        {
            "finding_id": "F04",
            "status": "REPEATED_PATTERN" if threshold_both else "PARTIAL_PATTERN",
            "title": "S级共同结果是安全工作阈值被抬高",
            "evidence": (
                "全部S/H triad的校准阈值变化S−H为"
                f"{pair_values(threshold, 'mean_difference_S_minus_H')}，"
                "两套triad级bootstrap区间均高于零。"
            ),
            "boundary": "工作阈值属于训练后的直接结果诊断，不能作为训练前样本价值分数。",
        }
    )
    confirmation = tiers.loc[tiers["discovery_or_confirmation"] == "confirmation"]
    findings.append(
        {
            "finding_id": "F05",
            "status": "COUNTEREXAMPLE_FOUND"
            if int((confirmation["cohort_code"] == "S").sum()) == 0
            else "PARTIAL_PATTERN",
            "title": "五个确认seed没有复现S级结果",
            "evidence": (
                f"{len(confirmation)}个confirmation triad中S级为"
                f"{int((confirmation['cohort_code'] == 'S').sum())}；成功高度依赖训练状态。"
            ),
            "boundary": "Phase C只压力测试A02固定selection，不能外推为所有方法的确认期成功率。",
        }
    )
    if selection_contrasts is not None and not selection_contrasts.empty:
        direction = selection_contrasts.loc[
            (selection_contrasts["analysis_scope"] == "phase_A_same_machine")
            & (
                selection_contrasts["feature"]
                == "delta_mean_score_direction_changes"
            )
        ]
        late_error = selection_contrasts.loc[
            (selection_contrasts["analysis_scope"] == "phase_A_same_machine")
            & (
                selection_contrasts["feature"]
                == "delta_mean_error_rate_late_161_200"
            )
        ]
        candidate_direction = (
            not direction.empty
            and not late_error.empty
            and (direction["mean_difference_S_minus_H"] < 0).all()
            and (late_error["mean_difference_S_minus_H"] < 0).all()
        )
        findings.append(
            {
                "finding_id": "F06",
                "status": "PARTIAL_PATTERN"
                if candidate_direction
                else "INSUFFICIENT_EVIDENCE",
                "title": "较少震荡和较低后期错误是成功selection的候选画像",
                "evidence": (
                    "Phase A同机S−H：方向改变"
                    f"{pair_values(direction, 'mean_difference_S_minus_H', 3)}；"
                    "161–200轮错误率"
                    f"{pair_values(late_error, 'mean_difference_S_minus_H', 5)}。"
                ),
                "boundary": "相同selection可跨seed翻转，且这些轨迹是冻结OOF模型上的属性；它们不是充分条件或单图因果分数。",
            }
        )

    if prediction_tail_contrasts is not None and not prediction_tail_contrasts.empty:
        defect_tail = prediction_tail_contrasts.loc[
            (prediction_tail_contrasts["analysis_scope"] == "all")
            & (prediction_tail_contrasts["label"] == "defect")
            & (prediction_tail_contrasts["tail_scope"] == "operational")
            & (prediction_tail_contrasts["score_type"] == "raw")
            & (prediction_tail_contrasts["feature"] == "mean_shift")
        ]
        defect_repeated = (
            len(defect_tail) == 2
            and (defect_tail["mean_difference_S_minus_H"] > 0).all()
            and (defect_tail["bootstrap_95_low"] > 0).all()
        )
        findings.append(
            {
                "finding_id": "F07",
                "status": "REPEATED_PATTERN" if defect_repeated else "PARTIAL_PATTERN",
                "title": "极好组最稳定地保护了最弱95张defect",
                "evidence": (
                    "95张固定operational defect尾部的raw mean-shift S−H为"
                    f"{pair_values(defect_tail, 'mean_difference_S_minus_H')}，"
                    "两套bootstrap区间均高于零。"
                ),
                "boundary": "这是训练后固定尾部集合的机制证据，不代表每张训练defect都具有固定因果价值。",
            }
        )
        normal_tail = prediction_tail_contrasts.loc[
            (prediction_tail_contrasts["analysis_scope"] == "all")
            & (prediction_tail_contrasts["label"] == "normal")
            & (prediction_tail_contrasts["tail_scope"] == "operational")
            & (prediction_tail_contrasts["score_type"] == "raw")
            & (prediction_tail_contrasts["feature"] == "mean_shift")
        ]
        harmful_compresses_more = len(normal_tail) == 2 and (
            normal_tail["mean_difference_S_minus_H"] > 0
        ).all()
        findings.append(
            {
                "finding_id": "F08",
                "status": "COUNTEREXAMPLE_FOUND"
                if harmful_compresses_more
                else "PARTIAL_PATTERN",
                "title": "单纯把困难normal压得更低并不能解释成功",
                "evidence": (
                    "31,747张固定operational normal尾部的raw mean-shift S−H为"
                    f"{pair_values(normal_tail, 'mean_difference_S_minus_H')}；"
                    "正值表示H级反而把normal压得更多。"
                ),
                "boundary": "关键区别是弱defect是否同时下沉；normal压低不能脱离defect保护单独评价。",
            }
        )
    return findings


def run_extreme_cohort_analysis(
    *,
    v2_report_dir: str | Path,
    expert_package_root: str | Path,
    selection_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Execute the focused full-data analysis and publish one atomic report."""

    v2_root = Path(v2_report_dir).resolve()
    expert_root = Path(expert_package_root).resolve()
    selections = Path(selection_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists() or output.with_name(output.name + ".inprogress").exists():
        raise FileExistsError(f"Refusing to overwrite analysis output: {output}")
    for source in (v2_root, expert_root, selections):
        try:
            output.relative_to(source)
        except ValueError:
            continue
        raise CanonicalInputError(f"Output must not be inside read-only source: {source}")

    v2_paths = verify_manifested_inputs(
        v2_root, v2_root / "manifest.json", REQUIRED_V2_TABLES
    )
    expert_paths = verify_manifested_inputs(
        expert_root, expert_root / "FILE_MANIFEST.csv", REQUIRED_EXPERT_INPUTS
    )

    tables = {
        Path(relative).stem: pd.read_csv(path)
        for relative, path in v2_paths.items()
    }
    triad_deltas = tables["triad_control_deltas"]
    canonical_runs = tables["canonical_run_metrics"]
    if len(triad_deltas) != 160 or len(canonical_runs) != 240:
        raise CanonicalInputError("Formal v3 analysis requires 160 pairs and 240 runs")
    _verify_selection_index(selections, canonical_runs)

    tiers = classify_triad_cohorts(triad_deltas)
    expected_counts = {"S": 12, "A": 3, "B": 1, "M": 41, "H": 23}
    actual_counts = tiers["cohort_code"].value_counts().to_dict()
    if actual_counts != expected_counts:
        raise CanonicalInputError(
            f"Frozen cohort regression mismatch: {actual_counts} != {expected_counts}"
        )
    training_features = build_training_window_features(
        tables["paired_epoch_differences"], tiers
    )
    training_features = training_features.merge(
        tiers[["triad_id", "t_machine_id"]],
        on="triad_id",
        how="left",
        validate="many_to_one",
    )

    metadata = json.loads(
        expert_paths["inputs/oof_probabilities_metadata.json"].read_text(
            encoding="utf-8"
        )
    )
    shape = tuple(int(value) for value in metadata.get("shape", []))
    dtype = str(metadata.get("dtype", ""))
    if shape != (200, 120000) or dtype != "float64" or metadata.get("epoch_base") != 1:
        raise CanonicalInputError(f"Unexpected OOF metadata: {metadata}")
    mmap_path = expert_paths["inputs/oof_probabilities_float64.mmap"]
    expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if mmap_path.stat().st_size != expected_bytes:
        raise CanonicalInputError("OOF mmap byte size does not match metadata")
    sample_ids = pd.read_csv(
        expert_paths["inputs/sample_ids.csv"], dtype={"sample_id": str, "oof_fold": str}
    )
    sample_ids["oof_fold"] = sample_ids["oof_fold"].str.zfill(2)
    if (
        len(sample_ids) != 120000
        or sample_ids["sample_id"].nunique() != 120000
        or sample_ids["y_true"].value_counts().to_dict() != {1: 60000, 0: 60000}
        or sorted(sample_ids["oof_fold"].unique())
        != [f"{index:02d}" for index in range(10)]
    ):
        raise CanonicalInputError("OOF sample index identity check failed")
    probabilities = np.memmap(
        mmap_path, dtype=np.float64, mode="r", shape=shape
    )
    fold01 = sample_ids["oof_fold"].to_numpy() == "01"
    if int(fold01.sum()) != 12003:
        raise CanonicalInputError("fold01 affected-sample count is not 12,003")
    if not np.array_equal(probabilities[176, fold01], probabilities[177, fold01]):
        raise CanonicalInputError("fold01 epoch177/178 repair provenance changed")
    if np.array_equal(probabilities[176, ~fold01], probabilities[177, ~fold01]):
        raise CanonicalInputError("epoch177/178 unexpectedly duplicate outside fold01")
    operational_dynamics, epoch_operational, oof_audit = (
        compute_operational_sample_dynamics(probabilities, sample_ids, fn_limit=285)
    )
    del probabilities
    if (
        int(epoch_operational.loc[epoch_operational["TN_at_FN_limit"].idxmax(), "epoch"])
        != 149
    ):
        raise CanonicalInputError("OOF operational work-point regression changed")
    if not (epoch_operational["actual_FN"] == 285).all():
        raise CanonicalInputError("OOF FN285 work point did not achieve exactly 285 FN")
    frozen_epoch_gap = pd.read_csv(expert_paths["inputs/epoch_gap_metrics.csv"])
    if len(frozen_epoch_gap) != 200 or frozen_epoch_gap["epoch"].tolist() != list(
        range(1, 201)
    ):
        raise CanonicalInputError("Frozen epoch gap table is not complete 1..200")
    gap_pairs = (
        ("normal_q68_nearest", "normal_q68"),
        ("normal_q90_nearest", "normal_q90"),
        ("defect_q50_nearest", "defect_q50"),
        ("defect_q05_nearest", "defect_q05"),
        ("gap_q68_q050_nearest", "gap_q68_q050"),
        ("tail_gap_q90_q05_nearest", "tail_gap_q90_q05"),
    )
    epoch_join = epoch_operational.merge(
        frozen_epoch_gap, on="epoch", how="left", validate="one_to_one"
    )
    for computed, frozen in gap_pairs:
        if not np.allclose(
            epoch_join[computed].to_numpy(dtype=float),
            epoch_join[frozen].to_numpy(dtype=float),
            rtol=0.0,
            atol=5e-10,
        ):
            maximum = float(
                np.max(np.abs(epoch_join[computed] - epoch_join[frozen]))
            )
            raise CanonicalInputError(
                f"Nearest-quantile OOF audit failed for {frozen}; max={maximum}"
            )

    run_selection_summary, treatment_sets = summarize_selection_operational_features(
        canonical_runs, selections, operational_dynamics, tiers
    )
    selection_pairs = pair_selection_feature_deltas(run_selection_summary, tiers)
    pair_metadata = triad_deltas[
        [
            "triad_id",
            "control",
            "machine_pair",
            "any_resumed",
            "t_machine_id",
        ]
    ].drop_duplicates()
    selection_pairs = selection_pairs.merge(
        pair_metadata,
        on=["triad_id", "control"],
        how="left",
        validate="many_to_one",
    )
    selection_outcomes = summarize_selection_set_outcomes(treatment_sets)
    selection_outcomes["outcome_profile"] = np.select(
        [
            selection_outcomes["spans_exceptional_and_harmful"],
            (selection_outcomes["exceptional_count"] > 0)
            & (selection_outcomes["harmful_count"] == 0),
            (selection_outcomes["harmful_count"] > 0)
            & (selection_outcomes["exceptional_count"] == 0),
        ],
        ["SPANS_S_H", "S_WITHOUT_H", "H_WITHOUT_S"],
        default="NO_EXTREME_OUTCOME",
    )
    fixed_flips = build_fixed_selection_seed_flips(treatment_sets)
    selection_set_features = _selection_set_feature_summary(
        run_selection_summary, treatment_sets, selection_outcomes
    )

    outcome_pairs = build_outcome_mechanism_pairs(
        triad_deltas,
        canonical_runs,
        tables["calibration_diagnostics"],
        tiers,
    )
    training_contrast_columns = (
        "train_loss_extra_drop_epoch121_to_200",
        "train_loss_robust_drop_121_130_to_191_200",
        "train_loss_slope_121_200",
        "val_loss_slope_121_200",
        "val_loss_late_rebound",
        "mean_delta_top1_e161_200",
    )
    training_contrasts = summarize_extreme_feature_contrasts(
        training_features,
        training_contrast_columns,
        bootstrap_samples=5000,
    )
    training_stratified = build_stratified_extreme_contrasts(
        training_features, training_contrast_columns
    )
    training_leaveouts = _build_leaveout_suite(
        training_features,
        training_contrast_columns,
        ("training_seed", "condition_slot", "t_machine_id"),
    )
    outcome_contrasts = summarize_extreme_feature_contrasts(
        outcome_pairs,
        ("delta_threshold", "delta_raw_threshold", "delta_auroc", "delta_auroc_raw"),
        bootstrap_samples=5000,
    )
    outcome_stratified = build_stratified_extreme_contrasts(
        outcome_pairs,
        ("delta_threshold", "delta_raw_threshold", "delta_auroc", "delta_auroc_raw"),
    )
    outcome_leaveouts = _build_leaveout_suite(
        outcome_pairs,
        ("delta_threshold", "delta_raw_threshold", "delta_auroc", "delta_auroc_raw"),
        ("training_seed", "condition_slot", "t_machine_id"),
    )
    normal_selection_pairs = selection_pairs.loc[selection_pairs["scope"] == "normal"]
    selection_contrast_columns = tuple(
        column
        for column in (
            "delta_mean_operational_error_rate",
            "delta_mean_operational_forgetting_count",
            "delta_mean_operational_recovery_count",
            "delta_mean_score_direction_changes",
            "delta_mean_operational_correction",
            "delta_mean_error_rate_late_161_200",
            "delta_share_corrected",
            "delta_share_persistent_wrong",
            "delta_share_deteriorating",
        )
        if column in normal_selection_pairs.columns
    )
    selection_contrasts = summarize_extreme_feature_contrasts(
        normal_selection_pairs,
        selection_contrast_columns,
        bootstrap_samples=5000,
    )
    selection_stratified = build_stratified_extreme_contrasts(
        normal_selection_pairs, selection_contrast_columns
    )
    selection_leaveouts = _build_leaveout_suite(
        normal_selection_pairs,
        selection_contrast_columns,
        ("training_seed", "condition_slot", "t_machine_id"),
    )
    defect_selection_pairs = selection_pairs.loc[selection_pairs["scope"] == "defect"]
    defect_selection_contrasts = summarize_extreme_feature_contrasts(
        defect_selection_pairs,
        selection_contrast_columns,
        bootstrap_samples=5000,
    )

    prediction_tail_contrasts = build_prediction_tail_extreme_contrasts(
        tables["prediction_tail_detail"], tiers, bootstrap_samples=5000
    )

    frozen_selection_pairs = tables["selection_value_effects"].merge(
        tiers[["triad_id", "cohort_code", "budget", "t_machine_id"]],
        on="triad_id",
        how="left",
        validate="many_to_one",
    )
    frozen_selection_columns = tuple(
        column
        for column in (
            "selection_delta_mean_p_defect_mean",
            "selection_delta_correct_rate_mean",
            "selection_delta_std_p_defect_mean",
            "selection_delta_mean_gap_critical_score",
            "selection_delta_mean_gap_guard_score",
        )
        if column in frozen_selection_pairs.columns
    )
    frozen_selection_contrasts = summarize_extreme_feature_contrasts(
        frozen_selection_pairs,
        frozen_selection_columns,
        bootstrap_samples=5000,
    )

    treatment_composition = tables["selection_composition"].loc[
        tables["selection_composition"]["arm"] == "T"
    ].merge(
        treatment_sets[
            ["run_slot", "sample_set_digest", "cohort_code"]
        ],
        on="run_slot",
        how="left",
        validate="many_to_one",
    ).merge(
        selection_outcomes[["sample_set_digest", "outcome_profile"]],
        on="sample_set_digest",
        how="left",
        validate="many_to_one",
    )
    unique_selection_composition = treatment_composition.drop_duplicates(
        [
            "sample_set_digest",
            "y_true",
            "replay_role",
            "dynamic_bucket",
            "count",
            "share",
        ]
    ).reset_index(drop=True)

    tier_composition = build_tier_composition_audit(tiers)
    extreme_shortlist = tiers.loc[
        tiers["cohort_code"].isin(["S", "A", "B", "H"])
    ].copy()
    extreme_shortlist["min_delta_TN_across_controls"] = extreme_shortlist[
        ["delta_TN_R1", "delta_TN_R2"]
    ].min(axis=1)
    extreme_shortlist["max_delta_TN_across_controls"] = extreme_shortlist[
        ["delta_TN_R1", "delta_TN_R2"]
    ].max(axis=1)
    extreme_shortlist["max_delta_FN_across_controls"] = extreme_shortlist[
        ["delta_FN_R1", "delta_FN_R2"]
    ].max(axis=1)
    extreme_shortlist["min_delta_FN_across_controls"] = extreme_shortlist[
        ["delta_FN_R1", "delta_FN_R2"]
    ].min(axis=1)
    extreme_shortlist["selection_reason"] = np.where(
        extreme_shortlist["cohort_code"] == "H",
        "both_controls_less_TN_and_more_FN",
        "positive_or_high_value_tier",
    )
    tier_method_composition = (
        tiers.groupby(
            ["cohort_code", "phase", "method", "budget"],
            dropna=False,
        )["triad_id"]
        .nunique()
        .rename("triad_count")
        .reset_index()
        .sort_values(["cohort_code", "triad_count", "phase", "method"], ascending=[True, False, True, True])
        .reset_index(drop=True)
    )
    condition_seed_matrix = _condition_seed_tier_matrix(tiers)
    outcome_tier_summary = _group_numeric_summary(
        outcome_pairs,
        group_columns=("cohort_code", "control", "phase"),
        numeric_columns=(
            "delta_TN",
            "delta_FN",
            "delta_threshold",
            "delta_raw_threshold",
            "delta_auroc",
            "delta_auroc_raw",
        ),
        statistical_unit="triad",
    )
    findings = _build_findings(
        tiers,
        selection_outcomes,
        training_contrasts,
        outcome_contrasts,
        selection_contrasts=selection_contrasts,
        prediction_tail_contrasts=prediction_tail_contrasts,
    )

    report_tables: dict[str, pd.DataFrame] = {
        "triad_performance_tiers": tiers,
        "extreme_triads_shortlist": extreme_shortlist,
        "tier_composition_audit": tier_composition,
        "tier_method_composition": tier_method_composition,
        "condition_seed_tier_matrix": condition_seed_matrix,
        "training_window_features": training_features,
        "training_extreme_contrasts": training_contrasts,
        "training_stratified_extreme_contrasts": training_stratified,
        "training_leave_one_group_out": training_leaveouts,
        "late_loss_definition_audit": training_features[
            [
                "triad_id",
                "condition_slot",
                "phase",
                "training_seed",
                "control",
                "cohort_code",
                "train_loss_extra_drop_epoch121_to_200",
                "train_loss_robust_drop_121_130_to_191_200",
                "train_loss_slope_121_200",
            ]
        ],
        "oof_epoch_operational_fn285": epoch_operational,
        "oof_operational_sample_dynamics": operational_dynamics,
        "selection_run_operational_summary": run_selection_summary,
        "selection_feature_pair_deltas": selection_pairs,
        "selection_extreme_contrasts": selection_contrasts,
        "selection_stratified_extreme_contrasts": selection_stratified,
        "selection_leave_one_group_out": selection_leaveouts,
        "defect_selection_extreme_contrasts": defect_selection_contrasts,
        "frozen_selection_value_effects": frozen_selection_pairs,
        "frozen_selection_value_extreme_contrasts": frozen_selection_contrasts,
        "treatment_selection_composition": treatment_composition,
        "unique_selection_set_composition": unique_selection_composition,
        "treatment_selection_sets": treatment_sets,
        "selection_set_outcomes": selection_outcomes,
        "fixed_selection_seed_flips": fixed_flips,
        "selection_set_feature_summary": selection_set_features,
        "outcome_mechanism_pairs": outcome_pairs,
        "outcome_mechanism_tier_summary": outcome_tier_summary,
        "outcome_extreme_contrasts": outcome_contrasts,
        "outcome_stratified_extreme_contrasts": outcome_stratified,
        "outcome_leave_one_group_out": outcome_leaveouts,
        "prediction_tail_detail": tables["prediction_tail_detail"],
        "prediction_tail_extreme_contrasts": prediction_tail_contrasts,
        "r2_overlap_power_audit": tables["r2_overlap_power_audit"],
        "prediction_tail_summary": tables["prediction_tail_summary"],
        "training_curve_summary": tables["training_curve_summary"],
        "candidate_pattern_registry": pd.DataFrame(findings),
    }
    report_metadata: dict[str, Any] = {
        "schema_version": "stage1_gapvalue240_extreme_cohort_analysis_v3",
        "validated_runs": 240,
        "triads": 80,
        "comparisons": 160,
        "cohort_counts": expected_counts,
        "unique_treatment_sample_sets": int(
            treatment_sets["sample_set_digest"].nunique()
        ),
        "selection_sets_spanning_S_and_H": int(
            selection_outcomes["spans_exceptional_and_harmful"].sum()
        ),
        "oof_audit": oof_audit,
        "v2_report_dir": str(v2_root),
        "expert_package_root": str(expert_root),
        "selection_root": str(selections),
        "source_hashes": {
            **{
                relative: sha256_file(path)
                for relative, path in v2_paths.items()
            },
            **{
                relative: sha256_file(path)
                for relative, path in expert_paths.items()
            },
        },
        "scientific_boundaries": [
            "val_op internal discovery only; no blind/external claim",
            "R1 and R2 remain separate",
            "selection features use unique sample sets; outcome effects use triads",
            "threshold and AUROC are post-training diagnostics only",
            "Phase B extreme comparisons are machine-confounded exploratory evidence",
        ],
        "conclusion_boundary": (
            "仅限240-run冻结实验的val_op内部描述与机制分析；没有blind/external test。"
            "S/H由结果事后分层，R1/R2始终分开；Phase B极端组比较存在机器混杂。"
        ),
    }

    from .extreme_reporting import build_extreme_report

    result = build_extreme_report(
        output,
        tables=report_tables,
        metadata=report_metadata,
        findings=findings,
    )
    return {
        "status": "PASS",
        "output_dir": str(result),
        "cohort_counts": expected_counts,
        "unique_treatment_sample_sets": report_metadata[
            "unique_treatment_sample_sets"
        ],
        "selection_sets_spanning_S_and_H": report_metadata[
            "selection_sets_spanning_S_and_H"
        ],
    }
