"""Build the canonical 80-triad feature matrix without outcome leakage.

All inputs are already-derived, read-only evidence tables.  The builder keeps
R1/R2, T/R1/R2 and normal/guard replay roles explicit in feature names.  It
does not publish anything unless :func:`publish_unified_feature_matrix` is
called explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd


class UnifiedFeatureMatrixError(ValueError):
    """Raised when a source table would create an ambiguous or leaky matrix."""


@dataclass(frozen=True)
class UnifiedFeatureMatrixResult:
    matrix: pd.DataFrame
    feature_registry: pd.DataFrame
    role_registry: pd.DataFrame
    audit: dict[str, object]


_ARMS = ("T", "R1", "R2")
_ROLES = ("normal_replay", "defect_guard")
_SELECTION_KEYS = {
    "run_slot",
    "triad_id",
    "phase",
    "condition_id",
    "method",
    "budget",
    "guard_ratio",
    "arm",
    "training_seed",
    "selection_seed",
    "replay_role",
}
_OUTCOME_FIELDS = {
    "actual_FN_at_FN95",
    "actual_TN_at_TN68253",
    "delta_TN_R1",
    "delta_FN_R1",
    "delta_TN_R2",
    "delta_FN_R2",
    "G_TN",
    "G_FN",
    "HARM_TN",
    "HARM_FN",
    "dual_improvement",
    "high_value",
    "dual_harm",
    "exclusive_cohort",
    "tie_group_at_FN95",
    "tie_group_at_TN68253",
}
_EXECUTION_TOKENS = (
    "machine",
    "resume",
    "snapshot",
    "package",
    "release",
    "attempt",
)
_LEAKY_PREDICTOR = re.compile(
    r"(?:"
    r"actual_(?:TN|FN)|delta_+(?:TN|FN)(?:_|$)|TN_at_FN|FN_at_TN|"
    r"(?:^|__)G_(?:TN|FN)(?:$|__)|(?:^|__)HARM_(?:TN|FN)(?:$|__)|"
    r"dual_improvement|dual_harm|high_value|exclusive_cohort|raw_frontier|"
    r"final_prediction|val_op_prediction|operational_threshold|score_raw|"
    r"roc_auc|auroc|auprc"
    r")",
    flags=re.IGNORECASE,
)


def _check_columns(frame: pd.DataFrame, *, name: str) -> None:
    duplicates = frame.columns[frame.columns.duplicated()].tolist()
    if duplicates:
        raise UnifiedFeatureMatrixError(f"{name} has duplicate columns: {duplicates}")


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], *, name: str
) -> None:
    _check_columns(frame, name=name)
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise UnifiedFeatureMatrixError(f"{name} missing columns: {missing}")


def _assert_id_coverage(
    frame: pd.DataFrame,
    expected_ids: set[str],
    *,
    name: str,
    unique: bool = False,
) -> None:
    _require_columns(frame, ["triad_id"], name=name)
    ids = frame["triad_id"].astype(str)
    if unique and ids.duplicated().any():
        raise UnifiedFeatureMatrixError(f"{name} has duplicate triad IDs")
    if set(ids) != expected_ids:
        raise UnifiedFeatureMatrixError(f"{name} triad IDs do not match outcomes")


def _slug(value: object) -> str:
    text = str(value).strip()
    slug = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    return slug or "empty"


def _value_slugs(values: Iterable[object]) -> dict[str, str]:
    strings = sorted({str(value) for value in values})
    initial = {value: _slug(value) for value in strings}
    counts = pd.Series(list(initial.values())).value_counts().to_dict()
    return {
        value: (
            slug
            if counts[slug] == 1
            else f"{slug}_{sha1(value.encode('utf-8')).hexdigest()[:8]}"
        )
        for value, slug in initial.items()
    }


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _numeric(frame: pd.DataFrame, columns: Iterable[str], *, name: str) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        converted = pd.to_numeric(result[column], errors="coerce")
        invalid = result[column].notna() & converted.isna()
        if invalid.any():
            raise UnifiedFeatureMatrixError(f"{name}.{column} is not numeric")
        result[column] = converted
    return result


def _outcome_role(column: str) -> tuple[str, str, int]:
    if column == "triad_id":
        return "TRIAD_IDENTITY", "IDENTITY", 0
    if column in _OUTCOME_FIELDS or _LEAKY_PREDICTOR.search(column):
        return "OUTCOME", "OUTCOME", 200
    lowered = column.lower()
    if any(token in lowered for token in _EXECUTION_TOKENS):
        return "EXECUTION_CONFOUND", "EXECUTION_CONFOUND", 200
    if column in {
        "phase",
        "condition_slot",
        "condition_id",
        "method",
        "budget",
        "guard_ratio",
        "training_seed",
        "selection_seed",
        "discovery_or_confirmation",
        "metric_version",
        "R1_run_slot",
        "R2_run_slot",
        "treatment_run_slot",
        "treatment_selection_sha256",
    }:
        return "DESIGN_OR_LINEAGE", "DESIGN_CONFOUND", 0
    if column in {"prediction_rows", "normal_count", "defect_count"}:
        return "OUTCOME_AUDIT", "OUTCOME_AUDIT", 200
    return "DESCRIPTIVE", "DESCRIPTIVE_NOT_PREDICTOR", 200


def _registry_row(
    *,
    feature: str,
    source_table: str,
    source_field: str,
    feature_family: str,
    available_epoch: int,
    allowed: bool,
    analysis_role: str,
    use: str,
    base_feature: str = "",
    control: str = "",
    arm: str = "",
    replay_role: str = "",
) -> dict[str, object]:
    if allowed and _LEAKY_PREDICTOR.search(feature):
        raise UnifiedFeatureMatrixError(f"leaky predictor is forbidden: {feature}")
    if allowed and analysis_role in {
        "OUTCOME",
        "OUTCOME_AUDIT",
        "EXECUTION_CONFOUND",
        "DESIGN_CONFOUND",
        "SELECTION_DIGEST",
    }:
        raise UnifiedFeatureMatrixError(
            f"leaky predictor role is forbidden: {feature}/{analysis_role}"
        )
    return {
        "feature": feature,
        "feature_family": feature_family,
        "available_epoch": int(available_epoch),
        "allowed_as_predictor": bool(allowed),
        "use": use,
        "base_feature": base_feature or source_field,
        "source_table": source_table,
        "source_field": source_field,
        "analysis_role": analysis_role,
        "control": control,
        "arm": arm,
        "replay_role": replay_role,
    }


def _join_block(
    matrix: pd.DataFrame,
    block: pd.DataFrame,
    *,
    name: str,
) -> pd.DataFrame:
    _assert_id_coverage(
        block,
        set(matrix["triad_id"].astype(str)),
        name=name,
        unique=True,
    )
    overlap = (set(matrix.columns) & set(block.columns)) - {"triad_id"}
    if overlap:
        raise UnifiedFeatureMatrixError(f"{name} would create duplicate columns: {sorted(overlap)}")
    return matrix.merge(block, on="triad_id", how="left", validate="one_to_one")


def _validate_selection_roles(
    outcomes: pd.DataFrame,
    numeric: pd.DataFrame,
    categorical: pd.DataFrame,
    late: pd.DataFrame,
) -> pd.DataFrame:
    required = ["triad_id", "phase", "budget", "guard_ratio", "arm", "replay_role"]
    for name, frame in (
        ("selection_numeric", numeric),
        ("selection_categorical", categorical),
        ("selection_late", late),
    ):
        _require_columns(frame, required, name=name)
        invalid_arms = sorted(set(frame["arm"].astype(str)) - set(_ARMS))
        invalid_roles = sorted(set(frame["replay_role"].astype(str)) - set(_ROLES))
        if invalid_arms or invalid_roles:
            raise UnifiedFeatureMatrixError(
                f"{name} has invalid arms/roles: {invalid_arms}/{invalid_roles}"
            )

    late_cells = late[
        ["triad_id", "arm", "replay_role", "selected_count"]
    ].copy()
    if late_cells.duplicated(["triad_id", "arm", "replay_role"]).any():
        raise UnifiedFeatureMatrixError("selection_late has duplicate arm/replay-role cells")
    for outcome in outcomes.to_dict("records"):
        triad_id = str(outcome["triad_id"])
        phase = str(outcome["phase"])
        budget = int(outcome["budget"])
        guard_ratio = float(outcome["guard_ratio"])
        expected_roles = (
            {"normal_replay", "defect_guard"}
            if phase == "B"
            else {"normal_replay"}
        )
        triad = late_cells.loc[late_cells["triad_id"].astype(str) == triad_id]
        for arm in _ARMS:
            arm_rows = triad.loc[triad["arm"].astype(str) == arm]
            observed_roles = set(arm_rows["replay_role"].astype(str))
            if observed_roles != expected_roles:
                raise UnifiedFeatureMatrixError(
                    f"Phase {phase} {triad_id}/{arm} requires {sorted(expected_roles)}; "
                    f"defect_guard/normal replay roles observed={sorted(observed_roles)}"
                )
            counts = arm_rows.set_index("replay_role")["selected_count"].astype(int)
            expected_defect = int(round(budget * guard_ratio)) if phase == "B" else 0
            expected_normal = budget - expected_defect
            if int(counts["normal_replay"]) != expected_normal:
                raise UnifiedFeatureMatrixError(
                    f"{triad_id}/{arm} normal replay dose differs from contract"
                )
            if phase == "B" and int(counts["defect_guard"]) != expected_defect:
                raise UnifiedFeatureMatrixError(
                    f"Phase B {triad_id}/{arm} defect_guard dose differs from contract"
                )

    expected_cells = set(
        late_cells[["triad_id", "arm", "replay_role"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    for name, frame in (
        ("selection_numeric", numeric),
        ("selection_categorical", categorical),
    ):
        cells = set(
            frame[["triad_id", "arm", "replay_role"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        if cells != expected_cells:
            raise UnifiedFeatureMatrixError(
                f"{name} arm/replay-role cells differ from selection_late"
            )
    return late_cells


def _telemetry_block(
    paired: pd.DataFrame,
    source_registry: pd.DataFrame,
    triad_ids: set[str],
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object]]:
    _require_columns(paired, ["triad_id", "control"], name="paired_telemetry")
    _require_columns(
        source_registry,
        ("feature", "feature_family", "available_epoch", "allowed_as_predictor", "use"),
        name="telemetry_registry",
    )
    if source_registry["feature"].astype(str).duplicated().any():
        raise UnifiedFeatureMatrixError("telemetry_registry has duplicate features")
    if paired.duplicated(["triad_id", "control"]).any():
        raise UnifiedFeatureMatrixError("paired_telemetry has duplicate triad/control rows")
    _assert_id_coverage(paired, triad_ids, name="paired_telemetry")
    counts = paired.groupby("triad_id")["control"].agg(lambda values: set(map(str, values)))
    if not counts.map(lambda values: values == {"R1", "R2"}).all():
        raise UnifiedFeatureMatrixError("paired_telemetry must preserve exactly R1 and R2")

    registered = source_registry.set_index(source_registry["feature"].astype(str))
    available_delta = {
        column[len("delta__") :]: column
        for column in paired.columns
        if str(column).startswith("delta__")
    }
    allowed_bases = {
        str(row.feature)
        for row in source_registry.itertuples(index=False)
        if _as_bool(row.allowed_as_predictor)
    }
    non_paired_config = sorted(allowed_bases - set(available_delta))
    missing = sorted(set(non_paired_config) - set(paired.columns))
    unregistered = sorted(set(available_delta) - set(registered.index))
    if missing or unregistered:
        raise UnifiedFeatureMatrixError(
            f"telemetry registry/source mismatch: missing={missing[:5]}, "
            f"unregistered={unregistered[:5]}"
        )

    triad_order = sorted(triad_ids)
    block_data: dict[str, object] = {"triad_id": triad_order}
    registry_rows: list[dict[str, object]] = []
    paired_predictor_bases = sorted(allowed_bases & set(available_delta))
    for control in ("R1", "R2"):
        control_rows = paired.loc[paired["control"].astype(str) == control].copy()
        control_rows["triad_id"] = control_rows["triad_id"].astype(str)
        control_rows = control_rows.set_index("triad_id").loc[triad_order]
        for base in paired_predictor_bases:
            source_column = available_delta[base]
            feature = f"telemetry__{control}__{source_column}"
            block_data[feature] = pd.to_numeric(
                control_rows[source_column], errors="raise"
            ).to_numpy()
            source = registered.loc[base]
            registry_rows.append(
                _registry_row(
                    feature=feature,
                    source_table="triad_paired_telemetry_deltas",
                    source_field=source_column,
                    feature_family=str(source["feature_family"]),
                    available_epoch=int(source["available_epoch"]),
                    allowed=True,
                    analysis_role="TRAINING_TELEMETRY_PREDICTOR",
                    use=str(source["use"]),
                    base_feature=base,
                    control=control,
                )
            )
    for base in non_paired_config:
        feature = f"training_config__{_slug(base)}"
        values: list[object] = []
        for triad_id in triad_order:
            observed = paired.loc[
                paired["triad_id"].astype(str) == triad_id, base
            ].drop_duplicates()
            if len(observed) != 1:
                raise UnifiedFeatureMatrixError(
                    f"paired telemetry training config differs across controls: {triad_id}/{base}"
                )
            values.append(observed.iloc[0])
        block_data[feature] = values
        source = registered.loc[base]
        registry_rows.append(
            _registry_row(
                feature=feature,
                source_table="triad_paired_telemetry_deltas",
                source_field=base,
                feature_family=str(source["feature_family"]),
                available_epoch=int(source["available_epoch"]),
                allowed=False,
                analysis_role="CONSTANT_TRAINING_CONFIG",
                use="audited common training configuration; not a candidate predictor",
                base_feature=base,
            )
        )
    block = pd.DataFrame(block_data)
    return block, registry_rows, {
        "telemetry_base_features": len(paired_predictor_bases),
        "telemetry_matrix_features": len(paired_predictor_bases) * 2,
        "telemetry_constant_config_features": len(non_paired_config),
    }


def _long_numeric_block(
    frame: pd.DataFrame,
    *,
    source_table: str,
    prefix: str,
    feature_column: str | None,
    excluded_columns: set[str],
    triad_ids: set[str],
    analysis_role: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    required = ["triad_id", "arm", "replay_role"]
    if feature_column:
        required.append(feature_column)
    _require_columns(frame, required, name=source_table)
    _assert_id_coverage(frame, triad_ids, name=source_table)
    statistic_columns = [
        column
        for column in frame.columns
        if column not in excluded_columns
        and column not in {"triad_id", "arm", "replay_role", feature_column}
    ]
    if not statistic_columns:
        raise UnifiedFeatureMatrixError(f"{source_table} has no numeric statistics")
    numeric = _numeric(frame, statistic_columns, name=source_table)
    id_columns = ["triad_id", "arm", "replay_role"]
    if feature_column:
        id_columns.append(feature_column)
    long = numeric.melt(
        id_vars=id_columns,
        value_vars=statistic_columns,
        var_name="statistic",
        value_name="value",
    )
    if feature_column:
        long["matrix_feature"] = long.apply(
            lambda row: (
                f"{prefix}__{row['arm']}__{row['replay_role']}__"
                f"{_slug(row[feature_column])}__{row['statistic']}"
            ),
            axis=1,
        )
    else:
        long["matrix_feature"] = long.apply(
            lambda row: (
                f"{prefix}__{row['arm']}__{row['replay_role']}__{row['statistic']}"
            ),
            axis=1,
        )
    if long.duplicated(["triad_id", "matrix_feature"]).any():
        raise UnifiedFeatureMatrixError(f"{source_table} creates duplicate feature cells")
    block = long.pivot(index="triad_id", columns="matrix_feature", values="value").reset_index()
    block.columns.name = None
    registry_rows = []
    identities = long[
        ["matrix_feature", "arm", "replay_role", "statistic"]
        + ([feature_column] if feature_column else [])
    ].drop_duplicates("matrix_feature")
    for row in identities.itertuples(index=False):
        base = str(getattr(row, feature_column)) if feature_column else str(row.statistic)
        registry_rows.append(
            _registry_row(
                feature=str(row.matrix_feature),
                source_table=source_table,
                source_field=f"{base}.{row.statistic}" if feature_column else str(row.statistic),
                feature_family="SELECTION_SUMMARY",
                available_epoch=0,
                allowed=True,
                analysis_role=analysis_role,
                use="pretraining frozen selection composition",
                base_feature=base,
                arm=str(row.arm),
                replay_role=str(row.replay_role),
            )
        )
    return block, registry_rows


def _categorical_block(
    categorical: pd.DataFrame,
    role_cells: pd.DataFrame,
    triad_ids: set[str],
    *,
    low_cardinality_max_levels: int,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object]]:
    _require_columns(
        categorical,
        (
            "triad_id",
            "arm",
            "replay_role",
            "dimension",
            "value",
            "count",
            "share",
            "selected_count",
        ),
        name="selection_categorical",
    )
    _assert_id_coverage(categorical, triad_ids, name="selection_categorical")
    categorical = _numeric(
        categorical,
        ("count", "share", "selected_count"),
        name="selection_categorical",
    )
    sums = categorical.groupby(
        ["triad_id", "arm", "replay_role", "dimension"], dropna=False
    )["share"].sum()
    if not np.allclose(sums.to_numpy(dtype=float), 1.0, rtol=1e-7, atol=1e-7):
        raise UnifiedFeatureMatrixError("selection categorical shares must sum to one")

    dimensions = {
        str(dimension): sorted(
            categorical.loc[
                categorical["dimension"].astype(str) == str(dimension), "value"
            ]
            .astype(str)
            .unique()
        )
        for dimension in sorted(categorical["dimension"].astype(str).unique())
    }
    low = {
        dimension: values
        for dimension, values in dimensions.items()
        if len(values) <= low_cardinality_max_levels
    }
    high = {
        dimension: values
        for dimension, values in dimensions.items()
        if len(values) > low_cardinality_max_levels
    }
    slugs = {dimension: _value_slugs(values) for dimension, values in low.items()}
    records: list[dict[str, object]] = []
    registry_rows: list[dict[str, object]] = []
    registered: set[str] = set()
    grouped = {
        (str(triad), str(arm), str(role), str(dimension)): group
        for (triad, arm, role, dimension), group in categorical.groupby(
            ["triad_id", "arm", "replay_role", "dimension"],
            dropna=False,
            sort=False,
        )
    }
    for cell in role_cells.to_dict("records"):
        triad = str(cell["triad_id"])
        arm = str(cell["arm"])
        role = str(cell["replay_role"])
        row: dict[str, object] = {"triad_id": triad}
        for dimension, values in low.items():
            group = grouped.get((triad, arm, role, dimension))
            if group is None:
                raise UnifiedFeatureMatrixError(
                    f"selection categorical missing {triad}/{arm}/{role}/{dimension}"
                )
            shares = dict(zip(group["value"].astype(str), group["share"].astype(float)))
            for value in values:
                feature = (
                    f"selection_categorical__{arm}__{role}__{_slug(dimension)}__"
                    f"{slugs[dimension][value]}__share"
                )
                row[feature] = float(shares.get(value, 0.0))
                if feature not in registered:
                    registered.add(feature)
                    registry_rows.append(
                        _registry_row(
                            feature=feature,
                            source_table="selection_categorical_composition",
                            source_field=f"{dimension}.{value}.share",
                            feature_family="SELECTION_CATEGORICAL_LOW_CARDINALITY",
                            available_epoch=0,
                            allowed=True,
                            analysis_role="SELECTION_COMPOSITION_PREDICTOR",
                            use="pretraining low-cardinality category share",
                            base_feature=dimension,
                            arm=arm,
                            replay_role=role,
                        )
                    )
        for dimension in high:
            group = grouped.get((triad, arm, role, dimension))
            if group is None:
                raise UnifiedFeatureMatrixError(
                    f"selection categorical missing {triad}/{arm}/{role}/{dimension}"
                )
            shares = group["share"].to_numpy(dtype=float)
            metrics = {
                "entropy": float(-(shares * np.log(shares)).sum()),
                "hhi": float(np.square(shares).sum()),
                "max_share": float(shares.max()),
                "level_count": int(len(shares)),
            }
            for statistic, value in metrics.items():
                feature = (
                    f"selection_categorical__{arm}__{role}__{_slug(dimension)}__{statistic}"
                )
                row[feature] = value
                if feature not in registered:
                    registered.add(feature)
                    registry_rows.append(
                        _registry_row(
                            feature=feature,
                            source_table="selection_categorical_composition",
                            source_field=f"{dimension}.{statistic}",
                            feature_family="SELECTION_CATEGORICAL_HIGH_CARDINALITY",
                            available_epoch=0,
                            allowed=True,
                            analysis_role="SELECTION_DIVERSITY_PREDICTOR",
                            use="pretraining group diversity summary",
                            base_feature=dimension,
                            arm=arm,
                            replay_role=role,
                        )
                    )
        records.append(row)
    long_block = pd.DataFrame(records)
    if long_block.duplicated("triad_id").any():
        # Each triad has multiple arm/role records; coalesce their disjoint columns.
        feature_columns = [column for column in long_block if column != "triad_id"]
        block = long_block.groupby("triad_id", sort=True)[feature_columns].first().reset_index()
    else:
        block = long_block
    return block, registry_rows, {
        "categorical_low_dimensions": sorted(low),
        "categorical_high_dimensions": sorted(high),
        "categorical_matrix_features": len(registry_rows),
    }


def build_unified_feature_matrix(
    *,
    triad_outcomes: pd.DataFrame,
    paired_telemetry: pd.DataFrame,
    selection_numeric: pd.DataFrame,
    selection_categorical: pd.DataFrame,
    selection_late: pd.DataFrame,
    treatment_selection_sets: pd.DataFrame,
    checkpoint_triads: pd.DataFrame,
    resource_triads: pd.DataFrame,
    telemetry_registry: pd.DataFrame,
    expected_triads: int = 80,
    low_cardinality_max_levels: int = 32,
) -> UnifiedFeatureMatrixResult:
    """Return one row per triad plus complete time/role registries."""

    inputs = {
        "triad_outcomes": triad_outcomes,
        "paired_telemetry": paired_telemetry,
        "selection_numeric": selection_numeric,
        "selection_categorical": selection_categorical,
        "selection_late": selection_late,
        "treatment_selection_sets": treatment_selection_sets,
        "checkpoint_triads": checkpoint_triads,
        "resource_triads": resource_triads,
        "telemetry_registry": telemetry_registry,
    }
    for name, frame in inputs.items():
        _check_columns(frame, name=name)
    _require_columns(
        triad_outcomes,
        ("triad_id", "phase", "budget", "guard_ratio"),
        name="triad_outcomes",
    )
    if len(triad_outcomes) != expected_triads or triad_outcomes["triad_id"].nunique() != expected_triads:
        raise UnifiedFeatureMatrixError(
            f"Expected {expected_triads} unique triad outcomes, found {len(triad_outcomes)}"
        )
    matrix = triad_outcomes.copy()
    matrix["triad_id"] = matrix["triad_id"].astype(str)
    triad_ids = set(matrix["triad_id"])
    registry_rows: list[dict[str, object]] = []
    for column in matrix.columns:
        family, role, epoch = _outcome_role(str(column))
        registry_rows.append(
            _registry_row(
                feature=str(column),
                source_table="triad_outcomes_80",
                source_field=str(column),
                feature_family=family,
                available_epoch=epoch,
                allowed=False,
                analysis_role=role,
                use="label, identity, lineage, or confound only",
            )
        )

    role_cells = _validate_selection_roles(
        matrix, selection_numeric, selection_categorical, selection_late
    )
    for name, frame, unique in (
        ("treatment_selection_sets", treatment_selection_sets, True),
        ("checkpoint_triads", checkpoint_triads, True),
        ("resource_triads", resource_triads, True),
    ):
        _assert_id_coverage(frame, triad_ids, name=name, unique=unique)

    telemetry_block, telemetry_rows, telemetry_audit = _telemetry_block(
        paired_telemetry, telemetry_registry, triad_ids
    )
    matrix = _join_block(matrix, telemetry_block, name="telemetry feature block")
    registry_rows.extend(telemetry_rows)

    numeric_block, numeric_rows = _long_numeric_block(
        selection_numeric,
        source_table="selection_numeric_feature_summary",
        prefix="selection_numeric",
        feature_column="feature",
        excluded_columns=_SELECTION_KEYS,
        triad_ids=triad_ids,
        analysis_role="SELECTION_NUMERIC_PREDICTOR",
    )
    matrix = _join_block(matrix, numeric_block, name="selection numeric block")
    registry_rows.extend(numeric_rows)

    categorical_block, categorical_rows, categorical_audit = _categorical_block(
        selection_categorical,
        role_cells,
        triad_ids,
        low_cardinality_max_levels=low_cardinality_max_levels,
    )
    matrix = _join_block(matrix, categorical_block, name="selection categorical block")
    registry_rows.extend(categorical_rows)

    late_excluded = _SELECTION_KEYS | {
        "late_persistence_semantics",
        "blank_last_wrong_semantics",
    }
    late_block, late_rows = _long_numeric_block(
        selection_late,
        source_table="selection_late_persistence_summary",
        prefix="selection_late",
        feature_column=None,
        excluded_columns=late_excluded,
        triad_ids=triad_ids,
        analysis_role="SELECTION_LATE_PERSISTENCE_PREDICTOR",
    )
    matrix = _join_block(matrix, late_block, name="selection late block")
    registry_rows.extend(late_rows)

    _require_columns(
        treatment_selection_sets,
        ("triad_id", "sample_set_digest", "selected_count"),
        name="treatment_selection_sets",
    )
    digests = treatment_selection_sets[["triad_id", "sample_set_digest", "selected_count"]].copy()
    expected_budget = matrix.set_index("triad_id")["budget"].astype(int)
    actual_budget = digests.set_index(digests["triad_id"].astype(str))["selected_count"].astype(int)
    if not actual_budget.sort_index().equals(expected_budget.sort_index()):
        raise UnifiedFeatureMatrixError("Treatment selection-set counts differ from budget")
    digest_block = digests[["triad_id", "sample_set_digest"]].rename(
        columns={"sample_set_digest": "treatment_sample_set_digest"}
    )
    digest_block["triad_id"] = digest_block["triad_id"].astype(str)
    matrix = _join_block(matrix, digest_block, name="treatment selection digest block")
    registry_rows.append(
        _registry_row(
            feature="treatment_sample_set_digest",
            source_table="treatment_selection_sets_80",
            source_field="sample_set_digest",
            feature_family="SELECTION_DIGEST",
            available_epoch=0,
            allowed=False,
            analysis_role="SELECTION_DIGEST",
            use="grouped holdout identity only",
        )
    )

    checkpoint_features = [column for column in checkpoint_triads if column.startswith("ckpt__")]
    if not checkpoint_features:
        raise UnifiedFeatureMatrixError("checkpoint_triads has no ckpt__ features")
    checkpoint_block = checkpoint_triads[["triad_id", *checkpoint_features]].copy()
    checkpoint_block["triad_id"] = checkpoint_block["triad_id"].astype(str)
    checkpoint_block = _numeric(
        checkpoint_block, checkpoint_features, name="checkpoint_triads"
    )
    matrix = _join_block(matrix, checkpoint_block, name="checkpoint feature block")
    for feature in checkpoint_features:
        registry_rows.append(
            _registry_row(
                feature=feature,
                source_table="checkpoint_triad_features",
                source_field=feature,
                feature_family="CHECKPOINT_PARAMETER_DRIFT",
                available_epoch=200,
                allowed=True,
                analysis_role="CHECKPOINT_MECHANISM_PREDICTOR",
                use="epoch-200 mechanism feature",
            )
        )

    resource_features = [column for column in resource_triads if column != "triad_id"]
    resource_block = resource_triads[["triad_id"]].copy()
    resource_block["triad_id"] = resource_block["triad_id"].astype(str)
    for source in resource_features:
        feature = f"confound__resource__{source}"
        resource_block[feature] = resource_triads[source].to_numpy()
        registry_rows.append(
            _registry_row(
                feature=feature,
                source_table="resource_reliability_triads",
                source_field=source,
                feature_family="RESOURCE_OR_EXECUTION_CONFOUND",
                available_epoch=200,
                allowed=False,
                analysis_role="EXECUTION_CONFOUND",
                use="sensitivity stratification only",
            )
        )
    matrix = _join_block(matrix, resource_block, name="resource confound block")

    if not matrix.columns.is_unique:
        raise UnifiedFeatureMatrixError("Unified matrix has duplicate columns")
    role_registry = pd.DataFrame(registry_rows)
    if role_registry["feature"].duplicated().any():
        duplicate = role_registry.loc[
            role_registry["feature"].duplicated(keep=False), "feature"
        ].tolist()
        raise UnifiedFeatureMatrixError(f"Feature registry has duplicate columns: {duplicate}")
    if set(role_registry["feature"]) != set(matrix.columns):
        missing_registry = sorted(set(matrix.columns) - set(role_registry["feature"]))
        missing_matrix = sorted(set(role_registry["feature"]) - set(matrix.columns))
        raise UnifiedFeatureMatrixError(
            f"Registry/matrix mismatch: registry_missing={missing_registry[:5]}, "
            f"matrix_missing={missing_matrix[:5]}"
        )
    predictor_rows = role_registry.loc[role_registry["allowed_as_predictor"].astype(bool)]
    leaky = [
        feature
        for feature in predictor_rows["feature"].astype(str)
        if _LEAKY_PREDICTOR.search(feature)
    ]
    if leaky:
        raise UnifiedFeatureMatrixError(f"leaky predictor columns survived: {leaky[:5]}")
    feature_registry = role_registry[
        [
            "feature",
            "feature_family",
            "available_epoch",
            "allowed_as_predictor",
            "use",
            "base_feature",
            "source_table",
        ]
    ].copy()
    matrix = matrix.sort_values("triad_id", ignore_index=True)
    feature_registry = feature_registry.sort_values("feature", ignore_index=True)
    role_registry = role_registry.sort_values("feature", ignore_index=True)
    audit = {
        "triads": len(matrix),
        "matrix_columns": len(matrix.columns),
        "predictor_columns": int(role_registry["allowed_as_predictor"].sum()),
        "non_predictor_columns": int((~role_registry["allowed_as_predictor"].astype(bool)).sum()),
        "selection_numeric_features": len(numeric_rows),
        "selection_categorical_features": len(categorical_rows),
        "selection_late_features": len(late_rows),
        "checkpoint_features": len(checkpoint_features),
        "resource_confound_features": len(resource_features),
        **telemetry_audit,
        **categorical_audit,
    }
    return UnifiedFeatureMatrixResult(
        matrix=matrix,
        feature_registry=feature_registry,
        role_registry=role_registry,
        audit=audit,
    )


def _stage_csv(frame: pd.DataFrame, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return temporary


def publish_unified_feature_matrix(
    result: UnifiedFeatureMatrixResult,
    output_dir: str | Path,
) -> dict[str, int]:
    """Atomically publish the matrix and both registries on explicit request."""

    output = Path(output_dir).resolve()
    if not output.name.endswith(".inprogress"):
        raise ValueError("Unified feature output must remain .inprogress")
    tables = output / "tables"
    targets = {
        tables / "unified_triad_feature_matrix.csv": result.matrix,
        tables / "EXTENDED_FEATURE_TIME_REGISTRY.csv": result.feature_registry,
        tables / "FEATURE_ROLE_REGISTRY.csv": result.role_registry,
    }
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite unified feature tables: {existing}")
    staged: dict[Path, str] = {}
    try:
        for target, frame in targets.items():
            staged[target] = _stage_csv(frame, target)
        for target, temporary in staged.items():
            os.replace(temporary, target)
    finally:
        for temporary in staged.values():
            if os.path.exists(temporary):
                os.unlink(temporary)
    return {
        "triads": int(len(result.matrix)),
        "matrix_columns": int(len(result.matrix.columns)),
        "predictor_columns": int(result.feature_registry["allowed_as_predictor"].sum()),
        "registry_rows": int(len(result.feature_registry)),
    }
