"""Auditable subgroup metadata and sample-shift summaries.

Subgroup fields are read only from the frozen val_op manifests stored in a
canonical attempt.  Missing fields remain unavailable; this module does not
infer water level, source, or a primary defect class from filenames or label
ordering.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


CATEGORY_COLUMNS: tuple[str, ...] = (
    "VA",
    "RB",
    "OB",
    "PF",
    "DE",
    "FS",
    "IS",
    "RO",
    "IN",
    "AF",
    "BE",
    "FO",
    "GR",
    "PH",
    "PB",
    "OS",
    "OP",
    "OK",
    "ND",
    "Defect",
)

TARGET_DEFECT_COLUMNS: tuple[str, ...] = ("PF", "DE", "FS", "RB", "AF", "OB")
PRIMARY_CLASS_CANDIDATES: tuple[str, ...] = (
    "primary_defect_class",
    "train_primary_class",
    "eval_primary_class",
)

_OPTIONAL_METADATA = ("WaterLevel", "target_labels", "source_csv_path")
_SHIFT_COLUMNS = ("sample_id", "y_true", "raw_shift", "calibrated_shift")


def _source_basename(value) -> object:
    if pd.isna(value) or not str(value).strip():
        return pd.NA
    return str(value).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _read_manifest(path: Path, y_true: int) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen val_op manifest not found: {path}")
    frame = pd.read_csv(path, dtype={"canonical_image_relpath": "string"})
    if "canonical_image_relpath" not in frame:
        raise ValueError(f"Manifest lacks canonical_image_relpath: {path}")
    ids = frame["canonical_image_relpath"]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        raise ValueError(f"Manifest contains blank canonical identities: {path}")
    if ids.duplicated().any():
        raise ValueError(f"Manifest sample identities must be unique: {path}")
    frame = frame.copy()
    frame["sample_id"] = ids.astype(str)
    frame["y_true"] = int(y_true)
    return frame


def _available_in_both(
    normal: pd.DataFrame, defect: pd.DataFrame, column: str
) -> bool:
    return column in normal.columns and column in defect.columns


def load_subgroup_metadata(canonical_attempt: str | Path) -> pd.DataFrame:
    """Load sample-unique val_op subgroup metadata from a canonical attempt.

    The returned frame carries a ``field_availability`` mapping in
    ``DataFrame.attrs``.  Optional columns are always present in the returned
    frame, but contain ``NA`` when the source field is unavailable.
    """

    frozen = Path(canonical_attempt) / "01_manifests" / "frozen_inputs"
    normal = _read_manifest(frozen / "val_op_normal_manifest.csv", 0)
    defect = _read_manifest(frozen / "val_op_defect_manifest.csv", 1)

    availability: dict[str, object] = {"canonical_image_relpath": True}
    selected_columns = ["sample_id", "canonical_image_relpath", "y_true"]
    for column in _OPTIONAL_METADATA:
        available = _available_in_both(normal, defect, column)
        availability[column] = available
        if not available:
            normal[column] = pd.NA
            defect[column] = pd.NA
        selected_columns.append(column)

    primary_source = next(
        (
            column
            for column in PRIMARY_CLASS_CANDIDATES
            if _available_in_both(normal, defect, column)
        ),
        None,
    )
    availability["primary_defect_class"] = primary_source is not None
    availability["primary_defect_class_source"] = primary_source
    if primary_source is None:
        normal["primary_defect_class"] = pd.NA
        defect["primary_defect_class"] = pd.NA
    else:
        normal["primary_defect_class"] = normal[primary_source]
        defect["primary_defect_class"] = defect[primary_source]
    selected_columns.append("primary_defect_class")

    category_columns = [
        column
        for column in CATEGORY_COLUMNS
        if _available_in_both(normal, defect, column)
    ]
    availability["category_columns"] = category_columns
    selected_columns.extend(category_columns)

    metadata = pd.concat(
        [normal[selected_columns], defect[selected_columns]], ignore_index=True
    )
    if metadata["sample_id"].duplicated().any():
        duplicates = (
            metadata.loc[metadata["sample_id"].duplicated(False), "sample_id"]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"val_op metadata sample IDs must be unique across labels: {duplicates}"
        )

    if bool(availability["source_csv_path"]):
        metadata["source_basename"] = metadata["source_csv_path"].map(
            _source_basename
        )
        availability["source_basename"] = True
    else:
        metadata["source_basename"] = pd.NA
        availability["source_basename"] = False

    for column in category_columns:
        metadata[column] = pd.to_numeric(metadata[column], errors="raise").astype(
            "Int64"
        )
    metadata.attrs["field_availability"] = availability
    return metadata


def _normalise_shift_frames(
    sample_shifts: pd.DataFrame | Sequence[pd.DataFrame],
) -> pd.DataFrame:
    if isinstance(sample_shifts, pd.DataFrame):
        frames = [sample_shifts]
    else:
        frames = list(sample_shifts)
    if not frames:
        raise ValueError("At least one sample-shift frame is required")

    normalized = []
    for frame_index, frame in enumerate(frames):
        missing = sorted(set(_SHIFT_COLUMNS).difference(frame.columns))
        if missing:
            raise ValueError(f"Sample-shift frame missing required columns: {missing}")
        current = frame.copy()
        current["sample_id"] = current["sample_id"].astype(str)
        current["y_true"] = pd.to_numeric(
            current["y_true"], errors="raise"
        ).astype(int)
        if not current["y_true"].isin([0, 1]).all():
            raise ValueError("Sample-shift labels must be binary 0/1")
        for column in ("raw_shift", "calibrated_shift"):
            current[column] = pd.to_numeric(current[column], errors="raise")
            if not np.isfinite(current[column].to_numpy(dtype=float)).all():
                raise ValueError(f"Sample-shift column contains NaN/Inf: {column}")
        if "control" not in current:
            current["control"] = "unspecified"
        current["control"] = current["control"].astype(str)
        current["_frame_index"] = frame_index
        normalized.append(current)
    return pd.concat(normalized, ignore_index=True)


def _display_value(value) -> str:
    if pd.isna(value) or not str(value).strip():
        return "<missing>"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _summary_record(
    group: pd.DataFrame,
    *,
    control: str,
    dimension: str,
    value: str,
) -> dict:
    return {
        "control": str(control),
        "subgroup_dimension": dimension,
        "subgroup_value": value,
        "n": int(len(group)),
        "normal_n": int((group["y_true"] == 0).sum()),
        "defect_n": int((group["y_true"] == 1).sum()),
        "raw_beneficial_rate": float(group["raw_beneficial"].mean()),
        "raw_mean_shift": float(group["raw_shift"].mean()),
        "calibrated_beneficial_rate": float(
            group["calibrated_beneficial"].mean()
        ),
        "calibrated_mean_shift": float(group["calibrated_shift"].mean()),
    }


def summarize_shift_subgroups(
    sample_shifts: pd.DataFrame | Sequence[pd.DataFrame],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate raw/calibrated T-control shifts over available subgroups."""

    availability = dict(metadata.attrs.get("field_availability", {}))
    required_metadata = {"sample_id", "y_true"}
    missing_metadata = sorted(required_metadata.difference(metadata.columns))
    if missing_metadata:
        raise ValueError(f"Subgroup metadata missing columns: {missing_metadata}")
    if metadata["sample_id"].duplicated().any():
        raise ValueError("Subgroup metadata sample IDs must be unique")

    shifts = _normalise_shift_frames(sample_shifts)
    meta = metadata.copy()
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta["metadata_y_true"] = pd.to_numeric(meta["y_true"], errors="raise").astype(
        int
    )
    meta = meta.drop(columns=["y_true"])
    merged = shifts.merge(
        meta,
        on="sample_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    missing_ids = merged.loc[merged["_merge"] != "both", "sample_id"].unique()
    if len(missing_ids):
        raise ValueError(
            f"Sample-shift IDs missing subgroup metadata: {missing_ids[:10].tolist()}"
        )
    if not (merged["y_true"] == merged["metadata_y_true"]).all():
        raise ValueError("Sample-shift label does not match frozen subgroup metadata")
    merged = merged.drop(columns=["_merge", "metadata_y_true"])

    merged["raw_beneficial"] = np.where(
        merged["y_true"] == 0, merged["raw_shift"] < 0, merged["raw_shift"] > 0
    )
    merged["calibrated_beneficial"] = np.where(
        merged["y_true"] == 0,
        merged["calibrated_shift"] < 0,
        merged["calibrated_shift"] > 0,
    )

    records: list[dict] = []
    for control, control_frame in merged.groupby("control", sort=True):
        records.append(
            _summary_record(
                control_frame,
                control=str(control),
                dimension="overall",
                value="all",
            )
        )
        dimensions = ["WaterLevel", "target_labels", "source_basename"]
        if bool(availability.get("primary_defect_class", False)):
            dimensions.append("primary_defect_class")
        for dimension in dimensions:
            if not bool(availability.get(dimension, False)):
                continue
            displayed = control_frame[dimension].map(_display_value)
            for value, indices in displayed.groupby(displayed, sort=True).groups.items():
                records.append(
                    _summary_record(
                        control_frame.loc[indices],
                        control=str(control),
                        dimension=dimension,
                        value=str(value),
                    )
                )

        available_categories = set(availability.get("category_columns", []))
        for category in TARGET_DEFECT_COLUMNS:
            if category not in available_categories:
                continue
            subset = control_frame[
                pd.to_numeric(control_frame[category], errors="raise").eq(1)
            ]
            if not subset.empty:
                records.append(
                    _summary_record(
                        subset,
                        control=str(control),
                        dimension="defect_class",
                        value=category,
                    )
                )

    result = pd.DataFrame(records).sort_values(
        ["control", "subgroup_dimension", "subgroup_value"], ignore_index=True
    )
    result.attrs["field_availability"] = availability
    return result
