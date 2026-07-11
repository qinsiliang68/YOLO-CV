from __future__ import annotations
from pathlib import Path
import pandas as pd

from .errors import ValidationError
from .util import atomic_write_bytes


def load_assignments(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"oof_fold":"string", "canonical_image_relpath":"string"})
    required = {"canonical_image_relpath","oof_y_true","oof_fold","oof_group_id","oof_source_manifest","train_primary_class"}
    missing = required - set(df.columns)
    if missing: raise ValidationError(f"Assignment table missing columns: {sorted(missing)}")
    df["oof_fold"] = df["oof_fold"].str.zfill(2)
    if set(df["oof_fold"].dropna().unique()) != {f"{i:02d}" for i in range(10)}:
        raise ValidationError("oof_fold must be strings 00..09")
    if len(df) != 120000: raise ValidationError(f"Expected 120000 assignments, got {len(df)}")
    if df["canonical_image_relpath"].duplicated().any(): raise ValidationError("Duplicate canonical identities")
    return df


def materialize_master_index(assignments_path: str | Path, output_path: str | Path, overwrite: bool = False) -> Path:
    a = load_assignments(assignments_path)
    out = pd.DataFrame({
        "sample_id": a["canonical_image_relpath"],
        "canonical_image_relpath": a["canonical_image_relpath"],
        "y_true": a["oof_y_true"].astype("int8"),
        "oof_fold": a["oof_fold"],
        "oof_group_id": a["oof_group_id"].astype("string"),
        "oof_group_source": a.get("oof_group_source", pd.Series([None]*len(a), dtype="string")),
        "train_primary_class": a["train_primary_class"].astype("string"),
        "source_manifest": a["oof_source_manifest"].astype("string"),
        "source_csv_path": a.get("source_csv_path"),
        "source_csv_row_number": a.get("source_csv_row_number"),
        "filename": a.get("Filename"),
    })
    return atomic_write_bytes(output_path, out.to_csv(index=False).encode("utf-8"), overwrite)
