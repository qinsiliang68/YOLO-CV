from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_gapvalue240.selection_mechanisms import (
    build_budget_nesting,
    build_method_overlaps,
    build_same_selection_reversals,
    build_selection_summaries,
    build_triad_overlap_audit,
    load_frozen_sample_features,
    verify_and_load_selections,
)


def _atomic_csv(frame: pd.DataFrame, path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing analysis table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict, path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing analysis summary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit frozen GapValue selections and sample-dynamics mechanisms."
    )
    parser.add_argument("--canonical-inventory", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--selection-index", required=True, type=Path)
    parser.add_argument("--selection-root", required=True, type=Path)
    parser.add_argument("--value-table", required=True, type=Path)
    parser.add_argument("--dynamics", required=True, type=Path)
    parser.add_argument("--master-index", required=True, type=Path)
    parser.add_argument("--triad-outcomes", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    canonical = pd.read_csv(args.canonical_inventory, dtype={"run_slot": "string"})
    matrix = pd.read_csv(args.matrix, dtype={"run_slot": "string"})
    additions = [
        field
        for field in ("condition_slot", "discovery_or_confirmation")
        if field not in canonical.columns and field in matrix.columns
    ]
    if additions:
        canonical = canonical.merge(
            matrix[["run_slot", *additions]], on="run_slot", validate="one_to_one"
        )
    features, field_registry = load_frozen_sample_features(
        args.value_table, args.dynamics, args.master_index
    )
    selected, sha_audit = verify_and_load_selections(
        canonical, args.selection_index, args.selection_root, features
    )
    categorical, numeric, late = build_selection_summaries(selected)
    triad_overlap = build_triad_overlap_audit(selected)
    method_overlap = build_method_overlaps(selected)
    budget_nesting = build_budget_nesting(selected)
    outcomes = pd.read_csv(args.triad_outcomes, dtype={"triad_id": "string"})
    treatment_sets, digest_outcomes, reversals = build_same_selection_reversals(
        selected, outcomes
    )

    tables = {
        "selection_field_usage_registry.csv": field_registry,
        "selection_sha_audit_240.csv": sha_audit,
        "selection_categorical_composition.csv": categorical,
        "selection_numeric_feature_summary.csv": numeric,
        "selection_late_persistence_summary.csv": late,
        "selection_triad_overlap_audit.csv": triad_overlap,
        "selection_method_overlap.csv": method_overlap,
        "selection_budget_nesting.csv": budget_nesting,
        "treatment_selection_sets_80.csv": treatment_sets,
        "treatment_selection_digest_outcomes.csv": digest_outcomes,
        "same_selection_seed_reversals.csv": reversals,
    }
    for name, table in tables.items():
        _atomic_csv(table, args.output_dir / name, overwrite=args.overwrite)

    r2 = triad_overlap.loc[
        (triad_overlap["scope"] == "all")
        & (triad_overlap["left_arm"] == "T")
        & (triad_overlap["right_arm"] == "R2")
    ]
    summary = {
        "status": "PASS",
        "canonical_runs": int(canonical["run_slot"].nunique()),
        "triads": int(canonical["triad_id"].nunique()),
        "selected_rows": len(selected),
        "sha_validated_runs": len(sha_audit),
        "field_registry_rows": len(field_registry),
        "gradient_fields_not_collected": int(
            (field_registry["availability"] == "NOT_COLLECTED").sum()
        ),
        "treatment_selection_sets": len(treatment_sets),
        "unique_treatment_selection_digests": len(digest_outcomes),
        "same_selection_reversal_digests": int(
            digest_outcomes["spans_dual_improvement_and_dual_harm"].sum()
        ),
        "same_selection_reversal_triads": len(reversals),
        "r2_effective_unique_contrast_rate_mean": float(
            r2["effective_unique_contrast_rate"].mean()
        ),
        "r2_effective_unique_contrast_rate_min": float(
            r2["effective_unique_contrast_rate"].min()
        ),
        "phase_b_manifests_with_role_local_rank_reuse": int(
            sha_audit["global_run_rank_has_duplicates"].sum()
        ),
        "output_tables": {name: len(table) for name, table in tables.items()},
    }
    _atomic_json(
        summary,
        args.output_dir / "selection_mechanism_summary.json",
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
