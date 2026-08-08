"""Publish canonical 240-run raw-prediction and exact NP-frontier evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Callable
from uuid import uuid4

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_gapvalue240.raw_frontier_analysis import (  # noqa: E402
    RawFrontierError,
    analyze_canonical_pairs,
    analyze_canonical_val_cal,
    build_control_reference_from_index,
    load_canonical_prediction_index,
)


TABLE_FILENAMES = (
    "raw_frontier_run_probability_metrics.csv",
    "raw_frontier_equivalence_audit.csv",
    "raw_frontier_paired_dominance.csv",
    "raw_frontier_paired_tail_shift_summary.csv",
    "control_reference.csv",
)
SUMMARY_FILENAME = "raw_frontier_analysis_summary.json"
VAL_CAL_TABLE_FILENAMES = (
    "val_cal_probability_metrics.csv",
    "val_cal_ranking_frontier_audit.csv",
    "val_cal_platt_audit.csv",
)
VAL_CAL_SUMMARY_FILENAME = "val_cal_analysis_summary.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _under_inprogress(path: Path) -> bool:
    return any(
        current.name.endswith(".inprogress") for current in (path, *path.parents)
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def publish_tables_atomic(
    output_tables_dir: str | Path,
    tables: dict[str, pd.DataFrame],
    summary: dict[str, object],
    *,
    overwrite: bool,
    _table_filenames: tuple[str, ...] = TABLE_FILENAMES,
    _summary_filename: str = SUMMARY_FILENAME,
) -> dict[str, str]:
    """Atomically publish this module's fixed outputs; summary is written last."""

    output = Path(output_tables_dir).resolve()
    if not _under_inprogress(output):
        raise RawFrontierError(
            f"Output must remain under a .inprogress analysis directory: {output}"
        )
    if set(tables) != set(_table_filenames):
        raise RawFrontierError(
            "Publisher table names must be exactly "
            f"{list(_table_filenames)}, found {sorted(tables)}"
        )
    output.mkdir(parents=True, exist_ok=True)
    destinations = {name: output / name for name in _table_filenames}
    destinations[_summary_filename] = output / _summary_filename
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing and not overwrite:
        raise RawFrontierError(
            "Raw-frontier outputs already exist; pass --overwrite to replace only "
            f"this module's fixed files: {existing}"
        )

    token = uuid4().hex
    temporary: dict[str, Path] = {
        name: output / f".{name}.{token}.tmp" for name in destinations
    }
    output_metadata: dict[str, dict[str, object]] = {}
    try:
        for name in _table_filenames:
            table = tables[name]
            if not isinstance(table, pd.DataFrame):
                raise RawFrontierError(f"{name} is not a pandas DataFrame")
            temp = temporary[name]
            table.to_csv(temp, index=False, lineterminator="\n")
            with temp.open("r+b") as handle:
                os.fsync(handle.fileno())
            verified = pd.read_csv(temp)
            if len(verified) != len(table) or verified.columns.tolist() != table.columns.tolist():
                raise RawFrontierError(f"Atomic staging verification failed for {name}")
            output_metadata[name] = {
                "row_count": int(len(table)),
                "column_count": int(len(table.columns)),
                "size_bytes": int(temp.stat().st_size),
                "sha256": _sha256(temp),
            }

        final_summary = dict(summary)
        final_summary["output_files"] = output_metadata
        summary_temp = temporary[_summary_filename]
        summary_temp.write_text(
            json.dumps(final_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with summary_temp.open("r+b") as handle:
            os.fsync(handle.fileno())
        json.loads(summary_temp.read_text(encoding="utf-8"))

        # The completeness marker is the JSON summary, so publish it last.
        for name in _table_filenames:
            os.replace(temporary[name], destinations[name])
        os.replace(summary_temp, destinations[_summary_filename])
    finally:
        for temp in temporary.values():
            if temp.exists():
                temp.unlink()
    return {name: str(path) for name, path in destinations.items()}


def publish_val_cal_tables_atomic(
    output_tables_dir: str | Path,
    tables: dict[str, pd.DataFrame],
    summary: dict[str, object],
    *,
    overwrite: bool,
) -> dict[str, str]:
    """Publish val-cal outputs without touching existing val-op artifacts."""

    return publish_tables_atomic(
        output_tables_dir,
        tables,
        summary,
        overwrite=overwrite,
        _table_filenames=VAL_CAL_TABLE_FILENAMES,
        _summary_filename=VAL_CAL_SUMMARY_FILENAME,
    )


def run_raw_frontier_publication(
    *,
    inventory_path: str | Path,
    extracted_root: str | Path,
    output_tables_dir: str | Path,
    overwrite: bool = False,
    notify: Callable[[str], None] = print,
) -> dict[str, object]:
    """Run the canonical analysis and atomically publish its compact tables."""

    inventory_path = Path(inventory_path).resolve()
    extracted_root = Path(extracted_root).resolve()
    output_tables_dir = Path(output_tables_dir).resolve()
    if _is_within(output_tables_dir, extracted_root):
        raise RawFrontierError(
            "Analysis output cannot be placed inside the read-only extracted source root"
        )
    if not _under_inprogress(output_tables_dir):
        raise RawFrontierError(
            "Analysis output must remain under a .inprogress analysis directory"
        )
    prospective = [output_tables_dir / name for name in (*TABLE_FILENAMES, SUMMARY_FILENAME)]
    existing = [str(path) for path in prospective if path.exists()]
    if existing and not overwrite:
        raise RawFrontierError(
            "Raw-frontier outputs already exist; pass --overwrite to replace only "
            f"this module's fixed files: {existing}"
        )

    started_at = _utc_now()
    start = time.perf_counter()
    notify("[raw-frontier] Loading canonical 240-run inventory")
    index = load_canonical_prediction_index(inventory_path, extracted_root)
    notify("[raw-frontier] Building fixed sample tails from 160 R1/R2 controls")
    reference = build_control_reference_from_index(index)
    reference_seconds = time.perf_counter() - start
    notify("[raw-frontier] Streaming 80 triads and recomputing raw/calibrated frontiers")
    analysis_start = time.perf_counter()
    result = analyze_canonical_pairs(index, reference)
    analysis_seconds = time.perf_counter() - analysis_start

    metrics = result["run_probability_metrics"]
    equivalence = result["frontier_equivalence_audit"]
    paired = result["paired_frontier_dominance"]
    tails = result["paired_tail_shift_summary"]
    expected_rows = {
        "run_probability_metrics": 480,
        "frontier_equivalence_audit": 240,
        "paired_frontier_dominance": 320,
        "paired_tail_shift_summary": 1_920,
    }
    observed_rows = {
        "run_probability_metrics": len(metrics),
        "frontier_equivalence_audit": len(equivalence),
        "paired_frontier_dominance": len(paired),
        "paired_tail_shift_summary": len(tails),
    }
    if observed_rows != expected_rows:
        raise RawFrontierError(
            f"Canonical analysis row-count gate failed: {observed_rows}"
        )
    diagnostics = metrics[["auroc", "auprc", "brier", "log_loss", "ece"]]
    if not diagnostics.notna().all().all():
        raise RawFrontierError("Probability diagnostics contain missing values")
    raw_pairs = paired[paired.score_type.astype(str) == "raw"].copy()
    if len(raw_pairs) != 160:
        raise RawFrontierError(f"Expected 160 raw T-control pairs, found {len(raw_pairs)}")
    dual_safe = (
        raw_pairs.groupby("triad_id", sort=True)["safe_frontier_dominant"]
        .all()
        .sum()
    )
    dual_full = (
        raw_pairs.groupby("triad_id", sort=True)["full_frontier_dominant"]
        .all()
        .sum()
    )
    normal_reference = reference[reference.y_true.astype(int) == 0]
    defect_reference = reference[reference.y_true.astype(int) == 1]
    compact_reference = reference[
        [
            "sample_id",
            "y_true",
            "control_median_score_raw",
            "operational_tail",
            "distribution_tail",
            "reference_risk_rank_within_class",
        ]
    ].copy()
    completed_at = _utc_now()
    summary: dict[str, object] = {
        "status": "COMPLETE",
        "analysis_stage": "RAW_PREDICTION_AND_EXACT_NP_FRONTIER",
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "source_is_read_only": True,
        "inventory_path": str(inventory_path),
        "inventory_sha256": _sha256(inventory_path),
        "extracted_root": str(extracted_root),
        "canonical_runs": int(index.run_slot.nunique()),
        "canonical_triads": int(index.triad_id.nunique()),
        "canonical_controls": int(index.arm.astype(str).isin({"R1", "R2"}).sum()),
        "raw_control_pairs": int(len(raw_pairs)),
        "probability_metric_rows": int(len(metrics)),
        "paired_frontier_rows": int(len(paired)),
        "paired_tail_shift_rows": int(len(tails)),
        "raw_calibrated_frontier_exact_runs": int(
            equivalence.raw_calibrated_frontier_exact.astype(bool).sum()
        ),
        "raw_safe_frontier_dominant_pairs": int(
            raw_pairs.safe_frontier_dominant.astype(bool).sum()
        ),
        "raw_full_frontier_dominant_pairs": int(
            raw_pairs.full_frontier_dominant.astype(bool).sum()
        ),
        "raw_dual_control_safe_frontier_dominant_triads": int(dual_safe),
        "raw_dual_control_full_frontier_dominant_triads": int(dual_full),
        "operational_normal_tail_count": int(normal_reference.operational_tail.sum()),
        "operational_defect_tail_count": int(defect_reference.operational_tail.sum()),
        "distribution_normal_tail_count": int(
            normal_reference.distribution_tail.sum()
        ),
        "distribution_defect_tail_count": int(
            defect_reference.distribution_tail.sum()
        ),
        "control_reference_path_set_sha256": str(
            reference.control_path_set_sha256.iloc[0]
        ),
        "threshold_rule": "predict defect when score >= threshold",
        "frontier_rule": "maximum TN under the same integer FN<=k budget; whole tie groups",
        "reference_rule": "per-sample median score_raw across canonical R1/R2 controls only",
        "reference_seconds": round(reference_seconds, 6),
        "analysis_seconds": round(analysis_seconds, 6),
        "total_compute_seconds": round(time.perf_counter() - start, 6),
    }
    tables = {
        "raw_frontier_run_probability_metrics.csv": metrics,
        "raw_frontier_equivalence_audit.csv": equivalence,
        "raw_frontier_paired_dominance.csv": paired,
        "raw_frontier_paired_tail_shift_summary.csv": tails,
        "control_reference.csv": compact_reference,
    }
    notify("[raw-frontier] Atomically publishing five tables and summary JSON")
    published = publish_tables_atomic(
        output_tables_dir,
        tables,
        summary,
        overwrite=overwrite,
    )
    final_summary = json.loads(
        Path(published[SUMMARY_FILENAME]).read_text(encoding="utf-8")
    )
    notify(
        "[raw-frontier] COMPLETE: "
        f"runs={final_summary['canonical_runs']}, "
        f"pairs={final_summary['raw_control_pairs']}, "
        f"safe_dominant={final_summary['raw_safe_frontier_dominant_pairs']}"
    )
    return final_summary


def run_val_cal_publication(
    *,
    inventory_path: str | Path,
    extracted_root: str | Path,
    output_tables_dir: str | Path,
    overwrite: bool = False,
    notify: Callable[[str], None] = print,
) -> dict[str, object]:
    """Stream all canonical val-cal files and publish calibration evidence."""

    inventory_path = Path(inventory_path).resolve()
    extracted_root = Path(extracted_root).resolve()
    output_tables_dir = Path(output_tables_dir).resolve()
    if _is_within(output_tables_dir, extracted_root):
        raise RawFrontierError(
            "Analysis output cannot be placed inside the read-only extracted source root"
        )
    if not _under_inprogress(output_tables_dir):
        raise RawFrontierError(
            "Analysis output must remain under a .inprogress analysis directory"
        )
    prospective = [
        output_tables_dir / name
        for name in (*VAL_CAL_TABLE_FILENAMES, VAL_CAL_SUMMARY_FILENAME)
    ]
    existing = [str(path) for path in prospective if path.exists()]
    if existing and not overwrite:
        raise RawFrontierError(
            "val_cal outputs already exist; pass --overwrite to replace only "
            f"the val_cal files: {existing}"
        )

    started_at = _utc_now()
    start = time.perf_counter()
    notify("[val-cal] Loading canonical inventory and resolving 240 val_cal/Platt files")
    index = load_canonical_prediction_index(
        inventory_path,
        extracted_root,
        require_val_cal=True,
    )
    notify("[val-cal] Streaming predictions and recomputing calibration/ranking audits")
    result = analyze_canonical_val_cal(index)
    metrics = result["val_cal_probability_metrics"]
    ranking = result["val_cal_ranking_frontier_audit"]
    platt = result["val_cal_platt_audit"]
    observed = {
        "val_cal_probability_metrics": len(metrics),
        "val_cal_ranking_frontier_audit": len(ranking),
        "val_cal_platt_audit": len(platt),
    }
    expected = {
        "val_cal_probability_metrics": 480,
        "val_cal_ranking_frontier_audit": 240,
        "val_cal_platt_audit": 240,
    }
    if observed != expected:
        raise RawFrontierError(f"val_cal row-count gate failed: {observed}")
    if metrics.run_slot.nunique() != 240 or ranking.run_slot.nunique() != 240:
        raise RawFrontierError("val_cal tables do not cover exactly 240 canonical runs")
    if ranking.val_cal_identity_sha256.nunique() != 1:
        raise RawFrontierError("val_cal sample identity/label digest differs across runs")
    if not metrics[["auroc", "auprc", "brier", "log_loss", "ece"]].notna().all().all():
        raise RawFrontierError("val_cal diagnostics contain missing values")

    metric_wide = metrics.pivot(
        index="run_slot",
        columns="score_type",
        values=["auroc", "auprc", "brier", "log_loss", "ece"],
    )
    metric_deltas = {
        metric: metric_wide[(metric, "calibrated")] - metric_wide[(metric, "raw")]
        for metric in ("auroc", "auprc", "brier", "log_loss", "ece")
    }
    summary: dict[str, object] = {
        "status": "COMPLETE",
        "analysis_stage": "VAL_CAL_CALIBRATION_AND_RANKING_AUDIT",
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "source_is_read_only": True,
        "inventory_path": str(inventory_path),
        "inventory_sha256": _sha256(inventory_path),
        "extracted_root": str(extracted_root),
        "canonical_runs": int(index.run_slot.nunique()),
        "canonical_triads": int(index.triad_id.nunique()),
        "probability_metric_rows": int(len(metrics)),
        "ranking_frontier_audit_rows": int(len(ranking)),
        "platt_audit_rows": int(len(platt)),
        "val_cal_identity_sha256": str(ranking.val_cal_identity_sha256.iloc[0]),
        "raw_calibrated_frontier_exact_runs": int(
            ranking.raw_calibrated_frontier_exact.astype(bool).sum()
        ),
        "monotonic_nondecreasing_runs": int(
            ranking.monotonic_nondecreasing.astype(bool).sum()
        ),
        "monotonic_violation_runs": int(
            (ranking.monotonic_violation_count.astype(int) > 0).sum()
        ),
        "total_monotonic_violations": int(
            ranking.monotonic_violation_count.astype(int).sum()
        ),
        "coefficient_positive_runs": int(platt.coefficient_positive.astype(bool).sum()),
        "coefficient_zero_runs": int(platt.coefficient_zero.astype(bool).sum()),
        "coefficient_negative_runs": int(platt.coefficient_negative.astype(bool).sum()),
        "coefficient_min": float(platt.coefficient.min()),
        "coefficient_median": float(platt.coefficient.median()),
        "coefficient_max": float(platt.coefficient.max()),
        "intercept_min": float(platt.intercept.min()),
        "intercept_median": float(platt.intercept.median()),
        "intercept_max": float(platt.intercept.max()),
        "clip_low_unique": sorted(platt.clip_low.astype(float).unique().tolist()),
        "clip_high_unique": sorted(platt.clip_high.astype(float).unique().tolist()),
        "source_prevalence_unique": sorted(
            platt.source_prevalence.astype(float).unique().tolist()
        ),
        "deployment_prevalence_unique": sorted(
            platt.deployment_prevalence.astype(float).unique().tolist()
        ),
        "max_source_prevalence_abs_error": float(
            platt.source_prevalence_abs_error.max()
        ),
        "max_abs_transform_residual": float(platt.max_abs_transform_residual.max()),
        "transform_recomputed_exact_1e12_runs": int(
            platt.transform_recomputed_exact_1e12.astype(bool).sum()
        ),
        "runs_with_low_clipping": int((platt.raw_clipped_low_count > 0).sum()),
        "runs_with_high_clipping": int((platt.raw_clipped_high_count > 0).sum()),
        "total_low_clipped_predictions": int(platt.raw_clipped_low_count.sum()),
        "total_high_clipped_predictions": int(platt.raw_clipped_high_count.sum()),
        "mean_auroc_delta_calibrated_minus_raw": float(metric_deltas["auroc"].mean()),
        "mean_auprc_delta_calibrated_minus_raw": float(metric_deltas["auprc"].mean()),
        "mean_brier_delta_calibrated_minus_raw": float(metric_deltas["brier"].mean()),
        "mean_log_loss_delta_calibrated_minus_raw": float(
            metric_deltas["log_loss"].mean()
        ),
        "mean_ece_delta_calibrated_minus_raw": float(metric_deltas["ece"].mean()),
        "total_compute_seconds": round(time.perf_counter() - start, 6),
    }
    tables = {
        "val_cal_probability_metrics.csv": metrics,
        "val_cal_ranking_frontier_audit.csv": ranking,
        "val_cal_platt_audit.csv": platt,
    }
    notify("[val-cal] Atomically publishing three val_cal tables and summary JSON")
    published = publish_val_cal_tables_atomic(
        output_tables_dir,
        tables,
        summary,
        overwrite=overwrite,
    )
    final_summary = json.loads(
        Path(published[VAL_CAL_SUMMARY_FILENAME]).read_text(encoding="utf-8")
    )
    notify(
        "[val-cal] COMPLETE: "
        f"runs={final_summary['canonical_runs']}, "
        f"frontier_exact={final_summary['raw_calibrated_frontier_exact_runs']}, "
        f"monotonic_violations={final_summary['monotonic_violation_runs']}"
    )
    return final_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument(
        "--output-tables-dir",
        "--output-dir",
        dest="output_tables_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the selected stage's fixed output files.",
    )
    parser.add_argument(
        "--stage",
        choices=("val-op", "val-cal"),
        default="val-op",
        help="Publish the operational frontier or calibration-split audit.",
    )
    args = parser.parse_args()
    runner = (
        run_raw_frontier_publication
        if args.stage == "val-op"
        else run_val_cal_publication
    )
    summary = runner(
        inventory_path=args.inventory,
        extracted_root=args.extracted_root,
        output_tables_dir=args.output_tables_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
