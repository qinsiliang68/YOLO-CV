"""Read-only orchestration for the Stage1 GapValue 240-run deep analysis."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .deep_analysis import CanonicalInputError, ingest_canonical_runs
from .deep_mechanisms import (
    audit_selections,
    build_control_consensus,
    calibration_diagnostics,
    define_reference_tails,
    pair_prediction_tail_shifts,
    summarize_sample_shift_consistency,
    summarize_selected_score_signs,
)
from .deep_reporting import build_deep_report
from .deep_statistics import (
    build_a02_summaries,
    build_budget_comparisons,
    build_condition_summaries,
    build_cross_method_comparisons,
    build_direct_treatment_comparisons,
    build_guard_comparisons,
    build_sensitivity_summaries,
    build_triad_deltas,
)
from .deep_subgroups import load_subgroup_metadata, summarize_shift_subgroups
from .statistics import paired_summary


def validate_completeness_audit(
    audit_path: str | Path,
    *,
    expected_runs: int = 240,
    expected_triads: int = 80,
    expected_comparisons: int = 160,
) -> dict[str, Any]:
    """Validate transfer completeness and authorize only proven snapshot merging."""

    path = Path(audit_path).resolve()
    if not path.is_file():
        raise CanonicalInputError(f"Missing completeness audit: {path}")
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CanonicalInputError(f"Invalid completeness audit: {path}: {exc}") from exc
    if not isinstance(audit, dict):
        raise CanonicalInputError("Completeness audit must be a JSON object")
    if audit.get("material_completeness") != "COMPLETE":
        raise CanonicalInputError(
            f"Material completeness is not COMPLETE: {audit.get('material_completeness')}"
        )
    for field, expected in (
        ("unique_validated_run_count", expected_runs),
        ("triad_count", expected_triads),
        ("paired_comparison_count", expected_comparisons),
    ):
        if int(audit.get(field, -1)) != expected:
            raise CanonicalInputError(
                f"Completeness audit {field} expected {expected}, "
                f"found {audit.get(field)}"
            )
    for field in ("missing_runs", "extra_runs", "duplicate_validated_runs"):
        if audit.get(field, []):
            raise CanonicalInputError(
                f"Completeness audit contains non-empty {field}: {audit[field]}"
            )

    snapshots = list(audit.get("input_snapshot_ids", []))
    semantic_authorized = len(snapshots) <= 1
    if len(snapshots) > 1:
        semantic = audit.get("input_snapshot_semantic_audit")
        if not isinstance(semantic, dict):
            raise CanonicalInputError(
                "Multiple input snapshots lack semantic equivalence audit"
            )
        semantic_authorized = bool(
            int(semantic.get("snapshot_count", -1)) == len(snapshots)
            and semantic.get("newline_normalized_all_frozen_csv_equal") is True
        )
        if not semantic_authorized:
            raise CanonicalInputError(
                "Multiple input snapshots are not semantically proven equivalent"
            )
    result = dict(audit)
    result["semantic_snapshot_exception_authorized"] = semantic_authorized
    return result


def build_threshold_frontier(
    runs: pd.DataFrame,
    *,
    fn_budgets: Iterable[int] = range(0, 201),
    sweep_loader: Callable[[pd.Series], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build the A02 FN-budget frontier using whole tie groups.

    Sweep rows are expected in descending threshold order.  The first row with
    ``FN <= budget`` is therefore the highest achievable threshold.
    """

    required = {
        "run_slot",
        "condition_slot",
        "arm",
        "phase",
        "training_seed",
        "attempt_dir",
    }
    missing = required - set(runs.columns)
    if missing:
        raise CanonicalInputError(
            f"Threshold frontier run table missing columns: {sorted(missing)}"
        )
    selected = runs[runs["condition_slot"].astype(str) == "A02"].copy()
    if selected.empty:
        return pd.DataFrame()

    def default_loader(row: pd.Series) -> pd.DataFrame:
        return pd.read_csv(
            Path(str(row["attempt_dir"])) / "05_metrics/threshold_sweep.csv"
        )

    loader = sweep_loader or default_loader
    budgets = [int(value) for value in fn_budgets]
    if any(value < 0 for value in budgets):
        raise CanonicalInputError("FN budgets must be non-negative")
    records: list[dict[str, Any]] = []
    for _, row in selected.sort_values("run_slot").iterrows():
        sweep = loader(row)
        sweep_required = {"threshold", "TP", "FP", "TN", "FN", "tie_group_size"}
        if not sweep_required.issubset(sweep.columns):
            raise CanonicalInputError(
                f"{row.run_slot} threshold sweep missing columns: "
                f"{sorted(sweep_required-set(sweep.columns))}"
            )
        if sweep.empty:
            raise CanonicalInputError(f"{row.run_slot} threshold sweep is empty")
        numeric = sweep[list(sweep_required - {"threshold"})].apply(
            pd.to_numeric, errors="raise"
        )
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise CanonicalInputError(
                f"{row.run_slot} threshold sweep contains NaN/Inf counts"
            )
        for budget in budgets:
            feasible = sweep[pd.to_numeric(sweep["FN"]) <= budget]
            if feasible.empty:
                record = {
                    "constraint_status": "CONSTRAINT_UNREACHABLE",
                    "threshold": np.nan,
                    "actual_TP": -1,
                    "actual_FP": -1,
                    "actual_TN": -1,
                    "actual_FN": -1,
                    "tie_group_size": 0,
                }
            else:
                point = feasible.iloc[0]
                record = {
                    "constraint_status": "OK",
                    "threshold": float(point["threshold"]),
                    "actual_TP": int(point["TP"]),
                    "actual_FP": int(point["FP"]),
                    "actual_TN": int(point["TN"]),
                    "actual_FN": int(point["FN"]),
                    "tie_group_size": int(point["tie_group_size"]),
                }
            records.append(
                {
                    "run_slot": str(row["run_slot"]),
                    "phase": str(row["phase"]),
                    "condition_slot": str(row["condition_slot"]),
                    "arm": str(row["arm"]),
                    "training_seed": int(row["training_seed"]),
                    "fn_budget": budget,
                    **record,
                }
            )
    return pd.DataFrame(records)


def _dual_control_status(
    frame: pd.DataFrame,
    *,
    condition_slot: str,
    require_tn_gate: bool = True,
) -> tuple[str, str]:
    if "condition_slot" not in frame.columns or "control" not in frame.columns:
        return "NOT_TESTABLE", "Both R1 and R2 evidence rows are required."
    rows = frame[frame["condition_slot"].astype(str) == condition_slot]
    if rows.empty or set(rows.get("control", pd.Series(dtype=str)).astype(str)) != {
        "R1",
        "R2",
    }:
        return "NOT_TESTABLE", "Both R1 and R2 evidence rows are required."
    safety = rows.get("safety_noninferior", pd.Series(False, index=rows.index)).fillna(
        False
    )
    tn = rows.get(
        "confirmed_TN_improvement", pd.Series(False, index=rows.index)
    ).fillna(False)
    if bool(safety.all()) and (not require_tn_gate or bool(tn.all())):
        return "SUPPORTED", "Both frozen controls pass the registered gates."
    direction_bad = (
        (pd.to_numeric(rows.get("mean_delta_FN"), errors="coerce") > 0).any()
        or (
            require_tn_gate
            and (pd.to_numeric(rows.get("mean_delta_TN"), errors="coerce") <= 0).any()
        )
    )
    if direction_bad or not bool(safety.all()):
        return "NOT_SUPPORTED", "At least one control fails safety or effect direction."
    return "INCONCLUSIVE", "Direction is favourable but registered gates are not met."


def build_hypothesis_registry(
    *,
    a02_summaries: pd.DataFrame,
    condition_summaries: pd.DataFrame,
    budget_comparisons: pd.DataFrame,
    guard_comparisons: pd.DataFrame,
    r2_overlap: pd.DataFrame,
) -> pd.DataFrame:
    """Classify the predeclared scientific questions conservatively."""

    pooled = a02_summaries[
        a02_summaries.get("analysis_cohort", pd.Series(dtype=str)).astype(str)
        == "pooled"
    ].copy()
    h1_rows = pooled[pooled.get("control", pd.Series(dtype=str)) == "R1"]
    h2_rows = pooled[pooled.get("control", pd.Series(dtype=str)) == "R2"]

    def one_control(rows: pd.DataFrame, control: str) -> tuple[str, str]:
        if rows.empty:
            return "NOT_TESTABLE", f"No A02 contract pooled-8 evidence for {control}."
        row = rows.iloc[0]
        numeric_evidence = (
            f"mean delta_FN={float(row.get('mean_delta_FN', np.nan)):.3f}, "
            f"FN upper95={float(row.get('FN_one_sided_95_upper', np.nan)):.3f}, "
            f"worst delta_FN={float(row.get('worst_delta_FN', np.nan)):.3f}, "
            f"mean delta_TN={float(row.get('mean_delta_TN', np.nan)):.3f}, "
            f"TN lower95={float(row.get('TN_one_sided_95_lower', np.nan)):.3f}."
        )
        if bool(row.get("safety_noninferior", False)) and bool(
            row.get("confirmed_TN_improvement", False)
        ):
            return (
                "SUPPORTED",
                f"A02 contract pooled-8 passes both gates vs {control}; "
                + numeric_evidence,
            )
        mean_fn = float(row.get("mean_delta_FN", np.nan))
        mean_tn = float(row.get("mean_delta_TN", np.nan))
        if (
            not bool(row.get("safety_noninferior", False))
            or mean_fn > 0
            or mean_tn <= 0
        ):
            return (
                "NOT_SUPPORTED",
                f"A02 contract pooled-8 fails registered evidence vs {control}; "
                + numeric_evidence,
            )
        return (
            "INCONCLUSIVE",
            f"A02 contract pooled-8 direction is favourable but not gated vs {control}; "
            + numeric_evidence,
        )

    records: list[dict[str, Any]] = []
    for hypothesis_id, label, rows, control in (
        ("H1_A02_vs_R1", "GapCritical-Strict B3000 outperforms global random", h1_rows, "R1"),
        ("H2_A02_vs_R2", "GapCritical-Strict B3000 outperforms matched random", h2_rows, "R2"),
    ):
        status, rationale = one_control(rows, control)
        if control == "R2" and not r2_overlap.empty:
            overlap_column = (
                "effective_unique_contrast"
                if "effective_unique_contrast" in r2_overlap.columns
                else "effective_unique_contrast_rate"
            )
            contrast = pd.to_numeric(
                r2_overlap.get(overlap_column), errors="coerce"
            ).median()
            rationale += (
                f" Median effective unique contrast={contrast:.3f}; "
                "R2 is a high-overlap, low-power near-treatment control."
            )
        records.append(
            {
                "hypothesis_id": hypothesis_id,
                "hypothesis": label,
                "status": status,
                "rationale": rationale,
                "causal_claim_allowed": False,
            }
        )

    method_slots = {"A05", "A06", "A07", "A08", "A09", "A10", "A11", "A12"}
    methods = condition_summaries[
        condition_summaries.get("condition_slot", pd.Series(dtype=str))
        .astype(str)
        .isin(method_slots)
    ]
    primary_statuses = {
        record["status"]
        for record in records
        if record["hypothesis_id"] in {"H1_A02_vs_R1", "H2_A02_vs_R2"}
    }
    if methods.empty:
        method_status = "NOT_TESTABLE"
    elif "NOT_SUPPORTED" in primary_statuses:
        method_status = "NOT_SUPPORTED"
    else:
        method_status = "INCONCLUSIVE"
    records.append(
        {
            "hypothesis_id": "H3_vs_traditional_rankings",
            "hypothesis": "GapCritical is stronger than traditional static rankings",
            "status": method_status,
            "rationale": (
                "Cross-method effect tables are required for interpretation; "
                "condition summaries alone do not create a causal ranking."
            ),
            "causal_claim_allowed": False,
        }
    )

    phase_a = (
        condition_summaries[
            condition_summaries["phase"].astype(str) == "A"
        ]
        if "phase" in condition_summaries.columns
        else condition_summaries
    )
    a02_status, _ = _dual_control_status(phase_a, condition_slot="A02")
    a13 = condition_summaries[
        condition_summaries.get("condition_slot", pd.Series(dtype=str)).astype(str)
        == "A13"
    ]
    negative_direction = (
        len(a13) == 2
        and set(a13["control"].astype(str)) == {"R1", "R2"}
        and (pd.to_numeric(a13["mean_delta_FN"], errors="coerce") > 0).all()
        and (pd.to_numeric(a13["mean_delta_TN"], errors="coerce") < 0).all()
    )
    if a02_status == "SUPPORTED" and negative_direction:
        negative_status = "SUPPORTED"
    elif a13.empty:
        negative_status = "NOT_TESTABLE"
    else:
        negative_status = "NOT_SUPPORTED"
    records.append(
        {
            "hypothesis_id": "H4_directional_negative_control",
            "hypothesis": "GapCritical and BottomGap produce opposite directions",
            "status": negative_status,
            "rationale": (
                "Requires dual-control positive A02 evidence and dual-control "
                "opposite-direction A13 evidence."
            ),
            "causal_claim_allowed": False,
        }
    )

    strict_budget_rows = phase_a[
        phase_a.get("condition_slot", pd.Series(dtype=str))
        .astype(str)
        .isin({"A01", "A02", "A03"})
    ]
    if budget_comparisons.empty or strict_budget_rows.empty:
        budget_status = "NOT_TESTABLE"
    else:
        harmful_large_budgets = strict_budget_rows[
            strict_budget_rows["condition_slot"].astype(str).isin({"A02", "A03"})
        ]
        budget_status = (
            "NOT_SUPPORTED"
            if not harmful_large_budgets.empty
            and (
                pd.to_numeric(
                    harmful_large_budgets["mean_delta_TN"], errors="coerce"
                )
                < 0
            ).all()
            and (
                pd.to_numeric(
                    harmful_large_budgets["mean_delta_FN"], errors="coerce"
                )
                > 0
            ).all()
            else "INCONCLUSIVE"
        )
    records.append(
        {
            "hypothesis_id": "H5_budget_response",
            "hypothesis": "Replay response varies coherently across B600/B3000/B6000",
            "status": budget_status,
            "rationale": (
                "Budget contrasts are exploratory because optimizer steps differ by budget."
            ),
            "causal_claim_allowed": False,
        }
    )
    dynamics_ablation_slots = {"A16", "A17", "A19"}
    available_slots = set(
        condition_summaries.get(
            "condition_slot", pd.Series(dtype=str)
        ).astype(str)
    )
    if not dynamics_ablation_slots.issubset(available_slots):
        early_late_status = "NOT_TESTABLE"
    elif a02_status == "NOT_SUPPORTED":
        early_late_status = "NOT_SUPPORTED"
    else:
        early_late_status = "INCONCLUSIVE"
    records.append(
        {
            "hypothesis_id": "H6_not_simple_early_late",
            "hypothesis": "Gap-linked dynamics add evidence beyond simple early/late change",
            "status": early_late_status,
            "rationale": "Residual and time-matched ablations are descriptive n=3 evidence.",
            "causal_claim_allowed": False,
        }
    )
    records.append(
        {
            "hypothesis_id": "H7_defect_guard",
            "hypothesis": "Defect guard protects FN without sacrificing TN",
            "status": "INCONCLUSIVE",
            "rationale": (
                "Guard contrasts are exploratory; mixed-machine Phase B evidence "
                "cannot establish an unqualified causal guard effect."
            ),
            "causal_claim_allowed": False,
        }
    )
    records.append(
        {
            "hypothesis_id": "H8_robust_to_execution_factors",
            "hypothesis": "A02 evidence is robust to machine, resume, and snapshot factors",
            "status": (
                "NOT_TESTABLE"
                if pooled.empty
                else "INCONCLUSIVE"
                if bool(pooled.get("machine_confounded", pd.Series(True)).any())
                else "SUPPORTED"
            ),
            "rationale": (
                "Pooled A02 confirmation is machine-confounded; sensitivity tables "
                "must be consulted rather than statistically correcting it away."
            ),
            "causal_claim_allowed": False,
        }
    )
    return pd.DataFrame(records)


def _read_predictions(attempt_dir: str | Path, split: str) -> pd.DataFrame:
    return pd.read_csv(
        Path(attempt_dir) / f"04_predictions/{split}_predictions.csv",
        dtype={"sample_id": "string"},
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _selection_sha_audit(
    runs: pd.DataFrame, selection_root: str | Path
) -> pd.DataFrame:
    root = Path(selection_root).resolve()
    records: list[dict[str, Any]] = []
    for _, row in runs.iterrows():
        path = root / str(row["run_slot"]) / "selection_manifest.csv"
        if not path.is_file():
            raise CanonicalInputError(f"Missing analysis selection manifest: {path}")
        actual = _sha256(path)
        expected = str(row["selection_sha256"]).upper()
        match = actual == expected
        records.append(
            {
                "run_slot": row["run_slot"],
                "expected_selection_sha256": expected,
                "actual_selection_sha256": actual,
                "exact_match": match,
            }
        )
        if not match:
            raise CanonicalInputError(
                f"{row['run_slot']} analysis selection SHA differs from canonical run"
            )
    return pd.DataFrame(records)


def _load_frozen_prediction_identity(
    attempt_dir: str | Path, split: str
) -> pd.DataFrame:
    root = Path(attempt_dir) / "01_manifests/frozen_inputs"
    parts: list[pd.DataFrame] = []
    for role, label in (("normal", 0), ("defect", 1)):
        path = root / f"{split}_{role}_manifest.csv"
        if not path.is_file():
            raise CanonicalInputError(f"Missing frozen prediction manifest: {path}")
        frame = pd.read_csv(
            path,
            usecols=["canonical_image_relpath", "Defect"],
            dtype={"canonical_image_relpath": "string"},
        )
        if frame["canonical_image_relpath"].isna().any() or frame[
            "canonical_image_relpath"
        ].duplicated().any():
            raise CanonicalInputError(f"Invalid frozen sample IDs: {path}")
        defect = pd.to_numeric(frame["Defect"], errors="raise").astype(int)
        if not (defect == label).all():
            raise CanonicalInputError(f"Frozen manifest class mismatch: {path}")
        parts.append(
            pd.DataFrame(
                {
                    "sample_id": frame["canonical_image_relpath"].astype(str),
                    "y_true": label,
                }
            )
        )
    expected = pd.concat(parts, ignore_index=True)
    if expected["sample_id"].duplicated().any():
        raise CanonicalInputError(
            f"Frozen {split} normal/defect manifests overlap"
        )
    return expected.sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def _validate_prediction_identity(
    predictions: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    run_slot: str,
    split: str,
) -> dict[str, Any]:
    required = {"sample_id", "y_true"}
    if not required.issubset(predictions.columns):
        raise CanonicalInputError(
            f"{run_slot} {split} predictions lack sample identity columns"
        )
    actual = predictions[["sample_id", "y_true"]].copy()
    if len(actual) != 120_000:
        raise CanonicalInputError(
            f"{run_slot} {split} expected 120000 rows, found {len(actual)}"
        )
    if actual["sample_id"].isna().any() or actual["sample_id"].duplicated().any():
        raise CanonicalInputError(
            f"{run_slot} {split} predictions contain missing/duplicate sample IDs"
        )
    actual["sample_id"] = actual["sample_id"].astype(str)
    actual["y_true"] = pd.to_numeric(actual["y_true"], errors="raise").astype(int)
    normal_count = int((actual["y_true"] == 0).sum())
    defect_count = int((actual["y_true"] == 1).sum())
    if normal_count != 100_000 or defect_count != 20_000:
        raise CanonicalInputError(
            f"{run_slot} {split} class counts differ: "
            f"normal={normal_count}, defect={defect_count}"
        )
    actual = actual.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if len(expected) != len(actual) or not np.array_equal(
        actual["sample_id"].to_numpy(), expected["sample_id"].to_numpy()
    ):
        raise CanonicalInputError(
            f"{run_slot} {split} sample ID set differs from frozen manifests"
        )
    if not np.array_equal(
        actual["y_true"].to_numpy(dtype=int),
        expected["y_true"].to_numpy(dtype=int),
    ):
        raise CanonicalInputError(
            f"{run_slot} {split} labels differ from frozen manifests"
        )
    return {
        "run_slot": run_slot,
        "split": split,
        "row_count": len(actual),
        "normal_count": normal_count,
        "defect_count": defect_count,
        "unique_sample_count": int(actual["sample_id"].nunique()),
        "sample_ids_exact": True,
        "labels_exact": True,
        "status": "PASS",
    }


def _build_epoch_curves(runs: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    metadata = [
        "run_slot",
        "triad_id",
        "condition_slot",
        "condition_id",
        "phase",
        "arm",
        "training_seed",
        "discovery_or_confirmation",
        "machine_id",
        "input_snapshot_id",
        "resume_count",
    ]
    for _, row in runs.iterrows():
        frame = pd.read_csv(
            Path(str(row["attempt_dir"])) / "02_logs/epoch_training_metrics.csv"
        )
        for column in metadata:
            frame[column] = row[column]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _calibration_table(
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    identity_records: list[dict[str, Any]] = []
    expected_by_snapshot: dict[tuple[str, str], pd.DataFrame] = {}
    for _, row in runs.iterrows():
        for split in ("val_cal", "val_op"):
            key = (str(row["input_snapshot_id"]), split)
            if key not in expected_by_snapshot:
                expected_by_snapshot[key] = _load_frozen_prediction_identity(
                    row["attempt_dir"], split
                )
            predictions = _read_predictions(row["attempt_dir"], split)
            identity_records.append(
                _validate_prediction_identity(
                    predictions,
                    expected_by_snapshot[key],
                    run_slot=str(row["run_slot"]),
                    split=split,
                )
            )
            diagnostics = calibration_diagnostics(
                predictions
            )
            records.append(
                {
                    "run_slot": row["run_slot"],
                    "triad_id": row["triad_id"],
                    "condition_slot": row["condition_slot"],
                    "phase": row["phase"],
                    "arm": row["arm"],
                    "training_seed": row["training_seed"],
                    "split": split,
                    **diagnostics,
                }
            )
    return pd.DataFrame(records), pd.DataFrame(identity_records)


def _training_contract_audit(
    runs: pd.DataFrame, training: pd.DataFrame
) -> pd.DataFrame:
    expected_steps = {600: 943, 3000: 961, 6000: 985}
    merged = runs[
        ["run_slot", "budget", "attempt_dir", "resume_count"]
    ].merge(training, on="run_slot", validate="one_to_one")
    records: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        budget = int(row["budget"])
        if budget not in expected_steps:
            raise CanonicalInputError(
                f"{row['run_slot']} has unsupported replay budget {budget}"
            )
        expected = expected_steps[budget]
        epochs_ok = int(row["completed_epochs"]) == 200
        steps_ok = int(row["expected_steps_per_epoch"]) == expected
        simple_total = expected * 200
        actual_total = int(row["optimizer_steps_total"])
        audit_path = (
            Path(str(row["attempt_dir"]))
            / "02_logs/training_execution_audit.json"
        )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        reconciliation = audit.get("partial_step_reconciliation")
        audit_repair = audit.get("audit_repair")
        discarded_partial_steps = int(
            audit.get("discarded_partial_optimizer_steps_total", 0) or 0
        )
        partial_reconciled = bool(
            audit.get("partial_step_reconciled", False)
            or (
                isinstance(reconciliation, dict)
                and reconciliation.get("status") in {"PASS", "RECONCILED"}
            )
            or (
                isinstance(audit_repair, dict)
                and audit_repair.get("repair_schema")
                == "stage1_gapvalue240.audit_partial_step_reconciliation.v1"
                and discarded_partial_steps > 0
            )
        )
        total_ok = actual_total == simple_total
        if not epochs_ok or not steps_ok:
            raise CanonicalInputError(
                f"{row['run_slot']} violates epoch/steps training contract"
            )
        if not total_ok and not partial_reconciled:
            raise CanonicalInputError(
                f"{row['run_slot']} optimizer-step mismatch lacks reconciliation proof"
            )
        records.append(
            {
                "run_slot": row["run_slot"],
                "budget": budget,
                "completed_epochs": int(row["completed_epochs"]),
                "expected_steps_per_epoch_by_budget": expected,
                "recorded_steps_per_epoch": int(row["expected_steps_per_epoch"]),
                "simple_expected_optimizer_steps": simple_total,
                "recorded_optimizer_steps_total": actual_total,
                "epochs_exact": epochs_ok,
                "steps_per_epoch_exact": steps_ok,
                "optimizer_steps_exact": total_ok,
                "partial_step_reconciled": partial_reconciled,
                "discarded_partial_optimizer_steps": discarded_partial_steps,
                "resume_count": int(row["resume_count"]),
                "status": "PASS" if total_ok else "PASS_RECONCILED",
            }
        )
    return pd.DataFrame(records)


def _selection_identity_quality(
    matrix: pd.DataFrame,
    selection_root: str | Path,
    master_sample_index: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    master_path = Path(master_sample_index).resolve()
    if not master_path.is_file():
        raise CanonicalInputError(f"Missing master sample index: {master_path}")
    master = pd.read_csv(
        master_path,
        usecols=[
            "sample_id",
            "y_true",
            "oof_fold",
            "oof_group_id",
            "train_primary_class",
        ],
        dtype={
            "sample_id": "string",
            "oof_fold": "string",
            "oof_group_id": "string",
            "train_primary_class": "string",
        },
    )
    if master["sample_id"].isna().any() or master["sample_id"].duplicated().any():
        raise CanonicalInputError("master_sample_index sample IDs must be unique")
    root = Path(selection_root).resolve()
    enriched: dict[str, pd.DataFrame] = {}
    quality_records: list[dict[str, Any]] = []
    for _, row in matrix.iterrows():
        slot = str(row["run_slot"])
        selection = pd.read_csv(
            root / slot / "selection_manifest.csv",
            dtype={"sample_id": "string", "oof_fold": "string"},
        )
        if selection["sample_id"].isna().any() or selection["sample_id"].duplicated().any():
            raise CanonicalInputError(f"{slot} selection IDs are missing/duplicated")
        merged = selection.merge(
            master,
            on="sample_id",
            how="left",
            validate="one_to_one",
            suffixes=("_selection", "_master"),
            indicator=True,
        )
        if not (merged["_merge"] == "both").all():
            raise CanonicalInputError(
                f"{slot} selection does not join exactly to master_sample_index"
            )
        if not (
            pd.to_numeric(merged["y_true_selection"], errors="raise").astype(int)
            == pd.to_numeric(merged["y_true_master"], errors="raise").astype(int)
        ).all():
            raise CanonicalInputError(f"{slot} labels disagree with master sample index")
        if not (
            merged["oof_fold_selection"].astype(str)
            == merged["oof_fold_master"].astype(str)
        ).all():
            raise CanonicalInputError(f"{slot} folds disagree with master sample index")
        group_counts = merged["oof_group_id"].value_counts(dropna=False)
        shares = group_counts.to_numpy(dtype=float) / len(merged)
        entropy = float(-(shares * np.log(shares)).sum())
        fold_counts = (
            merged["oof_fold_master"].astype(str).value_counts().sort_index()
        )
        fold_shares = fold_counts.reindex(
            [f"{index:02d}" for index in range(10)], fill_value=0
        ).to_numpy(dtype=float) / len(merged)
        feature_record: dict[str, Any] = {}
        for feature in ("mean_p_defect", "correct_rate", "std_p_defect"):
            values = pd.to_numeric(merged[feature], errors="raise")
            if not np.isfinite(values).all():
                raise CanonicalInputError(f"{slot} {feature} contains NaN/Inf")
            feature_record[f"{feature}_mean"] = float(values.mean())
            feature_record[f"{feature}_std"] = float(values.std(ddof=0))
        quality_records.append(
            {
                "run_slot": slot,
                "triad_id": row["triad_id"],
                "condition_slot": row["condition_slot"],
                "arm": row["arm"],
                "budget": int(row["budget"]),
                "training_seed": int(row["training_seed"]),
                "unique_group_count": int(group_counts.size),
                "largest_group_share": float(shares.max()),
                "group_entropy": entropy,
                "normalized_group_entropy": (
                    entropy / np.log(group_counts.size)
                    if group_counts.size > 1
                    else 0.0
                ),
                "fold_counts_json": json.dumps(
                    {str(key): int(value) for key, value in fold_counts.items()},
                    sort_keys=True,
                ),
                "fold_total_variation_from_uniform": float(
                    0.5 * np.abs(fold_shares - 0.1).sum()
                ),
                "primary_class_counts_json": json.dumps(
                    {
                        str(key): int(value)
                        for key, value in merged[
                            "train_primary_class"
                        ].value_counts(dropna=False).items()
                    },
                    sort_keys=True,
                ),
                **feature_record,
            }
        )
        enriched[slot] = merged

    smd_records: list[dict[str, Any]] = []
    for triad_id, group in matrix.groupby("triad_id", sort=True):
        arms = {str(row["arm"]): row for _, row in group.iterrows()}
        treatment = enriched[str(arms["T"]["run_slot"])]
        for control in ("R1", "R2"):
            comparison = enriched[str(arms[control]["run_slot"])]
            record: dict[str, Any] = {
                "triad_id": triad_id,
                "condition_slot": arms["T"]["condition_slot"],
                "control": control,
            }
            smd_values = []
            for feature in ("mean_p_defect", "correct_rate", "std_p_defect"):
                left = pd.to_numeric(treatment[feature], errors="raise").to_numpy()
                right = pd.to_numeric(comparison[feature], errors="raise").to_numpy()
                pooled = np.sqrt((np.var(left, ddof=1) + np.var(right, ddof=1)) / 2)
                smd = (
                    float((np.mean(left) - np.mean(right)) / pooled)
                    if pooled > 0
                    else 0.0
                    if np.mean(left) == np.mean(right)
                    else np.inf
                )
                record[f"smd_{feature}"] = smd
                smd_values.append(abs(smd))
            record["max_abs_smd"] = float(max(smd_values))
            smd_records.append(record)
    return pd.DataFrame(quality_records), pd.DataFrame(smd_records)


def _a02_extended_sensitivity(
    deltas: pd.DataFrame, runs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    a02 = deltas[deltas["condition_slot"].astype(str) == "A02"].copy()
    triad_resumed = (
        runs.groupby("triad_id")["resume_count"].max().gt(0).rename("triad_any_resumed")
    )
    a02 = a02.merge(
        triad_resumed, left_on="triad_id", right_index=True, validate="many_to_one"
    )
    records: list[dict[str, Any]] = []
    selectors: list[tuple[str, pd.Series]] = [
        ("all", pd.Series(True, index=a02.index)),
        ("current_pair_no_resume", ~a02["any_resumed"].astype(bool)),
        ("full_triad_no_resume", ~a02["triad_any_resumed"].astype(bool)),
        ("same_machine", a02["same_machine"].fillna(False)),
        ("cross_machine", a02["same_machine"].eq(False)),
    ]
    snapshot_pair = (
        a02["t_input_snapshot_id"].astype(str)
        + "__"
        + a02["control_input_snapshot_id"].astype(str)
    )
    for value in sorted(snapshot_pair.unique()):
        selectors.append((f"snapshot_pair:{value}", snapshot_pair == value))
    resume_pair_max = a02[["t_resume_count", "control_resume_count"]].max(axis=1)
    for value in sorted(resume_pair_max.unique()):
        selectors.append((f"resume_pair_max:{int(value)}", resume_pair_max == value))
    for control in ("R1", "R2"):
        control_rows = a02[a02["control"] == control]
        for analysis_set, mask in selectors:
            group = control_rows.loc[mask.reindex(control_rows.index, fill_value=False)]
            if group.empty:
                continue
            summary = paired_summary(group["delta_FN"], group["delta_TN"])
            records.append(
                {
                    "condition_slot": "A02",
                    "control": control,
                    "analysis_set": analysis_set,
                    "n": len(group),
                    **summary,
                }
            )

    loo_records: list[dict[str, Any]] = []
    for control in ("R1", "R2"):
        control_rows = a02[a02["control"] == control]
        for omitted in sorted(control_rows["training_seed"].unique()):
            group = control_rows[control_rows["training_seed"] != omitted]
            summary = paired_summary(group["delta_FN"], group["delta_TN"])
            loo_records.append(
                {
                    "condition_slot": "A02",
                    "control": control,
                    "omitted_seed": int(omitted),
                    "n": len(group),
                    **summary,
                }
            )
    return pd.DataFrame(records), pd.DataFrame(loo_records)


def _tail_plot_summary(detail: pd.DataFrame) -> pd.DataFrame:
    selected = detail[
        (detail["scope"].astype(str) == "operational")
        & (detail["score_type"].astype(str) == "raw")
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    grouped = (
        selected.groupby(["condition_id", "condition_slot", "control", "label"])
        .agg(
            mean_tail_shift=("mean_shift", "mean"),
            comparison_count=("triad_id", "nunique"),
        )
        .reset_index()
    )
    shift = grouped.pivot(
        index=["condition_id", "condition_slot", "control"],
        columns="label",
        values="mean_tail_shift",
    ).reset_index()
    counts = (
        grouped.groupby(["condition_id", "condition_slot", "control"])[
            "comparison_count"
        ]
        .max()
        .reset_index()
    )
    result = shift.merge(
        counts,
        on=["condition_id", "condition_slot", "control"],
        validate="one_to_one",
    )
    return result.rename(
        columns={
            "normal": "delta_normal_tail_score",
            "defect": "delta_defect_tail_score",
        }
    )


def _a02_training_curves(epoch_curves: pd.DataFrame) -> pd.DataFrame:
    selected = epoch_curves[
        epoch_curves["condition_slot"].astype(str) == "A02"
    ].copy()
    group_columns = ["arm", "epoch"]
    if "discovery_or_confirmation" in selected.columns:
        group_columns.insert(0, "discovery_or_confirmation")
    result = (
        selected.groupby(group_columns, sort=True)
        .agg(
            run_count=("run_slot", "nunique"),
            top1_mean=("metrics/accuracy_top1", "mean"),
            top1_std=("metrics/accuracy_top1", lambda x: float(x.std(ddof=0))),
            val_loss_mean=("val/loss", "mean"),
            val_loss_std=("val/loss", lambda x: float(x.std(ddof=0))),
            train_loss_mean=("train/loss", "mean"),
        )
        .reset_index()
    )
    # Stable reporting aliases.  The *_mean columns remain for explicitness.
    result["top1"] = result["top1_mean"]
    result["val_loss"] = result["val_loss_mean"]
    result["series"] = (
        result["discovery_or_confirmation"].astype(str)
        + "/"
        + result["arm"].astype(str)
        if "discovery_or_confirmation" in result.columns
        else result["arm"].astype(str)
    )
    return result


def _a02_frontier_summary(frontier: pd.DataFrame) -> pd.DataFrame:
    if frontier.empty:
        return frontier.copy()
    result = (
        frontier.groupby(["phase", "arm", "fn_budget"], sort=True)
        .agg(
            run_count=("run_slot", "nunique"),
            FN=("actual_FN", "mean"),
            FN_std=("actual_FN", lambda x: float(x.std(ddof=0))),
            TN=("actual_TN", "mean"),
            TN_std=("actual_TN", lambda x: float(x.std(ddof=0))),
        )
        .reset_index()
    )
    result["analysis_cohort"] = result["phase"].map(
        {"A": "discovery", "C": "confirmation"}
    ).fillna(result["phase"].astype(str))
    result["series"] = (
        result["analysis_cohort"].astype(str) + "/" + result["arm"].astype(str)
    )
    return result


def _reliability_tables(
    runs: pd.DataFrame,
    training: pd.DataFrame,
    completeness: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    merged = runs.merge(training, on="run_slot", how="one_to_one", validate="one_to_one")
    execution_records: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        audit = json.loads(
            (
                Path(str(row["attempt_dir"]))
                / "02_logs/training_execution_audit.json"
            ).read_text(encoding="utf-8")
        )
        segments = audit.get("resume_segments", [])
        duration = 0.0
        complete_durations = True
        for segment in segments:
            start = segment.get("started_at")
            end = segment.get("ended_at")
            if start is None or end is None:
                complete_durations = False
                continue
            duration += max(0.0, float(end) - float(start))
        execution_records.append(
            {
                "run_slot": row["run_slot"],
                "machine_id": row["machine_id"],
                "resume_count": int(row["resume_count"]),
                "segment_count": len(segments),
                "training_duration_seconds": duration if complete_durations else np.nan,
                "training_duration_hours": (
                    duration / 3600.0 if complete_durations else np.nan
                ),
                "duration_complete": complete_durations,
                "optimizer_steps_total": int(row["optimizer_steps_total"]),
                "epochs": int(row["completed_epochs"]),
            }
        )
    execution = pd.DataFrame(execution_records)
    merged = merged.merge(
        execution[
            [
                "run_slot",
                "training_duration_seconds",
                "training_duration_hours",
                "duration_complete",
                "segment_count",
            ]
        ],
        on="run_slot",
        validate="one_to_one",
    )
    machine = (
        merged.groupby("machine_id", dropna=False)
        .agg(
            run_count=("run_slot", "size"),
            resumed_run_count=("resume_count", lambda x: int((x > 0).sum())),
            duration_complete_run_count=("duration_complete", "sum"),
            duration_missing_run_count=(
                "duration_complete",
                lambda x: int((~x.astype(bool)).sum()),
            ),
            optimizer_steps_total=("optimizer_steps_total", "sum"),
            total_training_hours=("training_duration_hours", "sum"),
            mean_training_hours=("training_duration_hours", "mean"),
            mean_final_top1=("final_top1", "mean"),
            mean_final_val_loss=("final_val_loss", "mean"),
        )
        .reset_index()
    )
    snapshot = (
        merged.groupby("input_snapshot_id", dropna=False)
        .agg(
            run_count=("run_slot", "size"),
            resumed_run_count=("resume_count", lambda x: int((x > 0).sum())),
            mean_TN_at_FN95=("TN_at_FN95", "mean"),
            mean_FN_at_TN68253=("FN_at_TN68253", "mean"),
            mean_gap=("gap_q68_q050", "mean"),
        )
        .reset_index()
    )
    resume = (
        merged.assign(resume_group=np.where(merged.resume_count > 0, "resumed", "continuous"))
        .groupby(["resume_group", "resume_count"], dropna=False)
        .agg(
            run_count=("run_slot", "size"),
            mean_TN_at_FN95=("TN_at_FN95", "mean"),
            mean_FN_at_TN68253=("FN_at_TN68253", "mean"),
            mean_final_top1=("final_top1", "mean"),
        )
        .reset_index()
    )
    historical = pd.DataFrame(completeness.get("nonvalidated_evidence", []))
    return {
        "run_execution_reliability": execution,
        "reliability_by_machine": machine,
        "snapshot_sensitivity": snapshot,
        "resume_sensitivity": resume,
        "historical_failed_attempts": historical,
    }


def run_deep_analysis(
    *,
    extracted_root: str | Path,
    inventory_path: str | Path,
    completeness_audit_path: str | Path,
    matrix_path: str | Path,
    selection_root: str | Path | None = None,
    value_table: str | Path | None = None,
    output_dir: str | Path,
    recompute_predictions: bool = True,
) -> Path:
    """Execute the frozen 240-run deep analysis and atomically build its report."""

    extracted = Path(extracted_root).resolve()
    output = Path(output_dir).resolve()
    if output == extracted or output.is_relative_to(extracted):
        raise CanonicalInputError(
            "Analysis output must not be inside the immutable extracted result root"
        )
    if output.exists() or output.with_name(output.name + ".inprogress").exists():
        raise FileExistsError(f"Refusing to overwrite analysis output: {output}")

    matrix_file = Path(matrix_path).resolve()
    selection_path = (
        Path(selection_root).resolve()
        if selection_root is not None
        else matrix_file.parent / "selections"
    )
    value_path = (
        Path(value_table).resolve()
        if value_table is not None
        else Path(__file__).resolve().parents[1]
        / "artifacts"
        / "stage1_sample_value_experiments"
        / "experiments"
        / "oof_dynamics_gap_value_20260708"
        / "02_sample_value_tables"
        / "sample_value_table.csv"
    )
    if not selection_path.is_dir():
        raise CanonicalInputError(f"Missing frozen selection root: {selection_path}")
    if not value_path.is_file():
        raise CanonicalInputError(f"Missing frozen sample value table: {value_path}")

    completeness = validate_completeness_audit(completeness_audit_path)
    ingestion = ingest_canonical_runs(
        extracted,
        inventory_path,
        matrix_file,
        recompute_predictions=recompute_predictions,
    )
    runs = ingestion.runs
    matrix = pd.read_csv(matrix_file, keep_default_na=False)
    deltas = build_triad_deltas(runs)
    condition_summaries = build_condition_summaries(deltas)
    a02_summaries = build_a02_summaries(deltas)
    sensitivity = build_sensitivity_summaries(deltas)
    cross_method = build_cross_method_comparisons(deltas)
    direct_treatment = build_direct_treatment_comparisons(runs)
    budget = build_budget_comparisons(deltas)
    guard = build_guard_comparisons(deltas)
    guard_vs_normal_only = build_direct_treatment_comparisons(
        runs,
        specs=tuple((f"B{index:02d}", "A02") for index in range(1, 7)),
    )
    a02_extended, a02_leave_one_out = _a02_extended_sensitivity(deltas, runs)

    selection_sha = _selection_sha_audit(runs, selection_path)
    selections = audit_selections(matrix, selection_path)
    selection_quality, selection_smd = _selection_identity_quality(
        matrix,
        selection_path,
        matrix_file.parent / "master_sample_index.csv",
    )
    r2_overlap = selections["triad_overlap"].query(
        "control == 'R2' and scope == 'all'"
    ).reset_index(drop=True)
    condition_lookup = (
        runs[runs["arm"].astype(str) == "T"][
            ["triad_id", "condition_id", "condition_slot"]
        ]
        .drop_duplicates("triad_id")
    )
    r2_overlap = r2_overlap.drop(columns=["condition_slot"], errors="ignore").merge(
        condition_lookup, on="triad_id", validate="one_to_one"
    )
    r2_overlap["effective_unique_contrast_rate"] = r2_overlap[
        "effective_unique_contrast"
    ]
    bottom_gap = pd.DataFrame(
        [
            summarize_selected_score_signs(
                matrix,
                selection_path,
                value_path,
                condition_slot="A13",
                score_column="gap_critical_score",
            )
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="gapvalue240_deep_", dir=output.parent) as temporary:
        temp = Path(temporary)
        control_paths = [
            Path(path) / "04_predictions/val_op_predictions.csv"
            for path in runs.loc[runs["arm"].isin(["R1", "R2"]), "attempt_dir"]
        ]
        consensus = build_control_consensus(
            control_paths, cache_path=temp / "control_scores.float32.memmap"
        )
        tails = define_reference_tails(consensus)

        tail_records: list[pd.DataFrame] = []
        a02_shift_frames: list[tuple[str, int, pd.DataFrame]] = []
        a02_subgroup_frames: list[pd.DataFrame] = []
        by_slot = runs.set_index("run_slot")
        for _, delta in deltas.iterrows():
            treatment = _read_predictions(
                by_slot.loc[delta["t_run_slot"], "attempt_dir"], "val_op"
            )
            control = _read_predictions(
                by_slot.loc[delta["control_run_slot"], "attempt_dir"], "val_op"
            )
            summary, sample_shifts = pair_prediction_tail_shifts(
                treatment, control, tails
            )
            for column in (
                "triad_id",
                "condition_slot",
                "condition_id",
                "phase",
                "training_seed",
                "control",
                "machine_pair",
                "any_resumed",
            ):
                summary[column] = delta[column]
            tail_records.append(summary)
            if str(delta["condition_slot"]) == "A02":
                sample_shifts = sample_shifts.copy()
                sample_shifts["triad_id"] = str(delta["triad_id"])
                sample_shifts["control"] = str(delta["control"])
                sample_shifts["training_seed"] = int(delta["training_seed"])
                a02_shift_frames.append(
                    (
                        str(delta["control"]),
                        int(delta["training_seed"]),
                        sample_shifts,
                    )
                )
                a02_subgroup_frames.append(sample_shifts)
        tail_summary = pd.concat(tail_records, ignore_index=True)
        a02_consistency = summarize_sample_shift_consistency(a02_shift_frames)
        subgroup_metadata = load_subgroup_metadata(runs.iloc[0]["attempt_dir"])
        subgroup_summary = summarize_shift_subgroups(
            a02_subgroup_frames, subgroup_metadata
        )
        subgroup_availability = subgroup_metadata.attrs.get(
            "field_availability", {}
        )
        subgroup_availability_table = pd.DataFrame(
            [
                {
                    "field": str(field),
                    "available": (
                        bool(value) if isinstance(value, bool) else bool(value)
                    ),
                    "details": (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (list, dict))
                        else str(value)
                    ),
                }
                for field, value in subgroup_availability.items()
            ]
        )

        calibration, prediction_identity = _calibration_table(runs)
        epoch_curves = _build_epoch_curves(runs)
        if len(epoch_curves) != 48_000:
            raise CanonicalInputError(
                f"Expected 48,000 epoch curve rows, found {len(epoch_curves)}"
            )
        frontier_detail = build_threshold_frontier(runs)
        frontier = _a02_frontier_summary(frontier_detail)
        a02_curves = _a02_training_curves(epoch_curves)
        tail_plot = _tail_plot_summary(tail_summary)
        training_contract = _training_contract_audit(
            runs, ingestion.training_summaries
        )
        reliability = _reliability_tables(
            runs, ingestion.training_summaries, completeness
        )
        hypotheses = build_hypothesis_registry(
            a02_summaries=a02_summaries,
            condition_summaries=condition_summaries,
            budget_comparisons=budget,
            guard_comparisons=guard,
            r2_overlap=r2_overlap,
        )
        pooled_a02 = a02_summaries[
            a02_summaries["analysis_cohort"].astype(str) == "pooled"
        ].set_index("control")
        discovery_a02 = a02_summaries[
            a02_summaries["analysis_cohort"].astype(str) == "discovery"
        ].set_index("control")
        confirmation_a02 = a02_summaries[
            a02_summaries["analysis_cohort"].astype(str) == "confirmation"
        ].set_index("control")

        def effect_sentence(frame: pd.DataFrame, control: str) -> str:
            if control not in frame.index:
                return f"{control}: unavailable"
            row = frame.loc[control]
            return (
                f"{control}: n={int(row['n'])}, mean ΔFN="
                f"{float(row['mean_delta_FN']):+.3f}, FN upper95="
                f"{float(row['FN_one_sided_95_upper']):+.3f}, worst ΔFN="
                f"{float(row['worst_delta_FN']):+.3f}, mean ΔTN="
                f"{float(row['mean_delta_TN']):+.3f}, TN lower95="
                f"{float(row['TN_one_sided_95_lower']):+.3f}"
            )

        hypothesis_status = hypotheses.set_index("hypothesis_id")["status"].to_dict()
        primary_result = (
            "A02 GapCritical-Strict B3000 未通过预注册双对照成功门槛："
            f"{effect_sentence(pooled_a02, 'R1')}；"
            f"{effect_sentence(pooled_a02, 'R2')}。"
            "因此当前 240-run val_op 证据不支持把 A02 作为优于随机对照的主方法。"
        )
        tables: dict[str, pd.DataFrame] = {
            "canonical_run_metrics": runs,
            "metric_recompute_audit": ingestion.metric_audit,
            "training_curve_summary": ingestion.training_summaries,
            "epoch_training_curves": epoch_curves,
            "triad_control_deltas": deltas,
            "condition_control_summaries": condition_summaries,
            "a02_discovery_confirmation_combined": a02_summaries,
            "sensitivity_results": sensitivity,
            "cross_method_seed_paired": cross_method,
            "direct_treatment_comparisons": direct_treatment,
            "budget_response": budget,
            "guard_policy_contrasts": guard,
            "guard_vs_a02_normal_only_direct": guard_vs_normal_only,
            "a02_extended_sensitivity": a02_extended,
            "a02_leave_one_seed_out": a02_leave_one_out,
            "selection_sha_audit": selection_sha,
            "selection_run_audit": selections["run_audit"],
            "selection_composition": selections["composition"],
            "selection_triad_overlap": selections["triad_overlap"],
            "selection_method_overlap": selections["method_overlap"],
            "selection_identity_quality": selection_quality,
            "selection_feature_smd": selection_smd,
            "r2_overlap_power_audit": r2_overlap,
            "bottom_gap_signs": bottom_gap,
            "control_consensus": consensus,
            "fixed_reference_tails": tails,
            "prediction_tail_summary": tail_plot,
            "prediction_tail_detail": tail_summary,
            "sample_shift_consistency_a02": a02_consistency,
            "subgroup_field_availability": subgroup_availability_table,
            "a02_shift_subgroups": subgroup_summary,
            "calibration_diagnostics": calibration,
            "prediction_identity_audit": prediction_identity,
            "a02_threshold_frontier": frontier,
            "a02_threshold_frontier_detail": frontier_detail,
            "a02_training_curves": a02_curves,
            "training_contract_audit": training_contract,
            **reliability,
        }
        metadata = {
            "validated_runs": len(runs),
            "triads": runs["triad_id"].nunique(),
            "paired_comparisons": len(deltas),
            "epoch_rows": len(epoch_curves),
            "control_consensus_runs": len(control_paths),
            "prediction_identity_audits": len(prediction_identity),
            "val_op_prediction_files_verified": int(
                (prediction_identity["split"] == "val_op").sum()
            ),
            "val_cal_prediction_files_verified": int(
                (prediction_identity["split"] == "val_cal").sum()
            ),
            "input_snapshots": runs["input_snapshot_id"].nunique(),
            "resumed_runs": int((runs["resume_count"] > 0).sum()),
            "full_val_op_recompute": bool(recompute_predictions),
            "semantic_snapshot_exception_authorized": completeness[
                "semantic_snapshot_exception_authorized"
            ],
            "a02_vs_r1_status": hypothesis_status.get("H1_A02_vs_R1"),
            "a02_vs_r2_status": hypothesis_status.get("H2_A02_vs_R2"),
            "primary_result": primary_result,
            "conclusion_boundary": (
                "val_op internal discovery/replication only; no blind/external claim"
            ),
        }
        analysis_contract = {
            "analysis_id": "stage1_gapvalue240_full_analysis_v1",
            "source_policy": "canonical inventory attempts only; extracted root read-only",
            "expected_runs": 240,
            "expected_triads": 80,
            "expected_comparisons": 160,
            "primary_delta_FN": "FN_at_TN68253(T)-FN_at_TN68253(control)",
            "primary_delta_TN": "TN_at_FN95(T)-TN_at_FN95(control)",
            "safety_gate": "one-sided 95% t-CI upper <= +2 and worst seed <= +5",
            "tn_gate": "after safety, one-sided 95% t-CI lower > 0",
            "controls": "R1 and R2 are never pooled",
            "a02_primary_contract_cohort": "pooled-8; discovery-3 and confirmation-5 also reported separately",
            "a02_machine_rule": "Phase C and pooled-8 retain machine-confounded labels; no causal correction",
            "r2_rule": "high-overlap near-treatment mechanism control; effective unique contrast shown beside effects",
            "fixed_reference_tails": {
                "reference": "median score_raw across all 160 control runs",
                "operational_normal_high_risk_count": 31_747,
                "operational_defect_low_risk_count": 95,
                "tail_gap_normal_fraction": 0.10,
                "tail_gap_defect_fraction": 0.05,
            },
            "score_roles": {
                "score_raw": "model-output mechanism analysis",
                "score": "Platt-calibrated deployment probability",
            },
            "tie_rule": "score >= threshold; whole equal-score group moves together",
            "steps_per_epoch_by_budget": {600: 943, 3000: 961, 6000: 985},
            "three_seed_conditions": "exploratory; never labelled confirmed",
            "conclusion_boundary": metadata["conclusion_boundary"],
        }
        audits = {
            "completeness": completeness,
            "metric_recompute": ingestion.metric_audit,
            "selection_sha": selection_sha,
            "prediction_identity": prediction_identity,
            "training_contract": training_contract,
            "subgroup_field_availability": subgroup_availability,
            "source_boundaries": {
                "extracted_root": str(extracted),
                "inventory": str(Path(inventory_path).resolve()),
                "completeness_audit": str(
                    Path(completeness_audit_path).resolve()
                ),
                "matrix": str(matrix_file),
                "selection_root": str(selection_path),
                "value_table": str(value_path),
                "source_read_only": True,
                "output_outside_extracted_root": True,
            },
        }
        narrative = {
            "核心结论": primary_result,
            "三层 A02 结果": (
                "Discovery-3："
                f"{effect_sentence(discovery_a02, 'R1')}；"
                f"{effect_sentence(discovery_a02, 'R2')}。"
                "Confirmation-5："
                f"{effect_sentence(confirmation_a02, 'R1')}；"
                f"{effect_sentence(confirmation_a02, 'R2')}。"
                "Pooled-8 是合同统计，但包含机器混杂，不能消除 Phase C 的执行差异。"
            ),
            "分析边界": (
                "本报告检验训练动态选样的内部发现与复现证据，不把单个最高分 "
                "run 当作结论，也不作 blind/external 泛化声明。"
            ),
            "机器混杂": (
                "Phase C 五个确认 triad 的 treatment 与 controls 分处不同机器；"
                "pooled-8 只作为预注册算术结果，不能消除机器混杂。"
            ),
            "R2 对照解释": (
                "R2 是高重合、低功效的近 treatment 机制对照；R1 是完全不重合"
                "的主要随机基线，两者必须分别解释。"
                f"本批 R2 的中位有效独特对比比例为 "
                f"{float(r2_overlap['effective_unique_contrast'].median()):.2%}。"
            ),
            "BottomGap 负对照": (
                f"A13 实际包含 {int(bottom_gap.iloc[0]['negative_count'])} 个负分样本和 "
                f"{int(bottom_gap.iloc[0]['nonnegative_count'])} 个非负分样本；"
                "A13 数字方向有害，但 A02 本身也未形成预期正向结果，因此不能宣称"
                "已经建立正负方向因果对照。"
            ),
            "Defect guard": (
                "Phase B 仅有 3 seeds，且主要 guard treatment-control 比较存在机器混杂。"
                "报告保留剂量和策略的数值差异，但将 guard 结论标为 INCONCLUSIVE。"
            ),
            "子组": (
                "子组字段只使用冻结 val_op manifests 中真实存在的字段；"
                "primary_defect_class 缺失时明确标记不可用，不作推断。"
            ),
        }
        return build_deep_report(
            output,
            tables=tables,
            metadata=metadata,
            hypothesis_registry=hypotheses,
            analysis_contract=analysis_contract,
            audits=audits,
            narrative=narrative,
        )
