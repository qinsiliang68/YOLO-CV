from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .checkpointing import load_checkpoint
from .columnar import ColumnarManifest, write_zstd_parquet
from .errors import ErrorCode, SctsrError
from .ledger_schema import FRONTIER_SCHEMA
from .prediction_artifact import PredictionRow, validate_prediction_rows
from .serialization import atomic_write_json, sha256_file


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    fn_budget: int
    actual_fn: int
    tn: int
    fp: int
    tp: int
    threshold: float
    threshold_rule: str
    tie_size: int
    reachable: bool
    defect_count: int
    normal_count: int
    normalized_tn: float
    checkpoint_sha256: str
    prediction_artifact_sha256: str


@dataclass(frozen=True, slots=True)
class FrontierSummary:
    raw_frontier_normalized_auc: float
    tn_at_fn95: int | None
    fn_at_tn68253: int | None
    threshold_at_fn95: float | None
    threshold_at_tn68253: float | None
    target_tn_reachable: bool
    anchor_thresholds_equal: bool | None
    tie_group_count: int
    max_fn_budget: int
    defect_count: int
    normal_count: int


@dataclass(frozen=True, slots=True)
class _Boundary:
    threshold: float
    fn: int
    tn: int
    fp: int
    tp: int
    tie_size: int


def _no_positive_threshold(rows: Sequence[PredictionRow]) -> float:
    """Return a finite threshold that predicts no rows as defect.

    Canonical JSON forbids Infinity. Because probabilities are finite, one ULP
    above the largest observed value represents the same decision boundary as
    ``+inf`` while remaining serializable and reproducible.
    """

    maximum = max(float(row.p_defect_raw) for row in rows)
    threshold = math.nextafter(maximum, math.inf)
    if not math.isfinite(threshold):
        raise SctsrError(
            ErrorCode.SCHEMA_VALIDATION_FAILED,
            "Could not construct a finite no-positive prediction threshold",
            observed=maximum,
        )
    return threshold


def _boundaries(rows: Sequence[PredictionRow]) -> list[_Boundary]:
    defect_count = sum(row.y_true == 1 for row in rows)
    normal_count = len(rows) - defect_count
    boundaries = [_Boundary(_no_positive_threshold(rows), defect_count, normal_count, 0, 0, 0)]

    by_probability: dict[float, list[PredictionRow]] = {}
    for row in rows:
        by_probability.setdefault(float(row.p_defect_raw), []).append(row)

    tp = 0
    fp = 0
    for probability in sorted(by_probability, reverse=True):
        group = by_probability[probability]
        tp += sum(row.y_true == 1 for row in group)
        fp += sum(row.y_true == 0 for row in group)
        boundaries.append(
            _Boundary(
                threshold=probability,
                fn=defect_count - tp,
                tn=normal_count - fp,
                fp=fp,
                tp=tp,
                tie_size=len(group),
            )
        )
    return boundaries


def compute_tie_safe_frontier(
    rows: Sequence[PredictionRow],
    *,
    max_fn: int = 95,
    target_tn: int = 68_253,
    checkpoint_sha256: str = "UNREGISTERED",
    prediction_artifact_sha256: str = "UNREGISTERED",
) -> tuple[tuple[FrontierPoint, ...], FrontierSummary]:
    validate_prediction_rows(rows)
    for field, value in (
        ("checkpoint_sha256", checkpoint_sha256),
        ("prediction_artifact_sha256", prediction_artifact_sha256),
    ):
        if len(value) != 64 or value != value.upper() or any(char not in "0123456789ABCDEF" for char in value):
            raise SctsrError(
                ErrorCode.FRONTIER_INVALID,
                "Frontier computation requires canonical checkpoint and prediction artifact bindings",
                failing_field=field,
                observed=value,
            )
    if type(max_fn) is not int or max_fn < 0:
        raise ValueError("max_fn must be a non-negative integer")
    if type(target_tn) is not int or target_tn < 0:
        raise ValueError("target_tn must be a non-negative integer")

    boundaries = _boundaries(rows)
    defect_count = sum(row.y_true == 1 for row in rows)
    normal_count = len(rows) - defect_count
    points: list[FrontierPoint] = []

    for budget in range(max_fn + 1):
        feasible = [boundary for boundary in boundaries if boundary.fn <= budget]
        if not feasible:  # defensive: the all-positive boundary is always feasible for budget >= 0
            raise SctsrError(
                ErrorCode.SCHEMA_VALIDATION_FAILED,
                "No reachable tie-safe frontier point exists for the FN budget",
                observed=budget,
            )
        # Frozen tie breaker: largest TN, then higher threshold, then lower actual FN.
        best = max(feasible, key=lambda boundary: (boundary.tn, boundary.threshold, -boundary.fn))
        points.append(
            FrontierPoint(
                fn_budget=budget,
                actual_fn=best.fn,
                tn=best.tn,
                fp=best.fp,
                tp=best.tp,
                threshold=best.threshold,
                threshold_rule="p_defect_raw >= threshold",
                tie_size=best.tie_size,
                reachable=True,
                defect_count=defect_count,
                normal_count=normal_count,
                normalized_tn=best.tn / normal_count if normal_count else 0.0,
                checkpoint_sha256=checkpoint_sha256,
                prediction_artifact_sha256=prediction_artifact_sha256,
            )
        )

    if max_fn == 0:
        nauc = points[0].normalized_tn
    else:
        nauc = (
            sum(
                0.5 * (points[index].normalized_tn + points[index + 1].normalized_tn)
                for index in range(max_fn)
            )
            / max_fn
        )

    target_candidates = [boundary for boundary in boundaries if boundary.tn >= target_tn]
    if target_candidates:
        # Primary rule is minimum FN. Remaining keys are deterministic diagnostics only.
        target_best = min(
            target_candidates,
            key=lambda boundary: (boundary.fn, -boundary.tn, -boundary.threshold),
        )
        target_reachable = True
        target_fn: int | None = target_best.fn
        target_threshold: float | None = target_best.threshold
    else:
        target_reachable = False
        target_fn = None
        target_threshold = None

    fn_anchor = points[95] if max_fn >= 95 else None
    threshold_fn = fn_anchor.threshold if fn_anchor is not None else None
    summary = FrontierSummary(
        raw_frontier_normalized_auc=nauc,
        tn_at_fn95=fn_anchor.tn if fn_anchor is not None else None,
        fn_at_tn68253=target_fn,
        threshold_at_fn95=threshold_fn,
        threshold_at_tn68253=target_threshold,
        target_tn_reachable=target_reachable,
        anchor_thresholds_equal=(threshold_fn == target_threshold)
        if threshold_fn is not None and target_threshold is not None
        else None,
        tie_group_count=len(boundaries) - 1,
        max_fn_budget=max_fn,
        defect_count=defect_count,
        normal_count=normal_count,
    )
    return tuple(points), summary


def validate_checkpoint_for_evaluation(
    path: str | Path,
    *,
    epoch: int,
    mode: str,
    expected_sha256: str | None = None,
    expected_source_tree_digest: str | None = None,
    expected_training_seed: int | None = None,
) -> dict[str, Any]:
    checkpoint = Path(path)
    name = checkpoint.name.lower()
    if name == "best.pt" or "best.pt" in name:
        raise SctsrError(ErrorCode.BEST_PT_FORBIDDEN, "best.pt is forbidden")
    if mode == "formal" and epoch != 200:
        raise SctsrError(
            ErrorCode.CHECKPOINT_EPOCH_FORBIDDEN,
            "Formal endpoint is fixed at E200",
            observed=epoch,
            expected=200,
        )
    if mode == "trajectory" and epoch not in {120, 140, 150, 160, 180}:
        raise SctsrError(
            ErrorCode.CHECKPOINT_EPOCH_FORBIDDEN,
            "Trajectory evaluation accepts only preregistered anchors",
        )
    if mode not in {"formal", "trajectory", "synthetic"}:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Unknown evaluation mode")
    if not checkpoint.is_file():
        raise SctsrError(
            ErrorCode.PARENT_CHECKPOINT_INCOMPLETE,
            "Registered evaluation checkpoint is missing",
            artifact_path=str(checkpoint),
        )
    payload = load_checkpoint(checkpoint, expected_sha256=expected_sha256, expected_epoch=epoch)
    observed_sha = sha256_file(checkpoint)
    if expected_source_tree_digest is not None and str(payload["source_tree_digest"]).upper() != expected_source_tree_digest.upper():
        raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Evaluation checkpoint source tree digest mismatch")
    if expected_training_seed is not None and int(payload["training_seed"]) != int(expected_training_seed):
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Evaluation checkpoint training seed mismatch")
    return {
        "status": "PASS",
        "checkpoint_path": checkpoint.resolve().as_posix(),
        "checkpoint_sha256": observed_sha,
        "checkpoint_epoch": int(payload["epoch"]),
        "training_seed": int(payload["training_seed"]),
        "source_tree_digest": str(payload["source_tree_digest"]),
        "mode": mode,
        "selection_semantic": "NOT_FOR_METHOD_SELECTION" if mode == "trajectory" else "ENDPOINT_ONLY_NOT_FOR_SELECTION" if mode == "formal" else "SYNTHETIC_NOT_SCIENTIFIC_RESULT",
    }


def write_frontier_artifacts(
    points: Sequence[FrontierPoint],
    summary: FrontierSummary,
    *,
    frontier_path: str | Path,
    summary_path: str | Path,
    evaluation_mode: str,
    split_role: str,
    checkpoint_epoch: int,
) -> tuple[ColumnarManifest, dict[str, Any]]:
    if evaluation_mode == "formal":
        if checkpoint_epoch != 200 or split_role != "val_op" or len(points) != 96 or summary.max_fn_budget != 95:
            raise SctsrError(ErrorCode.FRONTIER_INVALID, "Formal frontier must be the 96-point E200 val_op endpoint")
        selection_semantic = "ENDPOINT_ONLY_NOT_FOR_SELECTION"
    elif evaluation_mode == "trajectory":
        if checkpoint_epoch not in {120, 140, 150, 160, 180} or split_role != "val_model":
            raise SctsrError(ErrorCode.FRONTIER_INVALID, "Trajectory frontier identity is not preregistered")
        selection_semantic = "NOT_FOR_METHOD_SELECTION"
    elif evaluation_mode == "synthetic":
        if split_role != "synthetic":
            raise SctsrError(ErrorCode.SYNTHETIC_RESULT_MISLABELLED, "Synthetic frontier has a non-synthetic split role")
        selection_semantic = "SYNTHETIC_NOT_SCIENTIFIC_RESULT"
    else:
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Unknown frontier evaluation mode")
    if split_role.lower() in {"test", "blind_test", "blind_holdout", "holdout_test"}:
        raise SctsrError(ErrorCode.TEST_ACCESS_FORBIDDEN, "Test and blind holdout frontier publication is forbidden")
    budgets = [point.fn_budget for point in points]
    if budgets != list(range(summary.max_fn_budget + 1)):
        raise SctsrError(ErrorCode.FRONTIER_INVALID, "Frontier budgets are incomplete, duplicated, or out of order")
    checkpoint_shas = {point.checkpoint_sha256 for point in points}
    prediction_shas = {point.prediction_artifact_sha256 for point in points}
    if len(checkpoint_shas) != 1 or len(prediction_shas) != 1:
        raise SctsrError(ErrorCode.FRONTIER_INVALID, "Frontier rows mix checkpoint or prediction artifact identities")
    manifest = write_zstd_parquet(
        [asdict(point) for point in points],
        frontier_path,
        schema_version="stage1.sctsr.frontier.v1",
        schema=FRONTIER_SCHEMA,
        require_run_epoch_partition=True,
    )
    payload: dict[str, Any] = {
        "schema_version": "stage1.sctsr.frontier_summary.v1",
        **asdict(summary),
        "evaluation_mode": evaluation_mode,
        "split_role": split_role,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_sha256": next(iter(checkpoint_shas)),
        "prediction_artifact_sha256": next(iter(prediction_shas)),
        "frontier_artifact_sha256": manifest.sha256,
        "frontier_row_count": len(points),
        "selection_semantic": selection_semantic,
        "two_anchor_thresholds_are_independent": True,
    }
    atomic_write_json(summary_path, payload)
    return manifest, payload
