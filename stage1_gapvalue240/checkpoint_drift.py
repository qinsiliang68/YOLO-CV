"""Stream-safe checkpoint metadata and three-anchor parameter drift analysis."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

import pandas as pd

from .util import sha256_file


@dataclass(frozen=True)
class StateDriftResult:
    tensor_drift: pd.DataFrame
    group_drift: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class CheckpointTripletResult:
    tensor_drift: pd.DataFrame
    group_drift: pd.DataFrame
    optimizer_groups: pd.DataFrame
    run_summary: dict[str, Any]
    best_epoch_candidates: dict[str, Any]


def _shape_text(tensor: Any) -> str:
    return "x".join(str(int(value)) for value in tensor.shape)


def _block_group(name: str) -> str:
    match = re.match(r"^model\.(\d+)(?:\.|$)", name)
    return f"block_{int(match.group(1))}" if match else name.split(".", 1)[0]


def _tensor_kind(name: str) -> str:
    if name.endswith("num_batches_tracked"):
        return "BUFFER_COUNTER"
    if name.endswith("running_mean") or name.endswith("running_var"):
        return "BUFFER_STATISTIC"
    if name.endswith(".weight"):
        return "WEIGHT"
    if name.endswith(".bias"):
        return "BIAS"
    return "OTHER"


def checkpoint_model_state(
    checkpoint: Mapping[str, Any],
    *,
    anchor: str,
) -> Mapping[str, Any]:
    """Return the trained model state used by each checkpoint anchor.

    Ultralytics strips ``best.pt`` to ``model`` and retains the resumable EMA in
    ``last.pt``.  The initial checkpoint also exposes ``model``.  Keeping this
    choice explicit prevents accidentally comparing the stale ``last.model``
    object against the final EMA used for inference.
    """

    if anchor not in {"initial", "best", "last"}:
        raise ValueError(f"Unsupported checkpoint anchor: {anchor}")
    value = checkpoint.get("ema") if anchor == "last" else checkpoint.get("model")
    if value is None and anchor == "last":
        value = checkpoint.get("model")
    if value is None:
        raise ValueError(f"Checkpoint has no usable model state for {anchor}")
    if isinstance(value, Mapping):
        return value
    state_dict = getattr(value, "state_dict", None)
    if not callable(state_dict):
        raise ValueError(f"Checkpoint {anchor} model state is not state-dict compatible")
    return state_dict()


def best_epoch_candidates(
    checkpoint: Mapping[str, Any],
    epoch_rows: pd.DataFrame,
    *,
    absolute_tolerance: float = 5e-7,
) -> dict[str, Any]:
    """Recover every epoch compatible with stripped ``best.pt`` metrics.

    A stripped Ultralytics best checkpoint has ``epoch=-1``.  When multiple
    epochs share the saved top-1/top-5 metrics, reporting a single inferred
    epoch would invent precision, so this function preserves the full tie set.
    """

    metrics = checkpoint.get("train_metrics") or {}
    required = {"epoch", "metrics/accuracy_top1", "metrics/accuracy_top5"}
    missing = required - set(epoch_rows.columns)
    if missing:
        raise ValueError(f"Epoch rows missing best-candidate columns: {sorted(missing)}")
    target_top1 = metrics.get("metrics/accuracy_top1")
    target_top5 = metrics.get("metrics/accuracy_top5")
    if target_top1 is None or target_top5 is None:
        return {
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "checkpoint_best_fitness": checkpoint.get("best_fitness"),
            "checkpoint_metric_top1": target_top1,
            "checkpoint_metric_top5": target_top5,
            "checkpoint_metric_val_loss": metrics.get("val/loss"),
            "checkpoint_metric_fitness": metrics.get("fitness"),
            "candidate_count": 0,
            "candidate_epochs": "",
            "candidate_epoch_earliest": pd.NA,
            "candidate_epoch_latest": pd.NA,
            "candidate_basis": "checkpoint_metrics_unavailable",
        }
    top1 = pd.to_numeric(epoch_rows["metrics/accuracy_top1"], errors="raise")
    top5 = pd.to_numeric(epoch_rows["metrics/accuracy_top5"], errors="raise")
    mask = (top1 - float(target_top1)).abs().le(absolute_tolerance) & (
        top5 - float(target_top5)
    ).abs().le(absolute_tolerance)
    candidates = sorted(pd.to_numeric(epoch_rows.loc[mask, "epoch"]).astype(int).tolist())
    return {
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_fitness": checkpoint.get("best_fitness"),
        "checkpoint_metric_top1": float(target_top1),
        "checkpoint_metric_top5": float(target_top5),
        "checkpoint_metric_val_loss": metrics.get("val/loss"),
        "checkpoint_metric_fitness": metrics.get("fitness"),
        "candidate_count": len(candidates),
        "candidate_epochs": ";".join(map(str, candidates)),
        "candidate_epoch_earliest": candidates[0] if candidates else pd.NA,
        "candidate_epoch_latest": candidates[-1] if candidates else pd.NA,
        "candidate_basis": "checkpoint_train_metrics_top1_top5",
    }


def compare_state_dicts(
    left_state: Mapping[str, Any],
    right_state: Mapping[str, Any],
    *,
    transition: str,
) -> StateDriftResult:
    """Compare two model state mappings without assuming identical heads."""

    import torch

    records: list[dict[str, Any]] = []
    accumulators: dict[str, dict[str, float]] = {}
    for name in sorted(set(left_state) | set(right_state)):
        left = left_state.get(name)
        right = right_state.get(name)
        row: dict[str, Any] = {
            "transition": transition,
            "tensor_name": name,
            "group_name": _block_group(name),
            "tensor_kind": _tensor_kind(name),
            "left_shape": _shape_text(left) if left is not None else "",
            "right_shape": _shape_text(right) if right is not None else "",
            "left_dtype": str(left.dtype) if left is not None else "",
            "right_dtype": str(right.dtype) if right is not None else "",
            "left_elements": int(left.numel()) if left is not None else 0,
            "right_elements": int(right.numel()) if right is not None else 0,
        }
        if left is None:
            row["comparison_status"] = "MISSING_LEFT"
        elif right is None:
            row["comparison_status"] = "MISSING_RIGHT"
        elif tuple(left.shape) != tuple(right.shape):
            row["comparison_status"] = "SHAPE_MISMATCH"
        elif not (torch.is_floating_point(left) and torch.is_floating_point(right)):
            row["comparison_status"] = "MATCHED_NON_FLOAT"
        else:
            left64 = left.detach().cpu().to(dtype=torch.float64).reshape(-1)
            right64 = right.detach().cpu().to(dtype=torch.float64).reshape(-1)
            delta = right64 - left64
            left_sq = float(torch.dot(left64, left64))
            right_sq = float(torch.dot(right64, right64))
            delta_sq = float(torch.dot(delta, delta))
            dot = float(torch.dot(left64, right64))
            left_l2 = math.sqrt(max(left_sq, 0.0))
            right_l2 = math.sqrt(max(right_sq, 0.0))
            delta_l2 = math.sqrt(max(delta_sq, 0.0))
            denominator = left_l2 * right_l2
            row.update(
                {
                    "comparison_status": "MATCHED",
                    "left_l2": left_l2,
                    "right_l2": right_l2,
                    "delta_l2": delta_l2,
                    "relative_l2": delta_l2 / max(left_l2, 1e-30),
                    "cosine_similarity": dot / denominator if denominator else math.nan,
                    "mean_abs_delta": float(delta.abs().mean()) if delta.numel() else 0.0,
                    "max_abs_delta": float(delta.abs().max()) if delta.numel() else 0.0,
                }
            )
            for group_name in (str(row["group_name"]), "ALL"):
                group = accumulators.setdefault(
                    group_name,
                    {
                        "matched_float_tensors": 0.0,
                        "matched_elements": 0.0,
                        "left_sq": 0.0,
                        "right_sq": 0.0,
                        "delta_sq": 0.0,
                        "dot": 0.0,
                    },
                )
                group["matched_float_tensors"] += 1
                group["matched_elements"] += int(left.numel())
                group["left_sq"] += left_sq
                group["right_sq"] += right_sq
                group["delta_sq"] += delta_sq
                group["dot"] += dot
        records.append(row)

    tensor_frame = pd.DataFrame(records)
    group_records: list[dict[str, Any]] = []
    for group_name, values in sorted(accumulators.items()):
        left_l2 = math.sqrt(max(values["left_sq"], 0.0))
        right_l2 = math.sqrt(max(values["right_sq"], 0.0))
        delta_l2 = math.sqrt(max(values["delta_sq"], 0.0))
        denominator = left_l2 * right_l2
        group_records.append(
            {
                "transition": transition,
                "group_name": group_name,
                "matched_float_tensors": int(values["matched_float_tensors"]),
                "matched_elements": int(values["matched_elements"]),
                "left_l2": left_l2,
                "right_l2": right_l2,
                "delta_l2": delta_l2,
                "relative_l2": delta_l2 / max(left_l2, 1e-30),
                "cosine_similarity": (
                    values["dot"] / denominator if denominator else math.nan
                ),
            }
        )
    status_counts = tensor_frame["comparison_status"].value_counts().to_dict()
    summary = {
        "transition": transition,
        "left_tensor_count": len(left_state),
        "right_tensor_count": len(right_state),
        "matched_float_tensors": int(status_counts.get("MATCHED", 0)),
        "matched_non_float_tensors": int(status_counts.get("MATCHED_NON_FLOAT", 0)),
        "shape_mismatch_tensors": int(status_counts.get("SHAPE_MISMATCH", 0)),
        "missing_left_tensors": int(status_counts.get("MISSING_LEFT", 0)),
        "missing_right_tensors": int(status_counts.get("MISSING_RIGHT", 0)),
    }
    return StateDriftResult(
        tensor_drift=tensor_frame,
        group_drift=pd.DataFrame(group_records),
        summary=summary,
    )


def checkpoint_optimizer_metadata(
    optimizer_state: Mapping[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Expose active versus scheduler-only optimizer groups from last.pt."""

    if optimizer_state is None:
        return pd.DataFrame(), {
            "optimizer_group_count": 0,
            "active_optimizer_group_count": 0,
            "empty_optimizer_group_count": 0,
            "optimizer_state_entries": 0,
        }
    groups = list(optimizer_state.get("param_groups", []))
    records = []
    for index, group in enumerate(groups):
        params = list(group.get("params", []))
        records.append(
            {
                "group_index": index,
                "active": bool(params),
                "parameter_count": len(params),
                "param_group": group.get("param_group"),
                "lr": group.get("lr"),
                "initial_lr": group.get("initial_lr"),
                "momentum": group.get("momentum"),
                "weight_decay": group.get("weight_decay"),
                "nesterov": group.get("nesterov"),
                "use_muon": bool(group.get("use_muon", False)),
            }
        )
    active = sum(bool(record["active"]) for record in records)
    summary = {
        "optimizer_group_count": len(records),
        "active_optimizer_group_count": active,
        "empty_optimizer_group_count": len(records) - active,
        "optimizer_state_entries": len(optimizer_state.get("state", {})),
    }
    return pd.DataFrame(records), summary


def _state_element_count(state: Mapping[str, Any]) -> int:
    return int(
        sum(
            int(value.numel())
            for value in state.values()
            if hasattr(value, "numel")
        )
    )


def scan_checkpoint_triplet(
    *,
    run_slot: str,
    initial_checkpoint: str | Path,
    best_checkpoint: str | Path,
    last_checkpoint: str | Path,
    epoch_rows: pd.DataFrame,
    known_sha256: Mapping[str, str] | None = None,
) -> CheckpointTripletResult:
    """Load exactly one initial/best/last triplet and release it after analysis."""

    import torch

    initial_path = Path(initial_checkpoint).resolve()
    best_path = Path(best_checkpoint).resolve()
    last_path = Path(last_checkpoint).resolve()
    for path in (initial_path, best_path, last_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    if not all(isinstance(value, Mapping) for value in (initial, best, last)):
        raise ValueError(f"Checkpoint triplet for {run_slot} must contain mapping roots")
    try:
        initial_state = checkpoint_model_state(initial, anchor="initial")
        best_state = checkpoint_model_state(best, anchor="best")
        last_state = checkpoint_model_state(last, anchor="last")
        initial_to_best = compare_state_dicts(
            initial_state,
            best_state,
            transition="initial_to_best",
        )
        best_to_last = compare_state_dicts(
            best_state,
            last_state,
            transition="best_to_last_ema",
        )
        tensor_drift = pd.concat(
            [initial_to_best.tensor_drift, best_to_last.tensor_drift],
            ignore_index=True,
        )
        tensor_drift.insert(0, "run_slot", str(run_slot))
        group_drift = pd.concat(
            [initial_to_best.group_drift, best_to_last.group_drift],
            ignore_index=True,
        )
        group_drift.insert(0, "run_slot", str(run_slot))
        optimizer_groups, optimizer_summary = checkpoint_optimizer_metadata(
            last.get("optimizer")
        )
        if len(optimizer_groups):
            optimizer_groups.insert(0, "run_slot", str(run_slot))
        best_candidates = best_epoch_candidates(best, epoch_rows)
        known = {str(key): str(value).upper() for key, value in (known_sha256 or {}).items()}
        run_summary: dict[str, Any] = {
            "run_slot": str(run_slot),
            "initial_checkpoint_sha256": known.get("initial") or sha256_file(initial_path),
            "best_checkpoint_sha256": known.get("best") or sha256_file(best_path),
            "last_checkpoint_sha256": known.get("last") or sha256_file(last_path),
            "initial_checkpoint_sha256_source": "provided_verified" if "initial" in known else "computed",
            "best_checkpoint_sha256_source": "provided_verified" if "best" in known else "computed",
            "last_checkpoint_sha256_source": "provided_verified" if "last" in known else "computed",
            "initial_checkpoint_version": initial.get("version"),
            "best_checkpoint_version": best.get("version"),
            "last_checkpoint_version": last.get("version"),
            "initial_state_tensor_count": len(initial_state),
            "best_state_tensor_count": len(best_state),
            "last_state_tensor_count": len(last_state),
            "initial_state_elements": _state_element_count(initial_state),
            "best_state_elements": _state_element_count(best_state),
            "last_state_elements": _state_element_count(last_state),
            "best_checkpoint_epoch": int(best.get("epoch", -1)),
            "last_checkpoint_epoch": int(last.get("epoch", -1)),
            "last_ema_updates": last.get("updates"),
            **{f"initial_to_best_{key}": value for key, value in initial_to_best.summary.items() if key != "transition"},
            **{f"best_to_last_{key}": value for key, value in best_to_last.summary.items() if key != "transition"},
            **optimizer_summary,
        }
        return CheckpointTripletResult(
            tensor_drift=tensor_drift,
            group_drift=group_drift,
            optimizer_groups=optimizer_groups,
            run_summary=run_summary,
            best_epoch_candidates={"run_slot": str(run_slot), **best_candidates},
        )
    finally:
        del initial, best, last
        gc.collect()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _canonical_attempt(extracted_root: Path, row: Mapping[str, Any]) -> Path:
    return (
        extracted_root
        / f"stage1_gapvalue240_{row['package']}_upload"
        / "runs"
        / str(row["run_slot"])
        / str(row["attempt_id"])
    )


def scan_inventory_checkpoints(
    *,
    extracted_root: str | Path,
    inventory_path: str | Path,
    initial_checkpoint: str | Path,
    epoch_curves_path: str | Path,
    output_dir: str | Path,
    expected_runs: int = 240,
    expected_epochs: int = 200,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Scan the canonical checkpoint triplet for every inventory row.

    The source attempts are never modified.  Five derived tables are published
    atomically below an existing analysis ``.inprogress`` directory.
    """

    root = Path(extracted_root).resolve()
    inventory_file = Path(inventory_path).resolve()
    initial_path = Path(initial_checkpoint).resolve()
    epoch_file = Path(epoch_curves_path).resolve()
    output = Path(output_dir).resolve()
    if not output.name.endswith(".inprogress"):
        raise ValueError("Checkpoint analysis output must remain .inprogress")
    inventory = pd.read_csv(inventory_file, keep_default_na=False)
    required = {"run_slot", "package", "attempt_id"}
    missing = required - set(inventory.columns)
    if missing:
        raise ValueError(f"Canonical inventory missing columns: {sorted(missing)}")
    if len(inventory) != expected_runs or inventory["run_slot"].nunique() != expected_runs:
        raise ValueError(
            f"Expected {expected_runs} unique canonical runs, found "
            f"{len(inventory)} rows/{inventory['run_slot'].nunique()} unique"
        )
    curves = pd.read_csv(epoch_file)
    counts = curves.groupby("run_slot", sort=False)["epoch"].nunique()
    if len(counts) != expected_runs or not counts.eq(expected_epochs).all():
        raise ValueError(
            f"Epoch grid does not prove {expected_runs}x{expected_epochs}: "
            f"runs={len(counts)}, bad={(~counts.eq(expected_epochs)).sum()}"
        )
    initial_sha = sha256_file(initial_path)

    tensor_frames: list[pd.DataFrame] = []
    group_frames: list[pd.DataFrame] = []
    optimizer_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(
        inventory.sort_values("run_slot").to_dict(orient="records"), start=1
    ):
        attempt = _canonical_attempt(root, row)
        if not attempt.is_dir():
            raise FileNotFoundError(attempt)
        best_path = attempt / "03_checkpoints/best.pt"
        last_path = attempt / "03_checkpoints/last.pt"
        result = scan_checkpoint_triplet(
            run_slot=str(row["run_slot"]),
            initial_checkpoint=initial_path,
            best_checkpoint=best_path,
            last_checkpoint=last_path,
            epoch_rows=curves.loc[curves["run_slot"].astype(str) == str(row["run_slot"])],
            known_sha256={"initial": initial_sha},
        )
        tensor_frames.append(result.tensor_drift)
        group_frames.append(result.group_drift)
        if len(result.optimizer_groups):
            optimizer_frames.append(result.optimizer_groups)
        identity = {
            key: row[key]
            for key in (
                "triad_id",
                "phase",
                "condition_id",
                "method",
                "budget",
                "guard_ratio",
                "arm",
                "training_seed",
                "selection_seed",
                "package",
                "machine_id",
                "attempt_id",
                "resume_count",
                "input_snapshot_id",
            )
            if key in row
        }
        summaries.append({**result.run_summary, **identity})
        candidates.append({**result.best_epoch_candidates, **identity})
        if progress is not None:
            progress(index, expected_runs, str(row["run_slot"]))

    tensor_frame = pd.concat(tensor_frames, ignore_index=True)
    group_frame = pd.concat(group_frames, ignore_index=True)
    optimizer_frame = (
        pd.concat(optimizer_frames, ignore_index=True)
        if optimizer_frames
        else pd.DataFrame()
    )
    summary_frame = pd.DataFrame(summaries).sort_values("run_slot", ignore_index=True)
    candidate_frame = pd.DataFrame(candidates).sort_values("run_slot", ignore_index=True)
    tables = output / "tables"
    _atomic_csv(tensor_frame, tables / "checkpoint_tensor_drift.csv")
    _atomic_csv(group_frame, tables / "checkpoint_block_drift.csv")
    _atomic_csv(optimizer_frame, tables / "checkpoint_optimizer_groups.csv")
    _atomic_csv(summary_frame, tables / "checkpoint_run_summary.csv")
    _atomic_csv(candidate_frame, tables / "checkpoint_best_epoch_candidates.csv")
    return {
        "canonical_runs": int(len(summary_frame)),
        "tensor_drift_rows": int(len(tensor_frame)),
        "block_drift_rows": int(len(group_frame)),
        "optimizer_group_rows": int(len(optimizer_frame)),
        "best_epoch_candidate_rows": int(len(candidate_frame)),
        "initial_checkpoint_sha256": initial_sha,
    }


def build_paired_checkpoint_deltas(
    group_drift: pd.DataFrame,
    run_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Pair every Treatment layer/transition against R1 and R2 separately."""

    required_metadata = {"run_slot", "triad_id", "arm"}
    missing_metadata = required_metadata - set(run_metadata.columns)
    if missing_metadata:
        raise ValueError(f"Run metadata missing columns: {sorted(missing_metadata)}")
    required_drift = {
        "run_slot",
        "transition",
        "group_name",
        "matched_float_tensors",
        "matched_elements",
        "left_l2",
        "right_l2",
        "delta_l2",
        "relative_l2",
        "cosine_similarity",
    }
    missing_drift = required_drift - set(group_drift.columns)
    if missing_drift:
        raise ValueError(f"Group drift missing columns: {sorted(missing_drift)}")
    if run_metadata["run_slot"].duplicated().any():
        raise ValueError("Run metadata contains duplicate run slots")
    metadata = run_metadata.copy()
    for triad_id, group in metadata.groupby("triad_id", sort=False):
        if sorted(group["arm"].astype(str).tolist()) != ["R1", "R2", "T"]:
            raise ValueError(f"Triad {triad_id} does not contain exactly T/R1/R2")
    identity_columns = [
        column
        for column in metadata.columns
        if column not in {"run_slot", "arm"}
    ]
    joined = group_drift.merge(
        metadata,
        on="run_slot",
        how="left",
        validate="many_to_one",
    )
    if joined["triad_id"].isna().any():
        raise ValueError("Checkpoint drift contains run slots absent from metadata")
    metric_columns = [
        "matched_float_tensors",
        "matched_elements",
        "left_l2",
        "right_l2",
        "delta_l2",
        "relative_l2",
        "cosine_similarity",
    ]
    keys = ["triad_id", "transition", "group_name"]
    treatment = joined.loc[joined["arm"].astype(str) == "T", keys + ["run_slot", *metric_columns]].rename(
        columns={"run_slot": "treatment_run_slot", **{column: f"treatment_{column}" for column in metric_columns}}
    )
    outputs: list[pd.DataFrame] = []
    for control_arm in ("R1", "R2"):
        control = joined.loc[
            joined["arm"].astype(str) == control_arm,
            keys + ["run_slot", *metric_columns],
        ].rename(
            columns={"run_slot": "control_run_slot", **{column: f"control_{column}" for column in metric_columns}}
        )
        paired = treatment.merge(control, on=keys, how="inner", validate="one_to_one")
        paired.insert(1, "control_arm", control_arm)
        for column in metric_columns:
            paired[f"delta_{column}"] = (
                paired[f"treatment_{column}"] - paired[f"control_{column}"]
            )
        outputs.append(paired)
    result = pd.concat(outputs, ignore_index=True)
    triad_identity = metadata.loc[metadata["arm"].astype(str) == "T", ["triad_id", *[column for column in identity_columns if column != "triad_id"]]]
    if len(triad_identity.columns) > 1:
        result = result.merge(
            triad_identity,
            on="triad_id",
            how="left",
            validate="many_to_one",
        )
    return result.sort_values(
        ["triad_id", "control_arm", "transition", "group_name"],
        ignore_index=True,
    )


def summarize_checkpoint_cohorts(
    paired: pd.DataFrame,
    triad_outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Describe layerwise T-control drift deltas within frozen outcome cohorts."""

    required_outcomes = {"triad_id", "exclusive_cohort"}
    missing = required_outcomes - set(triad_outcomes.columns)
    if missing:
        raise ValueError(f"Triad outcomes missing columns: {sorted(missing)}")
    if triad_outcomes["triad_id"].duplicated().any():
        raise ValueError("Triad outcomes contain duplicate triad IDs")
    joined = paired.merge(
        triad_outcomes,
        on="triad_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_outcome"),
    )
    if joined["exclusive_cohort"].isna().any():
        raise ValueError("Paired checkpoint rows are missing outcome cohorts")
    delta_columns = [
        column
        for column in joined.columns
        if column.startswith("delta_")
        and column not in {"delta_matched_float_tensors", "delta_matched_elements"}
    ]
    records: list[dict[str, Any]] = []
    keys = ["exclusive_cohort", "control_arm", "transition", "group_name"]
    for key_values, group in joined.groupby(keys, sort=True, dropna=False):
        identity = dict(zip(keys, key_values))
        for column in delta_columns:
            values = pd.to_numeric(group[column], errors="raise")
            records.append(
                {
                    **identity,
                    "metric": column,
                    "triad_count": int(group["triad_id"].nunique()),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(records)


def publish_checkpoint_pair_analysis(
    *,
    output_dir: str | Path,
    inventory_path: str | Path,
    triad_outcomes_path: str | Path,
    expected_triads: int = 80,
) -> dict[str, Any]:
    """Publish paired checkpoint-drift and frozen-cohort summary tables."""

    output = Path(output_dir).resolve()
    if not output.name.endswith(".inprogress"):
        raise ValueError("Checkpoint pair output must remain .inprogress")
    block_path = output / "tables/checkpoint_block_drift.csv"
    if not block_path.is_file():
        raise FileNotFoundError(block_path)
    inventory = pd.read_csv(Path(inventory_path).resolve(), keep_default_na=False)
    outcomes = pd.read_csv(Path(triad_outcomes_path).resolve(), keep_default_na=False)
    if inventory["triad_id"].nunique() != expected_triads:
        raise ValueError(
            f"Expected {expected_triads} inventory triads, found {inventory['triad_id'].nunique()}"
        )
    if outcomes["triad_id"].nunique() != expected_triads:
        raise ValueError(
            f"Expected {expected_triads} outcome triads, found {outcomes['triad_id'].nunique()}"
        )
    block_drift = pd.read_csv(block_path)
    paired = build_paired_checkpoint_deltas(block_drift, inventory)
    expected_pairs = expected_triads * 2
    structural_cells = block_drift[["transition", "group_name"]].drop_duplicates()
    expected_rows = expected_pairs * len(structural_cells)
    if len(paired) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} checkpoint pair rows, found {len(paired)}"
        )
    paired = paired.merge(
        outcomes,
        on="triad_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_outcome"),
    )
    summary = summarize_checkpoint_cohorts(
        paired.drop(columns=[column for column in paired.columns if column.endswith("_outcome")]),
        outcomes,
    )
    _atomic_csv(paired, output / "tables/checkpoint_paired_deltas.csv")
    _atomic_csv(summary, output / "tables/checkpoint_cohort_summary.csv")
    return {
        "triads": expected_triads,
        "paired_rows": int(len(paired)),
        "cohort_summary_rows": int(len(summary)),
        "structural_cells": int(len(structural_cells)),
    }
