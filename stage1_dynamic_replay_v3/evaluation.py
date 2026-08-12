"""Tie-safe raw safety frontier metrics and identity-bound prediction artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import uuid
from typing import Any, Iterable, Sequence

from .asset_identity import ManifestImageRecord
from .prediction_identity import (
    PredictionIdentityError,
    canonical_sample_label_digest,
    validate_prediction_identity,
)


PREDICTION_ARTIFACT_SCHEMA = "stage1.dynamic_oof_replay.val_op_predictions.v3"


class PredictionArtifactError(ValueError):
    """Raised when predictions or endpoint identities are incomplete or inconsistent."""


@dataclass(frozen=True)
class PredictionRow:
    sample_id: str
    y_true: int
    p_defect_raw: float
    checkpoint_epoch: int


@dataclass(frozen=True)
class FrontierPoint:
    fn_budget: int
    actual_fn: int
    tn: int
    threshold: float
    tie_group_size: int


@dataclass(frozen=True)
class RawSafetyFrontierMetrics:
    status: str
    frontier: tuple[FrontierPoint, ...]
    normalized_auc: float
    tn_at_fn_max: int
    fn_at_target_tn: int | None
    target_tn_reachable: bool
    defect_count: int
    normal_count: int


@dataclass(frozen=True)
class PredictionArtifact:
    status: str
    output_path: Path
    sidecar_path: Path
    row_count: int


@dataclass(frozen=True)
class PredictionArtifactValidation:
    status: str
    row_count: int
    sample_label_digest: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise PredictionArtifactError(f"prediction artifact already exists: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalized_rows(rows: Iterable[PredictionRow]) -> tuple[PredictionRow, ...]:
    normalized: list[PredictionRow] = []
    seen: set[str] = set()
    epochs: set[int] = set()
    for raw in rows:
        sample_id = str(raw.sample_id).strip().replace("\\", "/")
        label = int(raw.y_true)
        probability = float(raw.p_defect_raw)
        epoch = int(raw.checkpoint_epoch)
        if not sample_id or sample_id in seen:
            raise PredictionArtifactError(f"prediction sample identity is empty or duplicated: {sample_id!r}")
        if label not in (0, 1) or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise PredictionArtifactError(f"prediction value is invalid for {sample_id}")
        seen.add(sample_id)
        epochs.add(epoch)
        normalized.append(PredictionRow(sample_id, label, probability, epoch))
    if not normalized or len(epochs) != 1:
        raise PredictionArtifactError("prediction artifact must contain one non-empty checkpoint epoch")
    return tuple(normalized)


def compute_raw_safety_frontier(
    rows: Iterable[PredictionRow], *, max_fn: int = 95, target_tn: int = 68_253
) -> RawSafetyFrontierMetrics:
    predictions = _normalized_rows(rows)
    defect_count = sum(row.y_true == 1 for row in predictions)
    normal_count = sum(row.y_true == 0 for row in predictions)
    if defect_count <= max_fn or normal_count <= 0:
        raise PredictionArtifactError(
            f"frontier class counts are invalid for FN=0..{max_fn}: defect={defect_count}, normal={normal_count}"
        )
    groups: dict[float, list[PredictionRow]] = {}
    for row in predictions:
        groups.setdefault(row.p_defect_raw, []).append(row)
    true_positive = false_positive = 0
    candidates: list[tuple[int, int, float, int]] = []
    for threshold in sorted(groups, reverse=True):
        group = groups[threshold]
        true_positive += sum(row.y_true == 1 for row in group)
        false_positive += sum(row.y_true == 0 for row in group)
        false_negative = defect_count - true_positive
        true_negative = normal_count - false_positive
        candidates.append((false_negative, true_negative, threshold, len(group)))

    frontier: list[FrontierPoint] = []
    for budget in range(int(max_fn) + 1):
        eligible = [candidate for candidate in candidates if candidate[0] <= budget]
        if not eligible:
            raise PredictionArtifactError(f"raw frontier has no feasible threshold for FN budget {budget}")
        actual_fn, tn, threshold, tie_size = max(
            eligible,
            key=lambda item: (item[1], item[2], -item[0]),
        )
        frontier.append(FrontierPoint(budget, actual_fn, tn, threshold, tie_size))
    if max_fn == 0:
        normalized_auc = frontier[0].tn / normal_count
    else:
        ordinates = [point.tn / normal_count for point in frontier]
        normalized_auc = sum(
            (ordinates[index] + ordinates[index + 1]) * 0.5 for index in range(len(ordinates) - 1)
        ) / max_fn
    target_candidates = [candidate for candidate in candidates if candidate[1] >= int(target_tn)]
    fn_at_target = min(candidate[0] for candidate in target_candidates) if target_candidates else None
    return RawSafetyFrontierMetrics(
        status="PASS",
        frontier=tuple(frontier),
        normalized_auc=float(normalized_auc),
        tn_at_fn_max=frontier[-1].tn,
        fn_at_target_tn=fn_at_target,
        target_tn_reachable=fn_at_target is not None,
        defect_count=defect_count,
        normal_count=normal_count,
    )


def publish_prediction_artifact(
    *,
    rows: Iterable[PredictionRow],
    expected_records: Sequence[ManifestImageRecord],
    defect_manifest: str | Path,
    normal_manifest: str | Path,
    checkpoint: str | Path,
    output_path: str | Path,
) -> PredictionArtifact:
    predictions = _normalized_rows(rows)
    expected_identity = tuple((record.sample_id, record.y_true) for record in expected_records)
    observed_identity = tuple((row.sample_id, row.y_true) for row in predictions)
    try:
        identity = validate_prediction_identity(expected_identity, observed_identity)
    except PredictionIdentityError as exc:
        raise PredictionArtifactError(f"prediction identity mismatch: {exc}") from exc
    defect_path = Path(defect_manifest).resolve()
    normal_path = Path(normal_manifest).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    for name, path in (
        ("defect manifest", defect_path),
        ("normal manifest", normal_path),
        ("checkpoint", checkpoint_path),
    ):
        if not path.is_file():
            raise PredictionArtifactError(f"{name} is missing: {path}")
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["sample_id", "y_true", "p_defect_raw", "checkpoint_epoch"],
            )
            writer.writeheader()
            for row in sorted(predictions, key=lambda item: item.sample_id):
                writer.writerow(
                    {
                        "sample_id": row.sample_id,
                        "y_true": row.y_true,
                        "p_defect_raw": format(row.p_defect_raw, ".17g"),
                        "checkpoint_epoch": row.checkpoint_epoch,
                    }
                )
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists():
            raise PredictionArtifactError(f"prediction artifact already exists: {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    sidecar = destination.with_suffix(destination.suffix + ".manifest.json")
    payload = {
        "schema_version": PREDICTION_ARTIFACT_SCHEMA,
        "status": "COMPLETE",
        "row_count": len(predictions),
        "checkpoint_epoch": predictions[0].checkpoint_epoch,
        "prediction_sha256": _sha256(destination),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "defect_manifest_sha256": _sha256(defect_path),
        "normal_manifest_sha256": _sha256(normal_path),
        "expected_sample_label_digest": identity.expected_digest,
        "observed_sample_label_digest": identity.observed_digest,
    }
    _atomic_bytes(
        sidecar,
        (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    return PredictionArtifact("PASS", destination, sidecar, len(predictions))


def validate_prediction_artifact(
    output_path: str | Path,
    sidecar_path: str | Path,
    *,
    expected_records: Sequence[ManifestImageRecord],
) -> PredictionArtifactValidation:
    output = Path(output_path).resolve()
    sidecar = Path(sidecar_path).resolve()
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictionArtifactError(f"prediction sidecar is unreadable: {sidecar}") from exc
    if payload.get("schema_version") != PREDICTION_ARTIFACT_SCHEMA or payload.get("status") != "COMPLETE":
        raise PredictionArtifactError("prediction sidecar schema or status is invalid")
    if not output.is_file() or _sha256(output) != str(payload.get("prediction_sha256", "")).upper():
        raise PredictionArtifactError("prediction CSV sha256 differs from sidecar")
    rows: list[PredictionRow] = []
    with output.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", "y_true", "p_defect_raw", "checkpoint_epoch"}
        if not required.issubset(reader.fieldnames or ()):
            raise PredictionArtifactError("prediction CSV columns are invalid")
        for row in reader:
            rows.append(
                PredictionRow(
                    str(row["sample_id"]),
                    int(row["y_true"]),
                    float(row["p_defect_raw"]),
                    int(row["checkpoint_epoch"]),
                )
            )
    predictions = _normalized_rows(rows)
    if len(predictions) != int(payload.get("row_count", -1)):
        raise PredictionArtifactError("prediction row count differs from sidecar")
    expected_identity = tuple((record.sample_id, record.y_true) for record in expected_records)
    observed_identity = tuple((row.sample_id, row.y_true) for row in predictions)
    try:
        identity = validate_prediction_identity(expected_identity, observed_identity)
    except PredictionIdentityError as exc:
        raise PredictionArtifactError(f"prediction identity mismatch: {exc}") from exc
    expected_digest = canonical_sample_label_digest(expected_identity)
    if expected_digest != str(payload.get("expected_sample_label_digest", "")).upper():
        raise PredictionArtifactError("prediction expected identity digest differs from sidecar")
    if identity.observed_digest != str(payload.get("observed_sample_label_digest", "")).upper():
        raise PredictionArtifactError("prediction observed identity digest differs from sidecar")
    return PredictionArtifactValidation("PASS", len(predictions), identity.observed_digest)


__all__ = [
    "FrontierPoint",
    "PREDICTION_ARTIFACT_SCHEMA",
    "PredictionArtifact",
    "PredictionArtifactError",
    "PredictionArtifactValidation",
    "PredictionRow",
    "RawSafetyFrontierMetrics",
    "compute_raw_safety_frontier",
    "publish_prediction_artifact",
    "validate_prediction_artifact",
]
