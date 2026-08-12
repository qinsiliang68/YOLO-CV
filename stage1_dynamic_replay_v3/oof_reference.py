"""Build the immutable epoch-200 OOF reference used by Stage1 v3."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import uuid
from typing import Any, Iterable

import polars as pl

from .prediction_identity import canonical_sample_label_digest


OOF_REFERENCE_SCHEMA = "stage1.dynamic_oof_replay.oof_reference.v3"
_PREDICTION_COLUMNS = {"sample_id", "y_true", "oof_fold", "epoch", "p_defect_raw"}
_PROBABILITY_EPSILON = 1e-12


class OOFReferenceError(ValueError):
    """Raised when the OOF files cannot be bound to the training manifests."""


def binary_cross_entropy(y_true: int, p_defect: float) -> float:
    """Compute the evidence-table CE without importing the retired v3 controller."""

    label = int(y_true)
    if label not in (0, 1):
        raise ValueError(f"binary label must be 0 or 1: {y_true!r}")
    probability = float(p_defect)
    if not math.isfinite(probability):
        raise ValueError(f"probability must be finite: {p_defect!r}")
    probability = min(
        1.0 - _PROBABILITY_EPSILON,
        max(_PROBABILITY_EPSILON, probability),
    )
    return -math.log(probability if label == 1 else 1.0 - probability)


@dataclass(frozen=True)
class OOFReferenceResult:
    status: str
    output_path: Path
    sidecar_path: Path
    row_count: int
    defect_count: int
    normal_count: int
    fold_count: int
    sample_label_digest: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_manifest(path: Path, *, expected_count: int) -> dict[str, int]:
    if not path.is_file():
        if expected_count == 0:
            return {}
        raise OOFReferenceError(f"training manifest is missing: {path}")
    rows: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"canonical_image_relpath", "Defect"}
        if not required.issubset(reader.fieldnames or ()):
            raise OOFReferenceError(f"training manifest columns are invalid: {path}")
        for row in reader:
            sample_id = str(row["canonical_image_relpath"]).strip().replace("\\", "/")
            if not sample_id or sample_id in rows:
                raise OOFReferenceError(f"training manifest identity is empty or duplicated: {sample_id!r}")
            try:
                label = int(row["Defect"])
            except (TypeError, ValueError) as exc:
                raise OOFReferenceError(f"training manifest label is invalid for {sample_id}") from exc
            if label not in (0, 1):
                raise OOFReferenceError(f"training manifest label is invalid for {sample_id}: {label}")
            rows[sample_id] = label
    if len(rows) != expected_count:
        raise OOFReferenceError(
            f"training manifest identity count mismatch: expected {expected_count}, observed {len(rows)}"
        )
    return rows


def _prediction_files(jobs_root: Path) -> list[Path]:
    if not jobs_root.is_dir():
        raise OOFReferenceError(f"OOF jobs root is missing: {jobs_root}")
    files: list[Path] = []
    for root, _, names in os.walk(jobs_root):
        if "epoch_200_predictions.csv" in names:
            files.append(Path(root) / "epoch_200_predictions.csv")
    files.sort(key=lambda path: path.as_posix())
    if not files:
        raise OOFReferenceError(f"no epoch-200 OOF prediction files found under {jobs_root}")
    return files


def _read_predictions(files: Iterable[Path], jobs_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in files:
        file_digest = _sha256(path)
        relative = path.relative_to(jobs_root).as_posix()
        source_count = 0
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not _PREDICTION_COLUMNS.issubset(reader.fieldnames or ()):
                raise OOFReferenceError(f"OOF prediction columns are invalid: {path}")
            for raw in reader:
                sample_id = str(raw["sample_id"]).strip().replace("\\", "/")
                try:
                    label = int(raw["y_true"])
                    epoch = int(raw["epoch"])
                    probability = float(raw["p_defect_raw"])
                except (TypeError, ValueError) as exc:
                    raise OOFReferenceError(f"OOF prediction value is invalid in {path}") from exc
                fold = str(raw["oof_fold"]).strip()
                if not sample_id or label not in (0, 1) or epoch != 200 or not fold:
                    raise OOFReferenceError(f"OOF prediction identity or epoch is invalid in {path}")
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise OOFReferenceError(f"OOF probability is invalid for {sample_id}: {probability}")
                rows.append(
                    {
                        "sample_id": sample_id,
                        "y_true": label,
                        "oof_fold": fold,
                        "p_defect": probability,
                        "oof_ce": binary_cross_entropy(label, probability),
                        "source_prediction_relpath": relative,
                        "source_prediction_sha256": file_digest,
                    }
                )
                source_count += 1
        sources.append({"relative_path": relative, "sha256": file_digest, "row_count": source_count})
    return rows, sources


def build_oof_epoch200_reference(
    *,
    jobs_root: str | Path,
    defect_manifest: str | Path,
    normal_manifest: str | Path,
    output_path: str | Path,
    expected_rows: int = 120_000,
    expected_defect: int = 60_000,
    expected_normal: int = 60_000,
    expected_fold_count: int = 10,
) -> OOFReferenceResult:
    """Validate all OOF rows against manifests and atomically publish a reference table."""

    jobs = Path(jobs_root).resolve()
    defect_path = Path(defect_manifest).resolve()
    normal_path = Path(normal_manifest).resolve()
    destination = Path(output_path).resolve()
    if expected_rows != expected_defect + expected_normal:
        raise OOFReferenceError("expected row count does not equal defect plus normal counts")

    identity: dict[str, int] = {}
    for sample_id, label in _read_manifest(defect_path, expected_count=expected_defect).items():
        if label != 1:
            raise OOFReferenceError(f"defect manifest has a normal label: {sample_id}")
        identity[sample_id] = label
    for sample_id, label in _read_manifest(normal_path, expected_count=expected_normal).items():
        if label != 0:
            raise OOFReferenceError(f"normal manifest has a defect label: {sample_id}")
        if sample_id in identity:
            raise OOFReferenceError(f"training manifest identity is duplicated across classes: {sample_id}")
        identity[sample_id] = label

    prediction_files = _prediction_files(jobs)
    rows, sources = _read_predictions(prediction_files, jobs)
    observed_ids = [str(row["sample_id"]) for row in rows]
    if len(rows) != expected_rows or len(observed_ids) != len(set(observed_ids)):
        raise OOFReferenceError(
            f"OOF identity row count or uniqueness mismatch: expected {expected_rows}, observed {len(rows)}"
        )
    observed_identity = {str(row["sample_id"]): int(row["y_true"]) for row in rows}
    if observed_identity != identity:
        missing = sorted(set(identity) - set(observed_identity))[:5]
        extra = sorted(set(observed_identity) - set(identity))[:5]
        wrong_labels = sorted(
            sample_id
            for sample_id in set(identity) & set(observed_identity)
            if identity[sample_id] != observed_identity[sample_id]
        )[:5]
        raise OOFReferenceError(
            f"OOF identity mismatch: missing={missing}, extra={extra}, wrong_labels={wrong_labels}"
        )
    folds = {str(row["oof_fold"]) for row in rows}
    if len(folds) != expected_fold_count:
        raise OOFReferenceError(
            f"OOF fold count mismatch: expected {expected_fold_count}, observed {len(folds)}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        pl.DataFrame(rows).sort("sample_id").write_parquet(temporary, compression="zstd")
        # Windows rejects fsync on a read-only descriptor.
        with temporary.open("rb+") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    digest = canonical_sample_label_digest(identity.items())
    sidecar = destination.with_suffix(destination.suffix + ".manifest.json")
    payload = {
        "schema_version": OOF_REFERENCE_SCHEMA,
        "status": "PASS",
        "output_path": destination.name,
        "output_sha256": _sha256(destination),
        "row_count": len(rows),
        "defect_count": sum(int(row["y_true"]) == 1 for row in rows),
        "normal_count": sum(int(row["y_true"]) == 0 for row in rows),
        "fold_count": len(folds),
        "folds": sorted(folds),
        "sample_label_digest": digest,
        "defect_manifest_sha256": _sha256(defect_path),
        "normal_manifest_sha256": _sha256(normal_path) if normal_path.is_file() else None,
        "source_files": sources,
    }
    _atomic_bytes(
        sidecar,
        (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    return OOFReferenceResult(
        status="PASS",
        output_path=destination,
        sidecar_path=sidecar,
        row_count=len(rows),
        defect_count=payload["defect_count"],
        normal_count=payload["normal_count"],
        fold_count=len(folds),
        sample_label_digest=digest,
    )


__all__ = [
    "OOF_REFERENCE_SCHEMA",
    "OOFReferenceError",
    "OOFReferenceResult",
    "build_oof_epoch200_reference",
]
