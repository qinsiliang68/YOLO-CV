"""Exact sample and label identity checks for prediction artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable


class PredictionIdentityError(ValueError):
    pass


def _canonical(rows: Iterable[tuple[str, int]]) -> tuple[tuple[str, int], ...]:
    values = tuple((str(sample_id).replace("\\", "/"), int(label)) for sample_id, label in rows)
    identifiers = [sample_id for sample_id, _ in values]
    if len(identifiers) != len(set(identifiers)):
        raise PredictionIdentityError("prediction identity contains duplicate sample IDs")
    return tuple(sorted(values))


def _digest(rows: tuple[tuple[str, int], ...]) -> str:
    hasher = hashlib.sha256()
    for sample_id, label in rows:
        hasher.update(f"{sample_id}|{label}\n".encode("utf-8"))
    return hasher.hexdigest().upper()


def canonical_sample_label_digest(rows: Iterable[tuple[str, int]]) -> str:
    """Return the stable digest used to bind manifests and predictions."""

    return _digest(_canonical(rows))


@dataclass(frozen=True)
class PredictionIdentityReport:
    status: str
    row_count: int
    expected_digest: str
    observed_digest: str


def validate_prediction_identity(
    expected: Iterable[tuple[str, int]], observed: Iterable[tuple[str, int]]
) -> PredictionIdentityReport:
    expected_rows = _canonical(expected)
    observed_rows = _canonical(observed)
    expected_digest = _digest(expected_rows)
    observed_digest = _digest(observed_rows)
    if expected_rows != observed_rows:
        expected_ids = {sample_id for sample_id, _ in expected_rows}
        observed_ids = {sample_id for sample_id, _ in observed_rows}
        raise PredictionIdentityError(
            "prediction identity differs from the manifest: "
            f"missing={len(expected_ids - observed_ids)}, extra={len(observed_ids - expected_ids)}"
        )
    return PredictionIdentityReport("PASS", len(expected_rows), expected_digest, observed_digest)


__all__ = [
    "PredictionIdentityError",
    "PredictionIdentityReport",
    "canonical_sample_label_digest",
    "validate_prediction_identity",
]
