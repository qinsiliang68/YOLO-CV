from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from .checkpointing import load_checkpoint
from .columnar import ColumnarManifest, read_columnar, validate_columnar_file, write_zstd_parquet
from .errors import ErrorCode, SctsrError
from .ledger_schema import PREDICTION_SCHEMA
from .serialization import load_json, sha256_file, stable_digest


_HEX = frozenset("0123456789ABCDEF")
_FORMAL_SPLIT_ROLE = "val_op"
_TRAJECTORY_SPLIT_ROLE = "val_model"
_TEST_ROLES = {"test", "blind_test", "blind_holdout", "holdout_test"}
_ALLOWED_MODEL_VARIANTS = {"MODEL", "EMA", "MODEL_SYNTHETIC_CANARY"}
_ALLOWED_EVALUATION_MODES = {"formal", "trajectory", "synthetic"}
_TRAJECTORY_EPOCHS = {120, 140, 150, 160, 180}


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and value == value.upper() and all(char in _HEX for char in value)


@dataclass(frozen=True, slots=True)
class PredictionRow:
    run_id: str
    arm_id: str
    training_seed: int
    split_role: str
    split_manifest_path: str
    split_manifest_sha256: str
    sample_id: str
    y_true: int
    logit_normal: float
    logit_defect: float
    p_defect_raw: float
    checkpoint_epoch: int
    checkpoint_sha256: str
    model_variant: str
    source_tree_digest: str
    prediction_generation: int


@dataclass(frozen=True, slots=True)
class PredictionArtifactBinding:
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_epoch: int
    model_variant: str
    source_tree_digest: str
    training_seed: int
    split_role: str
    split_manifest_path: str
    split_manifest_sha256: str
    evaluation_mode: str
    selection_semantic: str

    def validate(self) -> Mapping[str, Any]:
        if self.evaluation_mode not in _ALLOWED_EVALUATION_MODES:
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Unknown prediction evaluation mode")
        if self.split_role.lower() in _TEST_ROLES:
            raise SctsrError(ErrorCode.TEST_ACCESS_FORBIDDEN, "SCTSR v4 prediction generation may not access test or blind holdout data")
        if not _is_sha256(self.checkpoint_sha256) or not _is_sha256(self.source_tree_digest) or not _is_sha256(self.split_manifest_sha256):
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction binding contains a non-canonical SHA-256")
        if self.model_variant not in _ALLOWED_MODEL_VARIANTS:
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction model/EMA selection is unregistered", observed=self.model_variant)
        if self.evaluation_mode == "formal":
            if self.checkpoint_epoch != 200:
                raise SctsrError(ErrorCode.CHECKPOINT_EPOCH_FORBIDDEN, "Formal prediction endpoint is fixed at E200")
            if self.split_role != _FORMAL_SPLIT_ROLE:
                raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Formal endpoint is restricted to the registered val_op split")
            if self.selection_semantic != "ENDPOINT_ONLY_NOT_FOR_SELECTION":
                raise SctsrError(ErrorCode.VAL_OP_SELECTION_FORBIDDEN, "val_op may only evaluate a frozen endpoint")
        elif self.evaluation_mode == "trajectory":
            if self.checkpoint_epoch not in _TRAJECTORY_EPOCHS:
                raise SctsrError(ErrorCode.CHECKPOINT_EPOCH_FORBIDDEN, "Trajectory prediction uses an unregistered checkpoint epoch")
            if self.split_role != _TRAJECTORY_SPLIT_ROLE:
                raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Trajectory diagnostics are restricted to val_model")
            if self.selection_semantic != "NOT_FOR_METHOD_SELECTION":
                raise SctsrError(ErrorCode.VAL_OP_SELECTION_FORBIDDEN, "Trajectory predictions must be marked not for method selection")
        else:
            if self.split_role != "synthetic" or self.selection_semantic != "SYNTHETIC_NOT_SCIENTIFIC_RESULT":
                raise SctsrError(ErrorCode.SYNTHETIC_RESULT_MISLABELLED, "Synthetic predictions require the synthetic-only semantic")

        checkpoint = Path(self.checkpoint_path)
        if checkpoint.name.lower() == "best.pt" or "best.pt" in checkpoint.name.lower():
            raise SctsrError(ErrorCode.BEST_PT_FORBIDDEN, "best.pt is forbidden for prediction generation")
        if not checkpoint.is_file():
            raise SctsrError(ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, "Registered prediction checkpoint is missing", artifact_path=str(checkpoint))
        payload = load_checkpoint(checkpoint, expected_sha256=self.checkpoint_sha256, expected_epoch=self.checkpoint_epoch)
        if str(payload["source_tree_digest"]).upper() != self.source_tree_digest:
            raise SctsrError(ErrorCode.SOURCE_TREE_MISMATCH, "Prediction checkpoint source tree does not match the binding")
        if int(payload["training_seed"]) != self.training_seed:
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction checkpoint training seed does not match the binding")
        if self.model_variant == "EMA" and not payload.get("ema_state"):
            raise SctsrError(ErrorCode.PARENT_CHECKPOINT_INCOMPLETE, "EMA prediction requested from a checkpoint without EMA state")
        return payload


def sample_label_identity_digest(rows: Sequence[PredictionRow]) -> str:
    return stable_digest(sorted((row.sample_id, row.y_true) for row in rows))


def _constant_identity(rows: Sequence[PredictionRow]) -> dict[str, Any]:
    fields = (
        "run_id", "arm_id", "training_seed", "split_role", "split_manifest_path",
        "split_manifest_sha256", "checkpoint_epoch", "checkpoint_sha256", "model_variant",
        "source_tree_digest", "prediction_generation",
    )
    identity: dict[str, Any] = {}
    for field in fields:
        values = {getattr(row, field) for row in rows}
        if len(values) != 1:
            raise SctsrError(
                ErrorCode.PREDICTION_IDENTITY_MISMATCH,
                "A prediction artifact mixed multiple registered identities",
                failing_field=field,
                observed=sorted(map(str, values)),
            )
        identity[field] = next(iter(values))
    return identity


def validate_prediction_rows(
    rows: Sequence[PredictionRow],
    *,
    expected_split_role: str | None = None,
    expected_sample_labels: Mapping[str, int] | None = None,
) -> None:
    if not rows:
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction artifact is empty")
    identity = _constant_identity(rows)
    if str(identity["split_role"]).lower() in _TEST_ROLES:
        raise SctsrError(ErrorCode.TEST_ACCESS_FORBIDDEN, "Test or blind holdout prediction access is forbidden")
    if expected_split_role is not None and identity["split_role"] != expected_split_role:
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction split role mismatch")
    for field in ("split_manifest_sha256", "checkpoint_sha256", "source_tree_digest"):
        if not _is_sha256(str(identity[field])):
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction identity contains a non-canonical SHA-256", failing_field=field)
    if not str(identity["run_id"]).strip() or not str(identity["arm_id"]).strip() or not str(identity["split_manifest_path"]).strip():
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction run, arm, and split manifest path must be registered")
    if identity["model_variant"] not in _ALLOWED_MODEL_VARIANTS:
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction model/EMA selection is invalid")
    if int(identity["prediction_generation"]) < 1:
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction generation must be positive")
    if not 1 <= int(identity["checkpoint_epoch"]) <= 200:
        raise SctsrError(ErrorCode.CHECKPOINT_EPOCH_FORBIDDEN, "Prediction checkpoint epoch is invalid")

    ids = [row.sample_id for row in rows]
    if any(not sample_id.strip() for sample_id in ids) or len(ids) != len(set(ids)):
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction artifact has empty or duplicate sample IDs")
    observed_labels: dict[str, int] = {}
    for row in rows:
        if row.y_true not in {0, 1}:
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction y_true must be binary")
        values = (float(row.logit_normal), float(row.logit_defect), float(row.p_defect_raw))
        if not all(math.isfinite(value) for value in values) or not 0.0 <= values[2] <= 1.0:
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction logits/probability are not finite or in range")
        maximum = max(values[0], values[1])
        expected_probability = math.exp(values[1] - maximum) / (
            math.exp(values[0] - maximum) + math.exp(values[1] - maximum)
        )
        if not math.isclose(values[2], expected_probability, rel_tol=1e-6, abs_tol=1e-7):
            raise SctsrError(
                ErrorCode.PREDICTION_IDENTITY_MISMATCH,
                "Raw defect probability does not match the registered two-class logits",
                observed=values[2],
                expected=expected_probability,
            )
        observed_labels[row.sample_id] = row.y_true
    if expected_sample_labels is not None:
        expected = {str(sample_id): int(label) for sample_id, label in expected_sample_labels.items()}
        missing = sorted(set(expected) - set(observed_labels))
        extra = sorted(set(observed_labels) - set(expected))
        wrong = sorted(sample_id for sample_id in set(expected) & set(observed_labels) if expected[sample_id] != observed_labels[sample_id])
        if missing or extra or wrong:
            raise SctsrError(
                ErrorCode.PREDICTION_IDENTITY_MISMATCH,
                "Prediction rows are not a complete one-to-one rendering of the registered split",
                observed={"missing": missing, "extra": extra, "wrong_label": wrong},
            )


def _manifest_rows(payload: Any) -> Sequence[Mapping[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        for key in ("rows", "samples", "records", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Split manifest has no registered row collection")


def load_split_sample_labels(path: str | Path) -> dict[str, int]:
    source = Path(path)
    if not source.is_file():
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Split manifest is missing", artifact_path=str(source))
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            rows: Sequence[Mapping[str, Any]] = list(csv.DictReader(handle))
    elif source.suffix.lower() in {".json", ".jsonl"}:
        if source.suffix.lower() == ".jsonl":
            with source.open("r", encoding="utf-8-sig") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
        else:
            with source.open("r", encoding="utf-8-sig") as handle:
                rows = _manifest_rows(json.load(handle))
    else:
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Unsupported split manifest format", artifact_path=str(source))
    output: dict[str, int] = {}
    for index, row in enumerate(rows):
        sample_id = row.get("sample_id", row.get("canonical_id", row.get("id")))
        label = row.get("y_true", row.get("label"))
        if sample_id is None or label is None:
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Split row lacks sample_id/y_true", failing_field=f"row[{index}]")
        key = str(sample_id)
        if key in output:
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Split manifest contains duplicate sample IDs", observed=key)
        try:
            parsed_label = int(label)
        except (TypeError, ValueError) as exc:
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Split label is not an integer", observed=label) from exc
        if parsed_label not in {0, 1}:
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Split labels must be binary", observed=parsed_label)
        output[key] = parsed_label
    if not output:
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Split manifest is empty")
    return output


def validate_prediction_binding(
    rows: Sequence[PredictionRow],
    binding: PredictionArtifactBinding,
    *,
    expected_sample_labels: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    payload = binding.validate()
    if binding.evaluation_mode != "synthetic":
        split_path = Path(binding.split_manifest_path)
        observed_manifest_sha = sha256_file(split_path)
        if observed_manifest_sha != binding.split_manifest_sha256:
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Split manifest SHA does not match the binding", observed=observed_manifest_sha, expected=binding.split_manifest_sha256)
        loaded_labels = load_split_sample_labels(split_path)
        if expected_sample_labels is not None and dict(expected_sample_labels) != loaded_labels:
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Caller split identity disagrees with the SHA-bound manifest")
        expected_sample_labels = loaded_labels
    elif expected_sample_labels is None:
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Synthetic prediction binding still requires an explicit split identity")
    validate_prediction_rows(rows, expected_split_role=binding.split_role, expected_sample_labels=expected_sample_labels)
    identity = _constant_identity(rows)
    expected_identity = {
        "training_seed": binding.training_seed,
        "split_role": binding.split_role,
        "split_manifest_path": binding.split_manifest_path,
        "split_manifest_sha256": binding.split_manifest_sha256,
        "checkpoint_epoch": binding.checkpoint_epoch,
        "checkpoint_sha256": binding.checkpoint_sha256,
        "model_variant": binding.model_variant,
        "source_tree_digest": binding.source_tree_digest,
    }
    mismatches = {field: {"observed": identity[field], "expected": value} for field, value in expected_identity.items() if identity[field] != value}
    if mismatches:
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction rows disagree with their checkpoint/split binding", observed=mismatches)
    return dict(payload)


def write_prediction_artifact(
    rows: Sequence[PredictionRow],
    path: str | Path,
    *,
    formal_endpoint: bool,
    binding: PredictionArtifactBinding | None = None,
    expected_sample_labels: Mapping[str, int] | None = None,
) -> tuple[ColumnarManifest, dict[str, object]]:
    validate_prediction_rows(rows, expected_sample_labels=expected_sample_labels)
    if formal_endpoint and binding is None:
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Formal prediction publication requires a loadable checkpoint and SHA-bound split binding")
    if binding is not None:
        validate_prediction_binding(rows, binding, expected_sample_labels=expected_sample_labels)
        if formal_endpoint and binding.evaluation_mode != "formal":
            raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Formal endpoint publication requires formal evaluation mode")
    elif formal_endpoint:
        raise AssertionError("unreachable")
    identity = _constant_identity(rows)
    digest = sample_label_identity_digest(rows)
    row_count = len(rows)
    materialized = [
        {
            **asdict(row),
            "sample_label_identity_digest": digest,
            "artifact_row_count": row_count,
        }
        for row in rows
    ]
    manifest = write_zstd_parquet(
        materialized,
        path,
        schema_version="stage1.sctsr.prediction_artifact.v2",
        schema=PREDICTION_SCHEMA,
        require_run_epoch_partition=True,
    )
    summary: dict[str, object] = {
        "schema_version": "stage1.sctsr.prediction_artifact_summary.v2",
        "row_count": row_count,
        "sample_label_identity_digest": digest,
        "run_id": identity["run_id"],
        "arm_id": identity["arm_id"],
        "training_seed": identity["training_seed"],
        "split_role": identity["split_role"],
        "split_manifest_path": identity["split_manifest_path"],
        "split_manifest_sha256": identity["split_manifest_sha256"],
        "checkpoint_epoch": identity["checkpoint_epoch"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "model_variant": identity["model_variant"],
        "source_tree_digest": identity["source_tree_digest"],
        "prediction_generation": identity["prediction_generation"],
        "prediction_artifact_sha256": manifest.sha256,
        "canonical_parquet": manifest.canonical_parquet,
        "formal_endpoint": formal_endpoint,
        "evaluation_mode": binding.evaluation_mode if binding is not None else "synthetic_unbound_compatibility",
        "selection_semantic": binding.selection_semantic if binding is not None else "SYNTHETIC_NOT_SCIENTIFIC_RESULT",
    }
    return manifest, summary


def read_registered_prediction_artifact(
    path: str | Path,
    *,
    summary_path: str | Path,
    checkpoint_path: str | Path,
    evaluation_mode: str,
    allow_synthetic_portable_fallback: bool = False,
) -> tuple[tuple[PredictionRow, ...], Mapping[str, Any], PredictionArtifactBinding]:
    """Load a published prediction partition and revalidate its whole identity.

    Raw JSON/CSV rows are intentionally unsupported: evaluation may consume
    only the immutable Parquet publication plus its small identity summary.
    """

    source = Path(path)
    if source.suffix.lower() != ".parquet":
        raise SctsrError(
            ErrorCode.PREDICTION_IDENTITY_MISMATCH,
            "Evaluation accepts only a registered Parquet prediction artifact",
            artifact_path=str(source),
        )
    summary_source = Path(summary_path)
    if not source.is_file() or not summary_source.is_file():
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction artifact or identity summary is missing")
    summary = load_json(summary_source)
    if not isinstance(summary, Mapping) or summary.get("schema_version") != "stage1.sctsr.prediction_artifact_summary.v2":
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction identity summary schema is invalid")
    validate_columnar_file(
        source,
        expected_rows=int(summary.get("row_count", -1)),
        expected_sha256=str(summary.get("prediction_artifact_sha256", "")),
        expected_schema_version="stage1.sctsr.prediction_artifact.v2",
        allow_synthetic_portable_fallback=allow_synthetic_portable_fallback,
    )
    raw_rows = read_columnar(source, allow_synthetic_portable_fallback=allow_synthetic_portable_fallback)
    names = {field.name for field in fields(PredictionRow)}
    expected_names = names | {"sample_label_identity_digest", "artifact_row_count"}
    for index, raw in enumerate(raw_rows):
        if set(raw) != expected_names:
            raise SctsrError(
                ErrorCode.PREDICTION_IDENTITY_MISMATCH,
                "Prediction row schema differs from the registered publication",
                failing_field=f"row[{index}]",
                observed={"missing": sorted(expected_names - set(raw)), "extra": sorted(set(raw) - expected_names)},
            )
    rows = tuple(PredictionRow(**{name: raw[name] for name in names}) for raw in raw_rows)
    validate_prediction_rows(rows)
    digest = sample_label_identity_digest(rows)
    if summary.get("sample_label_identity_digest") != digest:
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction sample-label digest differs from its summary")
    if any(raw["sample_label_identity_digest"] != digest or int(raw["artifact_row_count"]) != len(rows) for raw in raw_rows):
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction rows do not carry the publication-wide identity")
    first = rows[0]
    expected_summary = {
        "run_id": first.run_id,
        "arm_id": first.arm_id,
        "training_seed": first.training_seed,
        "split_role": first.split_role,
        "split_manifest_path": first.split_manifest_path,
        "split_manifest_sha256": first.split_manifest_sha256,
        "checkpoint_epoch": first.checkpoint_epoch,
        "checkpoint_sha256": first.checkpoint_sha256,
        "model_variant": first.model_variant,
        "source_tree_digest": first.source_tree_digest,
        "prediction_generation": first.prediction_generation,
        "evaluation_mode": evaluation_mode,
    }
    mismatches = {
        field: {"row": value, "summary": summary.get(field)}
        for field, value in expected_summary.items()
        if summary.get(field) != value
    }
    if mismatches:
        raise SctsrError(ErrorCode.PREDICTION_IDENTITY_MISMATCH, "Prediction rows differ from their summary identity", observed=mismatches)
    binding = PredictionArtifactBinding(
        checkpoint_path=Path(checkpoint_path).resolve().as_posix(),
        checkpoint_sha256=first.checkpoint_sha256,
        checkpoint_epoch=first.checkpoint_epoch,
        model_variant=first.model_variant,
        source_tree_digest=first.source_tree_digest,
        training_seed=first.training_seed,
        split_role=first.split_role,
        split_manifest_path=first.split_manifest_path,
        split_manifest_sha256=first.split_manifest_sha256,
        evaluation_mode=evaluation_mode,
        selection_semantic=str(summary.get("selection_semantic", "")),
    )
    expected_labels = {row.sample_id: row.y_true for row in rows} if evaluation_mode == "synthetic" else None
    validate_prediction_binding(rows, binding, expected_sample_labels=expected_labels)
    return rows, summary, binding
