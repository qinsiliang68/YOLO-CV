"""Exact analytic last-layer gradient diagnostics for fixed checkpoint probes."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
from pathlib import Path
import shutil
import time
from typing import Any, Callable
import uuid

import numpy as np
import pandas as pd

from .errors import ValidationError
from .util import atomic_write_json, sha256_file


class GradientProbeError(ValidationError):
    """Raised when fixed-probe features cannot define valid gradient targets."""


@dataclass(frozen=True)
class LastLayerGradientResult:
    metrics: pd.DataFrame
    normal_target_weight_gradient: np.ndarray
    normal_target_bias_gradient: np.ndarray
    defect_target_weight_gradient: np.ndarray
    defect_target_bias_gradient: np.ndarray


@dataclass(frozen=True)
class GradientProbeSpec:
    run_id: str
    arm_id: str
    checkpoint_epoch: int
    checkpoint: Path
    candidate_manifest: Path
    dataset_root: Path
    output_dir: Path
    yolo_root: Path
    gpu_id: str
    batch: int
    workers: int
    imgsz: int
    accepted_defect_names: tuple[str, ...]
    save_feature_payload: bool = True

    def __post_init__(self) -> None:
        for name in ("checkpoint", "candidate_manifest", "dataset_root", "output_dir", "yolo_root"):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        if self.checkpoint_epoch <= 0 or self.batch <= 0 or self.workers < 0 or self.imgsz <= 0:
            raise GradientProbeError("invalid gradient probe runtime settings")
        if not self.accepted_defect_names:
            raise GradientProbeError("accepted defect class names must not be empty")


@dataclass(frozen=True)
class GradientProbeRunResult:
    status: str
    skipped: bool
    row_count: int
    feature_dim: int
    output_dir: Path


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _target_gradient(
    features: np.ndarray,
    residuals: np.ndarray,
    mask: np.ndarray,
    name: str,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    count = int(mask.sum())
    if count == 0:
        raise GradientProbeError(f"{name} target group is empty")
    weight = np.einsum("nc,nd->cd", residuals[mask], features[mask]) / count
    bias = residuals[mask].mean(axis=0)
    norm_squared = float(np.square(weight).sum() + np.square(bias).sum())
    return weight, bias, norm_squared, count


def _alignment(
    features: np.ndarray,
    residuals: np.ndarray,
    sample_norm_squared: np.ndarray,
    target_weight: np.ndarray,
    target_bias: np.ndarray,
    target_norm_squared: float,
    target_mask: np.ndarray,
    target_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dot = np.einsum("nc,cd,nd->n", residuals, target_weight, features) + residuals @ target_bias
    denominator = np.sqrt(sample_norm_squared * target_norm_squared)
    cosine = np.divide(dot, denominator, out=np.full_like(dot, np.nan), where=denominator > 0)
    excluded_dot = dot.copy()
    excluded_cosine = cosine.copy()
    if target_count > 1:
        indices = np.flatnonzero(target_mask)
        excluded_dot[indices] = (
            target_count * dot[indices] - sample_norm_squared[indices]
        ) / (target_count - 1)
        excluded_target_norm_squared = (
            target_count**2 * target_norm_squared
            - 2 * target_count * dot[indices]
            + sample_norm_squared[indices]
        ) / (target_count - 1) ** 2
        excluded_target_norm_squared = np.maximum(excluded_target_norm_squared, 0.0)
        excluded_denominator = np.sqrt(
            sample_norm_squared[indices] * excluded_target_norm_squared
        )
        excluded_cosine[indices] = np.divide(
            excluded_dot[indices],
            excluded_denominator,
            out=np.full(len(indices), np.nan, dtype=np.float64),
            where=excluded_denominator > 0,
        )
    else:
        excluded_dot[target_mask] = np.nan
        excluded_cosine[target_mask] = np.nan
    return dot, cosine, excluded_dot, excluded_cosine


def _quadrants(normal_dot: np.ndarray, defect_dot: np.ndarray) -> list[str]:
    values: list[str] = []
    for normal, defect in zip(normal_dot, defect_dot):
        if normal == 0 or defect == 0:
            values.append("NEUTRAL_AXIS")
        elif normal > 0 and defect > 0:
            values.append("HELP_BOTH")
        elif normal > 0 and defect < 0:
            values.append("HELP_NORMAL_HARM_DEFECT")
        elif normal < 0 and defect > 0:
            values.append("HARM_NORMAL_HELP_DEFECT")
        else:
            values.append("HARM_BOTH")
    return values


def compute_last_layer_gradient_metrics(
    features: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    normal_target_mask: np.ndarray,
    defect_target_mask: np.ndarray,
    defect_class_index: int = 1,
) -> LastLayerGradientResult:
    """Compute exact CE gradients for the final linear layer without backpropagating per sample.

    A positive target dot product means a small gradient-descent step on the
    candidate is predicted, to first order, to reduce the corresponding target
    loss. Normal and defect axes remain separate; no arbitrary composite score
    is created.
    """

    feature_array = np.asarray(features, dtype=np.float64)
    logit_array = np.asarray(logits, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    normal_mask = np.asarray(normal_target_mask, dtype=bool)
    defect_mask = np.asarray(defect_target_mask, dtype=bool)
    if feature_array.ndim != 2 or logit_array.ndim != 2:
        raise GradientProbeError("features and logits must be two-dimensional")
    sample_count = feature_array.shape[0]
    class_count = logit_array.shape[1]
    if logit_array.shape[0] != sample_count or label_array.shape != (sample_count,):
        raise GradientProbeError("feature, logit, and label sample counts differ")
    if normal_mask.shape != (sample_count,) or defect_mask.shape != (sample_count,):
        raise GradientProbeError("target masks do not match sample count")
    if not 0 <= defect_class_index < class_count:
        raise GradientProbeError("defect class index is outside logits")
    if ((label_array < 0) | (label_array >= class_count)).any():
        raise GradientProbeError("labels are outside logits")
    if not np.isfinite(feature_array).all() or not np.isfinite(logit_array).all():
        raise GradientProbeError("features or logits contain non-finite values")
    if (normal_mask & defect_mask).any():
        raise GradientProbeError("normal and defect target groups overlap")

    probabilities = _softmax(logit_array)
    one_hot = np.eye(class_count, dtype=np.float64)[label_array]
    residuals = probabilities - one_hot
    feature_norm_squared = np.square(feature_array).sum(axis=1)
    residual_norm_squared = np.square(residuals).sum(axis=1)
    gradient_norm_squared = residual_norm_squared * (feature_norm_squared + 1.0)
    normal_weight, normal_bias, normal_norm_squared, normal_count = _target_gradient(
        feature_array, residuals, normal_mask, "normal"
    )
    defect_weight, defect_bias, defect_norm_squared, defect_count = _target_gradient(
        feature_array, residuals, defect_mask, "defect"
    )
    normal_dot, normal_cosine, normal_excluded_dot, normal_excluded_cosine = _alignment(
        feature_array,
        residuals,
        gradient_norm_squared,
        normal_weight,
        normal_bias,
        normal_norm_squared,
        normal_mask,
        normal_count,
    )
    defect_dot, defect_cosine, defect_excluded_dot, defect_excluded_cosine = _alignment(
        feature_array,
        residuals,
        gradient_norm_squared,
        defect_weight,
        defect_bias,
        defect_norm_squared,
        defect_mask,
        defect_count,
    )
    selected_probability = probabilities[np.arange(sample_count), label_array]
    defect_probability = probabilities[:, defect_class_index]
    normal_class_index = 0 if defect_class_index != 0 else 1
    metrics = pd.DataFrame(
        {
            "y_true": label_array,
            "loss": -np.log(np.clip(selected_probability, 1e-300, 1.0)),
            "p_defect": defect_probability,
            "logit_margin_defect_minus_normal": logit_array[:, defect_class_index]
            - logit_array[:, normal_class_index],
            "feature_norm": np.sqrt(feature_norm_squared),
            "residual_norm_el2n": np.sqrt(residual_norm_squared),
            "gradient_norm": np.sqrt(gradient_norm_squared),
            "normal_target_member": normal_mask,
            "defect_target_member": defect_mask,
            "normal_target_dot": normal_dot,
            "normal_target_cosine": normal_cosine,
            "normal_target_dot_self_excluded": normal_excluded_dot,
            "normal_target_cosine_self_excluded": normal_excluded_cosine,
            "defect_target_dot": defect_dot,
            "defect_target_cosine": defect_cosine,
            "defect_target_dot_self_excluded": defect_excluded_dot,
            "defect_target_cosine_self_excluded": defect_excluded_cosine,
            "alignment_quadrant": _quadrants(normal_excluded_dot, defect_excluded_dot),
        }
    )
    return LastLayerGradientResult(
        metrics=metrics,
        normal_target_weight_gradient=normal_weight,
        normal_target_bias_gradient=normal_bias,
        defect_target_weight_gradient=defect_weight,
        defect_target_bias_gradient=defect_bias,
    )


def _boolean_series(values: pd.Series, name: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise GradientProbeError(f"{name} contains invalid booleans")
    return normalized.isin({"true", "1"}).to_numpy(dtype=bool)


def _load_candidates(spec: GradientProbeSpec) -> tuple[pd.DataFrame, list[str]]:
    if not spec.checkpoint.is_file():
        raise FileNotFoundError(spec.checkpoint)
    if not spec.candidate_manifest.is_file():
        raise FileNotFoundError(spec.candidate_manifest)
    frame = pd.read_csv(spec.candidate_manifest, keep_default_na=False)
    required = {
        "sample_id",
        "y_true",
        "normal_target_member",
        "defect_target_member",
        "candidate_groups",
    }
    missing = required - set(frame.columns)
    if missing:
        raise GradientProbeError(f"gradient candidate manifest missing columns: {sorted(missing)}")
    if frame.sample_id.astype(str).duplicated().any():
        raise GradientProbeError("gradient candidate manifest contains duplicate sample_id")
    path_column = next(
        (name for name in ("image_path", "canonical_image_relpath", "source_image_path") if name in frame),
        None,
    )
    if path_column is None:
        raise GradientProbeError("gradient candidate manifest has no image path column")
    labels = pd.to_numeric(frame.y_true, errors="raise").astype(int)
    if not set(labels.unique()) <= {0, 1}:
        raise GradientProbeError("gradient candidate labels must be binary")
    normal_mask = _boolean_series(frame.normal_target_member, "normal_target_member")
    defect_mask = _boolean_series(frame.defect_target_member, "defect_target_member")
    if (normal_mask & defect_mask).any():
        raise GradientProbeError("gradient target groups overlap")
    if not normal_mask.any() or not defect_mask.any():
        raise GradientProbeError("both gradient target groups must be non-empty")
    paths: list[str] = []
    for value in frame[path_column].astype(str):
        path = Path(value)
        resolved = path if path.is_absolute() else spec.dataset_root / path
        resolved = resolved.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        paths.append(str(resolved))
    frame = frame.copy()
    frame["y_true"] = labels
    frame["normal_target_member"] = normal_mask
    frame["defect_target_member"] = defect_mask
    return frame, paths


def _defect_index(names: Any, accepted_names: tuple[str, ...]) -> int:
    mapping = (
        {int(key): str(value) for key, value in names.items()}
        if isinstance(names, dict)
        else {index: str(value) for index, value in enumerate(names)}
    )
    accepted = {value.lower() for value in accepted_names} | {"target_defect"}
    matches = [index for index, name in mapping.items() if name.lower() in accepted]
    if len(matches) != 1:
        raise GradientProbeError(f"cannot identify one defect class: names={mapping}")
    return matches[0]


def _default_feature_extractor(
    *,
    spec: GradientProbeSpec,
    image_paths: list[str],
    sample_ids: list[str],
    **_: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from .campaign_dynamic_training import _activate_local_ultralytics

    _activate_local_ultralytics(spec.yolo_root)
    import torch
    from ultralytics import YOLO

    model = None
    handle = None
    feature_batches: list[np.ndarray] = []
    logit_batches: list[np.ndarray] = []
    probability_rows: list[np.ndarray] = []
    started = time.perf_counter()
    try:
        model = YOLO(str(spec.checkpoint))
        head = model.model.model[-1]
        linear = getattr(head, "linear", None)
        if linear is None or int(getattr(linear, "out_features", -1)) != 2:
            raise GradientProbeError("gradient probe requires the binary final linear classification layer")

        def capture(_module, inputs, output) -> None:
            feature_batches.append(inputs[0].detach().float().cpu().numpy())
            logit_batches.append(output.detach().float().cpu().numpy())

        handle = linear.register_forward_hook(capture)
        for result in model.predict(
            source=image_paths,
            imgsz=spec.imgsz,
            batch=spec.batch,
            device=str(spec.gpu_id),
            workers=spec.workers,
            verbose=False,
            stream=True,
        ):
            if result.probs is None:
                raise GradientProbeError("gradient probe prediction did not return probabilities")
            probability_rows.append(result.probs.data.detach().float().cpu().numpy())
        feature_array = np.concatenate(feature_batches, axis=0)
        logit_array = np.concatenate(logit_batches, axis=0)
        expected = len(sample_ids)
        if len(probability_rows) != expected or len(feature_array) < expected or len(logit_array) < expected:
            raise GradientProbeError("gradient feature extraction row count mismatch")
        discarded = len(feature_array) - expected
        feature_array = feature_array[-expected:]
        logit_array = logit_array[-expected:]
        probabilities = np.stack(probability_rows)
        reconstructed = _softmax(logit_array.astype(np.float64))
        maximum_probability_difference = float(np.abs(reconstructed - probabilities).max())
        if maximum_probability_difference > 1e-5:
            raise GradientProbeError(
                f"captured logits do not reproduce prediction probabilities: {maximum_probability_difference}"
            )
        defect_class_index = _defect_index(model.names, spec.accepted_defect_names)
        return (
            feature_array.astype(np.float32, copy=False),
            logit_array.astype(np.float32, copy=False),
            {
                "extractor": "ultralytics_final_linear_forward_hook",
                "class_names": {str(key): str(value) for key, value in model.names.items()},
                "defect_class_index": defect_class_index,
                "hook_observation_rows": int(len(feature_array) + discarded),
                "discarded_warmup_rows": int(discarded),
                "maximum_probability_reconstruction_difference": maximum_probability_difference,
                "duration_seconds": time.perf_counter() - started,
            },
        )
    finally:
        if handle is not None:
            handle.remove()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _validate_existing_gradient_probe(spec: GradientProbeSpec) -> GradientProbeRunResult | None:
    if not spec.output_dir.exists():
        return None
    files = [item for item in spec.output_dir.iterdir()]
    if not files:
        return None
    manifest_path = spec.output_dir / "gradient_probe_manifest.json"
    scalar_path = spec.output_dir / "gradient_probe_scalars.parquet"
    feature_path = spec.output_dir / "gradient_feature_payload.npz"
    required = [manifest_path, scalar_path]
    if spec.save_feature_payload:
        required.append(feature_path)
    if not all(path.is_file() for path in required):
        raise GradientProbeError(f"gradient probe output is half-published: {spec.output_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "status": "COMPLETE",
        "run_id": spec.run_id,
        "arm_id": spec.arm_id,
        "checkpoint_epoch": spec.checkpoint_epoch,
        "checkpoint_sha256": sha256_file(spec.checkpoint),
        "candidate_manifest_sha256": sha256_file(spec.candidate_manifest),
        "scalar_sha256": sha256_file(scalar_path),
    }
    if spec.save_feature_payload:
        expected["feature_payload_sha256"] = sha256_file(feature_path)
    mismatch = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
    if mismatch:
        raise GradientProbeError(f"gradient probe manifest mismatch: {mismatch}")
    import polars as pl

    scalar_rows = int(pl.scan_parquet(scalar_path).select(pl.len()).collect().item())
    if scalar_rows != int(manifest.get("row_count", -1)):
        raise GradientProbeError("gradient probe scalar row count mismatch")
    return GradientProbeRunResult(
        "PASS",
        True,
        scalar_rows,
        int(manifest["feature_dim"]),
        spec.output_dir,
    )


def run_gradient_probe(
    spec: GradientProbeSpec,
    *,
    feature_extractor: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]] | None = None,
) -> GradientProbeRunResult:
    existing = _validate_existing_gradient_probe(spec)
    if existing is not None:
        return existing
    candidates, image_paths = _load_candidates(spec)
    staging = spec.output_dir.parent / f".{spec.output_dir.name}.{uuid.uuid4().hex}.tmpdir"
    started = time.perf_counter()
    try:
        staging.mkdir(parents=True)
        extractor = feature_extractor or _default_feature_extractor
        features, logits, extractor_metadata = extractor(
            spec=spec,
            image_paths=image_paths,
            sample_ids=candidates.sample_id.astype(str).tolist(),
            labels=candidates.y_true.to_numpy(dtype=np.int64),
        )
        feature_array = np.asarray(features, dtype=np.float32)
        logit_array = np.asarray(logits, dtype=np.float32)
        if feature_array.shape[0] != len(candidates) or logit_array.shape != (len(candidates), 2):
            raise GradientProbeError("extracted feature/logit shape does not match candidate manifest")
        defect_index = int(extractor_metadata.get("defect_class_index", 1))
        gradient = compute_last_layer_gradient_metrics(
            feature_array,
            logit_array,
            candidates.y_true.to_numpy(dtype=np.int64),
            normal_target_mask=candidates.normal_target_member.to_numpy(dtype=bool),
            defect_target_mask=candidates.defect_target_member.to_numpy(dtype=bool),
            defect_class_index=defect_index,
        )
        identity_columns = [
            column
            for column in candidates.columns
            if column not in {"y_true", "normal_target_member", "defect_target_member"}
        ]
        scalars = pd.concat(
            [candidates[identity_columns].reset_index(drop=True), gradient.metrics.reset_index(drop=True)],
            axis=1,
        )
        scalar_path = staging / "gradient_probe_scalars.parquet"
        import polars as pl

        pl.DataFrame(scalars.to_dict(orient="list"), infer_schema_length=None).write_parquet(
            scalar_path,
            compression="zstd",
            statistics=True,
        )
        feature_path = staging / "gradient_feature_payload.npz"
        if spec.save_feature_payload:
            np.savez_compressed(
                feature_path,
                sample_id=candidates.sample_id.astype(str).to_numpy(dtype=str),
                y_true=candidates.y_true.to_numpy(dtype=np.int8),
                features=feature_array,
                logits=logit_array,
                normal_target_weight_gradient=gradient.normal_target_weight_gradient.astype(np.float32),
                normal_target_bias_gradient=gradient.normal_target_bias_gradient.astype(np.float32),
                defect_target_weight_gradient=gradient.defect_target_weight_gradient.astype(np.float32),
                defect_target_bias_gradient=gradient.defect_target_bias_gradient.astype(np.float32),
            )
        manifest = {
            "schema_version": "stage1.gradient_probe.v1",
            "status": "COMPLETE",
            "run_id": spec.run_id,
            "arm_id": spec.arm_id,
            "checkpoint_epoch": spec.checkpoint_epoch,
            "checkpoint": str(spec.checkpoint),
            "checkpoint_sha256": sha256_file(spec.checkpoint),
            "candidate_manifest": str(spec.candidate_manifest),
            "candidate_manifest_sha256": sha256_file(spec.candidate_manifest),
            "row_count": len(scalars),
            "feature_dim": int(feature_array.shape[1]),
            "normal_target_count": int(candidates.normal_target_member.sum()),
            "defect_target_count": int(candidates.defect_target_member.sum()),
            "scalar_sha256": sha256_file(scalar_path),
            "feature_payload_saved": spec.save_feature_payload,
            "feature_payload_sha256": sha256_file(feature_path) if spec.save_feature_payload else None,
            "extractor_metadata": extractor_metadata,
            "duration_seconds": time.perf_counter() - started,
            "interpretation": (
                "Positive target dot predicts first-order target-loss reduction. Normal and defect axes are "
                "reported separately; same-target members use leave-one-out columns to remove self-inclusion."
            ),
        }
        atomic_write_json(staging / "gradient_probe_manifest.json", manifest)
        if spec.output_dir.exists():
            spec.output_dir.rmdir()
        staging.replace(spec.output_dir)
        return GradientProbeRunResult(
            "PASS",
            False,
            len(scalars),
            int(feature_array.shape[1]),
            spec.output_dir,
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


__all__ = [
    "GradientProbeError",
    "GradientProbeSpec",
    "GradientProbeRunResult",
    "LastLayerGradientResult",
    "compute_last_layer_gradient_metrics",
    "run_gradient_probe",
]
