"""Atomic fixed-probe and val_op prediction exports at preregistered checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Callable
import uuid

import numpy as np
import pandas as pd

from .errors import ExternalCommandError, ValidationError
from .subprocesses import run_logged
from .util import atomic_write_bytes, atomic_write_json, sha256_file


class CampaignPredictionError(ValidationError):
    """Raised when a key-checkpoint prediction artifact is incomplete or invalid."""


_DERIVED_PROBABILITY_FIELDS = (
    "score_raw",
    "p_defect",
    "p_normal",
    "probability_margin",
    "log_odds_defect",
    "entropy_nats",
    "predicted_y",
)


@dataclass(frozen=True)
class CausalProbeManifests:
    normal_manifest: Path
    defect_manifest: Path
    normal_count: int
    defect_count: int
    identity_path: Path


@dataclass(frozen=True)
class CampaignCheckpointPredictionSpec:
    run_id: str
    arm_id: str
    job_id: str
    checkpoint_epochs: tuple[int, ...]
    training_state_dir: Path
    output_dir: Path
    dataset_root: Path
    normal_train_manifest: Path
    defect_train_manifest: Path
    monitor_manifest: Path
    val_op_normal_manifest: Path
    val_op_defect_manifest: Path
    yolo_root: Path
    python_executable: str
    gpu_id: str
    batch: int
    workers: int
    imgsz: int
    accepted_defect_names: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "training_state_dir",
            "output_dir",
            "dataset_root",
            "normal_train_manifest",
            "defect_train_manifest",
            "monitor_manifest",
            "val_op_normal_manifest",
            "val_op_defect_manifest",
            "yolo_root",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        epochs = tuple(int(epoch) for epoch in self.checkpoint_epochs)
        if not epochs or len(epochs) != len(set(epochs)) or any(epoch <= 0 for epoch in epochs):
            raise CampaignPredictionError("checkpoint epochs must be unique positive integers")
        object.__setattr__(self, "checkpoint_epochs", epochs)
        if self.batch <= 0 or self.workers < 0 or self.imgsz <= 0:
            raise CampaignPredictionError("invalid checkpoint prediction runtime settings")


@dataclass(frozen=True)
class CampaignCheckpointPredictionResult:
    status: str
    epoch_count: int
    published_split_count: int
    skipped_split_count: int
    output_dir: Path


def _read_source_manifest(path: Path, name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, keep_default_na=False)
    identity = "canonical_image_relpath" if "canonical_image_relpath" in frame else "sample_id" if "sample_id" in frame else None
    if identity is None:
        raise CampaignPredictionError(f"{name} has no canonical sample identity")
    frame = frame.copy()
    frame["_sample_id"] = frame[identity].astype(str)
    if frame._sample_id.duplicated().any():
        raise CampaignPredictionError(f"{name} contains duplicate sample identities")
    return frame


def _ordered_probe_rows(source: pd.DataFrame, ids: list[str], name: str) -> pd.DataFrame:
    indexed = source.set_index("_sample_id", drop=False)
    missing = [sample_id for sample_id in ids if sample_id not in indexed.index]
    if missing:
        raise CampaignPredictionError(f"{name} monitor samples are missing from source manifest: {missing[:3]}")
    rows = indexed.loc[ids].drop(columns=["_sample_id"]).reset_index(drop=True)
    return rows


def build_causal_probe_manifests(
    monitor_manifest: str | Path,
    normal_train_manifest: str | Path,
    defect_train_manifest: str | Path,
    output_dir: str | Path,
) -> CausalProbeManifests:
    monitor_path = Path(monitor_manifest).resolve()
    normal_path = Path(normal_train_manifest).resolve()
    defect_path = Path(defect_train_manifest).resolve()
    output = Path(output_dir).resolve()
    monitor = pd.read_csv(monitor_path, keep_default_na=False)
    required = {"sample_id", "y_true"}
    if required - set(monitor.columns):
        raise CampaignPredictionError(f"monitor manifest missing columns: {sorted(required - set(monitor.columns))}")
    if monitor.sample_id.astype(str).duplicated().any():
        raise CampaignPredictionError("monitor manifest contains duplicate sample_id")
    labels = pd.to_numeric(monitor.y_true, errors="raise").astype(int)
    if not set(labels.unique()) <= {0, 1}:
        raise CampaignPredictionError("monitor manifest y_true must be binary")
    normal_source = _read_source_manifest(normal_path, "normal train manifest")
    defect_source = _read_source_manifest(defect_path, "defect train manifest")
    normal_ids = monitor.loc[labels.eq(0), "sample_id"].astype(str).tolist()
    defect_ids = monitor.loc[labels.eq(1), "sample_id"].astype(str).tolist()
    normal_rows = _ordered_probe_rows(normal_source, normal_ids, "normal")
    defect_rows = _ordered_probe_rows(defect_source, defect_ids, "defect")
    normal_output = output / "causal_probe_normal_manifest.csv"
    defect_output = output / "causal_probe_defect_manifest.csv"
    identity_path = output / "causal_probe_identity.json"
    expected_identity = {
        "schema_version": "stage1.causal_probe_manifests.v1",
        "monitor_manifest_sha256": sha256_file(monitor_path),
        "normal_train_manifest_sha256": sha256_file(normal_path),
        "defect_train_manifest_sha256": sha256_file(defect_path),
        "normal_count": len(normal_rows),
        "defect_count": len(defect_rows),
    }
    if output.exists() and any(output.iterdir()):
        if not all(path.is_file() for path in (normal_output, defect_output, identity_path)):
            raise CampaignPredictionError(f"causal probe manifests are half-published: {output}")
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        for key, value in expected_identity.items():
            if identity.get(key) != value:
                raise CampaignPredictionError(f"existing causal probe identity mismatch: {key}")
        if identity.get("normal_manifest_sha256") != sha256_file(normal_output) or identity.get(
            "defect_manifest_sha256"
        ) != sha256_file(defect_output):
            raise CampaignPredictionError("existing causal probe manifest checksum mismatch")
    else:
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(normal_output, normal_rows.to_csv(index=False, lineterminator="\n").encode("utf-8"))
        atomic_write_bytes(defect_output, defect_rows.to_csv(index=False, lineterminator="\n").encode("utf-8"))
        atomic_write_json(
            identity_path,
            {
                **expected_identity,
                "normal_manifest_sha256": sha256_file(normal_output),
                "defect_manifest_sha256": sha256_file(defect_output),
            },
        )
    return CausalProbeManifests(normal_output, defect_output, len(normal_rows), len(defect_rows), identity_path)


def _validate_prediction(path: Path, *, expected_defect: int, expected_normal: int) -> dict[str, Any]:
    frame = pd.read_csv(path)
    required = {"sample_id", "y_true", "score", *_DERIVED_PROBABILITY_FIELDS}
    if required - set(frame.columns):
        raise CampaignPredictionError(f"prediction output missing columns: {sorted(required - set(frame.columns))}")
    if frame.sample_id.astype(str).duplicated().any():
        raise CampaignPredictionError("prediction output contains duplicate sample_id")
    labels = pd.to_numeric(frame.y_true, errors="raise").astype(int)
    scores = pd.to_numeric(frame.score, errors="raise").to_numpy(float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise CampaignPredictionError("prediction scores are not finite probabilities")
    defect_count = int(labels.eq(1).sum())
    normal_count = int(labels.eq(0).sum())
    if defect_count != expected_defect or normal_count != expected_normal:
        raise CampaignPredictionError(
            f"prediction row counts mismatch: defect={defect_count}/{expected_defect}, "
            f"normal={normal_count}/{expected_normal}"
        )
    return {
        "row_count": len(frame),
        "defect_count": defect_count,
        "normal_count": normal_count,
        "sha256": sha256_file(path),
    }


def _enrich_prediction(path: Path) -> None:
    frame = pd.read_csv(path)
    required = {"sample_id", "y_true", "score"}
    if required - set(frame.columns):
        raise CampaignPredictionError(f"raw prediction output missing columns: {sorted(required - set(frame.columns))}")
    probability = pd.to_numeric(frame.score, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise CampaignPredictionError("raw prediction scores are not finite probabilities")
    normal = 1.0 - probability
    clipped_defect = np.clip(probability, 1e-15, 1.0)
    clipped_normal = np.clip(normal, 1e-15, 1.0)
    frame["score_raw"] = probability
    frame["p_defect"] = probability
    frame["p_normal"] = normal
    frame["probability_margin"] = probability - normal
    frame["log_odds_defect"] = np.log(clipped_defect) - np.log(clipped_normal)
    frame["entropy_nats"] = -(clipped_defect * np.log(clipped_defect) + clipped_normal * np.log(clipped_normal))
    frame["predicted_y"] = (probability >= 0.5).astype(np.int8)
    atomic_write_bytes(path, frame.to_csv(index=False, lineterminator="\n").encode("utf-8"), overwrite=True)


def _default_prediction_runner(spec: CampaignCheckpointPredictionSpec, **kwargs) -> dict[str, Any]:
    command = [
        str(spec.python_executable),
        "-m",
        "stage1_gapvalue240.prediction_worker",
        "--split",
        str(kwargs["split_name"]),
        "--checkpoint",
        str(kwargs["checkpoint"]),
        "--dataset-root",
        str(spec.dataset_root),
        "--defect-manifest",
        str(kwargs["defect_manifest"]),
        "--normal-manifest",
        str(kwargs["normal_manifest"]),
        "--output",
        str(kwargs["temporary_output"]),
        "--result-json",
        str(kwargs["result_json"]),
        "--yolo-root",
        str(spec.yolo_root),
        "--gpu-id",
        str(spec.gpu_id),
        "--batch",
        str(spec.batch),
        "--workers",
        str(spec.workers),
        "--imgsz",
        str(spec.imgsz),
    ]
    for name in spec.accepted_defect_names:
        command.extend(["--accepted-defect-name", name])
    try:
        run_logged(command, spec.yolo_root.parent, kwargs["log_path"], timeout=None)
    except ExternalCommandError as exc:
        result_path = Path(kwargs["result_json"])
        report = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        message = f"checkpoint prediction worker failed for {kwargs['split_name']}: {report.get('error', exc)}"
        if report.get("status") in {"INVALID_INPUT", "ARTIFACT_INVALID"}:
            raise CampaignPredictionError(message) from exc
        raise ExternalCommandError(message) from exc
    return json.loads(Path(kwargs["result_json"]).read_text(encoding="utf-8"))


def _existing_split(
    stable_output: Path,
    sidecar: Path,
    *,
    checkpoint_sha: str,
    expected_defect: int,
    expected_normal: int,
) -> bool:
    if not stable_output.exists() and not sidecar.exists():
        return False
    if not stable_output.is_file() or not sidecar.is_file():
        raise CampaignPredictionError(f"checkpoint prediction is half-published: {stable_output}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    validation = _validate_prediction(
        stable_output,
        expected_defect=expected_defect,
        expected_normal=expected_normal,
    )
    expected = {
        "status": "COMPLETE",
        "checkpoint_sha256": checkpoint_sha,
        "prediction_sha256": validation["sha256"],
        "row_count": validation["row_count"],
    }
    mismatch = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatch:
        raise CampaignPredictionError(f"checkpoint prediction sidecar mismatch: {mismatch}")
    return True


def run_key_checkpoint_predictions(
    spec: CampaignCheckpointPredictionSpec,
    *,
    prediction_runner: Callable[..., dict[str, Any]] | None = None,
) -> CampaignCheckpointPredictionResult:
    runner = prediction_runner
    probe = build_causal_probe_manifests(
        spec.monitor_manifest,
        spec.normal_train_manifest,
        spec.defect_train_manifest,
        spec.output_dir / "probe_manifests",
    )
    val_defect_count = len(pd.read_csv(spec.val_op_defect_manifest))
    val_normal_count = len(pd.read_csv(spec.val_op_normal_manifest))
    published = 0
    skipped = 0
    for epoch in spec.checkpoint_epochs:
        checkpoint = spec.training_state_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_sha = sha256_file(checkpoint)
        epoch_dir = spec.output_dir / f"epoch_{epoch:04d}"
        final_manifest = epoch_dir / "checkpoint_prediction_manifest.json"
        if final_manifest.is_file():
            final = json.loads(final_manifest.read_text(encoding="utf-8"))
            if final.get("status") != "COMPLETE" or final.get("checkpoint_sha256") != checkpoint_sha:
                raise CampaignPredictionError(f"completed checkpoint prediction identity mismatch: epoch {epoch}")
        split_specs = (
            (
                "val_op",
                spec.val_op_defect_manifest,
                spec.val_op_normal_manifest,
                val_defect_count,
                val_normal_count,
            ),
            (
                "causal_train_probe",
                probe.defect_manifest,
                probe.normal_manifest,
                probe.defect_count,
                probe.normal_count,
            ),
        )
        split_records: dict[str, Any] = {}
        for split_name, defect_manifest, normal_manifest, expected_defect, expected_normal in split_specs:
            stable = epoch_dir / f"{split_name}_predictions.csv"
            sidecar = epoch_dir / f"{split_name}_predictions.manifest.json"
            if _existing_split(
                stable,
                sidecar,
                checkpoint_sha=checkpoint_sha,
                expected_defect=expected_defect,
                expected_normal=expected_normal,
            ):
                skipped += 1
                split_records[split_name] = json.loads(sidecar.read_text(encoding="utf-8"))
                continue
            attempt = epoch_dir / "attempts" / f"{split_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            attempt.mkdir(parents=True, exist_ok=False)
            temporary = attempt / "predictions.csv"
            result_json = attempt / "worker_result.json"
            log_path = attempt / "worker.log"
            kwargs = {
                "split_name": split_name,
                "epoch": epoch,
                "checkpoint": checkpoint,
                "defect_manifest": defect_manifest,
                "normal_manifest": normal_manifest,
                "temporary_output": temporary,
                "result_json": result_json,
                "log_path": log_path,
            }
            report = runner(**kwargs) if runner is not None else _default_prediction_runner(spec, **kwargs)
            if report.get("status") != "PASS" or not temporary.is_file():
                raise CampaignPredictionError(f"checkpoint prediction did not pass: epoch={epoch}, split={split_name}")
            _enrich_prediction(temporary)
            validation = _validate_prediction(
                temporary,
                expected_defect=expected_defect,
                expected_normal=expected_normal,
            )
            epoch_dir.mkdir(parents=True, exist_ok=True)
            if stable.exists():
                raise CampaignPredictionError(f"refusing to overwrite stable prediction: {stable}")
            os.replace(temporary, stable)
            validation["sha256"] = sha256_file(stable)
            split_record = {
                "schema_version": "stage1.checkpoint_prediction_split.v1",
                "status": "COMPLETE",
                "run_id": spec.run_id,
                "arm_id": spec.arm_id,
                "job_id": spec.job_id,
                "epoch": epoch,
                "split": split_name,
                "checkpoint_sha256": checkpoint_sha,
                "prediction_sha256": validation["sha256"],
                "row_count": validation["row_count"],
                "defect_count": validation["defect_count"],
                "normal_count": validation["normal_count"],
                "derived_probability_fields": list(_DERIVED_PROBABILITY_FIELDS),
                "worker_result_sha256": sha256_file(result_json),
                "worker_log_sha256": sha256_file(log_path),
            }
            atomic_write_json(sidecar, split_record)
            split_records[split_name] = split_record
            published += 1
        final_payload = {
            "schema_version": "stage1.checkpoint_predictions.v1",
            "status": "COMPLETE",
            "run_id": spec.run_id,
            "arm_id": spec.arm_id,
            "job_id": spec.job_id,
            "epoch": epoch,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "splits": split_records,
        }
        if final_manifest.is_file():
            existing = json.loads(final_manifest.read_text(encoding="utf-8"))
            if existing != final_payload:
                raise CampaignPredictionError(f"completed checkpoint prediction manifest drift: epoch {epoch}")
        else:
            atomic_write_json(final_manifest, final_payload)
    return CampaignCheckpointPredictionResult("PASS", len(spec.checkpoint_epochs), published, skipped, spec.output_dir)


__all__ = [
    "CampaignPredictionError",
    "CausalProbeManifests",
    "CampaignCheckpointPredictionSpec",
    "CampaignCheckpointPredictionResult",
    "build_causal_probe_manifests",
    "run_key_checkpoint_predictions",
]
