"""Low-overhead process telemetry for the dynamic replay campaign.

The collector deliberately retains full epoch-level aggregates for all training
draws and per-sample trajectories only for frozen monitor or replay samples.
This preserves the causal evidence needed by the campaign without emitting the
roughly two billion rows implied by 120k samples x 200 epochs x 84 runs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import uuid
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from .errors import ValidationError
from .util import atomic_write_json, sha256_file, stable_hash


TELEMETRY_SCHEMA = "stage1.process_telemetry.v1"
_BASE_COLUMNS = {"canonical_image_relpath", "Filename"}
_REPLAY_COLUMNS = {"staged_filename", "sample_id", "y_true", "replay_role"}
_MONITOR_COLUMNS = {"sample_id", "monitor_group"}
_REPLAY_ROLES = {"normal_replay", "defect_guard"}


@dataclass(frozen=True)
class ProcessTelemetrySpec:
    run_id: str
    arm_id: str
    segment_id: str
    output_dir: Path
    base_normal_manifest: Path
    base_defect_manifest: Path
    replay_identity_manifest: Path
    monitor_manifest: Path
    expected_epoch_samples: int
    expected_replay_samples: int
    target_defect_class_index: int = 1

    def __post_init__(self) -> None:
        for name in (
            "output_dir",
            "base_normal_manifest",
            "base_defect_manifest",
            "replay_identity_manifest",
            "monitor_manifest",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        if not self.run_id or not self.arm_id or not self.segment_id:
            raise ValidationError("process telemetry identity fields must be non-empty")
        if self.expected_epoch_samples <= 0 or self.expected_replay_samples < 0:
            raise ValidationError("process telemetry expected counts are invalid")
        if self.target_defect_class_index < 0:
            raise ValidationError("target_defect_class_index must be non-negative")


@dataclass(frozen=True)
class _Identity:
    sample_id: str
    y_true: int
    exposure_role: str
    monitor_groups: str


def _read_csv(path: Path, required: set[str], name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, keep_default_na=False)
    missing = required - set(frame.columns)
    if missing:
        raise ValidationError(f"{name} missing columns: {sorted(missing)}")
    return frame


def _class_name(y_true: int) -> str:
    if y_true == 0:
        return "no_target"
    if y_true == 1:
        return "target_defect"
    raise ValidationError(f"binary Stage1 telemetry received y_true={y_true}")


def _identity_contract(spec: ProcessTelemetrySpec) -> tuple[dict[tuple[str, str], _Identity], str]:
    base_rows: list[tuple[str, str, int]] = []
    for path, y_true, name in (
        (spec.base_normal_manifest, 0, "base normal manifest"),
        (spec.base_defect_manifest, 1, "base defect manifest"),
    ):
        frame = _read_csv(path, _BASE_COLUMNS, name)
        for row in frame.itertuples(index=False):
            sample_id = str(getattr(row, "canonical_image_relpath"))
            filename = str(getattr(row, "Filename"))
            if not sample_id or not filename or Path(filename).name != filename:
                raise ValidationError(f"{name} contains an invalid sample identity")
            base_rows.append((sample_id, filename, y_true))
    sample_labels: dict[str, int] = {}
    lookup: dict[tuple[str, str], _Identity] = {}
    for sample_id, filename, y_true in base_rows:
        if sample_id in sample_labels:
            raise ValidationError(f"duplicate base sample_id: {sample_id}")
        sample_labels[sample_id] = y_true
        key = (_class_name(y_true), filename)
        if key in lookup:
            raise ValidationError(f"duplicate staged base identity: {key}")
        lookup[key] = _Identity(sample_id, y_true, f"base_{'normal' if y_true == 0 else 'defect'}", "")

    monitor = _read_csv(spec.monitor_manifest, _MONITOR_COLUMNS, "monitor manifest")
    monitor_groups: dict[str, set[str]] = defaultdict(set)
    for row in monitor.itertuples(index=False):
        sample_id = str(row.sample_id)
        group = str(row.monitor_group)
        if sample_id not in sample_labels:
            raise ValidationError(f"monitor sample_id absent from base manifests: {sample_id}")
        if not group:
            raise ValidationError(f"monitor group is empty for: {sample_id}")
        monitor_groups[sample_id].add(group)

    for key, identity in list(lookup.items()):
        groups = ";".join(sorted(monitor_groups.get(identity.sample_id, set())))
        lookup[key] = _Identity(identity.sample_id, identity.y_true, identity.exposure_role, groups)

    replay = _read_csv(spec.replay_identity_manifest, _REPLAY_COLUMNS, "replay identity manifest")
    if len(replay) != spec.expected_replay_samples:
        raise ValidationError(
            f"replay identity rows {len(replay)} != expected {spec.expected_replay_samples}"
        )
    staged_names: set[tuple[str, str]] = set()
    for row in replay.itertuples(index=False):
        filename = str(row.staged_filename)
        sample_id = str(row.sample_id)
        y_true = int(row.y_true)
        role = str(row.replay_role)
        if role not in _REPLAY_ROLES:
            raise ValidationError(f"invalid replay_role: {role}")
        expected_role = "normal_replay" if y_true == 0 else "defect_guard"
        if role != expected_role:
            raise ValidationError(f"replay_role {role} conflicts with y_true={y_true}")
        if not filename.startswith("replay__") or Path(filename).name != filename:
            raise ValidationError(f"invalid staged replay filename: {filename}")
        if sample_labels.get(sample_id) != y_true:
            raise ValidationError(f"replay sample identity/label is absent or inconsistent: {sample_id}")
        key = (_class_name(y_true), filename)
        if key in lookup or key in staged_names:
            raise ValidationError(f"duplicate staged replay identity: {key}")
        staged_names.add(key)
        groups = ";".join(sorted(monitor_groups.get(sample_id, set())))
        lookup[key] = _Identity(sample_id, y_true, role, groups)

    expected = len(base_rows) + len(replay)
    if expected != spec.expected_epoch_samples:
        raise ValidationError(
            f"identity index rows {expected} != expected epoch samples {spec.expected_epoch_samples}"
        )
    digest_rows = [
        {
            "class_name": key[0],
            "filename": key[1],
            "sample_id": value.sample_id,
            "y_true": value.y_true,
            "exposure_role": value.exposure_role,
            "monitor_groups": value.monitor_groups,
        }
        for key, value in sorted(lookup.items())
    ]
    return lookup, stable_hash(digest_rows)


def _extract_logits(preds: Any):
    return preds[1] if isinstance(preds, (list, tuple)) else preds


def _digest_update(digest: "hashlib._Hash", *parts: str | bytes) -> None:
    for part in parts:
        payload = part if isinstance(part, bytes) else str(part).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)


def _stats(values: list[float], prefix: str) -> dict[str, float | None]:
    if not values:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_min": None,
            f"{prefix}_q10": None,
            f"{prefix}_q50": None,
            f"{prefix}_q90": None,
            f"{prefix}_q95": None,
            f"{prefix}_q99": None,
            f"{prefix}_max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValidationError(f"non-finite {prefix} encountered in process telemetry")
    quantiles = np.quantile(array, [0.10, 0.50, 0.90, 0.95, 0.99])
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_std": float(array.std(ddof=0)),
        f"{prefix}_min": float(array.min()),
        f"{prefix}_q10": float(quantiles[0]),
        f"{prefix}_q50": float(quantiles[1]),
        f"{prefix}_q90": float(quantiles[2]),
        f"{prefix}_q95": float(quantiles[3]),
        f"{prefix}_q99": float(quantiles[4]),
        f"{prefix}_max": float(array.max()),
    }


def _blank_row(spec: ProcessTelemetrySpec, epoch: int, record_type: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": TELEMETRY_SCHEMA,
        "record_type": record_type,
        "run_id": spec.run_id,
        "arm_id": spec.arm_id,
        "segment_id": spec.segment_id,
        "epoch": int(epoch),
        "exposure_role": "",
        "sample_id": "",
        "y_true": None,
        "monitor_groups": "",
        "batch_count": None,
        "total_exposure_count": 0,
        "base_exposure_count": 0,
        "replay_normal_exposure_count": 0,
        "guard_defect_exposure_count": 0,
        "minibatch_order_sha256": "",
        "batch_size_sequence_sha256": "",
        "augmentation_realization_sha256": "",
        "augmentation_pooled_mean": None,
        "augmentation_pooled_std": None,
    }
    row.update(_stats([], "loss"))
    row.update(_stats([], "p_defect"))
    return row


def validate_process_telemetry_epoch(spec: ProcessTelemetrySpec, epoch: int) -> dict[str, Any]:
    parquet = spec.output_dir / f"epoch_{epoch:04d}_process_telemetry.parquet"
    sidecar = parquet.with_suffix(".json")
    if not parquet.is_file() or not sidecar.is_file():
        raise ValidationError(f"process telemetry epoch {epoch} is incomplete")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"process telemetry sidecar is unreadable: {sidecar}") from exc
    expected = {
        "schema_version": TELEMETRY_SCHEMA,
        "status": "COMPLETE",
        "run_id": spec.run_id,
        "arm_id": spec.arm_id,
        "segment_id": spec.segment_id,
        "epoch": int(epoch),
        "observed_epoch_samples": spec.expected_epoch_samples,
        "observed_replay_samples": spec.expected_replay_samples,
    }
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatches:
        raise ValidationError(f"process telemetry epoch {epoch} identity mismatch: {mismatches}")
    if metadata.get("parquet_sha256") != sha256_file(parquet):
        raise ValidationError(f"process telemetry epoch {epoch} checksum mismatch")
    try:
        import polars as pl

        row_count = int(pl.scan_parquet(parquet).select(pl.len()).collect().item())
    except Exception as exc:
        raise ValidationError(f"process telemetry parquet is unreadable: {parquet}") from exc
    if row_count != int(metadata.get("row_count", -1)):
        raise ValidationError(f"process telemetry epoch {epoch} row_count mismatch")
    return metadata


class ProcessTelemetryCollector:
    """Collect all-draw aggregates and bounded per-sample process trajectories."""

    def __init__(
        self,
        spec: ProcessTelemetrySpec,
        *,
        after_parquet_publish: Callable[[int], None] | None = None,
    ):
        self.spec = spec
        self.after_parquet_publish = after_parquet_publish
        self.lookup, self.identity_index_sha256 = _identity_contract(spec)
        self.spec.output_dir.mkdir(parents=True, exist_ok=True)
        self.active_epoch: int | None = None
        self.last_finished_epoch: int | None = None
        self.training_active = False
        self._reset()

    def _reset(self) -> None:
        self.batch_count = 0
        self.sample_count = 0
        self.replay_count = 0
        self.role_losses: dict[str, list[float]] = defaultdict(list)
        self.role_probabilities: dict[str, list[float]] = defaultdict(list)
        self.role_labels: dict[str, int] = {}
        self.sample_values: dict[str, dict[str, Any]] = {}
        self.order_digest = hashlib.sha256()
        self.batch_size_digest = hashlib.sha256()
        self.augmentation_digest = hashlib.sha256()
        self.augmentation_value_sum = 0.0
        self.augmentation_value_sum_squares = 0.0
        self.augmentation_value_count = 0

    @property
    def recording(self) -> bool:
        return self.active_epoch is not None and self.training_active

    def _paths(self, epoch: int) -> tuple[Path, Path]:
        parquet = self.spec.output_dir / f"epoch_{epoch:04d}_process_telemetry.parquet"
        return parquet, parquet.with_suffix(".json")

    def start_epoch(self, epoch: int) -> None:
        if self.active_epoch is not None:
            raise ValidationError(f"process telemetry epoch {self.active_epoch} is already active")
        parquet, sidecar = self._paths(epoch)
        if parquet.exists() or sidecar.exists():
            validate_process_telemetry_epoch(self.spec, epoch)
            raise ValidationError(f"process telemetry epoch {epoch} is already complete")
        self._reset()
        self.active_epoch = int(epoch)
        self.training_active = True

    def validate_class_names(self, names: Mapping[int, str] | Mapping[str, str]) -> None:
        normalized = {int(key): str(value) for key, value in names.items()}
        expected = {0: "no_target", self.spec.target_defect_class_index: "target_defect"}
        for index, value in expected.items():
            if normalized.get(index) != value:
                raise ValidationError(
                    f"classification index {index} must resolve to {value}, got {normalized.get(index)!r}"
                )

    def record_batch(self, logits: Any, batch: Mapping[str, Any]) -> None:
        if self.active_epoch is None:
            return
        import torch
        import torch.nn.functional as functional

        if not torch.is_tensor(logits) or logits.ndim != 2:
            raise ValidationError("process telemetry requires a 2D classification logit tensor")
        for key in ("cls", "im_file", "img"):
            if key not in batch:
                raise ValidationError(f"process telemetry batch missing {key}")
        labels_tensor = batch["cls"].detach().long()
        paths = [str(value) for value in batch["im_file"]]
        if logits.shape[0] != labels_tensor.shape[0] or logits.shape[0] != len(paths):
            raise ValidationError("process telemetry batch dimensions do not agree")
        if logits.shape[1] <= self.spec.target_defect_class_index:
            raise ValidationError("classification logits omit target_defect class")

        labels = labels_tensor.cpu().numpy()
        identities: list[_Identity] = []
        monitored_indices: list[int] = []
        for index, path_text in enumerate(paths):
            path = Path(path_text)
            key = (path.parent.name, path.name)
            identity = self.lookup.get(key)
            if identity is None:
                raise ValidationError(
                    f"training image is not present in the frozen identity index: {key}"
                )
            if int(labels[index]) != identity.y_true:
                raise ValidationError(
                    f"batch label conflicts with frozen identity for {identity.sample_id}: "
                    f"{int(labels[index])} != {identity.y_true}"
                )
            identities.append(identity)
            if identity.monitor_groups or identity.exposure_role in _REPLAY_ROLES:
                monitored_indices.append(index)

        with torch.no_grad():
            detached = logits.detach().float()
            losses = functional.cross_entropy(detached, labels_tensor, reduction="none").cpu().numpy()
            probabilities = detached.softmax(dim=1)[:, self.spec.target_defect_class_index].cpu().numpy()
            images = batch["img"].detach().float()
            if images.ndim != 4 or images.shape[0] != logits.shape[0]:
                raise ValidationError("process telemetry requires BCHW image tensors")
            pooled = functional.adaptive_avg_pool2d(images.mean(dim=1, keepdim=True), (8, 8))
            pooled = pooled.clamp(0.0, 1.0).mul(255).round()
            moments = torch.stack((pooled.sum(), pooled.square().sum())).cpu().tolist()
            selected_fingerprints: dict[int, bytes] = {}
            if monitored_indices:
                selected = pooled[monitored_indices].to(torch.uint8).cpu().numpy()
                selected_fingerprints = {
                    source_index: bytes(selected[local_index].reshape(-1).tolist())
                    for local_index, source_index in enumerate(monitored_indices)
                }

        self.augmentation_value_sum += float(moments[0])
        self.augmentation_value_sum_squares += float(moments[1])
        self.augmentation_value_count += int(pooled.numel())

        _digest_update(self.order_digest, f"BATCH:{self.batch_count}")
        _digest_update(self.batch_size_digest, str(len(paths)))
        _digest_update(
            self.augmentation_digest,
            f"BATCH:{self.batch_count}",
            f"SUM:{float(moments[0]):.6f}",
            f"SUMSQ:{float(moments[1]):.6f}",
            f"COUNT:{int(pooled.numel())}",
        )
        self.batch_count += 1
        for index, identity in enumerate(identities):
            loss = float(losses[index])
            probability = float(probabilities[index])
            if not math.isfinite(loss) or not math.isfinite(probability):
                raise ValidationError("non-finite sample telemetry value")
            _digest_update(self.order_digest, identity.sample_id, identity.exposure_role)
            self.role_losses[identity.exposure_role].append(loss)
            self.role_probabilities[identity.exposure_role].append(probability)
            self.role_labels[identity.exposure_role] = identity.y_true
            self.sample_count += 1
            if identity.exposure_role in _REPLAY_ROLES:
                self.replay_count += 1
            if identity.monitor_groups or identity.exposure_role in _REPLAY_ROLES:
                fingerprint = selected_fingerprints[index]
                _digest_update(
                    self.augmentation_digest,
                    identity.sample_id,
                    identity.exposure_role,
                    fingerprint,
                )
                sample = self.sample_values.setdefault(
                    identity.sample_id,
                    {
                        "y_true": identity.y_true,
                        "monitor_groups": set(),
                        "roles": defaultdict(int),
                        "losses": [],
                        "probabilities": [],
                        "augmentation_digest": hashlib.sha256(),
                    },
                )
                if sample["y_true"] != identity.y_true:
                    raise ValidationError(f"sample label changed during epoch: {identity.sample_id}")
                if identity.monitor_groups:
                    sample["monitor_groups"].update(identity.monitor_groups.split(";"))
                sample["roles"][identity.exposure_role] += 1
                sample["losses"].append(loss)
                sample["probabilities"].append(probability)
                _digest_update(sample["augmentation_digest"], identity.exposure_role, fingerprint)

    def _rows(self, epoch: int) -> list[dict[str, Any]]:
        all_losses = [value for values in self.role_losses.values() for value in values]
        all_probabilities = [value for values in self.role_probabilities.values() for value in values]
        rows: list[dict[str, Any]] = []
        epoch_row = _blank_row(self.spec, epoch, "EPOCH")
        if self.augmentation_value_count <= 0:
            raise ValidationError("augmentation telemetry contains no values")
        augmentation_mean = self.augmentation_value_sum / self.augmentation_value_count
        augmentation_variance = max(
            0.0,
            self.augmentation_value_sum_squares / self.augmentation_value_count
            - augmentation_mean**2,
        )
        epoch_row.update(
            {
                "exposure_role": "ALL",
                "batch_count": self.batch_count,
                "total_exposure_count": self.sample_count,
                "base_exposure_count": len(self.role_losses.get("base_normal", []))
                + len(self.role_losses.get("base_defect", [])),
                "replay_normal_exposure_count": len(self.role_losses.get("normal_replay", [])),
                "guard_defect_exposure_count": len(self.role_losses.get("defect_guard", [])),
                "minibatch_order_sha256": self.order_digest.hexdigest().upper(),
                "batch_size_sequence_sha256": self.batch_size_digest.hexdigest().upper(),
                "augmentation_realization_sha256": self.augmentation_digest.hexdigest().upper(),
                "augmentation_pooled_mean": float(augmentation_mean),
                "augmentation_pooled_std": float(math.sqrt(augmentation_variance)),
                **_stats(all_losses, "loss"),
                **_stats(all_probabilities, "p_defect"),
            }
        )
        rows.append(epoch_row)
        for role in sorted(self.role_losses):
            role_row = _blank_row(self.spec, epoch, "ROLE")
            count = len(self.role_losses[role])
            role_row.update(
                {
                    "exposure_role": role,
                    "y_true": self.role_labels[role],
                    "total_exposure_count": count,
                    "base_exposure_count": count if role.startswith("base_") else 0,
                    "replay_normal_exposure_count": count if role == "normal_replay" else 0,
                    "guard_defect_exposure_count": count if role == "defect_guard" else 0,
                    **_stats(self.role_losses[role], "loss"),
                    **_stats(self.role_probabilities[role], "p_defect"),
                }
            )
            rows.append(role_row)
        for sample_id in sorted(self.sample_values):
            sample = self.sample_values[sample_id]
            roles = sample["roles"]
            sample_row = _blank_row(self.spec, epoch, "SAMPLE")
            sample_row.update(
                {
                    "exposure_role": "MONITORED_SAMPLE",
                    "sample_id": sample_id,
                    "y_true": sample["y_true"],
                    "monitor_groups": ";".join(sorted(sample["monitor_groups"])),
                    "total_exposure_count": sum(roles.values()),
                    "base_exposure_count": roles.get("base_normal", 0) + roles.get("base_defect", 0),
                    "replay_normal_exposure_count": roles.get("normal_replay", 0),
                    "guard_defect_exposure_count": roles.get("defect_guard", 0),
                    "augmentation_realization_sha256": sample["augmentation_digest"].hexdigest().upper(),
                    **_stats(sample["losses"], "loss"),
                    **_stats(sample["probabilities"], "p_defect"),
                }
            )
            rows.append(sample_row)
        return rows

    def finish_epoch(self, epoch: int) -> Path:
        if self.active_epoch != epoch:
            raise ValidationError(f"process telemetry active epoch {self.active_epoch} != {epoch}")
        if self.sample_count != self.spec.expected_epoch_samples:
            raise ValidationError(
                f"observed epoch samples {self.sample_count} != expected {self.spec.expected_epoch_samples}"
            )
        if self.replay_count != self.spec.expected_replay_samples:
            raise ValidationError(
                f"observed replay samples {self.replay_count} != expected {self.spec.expected_replay_samples}"
            )
        rows = self._rows(epoch)
        parquet, sidecar = self._paths(epoch)
        role_summary = parquet.with_name(f"epoch_{epoch:04d}_role_loss_summary.json")
        temporary = parquet.with_name(f".tmp-{uuid.uuid4().hex[:8]}.parquet")
        try:
            import polars as pl

            pl.DataFrame(rows, infer_schema_length=None).write_parquet(
                temporary,
                compression="zstd",
                statistics=True,
            )
            with temporary.open("r+b") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, parquet)
            if self.after_parquet_publish is not None:
                self.after_parquet_publish(epoch)
            role_rows = [row for row in rows if row.get("record_type") in {"EPOCH", "ROLE"}]
            atomic_write_json(
                role_summary,
                {
                    "schema_version": "stage1.process_role_loss_summary.v1",
                    "status": "COMPLETE",
                    "run_id": self.spec.run_id,
                    "arm_id": self.spec.arm_id,
                    "segment_id": self.spec.segment_id,
                    "epoch": epoch,
                    "roles": role_rows,
                },
                overwrite=False,
            )
            atomic_write_json(
                sidecar,
                {
                    "schema_version": TELEMETRY_SCHEMA,
                    "status": "COMPLETE",
                    "run_id": self.spec.run_id,
                    "arm_id": self.spec.arm_id,
                    "segment_id": self.spec.segment_id,
                    "epoch": epoch,
                    "row_count": len(rows),
                    "sample_record_count": len(self.sample_values),
                    "observed_epoch_samples": self.sample_count,
                    "observed_replay_samples": self.replay_count,
                    "identity_index_sha256": self.identity_index_sha256,
                    "parquet_sha256": sha256_file(parquet),
                    "role_summary_relpath": role_summary.name,
                    "role_summary_sha256": sha256_file(role_summary),
                },
                overwrite=False,
            )
        except Exception:
            temporary.unlink(missing_ok=True)
            parquet.unlink(missing_ok=True)
            role_summary.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
            raise
        self.last_finished_epoch = epoch
        self.active_epoch = None
        self.training_active = False
        self._reset()
        return parquet

    def on_train_epoch_start(self, trainer: Any) -> None:
        self.start_epoch(int(trainer.epoch) + 1)

    def on_train_epoch_end(self, _trainer: Any) -> None:
        if self.active_epoch is None:
            raise ValidationError("process telemetry has no active training epoch")
        self.training_active = False

    def on_fit_epoch_end(self, trainer: Any) -> None:
        epoch = int(trainer.epoch) + 1
        if self.active_epoch is None:
            if self.last_finished_epoch is not None and epoch == self.last_finished_epoch + 1:
                return
            raise ValidationError(f"unexpected fit callback without active telemetry epoch: {epoch}")
        self.finish_epoch(epoch)


class TelemetryCriterion:
    """Transparent criterion wrapper that observes detached per-sample CE values."""

    def __init__(self, original: Any, collector: ProcessTelemetryCollector):
        self.original = original
        self.collector = collector

    def __call__(self, preds: Any, batch: Mapping[str, Any]):
        result = self.original(preds, batch)
        if self.collector.recording:
            self.collector.record_batch(_extract_logits(preds), batch)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        # EMA/checkpoint copies must contain only the original training
        # criterion; the live collector is process state, never model state.
        import copy

        return copy.deepcopy(self.original, memo)


class ProcessTelemetryInstaller:
    """Install live telemetry after EMA construction without mutating model methods."""

    def __init__(self, collector: ProcessTelemetryCollector):
        self.collector = collector

    def on_train_start(self, trainer: Any) -> None:
        self.collector.validate_class_names(trainer.data["names"])
        model = trainer.model
        while hasattr(model, "module"):
            model = model.module
        criterion = getattr(model, "criterion", None)
        if isinstance(criterion, TelemetryCriterion):
            raise ValidationError("process telemetry criterion was installed more than once")
        if criterion is None:
            criterion = model.init_criterion()
        model.criterion = TelemetryCriterion(criterion, self.collector)


__all__ = [
    "ProcessTelemetryCollector",
    "ProcessTelemetryInstaller",
    "ProcessTelemetrySpec",
    "TELEMETRY_SCHEMA",
    "TelemetryCriterion",
    "validate_process_telemetry_epoch",
]
