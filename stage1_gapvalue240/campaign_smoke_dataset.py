"""Immutable tiny real-data input package for local campaign engineering tests."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import uuid

import pandas as pd

from .errors import ValidationError
from .util import atomic_write_json, sha256_file, stable_hash


class LocalSmokeDatasetError(ValidationError):
    """Raised when the local real-data smoke subset cannot be frozen safely."""


@dataclass(frozen=True)
class LocalSmokeDataset:
    root: Path
    dataset_dir: Path
    base_normal_manifest: Path
    base_defect_manifest: Path
    replay_identity_manifest: Path
    monitor_manifest: Path
    expected_epoch_samples: int
    expected_replay_samples: int
    expected_steps_per_epoch: int
    validation_path: Path


_MANIFESTS = {
    "base_normal": "normal_train_manifest.csv",
    "base_defect": "train_manifest.csv",
    "val_normal": "normal_val_model_manifest.csv",
    "val_defect": "val_model_manifest.csv",
}


def _selected_manifest(
    dataset_root: Path,
    source: Path,
    *,
    rows: int,
    label: str,
) -> pd.DataFrame:
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source, keep_default_na=False)
    required = {"canonical_image_relpath", "Filename"}
    missing = required - set(frame.columns)
    if missing:
        raise LocalSmokeDatasetError(f"{label} source manifest missing columns: {sorted(missing)}")
    if len(frame) < rows:
        raise LocalSmokeDatasetError(f"{label} requires {rows} rows, source has {len(frame)}")
    selected = frame.head(rows).copy()
    if selected.canonical_image_relpath.astype(str).duplicated().any():
        raise LocalSmokeDatasetError(f"{label} contains duplicate sample identity")
    if selected.Filename.astype(str).duplicated().any():
        raise LocalSmokeDatasetError(f"{label} contains duplicate filename")
    for row in selected.itertuples(index=False):
        relative = Path(str(row.canonical_image_relpath).replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise LocalSmokeDatasetError(f"{label} contains unsafe image path: {relative}")
        source_image = (dataset_root / relative).resolve()
        try:
            source_image.relative_to(dataset_root)
        except ValueError as exc:
            raise LocalSmokeDatasetError(f"{label} image escapes dataset root: {relative}") from exc
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        if Path(str(row.Filename)).name != str(row.Filename):
            raise LocalSmokeDatasetError(f"{label} contains unsafe filename: {row.Filename}")
    return selected


def _link_rows(dataset_root: Path, frame: pd.DataFrame, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for row in frame.itertuples(index=False):
        source = (dataset_root / str(row.canonical_image_relpath)).resolve()
        target = destination / str(row.Filename)
        os.link(source, target)


def prepare_local_smoke_dataset(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    train_per_class: int,
    val_per_class: int,
    replay_normal: int,
    batch: int,
) -> LocalSmokeDataset:
    """Freeze one small real-image dataset shared by smoke and failure tests."""

    source_root = Path(dataset_root).resolve()
    output = Path(output_root).resolve()
    if min(train_per_class, val_per_class, replay_normal, batch) <= 0:
        raise LocalSmokeDatasetError("smoke subset sizes and batch must be positive")
    if replay_normal > train_per_class:
        raise LocalSmokeDatasetError("normal replay count exceeds the selected normal base pool")
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"local smoke dataset target is not empty: {output}")

    source_manifests = source_root / "manifests"
    selected = {
        role: _selected_manifest(
            source_root,
            source_manifests / name,
            rows=train_per_class if role.startswith("base_") else val_per_class,
            label=role,
        )
        for role, name in _MANIFESTS.items()
    }
    # A short random suffix keeps deep registered artifact roots usable on
    # Windows hosts that still enforce the legacy 260-character path limit.
    staging = output.parent / f".{output.name}.{uuid.uuid4().hex[:8]}.tmpdir"
    try:
        inputs = staging / "inputs"
        inputs.mkdir(parents=True)
        base_normal = inputs / "base_normal.csv"
        base_defect = inputs / "base_defect.csv"
        selected["base_normal"].to_csv(base_normal, index=False, lineterminator="\n")
        selected["base_defect"].to_csv(base_defect, index=False, lineterminator="\n")

        dataset = staging / "dataset"
        _link_rows(source_root, selected["base_normal"], dataset / "train/no_target")
        _link_rows(source_root, selected["base_defect"], dataset / "train/target_defect")
        _link_rows(source_root, selected["val_normal"], dataset / "val/no_target")
        _link_rows(source_root, selected["val_defect"], dataset / "val/target_defect")

        replay_rows: list[dict[str, object]] = []
        for rank, row in enumerate(
            selected["base_normal"].head(replay_normal).itertuples(index=False), start=1
        ):
            staged_name = f"replay__LOCAL_SMOKE__{rank:05d}__{row.Filename}"
            os.link(
                source_root / str(row.canonical_image_relpath),
                dataset / "train/no_target" / staged_name,
            )
            replay_rows.append(
                {
                    "staged_filename": staged_name,
                    "sample_id": str(row.canonical_image_relpath),
                    "y_true": 0,
                    "replay_role": "normal_replay",
                }
            )
        replay_identity = inputs / "replay_identity.csv"
        pd.DataFrame(replay_rows).to_csv(replay_identity, index=False, lineterminator="\n")
        monitor_rows = [
            *(
                {
                    "sample_id": str(row.canonical_image_relpath),
                    "monitor_group": "LOCAL_SMOKE_NORMAL",
                }
                for row in selected["base_normal"].head(replay_normal).itertuples(index=False)
            ),
            *(
                {
                    "sample_id": str(row.canonical_image_relpath),
                    "monitor_group": "LOCAL_SMOKE_DEFECT",
                }
                for row in selected["base_defect"].head(replay_normal).itertuples(index=False)
            ),
        ]
        monitor = inputs / "monitor.csv"
        pd.DataFrame(monitor_rows).to_csv(monitor, index=False, lineterminator="\n")

        expected_samples = train_per_class * 2 + replay_normal
        expected_steps = (expected_samples + batch - 1) // batch
        validation = staging / "SMOKE_DATASET_VALIDATION.json"
        atomic_write_json(
            validation,
            {
                "schema_version": "stage1.local_smoke_dataset.v1",
                "status": "PASS",
                "source_dataset_root": str(source_root),
                "source_manifest_sha256": {
                    role: sha256_file(source_manifests / name)
                    for role, name in _MANIFESTS.items()
                },
                "train_per_class": train_per_class,
                "val_per_class": val_per_class,
                "replay_normal": replay_normal,
                "batch": batch,
                "expected_epoch_samples": expected_samples,
                "expected_replay_samples": replay_normal,
                "expected_steps_per_epoch": expected_steps,
                "base_normal_manifest_sha256": sha256_file(base_normal),
                "base_defect_manifest_sha256": sha256_file(base_defect),
                "replay_identity_manifest_sha256": sha256_file(replay_identity),
                "monitor_manifest_sha256": sha256_file(monitor),
                "selected_identity_sha256": stable_hash(
                    [
                        {
                            "role": role,
                            "sample_ids": frame.canonical_image_relpath.astype(str).tolist(),
                        }
                        for role, frame in selected.items()
                    ]
                ),
            },
        )
        if output.exists():
            output.rmdir()
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return LocalSmokeDataset(
        root=output,
        dataset_dir=output / "dataset",
        base_normal_manifest=output / "inputs/base_normal.csv",
        base_defect_manifest=output / "inputs/base_defect.csv",
        replay_identity_manifest=output / "inputs/replay_identity.csv",
        monitor_manifest=output / "inputs/monitor.csv",
        expected_epoch_samples=expected_samples,
        expected_replay_samples=replay_normal,
        expected_steps_per_epoch=expected_steps,
        validation_path=output / "SMOKE_DATASET_VALIDATION.json",
    )


__all__ = [
    "LocalSmokeDataset",
    "LocalSmokeDatasetError",
    "prepare_local_smoke_dataset",
]
