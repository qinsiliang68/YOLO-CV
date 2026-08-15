from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .errors import ErrorCode, SctsrError
from .columnar import read_columnar, validate_columnar_file, write_zstd_parquet
from .rng_isolation import derive_counter_seed, replay_rng_domain
from .serialization import sha256_file, stable_digest


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode("ascii"))
    h.update(str(tuple(value.shape)).encode("ascii"))
    h.update(value.numpy().tobytes())
    return h.hexdigest().upper()


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    sample_id: str
    y_true: int
    source_path: str
    replay_role: str = "UNREGISTERED"
    oof_fold: int = -1
    oof_group_id: str = "UNREGISTERED"
    historical_dynamic_bucket: str = "UNREGISTERED"
    identity_group: str = "UNASSIGNED"
    oof_reference_probability: float | None = None
    oof_reference_reason: str = "REGISTERED_NOT_AVAILABLE"
    rho_candidate_signal: float | None = None
    rho_reason: str = "REGISTERED_NOT_REPORTED"


def _normalise_source_path(value: str) -> str:
    return str(value).replace("\\", "/").strip().casefold()


def _source_path_matches(observed: str, expected: str) -> bool:
    expected_token = _normalise_source_path(expected)
    if expected_token in {"", "unregistered", "not_applicable"}:
        return False
    observed_token = _normalise_source_path(observed)
    return (
        observed_token == expected_token
        or observed_token.endswith("/" + expected_token)
        or expected_token.endswith("/" + observed_token)
        # Historical hardlink staging intentionally rewrites the directory to
        # train/no_target or train/target_defect.  Filename plus the separately
        # checked binary label is the invariant identity bridge.
        or Path(observed_token).name == Path(expected_token).name
    )


class IdentityAugmentingDataset(torch.utils.data.Dataset):
    """Length-preserving wrapper that attaches canonical identity metadata.

    The wrapper never inserts replay samples into the base DataLoader. Replay
    access is by canonical identity through :meth:`replay_batch`.
    """

    def __init__(
        self,
        base_dataset: Any,
        identities: Sequence[DatasetIdentity],
        *,
        training_seed: int | None = None,
    ) -> None:
        if len(base_dataset) != len(identities):
            raise SctsrError(
                ErrorCode.DENOMINATOR_IDENTITY_MISMATCH,
                "Identity manifest length differs from upstream base dataset",
                observed=len(identities), expected=len(base_dataset),
            )
        raw_samples = getattr(base_dataset, "samples", None)
        if not isinstance(raw_samples, Sequence) or len(raw_samples) != len(base_dataset):
            raise SctsrError(
                ErrorCode.UPSTREAM_BINDING_FAILED,
                "Frozen Ultralytics classification dataset must expose one physical sample path and label per index",
            )
        by_filename_label: dict[tuple[str, int], DatasetIdentity] = {}
        for identity in identities:
            key = (Path(identity.source_path).name.casefold(), int(identity.y_true))
            if not key[0] or key in by_filename_label:
                raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "Identity manifest has an ambiguous filename/label binding", observed=key)
            by_filename_label[key] = identity
        aligned: list[DatasetIdentity] = []
        physical_paths: list[str] = []
        for raw_sample in raw_samples:
            if not isinstance(raw_sample, Sequence) or len(raw_sample) < 2:
                raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Classification dataset sample entry is malformed")
            physical_path, physical_label = str(raw_sample[0]), int(raw_sample[1])
            key = (Path(physical_path).name.casefold(), physical_label)
            try:
                identity = by_filename_label.pop(key)
            except KeyError as exc:
                raise SctsrError(
                    ErrorCode.PREDICTION_IDENTITY_MISMATCH,
                    "Physical classification sample is absent from the frozen identity manifest",
                    observed={"filename": key[0], "y_true": physical_label},
                    artifact_path=physical_path,
                ) from exc
            aligned.append(identity)
            physical_paths.append(physical_path)
        if by_filename_label:
            raise SctsrError(
                ErrorCode.PREDICTION_IDENTITY_MISMATCH,
                "Frozen identity manifest contains samples absent from the physical classification dataset",
                observed=sorted(by_filename_label)[:20],
            )
        self.base_dataset = base_dataset
        self.identities = tuple(aligned)
        self.physical_source_paths = tuple(physical_paths)
        self.index_by_id = {item.sample_id: index for index, item in enumerate(self.identities)}
        self.training_seed = training_seed
        self.epoch_context: int | None = None
        if len(self.index_by_id) != len(self.identities):
            raise SctsrError(ErrorCode.DUPLICATE_IDENTITY, "Base identity manifest contains duplicate sample IDs")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getattr__(self, name: str) -> Any:
        if name in {"base_dataset", "identities", "physical_source_paths", "index_by_id", "training_seed", "epoch_context"}:
            raise AttributeError(name)
        return getattr(self.base_dataset, name)

    def set_epoch_context(self, *, training_seed: int, epoch: int) -> None:
        if self.training_seed is not None and int(training_seed) != int(self.training_seed):
            raise SctsrError(
                ErrorCode.CONFIGURATION_MISMATCH,
                "Dataset augmentation seed identity differs from the prepared trainer",
                observed=training_seed,
                expected=self.training_seed,
            )
        if not 1 <= int(epoch) <= 200:
            raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Dataset epoch context is outside E1-E200")
        self.training_seed = int(training_seed)
        self.epoch_context = int(epoch)

    def _materialize(
        self,
        index: int,
        *,
        domain: str,
        epoch: int | None = None,
        token: str | None = None,
        training_seed: int | None = None,
    ) -> dict[str, Any]:
        identity = self.identities[index]
        effective_seed = self.training_seed if training_seed is None else int(training_seed)
        effective_epoch = self.epoch_context if epoch is None else int(epoch)
        if effective_seed is None or effective_epoch is None:
            # Unit-only compatibility path. Formal evidence rejects unbound
            # contexts before it reaches the training loop.
            augmentation_seed = int(torch.initial_seed())
            raw = self.base_dataset[index]
            augmentation_domain = "UNBOUND_SYNTHETIC_UNIT_ONLY"
        else:
            materialization_token = identity.sample_id if token is None else token
            augmentation_seed = derive_counter_seed(domain, effective_seed, effective_epoch, materialization_token)
            with replay_rng_domain(domain, effective_seed, effective_epoch, materialization_token):
                raw = self.base_dataset[index]
            augmentation_domain = domain
        if not isinstance(raw, Mapping):
            raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Ultralytics classification dataset must return a mapping")
        item = dict(raw)
        observed_source = self.physical_source_paths[index]
        if not _source_path_matches(observed_source, identity.source_path):
            raise SctsrError(
                ErrorCode.PREDICTION_IDENTITY_MISMATCH,
                "Dataset source path differs from frozen identity manifest",
                observed=observed_source, expected=identity.source_path, artifact_path=identity.source_path,
            )
        label = int(torch.as_tensor(item.get("cls", identity.y_true)).reshape(-1)[0].item())
        if label != identity.y_true:
            raise SctsrError(
                ErrorCode.PREDICTION_IDENTITY_MISMATCH,
                "Dataset label differs from frozen identity manifest",
                observed=label, expected=identity.y_true, artifact_path=identity.source_path,
            )
        item["sample_id"] = identity.sample_id
        item["sample_ids"] = identity.sample_id
        item["source_path"] = identity.source_path
        item["replay_role"] = identity.replay_role
        item["oof_fold"] = identity.oof_fold
        item["oof_group_id"] = identity.oof_group_id
        item["historical_dynamic_bucket"] = identity.historical_dynamic_bucket
        item["identity_group"] = identity.identity_group
        item["augmentation_seed"] = str(augmentation_seed)
        item["augmentation_seeds"] = str(augmentation_seed)
        item["augmentation_domain"] = augmentation_domain
        if "img" in item and isinstance(item["img"], torch.Tensor):
            item["augmentation_digest"] = tensor_digest(item["img"])
            item["augmentation_digests"] = item["augmentation_digest"]
        return item

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._materialize(index, domain="base_augmentation")

    def replay_batch(
        self,
        sample_ids: Sequence[str],
        *,
        epoch: int | None = None,
        base_step_index: int = 0,
        training_seed: int | None = None,
    ) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        for sample_id in sample_ids:
            try:
                index = self.index_by_id[str(sample_id)]
            except KeyError as exc:
                raise SctsrError(ErrorCode.IDENTITY_NOT_IN_BASE, "Replay identity is not in canonical base", observed=sample_id) from exc
            items.append(
                self._materialize(
                    index,
                    domain="replay_augmentation",
                    epoch=epoch,
                    token=f"{base_step_index}:{sample_id}",
                    training_seed=training_seed,
                )
            )
        if not items:
            return {"img": torch.empty((0,)), "cls": torch.empty((0,), dtype=torch.long), "sample_ids": (), "augmentation_digests": ()}
        images = torch.stack([torch.as_tensor(item["img"]) for item in items])
        labels = torch.as_tensor([int(torch.as_tensor(item["cls"]).reshape(-1)[0]) for item in items], dtype=torch.long)
        return {
            "img": images,
            "cls": labels,
            "sample_ids": tuple(str(item["sample_id"]) for item in items),
            "source_paths": tuple(str(item.get("source_path", "UNREGISTERED")) for item in items),
            "replay_roles": tuple(str(item.get("replay_role", "UNREGISTERED")) for item in items),
            "oof_folds": tuple(int(item.get("oof_fold", -1)) for item in items),
            "oof_group_ids": tuple(str(item.get("oof_group_id", "UNREGISTERED")) for item in items),
            "historical_dynamic_buckets": tuple(str(item.get("historical_dynamic_bucket", "UNREGISTERED")) for item in items),
            "identity_groups": tuple(str(item.get("identity_group", "UNASSIGNED")) for item in items),
            "augmentation_digests": tuple(str(item.get("augmentation_digest", "NOT_RECORDED")) for item in items),
            "augmentation_seeds": tuple(int(item["augmentation_seed"]) for item in items),
        }


class IdentityFilteringDataset(torch.utils.data.Dataset):
    """Expose exactly one registered evaluation identity set from an upstream dataset.

    The upstream classification directory may still contain historical rows
    excluded by the SCTSR content-disjointness overlay. This wrapper filters
    those rows before any evaluation batch is constructed and preserves the
    upstream transform and dataset attributes through delegation.
    """

    def __init__(self, base_dataset: Any, identities: Sequence[DatasetIdentity]) -> None:
        raw_samples = getattr(base_dataset, "samples", None)
        if not isinstance(raw_samples, Sequence) or len(raw_samples) != len(base_dataset):
            raise SctsrError(
                ErrorCode.UPSTREAM_BINDING_FAILED,
                "Frozen validation dataset must expose one physical sample path and label per index",
            )
        expected: dict[tuple[str, int], DatasetIdentity] = {}
        for identity in identities:
            key = (Path(identity.source_path).name.casefold(), int(identity.y_true))
            if not key[0] or key in expected:
                raise SctsrError(
                    ErrorCode.DUPLICATE_IDENTITY,
                    "Registered validation identities have an ambiguous filename/label binding",
                    observed=key,
                )
            expected[key] = identity
        selected: list[tuple[int, DatasetIdentity, str, tuple[Any, ...]]] = []
        for index, raw_sample in enumerate(raw_samples):
            if not isinstance(raw_sample, Sequence) or len(raw_sample) < 2:
                raise SctsrError(ErrorCode.UPSTREAM_BINDING_FAILED, "Classification validation sample is malformed")
            physical_path, physical_label = str(raw_sample[0]), int(raw_sample[1])
            key = (Path(physical_path).name.casefold(), physical_label)
            identity = expected.pop(key, None)
            if identity is None:
                continue
            if not _source_path_matches(physical_path, identity.source_path):
                raise SctsrError(
                    ErrorCode.PREDICTION_IDENTITY_MISMATCH,
                    "Validation physical path differs from the registered identity",
                    observed=physical_path,
                    expected=identity.source_path,
                )
            selected.append((index, identity, physical_path, tuple(raw_sample)))
        if expected:
            raise SctsrError(
                ErrorCode.PREDICTION_IDENTITY_MISMATCH,
                "Registered validation identities are absent from the physical classification dataset",
                observed=sorted(expected)[:20],
            )
        if not selected:
            raise SctsrError(ErrorCode.ASSET_VALIDATION_FAILED, "Effective validation dataset is empty")
        self.base_dataset = base_dataset
        self._base_indices = tuple(row[0] for row in selected)
        self.identities = tuple(row[1] for row in selected)
        self.physical_source_paths = tuple(row[2] for row in selected)
        self.samples = tuple(row[3] for row in selected)

    def __len__(self) -> int:
        return len(self._base_indices)

    def __getitem__(self, index: int) -> Any:
        return self.base_dataset[self._base_indices[index]]

    def __getattr__(self, name: str) -> Any:
        if name in {"base_dataset", "_base_indices", "identities", "physical_source_paths", "samples"}:
            raise AttributeError(name)
        return getattr(self.base_dataset, name)


def validate_materialized_dataset_bytes(
    dataset: Any,
    expected_content: Mapping[str, Mapping[str, Any]],
    *,
    role: str,
    dataset_root: str | Path | None = None,
    evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    """Hash the exact physical files exposed by an upstream dataset."""

    identities = getattr(dataset, "identities", None)
    physical_paths = getattr(dataset, "physical_source_paths", None)
    if (
        not isinstance(identities, Sequence)
        or not isinstance(physical_paths, Sequence)
        or len(identities) != len(physical_paths)
    ):
        raise SctsrError(
            ErrorCode.UPSTREAM_BINDING_FAILED,
            "Prepared dataset lacks auditable physical identity paths",
            observed=role,
        )
    evidence: list[dict[str, Any]] = []
    total_bytes = 0
    canonical_root = None if dataset_root is None else Path(dataset_root).resolve()
    selected_paths: set[Path] = set()
    scan_roots: set[Path] = set()
    for identity, raw_path in zip(identities, physical_paths, strict=True):
        sample_id = str(identity.sample_id)
        expected = expected_content.get(sample_id)
        path = Path(str(raw_path)).resolve()
        if canonical_root is not None and path != canonical_root and canonical_root not in path.parents:
            raise SctsrError(
                ErrorCode.DATASET_CONTENT_MISMATCH,
                "Materialized loader path escapes the registered dataset root",
                artifact_path=str(path),
            )
        if expected is None or not path.is_file():
            raise SctsrError(
                ErrorCode.DATASET_CONTENT_MISMATCH,
                "Materialized dataset identity or physical file is absent from the frozen content ledger",
                observed=sample_id,
                artifact_path=str(path),
            )
        observed_bytes = path.stat().st_size
        observed_sha = sha256_file(path)
        expected_bytes = int(expected["image_bytes"])
        expected_sha = str(expected["image_sha256"]).upper()
        if observed_bytes != expected_bytes or observed_sha != expected_sha:
            raise SctsrError(
                ErrorCode.DATASET_CONTENT_MISMATCH,
                "Materialized dataset bytes differ from the frozen content ledger",
                observed={"bytes": observed_bytes, "sha256": observed_sha},
                expected={"bytes": expected_bytes, "sha256": expected_sha},
                artifact_path=str(path),
            )
        total_bytes += observed_bytes
        selected_paths.add(path)
        scan_roots.add(path.parent)
        canonical_path = None if canonical_root is None else (canonical_root / Path(sample_id)).resolve()
        if canonical_path is not None:
            try:
                canonical_path.relative_to(canonical_root)
            except ValueError as exc:
                raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Canonical materialized identity escapes the registered dataset root", observed=sample_id) from exc
            if not canonical_path.is_file() or canonical_path.stat().st_size != expected_bytes or sha256_file(canonical_path) != expected_sha:
                raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Canonical source bytes differ from the frozen content ledger", observed=sample_id, artifact_path=str(canonical_path))
        stat_result = path.stat()
        evidence.append(
            {
                "role": role,
                "sample_id": sample_id,
                "y_true": int(identity.y_true),
                "loader_path": str(raw_path),
                "loader_path_resolved": path.as_posix(),
                "canonical_source_path": "NOT_BOUND" if canonical_path is None else canonical_path.as_posix(),
                "image_bytes": observed_bytes,
                "image_sha256": observed_sha,
                "physical_file_identity": f"{stat_result.st_dev}:{stat_result.st_ino}",
                "samefile_as_canonical": False if canonical_path is None else os.path.samefile(path, canonical_path),
            }
        )
    for selected in selected_paths:
        current = selected
        while True:
            stat_result = current.lstat()
            attributes = int(getattr(stat_result, "st_file_attributes", 0))
            if current.is_symlink() or attributes & 0x400:
                raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Materialized dataset path uses a symlink or reparse point", artifact_path=str(current))
            if canonical_root is None or current == canonical_root or canonical_root not in current.parents:
                break
            current = current.parent
    for scan_root in scan_roots:
        for entry in scan_root.rglob("*"):
            resolved = entry.resolve()
            attributes = int(getattr(entry.lstat(), "st_file_attributes", 0))
            if entry.is_symlink() or attributes & 0x400:
                raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Materialized dataset contains a symlink or reparse point", artifact_path=str(entry))
            if entry.is_file() and resolved not in selected_paths:
                raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Materialized dataset contains an unregistered extra file", artifact_path=str(entry))
            if entry.is_dir() and not any(resolved in selected.parents for selected in selected_paths):
                raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Materialized dataset contains an unregistered extra directory", artifact_path=str(entry))
    payload = {
        "schema_version": "stage1.sctsr.materialized_dataset_binding.v2",
        "status": "PASS",
        "role": role,
        "row_count": len(evidence),
        "physical_bytes_verified": total_bytes,
        "dataset_root": "NOT_BOUND" if canonical_root is None else canonical_root.as_posix(),
        "materialized_roots": sorted(path.as_posix() for path in scan_roots),
        "materialized_content_digest": stable_digest(
            sorted(evidence, key=lambda row: row["sample_id"])
        ),
    }
    if evidence_path is not None:
        manifest = write_zstd_parquet(
            sorted(evidence, key=lambda row: row["sample_id"]),
            evidence_path,
            schema_version="stage1.sctsr.materialized_dataset_rows.v2",
        )
        payload["evidence"] = {
            "path": Path(manifest.path).resolve().as_posix(),
            "bytes": manifest.bytes,
            "sha256": manifest.sha256,
            "row_count": manifest.row_count,
            "schema_version": manifest.schema_version,
            "schema_digest": manifest.schema_digest,
            "compression": manifest.compression,
        }
    return {**payload, "binding_digest": stable_digest(payload)}


def revalidate_materialized_dataset_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Re-hash every prepared loader file from its immutable row ledger."""

    if binding.get("schema_version") != "stage1.sctsr.materialized_dataset_binding.v2":
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Materialized dataset binding schema is not registered")
    core = {key: value for key, value in binding.items() if key != "binding_digest"}
    if binding.get("binding_digest") != stable_digest(core):
        raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Materialized dataset binding digest is invalid")
    evidence = binding.get("evidence")
    if not isinstance(evidence, Mapping):
        raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Materialized dataset binding lacks row-level evidence")
    path = Path(str(evidence.get("path", ""))).resolve()
    validate_columnar_file(
        path,
        expected_rows=int(evidence.get("row_count", -1)),
        expected_schema_version=str(evidence.get("schema_version", "")),
        expected_schema_digest=str(evidence.get("schema_digest", "")),
        expected_sha256=str(evidence.get("sha256", "")),
    )
    rows = read_columnar(path)
    observed: list[dict[str, Any]] = []
    for row in rows:
        physical = Path(str(row["loader_path_resolved"])).resolve()
        if not physical.is_file():
            raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "A bound materialized dataset file is missing", artifact_path=str(physical))
        stat_result = physical.stat()
        observed_row = dict(row)
        observed_row["image_bytes"] = stat_result.st_size
        observed_row["image_sha256"] = sha256_file(physical)
        observed_row["physical_file_identity"] = f"{stat_result.st_dev}:{stat_result.st_ino}"
        canonical = str(row.get("canonical_source_path", "NOT_BOUND"))
        observed_row["samefile_as_canonical"] = False if canonical == "NOT_BOUND" else os.path.samefile(physical, Path(canonical))
        if observed_row != row:
            raise SctsrError(
                ErrorCode.DATASET_CONTENT_MISMATCH,
                "Materialized dataset bytes or physical file identity changed after setup",
                observed={key: observed_row[key] for key in ("sample_id", "image_bytes", "image_sha256", "physical_file_identity")},
                expected={key: row[key] for key in ("sample_id", "image_bytes", "image_sha256", "physical_file_identity")},
                artifact_path=str(physical),
            )
        observed.append(observed_row)
    if stable_digest(sorted(observed, key=lambda row: row["sample_id"])) != binding.get("materialized_content_digest"):
        raise SctsrError(ErrorCode.DATASET_CONTENT_MISMATCH, "Materialized dataset aggregate digest changed after setup")
    return {"status": "PASS", "role": binding["role"], "row_count": len(observed), "binding_digest": binding["binding_digest"]}


def load_identity_manifest(path: str | Path) -> tuple[DatasetIdentity, ...]:
    import csv
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    def optional_float(value: Any) -> float | None:
        token = "" if value is None else str(value).strip()
        return None if token == "" else float(token)

    result = []
    for row in rows:
        result.append(DatasetIdentity(
            sample_id=str(row["sample_id"]),
            y_true=int(row.get("y_true", row.get("label", 0))),
            source_path=str(row.get("source_path", row.get("im_file", "UNREGISTERED"))),
            replay_role=str(row.get("replay_role", "UNREGISTERED")),
            oof_fold=int(row.get("oof_fold", -1)),
            oof_group_id=str(row.get("oof_group_id", "UNREGISTERED")),
            historical_dynamic_bucket=str(row.get("historical_dynamic_bucket", row.get("dynamic_bucket", "UNREGISTERED"))),
            identity_group=str(row.get("identity_group", "UNASSIGNED")),
            oof_reference_probability=optional_float(row.get("oof_reference_probability")),
            oof_reference_reason=str(row.get("oof_reference_reason") or "REGISTERED_NOT_AVAILABLE"),
            rho_candidate_signal=optional_float(row.get("rho_candidate_signal")),
            rho_reason=str(row.get("rho_reason") or "REGISTERED_NOT_REPORTED"),
        ))
    return tuple(result)
