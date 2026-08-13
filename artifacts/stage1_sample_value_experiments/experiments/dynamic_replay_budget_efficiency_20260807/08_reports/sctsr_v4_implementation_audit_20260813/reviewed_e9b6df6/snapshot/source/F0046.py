from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .errors import ErrorCode, SctsrError
from .rng_isolation import derive_counter_seed, replay_rng_domain


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
