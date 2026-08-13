from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from .errors import ErrorCode, SctsrError
from .rng_isolation import derive_counter_seed
from .serialization import stable_digest


@dataclass(frozen=True, slots=True)
class BaseEpochRngReceipt:
    training_seed: int
    epoch: int
    base_order_seed: int
    base_augmentation_seed: int
    sampler_type: str
    dataloader_type: str
    dataloader_reset: bool
    receipt_digest: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_counter_domain_base_loader(trainer: Any, *, training_seed: int, epoch: int) -> BaseEpochRngReceipt:
    """Reset the frozen loader onto independent order/augmentation domains.

    Ultralytics' stock loader creates its persistent iterator during trainer
    setup. A child constructed at E120 would otherwise restart that iterator's
    hidden generator at its initial state instead of continuing an auditable
    epoch identity. SCTSR deliberately rematerializes it at every epoch using
    two distinct counter-domain generators. The dataset wrapper then uses the
    sample-specific base-augmentation subdomain for each transform.
    """

    if not 1 <= int(epoch) <= 200 or int(training_seed) < 0:
        raise SctsrError(ErrorCode.CONFIGURATION_MISMATCH, "Base RNG epoch/seed identity is invalid")
    loader = getattr(trainer, "train_loader", None)
    dataset = getattr(loader, "dataset", None)
    if loader is None or dataset is None or not callable(getattr(dataset, "set_epoch_context", None)):
        raise SctsrError(
            ErrorCode.UPSTREAM_BINDING_FAILED,
            "Formal base loader must expose the SCTSR epoch-aware identity dataset",
        )
    sampler = getattr(loader, "sampler", None)
    if sampler is None or not hasattr(sampler, "generator"):
        raise SctsrError(
            ErrorCode.UPSTREAM_BINDING_FAILED,
            "Formal single-GPU base loader must expose a generator-bound random sampler",
        )
    reset = getattr(loader, "reset", None)
    if not callable(reset):
        raise SctsrError(
            ErrorCode.UPSTREAM_BINDING_FAILED,
            "Formal base loader must support deterministic persistent-iterator reset",
        )
    dataset.set_epoch_context(training_seed=training_seed, epoch=epoch)
    base_order_seed = derive_counter_seed("base_order", training_seed, epoch)
    base_augmentation_seed = derive_counter_seed("base_augmentation", training_seed, epoch)
    sampler.generator = torch.Generator().manual_seed(base_order_seed)
    loader.generator = torch.Generator().manual_seed(base_augmentation_seed)

    # Explicitly stop the iterator's old workers before replacing it. This is
    # a correctness action, not OOM recovery or a hidden resource downgrade.
    old_iterator = getattr(loader, "iterator", None)
    shutdown = getattr(old_iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    reset()
    payload = {
        "training_seed": int(training_seed),
        "epoch": int(epoch),
        "base_order_seed": base_order_seed,
        "base_augmentation_seed": base_augmentation_seed,
        "sampler_type": f"{type(sampler).__module__}.{type(sampler).__qualname__}",
        "dataloader_type": f"{type(loader).__module__}.{type(loader).__qualname__}",
        "dataloader_reset": True,
    }
    return BaseEpochRngReceipt(**payload, receipt_digest=stable_digest(payload))
