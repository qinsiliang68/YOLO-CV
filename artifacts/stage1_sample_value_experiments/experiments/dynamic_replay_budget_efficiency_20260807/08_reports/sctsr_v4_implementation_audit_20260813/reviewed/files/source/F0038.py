from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from .errors import ErrorCode, SctsrError


@dataclass
class BatchNormSnapshot:
    buffers: dict[str, dict[str, torch.Tensor | None]]

    def digest(self) -> str:
        h = hashlib.sha256()
        for module_name in sorted(self.buffers):
            h.update(module_name.encode("utf-8"))
            for name in ("running_mean", "running_var", "num_batches_tracked"):
                h.update(name.encode("utf-8"))
                value = self.buffers[module_name][name]
                if value is None:
                    h.update(b"NONE")
                else:
                    tensor = value.detach().cpu().contiguous()
                    h.update(str(tensor.dtype).encode("ascii"))
                    h.update(str(tuple(tensor.shape)).encode("ascii"))
                    h.update(tensor.numpy().tobytes())
        return h.hexdigest().upper()


def capture_batchnorm_buffers(model: torch.nn.Module) -> BatchNormSnapshot:
    buffers: dict[str, dict[str, torch.Tensor | None]] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, _BatchNorm):
            buffers[module_name] = {
                "running_mean": None if module.running_mean is None else module.running_mean.detach().clone(),
                "running_var": None if module.running_var is None else module.running_var.detach().clone(),
                "num_batches_tracked": None if module.num_batches_tracked is None else module.num_batches_tracked.detach().clone(),
            }
    return BatchNormSnapshot(buffers)


def restore_batchnorm_buffers(model: torch.nn.Module, snapshot: BatchNormSnapshot) -> None:
    modules = dict(model.named_modules())
    for module_name, values in snapshot.buffers.items():
        module = modules.get(module_name)
        if not isinstance(module, _BatchNorm):
            raise SctsrError(ErrorCode.BN_BUFFER_NOT_RESTORED, "BatchNorm module topology changed during replay", observed=module_name)
        for field_name, saved in values.items():
            current = getattr(module, field_name)
            if saved is None:
                if current is not None:
                    raise SctsrError(ErrorCode.BN_BUFFER_NOT_RESTORED, "BatchNorm buffer nullability changed", failing_field=f"{module_name}.{field_name}")
            else:
                if current is None:
                    raise SctsrError(ErrorCode.BN_BUFFER_NOT_RESTORED, "BatchNorm buffer disappeared", failing_field=f"{module_name}.{field_name}")
                current.copy_(saved.to(device=current.device, dtype=current.dtype))


@contextlib.contextmanager
def preserve_batchnorm_buffers(model: torch.nn.Module) -> Iterator[dict[str, str]]:
    before = capture_batchnorm_buffers(model)
    try:
        yield {"before_digest": before.digest()}
    finally:
        restore_batchnorm_buffers(model, before)
        after = capture_batchnorm_buffers(model)
        if after.digest() != before.digest():
            raise SctsrError(
                ErrorCode.BN_BUFFER_NOT_RESTORED,
                "BatchNorm running buffers were not restored byte-for-byte after replay",
                observed=after.digest(),
                expected=before.digest(),
            )
