from __future__ import annotations

import contextlib
import hashlib
import pickle
import random
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from .errors import ErrorCode, SctsrError


@dataclass
class RngSnapshot:
    python_state: object
    numpy_state: tuple
    torch_cpu_state: torch.Tensor
    torch_cuda_states: tuple[torch.Tensor, ...]

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(pickle.dumps(self.python_state, protocol=5))
        h.update(pickle.dumps(self.numpy_state, protocol=5))
        h.update(self.torch_cpu_state.detach().cpu().contiguous().numpy().tobytes())
        for state in self.torch_cuda_states:
            h.update(state.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest().upper()


def capture_global_rng() -> RngSnapshot:
    cuda_states: tuple[torch.Tensor, ...]
    if torch.cuda.is_available():
        cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    else:
        cuda_states = ()
    return RngSnapshot(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_cpu_state=torch.get_rng_state().clone(),
        torch_cuda_states=cuda_states,
    )


def restore_global_rng(snapshot: RngSnapshot) -> None:
    random.setstate(snapshot.python_state)
    np.random.set_state(snapshot.numpy_state)
    torch.set_rng_state(snapshot.torch_cpu_state)
    if snapshot.torch_cuda_states:
        if not torch.cuda.is_available() or len(snapshot.torch_cuda_states) != torch.cuda.device_count():
            raise SctsrError(
                ErrorCode.RNG_NOT_RESTORED,
                "CUDA RNG topology differs from the captured snapshot",
                observed=torch.cuda.device_count() if torch.cuda.is_available() else 0,
                expected=len(snapshot.torch_cuda_states),
            )
        torch.cuda.set_rng_state_all(list(snapshot.torch_cuda_states))


def derive_counter_seed(domain: str, training_seed: int, epoch: int, optional_sample_or_step: object = "") -> int:
    payload = f"{domain}\0{training_seed}\0{epoch}\0{optional_sample_or_step}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


@contextlib.contextmanager
def replay_rng_domain(domain: str, training_seed: int, epoch: int, token: object = "") -> Iterator[dict[str, str | int]]:
    before = capture_global_rng()
    seed64 = derive_counter_seed(domain, training_seed, epoch, token)
    seed32 = seed64 % (2**32)
    random.seed(seed64)
    np.random.seed(seed32)
    torch.manual_seed(seed64 % (2**63 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed64 % (2**63 - 1))
    fork = capture_global_rng()
    try:
        yield {"seed": seed64, "before_digest": before.digest(), "fork_digest": fork.digest()}
    finally:
        restore_global_rng(before)
        after = capture_global_rng()
        if after.digest() != before.digest():
            raise SctsrError(
                ErrorCode.RNG_NOT_RESTORED,
                "Global RNG state was not restored after replay",
                observed=after.digest(),
                expected=before.digest(),
            )
