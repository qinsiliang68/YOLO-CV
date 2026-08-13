from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .identity_pool import (
    IdentityPool,
    IdentityRecord,
    T_SELECTION_SEMANTIC,
    make_pool,
    partition_five_groups,
)
from .random_controls import PoolBuildResult, build_r1_global_random, build_r2_matched_random
from .rng_isolation import derive_counter_seed
from .serialization import stable_digest
from .terminal_field_guard import TerminalFieldGuard

SYNTHETIC_BASE_DENOMINATOR = 2_000
SYNTHETIC_BASE_MANIFEST_SHA = "1" * 64
SYNTHETIC_SOURCE_MANIFEST_SHA = "2" * 64


class TinyClassifier(torch.nn.Module):
    """Small network with BatchNorm so replay state-isolation is testable."""

    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(4),
            torch.nn.SiLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


@dataclass(frozen=True, slots=True)
class SyntheticFixture:
    training_seed: int
    base_records: tuple[IdentityRecord, ...]
    base_ids: tuple[str, ...]
    record_by_id: dict[str, IdentityRecord]
    features: dict[str, torch.Tensor]
    t_pool: IdentityPool
    r1_result: PoolBuildResult
    r2_result: PoolBuildResult
    groups_by_pool: dict[str, dict[str, tuple[IdentityRecord, ...]]]

    @property
    def base_denominator(self) -> int:
        return SYNTHETIC_BASE_DENOMINATOR


def _counter_token(domain: str, seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{domain}\0{seed}\0{sample_id}".encode("utf-8")).hexdigest()


def _records() -> tuple[IdentityRecord, ...]:
    rows: list[IdentityRecord] = []
    buckets = ("EASY", "BOUNDARY", "FORGETTING", "PERSISTENT_ERROR")
    for index in range(SYNTHETIC_BASE_DENOMINATOR):
        y = index % 2
        fold = (index // 2) % 10
        group = f"bucket_{(index // 20) % 20:02d}"
        dynamic = buckets[(index // 40) % len(buckets)]
        rows.append(
            IdentityRecord(
                sample_id=f"SYN_{index:06d}",
                y_true=y,
                replay_role="NORMAL_REPLAY" if y == 0 else "DEFECT_GUARD_REPLAY",
                historical_dynamic_bucket=dynamic,
                oof_fold=fold,
                oof_group_id=group,
                group_source="numeric_filename_bucket",
                base_manifest_membership=True,
            )
        )
    return tuple(rows)


def _features(records: Sequence[IdentityRecord], seed: int) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for record in records:
        generator = torch.Generator().manual_seed(derive_counter_seed("synthetic_fixture", seed, 0, record.sample_id))
        center = 0.8 if record.y_true else -0.8
        # Some boundary overlap and deterministic ties are intentional.
        bucket_shift = {
            "EASY": 0.35,
            "BOUNDARY": -0.55,
            "FORGETTING": -0.15,
            "PERSISTENT_ERROR": -0.75,
        }[record.historical_dynamic_bucket]
        sign = 1.0 if record.y_true else -1.0
        tensor = torch.randn(4, generator=generator) * 0.18 + center + sign * bucket_shift
        output[record.sample_id] = tensor.float()
    # Two distinct identities deliberately share the exact same raw input.
    # This produces a real model-output probability tie without rounding or
    # rewriting predictions after inference.
    output["SYN_000001"] = output["SYN_000000"].clone()
    return output


def _choose_t(records: Sequence[IdentityRecord], seed: int) -> tuple[IdentityRecord, ...]:
    # Choose exactly 2.5% while spreading selections across exact R2 strata.
    by_stratum: dict[tuple[int, str, int, str], list[IdentityRecord]] = {}
    for record in records:
        by_stratum.setdefault(record.stratum(), []).append(record)
    ordered_strata = sorted(by_stratum)
    target = 50
    chosen: list[IdentityRecord] = []
    cursor = 0
    while len(chosen) < target:
        stratum = ordered_strata[cursor % len(ordered_strata)]
        candidates = sorted(by_stratum[stratum], key=lambda r: (_counter_token("T_STRESS", seed, r.sample_id), r.sample_id))
        rank = cursor // len(ordered_strata)
        if rank < len(candidates) - 1:  # retain at least one zero-overlap candidate
            chosen.append(candidates[rank])
        cursor += 1
    return tuple(chosen)


def build_synthetic_fixture(*, training_seed: int = 20260812) -> SyntheticFixture:
    base = _records()
    base_ids = tuple(record.sample_id for record in base)
    features = _features(base, training_seed)
    t_records = _choose_t(base, training_seed)
    t_pool = make_pool(
        pool_id="SYNTHETIC_T_STRESS",
        pool_role="T_STRESS",
        records=t_records,
        base_denominator=SYNTHETIC_BASE_DENOMINATOR,
        base_manifest_sha256=SYNTHETIC_BASE_MANIFEST_SHA,
        source_manifest_path="SYNTHETIC_T_CANONICAL",
        source_manifest_sha256=SYNTHETIC_SOURCE_MANIFEST_SHA,
        construction_seed=None,
        selection_semantic=T_SELECTION_SEMANTIC,
    )
    r1 = build_r1_global_random(
        base,
        base_denominator=SYNTHETIC_BASE_DENOMINATOR,
        base_manifest_sha256=SYNTHETIC_BASE_MANIFEST_SHA,
        source_manifest_sha256=SYNTHETIC_SOURCE_MANIFEST_SHA,
        selection_seed=training_seed + 1,
        t_ids={record.sample_id for record in t_pool.records},
    )
    raw_rows = []
    for record in base:
        raw_rows.append(
            {
                "sample_id": record.sample_id,
                "y_true": record.y_true,
                "replay_role": record.replay_role,
                "historical_dynamic_bucket": record.historical_dynamic_bucket,
                "oof_fold": record.oof_fold,
                "oof_group_id": record.oof_group_id,
                "group_source": record.group_source,
                "base_manifest_membership": record.base_manifest_membership,
                # Forbidden terminal fields deliberately coexist in the source row.
                "loss": 999.0,
                "RHO": 999.0,
                "future_epoch_outcome": "MUST_NOT_BE_ACCESSED",
            }
        )
    r2 = build_r2_matched_random(
        raw_rows,
        t_pool=t_pool,
        base_denominator=SYNTHETIC_BASE_DENOMINATOR,
        base_manifest_sha256=SYNTHETIC_BASE_MANIFEST_SHA,
        source_manifest_sha256=SYNTHETIC_SOURCE_MANIFEST_SHA,
        selection_seed=training_seed + 2,
        guard=TerminalFieldGuard(),
    )
    groups = {
        "T": partition_five_groups(t_pool, base_denominator=SYNTHETIC_BASE_DENOMINATOR),
        "R1": partition_five_groups(r1.pool, base_denominator=SYNTHETIC_BASE_DENOMINATOR),
        "R2": partition_five_groups(r2.pool, base_denominator=SYNTHETIC_BASE_DENOMINATOR),
    }
    return SyntheticFixture(
        training_seed=training_seed,
        base_records=base,
        base_ids=base_ids,
        record_by_id={record.sample_id: record for record in base},
        features=features,
        t_pool=t_pool,
        r1_result=r1,
        r2_result=r2,
        groups_by_pool=groups,
    )


def _augmentation(feature: torch.Tensor, *, domain: str, training_seed: int, epoch: int, token: str) -> tuple[torch.Tensor, str]:
    seed = derive_counter_seed(domain, training_seed, epoch, token)
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(feature.shape, generator=generator, dtype=feature.dtype) * 0.015
    scale = 0.98 + (seed % 41) / 1000.0
    augmented = feature * scale + noise
    digest = stable_digest({"domain": domain, "seed": seed, "token": token, "scale": scale})
    return augmented.reshape(1, 2, 2), digest


def make_base_loader(fixture: SyntheticFixture, *, epoch: int, batch_size: int = 128) -> list[dict[str, object]]:
    ordered = sorted(
        fixture.base_records,
        key=lambda record: (_counter_token(f"base_order:{epoch}", fixture.training_seed, record.sample_id), record.sample_id),
    )
    batches: list[dict[str, object]] = []
    for start in range(0, len(ordered), batch_size):
        records = ordered[start : start + batch_size]
        augmented = [
            _augmentation(
                fixture.features[record.sample_id],
                domain="base_augmentation",
                training_seed=fixture.training_seed,
                epoch=epoch,
                token=record.sample_id,
            )
            for record in records
        ]
        batches.append(
            {
                "images": torch.stack([item[0] for item in augmented]),
                "labels": torch.tensor([record.y_true for record in records], dtype=torch.long),
                "sample_ids": tuple(record.sample_id for record in records),
                "augmentation_digests": tuple(item[1] for item in augmented),
            }
        )
    return batches


def make_replay_provider(fixture: SyntheticFixture) -> Callable[[Sequence[str], int, int, int], dict[str, object]]:
    def provider(sample_ids: Sequence[str], epoch: int, step_index: int, training_seed: int) -> dict[str, object]:
        records = [fixture.record_by_id[str(sample_id)] for sample_id in sample_ids]
        augmented = [
            _augmentation(
                fixture.features[record.sample_id],
                domain="replay_augmentation",
                training_seed=training_seed,
                epoch=epoch,
                token=f"{step_index}:{record.sample_id}",
            )
            for record in records
        ]
        return {
            "images": torch.stack([item[0] for item in augmented]),
            "labels": torch.tensor([record.y_true for record in records], dtype=torch.long),
            "sample_ids": tuple(sample_ids),
            "augmentation_digests": tuple(item[1] for item in augmented),
        }
    return provider


def synthetic_split(fixture: SyntheticFixture, *, size: int = 160) -> tuple[IdentityRecord, ...]:
    # Pin the two distinct identities with identical feature bytes into every
    # evaluation split so tie-safe frontier handling is exercised reliably.
    if size < 2:
        raise ValueError("Synthetic evaluation split must contain the registered tie pair")
    pinned_ids = {"SYN_000000", "SYN_000001"}
    pinned = sorted((record for record in fixture.base_records if record.sample_id in pinned_ids), key=lambda row: row.sample_id)
    ordered = sorted(
        (record for record in fixture.base_records if record.sample_id not in pinned_ids),
        key=lambda row: (_counter_token("synthetic_split", fixture.training_seed, row.sample_id), row.sample_id),
    )
    return tuple(pinned + ordered[: size - len(pinned)])
