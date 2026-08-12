"""Frozen seed derivation for the v3 campaign."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib


CONTRACT_ID = "stage1_dynamic_oof_replay_v3_20260809"
HISTORICAL_SEEDS = frozenset(
    {1829910790, 1385197443, 1343723075, 440790218, 1625332720, 1024521099, 216172108, 53303004}
)
V2_RESERVED_SEEDS = frozenset(
    {
        775599971, 982248093, 759103031, 1710119176, 1136302086, 1239936874, 1280388208, 247474179,
        2092795045, 683913534, 1438579979, 1652954292, 1455009725, 1628866060, 2114252210, 808380771,
        1194832389, 229020074, 1768572726, 1197081573, 657452570, 306190051, 710223846, 1296693113,
        1112440905, 1464755301, 367902785, 1661228750, 678316277, 605222362,
    }
)
HISTORICAL_AND_V2_RESERVED_SEEDS = HISTORICAL_SEEDS | V2_RESERVED_SEEDS


@dataclass(frozen=True)
class SeedRecord:
    seed_id: str
    seed_scope: str
    scope_index: int
    training_seed: int
    derivation_sha256: str


def _scope(scope: str, count: int, prefix: str, used: set[int]) -> list[SeedRecord]:
    rows: list[SeedRecord] = []
    derivation_index = 1
    while len(rows) < count:
        token = f"{CONTRACT_ID}|{scope}|{derivation_index}"
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        # The preregistration uses the leading 31 bits, not the low 31 bits of
        # the first 32-bit word.
        value = int(digest[:8], 16) >> 1
        derivation_index += 1
        if value == 0 or value in used:
            continue
        rows.append(SeedRecord(f"{prefix}{len(rows) + 1:03d}", scope, len(rows) + 1, value, digest.upper()))
        used.add(value)
    return rows


def build_seed_registry() -> tuple[SeedRecord, ...]:
    used = set(HISTORICAL_AND_V2_RESERVED_SEEDS)
    rows = [
        *_scope("DISCOVERY", 10, "D", used),
        *_scope("RANKING_ABLATION", 10, "A", used),
        *_scope("UNSEEN_CONFIRMATION", 14, "C", used),
    ]
    return tuple(rows)


__all__ = [
    "CONTRACT_ID",
    "HISTORICAL_AND_V2_RESERVED_SEEDS",
    "HISTORICAL_SEEDS",
    "SeedRecord",
    "V2_RESERVED_SEEDS",
    "build_seed_registry",
]
