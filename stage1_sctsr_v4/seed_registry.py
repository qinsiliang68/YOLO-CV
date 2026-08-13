from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .errors import ErrorCode, SctsrError
from .serialization import stable_digest


@dataclass(frozen=True, slots=True)
class SeedRegistry:
    schema_version: str
    state: str
    historical_training_seeds: tuple[int, ...]
    selection_seeds: tuple[int, ...]
    discovery_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    required_discovery_count: int = 8
    required_confirmation_count: int = 14

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SeedRegistry":
        expected = {
            "schema_version",
            "state",
            "historical_training_seeds",
            "selection_seeds",
            "discovery_seeds",
            "confirmation_seeds",
            "required_counts_when_formal",
        }
        if set(value) != expected:
            raise SctsrError(
                ErrorCode.SEED_REGISTRY_INVALID,
                "Seed registry fields do not exactly match the registered schema",
                observed={"missing": sorted(expected - set(value)), "extra": sorted(set(value) - expected)},
            )

        def values(name: str) -> tuple[int, ...]:
            raw = value[name]
            if not isinstance(raw, list) or any(type(seed) is not int for seed in raw):
                raise SctsrError(ErrorCode.SEED_REGISTRY_INVALID, f"{name} must be a JSON array of integer seed IDs")
            return tuple(raw)

        counts = value["required_counts_when_formal"]
        if not isinstance(counts, Mapping) or set(counts) != {"discovery", "confirmation"}:
            raise SctsrError(ErrorCode.SEED_REGISTRY_INVALID, "Formal seed count contract is malformed")
        if any(type(counts[name]) is not int for name in ("discovery", "confirmation")):
            raise SctsrError(ErrorCode.SEED_REGISTRY_INVALID, "Formal seed counts must be integers")
        return cls(
            schema_version=str(value["schema_version"]),
            state=str(value["state"]),
            historical_training_seeds=values("historical_training_seeds"),
            selection_seeds=values("selection_seeds"),
            discovery_seeds=values("discovery_seeds"),
            confirmation_seeds=values("confirmation_seeds"),
            required_discovery_count=int(counts["discovery"]),
            required_confirmation_count=int(counts["confirmation"]),
        )

    @property
    def digest(self) -> str:
        return stable_digest(self)

    def validate(
        self,
        *,
        formal: bool = False,
        release_authorization_verified: bool = False,
        expected_registry_digest: str | None = None,
    ) -> None:
        if self.schema_version != "stage1.sctsr.seed_registry.v1":
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Unknown seed registry schema")
        if self.required_discovery_count != 8 or self.required_confirmation_count != 14:
            raise SctsrError(ErrorCode.SEED_REGISTRY_INVALID, "Formal seed count contract must remain 8 discovery and 14 confirmation")
        groups: dict[str, Sequence[int]] = {
            "historical": self.historical_training_seeds,
            "selection": self.selection_seeds,
            "discovery": self.discovery_seeds,
            "confirmation": self.confirmation_seeds,
        }
        for name, seeds in groups.items():
            if any(type(seed) is not int or seed < 0 or seed >= 2**63 for seed in seeds):
                raise SctsrError(ErrorCode.SEED_REGISTRY_INVALID, f"Invalid seed inside {name}", observed=list(seeds))
            if len(seeds) != len(set(seeds)):
                raise SctsrError(ErrorCode.SEED_REGISTRY_INVALID, f"Duplicate seed inside {name}", observed=list(seeds))
        owners: dict[int, str] = {}
        for name, seeds in groups.items():
            for seed in seeds:
                if seed in owners:
                    raise SctsrError(
                        ErrorCode.SEED_REGISTRY_INVALID,
                        "Historical, selection, discovery, and confirmation seeds must be disjoint",
                        observed={"seed": seed, "first": owners[seed], "second": name},
                    )
                owners[seed] = name
        if expected_registry_digest is not None and self.digest != expected_registry_digest.upper():
            raise SctsrError(ErrorCode.IDENTITY_DIGEST_MISMATCH, "Seed registry digest does not match the release binding", observed=self.digest, expected=expected_registry_digest.upper())
        if formal:
            if not release_authorization_verified:
                raise SctsrError(ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED, "A formal-looking seed state string is not release authorization")
            if len(self.discovery_seeds) != 8 or len(self.confirmation_seeds) != 14:
                raise SctsrError(
                    ErrorCode.SEED_REGISTRY_INVALID,
                    "Formal release requires exactly 8 discovery and 14 confirmation seeds",
                    observed={"discovery": len(self.discovery_seeds), "confirmation": len(self.confirmation_seeds)},
                    expected={"discovery": 8, "confirmation": 14},
                )
            if self.state != "FROZEN_BY_RELEASE_AUTHORITY":
                raise SctsrError(ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED, "Formal seed registry is not frozen by release authority")
            return
        if self.state == "FORMAL_SEEDS_BLOCKED_UNTIL_RELEASE":
            if self.discovery_seeds or self.confirmation_seeds:
                raise SctsrError(ErrorCode.FORMAL_SEEDS_BLOCKED_UNTIL_RELEASE, "Blocked registry may not contain formal discovery or confirmation seed values")
        elif self.state != "SYNTHETIC_FIXTURE_ONLY":
            raise SctsrError(ErrorCode.SEED_REGISTRY_INVALID, "Development seed registry has an invalid state")
