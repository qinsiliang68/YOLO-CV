from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .errors import ErrorCode, SctsrError
from .serialization import stable_digest


REQUIRED_SCHEMAS = {
    "contract": "stage1.sctsr.contract.v1",
    "asset_registry": "stage1.sctsr.asset_registry.v1",
    "arms": "stage1.sctsr.arms.v1",
    "runtime_policy": "stage1.sctsr.runtime_policy.v1",
    "disabled_phase2": "stage1.sctsr.disabled_phase2.v1",
    "seed_registry": "stage1.sctsr.seed_registry.v1",
    "formal_release": "stage1.sctsr.formal_release.v1",
    "release_trust": "stage1.sctsr.release_trust.v1",
}


@dataclass(frozen=True, slots=True)
class SchemaRegistry:
    schema_version: str
    schemas: Mapping[str, str]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SchemaRegistry":
        return cls(str(value.get("schema_version", "")), dict(value.get("schemas", {})))

    @property
    def digest(self) -> str:
        return stable_digest({"schema_version": self.schema_version, "schemas": dict(self.schemas)})

    def validate(self) -> None:
        if self.schema_version != "stage1.sctsr.schema_registry.v1":
            raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Unknown schema registry version")
        missing = {key: value for key, value in REQUIRED_SCHEMAS.items() if self.schemas.get(key) != value}
        if missing:
            raise SctsrError(
                ErrorCode.SCHEMA_VALIDATION_FAILED,
                "Schema registry is missing or changes frozen schema identities",
                observed=dict(self.schemas),
                expected=REQUIRED_SCHEMAS,
            )
