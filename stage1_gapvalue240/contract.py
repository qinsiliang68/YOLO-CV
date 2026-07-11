from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ContractError
from .util import read_yaml, stable_hash


@dataclass(frozen=True)
class Contract:
    path: Path
    data: dict[str, Any]
    sha256: str

    @property
    def contract_id(self) -> str:
        return str(self.data["contract_id"])

    @property
    def artifact_root_relative(self) -> str:
        version = self.data["contract_version"].replace(".", "_")
        return f"artifacts/stage1_sample_value_experiments/contracts/gapvalue240_v{version}"


def compute_contract_hash(data: dict[str, Any]) -> str:
    body = copy.deepcopy(data)
    body.pop("contract_sha256", None)
    return stable_hash(body)


def load_contract(path: str | Path, verify_hash: bool = True) -> Contract:
    path = Path(path).resolve()
    data = read_yaml(path)
    required = ["contract_id", "contract_version", "repository", "conditions", "training", "controls"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ContractError(f"Contract missing keys: {missing}")
    actual = compute_contract_hash(data)
    expected = str(data.get("contract_sha256", "")).upper()
    if verify_hash and actual != expected:
        raise ContractError(f"Contract hash mismatch: expected={expected}, actual={actual}")
    return Contract(path=path, data=data, sha256=actual)


def validate_contract_semantics(contract: Contract) -> list[str]:
    d = contract.data
    issues: list[str] = []
    if len(d["conditions"]["phase_a"]) != 19: issues.append("Phase A must contain 19 conditions")
    if len(d["conditions"]["phase_b"]) != 6: issues.append("Phase B must contain 6 conditions")
    seeds = d["training"]["training_seeds"]
    if len(seeds["discovery"]) != 3: issues.append("Discovery seeds must contain 3 values")
    if len(seeds["confirmation_new"]) != 5: issues.append("Confirmation seeds must contain 5 values")
    if d["multi_machine"]["validated_run_count"] != 240: issues.append("validated_run_count must be 240")
    if d["replay"]["base_samples"] != 120000: issues.append("base_samples must be 120000")
    folds = [str(x).zfill(2) for x in d["data_semantics"]["fold_values"]]
    if folds != [f"{i:02d}" for i in range(10)]: issues.append(f"Invalid fold values: {folds}")
    return issues
