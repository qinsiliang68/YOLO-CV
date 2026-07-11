from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import pandas as pd

from .contract import Contract
from .seeds import derive_seed
from .util import atomic_write_bytes

@dataclass(frozen=True)
class RunSpec:
    run_slot: str
    triad_id: str
    phase: str
    condition_slot: str
    condition_id: str
    method: str
    budget: int
    guard_ratio: float
    arm: str
    training_seed: int
    selection_seed: int
    discovery_or_confirmation: str


def build_run_specs(contract: Contract) -> list[RunSpec]:
    d = contract.data
    specs: list[RunSpec] = []
    run_idx = 1
    triad_idx = 1
    discovery = list(d["training"]["training_seeds"]["discovery"])
    confirmation = list(d["training"]["training_seeds"]["confirmation_new"])
    for phase_key, phase_name in [("phase_a", "A"), ("phase_b", "B")]:
        for cond in d["conditions"][phase_key]:
            for seed in discovery:
                triad_id = f"TRIAD_{triad_idx:03d}"
                cid = f"{cond['slot']}_{cond['method']}_B{cond['budget']}"
                for arm in ["T", "R1", "R2"]:
                    specs.append(RunSpec(
                        run_slot=f"RUN_{run_idx:03d}", triad_id=triad_id, phase=phase_name,
                        condition_slot=cond["slot"], condition_id=cid, method=cond["method"],
                        budget=int(cond["budget"]), guard_ratio=float(cond.get("guard_ratio", 0.0)), arm=arm,
                        training_seed=int(seed), selection_seed=derive_seed(contract.contract_id, cid, seed, arm),
                        discovery_or_confirmation="discovery"))
                    run_idx += 1
                triad_idx += 1
    source = next(c for c in d["conditions"]["phase_a"] if c["slot"] == d["conditions"]["phase_c"]["source_slot"])
    for seed in confirmation:
        triad_id = f"TRIAD_{triad_idx:03d}"
        cid = f"C_{source['slot']}_{source['method']}_B{source['budget']}"
        for arm in ["T", "R1", "R2"]:
            specs.append(RunSpec(
                run_slot=f"RUN_{run_idx:03d}", triad_id=triad_id, phase="C",
                condition_slot=source["slot"], condition_id=cid, method=source["method"],
                budget=int(source["budget"]), guard_ratio=0.0, arm=arm,
                training_seed=int(seed), selection_seed=derive_seed(contract.contract_id, cid, seed, arm),
                discovery_or_confirmation="confirmation"))
            run_idx += 1
        triad_idx += 1
    if len(specs) != 240 or triad_idx - 1 != 80:
        raise RuntimeError(f"Matrix count error: runs={len(specs)}, triads={triad_idx-1}")
    return specs


def write_matrix(specs: list[RunSpec], path: str | Path, overwrite: bool = False) -> Path:
    df = pd.DataFrame([asdict(x) for x in specs])
    data = df.to_csv(index=False).encode("utf-8")
    return atomic_write_bytes(path, data, overwrite=overwrite)


def load_matrix(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"run_slot":"string","triad_id":"string","condition_slot":"string","arm":"string"})
    if len(df) != 240 or df["triad_id"].nunique() != 80: raise ValueError("Frozen matrix must contain 240 runs / 80 triads")
    if set(df["arm"]) != {"T","R1","R2"}: raise ValueError("Matrix arms invalid")
    return df
