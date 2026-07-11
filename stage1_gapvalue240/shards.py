from __future__ import annotations
from pathlib import Path
import pandas as pd
from .util import atomic_write_bytes,atomic_write_json


def write_machine_shards(matrix:pd.DataFrame,output_dir:str|Path)->None:
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    triads=sorted(matrix.triad_id.unique())
    if len(triads)!=80: raise ValueError("Expected 80 triads")
    allocation=[]
    for i in range(10):
        assigned=triads[i::10]; shard=matrix[matrix.triad_id.isin(assigned)].copy()
        path=output_dir/f"machine_{i+1:02d}_jobs.csv"; atomic_write_bytes(path,shard.to_csv(index=False).encode(),overwrite=True)
        allocation.append({"machine_id":f"machine_{i+1:02d}","triads":assigned,"runs":len(shard)})
    for i in [11,12]:
        path=output_dir/f"machine_{i:02d}_jobs.csv"; atomic_write_bytes(path,matrix.iloc[:0].to_csv(index=False).encode(),overwrite=True)
        allocation.append({"machine_id":f"machine_{i:02d}","triads":[],"runs":0,"role":"reserve"})
    atomic_write_json(output_dir/"allocation.json",allocation,overwrite=True)
