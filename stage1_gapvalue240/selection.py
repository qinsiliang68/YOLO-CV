from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import pandas as pd

from .contract import Contract
from .errors import ValidationError
from .matching import global_random, matched_random
from .matrix import RunSpec
from .util import atomic_write_bytes, atomic_write_json, sha256_file

@dataclass(frozen=True)
class SelectionArtifact:
    csv_path: Path
    audit_path: Path
    sha256: str


def _decorate(df:pd.DataFrame,spec:RunSpec,role:str,source_method:str)->pd.DataFrame:
    x=df.copy()
    x["run_slot"]=spec.run_slot; x["triad_id"]=spec.triad_id; x["condition_id"]=spec.condition_id
    x["arm"]=spec.arm; x["training_seed"]=spec.training_seed; x["selection_seed"]=spec.selection_seed
    x["replay_role"]=role; x["source_method"]=source_method
    if "rank" not in x: x["rank"]=range(1,len(x)+1)
    keep=["run_slot","triad_id","condition_id","arm","training_seed","selection_seed","rank","sample_id","y_true",
          "oof_fold","dynamic_bucket","mean_p_defect","correct_rate","std_p_defect","replay_role","source_method"]
    for c in keep:
        if c not in x: x[c]=None
    return x[keep]


def phase_a_selection(spec:RunSpec, ranking:pd.DataFrame, value_data:pd.DataFrame, contract:Contract, match_context=None)->tuple[pd.DataFrame,dict]:
    t=ranking.head(spec.budget).copy()
    if len(t)!=spec.budget: raise ValidationError(f"Treatment ranking too small: {spec.condition_id}")
    clean=value_data[(value_data.y_true==0)&value_data.is_clean].copy()
    if spec.arm=="T": selected=t; audit={"arm":"T","method":spec.method,"budget":spec.budget,"treatment_sha":None}
    elif spec.arm=="R1":
        result=global_random(clean,spec.budget,spec.selection_seed,set(t.sample_id)); selected=result.selected; audit=result.audit
    else:
        q=contract.data["controls"]["r2"]["quantile_bins"]
        result=matched_random(t,clean,spec.selection_seed,q,contract.data["controls"]["r2"]["validation"]["maximum_absolute_smd"],True,context=match_context)
        selected=result.selected; audit=result.audit
    return _decorate(selected,spec,"normal_replay",spec.method),audit


def phase_b_selection(spec:RunSpec, normal_top3000:pd.DataFrame, defect_ranking:pd.DataFrame,
                      value_data:pd.DataFrame,contract:Contract, match_context=None)->tuple[pd.DataFrame,dict]:
    ratio=spec.guard_ratio; defect_n=int(round(spec.budget*ratio)); normal_n=spec.budget-defect_n
    normals=normal_top3000.head(normal_n).copy()
    targeted=defect_ranking.head(defect_n).copy()
    clean_defect=value_data[(value_data.y_true==1)&value_data.is_clean].copy()
    if spec.arm=="T": defects=targeted; daudit={"arm":"T","method":spec.method}
    elif spec.arm=="R1":
        r=global_random(clean_defect,defect_n,spec.selection_seed,set(targeted.sample_id)); defects=r.selected; daudit=r.audit
    else:
        q=contract.data["controls"]["r2"]["quantile_bins"]
        r=matched_random(targeted,clean_defect,spec.selection_seed,q,contract.data["controls"]["r2"]["validation"]["maximum_absolute_smd"],True,context=match_context)
        defects=r.selected; daudit=r.audit
    n=_decorate(normals,spec,"normal_replay","GapCritical-Strict")
    d=_decorate(defects,spec,"defect_guard",spec.method)
    out=pd.concat([n,d],ignore_index=True)
    if len(out)!=spec.budget or out.sample_id.duplicated().any():
        raise ValidationError("Phase B selection must have exact unique replay slots")
    return out,{"normal_count":normal_n,"defect_count":defect_n,"normal_ids_sha256":_ids_hash(normals),"defect_audit":daudit}


def _ids_hash(df:pd.DataFrame)->str:
    import hashlib
    return hashlib.sha256("\n".join(df.sample_id.astype(str)).encode()).hexdigest().upper()


def write_selection(df:pd.DataFrame,audit:dict,output_dir:str|Path,overwrite:bool=False)->SelectionArtifact:
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    csv_path=output_dir/"selection_manifest.csv"; audit_path=output_dir/"selection_audit.json"
    atomic_write_bytes(csv_path,df.to_csv(index=False).encode("utf-8"),overwrite)
    audit=dict(audit); audit.update({"rows":len(df),"unique_samples":int(df.sample_id.nunique()),"selection_sha256":sha256_file(csv_path)})
    atomic_write_json(audit_path,audit,overwrite)
    return SelectionArtifact(csv_path,audit_path,audit["selection_sha256"])
