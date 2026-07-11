from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from .errors import ValidationError
from .util import atomic_write_bytes, atomic_write_json, sha256_file

IDENTITY="canonical_image_relpath"

@dataclass(frozen=True)
class ReplayManifests:
    train_manifest:Path
    normal_train_manifest:Path
    audit_manifest:Path
    summary_path:Path


def _read_base(path:Path)->pd.DataFrame:
    df=pd.read_csv(path,dtype={IDENTITY:"string"})
    if IDENTITY not in df: raise ValidationError(f"Base manifest missing {IDENTITY}: {path}")
    if df[IDENTITY].duplicated().any(): raise ValidationError(f"Base manifest has duplicate identities: {path}")
    return df


def build_replay_manifests(base_train:str|Path,base_normal:str|Path,selection_path:str|Path,output_dir:str|Path,
                           expected_base_total:int=120000,overwrite:bool=False)->ReplayManifests:
    base_train=Path(base_train); base_normal=Path(base_normal); output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    defect=_read_base(base_train); normal=_read_base(base_normal)
    if len(defect)+len(normal)!=expected_base_total: raise ValidationError(f"Base manifests total {len(defect)+len(normal)} != {expected_base_total}")
    sel=pd.read_csv(selection_path,dtype={"sample_id":"string"})
    required_selection={"sample_id","y_true","replay_role","run_slot","rank"}
    missing_selection=required_selection-set(sel.columns)
    if missing_selection: raise ValidationError(f"Selection manifest missing columns: {sorted(missing_selection)}")
    if sel.sample_id.duplicated().any(): raise ValidationError("Selection IDs must be unique")
    if sel.run_slot.astype(str).nunique()!=1: raise ValidationError("Selection manifest must contain exactly one run_slot")
    normal_map=normal.set_index(IDENTITY,drop=False); defect_map=defect.set_index(IDENTITY,drop=False)
    nids=sel.loc[sel.y_true==0,"sample_id"]; dids=sel.loc[sel.y_true==1,"sample_id"]
    missing_n=set(nids)-set(normal_map.index); missing_d=set(dids)-set(defect_map.index)
    if missing_n or missing_d: raise ValidationError(f"Selection IDs absent from base manifests: normal={len(missing_n)}, defect={len(missing_d)}")
    nreplay=normal_map.loc[nids].reset_index(drop=True) if len(nids) else normal.iloc[:0].copy()
    dreplay=defect_map.loc[dids].reset_index(drop=True) if len(dids) else defect.iloc[:0].copy()
    if "Filename" not in normal or "Filename" not in defect: raise ValidationError("Current trainer requires Filename in base manifests")
    slot_by_id={str(row.sample_id):int(index) for index,row in enumerate(sel.itertuples(index=False),start=1)}
    run_slot=str(sel.run_slot.iloc[0])
    def rename_replay(frame:pd.DataFrame)->pd.DataFrame:
        out=frame.copy()
        out["Filename"]=[f"replay__{run_slot}__{slot_by_id[str(sample_id)]:05d}__{Path(str(filename)).name}"
                         for sample_id,filename in zip(out[IDENTITY],out["Filename"])]
        return out
    nreplay=rename_replay(nreplay); dreplay=rename_replay(dreplay)
    out_n=pd.concat([normal,nreplay],ignore_index=True); out_d=pd.concat([defect,dreplay],ignore_index=True)
    if out_n.Filename.duplicated().any() or out_d.Filename.duplicated().any():
        raise ValidationError("Replay Filename values must be unique within each trainer class directory")
    train_out=output_dir/"train_manifest.csv"; normal_out=output_dir/"normal_train_manifest.csv"; audit_out=output_dir/"run_manifest.csv"
    atomic_write_bytes(train_out,out_d.to_csv(index=False).encode(),overwrite)
    atomic_write_bytes(normal_out,out_n.to_csv(index=False).encode(),overwrite)
    base_a=pd.concat([
        pd.DataFrame({"sample_id":defect[IDENTITY],"y_true":1,"role":"base","exposure_index":0}),
        pd.DataFrame({"sample_id":normal[IDENTITY],"y_true":0,"role":"base","exposure_index":0})],ignore_index=True)
    replay_a=sel[["sample_id","y_true","replay_role"]].rename(columns={"replay_role":"role"}).copy(); replay_a["exposure_index"]=1
    audit=pd.concat([base_a,replay_a],ignore_index=True)
    atomic_write_bytes(audit_out,audit.to_csv(index=False).encode(),overwrite)
    summary={"base_train_rows":len(defect),"base_normal_rows":len(normal),"replay_defect_rows":len(dreplay),"replay_normal_rows":len(nreplay),
             "epoch_samples":len(out_d)+len(out_n),"train_manifest_sha256":sha256_file(train_out),"normal_manifest_sha256":sha256_file(normal_out),
             "run_manifest_sha256":sha256_file(audit_out)}
    summary_path=output_dir/"manifest_summary.json"; atomic_write_json(summary_path,summary,overwrite)
    return ReplayManifests(train_out,normal_out,audit_out,summary_path)
