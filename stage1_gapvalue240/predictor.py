from __future__ import annotations
import importlib
import sys
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

from .errors import ValidationError
from .util import atomic_write_bytes

PATH_COLUMNS=("canonical_image_relpath","image_path","source_image_path")

def _read_eval_manifest(path:Path,y_true:int,dataset_root:Path)->pd.DataFrame:
    df=pd.read_csv(path,dtype={"canonical_image_relpath":"string"})
    id_col="canonical_image_relpath" if "canonical_image_relpath" in df else ("sample_id" if "sample_id" in df else None)
    if id_col is None: raise ValidationError(f"Evaluation manifest lacks canonical identity: {path}")
    chosen=None
    for c in PATH_COLUMNS:
        if c in df: chosen=c; break
    if chosen is None: raise ValidationError(f"Evaluation manifest lacks an accepted path column {PATH_COLUMNS}: {path}")
    def resolve(value):
        p=Path(str(value))
        return p if p.is_absolute() else dataset_root/p
    out=pd.DataFrame({"sample_id":df[id_col].astype(str),"image_path":df[chosen].map(resolve),"y_true":int(y_true)})
    if out.sample_id.duplicated().any(): raise ValidationError(f"Duplicate identities in {path}")
    missing=[str(p) for p in out.image_path if not p.exists()]
    if missing: raise FileNotFoundError(f"Missing {len(missing)} evaluation images; first={missing[0]}")
    return out

def _defect_index(names,accepted:list[str])->int:
    mapping={int(k):str(v) for k,v in names.items()} if isinstance(names,dict) else {i:str(v) for i,v in enumerate(names)}
    accepted_lower={str(x).lower() for x in accepted}|{"target_defect"}
    hits=[i for i,n in mapping.items() if n.lower() in accepted_lower]
    if len(hits)!=1: raise ValidationError(f"Cannot identify exactly one defect class from model names={mapping}, accepted={accepted}")
    return hits[0]

def _local_yolo(yolo_root:Path):
    yolo_root=Path(yolo_root).resolve()
    if not yolo_root.exists(): raise FileNotFoundError(f"Missing local YOLOv11 source: {yolo_root}")
    sys.path.insert(0,str(yolo_root))
    try:
        module=importlib.import_module("ultralytics")
    except Exception as exc: raise RuntimeError(f"Ultralytics import failed: {exc}") from exc
    module_path=Path(module.__file__).resolve()
    try: module_path.relative_to(yolo_root)
    except ValueError as exc: raise RuntimeError(f"Ultralytics resolved outside frozen YOLO root: {module_path}") from exc
    return module.YOLO

def _predict_paths(checkpoint:Path,rows:pd.DataFrame,output:Path,gpu_id,batch:int,workers:int,imgsz:int,
                   accepted_defect_names:list[str],yolo_root:Path)->Path:
    YOLO=_local_yolo(yolo_root)
    model=YOLO(str(checkpoint)); defect_idx=_defect_index(model.names,accepted_defect_names)
    records=[]; paths=rows.image_path.astype(str).tolist(); ids=rows.sample_id.astype(str).tolist(); labels=rows.y_true.astype(int).tolist()
    chunk_size=max(batch*16,1024)
    offset=0
    while offset<len(paths):
        chunk=paths[offset:offset+chunk_size]
        results=model.predict(source=chunk,imgsz=imgsz,batch=batch,device=str(gpu_id),workers=workers,verbose=False,stream=True)
        count=0
        for result in results:
            if result.probs is None: raise ValidationError("Model prediction did not return classification probabilities")
            probs=result.probs.data.detach().float().cpu().numpy()
            records.append((ids[offset+count],labels[offset+count],float(probs[defect_idx]))); count+=1
        if count!=len(chunk): raise ValidationError(f"Prediction count mismatch: expected {len(chunk)}, got {count}")
        offset+=len(chunk)
    out=pd.DataFrame(records,columns=["sample_id","y_true","score"])
    return atomic_write_bytes(output,out.to_csv(index=False).encode(),overwrite=True)

def predict_split(checkpoint:str|Path,dataset_root:str|Path,defect_manifest:str|Path,normal_manifest:str|Path,output:str|Path,
                  gpu_id=0,batch:int=256,workers:int=8,imgsz:int=224,accepted_defect_names:list[str]|None=None,
                  yolo_root:str|Path="YOLOv11")->Path:
    dataset_root=Path(dataset_root); defect=_read_eval_manifest(Path(defect_manifest),1,dataset_root); normal=_read_eval_manifest(Path(normal_manifest),0,dataset_root)
    rows=pd.concat([defect,normal],ignore_index=True)
    if rows.sample_id.duplicated().any(): raise ValidationError("Defect/normal evaluation manifests overlap")
    return _predict_paths(Path(checkpoint),rows,Path(output),gpu_id,batch,workers,imgsz,
                          accepted_defect_names or ["defect","def","1","abnormal","target_defect"],Path(yolo_root))
