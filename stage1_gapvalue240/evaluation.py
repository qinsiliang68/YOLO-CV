from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .calibration import fit_platt
from .errors import ValidationError
from .metrics import operational_metrics
from .util import atomic_write_bytes,atomic_write_json


def _normalize_prediction_columns(df:pd.DataFrame)->pd.DataFrame:
    aliases={"prob_defect":"score","p_defect":"score","p_defect_raw":"score","probability":"score","label":"y_true","canonical_image_relpath":"sample_id"}
    x=df.rename(columns={k:v for k,v in aliases.items() if k in df and v not in df}).copy()
    required={"sample_id","y_true","score"}; missing=required-set(x.columns)
    if missing: raise ValidationError(f"Prediction output missing {sorted(missing)}")
    return x[["sample_id","y_true","score"]]


def finalize_evaluation(raw_val_cal:str|Path,raw_val_op:str|Path,output_dir:str|Path,deployment_prevalence:float=.5,
                        overwrite:bool=False)->dict:
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    cal=_normalize_prediction_columns(pd.read_csv(raw_val_cal)); op=_normalize_prediction_columns(pd.read_csv(raw_val_op))
    model=fit_platt(cal.score.to_numpy(),cal.y_true.to_numpy(),deployment_prevalence)
    cal["score_raw"]=cal.score; cal["score"]=model.transform(cal.score.to_numpy())
    op["score_raw"]=op.score; op["score"]=model.transform(op.score.to_numpy())
    metrics,sweep=operational_metrics(op[["sample_id","y_true","score"]])
    atomic_write_bytes(output_dir/"val_cal_predictions.csv",cal.to_csv(index=False).encode(),overwrite)
    atomic_write_bytes(output_dir/"val_op_predictions.csv",op.to_csv(index=False).encode(),overwrite)
    atomic_write_json(output_dir/"platt_calibration.json",model.to_dict(),overwrite)
    atomic_write_json(output_dir/"operational_metrics.json",metrics,overwrite)
    atomic_write_bytes(output_dir/"threshold_sweep.csv",sweep.to_csv(index=False).encode(),overwrite)
    return metrics
