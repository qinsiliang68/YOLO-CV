from __future__ import annotations
from dataclasses import dataclass,asdict
import math
import numpy as np
import pandas as pd

from .errors import ValidationError

@dataclass(frozen=True)
class OperatingPoint:
    requested_constraint:str
    actual_FN:int
    actual_TN:int
    actual_TP:int
    actual_FP:int
    threshold:float
    constraint_status:str
    tie_group_size:int


def validate_predictions(df:pd.DataFrame)->pd.DataFrame:
    required={"sample_id","y_true","score"}; missing=required-set(df.columns)
    if missing: raise ValidationError(f"Prediction table missing {sorted(missing)}")
    x=df.copy()
    if x.sample_id.duplicated().any(): raise ValidationError("Prediction sample IDs must be unique")
    if not set(x.y_true.dropna().astype(int).unique()).issubset({0,1}): raise ValidationError("y_true must be 0/1")
    if not np.isfinite(x.score.astype(float)).all(): raise ValidationError("Scores contain NaN/Inf")
    return x


def tie_safe_sweep(y_true:np.ndarray,scores:np.ndarray)->pd.DataFrame:
    y=np.asarray(y_true,dtype=np.int8); s=np.asarray(scores,dtype=np.float64)
    if len(y)!=len(s) or len(y)==0: raise ValidationError("Invalid prediction arrays")
    order=np.argsort(-s,kind="mergesort"); y=y[order]; s=s[order]
    total_pos=int((y==1).sum()); total_neg=int((y==0).sum())
    change=np.r_[True,s[1:]!=s[:-1]]; starts=np.flatnonzero(change); ends=np.r_[starts[1:],len(s)]
    rows=[{"threshold":math.inf,"TP":0,"FP":0,"TN":total_neg,"FN":total_pos,"tie_group_size":0}]
    tp=fp=0
    for st,en in zip(starts,ends):
        g=y[st:en]; tp+=int((g==1).sum()); fp+=int((g==0).sum())
        rows.append({"threshold":float(s[st]),"TP":tp,"FP":fp,"TN":total_neg-fp,"FN":total_pos-tp,"tie_group_size":int(en-st)})
    return pd.DataFrame(rows)


def tn_at_fn(sweep:pd.DataFrame,max_fn:int=95)->OperatingPoint:
    ok=sweep[sweep.FN<=max_fn]
    if ok.empty:
        return OperatingPoint(f"FN<={max_fn}",-1,-1,-1,-1,float("nan"),"CONSTRAINT_UNREACHABLE",0)
    # Sweep is ordered highest threshold first. First satisfying row is the highest achievable threshold.
    r=ok.iloc[0]
    return OperatingPoint(f"FN<={max_fn}",int(r.FN),int(r.TN),int(r.TP),int(r.FP),float(r.threshold),"OK",int(r.tie_group_size))


def fn_at_tn(sweep:pd.DataFrame,min_tn:int=68253)->OperatingPoint:
    ok=sweep[sweep.TN>=min_tn]
    if ok.empty:
        return OperatingPoint(f"TN>={min_tn}",-1,-1,-1,-1,float("nan"),"CONSTRAINT_UNREACHABLE",0)
    # Last satisfying row is the lowest achievable threshold.
    r=ok.iloc[-1]
    return OperatingPoint(f"TN>={min_tn}",int(r.FN),int(r.TN),int(r.TP),int(r.FP),float(r.threshold),"OK",int(r.tie_group_size))


def operational_metrics(df:pd.DataFrame,fn_limit:int=95,tn_target:int=68253)->tuple[dict,pd.DataFrame]:
    x=validate_predictions(df); sw=tie_safe_sweep(x.y_true.to_numpy(),x.score.to_numpy())
    a=tn_at_fn(sw,fn_limit); b=fn_at_tn(sw,tn_target)
    n=x.loc[x.y_true==0,"score"].to_numpy(); d=x.loc[x.y_true==1,"score"].to_numpy()
    summary={
        "metric_version":"operational_v2_tie_safe","row_count":len(x),"normal_count":int(len(n)),"defect_count":int(len(d)),
        "TN_at_FN95":asdict(a),"FN_at_TN68253":asdict(b),
        "normal_q68":float(np.quantile(n,.68)),"normal_q90":float(np.quantile(n,.90)),
        "defect_q50":float(np.quantile(d,.50)),"defect_q05":float(np.quantile(d,.05)),
    }
    summary["gap_q68_q050"]=summary["defect_q50"]-summary["normal_q68"]
    summary["tail_gap_q90_q05"]=summary["defect_q05"]-summary["normal_q90"]
    return summary,sw
