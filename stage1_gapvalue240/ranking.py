from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .errors import ValidationError
from .oof import compute_epoch_gap_metrics, score_from_windows

@dataclass(frozen=True)
class RankingResult:
    method: str
    table: pd.DataFrame
    metadata: dict


def load_value_assets(value_path: str | Path, assignments_path: str | Path) -> pd.DataFrame:
    v=pd.read_csv(value_path,dtype={"sample_id":"string","oof_fold":"string"})
    a=pd.read_csv(assignments_path,usecols=["canonical_image_relpath","oof_group_id","train_primary_class","oof_y_true","oof_fold"],
                  dtype={"canonical_image_relpath":"string","oof_fold":"string","oof_group_id":"string","train_primary_class":"string"})
    v["oof_fold"]=v["oof_fold"].str.zfill(2); a["oof_fold"]=a["oof_fold"].str.zfill(2)
    d=v.merge(a,left_on="sample_id",right_on="canonical_image_relpath",how="left",validate="one_to_one",suffixes=("","_assignment"))
    if d["canonical_image_relpath"].isna().any(): raise ValidationError("Value rows missing assignment identity")
    if not np.array_equal(d.y_true.to_numpy(),d.oof_y_true.to_numpy()): raise ValidationError("Value/assignment labels differ")
    d["is_clean"]=d.dynamic_bucket.ne("possible_noise_or_label_issue")
    d["endpoint_trend_score"]=(d.p_defect_start-d.p_defect_end).clip(lower=0)
    d["boundary_0p5_score"]=1-2*(d.mean_p_defect-.5).abs()
    d["persistent_0p5_score"]=1-d.correct_rate
    return d


def _rank(df: pd.DataFrame, method: str, score: str, ascending: bool, pool_id: str) -> RankingResult:
    out=df.sort_values([score,"sample_id"],ascending=[ascending,True],kind="mergesort").copy()
    out.insert(0,"rank",np.arange(1,len(out)+1,dtype=np.int64))
    out["method"]=method; out["method_eligible_pool_id"]=pool_id; out["raw_score"]=out[score]
    cols=["rank","sample_id","y_true","oof_fold","oof_group_id","train_primary_class","dynamic_bucket",
          "mean_p_defect","correct_rate","std_p_defect","raw_score","method","method_eligible_pool_id"]
    return RankingResult(method,out[cols],{"pool_id":pool_id,"candidate_count":len(out),"score_column":score,"ascending":ascending})


def direct_ranking(data: pd.DataFrame, method: str) -> RankingResult:
    n=data[(data.y_true==0)&data.is_clean]
    if method=="Confidence-Clean": return _rank(n,method,"mean_p_defect",False,"clean_normal")
    if method=="Boundary-0.5-Clean": return _rank(n,method,"boundary_0p5_score",False,"clean_normal")
    if method=="Persistent-0.5-Clean": return _rank(n,method,"persistent_0p5_score",False,"clean_normal")
    if method=="EndpointTrend": return _rank(n[n.dynamic_bucket=="learnable_hard"],method,"endpoint_trend_score",False,"learnable_normal")
    if method=="GapCritical-Strict": return _rank(n[(n.dynamic_bucket=="learnable_hard")&n.gap_critical_score.gt(0)],method,"gap_critical_score",False,"strict_normal")
    if method=="GapCritical-Global": return _rank(n[n.gap_critical_score.gt(0)],method,"gap_critical_score",False,"global_clean_normal_positive_gap")
    if method=="BottomGap-3000-stress-control": return _rank(n,method,"gap_critical_score",True,"clean_normal")
    if method=="FoldBalanced-Gap":
        x=n[(n.dynamic_bucket=="learnable_hard")&n.gap_critical_score.gt(0)].copy()
        x["within_rank"]=x.groupby("oof_fold")["gap_critical_score"].rank(method="first",ascending=False)
        x=x.sort_values(["within_rank","gap_critical_score","sample_id"],ascending=[True,False,True],kind="mergesort")
        x["fold_balanced_score"]=-x.within_rank+1e-6*x.gap_critical_score
        return _rank(x,method,"fold_balanced_score",False,"strict_normal")
    if method=="GroupDiverse-Gap":
        x=n[(n.dynamic_bucket=="learnable_hard")&n.gap_critical_score.gt(0)].copy()
        x["within_group_rank"]=x.groupby("oof_group_id")["gap_critical_score"].rank(method="first",ascending=False)
        x=x.sort_values(["within_group_rank","gap_critical_score","sample_id"],ascending=[True,False,True],kind="mergesort")
        x["group_diverse_score"]=-x.within_group_rank+1e-6*x.gap_critical_score
        return _rank(x,method,"group_diverse_score",False,"strict_normal")
    if method in {"LowConfidence-Defect","Persistent-FN","GapGuard-Raw","GapGuard-ClassStrat"}:
        d=data[(data.y_true==1)&data.is_clean].copy()
        if method=="LowConfidence-Defect": return _rank(d,method,"mean_p_defect",True,"clean_defect")
        if method=="Persistent-FN": return _rank(d,method,"persistent_0p5_score",False,"clean_defect")
        d=d[d.gap_guard_score.gt(0)]
        if method=="GapGuard-Raw": return _rank(d,method,"gap_guard_score",False,"clean_defect_positive_guard")
        d["within_class_rank"]=d.groupby("train_primary_class")["gap_guard_score"].rank(method="first",ascending=False)
        d["class_count"]=d.groupby("train_primary_class")["sample_id"].transform("size")
        # Sorting by within-class percentile yields approximately proportional class allocation at every prefix.
        d["class_strat_score"]=1.0-(d.within_class_rank-1.0)/d.class_count+1e-12*d.gap_guard_score
        return _rank(d,method,"class_strat_score",False,"clean_defect_positive_guard")
    raise KeyError(f"Unknown direct method: {method}")


def _top_bottom(metrics: pd.DataFrame, col: str, n: int=40) -> tuple[list[int],list[int]]:
    good=metrics.nlargest(n,col).epoch.astype(int).tolist(); bad=metrics.nsmallest(n,col).epoch.astype(int).tolist()
    return good,bad


def dynamic_scores(matrix: np.ndarray, labels: np.ndarray, folds: np.ndarray, epoch_metrics: pd.DataFrame) -> dict[str,np.ndarray]:
    out:dict[str,np.ndarray]={}
    good,bad=_top_bottom(epoch_metrics,"tail_gap_q90_q05"); out["TailGap-Strict"]=score_from_windows(matrix,good,bad)
    out["WindowEarlyLate40"]=score_from_windows(matrix,range(161,201),range(1,41))
    t=np.linspace(-1,1,200); coef=np.polyfit(t,epoch_metrics.gap_q68_q050.to_numpy(),3); resid=epoch_metrics.gap_q68_q050.to_numpy()-np.polyval(coef,t)
    good=(np.argsort(resid)[-40:]+1).tolist(); bad=(np.argsort(resid)[:40]+1).tolist(); out["GapResidual-Strict"]=score_from_windows(matrix,good,bad)
    good=[];bad=[]
    for start in [1,51,101,151]:
        block=epoch_metrics[(epoch_metrics.epoch>=start)&(epoch_metrics.epoch<=start+49)]
        good+=block.nlargest(10,"gap_q68_q050").epoch.astype(int).tolist(); bad+=block.nsmallest(10,"gap_q68_q050").epoch.astype(int).tolist()
    out["GapCritical-Strict-TimeMatched"]=score_from_windows(matrix,good,bad)
    ex=compute_epoch_gap_metrics(matrix,labels,folds,exclude_fold_epoch=("01",178)); good,bad=_top_bottom(ex,"gap_q68_q050")
    exscore=score_from_windows(matrix,good,bad)
    if 178 in good or 178 in bad:
        # fold 01 sample values at repaired epoch are excluded from their own mean.
        mask=folds=="01"; g=[e for e in good if e!=178]; b=[e for e in bad if e!=178]
        exscore[mask]=score_from_windows(matrix[:,mask],g,b)
    out["Exclude178-GapStrict"]=exscore
    lofo=np.empty(matrix.shape[1],dtype=np.float64)
    for fold in [f"{i:02d}" for i in range(10)]:
        other=folds!=fold; metrics=compute_epoch_gap_metrics(matrix[:,other],labels[other])
        good,bad=_top_bottom(metrics,"gap_q68_q050"); own=folds==fold
        lofo[own]=score_from_windows(matrix[:,own],good,bad)
    out["LOFO-GapStrict"]=lofo
    return out


def dynamic_ranking(data: pd.DataFrame, score: np.ndarray, method: str) -> RankingResult:
    if len(score)!=len(data): raise ValidationError("Dynamic score length mismatch")
    x=data.copy(); x["dynamic_variant_score"]=score
    n=x[(x.y_true==0)&x.is_clean&(x.dynamic_bucket=="learnable_hard")&x.dynamic_variant_score.gt(0)]
    return _rank(n,method,"dynamic_variant_score",False,"strict_normal")
