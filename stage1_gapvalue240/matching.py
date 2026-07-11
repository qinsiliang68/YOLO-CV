from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import itertools
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .errors import InfeasibleMatchError, ValidationError

FEATURES=["mean_p_defect","correct_rate","std_p_defect"]
BIN_COLS=[f"{f}_bin" for f in FEATURES]

@dataclass(frozen=True)
class MatchResult:
    selected: pd.DataFrame
    audit: dict[str,Any]

@dataclass
class MatchContext:
    pool: pd.DataFrame
    bin_metadata: dict[str,Any]
    mu: pd.Series
    sd: pd.Series
    z: np.ndarray
    id_to_index: dict[str,int]
    exact_map: dict[tuple, np.ndarray]
    fold_indices: dict[str,np.ndarray]
    fold_nn: dict[str,NearestNeighbors]
    fold_z: dict[str,np.ndarray]


def _quantile_codes(series: pd.Series, q: int) -> tuple[pd.Series,list[float]]:
    clean=series.astype(float)
    try:
        codes,edges=pd.qcut(clean,q=q,labels=False,retbins=True,duplicates="drop")
    except ValueError:
        codes=pd.Series(np.zeros(len(clean),dtype=int),index=clean.index); edges=np.array([clean.min(),clean.max()])
    return codes.astype("Int64"),[float(x) for x in edges]


def add_match_bins(pool: pd.DataFrame, requested: dict[str,int]) -> tuple[pd.DataFrame,dict]:
    x=pool.copy().reset_index(drop=True); meta={}
    for f,q in requested.items():
        codes,edges=_quantile_codes(x[f],q)
        x[f"{f}_bin"]=codes.astype(int)
        meta[f]={"requested_bins":q,"actual_bins":int(codes.nunique()),"edges":edges}
    return x,meta


def build_match_context(pool:pd.DataFrame,quantile_bins:dict[str,int]|None=None)->MatchContext:
    quantile_bins=quantile_bins or {"mean_p_defect":20,"correct_rate":20,"std_p_defect":10}
    required={"sample_id","y_true","oof_fold","dynamic_bucket",*FEATURES}
    missing=required-set(pool.columns)
    if missing: raise ValidationError(f"Match pool missing columns: {sorted(missing)}")
    if pool.sample_id.duplicated().any(): raise ValidationError("Match pool sample IDs are not unique")
    x,meta=add_match_bins(pool,quantile_bins)
    x["oof_fold"]=x.oof_fold.astype(str).str.zfill(2)
    mu=x[FEATURES].mean(); sd=x[FEATURES].std(ddof=0).replace(0,1)
    z=((x[FEATURES]-mu)/sd).to_numpy(np.float64)
    id_to_index={str(s):int(i) for i,s in enumerate(x.sample_id)}
    exact_map={}
    for key,g in x.groupby(["oof_fold",*BIN_COLS],sort=False,dropna=False): exact_map[tuple(key)]=g.index.to_numpy(np.int64)
    fold_indices={}; fold_nn={}; fold_z={}
    for fold,g in x.groupby("oof_fold",sort=True):
        idx=g.index.to_numpy(np.int64); fz=z[idx]
        nn=NearestNeighbors(algorithm="auto",metric="euclidean").fit(fz)
        fold_indices[str(fold)]=idx; fold_nn[str(fold)]=nn; fold_z[str(fold)]=fz
    return MatchContext(x,meta,mu,sd,z,id_to_index,exact_map,fold_indices,fold_nn,fold_z)


def standardized_mean_differences(t: pd.DataFrame, c: pd.DataFrame, features=FEATURES) -> dict[str,float]:
    out={}
    for f in features:
        pooled=float(np.sqrt((t[f].var(ddof=1)+c[f].var(ddof=1))/2))
        out[f]=0.0 if pooled==0 else float((t[f].mean()-c[f].mean())/pooled)
    return out


def distribution_report(t: pd.DataFrame,c:pd.DataFrame,col:str) -> dict[str,Any]:
    tv=t[col].astype(str).value_counts(normalize=True); cv=c[col].astype(str).value_counts(normalize=True)
    keys=sorted(set(tv.index)|set(cv.index))
    return {"treatment":{k:float(tv.get(k,0)) for k in keys},"control":{k:float(cv.get(k,0)) for k in keys},
            "total_variation":float(.5*sum(abs(float(tv.get(k,0))-float(cv.get(k,0))) for k in keys))}


def global_random(pool: pd.DataFrame,budget:int,seed:int,exclude_ids:set[str]) -> MatchResult:
    c=pool[~pool.sample_id.isin(exclude_ids)].copy()
    if len(c)<budget: raise InfeasibleMatchError(f"R1 pool too small: need={budget}, available={len(c)}")
    selected=c.sample(n=budget,replace=False,random_state=seed).copy()
    return MatchResult(selected,{"method":"Global-Random","budget":budget,"seed":seed,"overlap_count":0,"candidate_count":len(c)})


def _neighbor_bin_keys(row:pd.Series,level:str)->Iterable[tuple]:
    fold=str(row.oof_fold)
    b0,b1,b2=(int(row[c]) for c in BIN_COLS)
    if level=="L0": yield (fold,b0,b1,b2); return
    if level=="L1":
        for d2 in [-1,0,1]: yield (fold,b0,b1,b2+d2)
        return
    if level=="L2":
        for d1,d2 in itertools.product([-1,0,1],repeat=2): yield (fold,b0,b1+d1,b2+d2)
        return
    raise ValueError(level)


def _best_from_indices(indices:Iterable[int],target_z:np.ndarray,ctx:MatchContext,used:set[int],treatment_indices:set[int])->int|None:
    available=[i for i in indices if i not in used and i not in treatment_indices]
    if not available: return None
    arr=np.asarray(available,dtype=np.int64)
    dist=((ctx.z[arr]-target_z)**2).sum(axis=1)
    return int(arr[int(np.argmin(dist))])


def matched_random(treatment: pd.DataFrame, global_clean_pool: pd.DataFrame|None, seed:int,
                   quantile_bins:dict[str,int]|None=None, max_abs_smd:float=.10,
                   allow_forced_overlap:bool=True, context:MatchContext|None=None) -> MatchResult:
    """Build a deterministic, same-fold hardness-matched control efficiently.

    Binning and nearest-neighbor indexes can be reused through ``MatchContext``.
    It first maximizes disjoint matches. If the frozen SMD gate is infeasible,
    it replaces the worst matched pairs with their corresponding treatment
    sample, recording every forced overlap and effective unique contrast.
    """
    if context is None:
        if global_clean_pool is None: raise ValueError("global_clean_pool or context is required")
        context=build_match_context(global_clean_pool,quantile_bins)
    ctx=context; pool=ctx.pool
    required={"sample_id","y_true","oof_fold","dynamic_bucket",*FEATURES}
    missing=required-set(treatment.columns)
    if missing: raise ValidationError(f"Treatment missing match columns: {sorted(missing)}")
    if treatment.sample_id.duplicated().any(): raise ValidationError("Treatment IDs are not unique")
    missing_ids=set(treatment.sample_id.astype(str))-set(ctx.id_to_index)
    if missing_ids: raise ValidationError(f"Treatment IDs absent from match pool: {len(missing_ids)}")
    t_indices=np.array([ctx.id_to_index[str(s)] for s in treatment.sample_id],dtype=np.int64)
    t=pool.iloc[t_indices].reset_index(drop=True)
    treatment_index_set=set(map(int,t_indices))
    rng=np.random.default_rng(seed)
    scarcity=[]
    for pos,row in t.iterrows():
        exact=ctx.exact_map.get((str(row.oof_fold),*(int(row[c]) for c in BIN_COLS)),np.empty(0,dtype=np.int64))
        available_count=sum(int(i not in treatment_index_set) for i in exact)
        scarcity.append((available_count,float(rng.random()),pos))
    used:set[int]=set(); selected_indices=[]; levels=[]; distances=[]
    for _,__,pos in sorted(scarcity):
        row=t.iloc[pos]; target_idx=int(t_indices[pos]); target_z=ctx.z[target_idx]
        chosen=None; level="L3"
        for lev in ["L0","L1","L2"]:
            candidates=[]
            for key in _neighbor_bin_keys(row,lev): candidates.extend(ctx.exact_map.get(key,()))
            chosen=_best_from_indices(candidates,target_z,ctx,used,treatment_index_set)
            if chosen is not None: level=lev; break
        if chosen is None:
            fold=str(row.oof_fold); fold_idx=ctx.fold_indices[fold]; nn=ctx.fold_nn[fold]
            k=min(max(64,int(len(t)*.05)),len(fold_idx))
            while True:
                _,loc=nn.kneighbors(target_z.reshape(1,-1),n_neighbors=k)
                for local in loc[0]:
                    idx=int(fold_idx[int(local)])
                    if idx not in used and idx not in treatment_index_set:
                        chosen=idx; break
                if chosen is not None or k==len(fold_idx): break
                k=min(len(fold_idx),k*2)
        if chosen is None:
            chosen=target_idx; level="L4_FORCED_OVERLAP"
        used.add(chosen); selected_indices.append(chosen); levels.append(level)
        distances.append(float(np.linalg.norm(ctx.z[chosen]-target_z)))
    selected=pool.iloc[selected_indices].copy().reset_index(drop=True)
    selected["matched_treatment_id"]=t.sample_id.to_numpy()
    selected["fallback_level"]=levels; selected["match_distance"]=distances
    def smd_obj(frame:pd.DataFrame)->tuple[float,dict[str,float]]:
        s=standardized_mean_differences(t,frame); return max(abs(v) for v in s.values()),s
    obj,smd=smd_obj(selected)
    if obj>max_abs_smd:
        if not allow_forced_overlap: raise InfeasibleMatchError(f"Disjoint R2 fails SMD: {smd}")
        order=np.argsort(-selected.match_distance.to_numpy())
        for pos in order:
            if selected.iloc[pos].sample_id==t.iloc[pos].sample_id: continue
            # Pair order is preserved, so replacement is unique and same-fold by construction.
            selected.loc[pos,pool.columns]=t.iloc[pos][pool.columns].to_numpy()
            selected.loc[pos,"fallback_level"]="L4_FORCED_OVERLAP"; selected.loc[pos,"match_distance"]=0.0
            obj,smd=smd_obj(selected)
            if obj<=max_abs_smd: break
    obj,smd=smd_obj(selected)
    if obj>max_abs_smd+1e-12: raise InfeasibleMatchError(f"R2 did not reach SMD gate: {smd}")
    if selected.sample_id.duplicated().any(): raise ValidationError("Matched control contains duplicate samples")
    overlap=set(selected.sample_id.astype(str))&set(t.sample_id.astype(str))
    fallback_counts={str(k):int(v) for k,v in selected.fallback_level.value_counts().items()}
    audit={
        "method":"Method-Matched-Random-v1.1","seed":int(seed),"budget":len(t),"candidate_count":len(pool),
        "bin_metadata":ctx.bin_metadata,"smd":smd,"max_abs_smd":obj,"forced_overlap_count":len(overlap),
        "overlap_count":len(overlap),"overlap_rate":len(overlap)/len(t),
        "jaccard":len(overlap)/len(set(selected.sample_id.astype(str))|set(t.sample_id.astype(str))),
        "unique_to_treatment":len(set(t.sample_id.astype(str))-set(selected.sample_id.astype(str))),
        "unique_to_control":len(set(selected.sample_id.astype(str))-set(t.sample_id.astype(str))),
        "effective_unique_contrast":1-len(overlap)/len(t),"fallback_counts":fallback_counts,
        "dynamic_bucket_balance":distribution_report(t,selected,"dynamic_bucket"),"fold_balance":distribution_report(t,selected,"oof_fold"),
    }
    return MatchResult(selected.reset_index(drop=True),audit)
