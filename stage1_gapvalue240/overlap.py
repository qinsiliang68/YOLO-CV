from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class OverlapDecision:
    candidate: str
    retained_as: str
    max_overlap: float
    replaced: bool
    compared_against: str | None


def overlap_rate(a:pd.DataFrame,b:pd.DataFrame,k:int=3000)->float:
    return len(set(a.head(k).sample_id)&set(b.head(k).sample_id))/k


def apply_overlap_gate(main_name:str, rankings:dict[str,pd.DataFrame], candidates:list[str], replacements:list[str],
                       threshold:float=.95,k:int=3000)->tuple[list[str],list[OverlapDecision]]:
    retained=[main_name]; decisions=[]; repl_iter=iter(replacements)
    for name in candidates:
        max_ov=-1.0; against=None
        for r in retained:
            ov=overlap_rate(rankings[name],rankings[r],k)
            if ov>max_ov: max_ov,against=ov,r
        if max_ov>=threshold:
            replacement=next(repl_iter,None)
            if replacement is None: raise RuntimeError("Overlap replacements exhausted")
            retained.append(replacement); decisions.append(OverlapDecision(name,replacement,max_ov,True,against))
        else:
            retained.append(name); decisions.append(OverlapDecision(name,name,max_ov,False,against))
    return retained,decisions
