from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def one_sided_t_ci(values,confidence=.95,side='lower')->float:
    x=np.asarray(values,dtype=float)
    if len(x)<2: return float('nan')
    mean=float(x.mean()); se=float(x.std(ddof=1)/math.sqrt(len(x))); q=float(student_t.ppf(confidence,len(x)-1))
    return mean-q*se if side=='lower' else mean+q*se


def paired_summary(delta_fn,delta_tn,upper_fn_max=2,worst_fn_max=5)->dict:
    dfn=np.asarray(delta_fn,dtype=float); dtn=np.asarray(delta_tn,dtype=float)
    upper=one_sided_t_ci(dfn,.95,'upper'); lower=one_sided_t_ci(dtn,.95,'lower')
    return {
        'n':len(dfn),'mean_delta_FN':float(dfn.mean()),'std_delta_FN':float(dfn.std(ddof=1)) if len(dfn)>1 else 0.0,
        'worst_delta_FN':float(dfn.max()),'FN_one_sided_95_upper':upper,
        'safety_noninferior':bool(upper<=upper_fn_max and dfn.max()<=worst_fn_max),
        'strict_safety_improvement':bool(dfn.mean()<=0),
        'mean_delta_TN':float(dtn.mean()),'std_delta_TN':float(dtn.std(ddof=1)) if len(dtn)>1 else 0.0,
        'worst_delta_TN':float(dtn.min()),'TN_one_sided_95_lower':lower,
        'confirmed_TN_improvement':bool(upper<=upper_fn_max and dfn.max()<=worst_fn_max and lower>0),
        'FN_win_rate':float((dfn<0).mean()),'TN_win_rate':float((dtn>0).mean()),
    }
