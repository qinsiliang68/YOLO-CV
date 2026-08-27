from __future__ import annotations
import sys, itertools, json
from pathlib import Path
import numpy as np
sys.path.insert(0, '/mnt/data/review_stage1_fresh/Stage1_BudgetedReplay_Learnability_20260809/src')
from stage1_replay.metrics import operating_point_metrics as op_old, confusion_at_threshold as c_old
from stage1_replay.eval.metrics import grouped_operating_curve, tn_at_fn_limit, fn_at_tn_target, confusion_at_threshold as c_new

rng=np.random.default_rng(123)
fail=[]
checks=0
for n in range(2,18):
    for rep in range(30):
        y=rng.integers(0,2,n)
        if len(np.unique(y))<2: continue
        # deliberately create ties
        p=np.round(rng.random(n), decimals=rng.integers(0,4))
        npos=(y==1).sum(); nneg=(y==0).sum()
        for max_fn in sorted(set([0, min(1,npos-1), max(0,npos//2), npos-1])):
            for target_tn in sorted(set([1, min(2,nneg), max(1,nneg//2), nneg])):
                checks+=1
                old=op_old(y,p,max_fn=max_fn,target_tn=target_tn)
                # brute thresholds: -inf, +inf, every score and nextafter(score,+inf)
                ts=[-np.inf,np.inf]
                for s in np.unique(p): ts += [float(s), float(np.nextafter(s,np.inf))]
                conf=[c_old(y,p,t) for t in ts]
                feasible_fn=[x for x in conf if x.fn<=max_fn]
                best_tn=max(x.tn for x in feasible_fn)
                if old['TN_at_FN_limit']!=best_tn:
                    fail.append(('old_fn',y.tolist(),p.tolist(),max_fn,target_tn,old,best_tn))
                    break
                feasible_tn=[x for x in conf if x.tn>=target_tn]
                best_fn=min(x.fn for x in feasible_tn)
                # old threshold_for_target_tn chooses smallest threshold to meet TN rank, which also minimizes FN monotonically? Increasing threshold increases both TN and FN, so minimum FN under TN>=target is at smallest feasible threshold.
                if old['FN_at_TN_target']!=best_fn:
                    fail.append(('old_tn',y.tolist(),p.tolist(),max_fn,target_tn,old,best_fn))
                    break
                # new API
                fnpt=tn_at_fn_limit(y,p,max_fn=max_fn)
                tnpt=fn_at_tn_target(y,p,target_tn=target_tn)
                if fnpt.tn != best_tn:
                    fail.append(('new_fn',y.tolist(),p.tolist(),max_fn,target_tn,fnpt.to_dict(),best_tn))
                    break
                if tnpt.fn != best_fn:
                    fail.append(('new_tn',y.tolist(),p.tolist(),max_fn,target_tn,tnpt.to_dict(),best_fn))
                    break
            if fail: break
        if fail: break
    if fail: break
print(json.dumps({'checks':checks,'fail_count':len(fail),'first_failure': fail[0] if fail else None}, indent=2))
