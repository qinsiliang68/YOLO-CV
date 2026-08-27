from __future__ import annotations
import sys, json, tempfile
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, '/mnt/data/review_stage1_fresh/Stage1_BudgetedReplay_Learnability_20260809/src')
from stage1_replay.data.convert import convert_long_oof_to_mmap
from stage1_replay.data.oof import load_mmap_dynamics, load_long_dynamics
from stage1_replay.features.dynamics import compute_dynamics_features
from stage1_replay.features.reliability import apply_reliability_gate
from stage1_replay.exceptions import SchemaError

rows=[]
# Proper two-fold OOF repeated over two seeds: each sample appears only under its held-out fold in each seed.
assign={'s0':0,'s1':0,'s2':1,'s3':1}
labels={'s0':0,'s1':1,'s2':0,'s3':1}
for seed in [11,22]:
    for sid,fold in assign.items():
        y=labels[sid]
        for epoch in [1,2,3]:
            p=(0.2 + 0.2*epoch) if y else (0.8 - 0.2*epoch)
            rows.append({'sample_id':sid,'label':y,'epoch':epoch,'p_defect':p,'seed':seed,'fold':fold})
df=pd.DataFrame(rows)
out={'rows':len(df),'proper_oof_expected_trajectories_per_sample':2,'fold_assignments':assign}
with tempfile.TemporaryDirectory() as td:
    td=Path(td); src=td/'oof.csv'; df.to_csv(src,index=False)
    try:
        convert_long_oof_to_mmap(src, td/'strict')
        out['strict_result']='unexpected_success'
    except Exception as e:
        out['strict_result']=f'{type(e).__name__}: {e}'
    result=convert_long_oof_to_mmap(src, td/'allow', allow_missing=True)
    cube=load_mmap_dynamics(result['metadata_path'])
    out['cube_shape']=list(cube.shape)
    out['trajectory_ids']=cube.trajectory_ids.tolist()
    out['missing_cells']=int(np.isnan(cube.p_defect).sum())
    out['observed_fraction_global']=float(np.isfinite(cube.p_defect).mean())
    feat=compute_dynamics_features(cube, windows=[('early',1,1),('late',2,3)])
    feat=apply_reliability_gate(feat, reference_loss_max_quantile=.95, never_learned_max=.75, minimum_observed_fraction=.98, minimum_late_consistency=.20, aum_min_quantile=.01, mode='exclude')
    cols=['sample_id','oof_trajectory_count','oof_observed_fraction','oof_never_learned_fraction','oof_late_stable_correct_fraction','eligible_reliability','reliability_flag_count']
    out['features']=feat[cols].to_dict('records')
    out['eligible_count']=int(feat['eligible_reliability'].sum())
print(json.dumps(out,indent=2))
