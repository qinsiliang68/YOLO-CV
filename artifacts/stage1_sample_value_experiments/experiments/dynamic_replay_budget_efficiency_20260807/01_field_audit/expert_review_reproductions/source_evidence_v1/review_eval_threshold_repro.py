from __future__ import annotations
import json
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, '/mnt/data/review_stage1_fresh/Stage1_BudgetedReplay_Learnability_20260809/src')
from stage1_replay.eval.pipeline import evaluate_prediction_sets
from stage1_replay.eval.metrics import select_threshold_on_validation, confusion_at_threshold
from stage1_replay.metrics import threshold_for_target_tn

# Keep calibration effectively monotone; rankings/threshold distinction is the issue.
def frame(name, labels, probs):
    return pd.DataFrame({'sample_id':[f'{name}_{i}' for i in range(len(labels))], 'label':labels, 'p_defect':probs})

# Validation: 4 positives and 6 negatives. max_fn=1 permits threshold around 0.70;
# target_tn=5 requires threshold around 0.85. These are intentionally distinct.
val_cal = frame('cal', [1,1,1,1,0,0,0,0,0,0], [.95,.85,.75,.65,.70,.60,.50,.40,.30,.20])
val_op  = frame('op',  [1,1,1,1,0,0,0,0,0,0], [.95,.85,.75,.65,.90,.80,.70,.60,.50,.40])
test    = frame('test',[1,1,1,1,0,0,0,0,0,0], [.96,.86,.76,.66,.91,.81,.71,.61,.51,.41])
res, frames = evaluate_prediction_sets(val_cal=val_cal,val_op=val_op,test=test,calibration_method='temperature',max_fn=1,target_tn=5,canonical_source='valop_fixed_threshold')
op=frames['val_op']; tst=frames['test']
fn_thr=select_threshold_on_validation(op.label,op.p_defect_calibrated,max_fn=1)
tn_thr=threshold_for_target_tn(op.label.to_numpy(),op.p_defect_calibrated.to_numpy(),5)
fn_conf=confusion_at_threshold(tst.label,tst.p_defect_calibrated,fn_thr).to_dict()
tn_conf=confusion_at_threshold(tst.label,tst.p_defect_calibrated,tn_thr).to_dict()
out={
 'pipeline_canonical':{'TN_at_FN1':res['TN_at_FN1'],'FN_at_TN5':res['FN_at_TN5']},
 'fn_threshold':float(fn_thr),'tn_threshold':float(tn_thr),
 'test_at_fn_threshold':fn_conf,'test_at_tn_threshold':tn_conf,
 'bug_demonstrated': bool(fn_thr != tn_thr and res['FN_at_TN5'] == fn_conf['fn'] and fn_conf['fn'] != tn_conf['fn'])
}
Path('/mnt/data/review_eval_threshold_repro.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
