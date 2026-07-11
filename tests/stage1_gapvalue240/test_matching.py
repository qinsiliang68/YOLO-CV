import numpy as np,pandas as pd
from stage1_gapvalue240.matching import matched_random,global_random

def toy():
    rng=np.random.default_rng(4); rows=[]
    for f in ['00','01']:
        for i in range(100):
            rows.append({'sample_id':f'{f}_{i}','y_true':0,'oof_fold':f,'dynamic_bucket':'learnable_hard' if i<50 else 'ordinary',
                         'mean_p_defect':i/100+rng.normal(0,.01),'correct_rate':1-i/120,'std_p_defect':.05+i/500})
    return pd.DataFrame(rows)

def test_r1_disjoint_and_r2_audited():
    pool=toy(); t=pd.concat([pool[pool.oof_fold=='00'].tail(10),pool[pool.oof_fold=='01'].tail(10)])
    r1=global_random(pool,20,3,set(t.sample_id)); assert not(set(r1.selected.sample_id)&set(t.sample_id))
    r2=matched_random(t,pool,7,max_abs_smd=.1,allow_forced_overlap=True)
    assert len(r2.selected)==20 and r2.selected.sample_id.nunique()==20
    assert r2.audit['max_abs_smd']<=.1000001
    assert r2.audit['forced_overlap_count']>=0
