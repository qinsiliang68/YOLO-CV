import pandas as pd
from stage1_gapvalue240.metrics import operational_metrics,tie_safe_sweep

def test_ties_are_not_split():
    df=pd.DataFrame({'sample_id':[f's{i}' for i in range(8)],'y_true':[1,1,1,0,0,0,0,0],'score':[.9,.8,.8,.8,.8,.7,.6,.5]})
    summary,sweep=operational_metrics(df,fn_limit=1,tn_target=3)
    row=sweep[sweep.threshold==.8].iloc[0]
    assert row.tie_group_size==4
    assert row.TP==3 and row.FP==2
    assert summary['metric_version']=='operational_v2_tie_safe'
