import sys, json
sys.path.insert(0, '/mnt/data/review_stage1_fresh/Stage1_BudgetedReplay_Learnability_20260809/src')
import pandas as pd
from stage1_replay.selection.diversity import cluster_quota_select
rows=[]
# A top two share vA and cap video=1, forcing only one A; B has distinct videos.
for sid,cluster,score,vid in [
 ('a1','A',10,'vA'),('a2','A',9,'vA'),('a3','A',1,'vA'),
 ('b1','B',8,'vB1'),('b2','B',7,'vB2'),('b3','B',6,'vB3')]:
 rows.append(dict(sample_id=sid,cluster=cluster,score=score,video_id=vid))
df=pd.DataFrame(rows)
out=cluster_quota_select(df,k=4,cluster_column='cluster',quota='equal',caps={'video_id':1})
print(out.to_json(orient='records'))
print(out['cluster'].value_counts().to_dict())
with open('/mnt/data/review_quota_repro.json','w') as f: json.dump({'selected':out.to_dict('records'),'cluster_counts':out['cluster'].value_counts().to_dict()},f,indent=2)
