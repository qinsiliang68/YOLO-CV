import json, yaml
from pathlib import Path
root=Path('/mnt/data/review_stage1_fresh/Stage1_BudgetedReplay_Learnability_20260809')
out={}
p=yaml.safe_load((root/'configs/project_learnability.example.yaml').read_text())
e=p['experiment']; out['project_learnability']={'methods':len(e['methods']),'budgets':len(e['budgets']),'seeds':len(e['seeds']),'triads':len(e['methods'])*len(e['budgets'])*len(e['seeds']),'arms':3*len(e['methods'])*len(e['budgets'])*len(e['seeds'])}
s=yaml.safe_load((root/'configs/study_matrix.example.yaml').read_text()); out['study_matrix']={'methods':len(s['methods']),'budgets':len(s['budgets']),'seeds':len(s['seeds']),'triads':len(s['methods'])*len(s['budgets'])*len(s['seeds']),'arms':3*len(s['methods'])*len(s['budgets'])*len(s['seeds'])}
Path('/mnt/data/review_matrix_counts.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
