from __future__ import annotations
from pathlib import Path
import shutil
from .util import atomic_write_json


def plan_cleanup(root:str|Path)->dict:
    root=Path(root); candidates=[]
    for pattern,reason in [('**/__pycache__','python cache'),('**/*.pyc','python bytecode'),('**/*.tmp','temporary file'),('**/work','completed-run working directory')]:
        for p in root.glob(pattern):
            if p.exists(): candidates.append({'path':str(p),'reason':reason,'is_dir':p.is_dir()})
    return {'root':str(root),'candidates':candidates,'count':len(candidates)}


def execute_cleanup(root:str|Path,manifest_path:str|Path,dry_run:bool=True)->dict:
    plan=plan_cleanup(root); plan['dry_run']=dry_run
    atomic_write_json(manifest_path,plan,overwrite=True)
    if not dry_run:
        for item in plan['candidates']:
            p=Path(item['path'])
            if p.is_dir(): shutil.rmtree(p,ignore_errors=True)
            elif p.exists(): p.unlink()
    return plan
