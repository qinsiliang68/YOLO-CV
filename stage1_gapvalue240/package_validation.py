from __future__ import annotations
import importlib.util,json
from pathlib import Path
import pandas as pd
from .contract import load_contract,validate_contract_semantics
from .util import atomic_write_json,sha256_file


def validate_package(root:str|Path,output:str|Path)->dict:
    root=Path(root).resolve(); issues=[]; checks={}
    c=load_contract(root/'configs/stage1_gapvalue240/EXPERIMENT_CONTRACT.yaml'); sem=validate_contract_semantics(c)
    if sem: issues+=sem
    wrappers=list((root/'scripts/stage1_gapvalue240/runs').glob('run_*.py')); checks['run_wrapper_count']=len(wrappers)
    if len(wrappers)!=240: issues.append(f'Expected 240 run wrappers, got {len(wrappers)}')
    matrix=root/'artifacts/stage1_sample_value_experiments/contracts/gapvalue240_v1_1/planned/planned_run_slot_matrix_v1_1.csv'
    if matrix.exists():
        d=pd.read_csv(matrix); checks['planned_runs']=len(d); checks['planned_triads']=d.triad_id.nunique()
        if len(d)!=240 or d.triad_id.nunique()!=80: issues.append('Planned matrix count mismatch')
    else: issues.append('Missing planned matrix')
    for rel in ['stage1_gapvalue240/__init__.py','configs/stage1_gapvalue240/EXPERIMENT_CONTRACT.yaml',
                'schemas/stage1_gapvalue240/frozen_matrix.schema.json','docs/stage1_gapvalue240/INSTALL_AND_QUICKSTART.md',
                'tests/stage1_gapvalue240/test_package.py']:
        if not (root/rel).exists(): issues.append(f'Missing required file: {rel}')
    report={'status':'PASS' if not issues else 'FAIL','issues':issues,'checks':checks,'contract_sha256':c.sha256}
    atomic_write_json(output,report,overwrite=True)
    if issues: raise RuntimeError(f'Package validation failed: {issues[:5]}')
    return report
