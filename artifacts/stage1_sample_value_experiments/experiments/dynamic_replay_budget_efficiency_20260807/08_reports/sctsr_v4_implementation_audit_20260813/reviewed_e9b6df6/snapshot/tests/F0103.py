import json,os,subprocess,sys
from pathlib import Path

def run(root,*args):
    env=dict(os.environ,PYTHONPATH=str(root),SCTSR_ALLOW_SYNTHETIC_COLUMNAR_FALLBACK='1')
    return subprocess.run([sys.executable,*map(str,args)],cwd=root,env=env,text=True,capture_output=True)

def test_validate_contract_cli(repository_root,tmp_path):
    out=tmp_path/'receipt.json';r=run(repository_root,repository_root/'scripts/stage1_sctsr_v4/validate_contract.py','--contract',repository_root/'configs/stage1_sctsr_v4/contract_v1.json','--arms',repository_root/'configs/stage1_sctsr_v4/arms_phase1_v1.json','--schemas',repository_root/'configs/stage1_sctsr_v4/schema_registry_v1.json','--output',out);assert r.returncode==0,r.stderr;assert json.loads(out.read_text())['status']=='PASS'

def test_formal_cli_rejected_without_release(repository_root,tmp_path):
    out=tmp_path/'receipt.json';r=run(repository_root,repository_root/'scripts/stage1_sctsr_v4/run_common_parent.py','--repository-root',repository_root,'--output-root',tmp_path/'parent','--training-seed','1','--execution-mode','formal','--output',out);assert r.returncode!=0;assert json.loads(out.read_text())['error']['code']=='FORMAL_RELEASE_NOT_AUTHORIZED'

def test_validate_run_cli(repository_root,tmp_path,canary_root):
    out=tmp_path/'receipt.json';r=run(repository_root,repository_root/'scripts/stage1_sctsr_v4/validate_run.py','--run-root',canary_root,'--allow-synthetic-columnar-fallback','--output',out);assert r.returncode==0,r.stderr;assert json.loads(out.read_text())['result']['artifact_count']>100
