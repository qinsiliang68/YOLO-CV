from __future__ import annotations
import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
import argparse,json
from pathlib import Path
from stage1_gapvalue240.contract import load_contract
from stage1_gapvalue240.machine import load_machine_config
from stage1_gapvalue240.prepare import prepare_all

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--machine-config',required=True); a=p.parse_args(argv)
    m=load_machine_config(a.machine_config); c=load_contract(m.path_value('repo_root')/'configs/stage1_gapvalue240/EXPERIMENT_CONTRACT.yaml')
    result=prepare_all(m.path_value('repo_root'),m.path_value('oof_raw_root'),m.path_value('artifact_root'),c)
    print(json.dumps(result,indent=2,ensure_ascii=False)); return 0
if __name__=='__main__': raise SystemExit(main())
