from __future__ import annotations
import argparse,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from stage1_sctsr_v4.cli_support import add_output_argument,require_receipt_outside_artifact_root,run_cli
from stage1_sctsr_v4.synthetic_canary import run_synthetic_canary

def main()->int:
    p=argparse.ArgumentParser(description='Run real forward/backward SCTSR synthetic canary')
    p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--repository-root',type=Path,default=Path.cwd());p.add_argument('--training-seed',type=int,default=20260812);p.add_argument('--allow-synthetic-columnar-fallback',action='store_true');add_output_argument(p);a=p.parse_args()
    def action():
        require_receipt_outside_artifact_root(a.output,a.artifact_root)
        if a.allow_synthetic_columnar_fallback:os.environ['SCTSR_ALLOW_SYNTHETIC_COLUMNAR_FALLBACK']='1'
        return run_synthetic_canary(a.artifact_root,repository_root=a.repository_root,training_seed=a.training_seed)
    return run_cli('run_synthetic_canary',a.output,action)
if __name__=='__main__':raise SystemExit(main())
