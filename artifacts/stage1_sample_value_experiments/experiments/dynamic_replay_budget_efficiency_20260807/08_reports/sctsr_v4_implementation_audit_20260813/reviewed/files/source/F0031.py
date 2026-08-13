from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from stage1_sctsr_v4.cli_support import add_output_argument,require_receipt_outside_artifact_root,run_cli
from stage1_sctsr_v4.run_validation import validate_run_tree

def main()->int:
    p=argparse.ArgumentParser(description='Read-only SCTSR run audit');p.add_argument('--run-root',type=Path,required=True);p.add_argument('--allow-synthetic-columnar-fallback',action='store_true');add_output_argument(p);a=p.parse_args()
    def action():
        require_receipt_outside_artifact_root(a.output,a.run_root)
        return validate_run_tree(a.run_root,allow_synthetic_portable_fallback=a.allow_synthetic_columnar_fallback)
    return run_cli('validate_run',a.output,action)
if __name__=='__main__':raise SystemExit(main())
