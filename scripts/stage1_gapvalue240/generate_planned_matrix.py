from __future__ import annotations
import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
import argparse
from pathlib import Path
from stage1_gapvalue240.contract import load_contract
from stage1_gapvalue240.matrix import build_run_specs,write_matrix

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--contract',required=True); p.add_argument('--output',required=True); p.add_argument('--overwrite',action='store_true'); a=p.parse_args(argv)
    c=load_contract(a.contract); write_matrix(build_run_specs(c),a.output,a.overwrite); print(a.output); return 0
if __name__=='__main__': raise SystemExit(main())
