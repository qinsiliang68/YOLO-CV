from __future__ import annotations
import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
import argparse,json
from stage1_gapvalue240.cleanup import execute_cleanup

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--manifest',required=True); p.add_argument('--execute',action='store_true'); a=p.parse_args(argv)
    print(json.dumps(execute_cleanup(a.root,a.manifest,not a.execute),indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
