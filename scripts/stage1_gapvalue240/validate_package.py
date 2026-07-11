from __future__ import annotations
import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
import argparse,json
from stage1_gapvalue240.package_validation import validate_package

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--root',default='.'); p.add_argument('--output',default='PACKAGE_VALIDATION.json'); a=p.parse_args(argv)
    print(json.dumps(validate_package(a.root,a.output),indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
