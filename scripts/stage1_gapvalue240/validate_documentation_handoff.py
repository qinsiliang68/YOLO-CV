from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
import argparse, json
from stage1_gapvalue240.campaign_documentation_validation import validate_documentation_handoff

def main(argv=None):
    p=argparse.ArgumentParser(description='Validate Stage1 dynamic-campaign UTF-8 handoff documentation.')
    p.add_argument('--document',action='append',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args(argv)
    report=validate_documentation_handoff(a.document,output_path=a.output)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
