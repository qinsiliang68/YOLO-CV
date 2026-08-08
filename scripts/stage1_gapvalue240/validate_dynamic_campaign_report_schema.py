from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
import argparse
import json
from stage1_gapvalue240.campaign_schema_validation import validate_report_schema

def main(argv=None):
    parser=argparse.ArgumentParser(description="Validate one dynamic-campaign JSON report against a versioned schema.")
    parser.add_argument('--report',required=True)
    parser.add_argument('--schema',required=True)
    args=parser.parse_args(argv)
    payload=validate_report_schema(args.report,args.schema)
    print(json.dumps({'status':'PASS','schema_version':payload.get('schema_version')},indent=2))
    return 0
if __name__=='__main__':
    raise SystemExit(main())
