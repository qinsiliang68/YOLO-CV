from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
import argparse, json
from stage1_gapvalue240.campaign_resource_validation import build_epoch_resource_log

def main(argv=None):
    p=argparse.ArgumentParser(description='Aggregate sampled resources into exact per-epoch Stage1 telemetry.')
    p.add_argument('--audit',required=True)
    p.add_argument('--sampled-resource-log',required=True)
    p.add_argument('--output-csv',required=True)
    p.add_argument('--validation-output',required=True)
    p.add_argument('--expected-epochs',type=int,default=200)
    a=p.parse_args(argv)
    report=build_epoch_resource_log(a.audit,a.sampled_resource_log,output_csv=a.output_csv,validation_output=a.validation_output,expected_epochs=a.expected_epochs)
    print(json.dumps(report,indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
