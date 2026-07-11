from __future__ import annotations
import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
import argparse,json
from stage1_gapvalue240.machine import load_machine_config
from stage1_gapvalue240.aggregate import aggregate_and_write
from stage1_gapvalue240.reporting import generate_markdown_report,generate_html_report

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--machine-config',required=True); a=p.parse_args(argv)
    m=load_machine_config(a.machine_config); ar=m.path_value('artifact_root'); out=m.path_value('output_root')
    agg=out/'aggregate'; r=aggregate_and_write(out,ar/'generated/frozen_experiment_matrix.csv',agg)
    generate_markdown_report(agg,agg/'FINAL_REPORT.md'); generate_html_report(agg,agg/'FINAL_REPORT.html')
    print(json.dumps(r,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
