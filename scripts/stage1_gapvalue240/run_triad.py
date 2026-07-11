from __future__ import annotations
import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
import argparse,json
from stage1_gapvalue240.machine import load_machine_config
from stage1_gapvalue240.run_engine import run_all
from stage1_gapvalue240.matrix import load_matrix
from stage1_gapvalue240.scheduling import order_triad_rows

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--machine-config',required=True); p.add_argument('--triad-id',required=True); a=p.parse_args(argv)
    m=load_machine_config(a.machine_config); matrix=load_matrix(m.path_value('artifact_root')/'generated/frozen_experiment_matrix.csv')
    rows=order_triad_rows(matrix[matrix.triad_id==a.triad_id])
    if len(rows)!=3: raise RuntimeError(f'Expected 3 arms for {a.triad_id}')
    results=[]
    for slot in rows.run_slot: results.append(run_all(str(slot),a.machine_config).__dict__)
    print(json.dumps(results,default=str,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
