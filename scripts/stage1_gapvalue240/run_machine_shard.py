from __future__ import annotations
import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
import argparse,json
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from stage1_gapvalue240.machine import load_machine_config
from stage1_gapvalue240.matrix import load_matrix
from stage1_gapvalue240.run_engine import prepare_run,train_run,evaluate_run,validate_run
from stage1_gapvalue240.scheduling import order_triad_rows

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--machine-config',required=True); p.add_argument('--continue-on-error',action='store_true'); a=p.parse_args(argv)
    m=load_machine_config(a.machine_config); ar=m.path_value('artifact_root')
    shard=ar/'generated/machine_shards'/f"{m.data['machine_id']}_jobs.csv"; matrix=load_matrix(ar/'generated/frozen_experiment_matrix.csv')
    jobs=pd.read_csv(shard); slots=[]
    for triad in jobs.triad_id.drop_duplicates(): slots.extend(order_triad_rows(matrix[matrix.triad_id==triad]).run_slot.astype(str).tolist())
    if not slots: print(json.dumps({'machine_id':m.data['machine_id'],'runs':0,'role':'reserve'})); return 0
    results=[]
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='next-run-stager') as ex:
        current=prepare_run(slots[0],a.machine_config)
        for i,slot in enumerate(slots):
            future=ex.submit(prepare_run,slots[i+1],a.machine_config) if i+1<len(slots) else None
            try:
                train_run(slot,a.machine_config,current.attempt_id); evaluate_run(slot,a.machine_config,current.attempt_id)
                final=validate_run(slot,a.machine_config,current.attempt_id); results.append({'run_slot':slot,'attempt_dir':str(final.attempt_dir)})
            except Exception as exc:
                results.append({'run_slot':slot,'error':repr(exc)})
                if not a.continue_on_error: raise
            if future is not None: current=future.result()
    print(json.dumps(results,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
