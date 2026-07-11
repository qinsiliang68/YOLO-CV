from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .statistics import paired_summary
from .util import atomic_write_bytes,atomic_write_json


def collect_validated(output_root:str|Path,matrix_path:str|Path)->pd.DataFrame:
    output_root=Path(output_root); matrix=pd.read_csv(matrix_path,dtype={'run_slot':'string','triad_id':'string','arm':'string'})
    rows=[]
    for slot in matrix.run_slot:
        parent=output_root/'runs'/str(slot)
        attempts=[]
        if parent.exists():
            for status in parent.glob('attempt_*/08_status/VALIDATED'):
                attempts.append(status.parents[1])
        if len(attempts)>1: raise RuntimeError(f'Multiple validated attempts for {slot}: {attempts}')
        if not attempts: continue
        a=attempts[0]; ident=json.loads((a/'00_identity/run_identity.json').read_text()); m=json.loads((a/'05_metrics/operational_metrics.json').read_text())
        rows.append({'run_slot':slot,'attempt_dir':str(a),'attempt_id':ident['attempt_id'],
                     'TN_at_FN95':m['TN_at_FN95']['actual_TN'],'FN_at_TN68253':m['FN_at_TN68253']['actual_FN'],
                     'actual_FN_at_FN95':m['TN_at_FN95']['actual_FN'],'actual_TN_at_TN68253':m['FN_at_TN68253']['actual_TN'],
                     'gap_q68_q050':m['gap_q68_q050'],'tail_gap_q90_q05':m['tail_gap_q90_q05']})
    results=pd.DataFrame(rows)
    return matrix.merge(results,on='run_slot',how='left',validate='one_to_one')


def paired_results(run_results:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    valid=run_results.dropna(subset=['TN_at_FN95','FN_at_TN68253']).copy()
    wide=valid.pivot(index=['triad_id','condition_slot','condition_id','method','budget','guard_ratio','training_seed','phase'],columns='arm',
                     values=['TN_at_FN95','FN_at_TN68253'])
    complete=wide.dropna().reset_index()
    records=[]
    for _,r in complete.iterrows():
        for control in ['R1','R2']:
            records.append({'triad_id':r['triad_id'],'condition_slot':r['condition_slot'],'condition_id':r['condition_id'],'method':r['method'],
                            'budget':r['budget'],'guard_ratio':r['guard_ratio'],'training_seed':r['training_seed'],'phase':r['phase'],'control':control,
                            'delta_TN':float(r[('TN_at_FN95','T')]-r[('TN_at_FN95',control)]),
                            'delta_FN':float(r[('FN_at_TN68253','T')]-r[('FN_at_TN68253',control)])})
    deltas=pd.DataFrame(records)
    summaries=[]
    if len(deltas):
        for (slot,control),g in deltas.groupby(['condition_slot','control']):
            rec={'condition_slot':slot,'control':control,'method':g.method.iloc[0],'budget':int(g.budget.iloc[0]),'guard_ratio':float(g.guard_ratio.iloc[0])}
            rec.update(paired_summary(g.delta_FN,g.delta_TN)); summaries.append(rec)
        primary=deltas[deltas.condition_slot=='A02']
        if len(primary):
            # A02 discovery and Phase C both carry A02 condition_slot.
            for control,g in primary.groupby('control'):
                if len(g)==8:
                    rec={'condition_slot':'A02_PHASE_C_COMBINED','control':control,'method':'GapCritical-Strict','budget':3000,'guard_ratio':0.0}
                    rec.update(paired_summary(g.delta_FN,g.delta_TN)); summaries.append(rec)
    return deltas,pd.DataFrame(summaries)


def aggregate_and_write(output_root:str|Path,matrix_path:str|Path,aggregate_dir:str|Path)->dict:
    aggregate_dir=Path(aggregate_dir); aggregate_dir.mkdir(parents=True,exist_ok=True)
    runs=collect_validated(output_root,matrix_path); deltas,summaries=paired_results(runs)
    atomic_write_bytes(aggregate_dir/'run_results.csv',runs.to_csv(index=False).encode(),overwrite=True)
    atomic_write_bytes(aggregate_dir/'paired_deltas.csv',deltas.to_csv(index=False).encode(),overwrite=True)
    atomic_write_bytes(aggregate_dir/'paired_summaries.csv',summaries.to_csv(index=False).encode(),overwrite=True)
    report={'validated_runs':int(runs.TN_at_FN95.notna().sum()),'complete_triads':int(deltas.triad_id.nunique()) if len(deltas) else 0,
            'expected_validated_runs':240,'expected_triads':80,'status':'COMPLETE' if runs.TN_at_FN95.notna().sum()==240 else 'PARTIAL'}
    atomic_write_json(aggregate_dir/'aggregate_status.json',report,overwrite=True); return report
