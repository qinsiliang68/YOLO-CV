from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .statistics import paired_summary
from .util import atomic_write_bytes,atomic_write_json
from .validation import verify_permanent_artifact_manifest


def collect_validated(
    output_root:str|Path|list[str|Path],
    matrix_path:str|Path,
    expected_identity:dict|None=None,
    expected_per_run:dict[str,dict]|None=None,
)->pd.DataFrame:
    output_roots=[Path(x) for x in output_root] if isinstance(output_root,(list,tuple,set)) else [Path(output_root)]
    matrix=pd.read_csv(matrix_path,dtype={'run_slot':'string','triad_id':'string','arm':'string'})
    rows=[]
    for slot in matrix.run_slot:
        attempts=[]
        for root in output_roots:
            parent=root/'runs'/str(slot)
            if not parent.exists(): continue
            for attempt in parent.glob('attempt_*'):
                if not attempt.is_dir() or attempt.name.endswith('.inprogress'): continue
                status_json=attempt/'08_status/status.json'
                state=None
                if status_json.exists(): state=json.loads(status_json.read_text(encoding='utf-8')).get('state')
                elif (attempt/'08_status/VALIDATED').exists(): state='VALIDATED'
                if state!='VALIDATED': continue
                identity_path=attempt/'00_identity/run_identity.json'
                if not identity_path.exists(): continue
                ident=json.loads(identity_path.read_text(encoding='utf-8'))
                if bool(ident.get('dry_run')): continue
                if expected_identity and any(ident.get(k)!=v for k,v in expected_identity.items()): continue
                per_run=(expected_per_run or {}).get(str(slot),{})
                if any(ident.get(k)!=v for k,v in per_run.items()): continue
                postflight=attempt/'07_validation/postflight_report.json'
                artifact_manifest=attempt/'07_validation/artifact_manifest.csv'
                if not postflight.is_file() or not artifact_manifest.is_file(): continue
                try:
                    if json.loads(postflight.read_text(encoding='utf-8')).get('status')!='PASS': continue
                except (OSError,ValueError,TypeError): continue
                try:
                    verify_permanent_artifact_manifest(attempt,artifact_manifest)
                except Exception as exc:
                    raise RuntimeError(f'Validated artifact checksum failure for {slot}: {attempt}: {exc}') from exc
                attempts.append((attempt,ident))
        if len(attempts)>1: raise RuntimeError(f'Multiple validated attempts for {slot}: {attempts}')
        if not attempts: continue
        a,ident=attempts[0]; m=json.loads((a/'05_metrics/operational_metrics.json').read_text())
        rows.append({'run_slot':slot,'attempt_dir':str(a),'attempt_id':ident['attempt_id'],
                     'resume_mode':ident.get('resume_mode','none'),'resume_count':int(ident.get('resume_count',0)),
                     'TN_at_FN95':m['TN_at_FN95']['actual_TN'],'FN_at_TN68253':m['FN_at_TN68253']['actual_FN'],
                     'actual_FN_at_FN95':m['TN_at_FN95']['actual_FN'],'actual_TN_at_TN68253':m['FN_at_TN68253']['actual_TN'],
                     'gap_q68_q050':m['gap_q68_q050'],'tail_gap_q90_q05':m['tail_gap_q90_q05']})
    results=pd.DataFrame(rows,columns=[
        'run_slot','attempt_dir','attempt_id','resume_mode','resume_count','TN_at_FN95',
        'FN_at_TN68253','actual_FN_at_FN95','actual_TN_at_TN68253','gap_q68_q050','tail_gap_q90_q05'
    ])
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


def aggregate_and_write(
    output_root:str|Path|list[str|Path],
    matrix_path:str|Path,
    aggregate_dir:str|Path,
    *,
    expected_identity:dict|None=None,
    expected_per_run:dict[str,dict]|None=None,
)->dict:
    aggregate_dir=Path(aggregate_dir); aggregate_dir.mkdir(parents=True,exist_ok=True)
    runs=collect_validated(
        output_root,matrix_path,expected_identity=expected_identity,expected_per_run=expected_per_run
    ); deltas,summaries=paired_results(runs)
    atomic_write_bytes(aggregate_dir/'run_results.csv',runs.to_csv(index=False).encode(),overwrite=True)
    atomic_write_bytes(aggregate_dir/'paired_deltas.csv',deltas.to_csv(index=False).encode(),overwrite=True)
    atomic_write_bytes(aggregate_dir/'paired_summaries.csv',summaries.to_csv(index=False).encode(),overwrite=True)
    report={'validated_runs':int(runs.TN_at_FN95.notna().sum()),'complete_triads':int(deltas.triad_id.nunique()) if len(deltas) else 0,
            'expected_validated_runs':240,'expected_triads':80,'status':'COMPLETE' if runs.TN_at_FN95.notna().sum()==240 else 'PARTIAL'}
    atomic_write_json(aggregate_dir/'aggregate_status.json',report,overwrite=True); return report
