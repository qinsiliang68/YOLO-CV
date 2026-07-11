from __future__ import annotations
from pathlib import Path
import html
import pandas as pd
from .util import atomic_write_text


def generate_markdown_report(aggregate_dir:str|Path,output:str|Path)->Path:
    aggregate_dir=Path(aggregate_dir); summaries=pd.read_csv(aggregate_dir/'paired_summaries.csv') if (aggregate_dir/'paired_summaries.csv').exists() else pd.DataFrame()
    lines=['# Stage-1 GapValue 240-Run Results','',f'Validated summaries: {len(summaries)}','']
    if len(summaries): lines += ['## Paired summaries','',summaries.to_markdown(index=False),'']
    else: lines += ['No complete paired summaries are available yet.','']
    return atomic_write_text(output,'\n'.join(lines),overwrite=True)


def generate_html_report(aggregate_dir:str|Path,output:str|Path)->Path:
    aggregate_dir=Path(aggregate_dir); tables=[]
    for name in ['run_results.csv','paired_deltas.csv','paired_summaries.csv']:
        p=aggregate_dir/name
        if p.exists(): tables.append(f'<h2>{html.escape(name)}</h2>'+pd.read_csv(p).to_html(index=False,border=1))
    body='<!doctype html><meta charset="utf-8"><title>GapValue 240</title><style>body{font-family:Arial;margin:24px}table{border-collapse:collapse;font-size:12px}th,td{padding:4px}</style><h1>Stage-1 GapValue 240-Run Results</h1>'+''.join(tables)
    return atomic_write_text(output,body,overwrite=True)
