from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import pandas as pd

from .contract import Contract
from .errors import ValidationError
from .util import sha256_file,atomic_write_json


def reference_root(repo_root:Path,contract:Contract)->Path:
    return repo_root/contract.data["frozen_inputs"]["reference_root_relative"]


def validate_frozen_inputs(repo_root:Path,contract:Contract,output:Path|None=None)->dict[str,Any]:
    root=reference_root(repo_root,contract); issues=[]; rows=[]
    for name,spec in contract.data["frozen_inputs"]["files"].items():
        p=root/name
        if not p.exists(): issues.append(f"Missing frozen input: {p}"); continue
        h=sha256_file(p); ok=h==str(spec["sha256"]).upper()
        row={"file":name,"path":str(p),"size_bytes":p.stat().st_size,"sha256":h,"expected_sha256":spec["sha256"],"sha_ok":ok}
        if not ok: issues.append(f"Checksum mismatch: {name}")
        if name.endswith('.csv') and 'rows' in spec:
            count=sum(1 for _ in p.open('rb'))-1; row['rows']=count; row['expected_rows']=spec['rows']; row['rows_ok']=count==spec['rows']
            if count!=spec['rows']: issues.append(f"Row count mismatch: {name}")
        rows.append(row)
    report={"status":"PASS" if not issues else "FAIL","issues":issues,"files":rows}
    if output: atomic_write_json(output,report,overwrite=True)
    if issues: raise ValidationError(f"Frozen input validation failed: {issues[:3]}")
    return report
