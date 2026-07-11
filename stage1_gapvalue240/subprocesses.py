from __future__ import annotations
import os,subprocess,time
from pathlib import Path
from typing import Sequence
from .errors import ExternalCommandError
from .util import atomic_write_json


def run_logged(command:Sequence[str],cwd:str|Path,log_path:str|Path,env:dict[str,str]|None=None,timeout:int|None=None)->dict:
    cwd=Path(cwd); log_path=Path(log_path); log_path.parent.mkdir(parents=True,exist_ok=True)
    started=time.time()
    merged=os.environ.copy(); merged.update(env or {})
    with log_path.open("wb") as log:
        proc=subprocess.Popen(list(map(str,command)),cwd=cwd,stdout=log,stderr=subprocess.STDOUT,env=merged)
        try: code=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(); raise ExternalCommandError(f"Command timed out: {command}")
    result={"command":list(map(str,command)),"cwd":str(cwd),"returncode":code,"duration_seconds":time.time()-started,"log":str(log_path)}
    atomic_write_json(log_path.with_suffix(log_path.suffix+".result.json"),result,overwrite=True)
    if code!=0: raise ExternalCommandError(f"Command failed ({code}); see {log_path}")
    return result
