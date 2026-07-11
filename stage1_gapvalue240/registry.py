from __future__ import annotations
import json,os,time
from pathlib import Path
from typing import Any


def append_registry(path:str|Path,event:dict[str,Any])->None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    rec=dict(event); rec.setdefault("timestamp_unix",time.time())
    data=(json.dumps(rec,sort_keys=True,ensure_ascii=False)+"\n").encode()
    fd=os.open(path,os.O_APPEND|os.O_CREAT|os.O_WRONLY,0o644)
    try: os.write(fd,data); os.fsync(fd)
    finally: os.close(fd)
