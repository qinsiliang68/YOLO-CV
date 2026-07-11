from __future__ import annotations
import csv,subprocess,threading,time
from pathlib import Path

class ResourceMonitor:
    def __init__(self,path:Path,gpu_id:str|int,nvidia_smi:str="nvidia-smi",interval:float=10.0):
        self.path=path; self.gpu_id=str(gpu_id); self.nvidia_smi=nvidia_smi; self.interval=interval; self.stop_event=threading.Event(); self.thread=None
    def start(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.thread=threading.Thread(target=self._run,daemon=True); self.thread.start()
    def stop(self):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=self.interval+2)
    def _run(self):
        with self.path.open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f); w.writerow(['timestamp_unix','gpu_util_pct','memory_used_mb','memory_total_mb','temperature_c','power_w','status'])
            while not self.stop_event.is_set():
                try:
                    q='utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw'
                    r=subprocess.run([self.nvidia_smi,'--id='+self.gpu_id,'--query-gpu='+q,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                    vals=[x.strip() for x in r.stdout.strip().split(',')]; w.writerow([time.time(),*vals,'OK' if r.returncode==0 else 'ERROR'])
                except Exception as exc: w.writerow([time.time(),'','','','','',f'UNAVAILABLE:{exc}'])
                f.flush(); self.stop_event.wait(self.interval)
