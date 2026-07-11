from __future__ import annotations
import argparse, json, os, shutil, time, uuid
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .contract import load_contract,Contract
from .errors import GapValueError,ValidationError
from .evaluation import finalize_evaluation
from .integration import trainer_command
from .predictor import predict_split
from .machine import load_machine_config,MachineConfig
from .manifests import build_replay_manifests
from .matrix import load_matrix
from .monitor import ResourceMonitor
from .registry import append_registry
from .runtime import ensure_ultralytics_runtime
from .site_binding import bind_checkpoint
from .status import set_status
from .subprocesses import run_logged
from .util import atomic_write_json,environment_snapshot,sha256_file
from .validation import preflight,postflight

@dataclass(frozen=True)
class PreparedRun:
    run_slot:str
    attempt_id:str
    attempt_dir:Path
    run_row:dict


def _repo_contract(machine:MachineConfig)->Contract:
    return load_contract(machine.path_value("repo_root")/"configs/stage1_gapvalue240/EXPERIMENT_CONTRACT.yaml")

def _artifact_root(machine:MachineConfig)->Path:
    return machine.path_value("artifact_root")

def _matrix(machine:MachineConfig)->pd.DataFrame:
    return load_matrix(_artifact_root(machine)/"generated/frozen_experiment_matrix.csv")

def _row(machine:MachineConfig,run_slot:str)->dict:
    df=_matrix(machine); m=df[df.run_slot==run_slot]
    if len(m)!=1: raise ValidationError(f"Run slot not found uniquely: {run_slot}")
    return m.iloc[0].to_dict()

def _attempt_parent(machine:MachineConfig,run_slot:str)->Path:
    return machine.path_value("output_root")/"runs"/run_slot

def _new_attempt_id(row:dict)->str:
    return time.strftime("%Y%m%dT%H%M%S")+"_"+str(row["arm"])+"_"+uuid.uuid4().hex[:10]

def _find_attempt(machine:MachineConfig,run_slot:str,attempt_id:str)->Path:
    parent=_attempt_parent(machine,run_slot)
    for p in [parent/f"attempt_{attempt_id}.inprogress",parent/f"attempt_{attempt_id}"]:
        if p.exists(): return p
    raise FileNotFoundError(f"Attempt not found: {run_slot}/{attempt_id}")

def _unique_find(root:Path,name:str)->Path:
    hits=list(root.rglob(name))
    if len(hits)!=1: raise ValidationError(f"Expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]

def _link_or_copy(src:Path,dst:Path)->None:
    dst.parent.mkdir(parents=True,exist_ok=True)
    try: os.link(src,dst)
    except OSError: shutil.copy2(src,dst)

def prepare_run(run_slot:str,machine_config:str|Path,attempt_id:str|None=None,allow_new_attempt_after_validated:bool=False)->PreparedRun:
    machine=load_machine_config(machine_config); contract=_repo_contract(machine); row=_row(machine,run_slot)
    parent=_attempt_parent(machine,run_slot); parent.mkdir(parents=True,exist_ok=True)
    validated=list(parent.glob("attempt_*/08_status/VALIDATED"))
    if validated and not allow_new_attempt_after_validated: raise ValidationError(f"Validated attempt already exists for {run_slot}")
    attempt_id=attempt_id or _new_attempt_id(row); attempt=parent/f"attempt_{attempt_id}.inprogress"
    if attempt.exists() or (parent/f"attempt_{attempt_id}").exists(): raise FileExistsError(f"Attempt exists: {attempt_id}")
    for d in ["00_identity","01_manifests","02_logs","03_checkpoints","04_predictions","05_metrics","06_figures","07_validation","08_status","work"]:
        (attempt/d).mkdir(parents=True,exist_ok=True)
    set_status(attempt,"PLANNED",{"run_slot":run_slot,"attempt_id":attempt_id})
    try:
        selection=_artifact_root(machine)/"generated/selections"/run_slot/"selection_manifest.csv"
        if not selection.exists(): raise FileNotFoundError(selection)
        shutil.copy2(selection,attempt/"01_manifests/selection_manifest.csv")
        manifests=build_replay_manifests(machine.path_value("train_manifest"),machine.path_value("normal_train_manifest"),selection,
                                         attempt/"01_manifests",expected_base_total=contract.data["replay"]["base_samples"])
        shutil.copy2(machine.path_value("val_model_defect_manifest"),attempt/"01_manifests/val_model_manifest.csv")
        shutil.copy2(machine.path_value("val_model_normal_manifest"),attempt/"01_manifests/normal_val_model_manifest.csv")
        if bool(machine.data.get("dry_run", False)):
            binding={"full_sha256":"DRY_RUN","created_by":str(machine.data["machine_id"]),"dry_run":True}
        else:
            binding=bind_checkpoint(contract,machine.path_value("base_checkpoint"),_artifact_root(machine)/"generated/site_asset_binding.json",str(machine.data["machine_id"]))
        identity={"run_slot":run_slot,"attempt_id":attempt_id,"run_row":row,"contract_sha256":contract.sha256,
                  "matrix_sha256":sha256_file(_artifact_root(machine)/"generated/frozen_experiment_matrix.csv"),
                  "selection_sha256":sha256_file(selection),"checkpoint_binding":binding,"machine_id":machine.data["machine_id"]}
        atomic_write_json(attempt/"00_identity/run_identity.json",identity)
        atomic_write_json(attempt/"00_identity/environment.json",environment_snapshot())
        preflight(contract,machine,row,selection,manifests.summary_path,attempt/"07_validation/preflight_report.json")
        set_status(attempt,"STAGED",{"preflight":"PASS"})
        append_registry(machine.path_value("output_root")/"registry"/f"{machine.data['machine_id']}.jsonl",{"event":"STAGED","run_slot":run_slot,"attempt_id":attempt_id})
        return PreparedRun(run_slot,attempt_id,attempt,row)
    except Exception as exc:
        set_status(attempt,"FAILED_INPUT",{"error":repr(exc)})
        append_registry(machine.path_value("output_root")/"registry"/f"{machine.data['machine_id']}.jsonl",
                        {"event":"FAILED_INPUT","run_slot":run_slot,"attempt_id":attempt_id,"error":repr(exc)})
        raise

def train_run(run_slot:str,machine_config:str|Path,attempt_id:str)->PreparedRun:
    machine=load_machine_config(machine_config); contract=_repo_contract(machine); row=_row(machine,run_slot); attempt=_find_attempt(machine,run_slot,attempt_id)
    set_status(attempt,"RUNNING",{"started":time.time()})
    monitor=ResourceMonitor(attempt/"02_logs/gpu_usage.csv",machine.data["gpu_id"],str(machine.data.get("nvidia_smi_path") or "nvidia-smi")); monitor.start()
    try:
        if bool(machine.data.get("dry_run",False)):
            for name in ["best.pt","last.pt"]: (attempt/"03_checkpoints"/name).write_bytes(f"DRYRUN {run_slot} {name}\n".encode())
            pd.DataFrame([{"epoch":1,"loss":.1,"dry_run":True}]).to_csv(attempt/"02_logs/epoch_training_metrics.csv",index=False)
        else:
            work=attempt/"work/trainer"; work.mkdir(parents=True,exist_ok=True)
            cmd=trainer_command(contract,machine,attempt/"01_manifests/train_manifest.csv",attempt/"01_manifests/normal_train_manifest.csv",work,int(row["training_seed"]))
            scratch=machine.path_value("local_scratch_root",required=False) or machine.path_value("cache_root")
            scratch.mkdir(parents=True,exist_ok=True)
            ensure_ultralytics_runtime(machine.path_value("cache_root"))
            env={"CUDA_VISIBLE_DEVICES":str(machine.data["gpu_id"]),"TMPDIR":str(scratch),"TEMP":str(scratch),"TMP":str(scratch),
                 "YOLO_CONFIG_DIR":str(machine.path_value("cache_root"))}
            run_logged(cmd,machine.path_value("repo_root"),attempt/"02_logs/train.log",env=env,
                       timeout=int(machine.data.get("command_timeout_seconds") or 0) or None)
            _link_or_copy(_unique_find(work,"best.pt"),attempt/"03_checkpoints/best.pt")
            _link_or_copy(_unique_find(work,"last.pt"),attempt/"03_checkpoints/last.pt")
            metrics=list(work.rglob("results.csv"))
            if len(metrics)==1: shutil.copy2(metrics[0],attempt/"02_logs/epoch_training_metrics.csv")
        set_status(attempt,"TRAIN_COMPLETED",{"best_sha256":sha256_file(attempt/"03_checkpoints/best.pt"),"last_sha256":sha256_file(attempt/"03_checkpoints/last.pt")})
    except Exception as exc:
        try: set_status(attempt,"FAILED_TRAIN",{"error":repr(exc)})
        finally: monitor.stop()
        raise
    monitor.stop(); return PreparedRun(run_slot,attempt_id,attempt,row)

def _dry_predictions(seed:int,n_normal:int=1000,n_defect:int=200)->tuple[pd.DataFrame,pd.DataFrame]:
    rng=np.random.default_rng(seed)
    def make(n0,n1):
        y=np.r_[np.zeros(n0,dtype=int),np.ones(n1,dtype=int)]
        p=np.r_[rng.beta(2,12,n0),rng.beta(12,2,n1)]
        return pd.DataFrame({"sample_id":[f"toy_{i:06d}" for i in range(len(y))],"y_true":y,"score":p})
    return make(n_normal,n_defect),make(100000,20000)

def evaluate_run(run_slot:str,machine_config:str|Path,attempt_id:str)->PreparedRun:
    machine=load_machine_config(machine_config); contract=_repo_contract(machine); row=_row(machine,run_slot); attempt=_find_attempt(machine,run_slot,attempt_id)
    try:
        raw=attempt/"work/evaluator"; raw.mkdir(parents=True,exist_ok=True)
        if bool(machine.data.get("dry_run",False)):
            cal,op=_dry_predictions(int(row["training_seed"])); cal.to_csv(raw/"val_cal_predictions.csv",index=False); op.to_csv(raw/"val_op_predictions.csv",index=False)
        else:
            ev=contract.data.get("evaluation_adapter", {})
            if ev.get("mode") != "native_ultralytics_val_only":
                raise ValidationError(f"Unsupported evaluation adapter: {ev}")
            predict_split(attempt/"03_checkpoints/best.pt",machine.path_value("dataset_root"),
                          machine.path_value("val_cal_defect_manifest"),machine.path_value("val_cal_normal_manifest"),
                          raw/"val_cal_predictions.csv",machine.data["gpu_id"],int(machine.data.get("prediction_batch_size") or 256),
                          int(machine.data.get("prediction_workers") or machine.data["num_workers"]),int(contract.data["training"]["image_size"]),
                          ev.get("accepted_defect_class_names"),machine.path_value("repo_root")/"YOLOv11")
            predict_split(attempt/"03_checkpoints/best.pt",machine.path_value("dataset_root"),
                          machine.path_value("val_op_defect_manifest"),machine.path_value("val_op_normal_manifest"),
                          raw/"val_op_predictions.csv",machine.data["gpu_id"],int(machine.data.get("prediction_batch_size") or 256),
                          int(machine.data.get("prediction_workers") or machine.data["num_workers"]),int(contract.data["training"]["image_size"]),
                          ev.get("accepted_defect_class_names"),machine.path_value("repo_root")/"YOLOv11")
        cal_path=_unique_find(raw,"val_cal_predictions.csv"); op_path=_unique_find(raw,"val_op_predictions.csv")
        metrics=finalize_evaluation(cal_path,op_path,attempt/"05_metrics",contract.data["calibration"]["deployment_prevalence"])
        shutil.copy2(attempt/"05_metrics/val_cal_predictions.csv",attempt/"04_predictions/val_cal_predictions.csv")
        shutil.copy2(attempt/"05_metrics/val_op_predictions.csv",attempt/"04_predictions/val_op_predictions.csv")
        set_status(attempt,"EVALUATED",{"TN_at_FN95":metrics["TN_at_FN95"],"FN_at_TN68253":metrics["FN_at_TN68253"]})
    except Exception as exc:
        try: set_status(attempt,"FAILED_EVAL",{"error":repr(exc)})
        finally: pass
        raise
    return PreparedRun(run_slot,attempt_id,attempt,row)

def validate_run(run_slot:str,machine_config:str|Path,attempt_id:str)->PreparedRun:
    machine=load_machine_config(machine_config); row=_row(machine,run_slot); attempt=_find_attempt(machine,run_slot,attempt_id)
    try:
        postflight(attempt,attempt/"07_validation/postflight_report.json")
        artifacts=[]
        for p in sorted(x for x in attempt.rglob('*') if x.is_file()): artifacts.append({"relative_path":p.relative_to(attempt).as_posix(),"size_bytes":p.stat().st_size,"sha256":sha256_file(p)})
        atomic_write_json(attempt/"07_validation/artifact_manifest.json",artifacts)
        set_status(attempt,"VALIDATED",{"validated":time.time()})
        final=attempt.with_name(attempt.name.removesuffix('.inprogress'))
        if final.exists(): raise FileExistsError(final)
        attempt.rename(final); attempt=final
        append_registry(machine.path_value("output_root")/"registry"/f"{machine.data['machine_id']}.jsonl",{"event":"VALIDATED","run_slot":run_slot,"attempt_id":attempt_id,"path":str(final)})
    except Exception as exc:
        try: set_status(attempt,"INVALID_ARTIFACT",{"error":repr(exc)})
        finally: pass
        raise
    return PreparedRun(run_slot,attempt_id,attempt,row)

def run_all(run_slot:str,machine_config:str|Path,attempt_id:str|None=None,allow_new_attempt_after_validated:bool=False)->PreparedRun:
    p=prepare_run(run_slot,machine_config,attempt_id,allow_new_attempt_after_validated)
    train_run(run_slot,machine_config,p.attempt_id); evaluate_run(run_slot,machine_config,p.attempt_id)
    return validate_run(run_slot,machine_config,p.attempt_id)

def run_entry_cli(run_slot:str,argv:list[str]|None=None)->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--machine-config',required=True); ap.add_argument('--action',choices=['prepare','train','evaluate','validate','run'],default='run')
    ap.add_argument('--attempt-id'); ap.add_argument('--allow-new-attempt-after-validated',action='store_true'); args=ap.parse_args(argv)
    if args.action=='prepare': r=prepare_run(run_slot,args.machine_config,args.attempt_id,args.allow_new_attempt_after_validated)
    elif args.action=='train': r=train_run(run_slot,args.machine_config,args.attempt_id)
    elif args.action=='evaluate': r=evaluate_run(run_slot,args.machine_config,args.attempt_id)
    elif args.action=='validate': r=validate_run(run_slot,args.machine_config,args.attempt_id)
    else: r=run_all(run_slot,args.machine_config,args.attempt_id,args.allow_new_attempt_after_validated)
    print(json.dumps({"run_slot":r.run_slot,"attempt_id":r.attempt_id,"attempt_dir":str(r.attempt_dir)},ensure_ascii=False)); return 0
