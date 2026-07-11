from __future__ import annotations
import re,subprocess
from pathlib import Path
from typing import Any

from .contract import Contract
from .errors import ValidationError
from .machine import MachineConfig

TRAIN_FLAG_ALIASES={
 "manifest_dir":["--manifest-dir"],
 "runs_root":["--runs-root"],
 "work_root":["--work-root"],
 "dataset_root":["--dataset-root"],
 "yolo_root":["--yolo-root"],
 "seed":["--seed"],
}
OPTIONAL_TRAIN_FLAG_ALIASES={"workers":["--workers","--num-workers","--num_workers"],"device":["--device"]}
EVAL_FLAG_ALIASES={
 "checkpoint":["--checkpoint","--weights","--model"],
 "val_cal_defect":["--val-cal-defect-manifest","--val_cal_defect_manifest"],
 "val_cal_normal":["--val-cal-normal-manifest","--val_cal_normal_manifest"],
 "val_op_defect":["--val-op-defect-manifest","--val_op_defect_manifest"],
 "val_op_normal":["--val-op-normal-manifest","--val_op_normal_manifest"],
 "output_root":["--output-root","--output_root"],
}

def _help_text(python:str,script:Path,repo:Path)->str:
    r=subprocess.run([python,str(script),"--help"],cwd=repo,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
    return r.stdout

def _resolve_flags(help_text:str,aliases:dict[str,list[str]])->dict[str,str]:
    result={}
    for logical,opts in aliases.items():
        found=[o for o in opts if re.search(rf"(?<![\w-]){re.escape(o)}(?![\w-])",help_text)]
        if len(found)!=1: raise ValidationError(f"Cannot resolve unique CLI flag for {logical}; found={found}")
        result[logical]=found[0]
    return result

def _resolve_optional_flags(help_text:str,aliases:dict[str,list[str]])->dict[str,str]:
    result={}
    for logical,opts in aliases.items():
        found=[o for o in opts if re.search(rf"(?<![\w-]){re.escape(o)}(?![\w-])",help_text)]
        if len(found)>1: raise ValidationError(f"Ambiguous optional CLI flag for {logical}: {found}")
        if found: result[logical]=found[0]
    return result

def trainer_command(contract:Contract,machine:MachineConfig,train_manifest:Path,normal_manifest:Path,output_root:Path,seed:int)->list[str]:
    repo=machine.path_value("repo_root"); script=repo/contract.data["repository"]["existing_trainer"]
    if not script.exists(): raise FileNotFoundError(script)
    train_manifest=Path(train_manifest).resolve(); normal_manifest=Path(normal_manifest).resolve(); output_root=Path(output_root).resolve()
    if train_manifest.parent != normal_manifest.parent:
        raise ValidationError("Current Stage-1 trainer requires train and normal manifests in one manifest directory")
    if train_manifest.name != "train_manifest.csv" or normal_manifest.name != "normal_train_manifest.csv":
        raise ValidationError("Replay manifests must use the current trainer's canonical filenames")
    manifest_dir=train_manifest.parent
    for name in ("val_model_manifest.csv","normal_val_model_manifest.csv"):
        if not (manifest_dir/name).exists(): raise FileNotFoundError(manifest_dir/name)
    model_code=str(contract.data["training"]["model_code"])
    expected_name=f"yolo11{model_code}-cls.pt"
    base_checkpoint=machine.path_value("base_checkpoint")
    allowed_checkpoints={
        (repo/expected_name).resolve(),
        (repo/"YOLOv11"/expected_name).resolve(),
        (repo/"YOLOv11"/"weights"/expected_name).resolve(),
    }
    if base_checkpoint not in allowed_checkpoints:
        raise ValidationError(f"Current trainer cannot consume an arbitrary checkpoint path: {base_checkpoint}")
    python=str(machine.data.get("python_executable") or "python")
    help_text=_help_text(python,script,repo)
    flags=_resolve_flags(help_text,TRAIN_FLAG_ALIASES); optional=_resolve_optional_flags(help_text,OPTIONAL_TRAIN_FLAG_ALIASES)
    cmd=[python,str(script),*contract.data["training"]["trainer_cli_fixed_args"],flags["seed"],str(seed),
         flags["manifest_dir"],str(manifest_dir),flags["runs_root"],str(output_root/"runs"),
         flags["work_root"],str(output_root/"dataset"),flags["dataset_root"],str(machine.path_value("dataset_root")),
         flags["yolo_root"],str(repo/"YOLOv11")]
    if "workers" in optional: cmd += [optional["workers"],str(machine.data["num_workers"])]
    if "device" in optional: cmd += [optional["device"],str(machine.data["gpu_id"])]
    return cmd

def evaluator_command(contract:Contract,machine:MachineConfig,checkpoint:Path,output_root:Path)->list[str]:
    repo=machine.path_value("repo_root"); script=repo/contract.data["repository"]["existing_evaluator"]
    if not script.exists(): raise FileNotFoundError(script)
    python=str(machine.data.get("python_executable") or "python")
    flags=_resolve_flags(_help_text(python,script,repo),EVAL_FLAG_ALIASES)
    paths={k:machine.path_value(k) for k in ["val_cal_defect_manifest","val_cal_normal_manifest","val_op_defect_manifest","val_op_normal_manifest"]}
    cmd=[python,str(script),flags["checkpoint"],str(checkpoint),flags["val_cal_defect"],str(paths["val_cal_defect_manifest"]),
         flags["val_cal_normal"],str(paths["val_cal_normal_manifest"]),flags["val_op_defect"],str(paths["val_op_defect_manifest"]),
         flags["val_op_normal"],str(paths["val_op_normal_manifest"]),flags["output_root"],str(output_root)]
    return cmd
