from __future__ import annotations
import importlib,json,math,subprocess,platform,sys
from pathlib import Path
import pandas as pd

from .contract import Contract,validate_contract_semantics
from .errors import ValidationError
from .machine import MachineConfig
from .metrics import operational_metrics
from .util import atomic_write_json,environment_snapshot,sha256_file



def _major_minor(value: str) -> tuple[int, int]:
    parts=[]
    for token in str(value).split('.'):
        num=''.join(ch for ch in token if ch.isdigit())
        if num: parts.append(int(num))
        if len(parts)==2: break
    return tuple(parts[:2]) if len(parts)>=2 else (-1,-1)

def _repository_audit(repo: Path, contract: Contract) -> dict:
    expected=contract.data["repository"]
    branch=subprocess.check_output(["git","branch","--show-current"],cwd=repo,text=True).strip()
    commit=subprocess.check_output(["git","rev-parse","--short=12","HEAD"],cwd=repo,text=True).strip()
    protected=expected.get("protected_paths",[])
    status=subprocess.check_output(["git","status","--porcelain","--",*protected],cwd=repo,text=True).strip() if protected else ''
    return {"branch":branch,"commit":commit,"protected_status":status,"branch_ok":branch==expected["branch"],
            "commit_ok":commit.startswith(str(expected["commit"])),"protected_clean":not bool(status)}

def _environment_audit(contract: Contract,machine:MachineConfig) -> dict:
    snap=environment_snapshot(); expected=contract.data["environment"]; checks={}
    yolo_root=machine.path_value("repo_root")/"YOLOv11"
    sys.path.insert(0,str(yolo_root))
    module=importlib.import_module("ultralytics")
    module_path=Path(module.__file__).resolve()
    try: module_path.relative_to(yolo_root.resolve())
    except ValueError as exc: raise ValidationError(f"Ultralytics resolved outside local YOLOv11: {module_path}") from exc
    snap["ultralytics"]=str(module.__version__)
    mapping={"python":"python","numpy":"numpy","pandas":"pandas","scikit_learn":"sklearn","pytorch":"torch","ultralytics":"ultralytics"}
    for ek,sk in mapping.items():
        actual=snap.get(sk)
        if isinstance(actual,dict): checks[ek]={"expected":expected.get(ek),"actual":actual,"ok":False}; continue
        checks[ek]={"expected":str(expected.get(ek)),"actual":str(actual),"ok":_major_minor(str(actual))==_major_minor(str(expected.get(ek)))}
    return {"snapshot":snap,"checks":checks,"all_ok":all(x["ok"] for x in checks.values())}

def preflight(contract:Contract,machine:MachineConfig,run_row:dict,selection_path:Path,run_manifest_summary:Path,output:Path)->dict:
    issues=validate_contract_semantics(contract)
    repo_audit=None; env_audit=None
    if not bool(machine.data.get("dry_run", False)):
        try:
            repo_audit=_repository_audit(machine.path_value("repo_root"),contract)
            if not repo_audit["branch_ok"]: issues.append(f"Wrong git branch: {repo_audit['branch']}")
            if not repo_audit["commit_ok"]: issues.append(f"Wrong git commit: {repo_audit['commit']}")
            if not repo_audit["protected_clean"]: issues.append("Protected repository paths are modified")
        except Exception as exc: issues.append(f"Repository audit failed: {exc}")
        try:
            env_audit=_environment_audit(contract,machine)
            if not env_audit["all_ok"]: issues.append("Environment major/minor versions do not match contract")
        except Exception as exc: issues.append(f"Environment audit failed: {exc}")
    required_paths=["repo_root","dataset_root","artifact_root","output_root","cache_root","base_checkpoint","train_manifest","normal_train_manifest",
                    "val_model_defect_manifest","val_model_normal_manifest"]
    if not bool(machine.data.get("dry_run", False)):
        required_paths += ["val_cal_defect_manifest","val_cal_normal_manifest","val_op_defect_manifest","val_op_normal_manifest"]
    for key in required_paths:
        try:
            p=machine.path_value(key)
            if key not in {"output_root","cache_root","artifact_root"} and not p.exists(): issues.append(f"Missing path {key}: {p}")
        except Exception as exc: issues.append(str(exc))
    if not selection_path.exists(): issues.append(f"Missing selection: {selection_path}")
    else:
        s=pd.read_csv(selection_path)
        if len(s)!=int(run_row["budget"]): issues.append(f"Selection budget mismatch: {len(s)} vs {run_row['budget']}")
        if s.sample_id.duplicated().any(): issues.append("Selection duplicate IDs")
    try:
        summary=json.loads(run_manifest_summary.read_text())
        expected=120000+int(run_row["budget"])
        if summary["epoch_samples"]!=expected: issues.append(f"Epoch sample count {summary['epoch_samples']} != {expected}")
        expected_steps=math.ceil(expected/int(contract.data["training"]["batch_size"]))
        contract_steps=int(contract.data["training"]["expected_steps"][f"B{int(run_row['budget'])}"])
        if expected_steps!=contract_steps: issues.append(f"Step count {expected_steps} != contract {contract_steps}")
    except Exception as exc: issues.append(f"Manifest summary invalid: {exc}")
    report={"status":"PASS" if not issues else "FAIL","issues":issues,"contract_sha256":contract.sha256,"run":run_row,
            "environment":env_audit["snapshot"] if env_audit else environment_snapshot(),"repository_audit":repo_audit,"environment_audit":env_audit,
            "selection_sha256":sha256_file(selection_path) if selection_path.exists() else None}
    atomic_write_json(output,report,overwrite=True)
    if issues: raise ValidationError(f"Preflight failed; see {output}")
    return report


def postflight(attempt_dir:Path,output:Path)->dict:
    issues=[]
    metrics_path=attempt_dir/"05_metrics/operational_metrics.json"; pred_path=attempt_dir/"04_predictions/val_op_predictions.csv"
    for p in [metrics_path,pred_path,attempt_dir/"03_checkpoints/best.pt",attempt_dir/"03_checkpoints/last.pt"]:
        if not p.exists(): issues.append(f"Missing artifact: {p.relative_to(attempt_dir)}")
    recomputed=None
    if not issues:
        saved=json.loads(metrics_path.read_text()); pred=pd.read_csv(pred_path)
        recomputed,_=operational_metrics(pred[["sample_id","y_true","score"]])
        for key,sub in [("TN_at_FN95","actual_TN"),("TN_at_FN95","actual_FN"),("FN_at_TN68253","actual_TN"),("FN_at_TN68253","actual_FN")]:
            if saved[key][sub]!=recomputed[key][sub]: issues.append(f"Metric mismatch {key}.{sub}")
        for key in ["gap_q68_q050","tail_gap_q90_q05"]:
            if abs(saved[key]-recomputed[key])>1e-12: issues.append(f"Metric mismatch {key}")
    report={"status":"PASS" if not issues else "FAIL","issues":issues,"recomputed":recomputed}
    atomic_write_json(output,report,overwrite=True)
    if issues: raise ValidationError(f"Postflight failed; see {output}")
    return report
