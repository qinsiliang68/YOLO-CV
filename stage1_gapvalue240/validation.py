from __future__ import annotations
import importlib,json,math,subprocess,platform,sys
from pathlib import Path
from typing import Callable
import pandas as pd
import numpy as np
import yaml

from .contract import Contract,validate_contract_semantics
from .errors import ValidationError
from .machine import MachineConfig
from .metrics import operational_metrics
from .util import atomic_write_bytes,atomic_write_json,environment_snapshot,sha256_file


PERMANENT_ARTIFACT_DIRS = (
    "00_identity",
    "01_manifests",
    "02_logs",
    "03_checkpoints",
    "04_predictions",
    "05_metrics",
    "06_figures",
    "07_validation",
)


def write_permanent_artifact_manifest(
    attempt_dir: str | Path,
    output: str | Path,
) -> list[dict]:
    """Hash durable run evidence without walking staging or work images."""

    attempt = Path(attempt_dir).resolve()
    destination = Path(output).resolve()
    rows = _permanent_artifact_rows(attempt, destination)
    frame = pd.DataFrame(rows, columns=["relative_path", "size_bytes", "sha256"])
    atomic_write_bytes(destination, frame.to_csv(index=False).encode("utf-8"), overwrite=True)
    return rows


def _permanent_artifact_rows(attempt: Path, destination: Path) -> list[dict]:
    rows: list[dict] = []
    for dirname in PERMANENT_ARTIFACT_DIRS:
        root = attempt / dirname
        if not root.is_dir():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.resolve() == destination or path.name.endswith(".tmp"):
                continue
            rows.append({
                "relative_path": path.relative_to(attempt).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return rows


def verify_permanent_artifact_manifest(
    attempt_dir: str | Path,
    manifest_path: str | Path,
) -> dict:
    attempt = Path(attempt_dir).resolve()
    manifest = Path(manifest_path).resolve()
    if not manifest.is_file():
        raise ValidationError(f"Missing artifact manifest: {manifest}")
    expected = _permanent_artifact_rows(attempt, manifest)
    recorded = pd.read_csv(manifest, dtype={"relative_path": "string", "sha256": "string"})
    actual = recorded.to_dict("records")
    normalized = [
        {"relative_path": str(row["relative_path"]), "size_bytes": int(row["size_bytes"]), "sha256": str(row["sha256"])}
        for row in actual
    ]
    if normalized != expected:
        raise ValidationError("Permanent artifact manifest differs from current durable files")
    return {"status": "PASS", "artifact_count": len(expected), "manifest": str(manifest)}



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


def _manifest_identity(path: str | Path, y_true: int) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"canonical_image_relpath": "string", "sample_id": "string"})
    identity = "canonical_image_relpath" if "canonical_image_relpath" in frame else "sample_id" if "sample_id" in frame else None
    if identity is None:
        raise ValidationError(f"Manifest lacks canonical identity: {path}")
    out = pd.DataFrame({"sample_id": frame[identity].astype(str), "y_true": int(y_true)})
    if out.sample_id.duplicated().any():
        raise ValidationError(f"Manifest contains duplicate identities: {path}")
    return out


def _prediction_expectation(defect_manifest: str | Path, normal_manifest: str | Path) -> pd.DataFrame:
    expected = pd.concat([_manifest_identity(defect_manifest, 1), _manifest_identity(normal_manifest, 0)], ignore_index=True)
    if expected.sample_id.duplicated().any():
        raise ValidationError("Defect and normal evaluation manifests overlap")
    return expected


def strict_postflight(
    attempt_dir: Path,
    output: Path,
    expected: dict,
    checkpoint_validator: Callable[[Path], None] | None = None,
) -> dict:
    """Validate the scientific execution contract before an attempt may become VALIDATED."""
    attempt_dir = Path(attempt_dir)
    issues: list[str] = []
    epochs = int(expected["epochs"])
    steps = int(expected["steps_per_epoch"])
    batch = int(expected["batch_size"])
    metrics_path = attempt_dir / "05_metrics/operational_metrics.json"
    cal_path = attempt_dir / "04_predictions/val_cal_predictions.csv"
    op_path = attempt_dir / "04_predictions/val_op_predictions.csv"
    results_path = attempt_dir / "02_logs/epoch_training_metrics.csv"
    audit_path = attempt_dir / "02_logs/training_execution_audit.json"
    args_path = attempt_dir / "02_logs/args.yaml"
    resolved_args_path = attempt_dir / "02_logs/resolved_training_args.json"
    checkpoints = [attempt_dir / "03_checkpoints/best.pt", attempt_dir / "03_checkpoints/last.pt"]
    required = [metrics_path, cal_path, op_path, results_path, audit_path, args_path, resolved_args_path, *checkpoints]
    for path in required:
        if not path.exists(): issues.append(f"Missing artifact: {path.relative_to(attempt_dir)}")

    if results_path.exists():
        try:
            results = pd.read_csv(results_path)
            if len(results) != epochs: issues.append(f"Training epoch rows {len(results)} != {epochs}")
            if "epoch" not in results: issues.append("Training results lack epoch column")
            elif len(results) and int(results.epoch.iloc[-1]) != epochs:
                issues.append(f"Last training epoch {results.epoch.iloc[-1]} != {epochs}")
            numeric = results.select_dtypes(include=["number"])
            if numeric.size and not np.isfinite(numeric.to_numpy(dtype=float)).all(): issues.append("Training results contain NaN/Inf")
        except Exception as exc: issues.append(f"Training results invalid: {exc}")

    audit = None
    if audit_path.exists():
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if int(audit.get("completed_epochs", -1)) != epochs: issues.append("Training audit completed_epochs mismatch")
            if int(audit.get("expected_steps_per_epoch", -1)) != steps: issues.append("Training audit expected steps mismatch")
            observed = [int(x) for x in audit.get("observed_steps_per_epoch", [])]
            if observed != [steps] * epochs: issues.append("Observed steps per epoch mismatch")
            if int(audit.get("optimizer_steps_total", -1)) != epochs * steps: issues.append("Optimizer step total mismatch")
            if int(audit.get("effective_batch_size", -1)) != batch: issues.append("Effective batch size mismatch")
            if audit.get("loss_finite") is not True: issues.append("Training audit reports non-finite loss")
        except Exception as exc: issues.append(f"Training execution audit invalid: {exc}")

    if args_path.exists():
        try:
            args = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
            checks = {
                "epochs": epochs, "batch": batch, "imgsz": int(expected["imgsz"]), "seed": int(expected["seed"]),
                "patience": 0, "deterministic": True, "cache": False,
            }
            for key, value in checks.items():
                if args.get(key) != value: issues.append(f"Resolved args mismatch {key}: {args.get(key)!r} != {value!r}")
            model_name = Path(str(args.get("model", ""))).name
            allowed_model_names = {str(expected["model_filename"])}
            if audit and int(audit.get("resume_count", 0)) > 0:
                allowed_model_names.add("last.pt")
            if model_name not in allowed_model_names: issues.append("Resolved model filename mismatch")
        except Exception as exc: issues.append(f"Resolved args invalid: {exc}")

    if resolved_args_path.exists() and args_path.exists():
        try:
            resolved_record = json.loads(resolved_args_path.read_text(encoding="utf-8"))
            if resolved_record.get("args_yaml_sha256") != sha256_file(args_path):
                issues.append("Resolved training args hash does not match args.yaml")
            if resolved_record.get("resolved_args") != (yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}):
                issues.append("Resolved training args payload differs from args.yaml")
            optimization = resolved_record.get("optimization", {})
            augmentation = resolved_record.get("augmentation", {})
            required_optimization = {"optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "warmup_momentum", "warmup_bias_lr"}
            required_augmentation = {"hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr", "bgr", "mosaic", "mixup", "cutmix", "copy_paste", "auto_augment", "erasing"}
            if set(optimization) != required_optimization: issues.append("Resolved optimizer/learning-rate fields are incomplete")
            if set(augmentation) != required_augmentation: issues.append("Resolved augmentation fields are incomplete")
        except Exception as exc: issues.append(f"Resolved training args record invalid: {exc}")

    for checkpoint in checkpoints:
        if not checkpoint.exists(): continue
        try:
            if checkpoint.stat().st_size <= 0: raise ValidationError("empty checkpoint")
            if checkpoint_validator is not None: checkpoint_validator(checkpoint)
        except Exception as exc: issues.append(f"Checkpoint cannot be loaded {checkpoint.name}: {exc}")

    saved = recomputed = None
    if metrics_path.exists() and op_path.exists():
        try:
            saved = json.loads(metrics_path.read_text(encoding="utf-8")); op = pd.read_csv(op_path)
            recomputed, _ = operational_metrics(op[["sample_id", "y_true", "score"]])
            for metric, field in (("TN_at_FN95", "actual_TN"), ("TN_at_FN95", "actual_FN"),
                                  ("FN_at_TN68253", "actual_TN"), ("FN_at_TN68253", "actual_FN")):
                if saved[metric][field] != recomputed[metric][field]: issues.append(f"Metric mismatch {metric}.{field}")
            for key in ("gap_q68_q050", "tail_gap_q90_q05"):
                if abs(float(saved[key]) - float(recomputed[key])) > 1e-12: issues.append(f"Metric mismatch {key}")
        except Exception as exc: issues.append(f"Operational metric recomputation failed: {exc}")

    for split, prediction_path in (("val_cal", cal_path), ("val_op", op_path)):
        if not prediction_path.exists(): continue
        try:
            predicted = pd.read_csv(prediction_path, dtype={"sample_id": "string"})
            expected_rows = _prediction_expectation(expected[f"{split}_defect_manifest"], expected[f"{split}_normal_manifest"])
            actual_pairs = set(zip(predicted.sample_id.astype(str), predicted.y_true.astype(int)))
            expected_pairs = set(zip(expected_rows.sample_id.astype(str), expected_rows.y_true.astype(int)))
            if len(predicted) != len(expected_rows) or actual_pairs != expected_pairs:
                issues.append(f"{split} prediction identities/labels do not match frozen manifests")
            if predicted.sample_id.duplicated().any(): issues.append(f"{split} prediction IDs are duplicated")
            if not np.isfinite(predicted.score.astype(float)).all(): issues.append(f"{split} predictions contain NaN/Inf")
        except Exception as exc: issues.append(f"{split} prediction validation failed: {exc}")

    report = {"status": "PASS" if not issues else "FAIL", "issues": issues, "expected": expected,
              "training_audit": audit, "recomputed": recomputed}
    atomic_write_json(output, report, overwrite=True)
    if issues: raise ValidationError(f"Strict postflight failed; see {output}")
    return report
