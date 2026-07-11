from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .aiops import EXIT_RETRYABLE, EXIT_SUCCESS, EXIT_TERMINAL, aiops_status_payload, exit_code_for_exception
from .checkpoint_probe import run_checkpoint_probe_worker
from .contract import Contract, load_contract
from .errors import ExternalCommandError, GapValueError, ValidationError
from .evaluation import finalize_evaluation
from .execution_recovery import RunDecision, discover_run_action
from .locks import RunLock
from .machine import MachineConfig, load_machine_config
from .machine_assets import validate_machine_asset_report
from .manifests import build_replay_manifests
from .matrix import load_matrix
from .monitor import ResourceMonitor
from .prediction_controller import PredictionWorkerSpec, run_prediction_workers
from .registry import append_registry
from .runtime_contract import (
    RuntimeContract,
    load_runtime_contract,
    validate_runtime_links,
    validation_status_for_mode,
    verify_release_identity,
    verify_selection_against_index,
)
from .status import read_status, set_status
from .subprocesses import run_logged
from .util import atomic_write_bytes, atomic_write_json, sha256_file, stable_hash
from .validation import strict_postflight, verify_permanent_artifact_manifest, write_permanent_artifact_manifest


RUNTIME_CONTRACT_RELATIVE = Path("configs/stage1_gapvalue240/RUNTIME_CONTRACT_v1_2.yaml")
SCIENCE_CONTRACT_RELATIVE = Path("configs/stage1_gapvalue240/EXPERIMENT_CONTRACT.yaml")
PERMANENT_INPUT_NAMES = {
    "train_manifest": "base_train_defect_manifest.csv",
    "normal_train_manifest": "base_train_normal_manifest.csv",
    "val_model_defect_manifest": "base_val_model_defect_manifest.csv",
    "val_model_normal_manifest": "base_val_model_normal_manifest.csv",
    "val_cal_defect_manifest": "val_cal_defect_manifest.csv",
    "val_cal_normal_manifest": "val_cal_normal_manifest.csv",
    "val_op_defect_manifest": "val_op_defect_manifest.csv",
    "val_op_normal_manifest": "val_op_normal_manifest.csv",
}


@dataclass(frozen=True)
class PreparedRun:
    run_slot: str
    attempt_id: str
    attempt_dir: Path
    run_row: dict


@dataclass(frozen=True)
class RunContext:
    machine: MachineConfig
    science: Contract
    runtime: RuntimeContract | None
    run_row: dict
    selection_path: Path
    identity: dict[str, Any]
    input_report: dict[str, Any]


def _repo_root(machine: MachineConfig) -> Path:
    return machine.path_value("repo_root")


def _science_contract(machine: MachineConfig) -> Contract:
    return load_contract(_repo_root(machine) / SCIENCE_CONTRACT_RELATIVE)


def _runtime_contract(machine: MachineConfig) -> RuntimeContract | None:
    path = _repo_root(machine) / RUNTIME_CONTRACT_RELATIVE
    if path.is_file():
        return load_runtime_contract(path)
    if bool(machine.data.get("dry_run", False)):
        return None
    raise FileNotFoundError(f"Formal execution requires runtime contract v1.2: {path}")


def _artifact_root(machine: MachineConfig) -> Path:
    return machine.path_value("artifact_root")


def _matrix_path(machine: MachineConfig, runtime_links: dict | None) -> Path:
    if runtime_links is not None:
        return Path(runtime_links["queue"]["frozen_matrix"]["path"])
    return _artifact_root(machine) / "generated/frozen_experiment_matrix.csv"


def _row_from_matrix(matrix_path: Path, run_slot: str) -> dict:
    frame = load_matrix(matrix_path)
    match = frame.loc[frame.run_slot.astype(str) == str(run_slot)]
    if len(match) != 1:
        raise ValidationError(f"Run slot not found uniquely: {run_slot}")
    return match.iloc[0].to_dict()


def _manifest_source(machine: MachineConfig, key: str) -> Path:
    value = machine.path_value(key, required=False)
    if value is not None:
        return value
    if bool(machine.data.get("dry_run", False)):
        fallback = "val_model_defect_manifest" if "defect" in key else "val_model_normal_manifest"
        return machine.path_value(fallback)
    raise ValidationError(f"Formal machine config is missing manifest: {key}")


def _validate_live_assets(machine: MachineConfig, report_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    current: dict[str, Any] = {}
    runtime = load_runtime_contract(_repo_root(machine) / RUNTIME_CONTRACT_RELATIVE)
    cached = validate_machine_asset_report(
        runtime,
        report_path,
        expected_machine_id=str(machine.data["machine_id"]),
        minimum_image_verification="existence",
    )
    for role, spec in runtime.data["machine_assets"]["manifest_roles"].items():
        source = machine.path_value(str(spec["machine_config_key"]))
        recorded = report["manifests"][role]
        if source != Path(str(recorded["path"])).resolve():
            raise ValidationError(f"Machine manifest path changed since asset snapshot: {role}")
        actual_sha = sha256_file(source)
        if actual_sha != str(recorded["sha256"]).upper():
            raise ValidationError(f"Machine manifest SHA-256 changed since asset snapshot: {role}")
        current[role] = {"path": str(source), "sha256": actual_sha, "rows": int(recorded["rows"])}
    checkpoint = machine.path_value("base_checkpoint")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != str(runtime.data["checkpoint"]["sha256"]).upper():
        raise ValidationError("Machine checkpoint SHA-256 differs from runtime contract")
    return {**cached, "current_manifests": current, "checkpoint_sha256": checkpoint_sha}


def _load_context(machine_config: str | Path, run_slot: str) -> RunContext:
    machine = load_machine_config(machine_config)
    science = _science_contract(machine)
    runtime = _runtime_contract(machine)
    dry_run = bool(machine.data.get("dry_run", False))
    links: dict | None = None
    release: dict[str, Any]
    selection_check: dict[str, Any] | None = None
    asset_report: dict[str, Any] | None = None

    if runtime is not None:
        links = validate_runtime_links(runtime, _repo_root(machine))
        expected_artifact = (_repo_root(machine) / runtime.data["queue"]["artifact_root"]).resolve()
        if _artifact_root(machine) != expected_artifact:
            raise ValidationError(
                f"machine artifact_root differs from runtime contract: {_artifact_root(machine)} != {expected_artifact}"
            )
        if dry_run:
            head = __import__("subprocess").check_output(
                ["git", "rev-parse", "HEAD"], cwd=_repo_root(machine), text=True
            ).strip()
            release = verify_release_identity(
                runtime,
                _repo_root(machine),
                test_release_ref_override=head,
                allow_test_override=True,
            )
        else:
            release = verify_release_identity(runtime, _repo_root(machine))
        selection_check = verify_selection_against_index(runtime, _repo_root(machine), run_slot)
        selection_path = Path(selection_check["selection_path"])
        if dry_run and not machine.path_value("machine_asset_report", required=False):
            input_snapshot_id = "DRY_RUN_NO_MACHINE_ASSET_SNAPSHOT"
        else:
            report_path = machine.path_value("machine_asset_report")
            asset_report = _validate_live_assets(machine, report_path)
            input_snapshot_id = str(asset_report["content_snapshot_id"])
        matrix_path = _matrix_path(machine, links)
        matrix_sha = links["queue"]["frozen_matrix"]["sha256"]
        selection_index_sha = links["queue"]["selection_index"]["sha256"]
        release_ref = str(runtime.data["release"]["git_tag"])
        runtime_sha = runtime.sha256
        science_file_sha = links["science_contract"]["file_sha256"]
        checkpoint_sha = links["checkpoint"]["sha256"]
    else:
        # Legacy dry-run support is intentionally non-aggregatable and cannot be used formally.
        release = {"status": "DRY_RUN_LEGACY", "override_used": True}
        matrix_path = _matrix_path(machine, None)
        selection_path = _artifact_root(machine) / "generated/selections" / run_slot / "selection_manifest.csv"
        if not selection_path.is_file():
            raise FileNotFoundError(selection_path)
        matrix_sha = sha256_file(matrix_path)
        selection_index_sha = "DRY_RUN_LEGACY"
        release_ref = "DRY_RUN_UNRELEASED"
        runtime_sha = "DRY_RUN_LEGACY"
        science_file_sha = sha256_file(_repo_root(machine) / SCIENCE_CONTRACT_RELATIVE)
        checkpoint_sha = sha256_file(machine.path_value("base_checkpoint"))
        manifest_hashes = {
            key: sha256_file(_manifest_source(machine, key)) for key in PERMANENT_INPUT_NAMES
        }
        input_snapshot_id = stable_hash(manifest_hashes)

    row = _row_from_matrix(matrix_path, run_slot)
    selection_sha = sha256_file(selection_path)
    if selection_check is not None and selection_sha != selection_check["selection_sha256"]:
        raise ValidationError(f"Selection changed after runtime verification: {run_slot}")
    identity = {
        "run_slot": str(run_slot),
        "dry_run": dry_run,
        "release_ref": release_ref,
        "release_commit": release.get("expected_commit"),
        "runtime_contract_id": runtime.runtime_contract_id if runtime else "DRY_RUN_LEGACY",
        "runtime_contract_sha256": runtime_sha,
        "science_contract_file_sha256": science_file_sha,
        "science_contract_sha256": science.sha256,
        "matrix_sha256": matrix_sha,
        "selection_index_sha256": selection_index_sha,
        "selection_sha256": selection_sha,
        "checkpoint_sha256": checkpoint_sha,
        "input_snapshot_id": input_snapshot_id,
        "resume_mode": "native_approximate",
        "scientific_config_hash": stable_hash({
            "run_row": row,
            "training": science.data["training"],
            "replay": science.data["replay"],
            "calibration": science.data["calibration"],
            "evaluation_adapter": science.data["evaluation_adapter"],
        }),
    }
    input_report = {
        "status": "PASS",
        "release": release,
        "runtime_links": links,
        "selection": selection_check or {"selection_path": str(selection_path), "selection_sha256": selection_sha},
        "machine_assets": asset_report,
    }
    return RunContext(machine, science, runtime, row, selection_path, identity, input_report)


def _attempt_parent(machine: MachineConfig, run_slot: str) -> Path:
    return machine.path_value("output_root") / "runs" / run_slot


def _new_attempt_id(row: dict) -> str:
    return time.strftime("%Y%m%dT%H%M%S") + "_" + str(row["arm"]) + "_" + uuid.uuid4().hex[:10]


def _find_attempt(machine: MachineConfig, run_slot: str, attempt_id: str) -> Path:
    parent = _attempt_parent(machine, run_slot)
    matches = [path for path in (parent / f"attempt_{attempt_id}.inprogress", parent / f"attempt_{attempt_id}") if path.exists()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Attempt not found uniquely: {run_slot}/{attempt_id}")
    return matches[0]


def _attempt_id(attempt: Path) -> str:
    name = attempt.name.removesuffix(".inprogress")
    return name.removeprefix("attempt_")


def _status_payload(
    prepared: PreparedRun,
    phase: str,
    *,
    retryable: bool = False,
    error_code: str | None = None,
    last_epoch: int | None = None,
    resume_count: int = 0,
) -> dict:
    return aiops_status_payload(
        run_slot=prepared.run_slot,
        phase=phase,
        last_epoch=last_epoch,
        resume_count=resume_count,
        retryable=retryable,
        error_code=error_code,
        attempt_id=prepared.attempt_id,
    )


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temp)
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def _copy_frozen_inputs(context: RunContext, attempt: Path) -> dict[str, Path]:
    frozen = attempt / "01_manifests/frozen_inputs"
    frozen.mkdir(parents=True, exist_ok=True)
    copies: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    for key, filename in PERMANENT_INPUT_NAMES.items():
        source = _manifest_source(context.machine, key)
        destination = frozen / filename
        _atomic_copy(source, destination)
        source_sha = sha256_file(source)
        copied_sha = sha256_file(destination)
        if source_sha != copied_sha:
            raise ValidationError(f"Frozen manifest copy SHA mismatch: {key}")
        copies[key] = destination
        rows.append({
            "role": key,
            "source_path": str(source),
            "frozen_path": destination.relative_to(attempt).as_posix(),
            "sha256": copied_sha,
            "rows": len(pd.read_csv(destination, usecols=[0])),
        })
    atomic_write_bytes(
        attempt / "00_identity/input_checksums.csv",
        pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
    )
    return copies


def _prepare_run_unlocked(
    context: RunContext,
    attempt_id: str | None = None,
    supersedes_attempt: Path | None = None,
) -> PreparedRun:
    run_slot = str(context.run_row["run_slot"])
    parent = _attempt_parent(context.machine, run_slot)
    parent.mkdir(parents=True, exist_ok=True)
    attempt_id = attempt_id or _new_attempt_id(context.run_row)
    attempt = parent / f"attempt_{attempt_id}.inprogress"
    if attempt.exists() or (parent / f"attempt_{attempt_id}").exists():
        raise FileExistsError(f"Attempt exists: {attempt_id}")
    for dirname in (
        "00_identity", "01_manifests", "02_logs", "03_checkpoints", "04_predictions",
        "05_metrics", "06_figures", "07_validation", "08_status", "work",
    ):
        (attempt / dirname).mkdir(parents=True, exist_ok=True)
    prepared = PreparedRun(run_slot, attempt_id, attempt, context.run_row)
    set_status(attempt, "PLANNED", _status_payload(prepared, "prepare"))
    try:
        selection_copy = attempt / "01_manifests/selection_manifest.csv"
        _atomic_copy(context.selection_path, selection_copy)
        if sha256_file(selection_copy) != context.identity["selection_sha256"]:
            raise ValidationError("Attempt selection copy differs from frozen selection index")
        copies = _copy_frozen_inputs(context, attempt)
        replay = build_replay_manifests(
            copies["train_manifest"],
            copies["normal_train_manifest"],
            selection_copy,
            attempt / "01_manifests",
            expected_base_total=int(context.science.data["replay"]["base_samples"]),
        )
        _atomic_copy(copies["val_model_defect_manifest"], attempt / "01_manifests/val_model_manifest.csv")
        _atomic_copy(copies["val_model_normal_manifest"], attempt / "01_manifests/normal_val_model_manifest.csv")
        identity = {
            **context.identity,
            "attempt_id": attempt_id,
            "machine_id": str(context.machine.data["machine_id"]),
            "run_row": context.run_row,
            "resume_count": 0,
            "created_at_unix": time.time(),
            "supersedes_attempt_id": _attempt_id(supersedes_attempt) if supersedes_attempt is not None else None,
        }
        atomic_write_json(attempt / "00_identity/run_identity.json", identity)
        atomic_write_json(attempt / "00_identity/environment_controller.json", {
            "python": platform.python_version(), "platform": platform.platform(), "pid": os.getpid()
        })
        expected_epoch_samples = int(context.science.data["replay"]["base_samples"]) + int(context.run_row["budget"])
        replay_summary = json.loads(replay.summary_path.read_text(encoding="utf-8"))
        issues = []
        if int(replay_summary["epoch_samples"]) != expected_epoch_samples:
            issues.append("Replay manifest epoch sample count mismatch")
        if int(context.run_row["budget"]) != len(pd.read_csv(selection_copy)):
            issues.append("Selection budget mismatch")
        report = {
            "status": "PASS" if not issues else "FAIL",
            "issues": issues,
            "runtime_inputs": context.input_report,
            "run_identity": identity,
            "replay_manifest_summary": replay_summary,
        }
        atomic_write_json(attempt / "07_validation/preflight_report.json", report)
        if issues:
            raise ValidationError(f"Preflight failed: {issues}")
        set_status(attempt, "STAGED", _status_payload(prepared, "prepare"))
        append_registry(
            context.machine.path_value("output_root") / "registry" / f"{context.machine.data['machine_id']}.jsonl",
            {"event": "STAGED", "run_slot": run_slot, "attempt_id": attempt_id},
        )
        return prepared
    except Exception as exc:
        set_status(attempt, "FAILED_INPUT", {
            **_status_payload(prepared, "prepare", retryable=False, error_code="INPUT_VALIDATION_FAILED"),
            "error": repr(exc),
        })
        raise


def prepare_run(
    run_slot: str,
    machine_config: str | Path,
    attempt_id: str | None = None,
    allow_new_attempt_after_validated: bool = False,
) -> PreparedRun:
    context = _load_context(machine_config, run_slot)
    parent = _attempt_parent(context.machine, run_slot)
    with RunLock(parent / ".run.lock", {"run_slot": run_slot}, reclaim_dead_local=True):
        decision = _discover(context)
        if decision.action in {"SKIP_VALIDATED", "SKIP_DRY_RUN"}:
            if not allow_new_attempt_after_validated:
                assert decision.attempt_dir is not None
                return _prepared(context, decision.attempt_dir)
            return _prepare_run_unlocked(context, attempt_id, supersedes_attempt=decision.attempt_dir)
        if decision.action != "NEW_ATTEMPT":
            assert decision.attempt_dir is not None
            return _prepared(context, decision.attempt_dir)
        if decision.superseded_attempt is not None and read_status(decision.superseded_attempt)["state"] == "PLANNED":
            old = PreparedRun(run_slot, _attempt_id(decision.superseded_attempt), decision.superseded_attempt, context.run_row)
            set_status(decision.superseded_attempt, "FAILED_INPUT", {
                **_status_payload(old, "prepare", retryable=False, error_code="PREPARE_INTERRUPTED"),
                "error": decision.reason,
            })
        return _prepare_run_unlocked(context, attempt_id)


def _formal_train_command(context: RunContext, prepared: PreparedRun, resume_checkpoint: Path | None) -> list[str]:
    machine = context.machine
    attempt = prepared.attempt_dir
    python = str(machine.data.get("python_executable") or "python")
    command = [
        python,
        str(_repo_root(machine) / "scripts/stage1_gapvalue240/formal_train_worker.py"),
        "--contract", str(_repo_root(machine) / SCIENCE_CONTRACT_RELATIVE),
        "--dataset-root", str(machine.path_value("dataset_root")),
        "--staging-root", str(machine.path_value("staging_root")),
        "--base-train-defect-manifest", str(attempt / "01_manifests/frozen_inputs/base_train_defect_manifest.csv"),
        "--base-train-normal-manifest", str(attempt / "01_manifests/frozen_inputs/base_train_normal_manifest.csv"),
        "--base-val-defect-manifest", str(attempt / "01_manifests/frozen_inputs/base_val_model_defect_manifest.csv"),
        "--base-val-normal-manifest", str(attempt / "01_manifests/frozen_inputs/base_val_model_normal_manifest.csv"),
        "--run-train-defect-manifest", str(attempt / "01_manifests/train_manifest.csv"),
        "--run-train-normal-manifest", str(attempt / "01_manifests/normal_train_manifest.csv"),
        "--checkpoint", str(machine.path_value("base_checkpoint")),
        "--output-dir", str(attempt),
        "--yolo-root", str(_repo_root(machine) / "YOLOv11"),
        "--run-slot", prepared.run_slot,
        "--training-seed", str(int(prepared.run_row["training_seed"])),
        "--budget", str(int(prepared.run_row["budget"])),
        "--device", str(machine.data["gpu_id"]),
        "--workers", str(int(machine.data["num_workers"])),
        "--minimum-staging-free-gib", str(float(machine.data.get("minimum_staging_free_gib", 2))),
        "--minimum-output-free-gib", str(float(machine.data.get("minimum_output_free_gib", 20))),
        "--maximum-staging-files", str(int(machine.data.get("maximum_staging_files", 151000))),
        "--segment-id", f"{prepared.attempt_id}_{uuid.uuid4().hex[:8]}",
    ]
    if resume_checkpoint is not None:
        command += ["--resume-checkpoint", str(resume_checkpoint)]
    return command


def _dry_train_outputs(context: RunContext, prepared: PreparedRun) -> None:
    attempt = prepared.attempt_dir
    epochs = int(context.science.data["training"]["epochs"])
    steps = int(context.science.data["training"]["expected_steps"][f"B{int(prepared.run_row['budget'])}"])
    trainer = attempt / "trainer"
    (trainer / "weights").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"epoch": range(1, epochs + 1), "train/loss": np.linspace(0.3, 0.1, epochs)}).to_csv(
        trainer / "results.csv", index=False
    )
    args = {
        "model": str(context.machine.path_value("base_checkpoint")), "epochs": epochs, "batch": 128,
        "imgsz": 224, "patience": 0, "seed": int(prepared.run_row["training_seed"]),
        "deterministic": True, "cache": False, "optimizer": "auto", "lr0": 0.01,
        "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0005,
        "warmup_epochs": 3.0, "warmup_momentum": 0.8, "warmup_bias_lr": 0.1,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 0.0,
        "translate": 0.1, "scale": 0.5, "shear": 0.0, "perspective": 0.0,
        "flipud": 0.0, "fliplr": 0.5, "bgr": 0.0, "mosaic": 1.0, "mixup": 0.0,
        "cutmix": 0.0, "copy_paste": 0.0, "auto_augment": "randaugment", "erasing": 0.4,
    }
    (trainer / "args.yaml").write_text(yaml.safe_dump(args, sort_keys=False), encoding="utf-8")
    for name in ("best.pt", "last.pt"):
        (trainer / "weights" / name).write_bytes(f"DRY RUN {prepared.run_slot} {name}".encode("utf-8"))
    (attempt / "training_state").mkdir(exist_ok=True)
    _atomic_copy(trainer / "weights/last.pt", attempt / "training_state/last.pt")
    audit = {
        "schema_version": "stage1_gapvalue240.training_execution_audit.v1",
        "expected_epochs": epochs, "completed_epochs": epochs, "expected_steps_per_epoch": steps,
        "observed_steps_per_epoch": [steps] * epochs, "optimizer_steps_total": steps * epochs,
        "effective_batch_size": 128, "configured_args": {
            "epochs": epochs, "batch": 128, "imgsz": 224, "patience": 0,
            "seed": int(prepared.run_row["training_seed"]), "deterministic": True,
            "cache": False, "model": str(context.machine.path_value("base_checkpoint")),
        },
        "loss_finite": True, "resume_mode": "native_approximate", "resume_count": 0,
        "resume_segments": [{"segment_id": "dry", "start_epoch": 1, "end_epoch": epochs, "status": "COMPLETED"}],
    }
    atomic_write_json(attempt / "training_execution_audit.json", audit)
    atomic_write_json(attempt / "resolved_training_args.json", {
        "schema_version": "stage1_gapvalue240.resolved_training_args.v1",
        "args_yaml_sha256": sha256_file(trainer / "args.yaml"),
        "optimization": {key: args[key] for key in (
            "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
            "warmup_momentum", "warmup_bias_lr",
        )},
        "augmentation": {key: args[key] for key in (
            "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
            "perspective", "flipud", "fliplr", "bgr", "mosaic", "mixup", "cutmix",
            "copy_paste", "auto_augment", "erasing",
        )},
        "resolved_args": args,
    })
    atomic_write_json(attempt / "storage_preflight.json", {"status": "DRY_RUN", "dry_run": True})
    atomic_write_json(attempt / "formal_environment.json", {"status": "DRY_RUN", "dry_run": True})
    atomic_write_json(attempt / "checkpoint_preflight.json", {"status": "DRY_RUN", "dry_run": True})


def _sync_training_outputs(prepared: PreparedRun) -> dict:
    attempt = prepared.attempt_dir
    sources = {
        attempt / "trainer/results.csv": attempt / "02_logs/epoch_training_metrics.csv",
        attempt / "trainer/args.yaml": attempt / "02_logs/args.yaml",
        attempt / "training_execution_audit.json": attempt / "02_logs/training_execution_audit.json",
        attempt / "resolved_training_args.json": attempt / "02_logs/resolved_training_args.json",
        attempt / "trainer/weights/best.pt": attempt / "03_checkpoints/best.pt",
        attempt / "training_state/last.pt": attempt / "03_checkpoints/last.pt",
        attempt / "storage_preflight.json": attempt / "07_validation/storage_preflight.json",
        attempt / "formal_environment.json": attempt / "00_identity/environment_training.json",
        attempt / "checkpoint_preflight.json": attempt / "07_validation/checkpoint_preflight.json",
    }
    for source, destination in sources.items():
        if not source.is_file():
            raise ValidationError(f"Formal trainer output missing: {source}")
        _atomic_copy(source, destination)
    audit = json.loads((attempt / "02_logs/training_execution_audit.json").read_text(encoding="utf-8"))
    identity_path = attempt / "00_identity/run_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity.update({
        "resume_mode": str(audit["resume_mode"]),
        "resume_count": int(audit["resume_count"]),
        "resume_segments": audit["resume_segments"],
        "last_epoch": int(audit["completed_epochs"]),
    })
    atomic_write_json(identity_path, identity, overwrite=True)
    return audit


def _prepared(context: RunContext, attempt: Path) -> PreparedRun:
    _assert_attempt_identity(context, attempt)
    return PreparedRun(str(context.run_row["run_slot"]), _attempt_id(attempt), attempt, context.run_row)


def _assert_attempt_identity(context: RunContext, attempt: Path) -> None:
    identity_path = attempt / "00_identity/run_identity.json"
    if not identity_path.is_file():
        raise ValidationError(f"Attempt lacks frozen run identity: {attempt}")
    actual = json.loads(identity_path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"expected": expected, "actual": actual.get(key)}
        for key, expected in context.identity.items()
        if actual.get(key) != expected
    }
    if mismatches:
        raise ValidationError(f"Attempt runtime identity mismatch: {mismatches}")
    selection = attempt / "01_manifests/selection_manifest.csv"
    if not selection.is_file() or sha256_file(selection) != context.identity["selection_sha256"]:
        raise ValidationError("Attempt selection manifest differs from frozen runtime identity")
    checksums = attempt / "00_identity/input_checksums.csv"
    if not checksums.is_file():
        raise ValidationError("Attempt input checksum index is missing")
    for row in pd.read_csv(checksums).to_dict("records"):
        relative = Path(str(row["frozen_path"]).replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValidationError(f"Unsafe frozen input path in checksum index: {relative}")
        frozen = (attempt / relative).resolve()
        try:
            frozen.relative_to(attempt.resolve())
        except ValueError as exc:
            raise ValidationError(f"Frozen input escapes attempt directory: {frozen}") from exc
        if not frozen.is_file() or sha256_file(frozen) != str(row["sha256"]):
            raise ValidationError(f"Attempt frozen input checksum mismatch: {row['role']}")


def _train_run_unlocked(context: RunContext, attempt: Path, *, resume: bool = False) -> PreparedRun:
    prepared = _prepared(context, attempt)
    status = read_status(attempt)["state"]
    audit_existing = attempt / "training_execution_audit.json"
    previous_audit = json.loads(audit_existing.read_text(encoding="utf-8")) if audit_existing.is_file() else {}
    resume_count = int(previous_audit.get("resume_count", 0))
    last_epoch = int(previous_audit.get("completed_epochs", 0)) if previous_audit else None
    resume_checkpoint = attempt / "training_state/last.pt" if resume else None
    gpu_lock = RunLock(
        context.machine.path_value("output_root") / "locks" / f"gpu_{context.machine.data['gpu_id']}.lock",
        {"run_slot": prepared.run_slot, "attempt_id": prepared.attempt_id},
        reclaim_dead_local=True,
    )
    gpu_lock.acquire()
    segment_label = f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    monitor = ResourceMonitor(
        attempt / "02_logs" / f"gpu_usage_{segment_label}.csv",
        context.machine.data["gpu_id"],
        str(context.machine.data.get("nvidia_smi_path") or "nvidia-smi"),
    )
    monitor_started = False
    try:
        if resume and status != "RECOVERING":
            set_status(attempt, "RECOVERING", _status_payload(
                prepared, "train", last_epoch=last_epoch, resume_count=resume_count,
            ))
        if read_status(attempt)["state"] != "RUNNING":
            set_status(attempt, "RUNNING", _status_payload(
                prepared, "train", last_epoch=last_epoch, resume_count=resume_count,
            ))
        monitor.start()
        monitor_started = True
        if bool(context.machine.data.get("dry_run", False)):
            _dry_train_outputs(context, prepared)
        else:
            command = _formal_train_command(context, prepared, resume_checkpoint)
            scratch = context.machine.path_value("local_scratch_root", required=False) or context.machine.path_value("cache_root")
            scratch.mkdir(parents=True, exist_ok=True)
            env = {
                "TMPDIR": str(scratch), "TEMP": str(scratch), "TMP": str(scratch),
                "YOLO_CONFIG_DIR": str(context.machine.path_value("cache_root")),
            }
            timeout = int(context.machine.data.get("command_timeout_seconds") or 0) or None
            run_logged(
                command, _repo_root(context.machine),
                attempt / "02_logs" / f"train_{segment_label}.log", env=env, timeout=timeout,
            )
        audit = _sync_training_outputs(prepared)
        set_status(attempt, "TRAIN_COMPLETED", {
            **_status_payload(
                prepared, "train", last_epoch=int(audit["completed_epochs"]),
                resume_count=int(audit["resume_count"]),
            ),
            "best_sha256": sha256_file(attempt / "03_checkpoints/best.pt"),
            "last_sha256": sha256_file(attempt / "03_checkpoints/last.pt"),
        })
        return prepared
    except Exception as exc:
        try:
            set_status(attempt, "FAILED_TRAIN_RETRYABLE", {
                **_status_payload(
                    prepared, "train", retryable=True, error_code="TRAIN_WORKER_FAILED",
                    last_epoch=last_epoch, resume_count=resume_count,
                ),
                "error": repr(exc),
            })
        except Exception:
            pass
        raise
    finally:
        if monitor_started:
            monitor.stop()
        gpu_lock.release()


def train_run(run_slot: str, machine_config: str | Path, attempt_id: str) -> PreparedRun:
    context = _load_context(machine_config, run_slot)
    parent = _attempt_parent(context.machine, run_slot)
    with RunLock(parent / ".run.lock", {"run_slot": run_slot}, reclaim_dead_local=True):
        attempt = _find_attempt(context.machine, run_slot, attempt_id)
        state = read_status(attempt)["state"]
        resume = state in {"RUNNING", "RECOVERING", "FAILED_TRAIN_RETRYABLE"}
        return _train_run_unlocked(context, attempt, resume=resume)


def _predictions_from_manifests(defect: Path, normal: Path) -> pd.DataFrame:
    frames = []
    for path, label, score in ((defect, 1, 0.9), (normal, 0, 0.1)):
        frame = pd.read_csv(path, dtype={"canonical_image_relpath": "string"})
        frames.append(pd.DataFrame({
            "sample_id": frame["canonical_image_relpath"].astype(str),
            "y_true": label,
            "score": score,
        }))
    return pd.concat(frames, ignore_index=True)


def _publish_evaluation(segment: Path, attempt: Path, prevalence: float) -> dict:
    finalized = segment / "finalized"
    metrics = finalize_evaluation(
        segment / "val_cal_predictions.csv",
        segment / "val_op_predictions.csv",
        finalized,
        prevalence,
    )
    _atomic_copy(finalized / "val_cal_predictions.csv", attempt / "04_predictions/val_cal_predictions.csv")
    _atomic_copy(finalized / "val_op_predictions.csv", attempt / "04_predictions/val_op_predictions.csv")
    for name in ("platt_calibration.json", "operational_metrics.json", "threshold_sweep.csv"):
        _atomic_copy(finalized / name, attempt / "05_metrics" / name)
    return metrics


def _evaluate_run_unlocked(context: RunContext, attempt: Path) -> PreparedRun:
    prepared = _prepared(context, attempt)
    gpu_lock: RunLock | None = None
    if not bool(context.machine.data.get("dry_run", False)):
        gpu_lock = RunLock(
            context.machine.path_value("output_root") / "locks" / f"gpu_{context.machine.data['gpu_id']}.lock",
            {"run_slot": prepared.run_slot, "attempt_id": prepared.attempt_id, "phase": "evaluate"},
            reclaim_dead_local=True,
        )
        gpu_lock.acquire()
    segment = attempt / "work/evaluator" / f"segment_{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    try:
        segment.mkdir(parents=True, exist_ok=False)
        if bool(context.machine.data.get("dry_run", False)):
            _predictions_from_manifests(
                attempt / "01_manifests/frozen_inputs/val_cal_defect_manifest.csv",
                attempt / "01_manifests/frozen_inputs/val_cal_normal_manifest.csv",
            ).to_csv(segment / "val_cal_predictions.csv", index=False)
            _predictions_from_manifests(
                attempt / "01_manifests/frozen_inputs/val_op_defect_manifest.csv",
                attempt / "01_manifests/frozen_inputs/val_op_normal_manifest.csv",
            ).to_csv(segment / "val_op_predictions.csv", index=False)
        else:
            specs = [
                PredictionWorkerSpec(
                    "val_cal",
                    attempt / "01_manifests/frozen_inputs/val_cal_defect_manifest.csv",
                    attempt / "01_manifests/frozen_inputs/val_cal_normal_manifest.csv",
                    segment / "val_cal_predictions.csv", segment / "val_cal_worker.json",
                    attempt / "02_logs" / f"{segment.name}_val_cal.log",
                ),
                PredictionWorkerSpec(
                    "val_op",
                    attempt / "01_manifests/frozen_inputs/val_op_defect_manifest.csv",
                    attempt / "01_manifests/frozen_inputs/val_op_normal_manifest.csv",
                    segment / "val_op_predictions.csv", segment / "val_op_worker.json",
                    attempt / "02_logs" / f"{segment.name}_val_op.log",
                ),
            ]
            evaluation = context.science.data["evaluation_adapter"]
            scratch = context.machine.path_value("local_scratch_root", required=False) or context.machine.path_value("cache_root")
            env = {
                "TMPDIR": str(scratch), "TEMP": str(scratch), "TMP": str(scratch),
                "YOLO_CONFIG_DIR": str(context.machine.path_value("cache_root")),
            }
            run_prediction_workers(
                specs=specs,
                python_executable=str(context.machine.data.get("python_executable") or "python"),
                worker_script=_repo_root(context.machine) / "scripts/stage1_gapvalue240/predict_split_worker.py",
                cwd=_repo_root(context.machine),
                checkpoint=attempt / "03_checkpoints/best.pt",
                dataset_root=context.machine.path_value("dataset_root"),
                yolo_root=_repo_root(context.machine) / "YOLOv11",
                gpu_id=str(context.machine.data["gpu_id"]),
                batch=int(context.machine.data.get("prediction_batch_size") or 256),
                workers=int(context.machine.data.get("prediction_workers") or context.machine.data["num_workers"]),
                imgsz=int(context.science.data["training"]["image_size"]),
                accepted_defect_names=evaluation["accepted_defect_class_names"],
                controller_result_json=segment / "prediction_controller.json",
                timeout_seconds=int(context.machine.data.get("command_timeout_seconds") or 0) or None,
                env=env,
            )
        metrics = _publish_evaluation(
            segment,
            attempt,
            float(context.science.data["calibration"]["deployment_prevalence"]),
        )
        identity = json.loads((attempt / "00_identity/run_identity.json").read_text(encoding="utf-8"))
        set_status(attempt, "EVALUATED", {
            **_status_payload(
                prepared, "evaluate", last_epoch=identity.get("last_epoch"),
                resume_count=int(identity.get("resume_count", 0)),
            ),
            "TN_at_FN95": metrics["TN_at_FN95"],
            "FN_at_TN68253": metrics["FN_at_TN68253"],
        })
        return prepared
    except Exception as exc:
        identity_path = attempt / "00_identity/run_identity.json"
        identity = json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.is_file() else {}
        try:
            set_status(attempt, "FAILED_EVAL_RETRYABLE", {
                **_status_payload(
                    prepared, "evaluate", retryable=True, error_code="EVALUATION_WORKER_FAILED",
                    last_epoch=identity.get("last_epoch"), resume_count=int(identity.get("resume_count", 0)),
                ),
                "error": repr(exc),
            })
        except Exception:
            pass
        raise
    finally:
        if gpu_lock is not None:
            gpu_lock.release()


def evaluate_run(run_slot: str, machine_config: str | Path, attempt_id: str) -> PreparedRun:
    context = _load_context(machine_config, run_slot)
    parent = _attempt_parent(context.machine, run_slot)
    with RunLock(parent / ".run.lock", {"run_slot": run_slot}, reclaim_dead_local=True):
        return _evaluate_run_unlocked(context, _find_attempt(context.machine, run_slot, attempt_id))


def _probe_checkpoint(context: RunContext, attempt: Path, checkpoint: Path, *, resumable: bool) -> dict:
    token = f"{checkpoint.stem}_{uuid.uuid4().hex[:8]}"
    result = run_checkpoint_probe_worker(
        python_executable=str(context.machine.data.get("python_executable") or "python"),
        worker_script=_repo_root(context.machine) / "scripts/stage1_gapvalue240/checkpoint_probe_worker.py",
        checkpoint=checkpoint,
        result_json=attempt / "07_validation" / f"checkpoint_probe_{token}.json",
        log_path=attempt / "02_logs" / f"checkpoint_probe_{token}.log",
        cwd=_repo_root(context.machine),
        require_resume_state=resumable,
        yolo_root=_repo_root(context.machine) / "YOLOv11",
    )
    return result


def _strict_expected(context: RunContext, attempt: Path) -> dict:
    return {
        "epochs": int(context.science.data["training"]["epochs"]),
        "steps_per_epoch": int(context.science.data["training"]["expected_steps"][f"B{int(context.run_row['budget'])}"]),
        "batch_size": int(context.science.data["training"]["batch_size"]),
        "imgsz": int(context.science.data["training"]["image_size"]),
        "seed": int(context.run_row["training_seed"]),
        "model_filename": str(context.machine.path_value("base_checkpoint").name),
        "val_cal_defect_manifest": str(attempt / "01_manifests/frozen_inputs/val_cal_defect_manifest.csv"),
        "val_cal_normal_manifest": str(attempt / "01_manifests/frozen_inputs/val_cal_normal_manifest.csv"),
        "val_op_defect_manifest": str(attempt / "01_manifests/frozen_inputs/val_op_defect_manifest.csv"),
        "val_op_normal_manifest": str(attempt / "01_manifests/frozen_inputs/val_op_normal_manifest.csv"),
    }


def _cleanup_success_temporaries(attempt: Path) -> None:
    for path in (
        attempt / "trainer", attempt / "training_state", attempt / "work",
    ):
        if path.exists():
            shutil.rmtree(path)
    for name in (
        "training_execution_audit.json", "resolved_training_args.json", "storage_preflight.json",
        "formal_environment.json", "checkpoint_preflight.json",
    ):
        (attempt / name).unlink(missing_ok=True)


def _validate_run_unlocked(context: RunContext, attempt: Path) -> PreparedRun:
    prepared = _prepared(context, attempt)
    identity = json.loads((attempt / "00_identity/run_identity.json").read_text(encoding="utf-8"))
    if read_status(attempt)["state"] != "VALIDATING":
        set_status(attempt, "VALIDATING", _status_payload(
            prepared, "validate", last_epoch=identity.get("last_epoch"),
            resume_count=int(identity.get("resume_count", 0)),
        ))
    try:
        def checkpoint_validator(path: Path) -> None:
            if bool(context.machine.data.get("dry_run", False)):
                return
            result = _probe_checkpoint(context, attempt, path, resumable=False)
            if result.get("status") != "PASS":
                raise ValidationError(result.get("error", "checkpoint probe failed"))

        strict_postflight(
            attempt,
            attempt / "07_validation/postflight_report.json",
            _strict_expected(context, attempt),
            checkpoint_validator=checkpoint_validator,
        )
        _cleanup_success_temporaries(attempt)
        artifact_manifest = attempt / "07_validation/artifact_manifest.csv"
        write_permanent_artifact_manifest(attempt, artifact_manifest)
        verify_permanent_artifact_manifest(attempt, artifact_manifest)
        supersedes_id = identity.get("supersedes_attempt_id")
        if supersedes_id:
            previous = _find_attempt(context.machine, prepared.run_slot, str(supersedes_id))
            previous_state = read_status(previous)["state"]
            if previous_state not in {"VALIDATED", "DRY_RUN_VALIDATED", "SUPERSEDED"}:
                raise ValidationError(
                    f"Replacement target is no longer a completed attempt: {previous}: {previous_state}"
                )
            if previous_state != "SUPERSEDED":
                set_status(previous, "SUPERSEDED", {
                    "superseded_by_attempt_id": prepared.attempt_id,
                    "run_slot": prepared.run_slot,
                    "phase": "replacement",
                    "pid": os.getpid(),
                    "retryable": False,
                    "error_code": None,
                    "last_epoch": identity.get("last_epoch"),
                    "resume_count": int(identity.get("resume_count", 0)),
                })
        if attempt.name.endswith(".inprogress"):
            final = attempt.with_name(attempt.name.removesuffix(".inprogress"))
            if final.exists():
                raise FileExistsError(final)
            attempt.rename(final)
            attempt = final
        terminal = validation_status_for_mode(bool(context.machine.data.get("dry_run", False)), context.runtime)
        set_status(attempt, terminal, _status_payload(
            PreparedRun(prepared.run_slot, prepared.attempt_id, attempt, prepared.run_row),
            "complete", last_epoch=identity.get("last_epoch"),
            resume_count=int(identity.get("resume_count", 0)),
        ))
        append_registry(
            context.machine.path_value("output_root") / "registry" / f"{context.machine.data['machine_id']}.jsonl",
            {"event": terminal, "run_slot": prepared.run_slot, "attempt_id": prepared.attempt_id, "path": str(attempt)},
        )
        return PreparedRun(prepared.run_slot, prepared.attempt_id, attempt, prepared.run_row)
    except Exception as exc:
        try:
            set_status(attempt, "INVALID_ARTIFACT", {
                **_status_payload(
                    PreparedRun(prepared.run_slot, prepared.attempt_id, attempt, prepared.run_row),
                    "validate", retryable=False, error_code="POSTFLIGHT_FAILED",
                    last_epoch=identity.get("last_epoch"), resume_count=int(identity.get("resume_count", 0)),
                ),
                "error": repr(exc),
            })
        except Exception:
            pass
        raise


def validate_run(run_slot: str, machine_config: str | Path, attempt_id: str) -> PreparedRun:
    context = _load_context(machine_config, run_slot)
    parent = _attempt_parent(context.machine, run_slot)
    with RunLock(parent / ".run.lock", {"run_slot": run_slot}, reclaim_dead_local=True):
        return _validate_run_unlocked(context, _find_attempt(context.machine, run_slot, attempt_id))


def _resume_checkpoint_valid(context: RunContext, attempt: Path) -> bool:
    checkpoint = attempt / "training_state/last.pt"
    if bool(context.machine.data.get("dry_run", False)):
        return checkpoint.is_file() and checkpoint.stat().st_size > 0
    try:
        result = _probe_checkpoint(context, attempt, checkpoint, resumable=True)
        return result.get("status") == "PASS"
    except Exception:
        return False


def _discover(context: RunContext) -> RunDecision:
    expected = dict(context.identity)
    return discover_run_action(
        _attempt_parent(context.machine, str(context.run_row["run_slot"])),
        expected,
        checkpoint_validator=lambda path: _resume_checkpoint_valid(context, path.parents[1]),
    )


def _execute_decision(context: RunContext, decision: RunDecision) -> PreparedRun:
    if decision.action in {"SKIP_VALIDATED", "SKIP_DRY_RUN"}:
        assert decision.attempt_dir is not None
        return _prepared(context, decision.attempt_dir)
    if decision.action == "NEW_ATTEMPT":
        if decision.superseded_attempt is not None and read_status(decision.superseded_attempt).get("state") in {
            "RUNNING", "RECOVERING", "FAILED_TRAIN_RETRYABLE"
        }:
            old = _prepared(context, decision.superseded_attempt)
            try:
                set_status(decision.superseded_attempt, "FAILED_TRAIN", {
                    **_status_payload(old, "train", retryable=False, error_code="CORRUPT_RESUME_CHECKPOINT"),
                    "error": decision.reason,
                })
            except Exception:
                pass
        previous_state = (
            read_status(decision.superseded_attempt)["state"]
            if decision.superseded_attempt is not None else None
        )
        if decision.superseded_attempt is not None and previous_state == "PLANNED":
            old = PreparedRun(
                str(context.run_row["run_slot"]),
                _attempt_id(decision.superseded_attempt),
                decision.superseded_attempt,
                context.run_row,
            )
            set_status(decision.superseded_attempt, "FAILED_INPUT", {
                **_status_payload(old, "prepare", retryable=False, error_code="PREPARE_INTERRUPTED"),
                "error": decision.reason,
            })
        replacement = (
            decision.superseded_attempt
            if previous_state in {"VALIDATED", "DRY_RUN_VALIDATED"} else None
        )
        prepared = _prepare_run_unlocked(context, supersedes_attempt=replacement)
        trained = _train_run_unlocked(context, prepared.attempt_dir)
        evaluated = _evaluate_run_unlocked(context, trained.attempt_dir)
        return _validate_run_unlocked(context, evaluated.attempt_dir)
    assert decision.attempt_dir is not None
    if decision.action == "TRAIN":
        trained = _train_run_unlocked(context, decision.attempt_dir)
        evaluated = _evaluate_run_unlocked(context, trained.attempt_dir)
        return _validate_run_unlocked(context, evaluated.attempt_dir)
    if decision.action == "RESUME_TRAIN":
        trained = _train_run_unlocked(context, decision.attempt_dir, resume=True)
        evaluated = _evaluate_run_unlocked(context, trained.attempt_dir)
        return _validate_run_unlocked(context, evaluated.attempt_dir)
    if decision.action == "EVALUATE":
        evaluated = _evaluate_run_unlocked(context, decision.attempt_dir)
        return _validate_run_unlocked(context, evaluated.attempt_dir)
    if decision.action == "VALIDATE":
        return _validate_run_unlocked(context, decision.attempt_dir)
    raise AssertionError(f"Unknown execution decision: {decision}")


def run_all(
    run_slot: str,
    machine_config: str | Path,
    attempt_id: str | None = None,
    allow_new_attempt_after_validated: bool = False,
) -> PreparedRun:
    context = _load_context(machine_config, run_slot)
    parent = _attempt_parent(context.machine, run_slot)
    with RunLock(parent / ".run.lock", {"run_slot": run_slot}, reclaim_dead_local=True):
        if attempt_id is not None:
            attempt = _find_attempt(context.machine, run_slot, attempt_id)
            state = read_status(attempt)["state"]
            mapping = {
                "STAGED": "TRAIN", "RUNNING": "RESUME_TRAIN", "RECOVERING": "RESUME_TRAIN",
                "FAILED_TRAIN_RETRYABLE": "RESUME_TRAIN", "TRAIN_COMPLETED": "EVALUATE",
                "FAILED_EVAL_RETRYABLE": "EVALUATE", "EVALUATED": "VALIDATE",
                "INVALID_ARTIFACT": "VALIDATE", "VALIDATING": "VALIDATE",
                "VALIDATED": "SKIP_VALIDATED", "DRY_RUN_VALIDATED": "SKIP_DRY_RUN",
            }
            decision = RunDecision(mapping.get(state, "NEW_ATTEMPT"), attempt_dir=attempt)
        else:
            decision = _discover(context)
        if allow_new_attempt_after_validated and decision.action in {"SKIP_VALIDATED", "SKIP_DRY_RUN"}:
            decision = RunDecision("NEW_ATTEMPT", superseded_attempt=decision.attempt_dir, reason="explicit new attempt")
        return _execute_decision(context, decision)


def run_entry_cli(run_slot: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine-config", required=True)
    parser.add_argument("--action", choices=("prepare", "train", "evaluate", "validate", "run"), default="run")
    parser.add_argument("--attempt-id")
    parser.add_argument("--allow-new-attempt-after-validated", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            result = prepare_run(run_slot, args.machine_config, args.attempt_id, args.allow_new_attempt_after_validated)
        elif args.action == "train":
            result = train_run(run_slot, args.machine_config, args.attempt_id)
        elif args.action == "evaluate":
            result = evaluate_run(run_slot, args.machine_config, args.attempt_id)
        elif args.action == "validate":
            result = validate_run(run_slot, args.machine_config, args.attempt_id)
        else:
            result = run_all(run_slot, args.machine_config, args.attempt_id, args.allow_new_attempt_after_validated)
        print(json.dumps({
            "status": "PASS", "run_slot": result.run_slot, "attempt_id": result.attempt_id,
            "attempt_dir": str(result.attempt_dir), "exit_code": EXIT_SUCCESS,
        }, ensure_ascii=False))
        return EXIT_SUCCESS
    except BaseException as exc:
        code = exit_code_for_exception(exc)
        print(json.dumps({
            "status": "FAIL", "run_slot": run_slot, "error_type": type(exc).__name__,
            "error": str(exc), "retryable": code == EXIT_RETRYABLE, "exit_code": code,
        }, ensure_ascii=False), file=__import__("sys").stderr)
        return code
