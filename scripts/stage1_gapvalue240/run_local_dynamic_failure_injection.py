from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from stage1_gapvalue240.campaign_dynamic_training import (
    SmokeFailureInjection,
    run_dynamic_training_segment,
)
from stage1_gapvalue240.campaign_failure_smoke import (
    build_failure_smoke_segment,
    validate_interrupted_boundary,
    validate_telemetry_write_interruption,
)
from stage1_gapvalue240.campaign_process_telemetry import validate_process_telemetry_epoch
from stage1_gapvalue240.campaign_smoke import (
    LocalSmokeValidationSpec,
    validate_local_smoke_run,
)
from stage1_gapvalue240.campaign_smoke_dataset import (
    LocalSmokeDataset,
    prepare_local_smoke_dataset,
)
from stage1_gapvalue240.campaign_worker import archive_zero_epoch_attempt
from stage1_gapvalue240.errors import ValidationError
from stage1_gapvalue240.monitor import ResourceMonitor
from stage1_gapvalue240.util import atomic_write_json, sha256_file


ROOT = _BootstrapPath(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local real-data OOM, process-kill, write-interruption, and corruption drills."
    )
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--scratch-root", default="C:/gapvalue240_dynamic_failure_injection")
    parser.add_argument("--train-per-class", type=int, default=32)
    parser.add_argument("--val-per-class", type=int, default=8)
    parser.add_argument("--replay-normal", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=24681357)
    parser.add_argument("--marker-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--child-config", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _subset_payload(subset: LocalSmokeDataset) -> dict[str, Any]:
    return {
        "root": str(subset.root),
        "dataset_dir": str(subset.dataset_dir),
        "base_normal_manifest": str(subset.base_normal_manifest),
        "base_defect_manifest": str(subset.base_defect_manifest),
        "replay_identity_manifest": str(subset.replay_identity_manifest),
        "monitor_manifest": str(subset.monitor_manifest),
        "expected_epoch_samples": subset.expected_epoch_samples,
        "expected_replay_samples": subset.expected_replay_samples,
        "expected_steps_per_epoch": subset.expected_steps_per_epoch,
        "validation_path": str(subset.validation_path),
    }


def _subset_from_payload(payload: dict[str, Any]) -> LocalSmokeDataset:
    return LocalSmokeDataset(
        root=Path(payload["root"]).resolve(),
        dataset_dir=Path(payload["dataset_dir"]).resolve(),
        base_normal_manifest=Path(payload["base_normal_manifest"]).resolve(),
        base_defect_manifest=Path(payload["base_defect_manifest"]).resolve(),
        replay_identity_manifest=Path(payload["replay_identity_manifest"]).resolve(),
        monitor_manifest=Path(payload["monitor_manifest"]).resolve(),
        expected_epoch_samples=int(payload["expected_epoch_samples"]),
        expected_replay_samples=int(payload["expected_replay_samples"]),
        expected_steps_per_epoch=int(payload["expected_steps_per_epoch"]),
        validation_path=Path(payload["validation_path"]).resolve(),
    )


def _run_segment(built, resource_log: Path, *, nvidia_smi: str) -> tuple[Any, float]:
    monitor = ResourceMonitor(
        resource_log,
        built.training.device,
        nvidia_smi,
        interval=0.5,
        process_pid=os.getpid(),
        disk_path=built.training.output_dir,
    )
    monitor.set_phase("LOCAL_FAILURE_INJECTION")
    monitor.start()
    started = time.perf_counter()
    try:
        result = run_dynamic_training_segment(built.training)
    finally:
        monitor.stop()
    return result, time.perf_counter() - started


def _validate_completed(
    built,
    *,
    epochs: int,
    resource_log: Path,
    segment_ids: tuple[str, ...],
) -> dict[str, Any]:
    validation = validate_local_smoke_run(
        LocalSmokeValidationSpec(
            run_id=built.training.run_id,
            arm_id=built.training.arm_id,
            output_dir=built.training.output_dir,
            telemetry=built.telemetry,
            expected_epochs=epochs,
            expected_steps_per_epoch=built.training.expected_steps_per_epoch,
            canonical_lock_file_sha256=str(built.training.canonical_lock_file_sha256),
            declared_smoke_overrides=built.smoke_canonical_overrides,
            resource_log=resource_log,
            telemetry_segment_ids_by_epoch=segment_ids,
        )
    )
    return {
        "status": "PASS",
        "report_path": str(validation.report_path),
        "report_sha256": sha256_file(validation.report_path),
        "artifact_manifest_path": str(validation.artifact_manifest_path),
        "artifact_manifest_sha256": sha256_file(validation.artifact_manifest_path),
        "artifact_count": validation.artifact_count,
        "total_bytes": validation.total_bytes,
    }


def _child_main(config_path: Path) -> int:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "stage1.local_failure_child.v1":
        raise ValidationError("invalid local failure child config schema")
    subset = _subset_from_payload(payload["subset"])
    reached = Path(payload["reached_marker"]).resolve()
    continue_marker = Path(payload["continue_marker"]).resolve()
    built = build_failure_smoke_segment(
        repo_root=payload["repo_root"],
        subset=subset,
        output_dir=payload["output_dir"],
        run_id=payload["run_id"],
        total_epochs=int(payload["total_epochs"]),
        segment_start_epoch=1,
        segment_end_epoch=int(payload["total_epochs"]),
        batch=int(payload["batch"]),
        workers=int(payload["workers"]),
        device=str(payload["device"]),
        seed=int(payload["seed"]),
        segment_id=str(payload["segment_id"]),
        failure_injection=SmokeFailureInjection(
            mode="PAUSE_AT_EPOCH_START",
            target_epoch=int(payload["pause_epoch"]),
            marker_path=reached,
            continue_marker_path=continue_marker,
            timeout_seconds=float(payload["marker_timeout_seconds"]),
        ),
    )
    _run_segment(
        built,
        Path(payload["resource_log"]).resolve(),
        nvidia_smi=str(payload["nvidia_smi"]),
    )
    raise ValidationError("process-kill child returned without being terminated")


def _wait_and_kill(
    process: subprocess.Popen,
    reached_marker: Path,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while not reached_marker.is_file():
        returncode = process.poll()
        if returncode is not None:
            raise ValidationError(f"process-kill child exited before marker: {returncode}")
        if time.monotonic() >= deadline:
            process.kill()
            process.wait(timeout=30)
            raise TimeoutError("process-kill child did not reach the registered pause marker")
        time.sleep(0.2)
    marker = json.loads(reached_marker.read_text(encoding="utf-8"))
    process.kill()
    returncode = process.wait(timeout=60)
    if returncode == 0:
        raise ValidationError("process-kill child unexpectedly returned success")
    return {"marker": marker, "returncode": returncode}


def _scenario_oom(args: argparse.Namespace, repo: Path, subset: LocalSmokeDataset, root: Path) -> dict:
    output = root / "oom_zero_epoch_recovery"
    attempt_segment = "OOM_ATTEMPT_E001"
    injection = SmokeFailureInjection(
        mode="OOM_AT_BATCH_START",
        target_epoch=1,
        target_batch=1,
    )
    attempted = build_failure_smoke_segment(
        repo_root=repo,
        subset=subset,
        output_dir=output,
        run_id="LOCAL_FAILURE_OOM",
        total_epochs=args.epochs,
        segment_start_epoch=1,
        segment_end_epoch=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        segment_id=attempt_segment,
        failure_injection=injection,
    )
    failure_log = output / "resource_logs/oom_failed_attempt.csv"
    started = time.perf_counter()
    try:
        _run_segment(attempted, failure_log, nvidia_smi=args.nvidia_smi)
    except Exception as exc:
        import torch

        if not isinstance(exc, torch.cuda.OutOfMemoryError):
            raise
        injected_error = f"{type(exc).__name__}: {exc}"
    else:
        raise ValidationError("injected OOM scenario unexpectedly completed")
    archive = archive_zero_epoch_attempt(output)
    recovery_segment = "OOM_RECOVERY_E001"
    recovery = build_failure_smoke_segment(
        repo_root=repo,
        subset=subset,
        output_dir=output,
        run_id="LOCAL_FAILURE_OOM",
        total_epochs=args.epochs,
        segment_start_epoch=1,
        segment_end_epoch=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        segment_id=recovery_segment,
    )
    recovery_log = output / "resource_logs/oom_recovery.csv"
    result, recovery_seconds = _run_segment(
        recovery, recovery_log, nvidia_smi=args.nvidia_smi
    )
    validation = _validate_completed(
        recovery,
        epochs=args.epochs,
        resource_log=recovery_log,
        segment_ids=(recovery_segment,) * args.epochs,
    )
    return {
        "status": "PASS",
        "injected_error": injected_error,
        "total_duration_seconds": time.perf_counter() - started,
        "recovery_duration_seconds": recovery_seconds,
        "recovery_completed_epoch": result.completed_epoch,
        "failed_attempt_archive": str(archive),
        "failed_attempt_manifest_sha256": sha256_file(
            archive / "ATTEMPT_ARCHIVE_MANIFEST.json"
        ),
        "validation": validation,
    }


def _scenario_process_kill(
    args: argparse.Namespace,
    repo: Path,
    subset: LocalSmokeDataset,
    root: Path,
) -> dict:
    output = root / "process_kill_resume"
    reached = output / "failure_injection/reached_epoch_0002.json"
    continue_marker = output / "failure_injection/never_continue.marker"
    first_segment = "KILL_BEFORE_EPOCH_002"
    config = root / "process_kill_child_config.json"
    child_log = root / "process_kill_child.log"
    child_resource = output / "resource_logs/killed_process.csv"
    atomic_write_json(
        config,
        {
            "schema_version": "stage1.local_failure_child.v1",
            "repo_root": str(repo),
            "subset": _subset_payload(subset),
            "output_dir": str(output),
            "run_id": "LOCAL_FAILURE_KILL",
            "total_epochs": args.epochs,
            "batch": args.batch,
            "workers": args.workers,
            "device": args.device,
            "seed": args.seed,
            "segment_id": first_segment,
            "pause_epoch": 2,
            "reached_marker": str(reached),
            "continue_marker": str(continue_marker),
            "marker_timeout_seconds": args.marker_timeout_seconds,
            "resource_log": str(child_resource),
            "nvidia_smi": args.nvidia_smi,
        },
    )
    command = [sys.executable, str(Path(__file__).resolve()), "--child-config", str(config)]
    started = time.perf_counter()
    with child_log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=repo,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        killed = _wait_and_kill(
            process,
            reached,
            timeout_seconds=args.marker_timeout_seconds,
        )
    boundary = validate_interrupted_boundary(
        output,
        expected_completed_epoch=1,
        expected_next_epoch=2,
    )
    resume_segment = "RESUME_E002_TO_FINAL"
    resume = build_failure_smoke_segment(
        repo_root=repo,
        subset=subset,
        output_dir=output,
        run_id="LOCAL_FAILURE_KILL",
        total_epochs=args.epochs,
        segment_start_epoch=2,
        segment_end_epoch=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        segment_id=resume_segment,
        resume_checkpoint=output / "training_state/last.pt",
    )
    resume_log = output / "resource_logs/resumed_process.csv"
    result, resume_seconds = _run_segment(resume, resume_log, nvidia_smi=args.nvidia_smi)
    validation = _validate_completed(
        resume,
        epochs=args.epochs,
        resource_log=resume_log,
        segment_ids=(first_segment,) + (resume_segment,) * (args.epochs - 1),
    )
    return {
        "status": "PASS",
        "total_duration_seconds": time.perf_counter() - started,
        "resume_duration_seconds": resume_seconds,
        "child_returncode": killed["returncode"],
        "pause_marker": killed["marker"],
        "interrupted_boundary": boundary,
        "resumed_completed_epoch": result.completed_epoch,
        "child_log": str(child_log),
        "child_log_sha256": sha256_file(child_log),
        "validation": validation,
    }


def _scenario_write_and_corruption(
    args: argparse.Namespace,
    repo: Path,
    subset: LocalSmokeDataset,
    root: Path,
) -> dict:
    failed_output = root / "telemetry_write_interruption"
    failed = build_failure_smoke_segment(
        repo_root=repo,
        subset=subset,
        output_dir=failed_output,
        run_id="LOCAL_FAILURE_WRITE",
        total_epochs=1,
        segment_start_epoch=1,
        segment_end_epoch=1,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        segment_id="WRITE_INTERRUPTION_E001",
        failure_injection=SmokeFailureInjection(
            mode="TELEMETRY_WRITE_INTERRUPTION",
            target_epoch=1,
        ),
    )
    failed_log = failed_output / "resource_logs/write_interrupted.csv"
    try:
        _run_segment(failed, failed_log, nvidia_smi=args.nvidia_smi)
    except OSError as exc:
        injected_error = f"{type(exc).__name__}: {exc}"
    else:
        raise ValidationError("telemetry write interruption unexpectedly completed")
    interruption = validate_telemetry_write_interruption(failed_output, failed_epoch=1)

    hot_spare_output = root / "telemetry_write_hot_spare_rerun"
    clean_segment = "HOT_SPARE_RERUN_E001"
    clean = build_failure_smoke_segment(
        repo_root=repo,
        subset=subset,
        output_dir=hot_spare_output,
        run_id="LOCAL_FAILURE_WRITE_HOT_SPARE",
        total_epochs=1,
        segment_start_epoch=1,
        segment_end_epoch=1,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        segment_id=clean_segment,
    )
    clean_log = hot_spare_output / "resource_logs/hot_spare.csv"
    _result, clean_seconds = _run_segment(clean, clean_log, nvidia_smi=args.nvidia_smi)
    validation = _validate_completed(
        clean,
        epochs=1,
        resource_log=clean_log,
        segment_ids=(clean_segment,),
    )

    evidence = root / "corrupt_sidecar_evidence"
    evidence.mkdir()
    clean_parquet = hot_spare_output / "process_telemetry/epoch_0001_process_telemetry.parquet"
    clean_sidecar = clean_parquet.with_suffix(".json")
    corrupt_parquet = evidence / clean_parquet.name
    corrupt_sidecar = evidence / clean_sidecar.name
    shutil.copy2(clean_parquet, corrupt_parquet)
    metadata = json.loads(clean_sidecar.read_text(encoding="utf-8"))
    metadata["parquet_sha256"] = "0" * 64
    atomic_write_json(corrupt_sidecar, metadata)
    corrupt_spec = replace(clean.telemetry, output_dir=evidence)
    try:
        validate_process_telemetry_epoch(corrupt_spec, 1)
    except ValidationError as exc:
        corruption_error = f"{type(exc).__name__}: {exc}"
    else:
        raise ValidationError("corrupt telemetry sidecar was incorrectly accepted")
    return {
        "status": "PASS",
        "injected_error": injected_error,
        "interruption_validation": interruption,
        "hot_spare_duration_seconds": clean_seconds,
        "hot_spare_validation": validation,
        "corrupt_sidecar_rejected": True,
        "corruption_error": corruption_error,
        "corrupt_sidecar_path": str(corrupt_sidecar),
        "corrupt_sidecar_sha256": sha256_file(corrupt_sidecar),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.child_config:
        return _child_main(Path(args.child_config).resolve())
    if args.epochs < 2:
        raise ValueError("--epochs must be at least 2 for the process-kill resume drill")
    if min(args.train_per_class, args.val_per_class, args.replay_normal, args.batch) <= 0:
        raise ValueError("local failure smoke sizes and batch must be positive")
    if args.workers < 0 or args.marker_timeout_seconds <= 0:
        raise ValueError("invalid local failure smoke worker/timeout setting")
    repo = Path(args.repo_root).resolve()
    scratch = Path(args.scratch_root).resolve()
    if scratch.exists():
        raise FileExistsError(f"failure-injection scratch already exists: {scratch}")
    scratch.mkdir(parents=True)
    started = time.perf_counter()
    try:
        subset = prepare_local_smoke_dataset(
            repo / "data/final_sewerml_dataset",
            scratch / "shared_real_subset",
            train_per_class=args.train_per_class,
            val_per_class=args.val_per_class,
            replay_normal=args.replay_normal,
            batch=args.batch,
        )
        scenarios = {
            "oom_zero_epoch_recovery": _scenario_oom(args, repo, subset, scratch),
            "process_kill_checkpoint_resume": _scenario_process_kill(
                args, repo, subset, scratch
            ),
            "telemetry_write_and_sidecar_corruption": _scenario_write_and_corruption(
                args, repo, subset, scratch
            ),
        }
        report = scratch / "FAILURE_INJECTION_REPORT.json"
        atomic_write_json(
            report,
            {
                "schema_version": "stage1.local_real_failure_injection.v1",
                "status": "PASS",
                "repo_root": str(repo),
                "scratch_root": str(scratch),
                "duration_seconds": time.perf_counter() - started,
                "epochs": args.epochs,
                "batch": args.batch,
                "workers": args.workers,
                "device": args.device,
                "subset_validation": str(subset.validation_path),
                "subset_validation_sha256": sha256_file(subset.validation_path),
                "scenarios": scenarios,
            },
        )
        print(json.dumps(json.loads(report.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        atomic_write_json(
            scratch / "FAILURE_INJECTION_FAILED.json",
            {
                "schema_version": "stage1.local_real_failure_injection.v1",
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "duration_seconds": time.perf_counter() - started,
            },
            overwrite=True,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
