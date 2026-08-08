from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
import os
from pathlib import Path
import time

import pandas as pd

from stage1_gapvalue240.campaign_dynamic_training import DynamicTrainingSpec, run_dynamic_training_segment
from stage1_gapvalue240.campaign_benchmark import compare_training_numerical_parity
from stage1_gapvalue240.campaign_canonical_lock import load_canonical_training_lock
from stage1_gapvalue240.campaign_process_telemetry import ProcessTelemetrySpec
from stage1_gapvalue240.campaign_smoke import LocalSmokeValidationSpec, validate_local_smoke_run
from stage1_gapvalue240.campaign_smoke_dataset import prepare_local_smoke_dataset
from stage1_gapvalue240.errors import ValidationError
from stage1_gapvalue240.monitor import ResourceMonitor
from stage1_gapvalue240.util import atomic_write_json, sha256_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark dynamic process telemetry on a tiny real dataset.")
    parser.add_argument("--repo-root", default=str(_BootstrapPath(__file__).resolve().parents[2]))
    parser.add_argument("--scratch-root", default="C:/gapvalue240_dynamic_telemetry_benchmark")
    parser.add_argument("--train-per-class", type=int, default=32)
    parser.add_argument("--val-per-class", type=int, default=8)
    parser.add_argument("--replay-normal", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--telemetry-first", action="store_true")
    return parser.parse_args(argv)


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _run(
    *,
    run_id: str,
    output: Path,
    dataset: Path,
    repo: Path,
    base_normal: Path,
    base_defect: Path,
    replay_identity: Path,
    monitor: Path,
    expected_samples: int,
    expected_steps: int,
    args: argparse.Namespace,
    telemetry_enabled: bool,
) -> dict[str, object]:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    telemetry = None
    if telemetry_enabled:
        telemetry = ProcessTelemetrySpec(
            run_id=run_id,
            arm_id="BENCHMARK",
            segment_id=f"{run_id}_E001_{args.epochs:03d}",
            output_dir=output / "process_telemetry",
            base_normal_manifest=base_normal,
            base_defect_manifest=base_defect,
            replay_identity_manifest=replay_identity,
            monitor_manifest=monitor,
            expected_epoch_samples=expected_samples,
            expected_replay_samples=args.replay_normal,
        )
    canonical_lock = repo / "configs/stage1_gapvalue240/CANONICAL_TRAINING_LOCK_v1.json"
    lock = load_canonical_training_lock(canonical_lock)
    observed_dimensions = {
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": 224,
        "workers": args.workers,
    }
    smoke_overrides = tuple(
        key
        for key, value in observed_dimensions.items()
        if value != lock.immutable_args[key]
    )
    spec = DynamicTrainingSpec(
        run_id=run_id,
        arm_id="BENCHMARK",
        schedule_id=f"BENCHMARK_{args.epochs}_EPOCHS",
        selection_digest="A" * 64,
        active_selection_digest="B" * 64,
        dataset_dir=dataset,
        checkpoint=repo / "yolo11l-cls.pt",
        output_dir=output,
        yolo_root=repo / "YOLOv11",
        total_epochs=args.epochs,
        segment_start_epoch=1,
        segment_end_epoch=args.epochs,
        batch=args.batch,
        imgsz=224,
        seed=12345,
        device=args.device,
        workers=args.workers,
        expected_steps_per_epoch=expected_steps,
        retained_checkpoint_epochs=(args.epochs,),
        execution_mode="SMOKE",
        segment_id=f"{run_id}_E001_{args.epochs:03d}",
        process_telemetry=telemetry,
        canonical_lock_path=canonical_lock,
        canonical_lock_file_sha256=sha256_file(canonical_lock),
        smoke_canonical_overrides=smoke_overrides,
    )
    resource_log = output / "resource_logs/resource.csv"
    resource_monitor = ResourceMonitor(
        resource_log,
        args.device,
        interval=1.0,
        process_pid=os.getpid(),
        disk_path=output,
    )
    resource_monitor.set_phase("LOCAL_REAL_DATA_SMOKE")
    resource_monitor.start()
    started = time.perf_counter()
    try:
        result = run_dynamic_training_segment(spec)
    finally:
        resource_monitor.stop()
    duration = time.perf_counter() - started
    telemetry_bytes = _directory_bytes(output / "process_telemetry") if telemetry_enabled else 0
    telemetry_artifacts = []
    smoke_validation = None
    if telemetry_enabled:
        for epoch in range(1, args.epochs + 1):
            parquet = output / f"process_telemetry/epoch_{epoch:04d}_process_telemetry.parquet"
            sidecar = parquet.with_suffix(".json")
            if not parquet.is_file() or not sidecar.is_file():
                raise ValidationError(f"missing smoke telemetry for epoch {epoch}")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            telemetry_artifacts.append(
                {
                    "epoch": epoch,
                    "parquet": str(parquet),
                    "parquet_bytes": parquet.stat().st_size,
                    "parquet_sha256": sha256_file(parquet),
                    "sidecar": str(sidecar),
                    "status": metadata.get("status"),
                    "row_count": metadata.get("row_count"),
                    "observed_epoch_samples": metadata.get("observed_epoch_samples"),
                }
            )
        smoke_result = validate_local_smoke_run(
            LocalSmokeValidationSpec(
                run_id=run_id,
                arm_id="BENCHMARK",
                output_dir=output,
                telemetry=telemetry,
                expected_epochs=args.epochs,
                expected_steps_per_epoch=expected_steps,
                canonical_lock_file_sha256=sha256_file(canonical_lock),
                declared_smoke_overrides=smoke_overrides,
                resource_log=resource_log,
            )
        )
        smoke_validation = {
            "report_path": str(smoke_result.report_path),
            "artifact_manifest_path": str(smoke_result.artifact_manifest_path),
            "artifact_count": smoke_result.artifact_count,
            "total_bytes": smoke_result.total_bytes,
        }
    return {
        "run_id": run_id,
        "telemetry_enabled": telemetry_enabled,
        "duration_seconds": duration,
        "telemetry_bytes": telemetry_bytes,
        "total_output_bytes": _directory_bytes(output),
        "completed_epochs": result.completed_epoch,
        "telemetry_epoch_count": len(telemetry_artifacts),
        "telemetry_artifacts": telemetry_artifacts,
        "audit_path": str(result.audit_path),
        "stable_last_sha256": sha256_file(result.stable_last),
        "canonical_lock_file_sha256": sha256_file(canonical_lock),
        "declared_smoke_overrides": list(smoke_overrides),
        "resolved_args_path": str(result.resolved_args_path),
        "resource_log": str(resource_log),
        "resource_log_bytes": resource_log.stat().st_size,
        "smoke_validation": smoke_validation,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = Path(args.repo_root).resolve()
    scratch = Path(args.scratch_root).resolve()
    if scratch.exists():
        raise FileExistsError(f"benchmark scratch already exists: {scratch}")
    dataset_root = repo / "data/final_sewerml_dataset"
    try:
        subset = prepare_local_smoke_dataset(
            dataset_root,
            scratch,
            train_per_class=args.train_per_class,
            val_per_class=args.val_per_class,
            replay_normal=args.replay_normal,
            batch=args.batch,
        )
        dataset = subset.dataset_dir
        base_normal = subset.base_normal_manifest
        base_defect = subset.base_defect_manifest
        replay_identity = subset.replay_identity_manifest
        monitor = subset.monitor_manifest
        expected_samples = subset.expected_epoch_samples
        expected_steps = subset.expected_steps_per_epoch
        run_specs = []
        if not args.skip_baseline:
            run_specs.append(
                dict(
                    run_id="LOCAL_BASELINE",
                    output=scratch / "baseline",
                    dataset=dataset,
                    repo=repo,
                    base_normal=base_normal,
                    base_defect=base_defect,
                    replay_identity=replay_identity,
                    monitor=monitor,
                    expected_samples=expected_samples,
                    expected_steps=expected_steps,
                    args=args,
                    telemetry_enabled=False,
                )
            )
        run_specs.append(
            dict(
                run_id="LOCAL_TELEMETRY",
                output=scratch / "telemetry",
                dataset=dataset,
                repo=repo,
                base_normal=base_normal,
                base_defect=base_defect,
                replay_identity=replay_identity,
                monitor=monitor,
                expected_samples=expected_samples,
                expected_steps=expected_steps,
                args=args,
                telemetry_enabled=True,
            )
        )
        if args.telemetry_first:
            run_specs.sort(key=lambda item: not bool(item["telemetry_enabled"]))
        results = [_run(**run_spec) for run_spec in run_specs]
        payload = {
            "schema_version": "stage1.dynamic_telemetry_benchmark.v1",
            "status": "PASS",
            "train_per_class": args.train_per_class,
            "val_per_class": args.val_per_class,
            "replay_normal": args.replay_normal,
            "epochs": args.epochs,
            "batch": args.batch,
            "expected_epoch_samples": expected_samples,
            "expected_steps": expected_steps,
            "execution_order": [str(result["run_id"]) for result in results],
            "runs": results,
        }
        if len(results) == 2:
            by_id = {str(result["run_id"]): result for result in results}
            baseline_seconds = float(by_id["LOCAL_BASELINE"]["duration_seconds"])
            telemetry_seconds = float(by_id["LOCAL_TELEMETRY"]["duration_seconds"])
            payload["observed_pair_runtime_ratio"] = telemetry_seconds / baseline_seconds
            payload["runtime_comparison_note"] = (
                "Single-pair runtime is order-sensitive because the first run pays process/GPU/cache warmup; "
                "use crossed execution orders before estimating telemetry overhead."
            )
            parity = compare_training_numerical_parity(
                baseline_checkpoint=scratch
                / f"baseline/training_state/checkpoint_epoch_{args.epochs:04d}.pt",
                telemetry_checkpoint=scratch
                / f"telemetry/training_state/checkpoint_epoch_{args.epochs:04d}.pt",
                baseline_results=scratch / "baseline/trainer/results.csv",
                telemetry_results=scratch / "telemetry/trainer/results.csv",
                yolo_root=repo / "YOLOv11",
            )
            payload["numerical_parity"] = parity
            if not parity["numerical_parity_passed"]:
                payload["status"] = "FAILED_NUMERICAL_PARITY"
        atomic_write_json(scratch / "BENCHMARK_RESULT.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if payload["status"] != "PASS":
            raise ValidationError("process telemetry changed the numerical training outcome")
        return 0
    except Exception:
        atomic_write_json(
            scratch / "BENCHMARK_FAILED.json",
            {"status": "FAILED", "created_at_unix": time.time()},
            overwrite=True,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
