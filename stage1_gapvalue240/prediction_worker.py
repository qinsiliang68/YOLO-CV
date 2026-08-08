from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .predictor import predict_split
from .util import atomic_write_json, sha256_file


WORKER_EXIT_CODES = {
    "PASS": 0,
    "INVALID_INPUT": 2,
    "PREDICTION_FAILED": 3,
    "ARTIFACT_INVALID": 4,
}


@dataclass(frozen=True)
class PredictionJob:
    split: str
    checkpoint: Path
    dataset_root: Path
    defect_manifest: Path
    normal_manifest: Path
    output: Path
    result_json: Path
    yolo_root: Path
    gpu_id: str
    batch: int
    workers: int
    imgsz: int
    accepted_defect_names: tuple[str, ...]


def _resolved(job: PredictionJob) -> PredictionJob:
    return PredictionJob(
        split=job.split,
        checkpoint=Path(job.checkpoint).resolve(),
        dataset_root=Path(job.dataset_root).resolve(),
        defect_manifest=Path(job.defect_manifest).resolve(),
        normal_manifest=Path(job.normal_manifest).resolve(),
        output=Path(job.output).resolve(),
        result_json=Path(job.result_json).resolve(),
        yolo_root=Path(job.yolo_root).resolve(),
        gpu_id=str(job.gpu_id),
        batch=int(job.batch),
        workers=int(job.workers),
        imgsz=int(job.imgsz),
        accepted_defect_names=tuple(job.accepted_defect_names),
    )


def _validate_inputs(job: PredictionJob) -> tuple[int, int]:
    if job.split not in {"val_cal", "val_op", "causal_train_probe"}:
        raise ValueError(f"Unsupported split: {job.split}")
    for path in (job.checkpoint, job.defect_manifest, job.normal_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (job.dataset_root, job.yolo_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if job.output.exists():
        raise FileExistsError(f"Prediction output already exists: {job.output}")
    if job.result_json.exists():
        raise FileExistsError(f"Worker result already exists: {job.result_json}")
    if job.batch <= 0 or job.workers < 0 or job.imgsz <= 0:
        raise ValueError("batch/imgsz must be positive and workers must be nonnegative")
    if not job.accepted_defect_names:
        raise ValueError("accepted_defect_names must not be empty")
    return len(pd.read_csv(job.defect_manifest)), len(pd.read_csv(job.normal_manifest))


def _validate_output(path: Path, expected_defect: int, expected_normal: int) -> dict:
    frame = pd.read_csv(path)
    required = {"sample_id", "y_true", "score"}
    if required - set(frame.columns):
        raise ValueError(f"Prediction output missing columns: {sorted(required - set(frame.columns))}")
    if frame.sample_id.astype(str).duplicated().any():
        raise ValueError("Prediction output has duplicate sample_id")
    labels = frame.y_true.astype(int)
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError("Prediction output y_true is not binary")
    if not np.isfinite(frame.score.astype(float)).all():
        raise ValueError("Prediction output contains NaN/Inf")
    defect_count = int((labels == 1).sum())
    normal_count = int((labels == 0).sum())
    if defect_count != expected_defect or normal_count != expected_normal:
        raise ValueError(
            f"Prediction row counts differ from manifests: defect={defect_count}/{expected_defect}, "
            f"normal={normal_count}/{expected_normal}"
        )
    return {
        "row_count": int(len(frame)),
        "defect_count": defect_count,
        "normal_count": normal_count,
        "output_sha256": sha256_file(path),
    }


def execute_prediction_job(
    job: PredictionJob,
    *,
    predict_fn: Callable | None = None,
) -> dict:
    """Execute one split and always leave one atomic machine-readable result."""

    job = _resolved(job)
    if job.result_json.exists():
        raise FileExistsError(f"Worker result already exists: {job.result_json}")
    output_preexisting = job.output.exists()
    started = time.time()
    stage = "INPUT_VALIDATION"
    report = {
        "schema_version": "stage1_gapvalue240_prediction_worker_v1",
        "split": job.split,
        "pid": os.getpid(),
        "checkpoint": str(job.checkpoint),
        "dataset_root": str(job.dataset_root),
        "defect_manifest": str(job.defect_manifest),
        "normal_manifest": str(job.normal_manifest),
        "output": str(job.output),
        "result_json": str(job.result_json),
        "yolo_root": str(job.yolo_root),
        "gpu_id": job.gpu_id,
        "batch": job.batch,
        "workers": job.workers,
        "imgsz": job.imgsz,
        "accepted_defect_names": list(job.accepted_defect_names),
        "started_at_unix": started,
    }
    try:
        expected_defect, expected_normal = _validate_inputs(job)
        report.update(
            {
                "checkpoint_sha256": sha256_file(job.checkpoint),
                "defect_manifest_sha256": sha256_file(job.defect_manifest),
                "normal_manifest_sha256": sha256_file(job.normal_manifest),
                "expected_defect_count": expected_defect,
                "expected_normal_count": expected_normal,
            }
        )
        stage = "PREDICTION"
        predictor = predict_fn or predict_split
        predictor(
            checkpoint=job.checkpoint,
            dataset_root=job.dataset_root,
            defect_manifest=job.defect_manifest,
            normal_manifest=job.normal_manifest,
            output=job.output,
            gpu_id=job.gpu_id,
            batch=job.batch,
            workers=job.workers,
            imgsz=job.imgsz,
            accepted_defect_names=list(job.accepted_defect_names),
            yolo_root=job.yolo_root,
        )
        stage = "ARTIFACT_VALIDATION"
        report.update(_validate_output(job.output, expected_defect, expected_normal))
        report.update({"status": "PASS", "exit_code": WORKER_EXIT_CODES["PASS"]})
    except Exception as exc:
        if job.output.exists() and not output_preexisting:
            job.output.unlink()
        if stage == "INPUT_VALIDATION":
            status = "INVALID_INPUT"
        elif stage == "ARTIFACT_VALIDATION":
            status = "ARTIFACT_INVALID"
        else:
            status = "PREDICTION_FAILED"
        report.update(
            {
                "status": status,
                "exit_code": WORKER_EXIT_CODES[status],
                "failed_stage": stage,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    ended = time.time()
    report.update({"ended_at_unix": ended, "duration_seconds": ended - started})
    atomic_write_json(job.result_json, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Stage-1 val_cal or val_op prediction in an isolated process.")
    parser.add_argument("--split", choices=("val_cal", "val_op", "causal_train_probe"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--defect-manifest", type=Path, required=True)
    parser.add_argument("--normal-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--yolo-root", type=Path, required=True)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--accepted-defect-name", action="append", dest="accepted", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = execute_prediction_job(
        PredictionJob(
            split=args.split,
            checkpoint=args.checkpoint,
            dataset_root=args.dataset_root,
            defect_manifest=args.defect_manifest,
            normal_manifest=args.normal_manifest,
            output=args.output,
            result_json=args.result_json,
            yolo_root=args.yolo_root,
            gpu_id=args.gpu_id,
            batch=args.batch,
            workers=args.workers,
            imgsz=args.imgsz,
            accepted_defect_names=tuple(args.accepted),
        )
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
