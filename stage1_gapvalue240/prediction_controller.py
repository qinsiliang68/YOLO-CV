from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .errors import ExternalCommandError
from .subprocesses import run_logged
from .util import atomic_write_json, sha256_file


@dataclass(frozen=True)
class PredictionWorkerSpec:
    split: str
    defect_manifest: Path
    normal_manifest: Path
    output: Path
    result_json: Path
    log_path: Path


def build_prediction_worker_command(
    *,
    spec: PredictionWorkerSpec,
    python_executable: str,
    worker_script: Path,
    checkpoint: Path,
    dataset_root: Path,
    yolo_root: Path,
    gpu_id: str,
    batch: int,
    workers: int,
    imgsz: int,
    accepted_defect_names: Sequence[str],
) -> list[str]:
    command = [
        str(python_executable),
        str(Path(worker_script).resolve()),
        "--split",
        spec.split,
        "--checkpoint",
        str(Path(checkpoint).resolve()),
        "--dataset-root",
        str(Path(dataset_root).resolve()),
        "--defect-manifest",
        str(Path(spec.defect_manifest).resolve()),
        "--normal-manifest",
        str(Path(spec.normal_manifest).resolve()),
        "--output",
        str(Path(spec.output).resolve()),
        "--result-json",
        str(Path(spec.result_json).resolve()),
        "--yolo-root",
        str(Path(yolo_root).resolve()),
        "--gpu-id",
        str(gpu_id),
        "--batch",
        str(batch),
        "--workers",
        str(workers),
        "--imgsz",
        str(imgsz),
    ]
    for name in accepted_defect_names:
        command.extend(["--accepted-defect-name", str(name)])
    return command


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_worker_result(spec: PredictionWorkerSpec, result: dict) -> None:
    if result.get("status") != "PASS" or int(result.get("exit_code", -1)) != 0:
        raise ExternalCommandError(f"Prediction worker did not report PASS: {spec.split}: {result}")
    output = Path(spec.output).resolve()
    if not output.is_file():
        raise ExternalCommandError(f"Prediction worker output is missing: {output}")
    if str(output) != str(Path(result.get("output", "")).resolve()):
        raise ExternalCommandError(f"Prediction worker output path mismatch: {spec.split}")
    if sha256_file(output) != result.get("output_sha256"):
        raise ExternalCommandError(f"Prediction worker output checksum mismatch: {spec.split}")


def run_prediction_workers(
    *,
    specs: Sequence[PredictionWorkerSpec],
    python_executable: str,
    worker_script: Path,
    cwd: Path,
    checkpoint: Path,
    dataset_root: Path,
    yolo_root: Path,
    gpu_id: str,
    batch: int,
    workers: int,
    imgsz: int,
    accepted_defect_names: Sequence[str],
    controller_result_json: Path,
    timeout_seconds: float | None,
    env: dict[str, str] | None = None,
) -> dict:
    """Run val_cal and val_op sequentially in disposable GPU worker processes."""

    if [spec.split for spec in specs] != ["val_cal", "val_op"]:
        raise ValueError("Prediction workers must be ordered exactly as val_cal then val_op")
    controller_result_json = Path(controller_result_json).resolve()
    if controller_result_json.exists():
        raise FileExistsError(f"Controller result already exists: {controller_result_json}")
    started = time.time()
    worker_reports: list[dict] = []
    report = {
        "schema_version": "stage1_gapvalue240_prediction_controller_v1",
        "status": "RUNNING",
        "started_at_unix": started,
        "workers": worker_reports,
        "exit_code_mapping": {
            "0": "PASS",
            "2": "INVALID_INPUT",
            "3": "PREDICTION_FAILED",
            "4": "ARTIFACT_INVALID",
            "controller_timeout": "WORKER_TIMEOUT",
        },
    }
    try:
        for spec in specs:
            command = build_prediction_worker_command(
                spec=spec,
                python_executable=python_executable,
                worker_script=worker_script,
                checkpoint=checkpoint,
                dataset_root=dataset_root,
                yolo_root=yolo_root,
                gpu_id=gpu_id,
                batch=batch,
                workers=workers,
                imgsz=imgsz,
                accepted_defect_names=accepted_defect_names,
            )
            try:
                subprocess_result = run_logged(command, cwd, spec.log_path, env=env, timeout=timeout_seconds)
            except ExternalCommandError:
                subprocess_result = _load_json(Path(spec.log_path).with_suffix(Path(spec.log_path).suffix + ".result.json"))
                worker_result = _load_json(Path(spec.result_json))
                worker_reports.append(
                    {
                        "split": spec.split,
                        "status": "WORKER_TIMEOUT" if subprocess_result and subprocess_result.get("timed_out") else "WORKER_FAILED",
                        "subprocess": subprocess_result,
                        "worker": worker_result,
                    }
                )
                raise
            worker_result = _load_json(Path(spec.result_json))
            if worker_result is None:
                raise ExternalCommandError(f"Prediction worker result is missing: {spec.result_json}")
            _validate_worker_result(spec, worker_result)
            worker_result = dict(worker_result)
            worker_result["subprocess"] = subprocess_result
            worker_reports.append(worker_result)
        report["status"] = "PASS"
    except Exception as exc:
        report.update(
            {
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        ended = time.time()
        report.update({"ended_at_unix": ended, "duration_seconds": ended - started})
        atomic_write_json(controller_result_json, report)
        raise
    ended = time.time()
    report.update({"ended_at_unix": ended, "duration_seconds": ended - started})
    atomic_write_json(controller_result_json, report)
    return report
