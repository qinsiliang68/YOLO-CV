"""Canonical resource preflight and per-epoch resource-log validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable
import uuid

import pandas as pd
import yaml

from .errors import ValidationError
from .util import atomic_write_json, sha256_file


PREFLIGHT_SCHEMA = "stage1.disk_gpu_preflight_validation.v1"
RESOURCE_LOG_SCHEMA = "stage1.resource_log_validation.v1"
RESOURCE_COLUMNS = {
    "epoch",
    "gpu_util_pct",
    "gpu_memory_allocated_bytes",
    "gpu_memory_reserved_bytes",
    "gpu_power_w",
    "cpu_util_pct",
    "rss_bytes",
    "dataloader_wait_seconds",
    "train_compute_seconds",
    "eval_seconds",
    "checkpoint_seconds",
    "write_seconds",
    "queue_idle_seconds",
    "disk_free_bytes",
    "child_process_count",
}


def _default_gpu_probe(executable: str) -> dict[str, Any]:
    command = [
        executable,
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
    if result.returncode != 0:
        raise OSError(result.stdout.strip() or f"{executable} returned {result.returncode}")
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "utilization_gpu_pct": float(parts[2]),
                "memory_used_mib": float(parts[3]),
                "memory_total_mib": float(parts[4]),
                "power_w": float(parts[5]),
            }
        )
    if not rows:
        raise OSError("nvidia-smi returned no GPU rows")
    return {"gpus": rows, "command": command}


def _disk_benchmark(root: Path, bytes_to_write: int) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f".stage1_preflight_{uuid.uuid4().hex}.tmp"
    payload = b"0" * min(bytes_to_write, 1024 * 1024)
    remaining = bytes_to_write
    started = time.perf_counter()
    try:
        with path.open("wb") as handle:
            while remaining > 0:
                chunk = payload[: min(len(payload), remaining)]
                handle.write(chunk)
                remaining -= len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        elapsed = max(time.perf_counter() - started, 1e-9)
        observed = path.stat().st_size
        return {
            "bytes_written": observed,
            "duration_seconds": elapsed,
            "write_mib_per_second": observed / elapsed / (1024**2),
        }
    finally:
        path.unlink(missing_ok=True)


def run_disk_gpu_preflight(
    machine_config_path: str | Path,
    canonical_lock_path: str | Path,
    *,
    output_path: str | Path,
    required_output_free_bytes: int = 20 * 1024**3,
    benchmark_bytes: int = 4 * 1024**2,
    require_gpu: bool = True,
    gpu_probe: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config_path = Path(machine_config_path).resolve()
    lock_path = Path(canonical_lock_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    immutable = dict(lock.get("immutable_args", {}))
    issues: list[str] = []
    if int(immutable.get("batch", -1)) != 128:
        issues.append("canonical batch must remain 128")
    if int(immutable.get("workers", -1)) != 4:
        issues.append("canonical workers must remain 4")
    if int(config.get("num_workers", -1)) != 4:
        issues.append("machine num_workers must remain 4")
    output_root = Path(str(config.get("output_root", ""))).expanduser()
    if not output_root.is_absolute():
        output_root = (config_path.parent / output_root).resolve()
    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        issues.append(f"output root cannot be created: {exc}")
    disk = None
    benchmark = None
    if output_root.exists():
        usage = shutil.disk_usage(output_root)
        disk = {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}
        if usage.free < required_output_free_bytes:
            issues.append(
                f"output disk free bytes {usage.free} below required {required_output_free_bytes}"
            )
        try:
            benchmark = _disk_benchmark(output_root, benchmark_bytes)
        except OSError as exc:
            issues.append(f"disk write benchmark failed: {exc}")
    cpu_ram: dict[str, Any] = {}
    try:
        import psutil

        virtual = psutil.virtual_memory()
        cpu_ram = {
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "cpu_util_pct": psutil.cpu_percent(interval=0.05),
            "ram_total_bytes": virtual.total,
            "ram_available_bytes": virtual.available,
            "process_rss_bytes": psutil.Process(os.getpid()).memory_info().rss,
            "child_process_count": len(psutil.Process(os.getpid()).children(recursive=True)),
        }
    except Exception as exc:
        issues.append(f"CPU/RAM probe failed: {exc}")
    gpu = None
    gpu_status = "PASS"
    probe = gpu_probe or _default_gpu_probe
    try:
        gpu = probe(str(config.get("nvidia_smi_path", "nvidia-smi")))
    except Exception as exc:
        gpu_status = "NOT_RUN_NO_GPU"
        if require_gpu:
            issues.append(f"GPU probe failed: {exc}")
        else:
            gpu = {"error": str(exc)}
    non_gpu_issues = [issue for issue in issues if not issue.startswith("GPU probe failed")]
    if non_gpu_issues:
        status = "FAIL"
    elif gpu_status == "NOT_RUN_NO_GPU":
        status = "NOT_RUN_NO_GPU"
    else:
        status = "PASS"
    report = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": status,
        "created_at_unix": time.time(),
        "issues": issues,
        "machine_id": config.get("machine_id"),
        "canonical_batch": immutable.get("batch"),
        "canonical_workers": immutable.get("workers"),
        "machine_workers": config.get("num_workers"),
        "machine_config_sha256": sha256_file(config_path),
        "canonical_lock_file_sha256": sha256_file(lock_path),
        "required_output_free_bytes": required_output_free_bytes,
        "disk": disk,
        "disk_write_benchmark": benchmark,
        "cpu_ram": cpu_ram,
        "gpu_probe_status": gpu_status,
        "gpu": gpu,
        "automatic_batch_or_worker_tuning": False,
    }
    atomic_write_json(output_path, report, overwrite=True)
    if status == "FAIL":
        raise ValidationError(f"disk/GPU preflight failed; see {output_path}")
    if require_gpu and status != "PASS":
        raise ValidationError(f"disk/GPU preflight requires a real GPU; see {output_path}")
    return report


def validate_resource_log(
    resource_log_path: str | Path,
    *,
    output_path: str | Path,
    expected_epochs: int = 200,
) -> dict[str, Any]:
    path = Path(resource_log_path).resolve()
    frame = pd.read_csv(path, keep_default_na=False)
    missing = RESOURCE_COLUMNS - set(frame.columns)
    issues: list[str] = []
    if missing:
        issues.append(f"resource log missing columns: {sorted(missing)}")
    else:
        epochs = pd.to_numeric(frame.epoch, errors="raise").astype(int).tolist()
        if epochs != list(range(1, expected_epochs + 1)):
            issues.append("resource log must cover every epoch exactly once")
        numeric = RESOURCE_COLUMNS - {"epoch"}
        for column in numeric:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.isna().any() or (values < 0).any():
                issues.append(f"resource log column contains missing/negative values: {column}")
    report = {
        "schema_version": RESOURCE_LOG_SCHEMA,
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "expected_epochs": expected_epochs,
        "row_count": len(frame),
        "resource_log_sha256": sha256_file(path),
    }
    atomic_write_json(output_path, report, overwrite=True)
    if issues:
        raise ValidationError(f"resource log validation failed; see {output_path}")
    return report


def build_epoch_resource_log(
    audit_path: str | Path,
    sampled_resource_log_path: str | Path,
    *,
    output_csv: str | Path,
    validation_output: str | Path,
    expected_epochs: int = 200,
) -> dict[str, Any]:
    """Aggregate low-overhead samples and dynamic-audit timings to one row per epoch."""

    audit_file = Path(audit_path).resolve()
    sample_file = Path(sampled_resource_log_path).resolve()
    try:
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"unreadable dynamic training audit: {audit_file}") from exc
    records = list(audit.get("epoch_records", []))
    if [int(row.get("epoch", -1)) for row in records] != list(range(1, expected_epochs + 1)):
        raise ValidationError("dynamic audit must cover every epoch before resource aggregation")
    samples = pd.read_csv(sample_file, keep_default_na=False)
    required_samples = {
        "timestamp_unix", "gpu_util_pct", "memory_used_mb", "power_w",
        "system_cpu_pct", "process_rss_bytes", "disk_free_bytes",
    }
    missing = required_samples - set(samples.columns)
    if missing:
        raise ValidationError(f"sampled resource log missing columns: {sorted(missing)}")
    samples["timestamp_unix"] = pd.to_numeric(samples.timestamp_unix, errors="coerce")
    rows: list[dict[str, Any]] = []
    for record in records:
        epoch = int(record["epoch"])
        start = float(record.get("epoch_started_at_unix") or 0.0)
        end = float(record.get("epoch_ended_at_unix") or record.get("train_ended_at_unix") or 0.0)
        if start <= 0 or end < start:
            raise ValidationError(f"epoch {epoch} has no valid wall-clock boundary")
        window = samples.loc[(samples.timestamp_unix >= start) & (samples.timestamp_unix <= end)]
        def numeric(column: str) -> pd.Series:
            return pd.to_numeric(window[column], errors="coerce").dropna()
        gpu_util = numeric("gpu_util_pct")
        memory = numeric("memory_used_mb")
        power = numeric("power_w")
        cpu = numeric("system_cpu_pct")
        rss = numeric("process_rss_bytes")
        disk = numeric("disk_free_bytes")
        rows.append({
            "epoch": epoch,
            "gpu_util_pct": float(gpu_util.mean()) if len(gpu_util) else 0.0,
            "gpu_memory_allocated_bytes": int(record.get("cuda_peak_allocated_bytes") or (memory.max() * 1024**2 if len(memory) else 0)),
            "gpu_memory_reserved_bytes": int(record.get("cuda_peak_reserved_bytes") or (memory.max() * 1024**2 if len(memory) else 0)),
            "gpu_power_w": float(power.mean()) if len(power) else 0.0,
            "cpu_util_pct": float(cpu.mean()) if len(cpu) else float(record.get("cpu_util_pct") or 0.0),
            "rss_bytes": int(max(float(record.get("rss_bytes") or 0), float(rss.max()) if len(rss) else 0)),
            "dataloader_wait_seconds": float(record.get("interbatch_wait_seconds") or 0.0),
            "train_compute_seconds": float(record.get("train_compute_seconds") or 0.0),
            "eval_seconds": float(record.get("eval_seconds") or 0.0),
            "checkpoint_seconds": float(record.get("checkpoint_seconds") or 0.0),
            "write_seconds": float(record.get("write_seconds") or 0.0),
            "queue_idle_seconds": float(record.get("queue_idle_seconds") or 0.0),
            "disk_free_bytes": int(disk.min()) if len(disk) else 0,
            "child_process_count": int(record.get("child_process_count") or 0),
        })
    output = Path(output_csv).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{uuid.uuid4().hex}.tmp")
    pd.DataFrame(rows, columns=sorted(RESOURCE_COLUMNS, key=lambda value: (value != "epoch", value))).to_csv(temporary, index=False)
    os.replace(temporary, output)
    return validate_resource_log(output, output_path=validation_output, expected_epochs=expected_epochs)


__all__ = [
    "PREFLIGHT_SCHEMA",
    "RESOURCE_LOG_SCHEMA",
    "RESOURCE_COLUMNS",
    "build_epoch_resource_log",
    "run_disk_gpu_preflight",
    "validate_resource_log",
]
