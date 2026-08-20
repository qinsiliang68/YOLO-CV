from __future__ import annotations

import csv
import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import psutil

from .columnar import partition_identity, write_zstd_parquet
from .errors import ErrorCode, SctsrError
from .ledger_schema import TELEMETRY_SCHEMA

TELEMETRY_CADENCE_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class TelemetryRow:
    timestamp_utc: str
    monotonic_seconds: float
    run_id: str
    arm_id: str
    training_seed: int
    epoch: int
    process_pid: int
    process_cpu_percent: float | None
    process_rss: int | None
    process_vms: int | None
    process_read_bytes: int | None
    process_write_bytes: int | None
    process_read_count: int | None
    process_write_count: int | None
    system_cpu_percent: float | None
    system_memory_total: int | None
    system_memory_available: int | None
    system_memory_used: int | None
    system_memory_percent: float | None
    gpu_index: int | None
    gpu_uuid: str | None
    gpu_name: str | None
    gpu_utilization: float | None
    gpu_memory_used: int | None
    gpu_memory_total: int | None
    gpu_temperature: float | None
    gpu_power: float | None
    cuda_allocated: int | None
    cuda_reserved: int | None
    cuda_max_allocated: int | None
    cuda_max_reserved: int | None
    run_volume_total: int | None
    run_volume_free: int | None
    run_volume_used: int | None
    artifact_volume_total: int | None
    artifact_volume_free: int | None
    artifact_volume_used: int | None
    process_provider_status: str
    process_provider_reason: str
    system_provider_status: str
    system_provider_reason: str
    gpu_provider_status: str
    gpu_provider_reason: str
    cuda_provider_status: str
    cuda_provider_reason: str
    disk_provider_status: str
    disk_provider_reason: str
    telemetry_provider_status: str
    provider_error_code: str | None
    row_generation: int


def _disk(path: str | Path) -> tuple[int, int, int]:
    usage = shutil.disk_usage(Path(path))
    return int(usage.total), int(usage.free), int(usage.used)


def _provider_status(errors: list[str]) -> tuple[str, str]:
    if errors:
        return "FAILED", ";".join(errors)
    return "PASS", "PRESENT"


def _nvidia_smi() -> tuple[dict[str, Any] | None, str]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None, "NVIDIA_SMI_NOT_AVAILABLE"
    query = "index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        completed = subprocess.run(
            [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
            encoding="utf-8",
            errors="replace",
        )
        rows = list(csv.reader(completed.stdout.splitlines(), skipinitialspace=True))
        if not rows or len(rows[0]) != 8:
            return None, "NVIDIA_SMI_MALFORMED_OUTPUT"
        index, uuid, name, utilization, memory_used, memory_total, temperature, power = rows[0]
        return {
            "gpu_index": int(index),
            "gpu_uuid": uuid.strip(),
            "gpu_name": name.strip(),
            "gpu_utilization": float(utilization),
            "gpu_memory_used": int(float(memory_used) * 1024 * 1024),
            "gpu_memory_total": int(float(memory_total) * 1024 * 1024),
            "gpu_temperature": float(temperature),
            "gpu_power": float(power),
        }, "PRESENT"
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return None, f"NVIDIA_SMI_{type(exc).__name__.upper()}"


def sample_telemetry(
    *,
    run_id: str,
    arm_id: str,
    training_seed: int,
    epoch: int,
    run_path: str | Path,
    artifact_path: str | Path,
    row_generation: int = 1,
    process_pid: int | None = None,
) -> TelemetryRow:
    # Timestamp the acquisition boundary, not the completion of the slowest
    # provider.  In particular, nvidia-smi can take hundreds of milliseconds
    # on Windows; recording its completion time turns provider latency into a
    # false cadence gap even when the sampler started exactly on schedule.
    sample_started_at_utc = datetime.now(timezone.utc).isoformat()
    sample_started_monotonic = time.monotonic()
    pid = int(process_pid or os.getpid())
    process_errors: list[str] = []
    system_errors: list[str] = []
    disk_errors: list[str] = []
    try:
        proc = psutil.Process(pid)
        process_cpu = float(proc.cpu_percent(interval=None))
        memory = proc.memory_info()
        rss, vms = int(memory.rss), int(memory.vms)
        io = proc.io_counters()
        read_bytes, write_bytes = int(io.read_bytes), int(io.write_bytes)
        read_count, write_count = int(io.read_count), int(io.write_count)
    except (psutil.Error, AttributeError, OSError) as exc:
        process_errors.append(type(exc).__name__)
        process_cpu = rss = vms = read_bytes = write_bytes = read_count = write_count = None
    try:
        system_cpu = float(psutil.cpu_percent(interval=None))
        virtual_memory = psutil.virtual_memory()
        memory_total = int(virtual_memory.total)
        memory_available = int(virtual_memory.available)
        memory_used = int(virtual_memory.used)
        memory_percent = float(virtual_memory.percent)
    except (psutil.Error, AttributeError, OSError) as exc:
        system_errors.append(type(exc).__name__)
        system_cpu = memory_total = memory_available = memory_used = memory_percent = None
    try:
        run_total, run_free, run_used = _disk(run_path)
        artifact_total, artifact_free, artifact_used = _disk(artifact_path)
    except OSError as exc:
        disk_errors.append(type(exc).__name__)
        run_total = run_free = run_used = artifact_total = artifact_free = artifact_used = None

    gpu, gpu_reason = _nvidia_smi()
    if gpu is None:
        gpu = {
            "gpu_index": None,
            "gpu_uuid": None,
            "gpu_name": None,
            "gpu_utilization": None,
            "gpu_memory_used": None,
            "gpu_memory_total": None,
            "gpu_temperature": None,
            "gpu_power": None,
        }
        gpu_status = "NOT_AVAILABLE"
    else:
        gpu_status = "PASS"

    cuda_reason = "NO_CUDA_DEVICE"
    cuda_status = "NOT_AVAILABLE"
    cuda_allocated = cuda_reserved = cuda_max_allocated = cuda_max_reserved = None
    try:
        import torch

        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            cuda_allocated = int(torch.cuda.memory_allocated(index))
            cuda_reserved = int(torch.cuda.memory_reserved(index))
            cuda_max_allocated = int(torch.cuda.max_memory_allocated(index))
            cuda_max_reserved = int(torch.cuda.max_memory_reserved(index))
            cuda_status, cuda_reason = "PASS", "PRESENT"
    except Exception as exc:  # hardware/provider errors are evidence, not fake zeroes
        cuda_status, cuda_reason = "FAILED", f"TORCH_CUDA_{type(exc).__name__.upper()}"

    process_status, process_reason = _provider_status(process_errors)
    system_status, system_reason = _provider_status(system_errors)
    disk_status, disk_reason = _provider_status(disk_errors)
    critical_pass = process_status == system_status == disk_status == "PASS"
    aggregate_status = "PASS" if critical_pass and gpu_status == cuda_status == "PASS" else "PASS_WITH_REGISTERED_NULLS" if critical_pass else "FAILED"
    provider_errors = []
    for provider, status, reason in (
        ("process", process_status, process_reason),
        ("system", system_status, system_reason),
        ("gpu", gpu_status, gpu_reason),
        ("cuda", cuda_status, cuda_reason),
        ("disk", disk_status, disk_reason),
    ):
        if status != "PASS":
            provider_errors.append(f"{provider}:{reason}")

    return TelemetryRow(
        timestamp_utc=sample_started_at_utc,
        monotonic_seconds=sample_started_monotonic,
        run_id=run_id,
        arm_id=arm_id,
        training_seed=training_seed,
        epoch=epoch,
        process_pid=pid,
        process_cpu_percent=process_cpu,
        process_rss=rss,
        process_vms=vms,
        process_read_bytes=read_bytes,
        process_write_bytes=write_bytes,
        process_read_count=read_count,
        process_write_count=write_count,
        system_cpu_percent=system_cpu,
        system_memory_total=memory_total,
        system_memory_available=memory_available,
        system_memory_used=memory_used,
        system_memory_percent=memory_percent,
        **gpu,
        cuda_allocated=cuda_allocated,
        cuda_reserved=cuda_reserved,
        cuda_max_allocated=cuda_max_allocated,
        cuda_max_reserved=cuda_max_reserved,
        run_volume_total=run_total,
        run_volume_free=run_free,
        run_volume_used=run_used,
        artifact_volume_total=artifact_total,
        artifact_volume_free=artifact_free,
        artifact_volume_used=artifact_used,
        process_provider_status=process_status,
        process_provider_reason=process_reason,
        system_provider_status=system_status,
        system_provider_reason=system_reason,
        gpu_provider_status=gpu_status,
        gpu_provider_reason=gpu_reason,
        cuda_provider_status=cuda_status,
        cuda_provider_reason=cuda_reason,
        disk_provider_status=disk_status,
        disk_provider_reason=disk_reason,
        telemetry_provider_status=aggregate_status,
        provider_error_code=";".join(provider_errors) if provider_errors else None,
        row_generation=row_generation,
    )


def validate_telemetry_for_closeout(
    rows: Sequence[TelemetryRow],
    *,
    cadence_seconds: float = TELEMETRY_CADENCE_SECONDS,
    tolerance: float = 0.35,
) -> None:
    if not rows:
        raise SctsrError(ErrorCode.TELEMETRY_UNAVAILABLE, "No telemetry rows are available")
    identity = (rows[0].run_id, rows[0].arm_id, rows[0].training_seed, rows[0].epoch)
    for row in rows:
        if (row.run_id, row.arm_id, row.training_seed, row.epoch) != identity:
            raise SctsrError(ErrorCode.TELEMETRY_UNAVAILABLE, "Telemetry partition mixes run/arm/seed/epoch identities")
        critical = (row.process_rss, row.system_memory_total, row.run_volume_free, row.artifact_volume_free)
        if all(value is None for value in critical) or row.telemetry_provider_status == "FAILED":
            raise SctsrError(ErrorCode.TELEMETRY_UNAVAILABLE, "Critical telemetry providers are unavailable")
        if any(value is not None and value <= 0 for value in critical):
            raise SctsrError(ErrorCode.TELEMETRY_UNAVAILABLE, "Critical telemetry may not use fake zero or negative values")
        for provider, fields in (
            ("process", (row.process_cpu_percent, row.process_rss, row.process_vms)),
            ("system", (row.system_cpu_percent, row.system_memory_total)),
            ("gpu", (row.gpu_index, row.gpu_uuid, row.gpu_name, row.gpu_utilization, row.gpu_memory_used, row.gpu_memory_total, row.gpu_temperature, row.gpu_power)),
            ("cuda", (row.cuda_allocated, row.cuda_reserved, row.cuda_max_allocated, row.cuda_max_reserved)),
            ("disk", (row.run_volume_total, row.run_volume_free, row.artifact_volume_total, row.artifact_volume_free)),
        ):
            status = getattr(row, f"{provider}_provider_status")
            reason = getattr(row, f"{provider}_provider_reason")
            if status not in {"PASS", "FAILED", "NOT_AVAILABLE"} or not reason or reason in {"unknown", "UNKNOWN"}:
                raise SctsrError(ErrorCode.TELEMETRY_UNAVAILABLE, "Telemetry provider status/reason is invalid", failing_field=provider)
            if status == "NOT_AVAILABLE" and any(value is not None for value in fields):
                raise SctsrError(ErrorCode.TELEMETRY_UNAVAILABLE, "Unavailable telemetry provider populated fake values", failing_field=provider)
        if row.row_generation < 1:
            raise SctsrError(ErrorCode.TELEMETRY_UNAVAILABLE, "Telemetry row lacks transaction generation")
    ordered = sorted(rows, key=lambda item: item.monotonic_seconds)
    if [item.monotonic_seconds for item in rows] != [item.monotonic_seconds for item in ordered]:
        raise SctsrError(ErrorCode.TELEMETRY_UNAVAILABLE, "Telemetry rows are not monotonic")
    for left, right in zip(ordered, ordered[1:]):
        observed = right.monotonic_seconds - left.monotonic_seconds
        if abs(observed - cadence_seconds) > tolerance:
            raise SctsrError(ErrorCode.TELEMETRY_UNAVAILABLE, "Telemetry cadence is outside tolerance", observed=observed, expected=cadence_seconds)


def write_telemetry_partition(rows: Sequence[TelemetryRow], path: str | Path):
    validate_telemetry_for_closeout(rows)
    run_id, epoch = partition_identity(path, required=True)
    if any(row.run_id != run_id or row.epoch != epoch for row in rows):
        raise SctsrError(ErrorCode.SCHEMA_VALIDATION_FAILED, "Telemetry row identity differs from its run/epoch partition")
    return write_zstd_parquet(
        [asdict(row) for row in rows],
        path,
        schema_version="stage1.sctsr.resource_telemetry.v1",
        schema=TELEMETRY_SCHEMA,
        require_run_epoch_partition=True,
    )


class TelemetrySampler:
    """One-second wall-clock sampler whose rows remain bound to one epoch."""

    def __init__(
        self,
        *,
        run_id: str,
        arm_id: str,
        training_seed: int,
        epoch: int,
        run_path: str | Path,
        artifact_path: str | Path,
        row_generation: int,
        cadence_seconds: float = TELEMETRY_CADENCE_SECONDS,
        sample_function: Callable[..., TelemetryRow] = sample_telemetry,
    ) -> None:
        if cadence_seconds != TELEMETRY_CADENCE_SECONDS:
            raise SctsrError(ErrorCode.TELEMETRY_UNAVAILABLE, "Formal telemetry cadence is frozen at exactly one second")
        self.kwargs = {
            "run_id": run_id,
            "arm_id": arm_id,
            "training_seed": training_seed,
            "epoch": epoch,
            "run_path": run_path,
            "artifact_path": artifact_path,
            "row_generation": row_generation,
        }
        self.cadence_seconds = cadence_seconds
        self.sample_function = sample_function
        self.rows: list[TelemetryRow] = []
        self.errors: list[BaseException] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        # Anchor the cadence to the first completed sample.  Some providers
        # (notably the first nvidia-smi probe on Windows) can take longer than
        # one second to warm up.  Anchoring before that call leaves ``target``
        # in the past and the old loop immediately emitted a catch-up sample,
        # creating a sub-second interval that closeout correctly rejected.
        # Re-anchor after any missed target so telemetry never catches up by
        # producing an artificial burst of closely spaced rows.
        target: float | None = None
        while not self._stop.is_set():
            try:
                row = self.sample_function(**self.kwargs)
                self.rows.append(row)
            except BaseException as exc:
                self.errors.append(exc)
                self._stop.set()
                return
            if target is None:
                target = row.monotonic_seconds + self.cadence_seconds
            else:
                target += self.cadence_seconds
                if target <= row.monotonic_seconds:
                    target = row.monotonic_seconds + self.cadence_seconds
            self._stop.wait(max(0.0, target - time.monotonic()))

    def start(self) -> "TelemetrySampler":
        if self._thread is not None:
            raise SctsrError(ErrorCode.TELEMETRY_UNAVAILABLE, "Telemetry sampler was already started")
        self._thread = threading.Thread(target=self._run, name=f"sctsr-telemetry-{self.kwargs['run_id']}-{self.kwargs['epoch']}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> tuple[TelemetryRow, ...]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise SctsrError(ErrorCode.TELEMETRY_UNAVAILABLE, "Telemetry sampler did not stop cleanly")
        if self.errors:
            raise SctsrError(ErrorCode.TELEMETRY_UNAVAILABLE, "Telemetry sampler failed", observed=type(self.errors[0]).__name__)
        return tuple(self.rows)

    def __enter__(self) -> "TelemetrySampler":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False
