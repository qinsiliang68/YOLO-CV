from __future__ import annotations

import csv
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any

import psutil


RESOURCE_COLUMNS = (
    "timestamp_unix",
    "interval_seconds",
    "runtime_phase",
    "gpu_util_pct",
    "memory_used_mb",
    "memory_total_mb",
    "temperature_c",
    "power_w",
    "process_pid",
    "process_rss_bytes",
    "process_cpu_pct",
    "system_cpu_pct",
    "system_ram_used_bytes",
    "system_ram_available_bytes",
    "disk_read_bytes",
    "disk_write_bytes",
    "disk_read_bytes_delta",
    "disk_write_bytes_delta",
    "disk_free_bytes",
    "status",
)


class ResourceMonitor:
    """Low-overhead sidecar monitor for GPU, process, host, and disk state."""

    def __init__(
        self,
        path: Path,
        gpu_id: str | int,
        nvidia_smi: str = "nvidia-smi",
        interval: float = 10.0,
        *,
        process_pid: int | None = None,
        disk_path: str | Path | None = None,
    ) -> None:
        if interval <= 0:
            raise ValueError("resource monitor interval must be positive")
        self.path = Path(path).resolve()
        self.gpu_id = str(gpu_id)
        self.nvidia_smi = str(nvidia_smi)
        self.interval = float(interval)
        self.process_pid = int(process_pid if process_pid is not None else os.getpid())
        self.disk_path = Path(disk_path or self.path.parent).resolve()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._phase = "INITIALIZING"
        self._phase_lock = threading.Lock()

    def set_phase(self, phase: str) -> None:
        value = str(phase).strip()
        if not value:
            raise ValueError("runtime phase must not be empty")
        with self._phase_lock:
            self._phase = value

    def _current_phase(self) -> str:
        with self._phase_lock:
            return self._phase

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("resource monitor is already running")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.disk_path.mkdir(parents=True, exist_ok=True)
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run,
            name=f"resource-monitor-{self.gpu_id}",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=self.interval + 6.0)

    def _gpu_values(self) -> tuple[list[str | float | None], str]:
        query = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
        try:
            result = subprocess.run(
                [
                    self.nvidia_smi,
                    f"--id={self.gpu_id}",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            values = [value.strip() for value in result.stdout.strip().split(",")]
            if result.returncode != 0 or len(values) != 5:
                detail = (result.stderr or result.stdout or f"returncode={result.returncode}").strip()
                return [None] * 5, f"GPU_ERROR:{detail}"
            return values, "OK"
        except Exception as exc:
            return [None] * 5, f"GPU_UNAVAILABLE:{type(exc).__name__}:{exc}"

    @staticmethod
    def _disk_counters() -> tuple[int, int]:
        counters = psutil.disk_io_counters()
        if counters is None:
            return 0, 0
        return int(counters.read_bytes), int(counters.write_bytes)

    def _host_values(self, process: psutil.Process) -> dict[str, Any]:
        memory = psutil.virtual_memory()
        read_bytes, write_bytes = self._disk_counters()
        try:
            process_rss = int(process.memory_info().rss)
            process_cpu = float(process.cpu_percent(interval=None))
        except psutil.Error:
            process_rss = 0
            process_cpu = 0.0
        return {
            "process_rss_bytes": process_rss,
            "process_cpu_pct": process_cpu,
            "system_cpu_pct": float(psutil.cpu_percent(interval=None)),
            "system_ram_used_bytes": int(memory.used),
            "system_ram_available_bytes": int(memory.available),
            "disk_read_bytes": read_bytes,
            "disk_write_bytes": write_bytes,
            "disk_free_bytes": int(shutil.disk_usage(self.disk_path).free),
        }

    def _run(self) -> None:
        process = psutil.Process(self.process_pid)
        process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        previous_read, previous_write = self._disk_counters()
        previous_time = time.time()
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=RESOURCE_COLUMNS)
            writer.writeheader()
            while not self.stop_event.is_set():
                now = time.time()
                gpu, status = self._gpu_values()
                try:
                    host = self._host_values(process)
                    read_delta = max(0, int(host["disk_read_bytes"]) - previous_read)
                    write_delta = max(0, int(host["disk_write_bytes"]) - previous_write)
                    previous_read = int(host["disk_read_bytes"])
                    previous_write = int(host["disk_write_bytes"])
                except Exception as exc:
                    host = {
                        "process_rss_bytes": None,
                        "process_cpu_pct": None,
                        "system_cpu_pct": None,
                        "system_ram_used_bytes": None,
                        "system_ram_available_bytes": None,
                        "disk_read_bytes": previous_read,
                        "disk_write_bytes": previous_write,
                        "disk_free_bytes": None,
                    }
                    read_delta = 0
                    write_delta = 0
                    status = f"{status};HOST_UNAVAILABLE:{type(exc).__name__}:{exc}"
                writer.writerow(
                    {
                        "timestamp_unix": now,
                        "interval_seconds": max(0.0, now - previous_time),
                        "runtime_phase": self._current_phase(),
                        "gpu_util_pct": gpu[0],
                        "memory_used_mb": gpu[1],
                        "memory_total_mb": gpu[2],
                        "temperature_c": gpu[3],
                        "power_w": gpu[4],
                        "process_pid": self.process_pid,
                        **host,
                        "disk_read_bytes_delta": read_delta,
                        "disk_write_bytes_delta": write_delta,
                        "status": status,
                    }
                )
                stream.flush()
                previous_time = now
                self.stop_event.wait(self.interval)


__all__ = ["RESOURCE_COLUMNS", "ResourceMonitor"]
