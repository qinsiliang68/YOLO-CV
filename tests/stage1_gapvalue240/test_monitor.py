from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import time

import pandas as pd

from stage1_gapvalue240.monitor import ResourceMonitor


def test_resource_monitor_records_gpu_cpu_ram_disk_and_runtime_phase(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="91, 7000, 8192, 68, 175.5\n", stderr="")

    monkeypatch.setattr("stage1_gapvalue240.monitor.subprocess.run", fake_run)
    path = tmp_path / "resource.csv"
    monitor = ResourceMonitor(
        path,
        gpu_id=0,
        interval=0.01,
        process_pid=__import__("os").getpid(),
        disk_path=tmp_path,
    )
    monitor.set_phase("STAGING")
    monitor.start()
    time.sleep(0.03)
    monitor.set_phase("TRAIN_COMPUTE")
    time.sleep(0.03)
    monitor.stop()

    frame = pd.read_csv(path)
    required = {
        "timestamp_unix",
        "runtime_phase",
        "gpu_util_pct",
        "memory_used_mb",
        "memory_total_mb",
        "temperature_c",
        "power_w",
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
    }
    assert required <= set(frame.columns)
    assert len(frame) >= 2
    assert set(frame.runtime_phase) <= {"STAGING", "TRAIN_COMPUTE"}
    assert frame.gpu_util_pct.eq(91).all()
    assert frame.process_rss_bytes.gt(0).all()
    assert frame.system_ram_available_bytes.gt(0).all()
    assert frame.disk_free_bytes.gt(0).all()
    assert frame.disk_read_bytes_delta.fillna(0).ge(0).all()
    assert frame.disk_write_bytes_delta.fillna(0).ge(0).all()


def test_resource_monitor_survives_missing_gpu_tool_and_stop_is_idempotent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def unavailable(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr("stage1_gapvalue240.monitor.subprocess.run", unavailable)
    path = tmp_path / "resource.csv"
    monitor = ResourceMonitor(path, gpu_id=0, interval=0.01, disk_path=tmp_path)
    monitor.start()
    time.sleep(0.02)
    monitor.stop()
    monitor.stop()

    frame = pd.read_csv(path)
    assert not frame.empty
    assert frame.status.astype(str).str.startswith("GPU_UNAVAILABLE:").all()
    assert frame.system_cpu_pct.notna().all()
