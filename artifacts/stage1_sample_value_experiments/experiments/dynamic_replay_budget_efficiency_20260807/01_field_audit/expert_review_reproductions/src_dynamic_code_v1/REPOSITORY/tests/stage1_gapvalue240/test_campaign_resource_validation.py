from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from stage1_gapvalue240.campaign_resource_validation import run_disk_gpu_preflight, validate_resource_log
from stage1_gapvalue240.errors import ValidationError


def _inputs(tmp_path: Path, *, workers: int = 4) -> tuple[Path, Path]:
    output = tmp_path / "out"
    config = tmp_path / "machine.yaml"
    config.write_text(yaml.safe_dump({
        "machine_id": "M01", "output_root": str(output), "num_workers": workers,
        "nvidia_smi_path": "nvidia-smi",
    }), encoding="utf-8")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"immutable_args": {"batch": 128, "workers": 4}}), encoding="utf-8")
    return config, lock


def test_preflight_passes_with_canonical_resources_and_injected_gpu(tmp_path: Path) -> None:
    config, lock = _inputs(tmp_path)
    report = run_disk_gpu_preflight(
        config, lock, output_path=tmp_path / "report.json", required_output_free_bytes=1,
        benchmark_bytes=1024, gpu_probe=lambda _exe: {"gpus": [{"index": 0}]},
    )
    assert report["status"] == "PASS"
    assert report["automatic_batch_or_worker_tuning"] is False


def test_preflight_reports_not_run_when_gpu_is_unavailable_but_optional(tmp_path: Path) -> None:
    config, lock = _inputs(tmp_path)

    def missing_gpu(_exe: str):
        raise OSError("nvidia-smi unavailable")

    report = run_disk_gpu_preflight(
        config, lock, output_path=tmp_path / "report.json", required_output_free_bytes=1,
        benchmark_bytes=1024, require_gpu=False, gpu_probe=missing_gpu,
    )
    assert report["status"] == "NOT_RUN_NO_GPU"
    assert report["gpu_probe_status"] == "NOT_RUN_NO_GPU"


def test_preflight_rejects_worker_drift(tmp_path: Path) -> None:
    config, lock = _inputs(tmp_path, workers=2)
    with pytest.raises(ValidationError, match="preflight"):
        run_disk_gpu_preflight(
            config, lock, output_path=tmp_path / "report.json", required_output_free_bytes=1,
            benchmark_bytes=1024, gpu_probe=lambda _exe: {"gpus": [{"index": 0}]},
        )


def test_resource_log_requires_all_epochs_and_fields(tmp_path: Path) -> None:
    rows = []
    for epoch in range(1, 4):
        rows.append({
            "epoch": epoch, "gpu_util_pct": 50, "gpu_memory_allocated_bytes": 1,
            "gpu_memory_reserved_bytes": 2, "gpu_power_w": 100, "cpu_util_pct": 20,
            "rss_bytes": 1000, "dataloader_wait_seconds": 1, "train_compute_seconds": 2,
            "eval_seconds": 0, "checkpoint_seconds": 0, "write_seconds": 0,
            "queue_idle_seconds": 0, "disk_free_bytes": 100000, "child_process_count": 0,
        })
    path = tmp_path / "resources.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    report = validate_resource_log(path, output_path=tmp_path / "out.json", expected_epochs=3)
    assert report["status"] == "PASS"


def test_epoch_resource_builder_aggregates_exact_epoch_windows(tmp_path: Path) -> None:
    from stage1_gapvalue240.campaign_resource_validation import build_epoch_resource_log
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "epoch_records": [
            {"epoch": 1, "epoch_started_at_unix": 10.0, "train_ended_at_unix": 11.0, "epoch_ended_at_unix": 12.0,
             "interbatch_wait_seconds": .2, "train_compute_seconds": .7, "eval_seconds": .1,
             "checkpoint_seconds": .05, "write_seconds": .01, "queue_idle_seconds": 0,
             "cuda_peak_allocated_bytes": 100, "cuda_peak_reserved_bytes": 120, "rss_bytes": 200,
             "cpu_util_pct": 30, "child_process_count": 4},
            {"epoch": 2, "epoch_started_at_unix": 20.0, "train_ended_at_unix": 21.0, "epoch_ended_at_unix": 22.0,
             "interbatch_wait_seconds": .3, "train_compute_seconds": .6, "eval_seconds": .2,
             "checkpoint_seconds": .04, "write_seconds": .02, "queue_idle_seconds": 0,
             "cuda_peak_allocated_bytes": 110, "cuda_peak_reserved_bytes": 130, "rss_bytes": 210,
             "cpu_util_pct": 31, "child_process_count": 4},
        ]
    }), encoding="utf-8")
    samples = tmp_path / "samples.csv"
    pd.DataFrame([
        {"timestamp_unix":10.5,"gpu_util_pct":80,"memory_used_mb":1,"power_w":100,"system_cpu_pct":20,"process_rss_bytes":180,"disk_free_bytes":1000},
        {"timestamp_unix":11.5,"gpu_util_pct":90,"memory_used_mb":1,"power_w":110,"system_cpu_pct":30,"process_rss_bytes":190,"disk_free_bytes":900},
        {"timestamp_unix":20.5,"gpu_util_pct":70,"memory_used_mb":1,"power_w":90,"system_cpu_pct":25,"process_rss_bytes":200,"disk_free_bytes":800},
    ]).to_csv(samples,index=False)
    report=build_epoch_resource_log(audit,samples,output_csv=tmp_path/'epoch.csv',validation_output=tmp_path/'validation.json',expected_epochs=2)
    assert report['status']=='PASS'
    frame=pd.read_csv(tmp_path/'epoch.csv')
    assert frame.epoch.tolist()==[1,2]
    assert frame.loc[0,'gpu_util_pct']==85
