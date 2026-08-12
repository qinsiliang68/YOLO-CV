from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.resource_reliability import (
    ResourceAuditError,
    analyze_canonical_resources,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_run(
    tmp_path: Path,
    *,
    slot: str,
    arm: str,
    machine: str,
    resume_count: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    attempt_id = f"attempt_{slot}_{arm}"
    attempt = tmp_path / "extract" / "P01" / "runs" / slot / attempt_id
    snapshot = "SNAPSHOT_A"
    segments = [
        {
            "start_epoch": 1,
            "end_epoch": 200 if resume_count == 0 else 100,
            "started_at": 1000.0,
            "ended_at": 1100.0 if resume_count == 0 else None,
            "status": "COMPLETED" if resume_count == 0 else "INTERRUPTED",
            "resumed": False,
        }
    ]
    if resume_count:
        segments.append(
            {
                "start_epoch": 101,
                "end_epoch": 200,
                "started_at": 1200.0,
                "ended_at": 1300.0,
                "status": "COMPLETED",
                "resumed": True,
            }
        )

    _write_json(
        attempt / "00_identity/environment_controller.json",
        {"pid": 10, "platform": "Windows-test", "python": "3.11"},
    )
    _write_json(
        attempt / "00_identity/environment_training.json",
        {
            "actual": {
                "python": "3.11",
                "pytorch": "2.11",
                "cuda_build": "12.8",
                "ultralytics": "8.4.66",
                "numpy": "1.26",
                "pandas": "3.0",
                "polars": "1.41",
                "scikit_learn": "1.8",
            },
            "checks": {"python": {"ok": True}, "pytorch": {"ok": True}},
            "status": "PASS",
        },
    )
    _write_json(
        attempt / "00_identity/run_identity.json",
        {
            "run_slot": slot,
            "attempt_id": attempt_id.removeprefix("attempt_"),
            "machine_id": machine,
            "input_snapshot_id": snapshot,
            "resume_count": resume_count,
            "resume_mode": "native_approximate",
            "last_epoch": 200,
            "resume_segments": segments,
        },
    )
    _write_json(
        attempt / "02_logs/training_execution_audit.json",
        {
            "completed_epochs": 200,
            "expected_epochs": 200,
            "expected_steps_per_epoch": 10,
            "optimizer_steps_total": 2000,
            "effective_batch_size": 128,
            "loss_finite": True,
            "resume_count": resume_count,
            "resume_mode": "native_approximate",
            "resume_segments": segments,
        },
    )
    _write_json(
        attempt / "07_validation/storage_preflight.json",
        {
            "status": "PASS",
            "dataset_volume": "windows:c:",
            "staging_volume": "windows:c:",
            "expected_staging_files": 100,
            "maximum_staging_files": 200,
            "minimum_output_free_bytes": 1000,
            "minimum_staging_free_bytes": 500,
            "output_free_bytes": 5000,
            "staging_free_bytes": 3000,
        },
    )
    _write_json(
        attempt / "07_validation/preflight_report.json",
        {
            "status": "PASS",
            "issues": [],
            "replay_manifest_summary": {"epoch_samples": 120600},
        },
    )
    _write_json(
        attempt / "08_status/status.json",
        {
            "run_slot": slot,
            "state": "VALIDATED",
            "phase": "complete",
            "resume_count": resume_count,
            "last_epoch": 200,
        },
    )
    pd.DataFrame(
        {
            "epoch": range(1, 201),
            "time": [float(i) for i in range(1, 201)],
        }
    ).to_csv(attempt / "02_logs/epoch_training_metrics.csv", index=False)

    gpu_logs = 1 + resume_count
    for index in range(gpu_logs):
        pd.DataFrame(
            {
                "timestamp_unix": [1000.0 + 200 * index, 1100.0 + 200 * index],
                "gpu_util_pct": [50, 90],
                "memory_used_mb": [1000, 2000 + index],
                "memory_total_mb": [24576, 24576],
                "temperature_c": [50, 70],
                "power_w": [100, 200],
                "status": ["OK", "OK"],
            }
        ).to_csv(
            attempt / f"02_logs/gpu_usage_20260701T00000{index}_abc.csv",
            index=False,
        )
    _write_json(
        attempt / "02_logs/train_20260701T000000_abc.log.result.json",
        {
            "status": "PASS",
            "returncode": 0,
            "duration_seconds": 100.0,
            "started_at_unix": 1000.0,
            "ended_at_unix": 1100.0,
            "timed_out": False,
        },
    )

    inventory = {
        "run_slot": slot,
        "triad_id": "TRIAD_001",
        "phase": "A",
        "condition_id": "A01_Test",
        "method": "Test",
        "budget": 600,
        "guard_ratio": 0.0,
        "arm": arm,
        "training_seed": 11,
        "selection_seed": 20 + len(slot),
        "package": "P01",
        "machine_id": machine,
        "attempt_id": attempt_id,
        "resume_count": resume_count,
        "selection_sha256": "A" * 64,
        "release_ref": "tag",
        "release_commit": "b" * 40,
        "input_snapshot_id": snapshot,
    }
    ledger = []
    for path in sorted(attempt.rglob("*")):
        if path.is_file():
            ledger.append(
                {
                    "run_slot": slot,
                    "package": "P01",
                    "attempt_id": attempt_id,
                    "attempt_dir": str(attempt),
                    "relative_path": path.relative_to(attempt).as_posix(),
                    "canonical_attempt": True,
                    "artifact_manifest_size_match": True,
                }
            )
    return inventory, ledger


def test_resource_audit_summarizes_gpu_resume_runtime_and_machine_pairing(
    tmp_path: Path,
) -> None:
    inventory_rows = []
    ledger_rows = []
    for slot, arm, machine, resumes in (
        ("RUN_001", "T", "machine_01", 0),
        ("RUN_002", "R1", "machine_01", 1),
        ("RUN_003", "R2", "machine_02", 0),
    ):
        inventory, ledger = _make_run(
            tmp_path,
            slot=slot,
            arm=arm,
            machine=machine,
            resume_count=resumes,
        )
        inventory_rows.append(inventory)
        ledger_rows.extend(ledger)
    inventory_path = tmp_path / "inventory.csv"
    ledger_path = tmp_path / "source_ledger.csv"
    pd.DataFrame(inventory_rows).to_csv(inventory_path, index=False)
    pd.DataFrame(ledger_rows).to_csv(ledger_path, index=False)

    result = analyze_canonical_resources(
        inventory_path,
        ledger_path,
        expected_runs=3,
        expected_triads=1,
        max_workers=2,
    )

    assert len(result.runs) == 3
    assert len(result.machines) == 2
    assert len(result.triads) == 1
    assert result.summary["canonical_runs"] == 3
    assert result.summary["resumed_runs"] == 1
    assert result.summary["input_snapshots"] == 1
    assert result.summary["cross_machine_triads"] == 1
    resumed = result.runs.set_index("run_slot").loc["RUN_002"]
    assert resumed["gpu_log_count"] == 2
    assert resumed["gpu_memory_peak_mb"] == 2001
    assert resumed["gpu_monitor_seconds"] == 200.0
    assert json.loads(resumed["gpu_status_counts_json"]) == {"OK": 4}
    assert resumed["resume_interrupted_segments"] == 1
    assert resumed["resume_completed_segments"] == 1
    assert resumed["resume_missing_end_timestamp_segments"] == 1
    assert resumed["resume_resumed_segments"] == 1
    assert resumed["identity_consistent"]
    assert resumed["validated_complete"]
    assert resumed["training_samples_per_second"] > 0
    triad = result.triads.iloc[0]
    assert triad["all_arms_same_machine"] == False  # noqa: E712
    assert triad["t_r1_same_machine"] == True  # noqa: E712
    assert triad["t_r2_same_machine"] == False  # noqa: E712


def test_resource_audit_rejects_noncanonical_or_missing_required_evidence(
    tmp_path: Path,
) -> None:
    inventory, ledger = _make_run(
        tmp_path,
        slot="RUN_001",
        arm="T",
        machine="machine_01",
        resume_count=0,
    )
    inventory_path = tmp_path / "inventory.csv"
    ledger_path = tmp_path / "source_ledger.csv"
    pd.DataFrame([inventory]).to_csv(inventory_path, index=False)
    broken = pd.DataFrame(ledger)
    broken.loc[
        broken["relative_path"] == "08_status/status.json", "canonical_attempt"
    ] = False
    broken.to_csv(ledger_path, index=False)

    with pytest.raises(ResourceAuditError, match="non-canonical ledger row"):
        analyze_canonical_resources(
            inventory_path,
            ledger_path,
            expected_runs=1,
            expected_triads=1,
        )


def test_real_canonical_240_resource_audit() -> None:
    inventory_path = Path(
        r"C:\baidunetdiskdownload\stage1_gapvalue240_extract_audit_20260728"
        r"\GLOBAL_VALIDATED_RUN_INVENTORY.csv"
    )
    ledger_path = Path(
        r"artifacts\stage1_sample_value_experiments\experiments"
        r"\oof_dynamics_gap_value_20260708\06_reports"
        r"\gapvalue240_goal_analysis_20260806_v1.inprogress\audit"
        r"\source_file_ledger.csv"
    )
    if not inventory_path.is_file() or not ledger_path.is_file():
        pytest.skip("Canonical 240-run source ledger is not mounted")

    result = analyze_canonical_resources(
        inventory_path,
        ledger_path,
        expected_runs=240,
        expected_triads=80,
        max_workers=16,
    )

    assert len(result.runs) == 240
    assert len(result.triads) == 80
    assert result.summary["resumed_runs"] == 15
    assert result.summary["input_snapshots"] == 2
    assert result.summary["cross_machine_triads"] == 24
    assert result.runs["validated_complete"].all()
    assert result.runs["identity_consistent"].all()
    # These are preserved monitor gaps, not silently converted to zero-utilization.
    # All 68 are five-second nvidia-smi query timeouts in six otherwise validated runs.
    assert result.runs["gpu_non_ok_samples"].sum() == 68
    assert (result.runs["gpu_non_ok_samples"] > 0).sum() == 6
