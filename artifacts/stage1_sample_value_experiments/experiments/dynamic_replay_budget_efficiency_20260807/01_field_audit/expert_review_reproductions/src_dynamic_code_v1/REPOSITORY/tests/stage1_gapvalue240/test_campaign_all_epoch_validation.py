from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.campaign_all_epoch_validation import validate_all_epoch_telemetry
from stage1_gapvalue240.errors import ValidationError
from stage1_gapvalue240.util import atomic_write_json, sha256_file


def _fixture(tmp_path: Path, *, epochs: int = 4, replay: int = 2) -> tuple[Path, Path, Path]:
    telemetry = tmp_path / "telemetry"
    telemetry.mkdir()
    schedule_rows = []
    records = []
    for epoch in range(1, epochs + 1):
        schedule_rows.append({
            "epoch": epoch,
            "segment_id": "S1" if epoch <= 2 else "S2",
            "base_sample_exposures": 10,
            "normal_replay_exposures": replay,
            "defect_guard_exposures": 0,
            "total_sample_exposures": 10 + replay,
            "expected_optimizer_steps": 3,
        })
        records.append({"epoch": epoch, "optimizer_steps_epoch": 3})
        parquet = telemetry / f"epoch_{epoch:04d}_process_telemetry.parquet"
        parquet.write_bytes(f"parquet-{epoch}".encode())
        role = telemetry / f"epoch_{epoch:04d}_role_loss_summary.json"
        roles = [] if replay == 0 else [{
            "record_type": "ROLE", "exposure_role": "normal_replay",
            "total_exposure_count": replay, "loss_mean": 0.2,
        }]
        atomic_write_json(role, {
            "schema_version": "stage1.process_role_loss_summary.v1", "status": "COMPLETE",
            "run_id": "R", "arm_id": "NR" if replay == 0 else "T", "segment_id": "S1", "epoch": epoch,
            "roles": roles,
        })
        atomic_write_json(parquet.with_suffix(".json"), {
            "schema_version": "stage1.process_telemetry.v1", "status": "COMPLETE",
            "run_id": "R", "arm_id": "NR" if replay == 0 else "T", "segment_id": "S1", "epoch": epoch,
            "row_count": 2, "observed_epoch_samples": 10 + replay,
            "observed_replay_samples": replay, "parquet_sha256": sha256_file(parquet),
            "role_summary_relpath": role.name, "role_summary_sha256": sha256_file(role),
        })
    schedule = tmp_path / "schedule.csv"
    pd.DataFrame(schedule_rows).to_csv(schedule, index=False)
    audit = tmp_path / "audit.json"
    atomic_write_json(audit, {
        "schema_version": "stage1.dynamic_training_audit.v2", "run_id": "R",
        "arm_id": "NR" if replay == 0 else "T", "total_epochs": epochs,
        "completed_epochs": epochs, "expected_steps_by_epoch": [3] * epochs,
        "observed_steps_by_epoch": [3] * epochs, "epoch_records": records,
        "segments": [
            {"segment_id": "S1", "start_epoch": 1, "end_epoch": 2, "status": "COMPLETED"},
            {"segment_id": "S2", "start_epoch": 3, "end_epoch": epochs, "status": "COMPLETED"},
        ],
    })
    return audit, schedule, telemetry


def test_all_epoch_validator_accepts_exact_coverage_and_dose(tmp_path: Path) -> None:
    audit, schedule, telemetry = _fixture(tmp_path)
    report = validate_all_epoch_telemetry(audit, schedule, telemetry, output_path=tmp_path / "out.json", expected_epochs=4)
    assert report["status"] == "PASS"
    assert report["cumulative_expected_replay_exposures"] == 8


def test_all_epoch_validator_rejects_missing_or_duplicate_epoch(tmp_path: Path) -> None:
    audit, schedule, telemetry = _fixture(tmp_path)
    frame = pd.read_csv(schedule)
    frame.loc[3, "epoch"] = 3
    frame.to_csv(schedule, index=False)
    with pytest.raises(ValidationError, match="all-epoch"):
        validate_all_epoch_telemetry(audit, schedule, telemetry, output_path=tmp_path / "out.json", expected_epochs=4)


def test_all_epoch_validator_enforces_no_replay_zero(tmp_path: Path) -> None:
    audit, schedule, telemetry = _fixture(tmp_path, replay=0)
    sidecar = telemetry / "epoch_0002_process_telemetry.json"
    payload = json.loads(sidecar.read_text())
    payload["observed_replay_samples"] = 1
    atomic_write_json(sidecar, payload, overwrite=True)
    with pytest.raises(ValidationError, match="all-epoch"):
        validate_all_epoch_telemetry(audit, schedule, telemetry, output_path=tmp_path / "out.json", expected_epochs=4)


def test_all_epoch_validator_rejects_atomic_half_write(tmp_path: Path) -> None:
    audit, schedule, telemetry = _fixture(tmp_path)
    (telemetry / "epoch_0004_process_telemetry.json").unlink()
    with pytest.raises(ValidationError, match="all-epoch"):
        validate_all_epoch_telemetry(audit, schedule, telemetry, output_path=tmp_path / "out.json", expected_epochs=4)
