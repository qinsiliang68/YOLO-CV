from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch

from stage1_gapvalue240.campaign_process_telemetry import (
    ProcessTelemetryCollector,
    ProcessTelemetrySpec,
)
from stage1_gapvalue240.campaign_smoke import (
    CampaignSmokeError,
    LocalSmokeValidationSpec,
    validate_local_smoke_run,
)
from stage1_gapvalue240.monitor import RESOURCE_COLUMNS
from stage1_gapvalue240.util import sha256_file


LOCK_SHA = "A" * 64


def _manifest(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _telemetry_spec(
    root: Path,
    *,
    epochs: int,
    segment_ids: tuple[str, ...] | None = None,
) -> ProcessTelemetrySpec:
    inputs = root / "inputs"
    normal = _manifest(
        inputs / "normal.csv",
        [
            {"canonical_image_relpath": "normal/n0.jpg", "Filename": "n0.jpg"},
            {"canonical_image_relpath": "normal/n1.jpg", "Filename": "n1.jpg"},
        ],
    )
    defect = _manifest(
        inputs / "defect.csv",
        [{"canonical_image_relpath": "defect/d0.jpg", "Filename": "d0.jpg"}],
    )
    replay = _manifest(
        inputs / "replay.csv",
        [
            {
                "staged_filename": "replay__SMOKE__00001__n0.jpg",
                "sample_id": "normal/n0.jpg",
                "y_true": 0,
                "replay_role": "normal_replay",
            }
        ],
    )
    monitor = _manifest(
        inputs / "monitor.csv",
        [
            {"sample_id": "normal/n0.jpg", "monitor_group": "NORMAL_CORE"},
            {"sample_id": "defect/d0.jpg", "monitor_group": "WEAK_DEFECT"},
        ],
    )
    spec = ProcessTelemetrySpec(
        run_id="LOCAL_SMOKE",
        arm_id="SMOKE",
        segment_id=f"LOCAL_SMOKE_E001_{epochs:03d}",
        output_dir=root / "process_telemetry",
        base_normal_manifest=normal,
        base_defect_manifest=defect,
        replay_identity_manifest=replay,
        monitor_manifest=monitor,
        expected_epoch_samples=4,
        expected_replay_samples=1,
    )
    resolved_segments = segment_ids or (spec.segment_id,) * epochs
    for epoch in range(1, epochs + 1):
        collector = ProcessTelemetryCollector(
            replace(spec, segment_id=resolved_segments[epoch - 1])
        )
        collector.start_epoch(epoch)
        collector.record_batch(
            torch.tensor([[2.0, -1.0], [1.0, 0.0], [-0.5, 1.5], [0.2, 0.8]]),
            {
                "cls": torch.tensor([0, 0, 1, 0]),
                "im_file": [
                    "C:/stage/train/no_target/n0.jpg",
                    "C:/stage/train/no_target/n1.jpg",
                    "C:/stage/train/target_defect/d0.jpg",
                    "C:/stage/train/no_target/replay__SMOKE__00001__n0.jpg",
                ],
                "img": torch.ones(4, 3, 16, 16),
            },
        )
        collector.finish_epoch(epoch)
    return spec


def _valid_run(
    tmp_path: Path,
    *,
    epochs: int = 3,
    segment_ids: tuple[str, ...] | None = None,
) -> LocalSmokeValidationSpec:
    root = tmp_path / "smoke"
    telemetry = _telemetry_spec(root, epochs=epochs, segment_ids=segment_ids)
    trainer = root / "trainer"
    (trainer / "weights").mkdir(parents=True)
    pd.DataFrame(
        [{"epoch": epoch, "train/loss": 0.25} for epoch in range(1, epochs + 1)]
    ).to_csv(trainer / "results.csv", index=False)
    (trainer / "weights/last.pt").write_bytes(b"last")
    (trainer / "weights/best.pt").write_bytes(b"best")
    (trainer / "args.yaml").write_text("epochs: 3\n", encoding="utf-8")
    state = root / "training_state"
    state.mkdir()
    (state / "last.pt").write_bytes(b"resume")
    checkpoint = state / f"checkpoint_epoch_{epochs:04d}.pt"
    checkpoint.write_bytes(b"resume")
    (state / f"checkpoint_epoch_{epochs:04d}.json").write_text(
        json.dumps({"epoch": epochs, "sha256": sha256_file(checkpoint), "resumable_expected": True}),
        encoding="utf-8",
    )
    (root / "dynamic_training_audit.json").write_text(
        json.dumps(
            {
                "schema_version": "stage1.dynamic_training_audit.v2",
                "run_id": "LOCAL_SMOKE",
                "arm_id": "SMOKE",
                "execution_mode": "SMOKE",
                "canonical_lock_file_sha256": LOCK_SHA,
                "smoke_canonical_overrides": ["batch", "epochs", "workers"],
                "completed_epochs": epochs,
                "expected_steps_by_epoch": [1] * epochs,
                "observed_steps_by_epoch": [1] * epochs,
                "loss_finite": True,
                "segments": [{"status": "COMPLETED", "end_epoch": epochs}],
                "epoch_records": [{"epoch": epoch} for epoch in range(1, epochs + 1)],
            }
        ),
        encoding="utf-8",
    )
    (root / "resolved_training_args.json").write_text(
        json.dumps(
            {
                "execution_mode": "SMOKE",
                "canonical_lock_file_sha256": LOCK_SHA,
                "canonical_lock_validation": {
                    "status": "PASS_WITH_DECLARED_SMOKE_OVERRIDES",
                    "declared_smoke_overrides": ["batch", "epochs", "workers"],
                },
            }
        ),
        encoding="utf-8",
    )
    resource = root / "resource_logs/resource.csv"
    resource.parent.mkdir(parents=True)
    row = {column: 0 for column in RESOURCE_COLUMNS}
    row.update(
        {
            "timestamp_unix": 1.0,
            "runtime_phase": "LOCAL_REAL_DATA_SMOKE",
            "process_rss_bytes": 1,
            "system_ram_available_bytes": 1,
            "disk_free_bytes": 1,
            "status": "OK",
        }
    )
    pd.DataFrame([row]).to_csv(resource, index=False)
    return LocalSmokeValidationSpec(
        run_id="LOCAL_SMOKE",
        arm_id="SMOKE",
        output_dir=root,
        telemetry=telemetry,
        expected_epochs=epochs,
        expected_steps_per_epoch=1,
        canonical_lock_file_sha256=LOCK_SHA,
        declared_smoke_overrides=("batch", "epochs", "workers"),
        resource_log=resource,
        telemetry_segment_ids_by_epoch=segment_ids,
    )


def test_local_smoke_validator_publishes_hashed_complete_evidence(tmp_path: Path) -> None:
    spec = _valid_run(tmp_path)

    result = validate_local_smoke_run(spec)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    manifest = pd.read_csv(result.artifact_manifest_path)
    assert report["status"] == "PASS"
    assert report["all_epoch_telemetry"] == "PASS"
    assert report["canonical_lock_binding"] == "PASS"
    assert report["checkpoint_resume_contract"] == "PASS"
    assert report["resource_log_contract"] == "PASS"
    assert report["artifact_count"] == len(manifest) == result.artifact_count
    assert manifest.relative_path.is_unique
    assert not manifest.relative_path.str.endswith(".tmp").any()
    assert set(report["validated_epochs"]) == {1, 2, 3}


def test_local_smoke_validator_fails_closed_on_missing_epoch_or_partial_file(
    tmp_path: Path,
) -> None:
    spec = _valid_run(tmp_path)
    missing = spec.telemetry.output_dir / "epoch_0002_process_telemetry.json"
    missing.unlink()
    with pytest.raises(CampaignSmokeError, match="epoch 2"):
        validate_local_smoke_run(spec)

    spec = _valid_run(tmp_path / "partial")
    (spec.output_dir / "process_telemetry/.epoch_0004.tmp").write_bytes(b"partial")
    with pytest.raises(CampaignSmokeError, match="temporary artifact"):
        validate_local_smoke_run(spec)


def test_local_smoke_validator_accepts_registered_segment_identity_per_epoch(
    tmp_path: Path,
) -> None:
    segments = ("BEFORE_KILL", "AFTER_RESUME", "AFTER_RESUME")
    spec = _valid_run(tmp_path, segment_ids=segments)

    result = validate_local_smoke_run(spec)

    details = json.loads(result.details_path.read_text(encoding="utf-8"))
    assert [row["segment_id"] for row in details["telemetry_epochs"]] == list(segments)
