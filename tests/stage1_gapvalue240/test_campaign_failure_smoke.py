from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.campaign_dynamic_training import SmokeFailureInjection
from stage1_gapvalue240.campaign_failure_smoke import (
    FailureSmokeError,
    build_failure_smoke_segment,
    validate_interrupted_boundary,
    validate_telemetry_write_interruption,
)
from stage1_gapvalue240.campaign_smoke_dataset import LocalSmokeDataset
from stage1_gapvalue240.util import sha256_file
from scripts.stage1_gapvalue240.run_local_dynamic_failure_injection import parse_args


ROOT = Path(__file__).resolve().parents[2]


def test_failure_injection_cli_uses_multi_epoch_real_smoke_defaults() -> None:
    args = parse_args(["--scratch-root", "C:/tmp/failure-smoke"])

    assert args.epochs == 3
    assert args.train_per_class == 32
    assert args.replay_normal == 4
    assert args.child_config is None


def _subset(tmp_path: Path) -> LocalSmokeDataset:
    root = tmp_path / "subset"
    dataset = root / "dataset"
    for relative in (
        "train/no_target",
        "train/target_defect",
        "val/no_target",
        "val/target_defect",
    ):
        (dataset / relative).mkdir(parents=True, exist_ok=True)
    inputs = root / "inputs"
    inputs.mkdir()
    normal = inputs / "base_normal.csv"
    defect = inputs / "base_defect.csv"
    replay = inputs / "replay_identity.csv"
    monitor = inputs / "monitor.csv"
    pd.DataFrame(
        [{"canonical_image_relpath": "n.jpg", "Filename": "n.jpg"}]
    ).to_csv(normal, index=False)
    pd.DataFrame(
        [{"canonical_image_relpath": "d.jpg", "Filename": "d.jpg"}]
    ).to_csv(defect, index=False)
    pd.DataFrame(
        [
            {
                "staged_filename": "replay__LOCAL__00001__n.jpg",
                "sample_id": "n.jpg",
                "y_true": 0,
                "replay_role": "normal_replay",
            }
        ]
    ).to_csv(replay, index=False)
    pd.DataFrame(
        [
            {"sample_id": "n.jpg", "monitor_group": "N"},
            {"sample_id": "d.jpg", "monitor_group": "D"},
        ]
    ).to_csv(monitor, index=False)
    validation = root / "SMOKE_DATASET_VALIDATION.json"
    validation.write_text('{"status":"PASS"}', encoding="utf-8")
    return LocalSmokeDataset(
        root=root,
        dataset_dir=dataset,
        base_normal_manifest=normal,
        base_defect_manifest=defect,
        replay_identity_manifest=replay,
        monitor_manifest=monitor,
        expected_epoch_samples=3,
        expected_replay_samples=1,
        expected_steps_per_epoch=1,
        validation_path=validation,
    )


def test_build_failure_smoke_segment_inherits_canonical_configuration(tmp_path: Path) -> None:
    subset = _subset(tmp_path)
    injection = SmokeFailureInjection(
        mode="OOM_AT_BATCH_START", target_epoch=1, target_batch=1
    )
    built = build_failure_smoke_segment(
        repo_root=ROOT,
        subset=subset,
        output_dir=tmp_path / "run",
        run_id="FAILURE_SMOKE",
        total_epochs=3,
        segment_start_epoch=1,
        segment_end_epoch=3,
        batch=4,
        workers=0,
        device="0",
        seed=7,
        segment_id="OOM_ATTEMPT",
        failure_injection=injection,
    )

    assert built.training.execution_mode == "SMOKE"
    assert built.training.smoke_failure_injection == injection
    assert built.training.canonical_lock_path.name == "CANONICAL_TRAINING_LOCK_v1.json"
    assert set(built.smoke_canonical_overrides) == {"epochs", "batch", "workers"}
    assert built.telemetry.expected_epoch_samples == 3
    assert built.telemetry.segment_id == "OOM_ATTEMPT"


def test_interruption_boundary_requires_exact_epoch_checkpoint_and_no_next_telemetry(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    state = output / "training_state"
    telemetry = output / "process_telemetry"
    trainer = output / "trainer"
    state.mkdir(parents=True)
    telemetry.mkdir()
    trainer.mkdir()
    checkpoint = state / "checkpoint_epoch_0001.pt"
    checkpoint.write_bytes(b"checkpoint")
    (state / "last.pt").write_bytes(b"checkpoint")
    (state / "checkpoint_epoch_0001.json").write_text(
        json.dumps({"epoch": 1, "sha256": sha256_file(checkpoint)}), encoding="utf-8"
    )
    parquet = telemetry / "epoch_0001_process_telemetry.parquet"
    parquet.write_bytes(b"telemetry")
    parquet.with_suffix(".json").write_text(
        json.dumps({"status": "COMPLETE", "epoch": 1, "parquet_sha256": sha256_file(parquet)}),
        encoding="utf-8",
    )
    pd.DataFrame([{"epoch": 1}]).to_csv(trainer / "results.csv", index=False)
    (output / "dynamic_training_audit.json").write_text(
        json.dumps(
            {
                "completed_epochs": 1,
                "epoch_records": [{"epoch": 1}],
                "segments": [{"status": "RUNNING"}],
            }
        ),
        encoding="utf-8",
    )

    report = validate_interrupted_boundary(
        output,
        expected_completed_epoch=1,
        expected_next_epoch=2,
    )
    assert report["status"] == "PASS"
    assert report["resume_checkpoint_sha256"] == sha256_file(state / "last.pt")

    (telemetry / "epoch_0002_process_telemetry.parquet").write_bytes(b"fake complete")
    with pytest.raises(FailureSmokeError, match="next epoch telemetry"):
        validate_interrupted_boundary(
            output,
            expected_completed_epoch=1,
            expected_next_epoch=2,
        )


def test_telemetry_write_interruption_requires_no_published_epoch(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "process_telemetry").mkdir()
    (output / "dynamic_training_audit.json").write_text(
        json.dumps(
            {
                "completed_epochs": 1,
                "segments": [{"status": "FAILED", "error": "OSError"}],
            }
        ),
        encoding="utf-8",
    )

    report = validate_telemetry_write_interruption(output, failed_epoch=1)
    assert report["status"] == "PASS"

    (output / "process_telemetry/epoch_0001_process_telemetry.json").write_text(
        '{"status":"COMPLETE"}', encoding="utf-8"
    )
    with pytest.raises(FailureSmokeError, match="published telemetry"):
        validate_telemetry_write_interruption(output, failed_epoch=1)
