from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.campaign_aiops_reporting import build_daily_campaign_status, validate_cycle_closeout
from stage1_gapvalue240.errors import ValidationError


def _files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    events = tmp_path / "events.csv"
    pd.DataFrame([
        {"job_id": "J1", "block_id": "B1", "machine_id": "M1", "state": "COMPLETE", "attempt_id": "A1", "updated_at_unix": 1000, "retry_count": 0, "bytes_written_24h": 100, "disk_free_bytes": 1000, "canonical_completion": True, "gpu_train_seconds": 10, "dataloader_wait_seconds": 1},
        {"job_id": "J2", "block_id": "B2", "machine_id": "M2", "state": "PENDING", "attempt_id": "A2", "updated_at_unix": 1000, "retry_count": 0, "bytes_written_24h": 0, "disk_free_bytes": 2000, "canonical_completion": False, "gpu_train_seconds": 0, "dataloader_wait_seconds": 0},
    ]).to_csv(events, index=False)
    release = tmp_path / "release.json"
    release.write_text(json.dumps({"release_id": "REL", "queue_registry_sha256": "A" * 64}), encoding="utf-8")
    assignment = tmp_path / "assignment.json"
    assignment.write_text(json.dumps({"assignment_id": "ASG", "job_ids": ["J1", "J2"]}), encoding="utf-8")
    lock = tmp_path / "lock.json"
    lock.write_text("{}", encoding="utf-8")
    return events, release, assignment, lock


def test_daily_status_is_read_only_and_reports_resources(tmp_path: Path) -> None:
    events, release, assignment, lock = _files(tmp_path)
    report = build_daily_campaign_status(
        events, release_path=release, assignment_manifest_path=assignment,
        canonical_lock_path=lock, output_json=tmp_path / "daily.json",
        output_markdown=tmp_path / "daily.md", now_unix=1001,
    )
    assert report["read_only"] is True
    assert report["scientific_jobs_created"] == 0
    assert report["counts"]["complete"] == 1
    assert (tmp_path / "daily.md").read_text(encoding="utf-8").startswith("# Stage1")


def test_closeout_rejects_pending_or_duplicate_completion(tmp_path: Path) -> None:
    events, release, assignment, lock = _files(tmp_path)
    with pytest.raises(ValidationError, match="closeout"):
        validate_cycle_closeout(
            events, expected_jobs=["J1", "J2"], release_path=release,
            assignment_manifest_path=assignment, canonical_lock_path=lock,
            output_path=tmp_path / "closeout.json",
        )


def test_aiops_rejects_blind_input_path(tmp_path: Path) -> None:
    blind = tmp_path / "blind_holdout"
    blind.mkdir()
    events, release, assignment, lock = _files(tmp_path)
    target = blind / "events.csv"
    target.write_bytes(events.read_bytes())
    with pytest.raises(ValidationError, match="blind"):
        build_daily_campaign_status(
            target, release_path=release, assignment_manifest_path=assignment,
            canonical_lock_path=lock, output_json=tmp_path / "daily.json",
            output_markdown=tmp_path / "daily.md",
        )
