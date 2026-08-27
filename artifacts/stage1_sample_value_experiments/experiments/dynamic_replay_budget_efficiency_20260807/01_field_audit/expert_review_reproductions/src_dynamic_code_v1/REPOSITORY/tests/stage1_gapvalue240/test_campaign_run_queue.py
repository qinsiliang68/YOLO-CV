from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.campaign_preregistration import build_campaign_preregistration
from stage1_gapvalue240.campaign_run_queue import (
    RunQueueError,
    build_campaign_run_queue,
    build_replay_identity_manifest,
)
from stage1_gapvalue240.util import sha256_file


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "configs/stage1_gapvalue240/CANONICAL_TRAINING_LOCK_v1.json"


def _write(path: Path, rows: list[dict[str, object]], columns: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def _selection(path: Path, selection_id: str, normals: int, defects: int) -> Path:
    rows = []
    selection_rank = 1
    for index in range(1, normals + 1):
        rows.append(
            {
                "selection_id": selection_id,
                "selection_rank": selection_rank,
                "role_rank": index,
                "sample_id": f"normal/n{index:05d}.jpg",
                "y_true": 0,
                "replay_role": "normal_replay",
            }
        )
        selection_rank += 1
    for index in range(1, defects + 1):
        rows.append(
            {
                "selection_id": selection_id,
                "selection_rank": selection_rank,
                "role_rank": index,
                "sample_id": f"defect/d{index:05d}.jpg",
                "y_true": 1,
                "replay_role": "defect_guard",
            }
        )
        selection_rank += 1
    return _write(path, rows)


def _prereg(tmp_path: Path) -> tuple[Path, Path]:
    treatment = _write(
        tmp_path / "sources/treatment.csv",
        [
            {
                "rank": index,
                "sample_id": f"normal/n{index:05d}.jpg",
                "y_true": 0,
                "oof_fold": (index - 1) % 10,
                "dynamic_bucket": "learnable_hard",
                "mean_p_defect": 1 - index / 10_000,
                "correct_rate": index / 10_000,
                "std_p_defect": 0.25,
                "source_method": "GapCritical-Strict",
            }
            for index in range(1, 6001)
        ],
    )
    monitor_source = _write(
        tmp_path / "sources/sample_value_table.csv",
        [
            {
                "sample_id": f"normal/n{index:05d}.jpg",
                "y_true": 0,
                "oof_fold": (index - 1) % 10,
                "dynamic_bucket": "learnable_hard",
                "mean_p_defect": 1 - index / 10_000,
                "correct_rate": index / 10_000,
                "std_p_defect": 0.25,
            }
            for index in range(1, 6001)
        ]
        + [
            {
                "sample_id": f"defect/d{index:05d}.jpg",
                "y_true": 1,
                "oof_fold": (index - 1) % 10,
                "dynamic_bucket": "learnable_hard" if index <= 300 else "easy_clean",
                "mean_p_defect": index / 10_000,
                "correct_rate": 0.5,
                "std_p_defect": 0.15,
            }
            for index in range(1, 6001)
        ],
    )
    root = tmp_path / "03_preregistration"
    build_campaign_preregistration(
        root,
        treatment_ranking_source=treatment,
        canonical_lock_path=LOCK,
        machine_count=10,
    )
    return root, monitor_source


def test_build_campaign_run_queue_freezes_gated_jobs_monitor_and_lock(tmp_path: Path) -> None:
    prereg, monitor_source = _prereg(tmp_path)
    output = tmp_path / "04_run_queue"

    result = build_campaign_run_queue(prereg, output, monitor_source=monitor_source)

    registry = pd.read_csv(result.job_registry, keep_default_na=False)
    assert len(registry) == 296
    assert set(registry.release_state) == {"ENGINEERING_GATE", "HELD"}
    assert not registry.queue_status.eq("READY").any()
    assert set(registry.loc[registry.cycle_id == "CYCLE_1", "queue_status"]) == {
        "HELD_ENGINEERING_GATE",
        "BLOCKED_ENGINEERING_GATE",
    }
    assert set(registry.loc[registry.cycle_id == "CYCLE_2", "queue_status"]) == {
        "HELD_SCIENTIFIC_GATE"
    }
    expected_lock = sha256_file(LOCK)
    assert set(registry.canonical_lock_file_sha256) == {expected_lock}
    lock_copy = output / registry.iloc[0].canonical_lock_relpath
    assert sha256_file(lock_copy) == expected_lock

    branched = registry[
        (registry.logical_arm_id == "C1_T_RHO_2P5_SAME_PEAK_TAPER")
        & (registry.segment_start_epoch == 141)
    ].iloc[0]
    assert branched.active_selection_rows == 2000
    assert branched.dependency_job_id
    assert branched.dependency_output_relpath
    template = pd.read_csv(output / branched.active_selection_template_relpath)
    assert template.sample_id.iloc[0] == "normal/n00001.jpg"
    assert len(template) == 2000

    monitor = pd.read_csv(result.monitor_manifest)
    assert monitor.sample_id.is_unique
    assert set(monitor.monitor_group) == {
        "NORMAL_REPLAY_CORE_0P5_CLASS_PERCENT",
        "NORMAL_REPLAY_REMAINDER_STRATIFIED",
        "WEAK_DEFECT_CORE_0P5_CLASS_PERCENT",
        "WEAK_DEFECT_BUFFER_0P5_TO_5_CLASS_PERCENT_STRATIFIED",
    }
    assert len(monitor) == 120

    validation = json.loads(result.validation_path.read_text(encoding="utf-8"))
    assert validation["status"] == "PASS"
    assert validation["job_count"] == 296
    assert validation["canonical_lock_file_sha256"] == expected_lock
    assert validation["cycle_job_counts"] == {"CYCLE_1": 88, "CYCLE_2": 208}
    assert len(list((output / "machines").glob("M*.csv"))) == 10


def test_build_campaign_run_queue_rejects_unknown_dependency_or_lock_drift(tmp_path: Path) -> None:
    prereg, monitor_source = _prereg(tmp_path)
    graph_path = prereg / "PHYSICAL_JOB_GRAPH.csv"
    graph = pd.read_csv(graph_path, keep_default_na=False)
    graph.loc[1, "dependency_job_id"] = "MISSING_JOB"
    graph.to_csv(graph_path, index=False)
    with pytest.raises(RunQueueError, match="unknown dependencies"):
        build_campaign_run_queue(
            prereg,
            tmp_path / "queue_missing_dependency",
            monitor_source=monitor_source,
        )

    prereg, monitor_source = _prereg(tmp_path / "lock_case")
    graph_path = prereg / "PHYSICAL_JOB_GRAPH.csv"
    graph = pd.read_csv(graph_path, keep_default_na=False)
    graph.loc[0, "canonical_lock_file_sha256"] = "0" * 64
    graph.to_csv(graph_path, index=False)
    with pytest.raises(RunQueueError, match="canonical lock"):
        build_campaign_run_queue(
            prereg,
            tmp_path / "queue_lock_drift",
            monitor_source=monitor_source,
        )


def test_replay_identity_manifest_maps_template_to_staged_filenames(tmp_path: Path) -> None:
    template = _selection(tmp_path / "template.csv", "T", 2, 1)
    normal = _write(
        tmp_path / "normal.csv",
        [
            {"canonical_image_relpath": "normal/n00001.jpg", "Filename": "N-1.jpg"},
            {"canonical_image_relpath": "normal/n00002.jpg", "Filename": "N-2.jpg"},
        ],
    )
    defect = _write(
        tmp_path / "defect.csv",
        [{"canonical_image_relpath": "defect/d00001.jpg", "Filename": "D-1.jpg"}],
    )

    result = build_replay_identity_manifest(
        template,
        normal,
        defect,
        run_slot="JOB_S001_A",
        output_path=tmp_path / "replay_identity.csv",
    )
    frame = pd.read_csv(result.path)
    assert frame.staged_filename.tolist() == [
        "replay__JOB_S001_A__00001__N-1.jpg",
        "replay__JOB_S001_A__00002__N-2.jpg",
        "replay__JOB_S001_A__00003__D-1.jpg",
    ]
    assert frame.replay_role.tolist() == ["normal_replay", "normal_replay", "defect_guard"]
    assert result.row_count == 3
    assert result.sha256 == sha256_file(result.path)
