from __future__ import annotations

import json
from pathlib import Path
import threading

import pandas as pd
import pytest

from stage1_gapvalue240.campaign_canary import (
    aggregate_coordination_root_canary,
    aggregate_ten_machine_real_data_canary,
    build_coordination_canary_commands,
    build_ten_machine_real_data_canary_commands,
    run_coordination_root_canary,
)
from stage1_gapvalue240.errors import ValidationError
from stage1_gapvalue240.util import atomic_write_json, sha256_file


MACHINES = tuple(f"machine_{index:02d}" for index in range(1, 11))


def test_local_ten_node_coordination_canary_has_one_atomic_winner(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    output = tmp_path / "node_reports"
    reports = []
    errors = []
    guard = threading.Lock()

    def run(machine_id: str) -> None:
        try:
            report = run_coordination_root_canary(
                root,
                machine_id=machine_id,
                campaign_id="CAMPAIGN",
                generation="GEN_001",
                expected_machine_ids=MACHINES,
                output_dir=output,
                visibility_timeout_seconds=10,
            )
        except BaseException as exc:
            with guard:
                errors.append(exc)
        else:
            with guard:
                reports.append(report)

    threads = [threading.Thread(target=run, args=(machine,)) for machine in MACHINES]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert errors == []
    assert len(reports) == 10
    aggregate = aggregate_coordination_root_canary(
        output,
        expected_machine_ids=MACHINES,
        campaign_id="CAMPAIGN",
        generation="GEN_001",
        output_path=tmp_path / "COORDINATION_ROOT_CANARY_AGGREGATE.json",
    )
    assert aggregate["status"] == "PASS"
    assert aggregate["atomic_competition_winner_count"] == 1
    assert aggregate["visible_token_matrix_complete"]


def test_coordination_aggregate_rejects_missing_or_duplicate_machine(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    for machine in MACHINES[:-1]:
        atomic_write_json(
            reports / f"{machine}.json",
            {
                "schema_version": "stage1.coordination_root_canary_node.v1",
                "status": "PASS",
                "machine_id": machine,
                "campaign_id": "C",
                "generation": "G",
                "coordination_root_id": "ROOT",
                "atomic_competition": "LOSER",
                "visible_tokens": list(MACHINES),
                "token_hashes": {value: "A" * 64 for value in MACHINES},
            },
        )
    with pytest.raises(ValidationError, match="coordination root canary aggregation"):
        aggregate_coordination_root_canary(
            reports,
            expected_machine_ids=MACHINES,
            campaign_id="C",
            generation="G",
            output_path=tmp_path / "aggregate.json",
        )


def test_coordination_command_builder_emits_ten_independent_commands(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    for machine in MACHINES:
        (configs / f"{machine}.yaml").write_text(f"machine_id: {machine}\n", encoding="utf-8")
    result = build_coordination_canary_commands(
        configs,
        output_dir=tmp_path / "commands",
        repo_root=tmp_path,
        campaign_id="C",
        generation="G",
        expected_machine_ids=MACHINES,
    )
    rows = pd.read_csv(result["commands_csv"], keep_default_na=False)
    assert len(rows) == 10
    assert rows.machine_id.nunique() == 10
    assert rows.command.str.contains("run_coordination_root_canary.py", regex=False).all()
    assert rows.command.str.contains("<SET_SHARED_COORDINATION_ROOT>", regex=False).all()


def _real_canary_report(path: Path, machine: str, job: str) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": "stage1.ten_machine_real_data_canary_node.v1",
            "status": "PASS",
            "machine_id": machine,
            "job_id": job,
            "canonical_lock_validation": "PASS",
            "machine_config_validation": "PASS",
            "dataset_identity_validation": "PASS",
            "workers": 4,
            "lease_validation": "PASS",
            "completed_epochs": 1,
            "telemetry_validation": "PASS",
            "checkpoint_sidecar_validation": "PASS",
            "resource_log_validation": "PASS",
            "gpu_memory_released": True,
            "child_workers_released": True,
        },
    )


def test_ten_machine_real_data_canary_commands_and_aggregate(tmp_path: Path) -> None:
    commands = pd.DataFrame(
        {
            "job_id": [f"JOB_{index:02d}" for index in range(1, 11)],
            "assigned_machine_id": MACHINES,
            "command": [f"uv run python worker.py --job-id JOB_{index:02d}" for index in range(1, 11)],
            "command_sha256": ["A" * 64 for _ in MACHINES],
        }
    )
    source = tmp_path / "STANDALONE_JOB_COMMANDS.csv"
    commands.to_csv(source, index=False)
    built = build_ten_machine_real_data_canary_commands(
        source,
        output_dir=tmp_path / "canary_commands",
        expected_machine_ids=MACHINES,
    )
    built_rows = pd.read_csv(built["commands_csv"], keep_default_na=False)
    assert len(built_rows) == 10
    assert built_rows.job_id.nunique() == 10
    reports = tmp_path / "reports"
    reports.mkdir()
    for row in built_rows.itertuples(index=False):
        _real_canary_report(reports / f"{row.assigned_machine_id}.json", row.assigned_machine_id, row.job_id)
    aggregate = aggregate_ten_machine_real_data_canary(
        reports,
        expected_machine_ids=MACHINES,
        expected_commands_path=built["commands_csv"],
        output_path=tmp_path / "TEN_MACHINE_REAL_DATA_CANARY.json",
    )
    assert aggregate["status"] == "PASS"
    assert aggregate["node_count"] == 10


def test_ten_machine_real_data_canary_requires_exactly_ten_unique_pass_nodes(tmp_path: Path) -> None:
    commands = pd.DataFrame(
        {
            "job_id": [f"JOB_{index:02d}" for index in range(1, 11)],
            "assigned_machine_id": MACHINES,
            "command": [f"worker --job-id JOB_{index:02d}" for index in range(1, 11)],
            "command_sha256": ["A" * 64 for _ in MACHINES],
        }
    )
    source = tmp_path / "commands.csv"
    commands.to_csv(source, index=False)
    reports = tmp_path / "reports"
    reports.mkdir()
    for index, machine in enumerate(MACHINES[:-1], start=1):
        _real_canary_report(reports / f"{machine}.json", machine, f"JOB_{index:02d}")
    with pytest.raises(ValidationError, match="ten-machine real-data canary aggregation"):
        aggregate_ten_machine_real_data_canary(
            reports,
            expected_machine_ids=MACHINES,
            expected_commands_path=source,
            output_path=tmp_path / "aggregate.json",
        )
