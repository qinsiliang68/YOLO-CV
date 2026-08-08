from __future__ import annotations

import json
from pathlib import Path

import yaml

from stage1_gapvalue240.campaign_dry_generation import dry_generate_campaign_v2
from stage1_gapvalue240.campaign_engineering_gate import REQUIRED_EVIDENCE_SCHEMAS
from tests.stage1_gapvalue240.test_campaign_run_queue import _prereg


def _raw_evidence(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True)
    result: dict[str, Path] = {}
    for evidence_type, schema in REQUIRED_EVIDENCE_SCHEMAS.items():
        path = root / f"{evidence_type}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": schema,
                    "status": "PASS",
                    "fixture": True,
                }
            ),
            encoding="utf-8",
        )
        result[evidence_type] = path
    return result


def _machines(repo_root: Path) -> tuple[Path, dict[str, str]]:
    root = repo_root / "configs/machines"
    root.mkdir(parents=True)
    mapping: dict[str, str] = {}
    for index in range(1, 11):
        slot = f"M{index:02d}"
        machine = f"machine_{index:02d}"
        path = root / f"{machine}.yaml"
        path.write_text(yaml.safe_dump({"machine_id": machine}), encoding="utf-8")
        mapping[slot] = machine
    return root, mapping


def test_dry_generation_builds_v2_chain_without_activation(tmp_path: Path) -> None:
    prereg, monitor_source = _prereg(tmp_path)
    repo_root = tmp_path / "repo"
    (repo_root / "scripts/stage1_gapvalue240").mkdir(parents=True)
    (repo_root / "scripts/stage1_gapvalue240/dynamic_campaign_train_worker.py").write_text(
        "raise SystemExit('fixture only')\n", encoding="utf-8"
    )
    machines, mapping = _machines(repo_root)

    report = dry_generate_campaign_v2(
        prereg,
        monitor_source,
        output_root=tmp_path / "dry_chain",
        repo_root=repo_root,
        machine_configs_dir=machines,
        slot_mapping=mapping,
        raw_evidence_reports=_raw_evidence(tmp_path / "evidence"),
        campaign_id="stage1_dynamic_replay_campaign_v1",
        pilot_seed_ids=("S001", "S002"),
    )

    assert report["status"] == "PASS"
    assert report["activation_status"] == "NOT_ACTIVATED_DRY_RUN"
    assert report["queue"]["schema_version"] == "stage1.dynamic_campaign_run_queue.v2"
    assert report["release"]["job_count"] > 0
    assert report["assignment"]["job_count"] == report["release"]["job_count"]
    assert not (tmp_path / "dry_chain/coordination/ACTIVE_ASSIGNMENT.json").exists()
