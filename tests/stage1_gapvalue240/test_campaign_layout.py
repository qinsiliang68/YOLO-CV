from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from stage1_gapvalue240.campaign_layout import (
    ACTIVE_PREREGISTRATION_DIRNAME,
    ACTIVE_RUN_QUEUE_DIRNAME,
    CAMPAIGN_ID,
    CAMPAIGN_STAGE_DIRS,
    CampaignLayoutError,
    active_preregistration_dir,
    active_run_queue_dir,
    initialize_campaign_layout,
)


def _seed_registry(root: Path) -> None:
    registry = root / "00_registry"
    registry.mkdir(parents=True)
    row = {
        "experiment_id": "legacy_40runs",
        "experiment_group": "legacy",
        "status": "legacy_result",
        "physical_root": "artifacts/legacy_40runs",
        "artifact_mode": "legacy_pointer",
        "note": "Do not move.",
    }
    (registry / "experiments.json").write_text(
        json.dumps(
            {
                "created_at": "2026-07-01T00:00:00",
                "family_root": root.as_posix(),
                "experiments": [row],
            }
        ),
        encoding="utf-8",
    )
    with (registry / "experiments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_initializes_parallel_campaign_without_rewriting_legacy_rows(tmp_path: Path) -> None:
    family = tmp_path / "stage1_sample_value_experiments"
    _seed_registry(family)

    result = initialize_campaign_layout(family, created_at="2026-08-07T12:00:00")

    registry = json.loads((family / "00_registry/experiments.json").read_text(encoding="utf-8"))
    assert registry["created_at"] == "2026-07-01T00:00:00"
    assert [row["experiment_id"] for row in registry["experiments"]] == [
        "legacy_40runs",
        CAMPAIGN_ID,
    ]
    campaign = family / "experiments" / CAMPAIGN_ID
    assert set(path.name for path in campaign.iterdir() if path.is_dir()) == set(
        CAMPAIGN_STAGE_DIRS.values()
    )
    plan = json.loads((campaign / "00_registry/experiment_plan.json").read_text(encoding="utf-8"))
    assert plan["stage_dirs"] == CAMPAIGN_STAGE_DIRS
    assert plan["deadline"] == "2026-09-10"
    assert plan["status"] == "planning_and_instrumentation"
    assert plan["active_runtime_dirs"] == {
        "preregistration": ACTIVE_PREREGISTRATION_DIRNAME,
        "run_queue": ACTIVE_RUN_QUEUE_DIRNAME,
    }
    assert plan["scientific_protocol_source"] == (
        "03_preregistration_v2/PREREGISTRATION_VALIDATION.json"
    )
    assert "default_arms" not in plan
    assert result["registry_action"] == "appended"


def test_campaign_layout_is_idempotent(tmp_path: Path) -> None:
    family = tmp_path / "stage1_sample_value_experiments"
    _seed_registry(family)
    initialize_campaign_layout(family, created_at="2026-08-07T12:00:00")

    result = initialize_campaign_layout(family, created_at="2026-08-08T12:00:00")

    registry = json.loads((family / "00_registry/experiments.json").read_text(encoding="utf-8"))
    assert sum(row["experiment_id"] == CAMPAIGN_ID for row in registry["experiments"]) == 1
    assert result["registry_action"] == "preserved"


def test_campaign_layout_preserves_an_evolved_campaign_readme(tmp_path: Path) -> None:
    family = tmp_path / "stage1_sample_value_experiments"
    _seed_registry(family)
    initialize_campaign_layout(family, created_at="2026-08-07T12:00:00")
    readme = family / "experiments" / CAMPAIGN_ID / "README.md"
    readme.write_text("# Owner-updated runtime status\n", encoding="utf-8")

    initialize_campaign_layout(family, created_at="2026-08-08T12:00:00")

    assert readme.read_text(encoding="utf-8") == "# Owner-updated runtime status\n"


def test_campaign_layout_rejects_conflicting_existing_identity(tmp_path: Path) -> None:
    family = tmp_path / "stage1_sample_value_experiments"
    _seed_registry(family)
    registry_path = family / "00_registry/experiments.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["experiments"].append(
        {
            "experiment_id": CAMPAIGN_ID,
            "experiment_group": "wrong",
            "status": "active",
            "physical_root": "wrong/root",
            "artifact_mode": "native_grouped_tree",
            "note": "conflict",
        }
    )
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(CampaignLayoutError, match="conflicts with campaign contract"):
        initialize_campaign_layout(family)


def test_active_campaign_paths_keep_v1_evidence_separate_from_v2_runtime(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / CAMPAIGN_ID

    assert ACTIVE_PREREGISTRATION_DIRNAME == "03_preregistration_v2"
    assert ACTIVE_RUN_QUEUE_DIRNAME == "04_run_queue_v2"
    assert active_preregistration_dir(campaign) == campaign / "03_preregistration_v2"
    assert active_run_queue_dir(campaign) == campaign / "04_run_queue_v2"
    assert active_preregistration_dir(campaign) != campaign / CAMPAIGN_STAGE_DIRS["03_preregistration"]
    assert active_run_queue_dir(campaign) != campaign / CAMPAIGN_STAGE_DIRS["04_run_queue"]
