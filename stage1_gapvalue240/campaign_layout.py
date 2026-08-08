"""Registry-safe layout for the post-240-run dynamic replay campaign."""

from __future__ import annotations

import csv
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any


class CampaignLayoutError(RuntimeError):
    """Raised when an existing registry conflicts with the campaign contract."""


CAMPAIGN_ID = "dynamic_replay_budget_efficiency_20260807"
CAMPAIGN_GROUP = "conditional_sample_value_dynamic_replay"
CAMPAIGN_STAGE_DIRS: dict[str, str] = {
    "00_registry": "00_registry",
    "01_field_audit": "01_field_audit",
    "02_literature": "02_literature",
    "03_preregistration": "03_preregistration",
    "04_run_queue": "04_run_queue",
    "05_training_runs": "05_training_runs",
    "06_process_exports": "06_process_exports",
    "07_evaluation": "07_evaluation",
    "08_reports": "08_reports",
    "09_aiops": "09_aiops",
}

# The unversioned directories above are immutable v1 evidence. Formal runtime
# entrypoints use these explicit v2 siblings so a worker cannot consume the
# historical 308-job queue by accident.
ACTIVE_PREREGISTRATION_DIRNAME = "03_preregistration_v2"
ACTIVE_RUN_QUEUE_DIRNAME = "04_run_queue_v2"


def active_preregistration_dir(campaign_root: str | Path) -> Path:
    return Path(campaign_root) / ACTIVE_PREREGISTRATION_DIRNAME


def active_run_queue_dir(campaign_root: str | Path) -> Path:
    return Path(campaign_root) / ACTIVE_RUN_QUEUE_DIRNAME


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(value: Any, path: Path) -> None:
    _atomic_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        path,
    )


def _atomic_registry_csv(rows: list[dict[str, str]], path: Path) -> None:
    columns = (
        "experiment_id",
        "experiment_group",
        "status",
        "physical_root",
        "artifact_mode",
        "note",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _campaign_registry_row(family_root: Path) -> dict[str, str]:
    workspace_artifact_root = Path("artifacts/stage1_sample_value_experiments")
    if family_root.name == "stage1_sample_value_experiments":
        physical_root = (
            workspace_artifact_root / "experiments" / CAMPAIGN_ID
        ).as_posix()
    else:
        physical_root = (family_root / "experiments" / CAMPAIGN_ID).as_posix()
    return {
        "experiment_id": CAMPAIGN_ID,
        "experiment_group": CAMPAIGN_GROUP,
        "status": "planning_and_instrumentation",
        "physical_root": physical_root,
        "artifact_mode": "native_grouped_tree",
        "note": (
            "Conditional sample-value campaign testing dynamic replay decay, weak-defect "
            "guarding, seed sensitivity, and a no-replay causal control before 2026-09-10."
        ),
    }


def _plan(created_at: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_id": CAMPAIGN_ID,
        "experiment_group": CAMPAIGN_GROUP,
        "status": "planning_and_instrumentation",
        "created_at": created_at,
        "deadline": "2026-09-10",
        "stage_dirs": CAMPAIGN_STAGE_DIRS,
        "legacy_evidence_dirs": {
            "preregistration": CAMPAIGN_STAGE_DIRS["03_preregistration"],
            "run_queue": CAMPAIGN_STAGE_DIRS["04_run_queue"],
        },
        "active_runtime_dirs": {
            "preregistration": ACTIVE_PREREGISTRATION_DIRNAME,
            "run_queue": ACTIVE_RUN_QUEUE_DIRNAME,
        },
        "scientific_protocol_source": (
            f"{ACTIVE_PREREGISTRATION_DIRNAME}/PREREGISTRATION_VALIDATION.json"
        ),
        "runtime_queue_source": (
            f"{ACTIVE_RUN_QUEUE_DIRNAME}/RUN_QUEUE_VALIDATION.json"
        ),
        "key_checkpoint_epochs": [120, 140, 150, 160, 180, 200],
        "layout_rule": "family/experiments/experiment_id/stage/registered_artifact",
        "source_experiments": [
            "hn_rn_40runs_20260630",
            "hn_band_120runs_20260707",
            "oof_dynamics_gap_value_20260708",
        ],
    }


def _readme() -> str:
    return "\n".join(
        [
            f"# {CAMPAIGN_ID}",
            "",
            "This experiment is parallel to the prior 40-run, 120-run, and 240-run/OOF work. ",
            "It does not replace or mutate those evidence packages.",
            "",
            "Primary question: can replay schedule control and weak-defect protection make a fixed ",
            "sample budget produce stable benefit across unseen training seeds?",
            "",
            "The current phase is field audit, literature synthesis, instrumentation, and pilot design. ",
            "Confirmatory training must not start until P0 telemetry gates pass.",
            "",
            "Deadline for the ten-machine campaign: `2026-09-10`.",
            "",
        ]
    )


def initialize_campaign_layout(
    family_root: str | Path,
    *,
    created_at: str | None = None,
) -> dict[str, str]:
    """Append the campaign registry entry and create its isolated stage tree."""

    root = Path(family_root).resolve()
    registry_dir = root / "00_registry"
    registry_json = registry_dir / "experiments.json"
    registry_csv = registry_dir / "experiments.csv"
    if not registry_json.is_file():
        raise CampaignLayoutError(f"Missing experiment family registry: {registry_json}")
    payload = json.loads(registry_json.read_text(encoding="utf-8"))
    rows = payload.get("experiments")
    if not isinstance(rows, list):
        raise CampaignLayoutError("Experiment registry has no experiments list")
    if len({str(row.get("experiment_id")) for row in rows}) != len(rows):
        raise CampaignLayoutError("Experiment registry already contains duplicate experiment IDs")

    expected = _campaign_registry_row(root)
    existing = next(
        (row for row in rows if row.get("experiment_id") == CAMPAIGN_ID),
        None,
    )
    if existing is None:
        rows.append(expected)
        action = "appended"
    else:
        contract_keys = ("experiment_group", "physical_root", "artifact_mode")
        conflicts = [key for key in contract_keys if existing.get(key) != expected[key]]
        if conflicts:
            raise CampaignLayoutError(
                f"Existing {CAMPAIGN_ID} conflicts with campaign contract: {conflicts}"
            )
        action = "preserved"

    timestamp = created_at or datetime.now().astimezone().isoformat(timespec="seconds")
    payload["experiments"] = rows
    _atomic_json(payload, registry_json)
    _atomic_registry_csv(rows, registry_csv)

    campaign_root = root / "experiments" / CAMPAIGN_ID
    for directory in CAMPAIGN_STAGE_DIRS.values():
        (campaign_root / directory).mkdir(parents=True, exist_ok=True)
    plan_path = campaign_root / "00_registry/experiment_plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("experiment_id") != CAMPAIGN_ID or plan.get("stage_dirs") != CAMPAIGN_STAGE_DIRS:
            raise CampaignLayoutError(f"Existing campaign plan conflicts with contract: {plan_path}")
    else:
        _atomic_json(_plan(timestamp), plan_path)
    readme_path = campaign_root / "README.md"
    if not readme_path.is_file():
        _atomic_text(_readme(), readme_path)
    return {
        "campaign_id": CAMPAIGN_ID,
        "campaign_root": str(campaign_root),
        "registry_action": action,
    }
