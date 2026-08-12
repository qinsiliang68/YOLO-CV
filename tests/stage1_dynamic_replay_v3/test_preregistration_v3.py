from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

from stage1_dynamic_replay_v3.preregistration_v3 import (
    REQUIRED_ARM_IDS,
    validate_preregistration_v3,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "stage1_sample_value_experiments"
    / "experiments"
    / "dynamic_replay_budget_efficiency_20260807"
)
PREREGISTRATION_ROOT = EXPERIMENT_ROOT / "03_preregistration_v3"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _copy_contract(tmp_path: Path) -> Path:
    target = tmp_path / "03_preregistration_v3"
    target.mkdir()
    for source in PREREGISTRATION_ROOT.iterdir():
        if source.is_file() and source.name != "PREREGISTRATION_VALIDATION.json":
            (target / source.name).write_bytes(source.read_bytes())
    return target


def test_repository_preregistration_v3_is_complete_but_not_a_result() -> None:
    report = validate_preregistration_v3(PREREGISTRATION_ROOT)

    assert report["status"] == "PASS"
    assert report["error_count"] == 0
    assert report["scientific_status"] == "PREREGISTERED_NOT_RUN"
    assert report["candidate_effectiveness_claim"] == "NOT_EVALUATED"
    assert report["formal_training_authorized"] is False
    assert report["engineering_gate_authorized"] is False
    assert report["blind_holdout_authorized"] is False
    assert set(report["arm_ids"]) == REQUIRED_ARM_IDS
    assert report["confirmation_seed_count"] == 14


def test_preregistration_v3_fails_closed_on_weighted_score_or_proxy_utility(
    tmp_path: Path,
) -> None:
    root = _copy_contract(tmp_path)
    contract_path = root / "SCIENTIFIC_CONTRACT.json"
    payload = _read_json(contract_path)
    payload["composition_contract"]["arbitrary_weighted_composite_allowed"] = True
    payload["signals"]["loss"]["utility_evidence"] = True
    _write_json(contract_path, payload)

    report = validate_preregistration_v3(root)

    assert report["status"] == "FAIL"
    codes = {error["code"] for error in report["errors"]}
    assert "ARBITRARY_WEIGHTED_SCORE_FORBIDDEN" in codes
    assert "SIGNAL_CANNOT_BE_UTILITY_EVIDENCE" in codes


def test_preregistration_v3_fails_closed_on_control_or_exposure_mismatch(
    tmp_path: Path,
) -> None:
    root = _copy_contract(tmp_path)
    matrix_path = root / "MINIMAL_FALSIFIABLE_MATRIX.csv"
    with matrix_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows = [row for row in rows if row["arm_id"] != "R2_METHOD_MATCHED_RANDOM"]
    for row in rows:
        if row["arm_id"] == "CURRENT_LOSS":
            row["replay_budget_group"] = "MISMATCHED_BUDGET"
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = validate_preregistration_v3(root)

    assert report["status"] == "FAIL"
    codes = {error["code"] for error in report["errors"]}
    assert "REQUIRED_ARM_MISSING" in codes
    assert "REPLAY_BUDGET_GROUP_MISMATCH" in codes


def test_preregistration_v3_fails_closed_on_step_lock_or_qrad_ladder_drift(
    tmp_path: Path,
) -> None:
    root = _copy_contract(tmp_path)
    contract_path = root / "SCIENTIFIC_CONTRACT.json"
    contract = _read_json(contract_path)
    contract["budget_contract"]["optimizer_step_lock"] = "replay adds steps"
    contract["budget_contract"]["replay_slots_per_epoch"] = 599
    _write_json(contract_path, contract)

    matrix_path = root / "MINIMAL_FALSIFIABLE_MATRIX.csv"
    with matrix_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    for row in rows:
        if row["arm_id"] == "T_QRAD":
            row["d_component"] = "OFF"
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = validate_preregistration_v3(root)

    assert report["status"] == "FAIL"
    codes = {error["code"] for error in report["errors"]}
    assert "OPTIMIZER_STEP_LOCK_MISMATCH" in codes
    assert "REPLAY_BUDGET_VALUE_MISMATCH" in codes
    assert "QRAD_LADDER_MISMATCH" in codes


def test_preregistration_v3_fails_closed_on_seed_metric_and_oracle_drift(
    tmp_path: Path,
) -> None:
    root = _copy_contract(tmp_path)
    statistics_path = root / "STATISTICAL_DECISION_RULES.json"
    statistics = _read_json(statistics_path)
    statistics["seed_design"]["confirmation_count"] = 10
    statistics["primary_endpoint"]["fn_max"] = 5
    statistics["checkpoint_contract"]["checkpoint_policy"] = "best.pt"
    statistics["checkpoint_contract"]["test_oracle_allowed"] = True
    _write_json(statistics_path, statistics)

    report = validate_preregistration_v3(root)

    assert report["status"] == "FAIL"
    codes = {error["code"] for error in report["errors"]}
    assert "CONFIRMATION_SEED_COUNT_MISMATCH" in codes
    assert "PRIMARY_FRONTIER_RANGE_MISMATCH" in codes
    assert "FIXED_CHECKPOINT_REQUIRED" in codes
    assert "TEST_ORACLE_FORBIDDEN" in codes


def test_preregistration_v3_fails_closed_on_missing_collection_identity_fields(
    tmp_path: Path,
) -> None:
    root = _copy_contract(tmp_path)
    schema_path = root / "DATA_COLLECTION_SCHEMA.json"
    schema = _read_json(schema_path)
    fields = schema["artifacts"]["epoch_exposure_ledger"]["required_fields"]
    fields.remove("optimizer_visible_replay_exposure_actual")
    fields.remove("cumulative_repeat_occurrences_actual")
    _write_json(schema_path, schema)

    report = validate_preregistration_v3(root)

    assert report["status"] == "FAIL"
    errors = {
        (error.get("field"), error["code"])
        for error in report["errors"]
    }
    assert (
        "epoch_exposure_ledger.required_fields",
        "REQUIRED_COLLECTION_FIELD_MISSING",
    ) in errors


def test_preregistration_v3_cli_writes_machine_audit(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts"
                / "stage1_dynamic_replay_v3"
                / "validate_preregistration_v3.py"
            ),
            "--preregistration-root",
            str(PREREGISTRATION_ROOT),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = _read_json(output)
    assert payload["status"] == "PASS"
    assert payload["validation_executed_training"] is False
