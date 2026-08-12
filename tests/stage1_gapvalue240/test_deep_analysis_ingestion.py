from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from stage1_gapvalue240.deep_analysis import (
    CanonicalInputError,
    _training_summary,
    ingest_canonical_runs,
)
from stage1_gapvalue240.metrics import operational_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _make_run(
    extracted_root: Path,
    *,
    slot: str,
    triad: str,
    arm: str,
    package: str,
    attempt_name: str,
    machine: str,
    selection_seed: int,
) -> tuple[dict, dict]:
    attempt = (
        extracted_root
        / f"stage1_gapvalue240_{package}_upload"
        / "runs"
        / slot
        / attempt_name
    )
    selection = attempt / "01_manifests/selection_manifest.csv"
    selection.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"sample_id": f"{slot}_sample", "arm": arm, "rank": 1}]
    ).to_csv(selection, index=False)
    selection_sha = _sha256(selection)

    prediction = pd.DataFrame(
        {
            "sample_id": ["n0", "n1", "n2", "d0", "d1"],
            "y_true": [0, 0, 0, 1, 1],
            # d0/n1 are tied; this exercises whole-tie-group handling.
            "score": [0.05, 0.5, 0.1, 0.5, 0.9],
            "score_raw": [-3.0, 0.0, -2.0, 0.0, 3.0],
        }
    )
    pred_path = attempt / "04_predictions/val_op_predictions.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(pred_path, index=False)
    metrics, _ = operational_metrics(prediction, fn_limit=1, tn_target=2)
    _write_json(attempt / "05_metrics/operational_metrics.json", metrics)

    epochs = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "time": [10.0, 20.0, 30.0],
            "train/loss": [0.5, 0.4, 0.3],
            "metrics/accuracy_top1": [0.7, 0.8, 0.75],
            "metrics/accuracy_top5": [1.0, 1.0, 1.0],
            "val/loss": [0.4, 0.2, 0.25],
        }
    )
    epoch_path = attempt / "02_logs/epoch_training_metrics.csv"
    epoch_path.parent.mkdir(parents=True, exist_ok=True)
    epochs.to_csv(epoch_path, index=False)
    _write_json(
        attempt / "02_logs/training_execution_audit.json",
        {
            "completed_epochs": 3,
            "expected_steps_per_epoch": 5,
            "optimizer_steps_total": 15,
            "resume_count": 1 if arm == "R2" else 0,
            "resume_mode": "native_approximate",
            "loss_finite": True,
        },
    )

    identity = {
        "attempt_id": attempt_name.removeprefix("attempt_"),
        "run_slot": slot,
        "machine_id": machine,
        "input_snapshot_id": "snapshot-A",
        "matrix_sha256": "matrix-sha",
        "release_ref": "release-v1",
        "release_commit": "commit-v1",
        "selection_sha256": selection_sha,
        "resume_count": 1 if arm == "R2" else 0,
        "resume_mode": "native_approximate",
        "dry_run": False,
        "run_row": {
            "run_slot": slot,
            "triad_id": triad,
            "phase": "A",
            "condition_slot": "A01",
            "condition_id": "A01_Test_B1",
            "method": "Test",
            "budget": 1,
            "guard_ratio": 0.0,
            "arm": arm,
            "training_seed": 11,
            "selection_seed": selection_seed,
            "discovery_or_confirmation": "discovery",
        },
    }
    _write_json(attempt / "00_identity/run_identity.json", identity)
    _write_json(
        attempt / "07_validation/postflight_report.json",
        {"status": "PASS", "issues": []},
    )
    _write_json(
        attempt / "08_status/status.json",
        {"state": "VALIDATED", "run_slot": slot},
    )

    required = [
        "00_identity/run_identity.json",
        "01_manifests/selection_manifest.csv",
        "02_logs/epoch_training_metrics.csv",
        "02_logs/training_execution_audit.json",
        "04_predictions/val_op_predictions.csv",
        "05_metrics/operational_metrics.json",
        "07_validation/postflight_report.json",
    ]
    rows = []
    for relative in required:
        path = attempt / relative
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = attempt / "07_validation/artifact_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    inventory_row = {
        "run_slot": slot,
        "triad_id": triad,
        "phase": "A",
        "condition_id": "A01_Test_B1",
        "method": "Test",
        "budget": 1,
        "guard_ratio": 0.0,
        "arm": arm,
        "training_seed": 11,
        "selection_seed": selection_seed,
        "package": package,
        "machine_id": machine,
        "attempt_id": attempt_name,
        "resume_count": 1 if arm == "R2" else 0,
        "selection_sha256": selection_sha,
        "release_ref": "release-v1",
        "release_commit": "commit-v1",
        "input_snapshot_id": "snapshot-A",
    }
    matrix_row = identity["run_row"]
    return inventory_row, matrix_row


@pytest.fixture
def canonical_fixture(tmp_path: Path) -> dict:
    extracted = tmp_path / "extracted"
    inventory_rows = []
    matrix_rows = []
    for index, arm in enumerate(("T", "R1", "R2"), start=1):
        inv, matrix = _make_run(
            extracted,
            slot=f"RUN_{index:03d}",
            triad="TRIAD_001",
            arm=arm,
            package="P35",
            attempt_name=f"attempt_exact_{arm}",
            machine=f"machine_{index:02d}",
            selection_seed=100 + index,
        )
        inventory_rows.append(inv)
        matrix_rows.append(matrix)

    # A tempting alternate attempt must never be discovered or selected.
    alternate = (
        extracted
        / "stage1_gapvalue240_P35_upload/runs/RUN_001/attempt_newer_but_not_inventory"
    )
    alternate.mkdir(parents=True)

    inventory = tmp_path / "inventory.csv"
    matrix = tmp_path / "matrix.csv"
    pd.DataFrame(inventory_rows).to_csv(inventory, index=False)
    pd.DataFrame(matrix_rows).to_csv(matrix, index=False)
    matrix_sha = _sha256(matrix)
    for row in inventory_rows:
        attempt = (
            extracted
            / f"stage1_gapvalue240_{row['package']}_upload"
            / "runs"
            / row["run_slot"]
            / row["attempt_id"]
        )
        identity_path = attempt / "00_identity/run_identity.json"
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        identity["matrix_sha256"] = matrix_sha
        _write_json(identity_path, identity)
        artifact = attempt / "07_validation/artifact_manifest.csv"
        manifest = pd.read_csv(artifact)
        match = manifest.relative_path == "00_identity/run_identity.json"
        manifest.loc[match, "size_bytes"] = identity_path.stat().st_size
        manifest.loc[match, "sha256"] = _sha256(identity_path)
        manifest.to_csv(artifact, index=False)
    return {
        "root": extracted,
        "inventory": inventory,
        "matrix": matrix,
        "inventory_rows": inventory_rows,
    }


def test_ingests_only_inventory_attempts_and_builds_training_summary(
    canonical_fixture: dict,
) -> None:
    result = ingest_canonical_runs(
        canonical_fixture["root"],
        canonical_fixture["inventory"],
        canonical_fixture["matrix"],
        expected_runs=3,
        expected_triads=1,
        expected_comparisons=2,
        expected_epochs=3,
        expected_prediction_rows=5,
        expected_normal_count=3,
        expected_defect_count=2,
        recompute_predictions=False,
    )

    assert len(result.runs) == 3
    assert result.runs["attempt_dir"].str.contains("attempt_exact_").all()
    assert set(result.runs["machine_id"]) == {
        "machine_01",
        "machine_02",
        "machine_03",
    }
    assert set(result.runs["input_snapshot_id"]) == {"snapshot-A"}
    assert result.runs.set_index("arm").loc["R2", "resume_count"] == 1
    assert result.training_summaries["completed_epochs"].eq(3).all()
    assert result.training_summaries["best_top1_epoch"].eq(2).all()
    assert result.training_summaries["final_top1"].eq(0.75).all()
    assert result.training_summaries["best_val_loss_epoch"].eq(2).all()
    assert result.training_summaries["best_val_loss"].eq(0.2).all()
    assert result.training_summaries["best_final_top1_gap"].tolist() == pytest.approx(
        [0.05] * 3
    )
    assert result.training_summaries[
        "final_minus_best_val_loss"
    ].tolist() == pytest.approx([0.05] * 3)
    assert result.training_summaries["last_window_epochs"].eq(3).all()
    assert result.training_summaries["overfit_flag"].eq(False).all()
    assert result.training_summaries["oscillation_flag"].eq(False).all()
    assert result.metric_audit["recomputed"].eq(False).all()


def test_full_recompute_matches_saved_tie_safe_metrics(canonical_fixture: dict) -> None:
    result = ingest_canonical_runs(
        canonical_fixture["root"],
        canonical_fixture["inventory"],
        canonical_fixture["matrix"],
        expected_runs=3,
        expected_triads=1,
        expected_comparisons=2,
        expected_epochs=3,
        expected_prediction_rows=5,
        expected_normal_count=3,
        expected_defect_count=2,
        recompute_predictions=True,
        fn_limit=1,
        tn_target=2,
    )

    assert result.metric_audit["recomputed"].all()
    assert result.metric_audit["exact_match"].all()
    assert result.metric_audit["integer_fields_exact"].all()
    assert result.metric_audit["raw_calibrated_integer_metrics_exact"].all()
    assert result.metric_audit["max_float_abs_error"].max() < 1e-12
    assert result.runs["TN_at_FN95"].notna().all()
    assert result.runs["FN_at_TN68253"].notna().all()
    assert result.runs["raw_TN_at_FN95"].equals(result.runs["TN_at_FN95"])
    assert result.runs["raw_FN_at_TN68253"].equals(
        result.runs["FN_at_TN68253"]
    )
    assert result.runs["raw_gap_q68_q050"].notna().all()
    assert result.runs["raw_tail_gap_q90_q05"].notna().all()


def test_full_recompute_rejects_raw_calibrated_operational_count_drift(
    canonical_fixture: dict,
) -> None:
    row = canonical_fixture["inventory_rows"][0]
    attempt = (
        canonical_fixture["root"]
        / f"stage1_gapvalue240_{row['package']}_upload"
        / "runs"
        / row["run_slot"]
        / row["attempt_id"]
    )
    prediction_path = attempt / "04_predictions/val_op_predictions.csv"
    predictions = pd.read_csv(prediction_path)
    # Keep calibrated scores unchanged, but invert raw ordering to make the
    # raw operational counts disagree with the monotonic calibrated result.
    predictions["score_raw"] = -predictions["score_raw"]
    predictions.to_csv(prediction_path, index=False)
    manifest_path = attempt / "07_validation/artifact_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    match = manifest["relative_path"] == "04_predictions/val_op_predictions.csv"
    manifest.loc[match, "size_bytes"] = prediction_path.stat().st_size
    manifest.loc[match, "sha256"] = _sha256(prediction_path)
    manifest.to_csv(manifest_path, index=False)

    with pytest.raises(
        CanonicalInputError,
        match="raw/calibrated operational integer metrics differ",
    ):
        ingest_canonical_runs(
            canonical_fixture["root"],
            canonical_fixture["inventory"],
            canonical_fixture["matrix"],
            expected_runs=3,
            expected_triads=1,
            expected_comparisons=2,
            expected_epochs=3,
            expected_prediction_rows=5,
            expected_normal_count=3,
            expected_defect_count=2,
            recompute_predictions=True,
            fn_limit=1,
            tn_target=2,
        )


def test_training_summary_flags_only_strong_overfit_and_oscillation(
    tmp_path: Path,
) -> None:
    def write_attempt(
        name: str,
        *,
        val_loss: list[float],
        top1: list[float],
    ) -> Path:
        attempt = tmp_path / name
        epoch_count = len(val_loss)
        (attempt / "02_logs").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "epoch": range(1, epoch_count + 1),
                "time": [float(index) for index in range(1, epoch_count + 1)],
                "train/loss": [
                    0.8 - 0.5 * index / max(epoch_count - 1, 1)
                    for index in range(epoch_count)
                ],
                "metrics/accuracy_top1": top1,
                "val/loss": val_loss,
            }
        ).to_csv(
            attempt / "02_logs/epoch_training_metrics.csv",
            index=False,
        )
        _write_json(
            attempt / "02_logs/training_execution_audit.json",
            {
                "completed_epochs": epoch_count,
                "expected_steps_per_epoch": 5,
                "optimizer_steps_total": epoch_count * 5,
                "loss_finite": True,
            },
        )
        return attempt

    decline = [0.5 - 0.4 * index / 19 for index in range(20)]
    rise = [0.12 + 0.28 * index / 19 for index in range(20)]
    top1_up = [0.7 + 0.2 * index / 19 for index in range(20)]
    top1_down = [0.895 - 0.095 * index / 19 for index in range(20)]
    overfit_attempt = write_attempt(
        "overfit",
        val_loss=decline + rise,
        top1=top1_up + top1_down,
    )
    overfit = _training_summary(
        overfit_attempt,
        run_slot="RUN_OVERFIT",
        expected_epochs=40,
    )
    assert overfit["overfit_flag"] is True
    assert overfit["oscillation_flag"] is False
    assert overfit["best_val_loss_epoch"] == 20
    assert overfit["final_minus_best_val_loss"] > 0.25

    oscillating_attempt = write_attempt(
        "oscillating",
        val_loss=[0.2 if index % 2 == 0 else 0.4 for index in range(20)],
        top1=[0.75 if index % 2 == 0 else 0.85 for index in range(20)],
    )
    oscillating = _training_summary(
        oscillating_attempt,
        run_slot="RUN_OSCILLATING",
        expected_epochs=20,
    )
    assert oscillating["oscillation_flag"] is True
    assert oscillating["overfit_flag"] is False
    assert oscillating["last_window_val_loss_direction_changes"] >= 10
    assert oscillating["last_window_top1_direction_changes"] >= 10


def test_selection_tamper_is_blocked(canonical_fixture: dict) -> None:
    row = canonical_fixture["inventory_rows"][0]
    selection = (
        canonical_fixture["root"]
        / f"stage1_gapvalue240_{row['package']}_upload"
        / "runs"
        / row["run_slot"]
        / row["attempt_id"]
        / "01_manifests/selection_manifest.csv"
    )
    selection.write_text(selection.read_text() + "tampered\n", encoding="utf-8")

    with pytest.raises(CanonicalInputError, match="selection SHA"):
        ingest_canonical_runs(
            canonical_fixture["root"],
            canonical_fixture["inventory"],
            canonical_fixture["matrix"],
            expected_runs=3,
            expected_triads=1,
            expected_comparisons=2,
            expected_epochs=3,
            expected_prediction_rows=5,
            expected_normal_count=3,
            expected_defect_count=2,
        )


def test_matrix_or_expected_count_mismatch_is_blocked(canonical_fixture: dict) -> None:
    matrix = pd.read_csv(canonical_fixture["matrix"])
    matrix.loc[matrix.run_slot == "RUN_002", "training_seed"] = 999
    matrix.to_csv(canonical_fixture["matrix"], index=False)

    with pytest.raises(CanonicalInputError, match="matrix"):
        ingest_canonical_runs(
            canonical_fixture["root"],
            canonical_fixture["inventory"],
            canonical_fixture["matrix"],
            expected_runs=3,
            expected_triads=1,
            expected_comparisons=2,
            expected_epochs=3,
            expected_prediction_rows=5,
            expected_normal_count=3,
            expected_defect_count=2,
        )

    with pytest.raises(CanonicalInputError, match="Expected 4 canonical runs"):
        ingest_canonical_runs(
            canonical_fixture["root"],
            canonical_fixture["inventory"],
            canonical_fixture["matrix"],
            expected_runs=4,
            expected_triads=1,
            expected_comparisons=2,
            expected_epochs=3,
            expected_prediction_rows=5,
            expected_normal_count=3,
            expected_defect_count=2,
        )
