from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from stage1_sctsr_v4.checkpointing import build_checkpoint_payload, save_checkpoint_atomic
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.fixed_step_runtime import ExponentialMovingAverage
from stage1_sctsr_v4.formal_cli import load_formal_identity
from stage1_sctsr_v4.run_validation import build_artifact_index, validate_run_tree
from stage1_sctsr_v4.serialization import atomic_write_json, sha256_file


class _Scaler:
    def state_dict(self):
        return {}


def _run(repository_root: Path, *arguments: object) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ, PYTHONPATH=str(repository_root))
    return subprocess.run(
        [sys.executable, *map(str, arguments)],
        cwd=repository_root,
        env=environment,
        text=True,
        capture_output=True,
    )


def _checkpoint(path: Path) -> str:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    payload = build_checkpoint_payload(
        model=model,
        ema=ExponentialMovingAverage.from_model(model),
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=_Scaler(),
        epoch=200,
        global_step=187_600,
        base_sampler_generation=200,
        canonical_training_lock_sha256="A" * 64,
        initial_checkpoint_sha256="B" * 64,
        base_manifest_sha256="C" * 64,
        training_seed=1,
        source_tree_digest="D" * 64,
        runtime_config_digest="E" * 64,
        asset_registry_digest="F" * 64,
    )
    return save_checkpoint_atomic(path, payload)


def test_validate_run_rejects_indexed_but_semantically_empty_shell(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    atomic_write_json(
        root / "RUN_MANIFEST.json",
        {
            "schema_version": "stage1.sctsr.synthetic_run_manifest.v1",
            "semantic": "SYNTHETIC_NOT_SCIENTIFIC_RESULT",
            "scientific_result": False,
        },
    )
    atomic_write_json(root / "ARTIFACT_INDEX.json", build_artifact_index(root))
    with pytest.raises(SctsrError) as caught:
        validate_run_tree(root)
    assert caught.value.code is ErrorCode.CLOSEOUT_NOT_VALIDATED


def test_validate_run_detects_missing_required_canary_partition_even_after_reindex(tmp_path, canary_root):
    root = tmp_path / "tampered"
    shutil.copytree(canary_root, root)
    target = next((root / "04_ledgers" / "occurrence").rglob("*.parquet"))
    target.unlink()
    atomic_write_json(root / "ARTIFACT_INDEX.json", build_artifact_index(root))
    with pytest.raises(SctsrError) as caught:
        validate_run_tree(root)
    assert caught.value.code is ErrorCode.CLOSEOUT_NOT_VALIDATED


def test_evaluate_cli_rejects_arbitrary_raw_json_even_with_loadable_checkpoint(repository_root, tmp_path, prediction_rows):
    checkpoint = tmp_path / "epoch_0200.pt"
    checkpoint_sha = _checkpoint(checkpoint)
    rows = tuple(
        replace(
            row,
            checkpoint_sha256=checkpoint_sha,
            source_tree_digest="D" * 64,
        )
        for row in prediction_rows[:8]
    )
    raw_predictions = tmp_path / "raw_predictions.json"
    raw_predictions.write_text(json.dumps([asdict(row) for row in rows]), encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    result = _run(
        repository_root,
        repository_root / "scripts" / "stage1_sctsr_v4" / "evaluate_checkpoint.py",
        "--checkpoint",
        checkpoint,
        "--checkpoint-epoch",
        200,
        "--mode",
        "synthetic",
        "--predictions",
        raw_predictions,
        "--frontier-output",
        tmp_path / "frontier.json",
        "--output",
        receipt,
    )
    assert result.returncode != 0
    assert json.loads(receipt.read_text(encoding="utf-8"))["error"]["code"] == ErrorCode.PREDICTION_IDENTITY_MISMATCH.value


def test_evaluate_cli_revalidates_registered_prediction_and_writes_partitioned_frontier(repository_root, tmp_path, canary_root):
    run_id = "SYNTH_T_U_20260812"
    predictions = canary_root / "06_predictions" / f"run_id={run_id}" / "epoch=0200" / "predictions.parquet"
    diagnostic = json.loads(
        (canary_root / "07_evaluation" / f"run_id={run_id}" / "epoch=0200" / "unreachable_target_diagnostic.json").read_text(encoding="utf-8")
    )
    prediction_summary = tmp_path / "prediction_summary.json"
    atomic_write_json(prediction_summary, diagnostic["prediction_summary"])
    checkpoint = canary_root / "05_checkpoints" / f"{run_id}_e200.pt"
    frontier = tmp_path / "run_id=EVAL_SYNTH_T_U" / "epoch=0200" / "frontier.parquet"
    receipt = tmp_path / "evaluate_receipt.json"
    result = _run(
        repository_root,
        repository_root / "scripts" / "stage1_sctsr_v4" / "evaluate_checkpoint.py",
        "--checkpoint",
        checkpoint,
        "--checkpoint-epoch",
        200,
        "--mode",
        "synthetic",
        "--predictions",
        predictions,
        "--prediction-summary",
        prediction_summary,
        "--frontier-output",
        frontier,
        "--output",
        receipt,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["result"]["frontier_row_count"] == 96
    assert frontier.is_file()


def test_formal_runner_cli_exposes_trust_and_binding_arguments(repository_root):
    for script in ("run_common_parent.py", "run_branch.py"):
        result = _run(repository_root, repository_root / "scripts" / "stage1_sctsr_v4" / script, "--help")
        assert result.returncode == 0
        assert "--release-trust-policy" in result.stdout
        assert "--contract" in result.stdout
        assert "--asset-registry" in result.stdout
        assert "--seed-registry" in result.stdout


def test_formal_closeout_rejects_missing_execution_attempt_snapshot(tmp_path):
    from stage1_sctsr_v4.run_validation import _validate_formal_tree

    manifest = {
        "schema_version": "stage1.sctsr.formal_run_manifest.v2",
        "execution_mode": "formal",
        "formal_training_authorized": True,
        "formal_training_started": True,
        "engineering_gate_generated": False,
        "assignments_generated": False,
        "pilot_release_generated": False,
        "blind_holdout_opened": False,
        "selector_trained": False,
        "method_effectiveness_claimed": False,
        "test_accessed": False,
        "best_pt_used": False,
        "run_role": "COMMON_PARENT",
    }

    with pytest.raises(SctsrError) as caught:
        _validate_formal_tree(tmp_path, manifest)

    assert caught.value.code is ErrorCode.CLOSEOUT_NOT_VALIDATED
    assert "execution attempt" in str(caught.value).lower()


def test_formal_identity_template_cannot_be_loaded_as_a_prepared_identity(repository_root):
    with pytest.raises(SctsrError) as caught:
        load_formal_identity(repository_root / "configs" / "stage1_sctsr_v4" / "formal_identity_template_v1.json")
    assert caught.value.code is ErrorCode.CONFIGURATION_MISMATCH


def test_closeout_rejects_synthetic_canary_as_full_implementation_completion(repository_root, tmp_path, canary_root):
    receipt = tmp_path / "closeout_cli.json"
    completion = tmp_path / "completion.json"
    result = _run(
        repository_root,
        repository_root / "scripts" / "stage1_sctsr_v4" / "closeout_run.py",
        "--run-root",
        canary_root,
        "--completion-output",
        completion,
        "--output",
        receipt,
    )
    assert result.returncode != 0
    assert json.loads(receipt.read_text(encoding="utf-8"))["error"]["code"] == ErrorCode.CLOSEOUT_NOT_VALIDATED.value


def test_closeout_requires_the_detailed_taskbook_self_audit(repository_root):
    result = _run(repository_root, repository_root / "scripts" / "stage1_sctsr_v4" / "closeout_run.py", "--help")
    assert result.returncode == 0
    assert "--detailed-self-audit" in result.stdout
    assert "--taskbook" in result.stdout
