from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from stage1_sctsr_v4.errors import ErrorCode


def _run(repository_root: Path, *arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *map(str, arguments)],
        cwd=repository_root,
        env=dict(os.environ, PYTHONPATH=str(repository_root)),
        text=True,
        capture_output=True,
    )


def test_branch_cli_exposes_identity_pool_and_parent_artifact_index(repository_root):
    result = _run(repository_root, repository_root / "scripts" / "stage1_sctsr_v4" / "run_branch.py", "--help")
    assert result.returncode == 0
    assert "--identity-pool" in result.stdout
    assert "--parent-artifact-index" in result.stdout
    assert "--execution-token" in result.stdout
    assert "--execution-claim-root" in result.stdout
    assert "--run-intent-acknowledgement" in result.stdout
    assert "--runbook-manifest" in result.stdout


def test_formal_parent_fails_before_output_without_execution_token_or_claim_registry(repository_root, tmp_path):
    receipt = tmp_path / "formal_parent_rejected.json"
    output_root = tmp_path / "must_not_exist"
    result = _run(
        repository_root,
        repository_root / "scripts" / "stage1_sctsr_v4" / "run_common_parent.py",
        "--repository-root",
        repository_root,
        "--output-root",
        output_root,
        "--training-seed",
        101,
        "--execution-mode",
        "formal",
        "--output",
        receipt,
    )

    assert result.returncode != 0
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert {"execution_token", "execution_claim_root"}.issubset(payload["error"]["observed"])
    assert {"run_intent_acknowledgement", "runbook_manifest"}.issubset(payload["error"]["observed"])
    assert not output_root.exists()


def test_formal_endpoint_model_split_and_batch_are_frozen_in_runtime_policy(repository_root):
    policy = json.loads(
        (repository_root / "configs/stage1_sctsr_v4/runtime_policy_v1.json").read_text(encoding="utf-8")
    )
    assert policy["formal_endpoint_epoch"] == 200
    assert policy["formal_endpoint_split_role"] == "val_op"
    assert policy["formal_endpoint_model_variant"] == "EMA"
    assert policy["formal_endpoint_batch_size"] == 128


def test_asset_cli_cannot_skip_registered_large_file_hashes(repository_root):
    result = _run(repository_root, repository_root / "scripts" / "stage1_sctsr_v4" / "validate_assets.py", "--help")
    assert result.returncode == 0
    assert "--skip-large-sha" not in result.stdout


def test_split_bundle_cli_only_exposes_registered_validation_roles(repository_root):
    result = _run(repository_root, repository_root / "scripts" / "stage1_sctsr_v4" / "build_split_identity_bundle.py", "--help")
    assert result.returncode == 0
    assert "{val_model,val_cal,val_op}" in result.stdout
    assert "test" not in result.stdout.lower()


def test_source_manifest_cli_uses_registered_source_roots(repository_root):
    result = _run(repository_root, repository_root / "scripts" / "stage1_sctsr_v4" / "build_source_tree_manifest.py", "--help")
    assert result.returncode == 0
    assert "--include-path" not in result.stdout
    assert "--manifest-output" in result.stdout


def test_pool_cli_cannot_accept_an_absolute_base_denominator(repository_root):
    result = _run(repository_root, repository_root / "scripts" / "stage1_sctsr_v4" / "build_identity_pools.py", "--help")
    assert result.returncode == 0
    assert "--base-denominator" not in result.stdout


def test_validate_schedule_rejects_partial_formal_matrix(repository_root, tmp_path, phase1_schedules):
    from stage1_sctsr_v4.schedule import schedule_to_dict
    from stage1_sctsr_v4.serialization import atomic_write_json

    one_schedule = tmp_path / "t_u.json"
    atomic_write_json(one_schedule, schedule_to_dict(phase1_schedules[next(key for key in phase1_schedules if key.value == "T_U")]))
    receipt = tmp_path / "receipt.json"
    result = _run(
        repository_root,
        repository_root / "scripts" / "stage1_sctsr_v4" / "validate_schedule.py",
        "--schedule",
        one_schedule,
        "--output",
        receipt,
    )
    assert result.returncode != 0
    assert json.loads(receipt.read_text(encoding="utf-8"))["error"]["code"] == ErrorCode.CONFIGURATION_MISMATCH.value


def test_build_identity_pool_rejects_placeholder_formal_hashes(repository_root, tmp_path):
    fixture = json.loads((repository_root / "tests" / "stage1_sctsr_v4" / "fixtures" / "tie_predictions.json").read_text(encoding="utf-8"))
    # The exact row values are irrelevant: trust validation must fail before
    # formal inputs are loaded or any output directory is created.
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps(fixture), encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    output_dir = tmp_path / "pool"
    result = _run(
        repository_root,
        repository_root / "scripts" / "stage1_sctsr_v4" / "build_identity_pools.py",
        "--pool",
        "T_STRESS",
        "--base-records",
        rows,
        "--t-records",
        rows,
        "--output-dir",
        output_dir,
        "--output",
        receipt,
    )
    assert result.returncode != 0
    assert json.loads(receipt.read_text(encoding="utf-8"))["error"]["code"] == ErrorCode.IDENTITY_DIGEST_MISMATCH.value
    assert not output_dir.exists()


def test_formal_identity_template_names_every_runtime_binding(repository_root):
    template = json.loads((repository_root / "configs" / "stage1_sctsr_v4" / "trainer_overrides_template_v1.json").read_text(encoding="utf-8"))
    assert {"model", "data", "device", "project", "name", "seed", "exist_ok", "resume"}.issubset(template)
    assert template["name"] == "trainer"
    assert template["exist_ok"] is False
    assert template["resume"] is False


def test_run_intent_control_plane_has_explicit_build_and_validation_clis(repository_root):
    runbook = _run(
        repository_root,
        repository_root / "scripts" / "stage1_sctsr_v4" / "build_runbook_manifest.py",
        "--help",
    )
    acknowledgement = _run(
        repository_root,
        repository_root / "scripts" / "stage1_sctsr_v4" / "build_run_intent_acknowledgement.py",
        "--help",
    )
    assert runbook.returncode == 0
    assert "--document" in runbook.stdout
    assert "--manifest-output" in runbook.stdout
    assert acknowledgement.returncode == 0
    assert "--acknowledge-all-required-statements" in acknowledgement.stdout
    assert "--acknowledgement-output" in acknowledgement.stdout
    assert "--execution-token" in acknowledgement.stdout
