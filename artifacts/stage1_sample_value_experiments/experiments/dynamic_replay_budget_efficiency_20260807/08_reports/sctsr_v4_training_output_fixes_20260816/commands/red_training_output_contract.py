from __future__ import annotations

import inspect
import subprocess
from dataclasses import fields
from pathlib import Path

from stage1_sctsr_v4 import dataset_adapter
from stage1_sctsr_v4 import formal_cli
from stage1_sctsr_v4 import formal_execution
from stage1_sctsr_v4 import prediction_runtime
from stage1_sctsr_v4 import random_controls
from stage1_sctsr_v4 import source_identity
from stage1_sctsr_v4 import training_system


def test_tc01_r2_builder_requires_content_identity():
    parameters = inspect.signature(random_controls.build_r2_matched_random).parameters
    assert "content_sha256_by_sample_id" in parameters


def test_tc02_endpoint_records_and_binds_actual_input_bytes():
    record_fields = {field.name for field in fields(prediction_runtime.RegisteredImageRecord)}
    assert {"image_bytes", "image_sha256"} <= record_fields
    assert hasattr(prediction_runtime, "build_endpoint_input_binding")
    parameters = inspect.signature(prediction_runtime.build_endpoint_input_binding).parameters
    assert {"dataset_root", "content_ledger_sha256", "content_ledger_identity_digest"} <= set(parameters)


def test_tc03_materialized_loader_has_terminal_revalidator():
    assert hasattr(dataset_adapter, "revalidate_materialized_dataset_binding")


def test_tc04_finalization_has_one_fenced_entrypoint():
    assert hasattr(formal_execution, "execute_fenced_finalization")


def test_tc05_execution_lease_has_terminal_failure_transition():
    assert hasattr(formal_execution, "mark_execution_failed")


def test_tc06_logical_job_is_storage_independent_and_namespaced():
    invariant_fields = set(formal_execution.LOGICAL_JOB_INVARIANT_FIELDS)
    assert "output_root_digest" not in invariant_fields
    parameters = inspect.signature(formal_execution.logical_job_digest).parameters
    assert {"experiment_id", "release_id"} <= set(parameters)


def test_tc07_validation_can_reprobe_live_runtime():
    assert hasattr(source_identity, "probe_runtime_environment")


def test_tc08_imported_adapter_has_an_origin_validator():
    assert hasattr(training_system, "validate_sctsr_adapter_import")


def test_tc09_binary_contract_exists_before_first_step():
    assert hasattr(formal_cli, "validate_binary_classification_contract")


def test_tc10_registered_evidence_text_is_forced_to_lf():
    root = Path(__file__).resolve().parent
    target = "docs/stage1_sctsr_v4/tdd_receipts/commit_01/RED.txt"
    result = subprocess.run(
        ["git", "check-attr", "eol", "--", target],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip().endswith("eol: lf")
