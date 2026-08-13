from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_v3_baseline_audit.py")
REPOSITORY_ROOT = next(parent for parent in MODULE_PATH.parents if (parent / ".git").exists())
BASELINE = "a70ba60485dd32c2f8b4268b8f28ea2d3549f42f"
FROZEN_SOURCE = "e9b6df61b0eb02e1d32c29175644f1c2af545afc"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_v3_baseline_audit", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pytest_output_parsing_distinguishes_passed_collected_and_skipped():
    module = _load_module()

    summary = module.parse_pytest_run_summary("183 passed, 1 skipped in 4.76s\n")
    collected = module.parse_collected_test_ids(
        "tests/x.py::test_a\n"
        "tests/x.py::test_b[value]\n"
        "2 tests collected in 0.03s\n"
    )

    assert summary == {"passed": 183, "failed": 0, "skipped": 1, "errors": 0}
    assert collected == ["tests/x.py::test_a", "tests/x.py::test_b[value]"]


def test_frozen_v3_source_and_test_trees_equal_the_registered_baseline():
    module = _load_module()

    snapshot = module.collect_git_snapshot(
        REPOSITORY_ROOT,
        baseline_commit=BASELINE,
        frozen_source_commit=FROZEN_SOURCE,
    )

    assert snapshot["protected_diff_paths"] == []
    assert snapshot["source_tree"]["baseline"] == snapshot["source_tree"]["frozen_source"]
    assert snapshot["test_tree"]["baseline"] == snapshot["test_tree"]["frozen_source"]
    assert snapshot["current_python_test_file_count"] == 34
    assert snapshot["historical_max_python_test_file_count"] == 34


def test_taskbook_count_contradiction_is_not_reported_as_acceptance_pass():
    module = _load_module()

    assessment = module.assess_counts(
        required_passed=231,
        current_python_test_file_count=34,
        historical_max_python_test_file_count=34,
        collected_test_ids=[f"tests/x.py::test_{index}" for index in range(184)],
        run_summary={"passed": 183, "failed": 0, "skipped": 1, "errors": 0},
        protected_tree_equal=True,
    )

    assert assessment["audit_truth_status"] == "PASS"
    assert assessment["sa266_status"] == "FAIL_TASKBOOK_COUNT_CONTRADICTION"
    assert assessment["missing_passes_to_taskbook_requirement"] == 48
    assert assessment["recoverable_missing_tracked_tests_detected"] is False
    assert assessment["implementation_regression_detected"] is False


def test_modified_protected_tree_fails_the_identity_audit_even_if_count_is_high():
    module = _load_module()

    assessment = module.assess_counts(
        required_passed=231,
        current_python_test_file_count=80,
        historical_max_python_test_file_count=80,
        collected_test_ids=[f"tests/x.py::test_{index}" for index in range(240)],
        run_summary={"passed": 240, "failed": 0, "skipped": 0, "errors": 0},
        protected_tree_equal=False,
    )

    assert assessment["audit_truth_status"] == "FAIL_PROTECTED_TREE_CHANGED"
    assert assessment["implementation_regression_detected"] is True
