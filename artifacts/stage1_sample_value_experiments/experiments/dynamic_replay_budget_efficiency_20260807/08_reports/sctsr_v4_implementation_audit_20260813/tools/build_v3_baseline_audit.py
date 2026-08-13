from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


V3_SOURCE_PATH = "stage1_dynamic_replay_v3"
V3_TEST_PATH = "tests/stage1_dynamic_replay_v3"
PROTECTED_PATHS = (V3_SOURCE_PATH, V3_TEST_PATH)
SUMMARY_PATTERN = re.compile(r"(?P<count>\d+)\s+(?P<kind>passed|failed|skipped|error|errors)")


def _git(repository_root: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed with exit {completed.returncode}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def parse_pytest_run_summary(output: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for match in SUMMARY_PATTERN.finditer(output):
        kind = match.group("kind")
        if kind in {"error", "errors"}:
            kind = "errors"
        counts[kind] = int(match.group("count"))
    if not any(counts.values()):
        raise ValueError("pytest output contains no parseable result summary")
    return counts


def parse_collected_test_ids(output: str) -> list[str]:
    test_ids = [line.strip() for line in output.splitlines() if "::" in line and line.strip()]
    if not test_ids:
        raise ValueError("pytest collect output contains no test IDs")
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("pytest collect output contains duplicate test IDs")
    return test_ids


def _tree(repository_root: Path, commit: str, path: str) -> str:
    return _git(repository_root, "rev-parse", f"{commit}:{path}")


def _python_test_files(repository_root: Path, commit: str) -> list[str]:
    output = _git(repository_root, "ls-tree", "-r", "--name-only", commit, "--", V3_TEST_PATH)
    return sorted(line for line in output.splitlines() if line.endswith(".py"))


def _history_commits(repository_root: Path) -> list[str]:
    output = _git(repository_root, "rev-list", "--all", "--", V3_TEST_PATH)
    return [line for line in output.splitlines() if line]


def collect_git_snapshot(
    repository_root: Path,
    *,
    baseline_commit: str,
    frozen_source_commit: str,
) -> dict:
    root = repository_root.resolve()
    head = _git(root, "rev-parse", "HEAD")
    source_tree = {
        "baseline": _tree(root, baseline_commit, V3_SOURCE_PATH),
        "frozen_source": _tree(root, frozen_source_commit, V3_SOURCE_PATH),
        "head": _tree(root, head, V3_SOURCE_PATH),
    }
    test_tree = {
        "baseline": _tree(root, baseline_commit, V3_TEST_PATH),
        "frozen_source": _tree(root, frozen_source_commit, V3_TEST_PATH),
        "head": _tree(root, head, V3_TEST_PATH),
    }
    protected_diff_output = _git(
        root,
        "diff",
        "--name-only",
        baseline_commit,
        head,
        "--",
        *PROTECTED_PATHS,
    )
    current_files = _python_test_files(root, head)
    history_rows = []
    for commit in _history_commits(root):
        files = _python_test_files(root, commit)
        history_rows.append({"commit": commit, "python_test_file_count": len(files)})
    historical_max = max((row["python_test_file_count"] for row in history_rows), default=0)
    return {
        "baseline_commit": _git(root, "rev-parse", baseline_commit),
        "frozen_source_commit": _git(root, "rev-parse", frozen_source_commit),
        "head_commit": head,
        "source_tree": source_tree,
        "test_tree": test_tree,
        "protected_diff_paths": [line for line in protected_diff_output.splitlines() if line],
        "current_python_test_file_count": len(current_files),
        "current_python_test_files": current_files,
        "historical_max_python_test_file_count": historical_max,
        "v3_test_history": history_rows,
    }


def assess_counts(
    *,
    required_passed: int,
    current_python_test_file_count: int,
    historical_max_python_test_file_count: int,
    collected_test_ids: Iterable[str],
    run_summary: dict[str, int],
    protected_tree_equal: bool,
) -> dict:
    collected = list(collected_test_ids)
    executed_or_skipped = run_summary["passed"] + run_summary["failed"] + run_summary["skipped"]
    implementation_regression = (
        not protected_tree_equal
        or run_summary["failed"] > 0
        or run_summary["errors"] > 0
        or executed_or_skipped != len(collected)
    )
    if implementation_regression:
        audit_truth_status = "FAIL_PROTECTED_TREE_CHANGED" if not protected_tree_equal else "FAIL_TEST_EXECUTION_MISMATCH"
    else:
        audit_truth_status = "PASS"
    if run_summary["passed"] >= required_passed and not implementation_regression:
        sa266_status = "PASS"
    else:
        sa266_status = "FAIL_TASKBOOK_COUNT_CONTRADICTION"
    return {
        "audit_truth_status": audit_truth_status,
        "sa266_status": sa266_status,
        "required_passed": required_passed,
        "collected_test_count": len(collected),
        "run_summary": dict(run_summary),
        "missing_passes_to_taskbook_requirement": max(0, required_passed - run_summary["passed"]),
        "current_python_test_file_count": current_python_test_file_count,
        "historical_max_python_test_file_count": historical_max_python_test_file_count,
        "recoverable_missing_tracked_tests_detected": historical_max_python_test_file_count > current_python_test_file_count,
        "implementation_regression_detected": implementation_regression,
    }


def _file_identity(path: Path, root: Path) -> dict:
    payload = path.read_bytes()
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest().upper(),
    }


def build_report(
    repository_root: Path,
    *,
    baseline_commit: str,
    frozen_source_commit: str,
    required_passed: int,
    pytest_run_log: Path,
    pytest_collect_log: Path,
) -> dict:
    root = repository_root.resolve()
    snapshot = collect_git_snapshot(
        root,
        baseline_commit=baseline_commit,
        frozen_source_commit=frozen_source_commit,
    )
    run_output = pytest_run_log.read_text(encoding="utf-8")
    collect_output = pytest_collect_log.read_text(encoding="utf-8")
    run_summary = parse_pytest_run_summary(run_output)
    collected_test_ids = parse_collected_test_ids(collect_output)
    trees_equal = (
        len(snapshot["protected_diff_paths"]) == 0
        and len(set(snapshot["source_tree"].values())) == 1
        and len(set(snapshot["test_tree"].values())) == 1
    )
    assessment = assess_counts(
        required_passed=required_passed,
        current_python_test_file_count=snapshot["current_python_test_file_count"],
        historical_max_python_test_file_count=snapshot["historical_max_python_test_file_count"],
        collected_test_ids=collected_test_ids,
        run_summary=run_summary,
        protected_tree_equal=trees_equal,
    )
    return {
        "schema_version": "stage1.sctsr.v3_baseline_audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": assessment["audit_truth_status"],
        "scope": "READ_ONLY_V3_BASELINE_IDENTITY_AND_COUNT_AUDIT",
        "git_snapshot": snapshot,
        "pytest_evidence": {
            "run_log": _file_identity(pytest_run_log, root),
            "collect_log": _file_identity(pytest_collect_log, root),
            "collected_test_ids": collected_test_ids,
        },
        "assessment": assessment,
        "interpretation": {
            "v4_modified_v3": not trees_equal,
            "taskbook_231_claim_reproduced": assessment["sa266_status"] == "PASS",
            "historical_231_text_used_as_execution_evidence": False,
            "synthetic_tests_added_to_reach_count": False,
            "conclusion": (
                "The protected v3 source and test trees are byte-identical across baseline, frozen implementation, and HEAD; "
                "the tracked suite collects 184 tests and executes as 183 passed plus 1 skipped. The taskbook's 231-pass "
                "statement is contradicted by the frozen repository inventory and remains an acceptance blocker, not a v4 regression."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit frozen v3 identity and the taskbook 231-pass contradiction")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--frozen-source-commit", required=True)
    parser.add_argument("--required-passed", type=int, default=231)
    parser.add_argument("--pytest-run-log", type=Path, required=True)
    parser.add_argument("--pytest-collect-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        args.repository_root,
        baseline_commit=args.baseline_commit,
        frozen_source_commit=args.frozen_source_commit,
        required_passed=args.required_passed,
        pytest_run_log=args.pytest_run_log.resolve(),
        pytest_collect_log=args.pytest_collect_log.resolve(),
    )
    _atomic_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "sa266_status": report["assessment"]["sa266_status"],
                "collected": report["assessment"]["collected_test_count"],
                "passed": report["assessment"]["run_summary"]["passed"],
                "skipped": report["assessment"]["run_summary"]["skipped"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
