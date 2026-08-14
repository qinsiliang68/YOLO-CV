from __future__ import annotations

from collections import Counter
import json
import os
import subprocess
import sys

import pytest

from stage1_sctsr_v4.asset_registry import load_asset_registry
from stage1_sctsr_v4.formal_pool_inputs import load_formal_pool_inputs
from stage1_sctsr_v4.identity_pool import IdentityRecord
from stage1_sctsr_v4.r2_specification_audit import (
    audit_r2_specifications,
    build_minimum_displacement_candidate,
)
from stage1_sctsr_v4.serialization import sha256_file, stable_digest


@pytest.fixture(scope="module")
def registered_inputs(repository_root):
    registry = load_asset_registry(repository_root / "configs/stage1_sctsr_v4/asset_registry_v1.json")
    return load_formal_pool_inputs(registry, repository_root)


def test_minimum_displacement_candidate_is_unique_zero_overlap_and_coarse_exact():
    base = (
        IdentityRecord("t0", 0, "normal_replay", "hard", 0, "g0"),
        IdentityRecord("t1", 0, "normal_replay", "hard", 0, "g0"),
        IdentityRecord("c0", 0, "normal_replay", "hard", 0, "g0"),
        IdentityRecord("c1", 0, "normal_replay", "hard", 0, "g1"),
        IdentityRecord("c2", 0, "normal_replay", "hard", 0, "g1"),
    )
    treatment = base[:2]

    candidate = build_minimum_displacement_candidate(base, treatment, selection_seed=17)

    assert len(candidate.records) == 2
    assert len({row.sample_id for row in candidate.records}) == 2
    assert not ({row.sample_id for row in candidate.records} & {row.sample_id for row in treatment})
    assert Counter(row.stratum()[:3] for row in candidate.records) == Counter(row.stratum()[:3] for row in treatment)
    assert candidate.minimum_displacement_count == 1
    assert candidate.observed_displacement_count == 1


def test_registered_r2_audit_reproduces_shortage_and_compares_falsifiable_options(registered_inputs):
    audit = audit_r2_specifications(
        registered_inputs.base_records,
        registered_inputs.t_pool.records,
        selection_seed=20260812,
    )

    assert audit["status"] == "SPECIFICATION_CHANGE_REQUIRED"
    assert audit["treatment"]["unique_count"] == 3000
    assert audit["strict_exact"]["shortage_strata"] == 172
    assert audit["strict_exact"]["shortage_occurrences"] == 378
    assert audit["strict_exact"]["zero_available_strata"] == 30
    assert audit["strict_exact"]["zero_available_occurrences"] == 61
    assert audit["options"]["exact_with_replacement"]["status"] == "INFEASIBLE"
    assert audit["options"]["matchable_t_subset"]["unique_count"] == 2622
    assert audit["options"]["drop_group_hash_random"]["group_total_variation"] == pytest.approx(0.3923333333333333)
    proposed = audit["options"]["minimum_displacement_zero_overlap"]
    assert proposed["status"] == "FEASIBLE_REQUIRES_PREREGISTERED_SPEC_CHANGE"
    assert proposed["unique_count"] == 3000
    assert proposed["overlap_with_t_count"] == 0
    assert proposed["exact_label_dynamic_fold"] is True
    assert proposed["joint_displacement_count"] == 378
    assert proposed["group_total_variation"] == pytest.approx(0.126)
    assert audit["recommended_option"] == "minimum_displacement_zero_overlap"


def test_r2_audit_cli_writes_a_byte_bound_report_without_pool_side_effect(repository_root, tmp_path):
    report = tmp_path / "R2_SPECIFICATION_AUDIT.json"
    receipt = tmp_path / "R2_SPECIFICATION_AUDIT_RECEIPT.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "stage1_sctsr_v4" / "audit_r2_specification.py"),
            "--repository-root",
            str(repository_root),
            "--asset-registry",
            str(repository_root / "configs" / "stage1_sctsr_v4" / "asset_registry_v1.json"),
            "--selection-seed",
            "20260812",
            "--audit-output",
            str(report),
            "--output",
            str(receipt),
        ],
        cwd=repository_root,
        env=dict(os.environ, PYTHONPATH=str(repository_root)),
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    audit = json.loads(report.read_text(encoding="utf-8"))
    cli = json.loads(receipt.read_text(encoding="utf-8"))
    assert audit["audit_digest"] == stable_digest({key: value for key, value in audit.items() if key != "audit_digest"})
    assert cli["status"] == "PASS"
    assert cli["result"]["audit_sha256"] == sha256_file(report)
    assert cli["result"]["audit_digest"] == audit["audit_digest"]
    assert not (tmp_path / "identity_pools").exists()
