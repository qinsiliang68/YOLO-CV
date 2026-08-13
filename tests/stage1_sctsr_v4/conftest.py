from __future__ import annotations

import os
import sys
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("SCTSR_ALLOW_SYNTHETIC_COLUMNAR_FALLBACK", "1")

@pytest.fixture(scope="session")
def repository_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def synthetic_fixture():
    from stage1_sctsr_v4.identity_pool import IdentityRecord, T_SELECTION_SEMANTIC, make_pool, partition_five_groups
    from stage1_sctsr_v4.random_controls import build_r1_global_random, build_r2_matched_random
    from stage1_sctsr_v4.terminal_field_guard import TerminalFieldGuard

    denominator = 2_000
    seed = 20260812
    buckets = ("EASY", "BOUNDARY", "FORGETTING", "PERSISTENT_ERROR")
    base = tuple(
        IdentityRecord(
            sample_id=f"SYN_{index:06d}",
            y_true=index % 2,
            replay_role="NORMAL_REPLAY" if index % 2 == 0 else "DEFECT_GUARD_REPLAY",
            historical_dynamic_bucket=buckets[(index // 40) % len(buckets)],
            oof_fold=(index // 2) % 10,
            oof_group_id=f"bucket_{(index // 20) % 20:02d}",
            group_source="numeric_filename_bucket",
            base_manifest_membership=True,
        )
        for index in range(denominator)
    )
    by_stratum = {}
    for record in base:
        by_stratum.setdefault(record.stratum(), []).append(record)
    chosen = []
    ordered_strata = sorted(by_stratum)
    cursor = 0
    while len(chosen) < 50:
        stratum = ordered_strata[cursor % len(ordered_strata)]
        ranked = sorted(
            by_stratum[stratum],
            key=lambda record: (
                hashlib.sha256(f"T_STRESS\0{seed}\0{record.sample_id}".encode()).hexdigest(),
                record.sample_id,
            ),
        )
        rank = cursor // len(ordered_strata)
        if rank < len(ranked) - 1:
            chosen.append(ranked[rank])
        cursor += 1
    t_pool = make_pool(
        pool_id="SYNTHETIC_T_STRESS",
        pool_role="T_STRESS",
        records=chosen,
        base_denominator=denominator,
        base_manifest_sha256="1" * 64,
        source_manifest_path="SYNTHETIC_T_CANONICAL",
        source_manifest_sha256="2" * 64,
        construction_seed=None,
        selection_semantic=T_SELECTION_SEMANTIC,
    )
    r1 = build_r1_global_random(
        base,
        base_denominator=denominator,
        base_manifest_sha256="1" * 64,
        source_manifest_sha256="2" * 64,
        selection_seed=seed + 1,
        t_ids={record.sample_id for record in t_pool.records},
    )
    rows = [
        {
            "sample_id": record.sample_id,
            "y_true": record.y_true,
            "replay_role": record.replay_role,
            "historical_dynamic_bucket": record.historical_dynamic_bucket,
            "oof_fold": record.oof_fold,
            "oof_group_id": record.oof_group_id,
            "group_source": record.group_source,
            "base_manifest_membership": record.base_manifest_membership,
            "loss": 999.0,
            "RHO": 999.0,
        }
        for record in base
    ]
    r2 = build_r2_matched_random(
        rows,
        t_pool=t_pool,
        base_denominator=denominator,
        base_manifest_sha256="1" * 64,
        source_manifest_sha256="2" * 64,
        selection_seed=seed + 2,
        guard=TerminalFieldGuard(),
    )
    return SimpleNamespace(
        training_seed=seed,
        base_denominator=denominator,
        base_records=base,
        base_ids=tuple(record.sample_id for record in base),
        t_pool=t_pool,
        r1_result=r1,
        r2_result=r2,
        groups_by_pool={
            "T": partition_five_groups(t_pool, base_denominator=denominator),
            "R1": partition_five_groups(r1.pool, base_denominator=denominator),
            "R2": partition_five_groups(r2.pool, base_denominator=denominator),
        },
    )


@pytest.fixture
def phase1_schedules(synthetic_fixture):
    from stage1_sctsr_v4.arm_spec import ArmId
    from stage1_sctsr_v4.schedule import build_schedule

    fixture = synthetic_fixture
    t = fixture.groups_by_pool["T"]
    r1 = fixture.groups_by_pool["R1"]
    r2 = fixture.groups_by_pool["R2"]
    denominator = fixture.base_denominator
    return {
        ArmId.NR: build_schedule(ArmId.NR, primary_groups=None, primary_digest="NONE", base_denominator=denominator),
        ArmId.R1_U: build_schedule(ArmId.R1_U, primary_groups=r1, primary_digest=fixture.r1_result.pool.spec.identity_digest, base_denominator=denominator),
        ArmId.R2_U: build_schedule(ArmId.R2_U, primary_groups=r2, primary_digest=fixture.r2_result.pool.spec.identity_digest, base_denominator=denominator),
        ArmId.T_U: build_schedule(ArmId.T_U, primary_groups=t, primary_digest=fixture.t_pool.spec.identity_digest, base_denominator=denominator),
        ArmId.R2_F: build_schedule(ArmId.R2_F, primary_groups=r2, primary_digest=fixture.r2_result.pool.spec.identity_digest, base_denominator=denominator),
        ArmId.T_F: build_schedule(ArmId.T_F, primary_groups=t, primary_digest=fixture.t_pool.spec.identity_digest, base_denominator=denominator),
        ArmId.T_TO_R2_AT_160: build_schedule(ArmId.T_TO_R2_AT_160, primary_groups=t, primary_digest=fixture.t_pool.spec.identity_digest, fallback_groups=r2, fallback_digest=fixture.r2_result.pool.spec.identity_digest, base_denominator=denominator),
        ArmId.T_TO_NR_AT_160: build_schedule(ArmId.T_TO_NR_AT_160, primary_groups=t, primary_digest=fixture.t_pool.spec.identity_digest, base_denominator=denominator),
    }


@pytest.fixture(scope="session")
def canary_root(tmp_path_factory: pytest.TempPathFactory, repository_root: Path) -> Path:
    from stage1_sctsr_v4.synthetic_canary import run_synthetic_canary

    root = tmp_path_factory.mktemp("sctsr_canary") / "run"
    result = run_synthetic_canary(root, repository_root=repository_root, training_seed=20260812)
    assert result["status"] == "PASS"
    return root


@pytest.fixture
def prediction_rows():
    from stage1_sctsr_v4.prediction_artifact import PredictionRow

    rows = []
    for i in range(240):
        y = i % 2
        # Deliberate ties every four rows.
        p = round(((i * 7) % 101) / 100, 2)
        rows.append(PredictionRow(
            run_id="R", arm_id="T_U", training_seed=1, split_role="synthetic",
            split_manifest_path="synthetic.json", split_manifest_sha256="A" * 64,
            sample_id=f"S{i:04d}", y_true=y, logit_normal=1.0-p, logit_defect=p,
            p_defect_raw=p, checkpoint_epoch=200, checkpoint_sha256="B" * 64,
            model_variant="EMA", source_tree_digest="C" * 64, prediction_generation=1,
        ))
    return tuple(rows)
