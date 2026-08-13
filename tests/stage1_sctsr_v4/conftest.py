from __future__ import annotations

import os
import sys
from pathlib import Path

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
    from stage1_sctsr_v4.synthetic_fixture import build_synthetic_fixture

    return build_synthetic_fixture(training_seed=20260812)


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
