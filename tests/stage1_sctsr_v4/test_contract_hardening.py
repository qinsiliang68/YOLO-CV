from __future__ import annotations

import json

import pytest

from stage1_sctsr_v4.contracts import (
    require_synthetic_or_authorized,
    validate_contract_mapping,
)
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.serialization import load_json


def _contract(repository_root):
    return load_json(repository_root / "configs/stage1_sctsr_v4/contract_v1.json")


def _arms(repository_root):
    return load_json(repository_root / "configs/stage1_sctsr_v4/arms_phase1_v1.json")["arms"]


def test_sa_030_forged_nonempty_release_signature_is_rejected(tmp_path):
    release = {
        "schema_version": "stage1.sctsr.formal_release.v1",
        "authorization": "SIGNED_SCTSR_V4_FORMAL_RELEASE",
        "formal_training_authorized": True,
        "release_id": "FORGED_RELEASE",
        "signature": "x",
        "baseline_main_commit": "a70ba60485dd32c2f8b4268b8f28ea2d3549f42f",
        "taskbook_blob_sha": "b201d021712e9c6614e119d35f0e14bdf405c6be",
    }
    path = tmp_path / "forged_release.json"
    path.write_text(json.dumps(release), encoding="utf-8")

    with pytest.raises(SctsrError) as exc:
        require_synthetic_or_authorized("formal", path)

    assert exc.value.code is ErrorCode.FORMAL_RELEASE_NOT_AUTHORIZED


def test_sa_031_contract_baseline_identity_is_immutable(repository_root):
    contract = _contract(repository_root)
    contract["baseline"]["commit"] = "0" * 40

    with pytest.raises(SctsrError) as exc:
        validate_contract_mapping(contract)

    assert exc.value.code is ErrorCode.CONFIGURATION_MISMATCH


def test_sa_032_absolute_budget_is_rejected_in_contract(repository_root):
    contract = _contract(repository_root)
    contract["rates"]["uniform"]["absolute_count"] = 600

    with pytest.raises(SctsrError) as exc:
        validate_contract_mapping(contract)

    assert exc.value.code is ErrorCode.ABSOLUTE_BUDGET_FORBIDDEN


def test_sa_033_absolute_budget_is_rejected_in_arm(repository_root):
    contract = _contract(repository_root)
    arms = _arms(repository_root)
    arms[1]["replay_count"] = 600

    with pytest.raises(SctsrError) as exc:
        validate_contract_mapping(contract, arms)

    assert exc.value.code is ErrorCode.ABSOLUTE_BUDGET_FORBIDDEN
