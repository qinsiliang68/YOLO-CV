from __future__ import annotations

from collections import Counter
from dataclasses import asdict

import pytest

from stage1_sctsr_v4.asset_registry import load_asset_registry
from stage1_sctsr_v4.dataset_content_ledger import load_registered_dataset_content_map
from stage1_sctsr_v4.errors import ErrorCode, SctsrError
from stage1_sctsr_v4.formal_pool_inputs import (
    R2_APPROVED_IDENTITY_DIGEST,
    R2_APPROVED_SELECTION_SEED,
    build_registered_r2,
    load_formal_pool_inputs,
)
from stage1_sctsr_v4.identity_pool import IdentityRecord, make_pool
from stage1_sctsr_v4.random_controls import (
    R2_MINIMUM_OOF_GROUP_DISPLACEMENT_POLICY,
    build_r2_matched_random,
)
from stage1_sctsr_v4.r2_addendum import (
    R2_APPROVED_CONTENT_MAP_DIGEST,
    validate_approved_r2_build,
    validate_r2_matching_policy_mapping,
)
from stage1_sctsr_v4.serialization import load_json, sha256_file, stable_digest
from stage1_sctsr_v4.terminal_field_guard import TerminalFieldGuard


def _mapping(record: IdentityRecord) -> dict[str, object]:
    return {
        "sample_id": record.sample_id,
        "y_true": record.y_true,
        "replay_role": record.replay_role,
        "historical_dynamic_bucket": record.historical_dynamic_bucket,
        "oof_fold": record.oof_fold,
        "oof_group_id": record.oof_group_id,
        "group_source": record.group_source,
        "base_manifest_membership": record.base_manifest_membership,
    }


def test_addendum_relaxes_only_oof_group_and_records_every_displacement():
    records = (
        IdentityRecord("t0", 0, "normal_replay", "hard", 0, "g0"),
        IdentityRecord("t1", 0, "normal_replay", "hard", 0, "g0"),
        IdentityRecord("c0", 0, "normal_replay", "hard", 0, "g0"),
        IdentityRecord("c1", 0, "normal_replay", "hard", 0, "g1"),
        IdentityRecord("c2", 0, "normal_replay", "hard", 0, "g1"),
    )
    t_pool = make_pool(
        pool_id="T",
        pool_role="T_STRESS",
        records=records[:2],
        base_denominator=80,
        base_manifest_sha256="A" * 64,
        source_manifest_path="T.csv",
        source_manifest_sha256="B" * 64,
        construction_seed=None,
        selection_semantic="HISTORICAL_SIGN_REVERSAL_STRESS_SET_NOT_VALIDATED_SELECTOR",
    )

    result = build_r2_matched_random(
        [_mapping(record) for record in records],
        t_pool=t_pool,
        base_denominator=80,
        base_manifest_sha256="A" * 64,
        source_manifest_sha256="C" * 64,
        selection_seed=17,
        guard=TerminalFieldGuard(),
        matching_policy=R2_MINIMUM_OOF_GROUP_DISPLACEMENT_POLICY,
    )

    assert len(result.pool.records) == 2
    assert not ({record.sample_id for record in result.pool.records} & {"t0", "t1"})
    assert Counter(record.stratum()[:3] for record in result.pool.records) == Counter(
        record.stratum()[:3] for record in t_pool.records
    )
    assert Counter(record.stratum() for record in result.pool.records) != Counter(
        record.stratum() for record in t_pool.records
    )
    assert result.audit.exact_coarse_stratum_quota is True
    assert result.audit.exact_joint_stratum_quota is False
    assert result.audit.relaxed_fields == ("oof_group_id",)
    assert result.audit.displacement_count == 1
    assert result.audit.displacement_lower_bound == 1
    assert len(result.audit.displacement_records) == 1
    displacement = result.audit.displacement_records[0]
    assert displacement.requested_oof_group_id == "g0"
    assert displacement.selected_oof_group_id == "g1"
    assert displacement.selected_sample_id in {"c1", "c2"}
    assert result.audit.displacement_ledger_digest == stable_digest(result.audit.displacement_records)


def test_addendum_does_not_turn_unknown_policy_into_another_relaxation():
    records = (
        IdentityRecord("t0", 0, "normal_replay", "hard", 0, "g0"),
        IdentityRecord("c0", 0, "normal_replay", "hard", 0, "g1"),
    )
    t_pool = make_pool(
        pool_id="T",
        pool_role="T_STRESS",
        records=records[:1],
        base_denominator=40,
        base_manifest_sha256="A" * 64,
        source_manifest_path="T.csv",
        source_manifest_sha256="B" * 64,
        construction_seed=None,
        selection_semantic="HISTORICAL_SIGN_REVERSAL_STRESS_SET_NOT_VALIDATED_SELECTOR",
    )
    with pytest.raises(SctsrError) as error:
        build_r2_matched_random(
            [_mapping(record) for record in records],
            t_pool=t_pool,
            base_denominator=40,
            base_manifest_sha256="A" * 64,
            source_manifest_sha256="C" * 64,
            selection_seed=17,
            matching_policy="RELAX_EVERYTHING",
        )
    assert error.value.code is ErrorCode.CONFIGURATION_MISMATCH


def test_registered_addendum_materializes_the_audited_3000_id_pool(repository_root):
    registry = load_asset_registry(repository_root / "configs/stage1_sctsr_v4/asset_registry_v1.json")
    inputs = load_formal_pool_inputs(registry, repository_root)

    result = build_registered_r2(
        inputs,
        base_denominator=registry.base_denominator,
        selection_seed=R2_APPROVED_SELECTION_SEED,
    )

    assert len(result.pool.records) == 3000
    assert result.pool.spec.identity_digest == R2_APPROVED_IDENTITY_DIGEST
    assert result.audit.displacement_count == 379
    assert result.audit.displacement_lower_bound == 379
    assert len(result.audit.displacement_records) == 379
    assert result.audit.relaxed_fields == ("oof_group_id",)
    assert not ({record.sample_id for record in result.pool.records} & {record.sample_id for record in inputs.t_pool.records})
    assert Counter(record.stratum()[:3] for record in result.pool.records) == Counter(
        record.stratum()[:3] for record in inputs.t_pool.records
    )
    content = load_registered_dataset_content_map(registry=registry, repository_root=repository_root)
    t_content = {str(content[record.sample_id]["image_sha256"]).upper() for record in inputs.t_pool.records}
    r2_content = {str(content[record.sample_id]["image_sha256"]).upper() for record in result.pool.records}
    assert len(r2_content) == 3000
    assert not (t_content & r2_content)
    assert result.audit.overlap_with_t_content_count == 0
    assert result.audit.excluded_content_duplicate_count >= 1
    validate_approved_r2_build(asdict(result.pool.spec), asdict(result.audit))


def test_registered_addendum_build_audit_rejects_unregistered_extra_relaxation(repository_root):
    registry = load_asset_registry(repository_root / "configs/stage1_sctsr_v4/asset_registry_v1.json")
    inputs = load_formal_pool_inputs(registry, repository_root)
    result = build_registered_r2(
        inputs,
        base_denominator=registry.base_denominator,
        selection_seed=R2_APPROVED_SELECTION_SEED,
    )
    audit = asdict(result.audit)
    audit["relaxed_fields"] = ["oof_group_id", "oof_fold"]
    with pytest.raises(SctsrError) as error:
        validate_approved_r2_build(asdict(result.pool.spec), audit)
    assert error.value.code is ErrorCode.CONFIGURATION_MISMATCH


def test_registered_addendum_rejects_an_unfrozen_selection_seed(repository_root):
    registry = load_asset_registry(repository_root / "configs/stage1_sctsr_v4/asset_registry_v1.json")
    inputs = load_formal_pool_inputs(registry, repository_root)
    with pytest.raises(SctsrError) as error:
        build_registered_r2(inputs, base_denominator=registry.base_denominator, selection_seed=1)
    assert error.value.code is ErrorCode.CONFIGURATION_MISMATCH


def test_machine_addendum_names_one_relaxed_field_and_one_shared_r2_pool(repository_root):
    policy_path = repository_root / "configs/stage1_sctsr_v4/r2_matching_policy_v1.json"
    policy = load_json(policy_path)

    assert policy["schema_version"] == "stage1.sctsr.r2_matching_policy.v2"
    assert policy["owner_decision"] == "APPROVED_CONTENT_DISJOINT_REPAIR_2026-08-16"
    assert policy["relaxed_fields"] == ["oof_group_id"]
    assert policy["exact_fields"] == ["y_true", "historical_dynamic_bucket", "oof_fold"]
    assert policy["applies_to_arms"] == ["R2_U", "R2_F", "T_TO_R2_AT_160"]
    assert policy["shared_identity_pool_required"] is True
    assert policy["unique_count"] == 3000
    assert policy["overlap_with_t"] == 0
    assert policy["overlap_with_t_content"] == 0
    assert policy["content_unique_within_r2"] is True
    assert policy["selection_seed"] == R2_APPROVED_SELECTION_SEED
    assert policy["expected_identity_digest"] == R2_APPROVED_IDENTITY_DIGEST
    validate_r2_matching_policy_mapping(policy)

    contract = load_json(repository_root / "configs/stage1_sctsr_v4/contract_v1.json")
    assert contract["r2_matching_policy"] == {
        "path": "configs/stage1_sctsr_v4/r2_matching_policy_v1.json",
        "sha256": sha256_file(policy_path),
        "policy_digest": policy["policy_digest"],
        "policy_id": policy["policy_id"],
        "selection_seed": R2_APPROVED_SELECTION_SEED,
        "expected_identity_digest": R2_APPROVED_IDENTITY_DIGEST,
        "expected_content_map_digest": R2_APPROVED_CONTENT_MAP_DIGEST,
        "expected_selected_content_digest": policy["expected_selected_content_digest"],
    }

    seed_registry = load_json(repository_root / "configs/stage1_sctsr_v4/seed_registry_schema_v1.json")
    assert seed_registry["selection_seeds"] == [R2_APPROVED_SELECTION_SEED]


def test_addendum_validator_rejects_relaxing_a_second_field(repository_root):
    policy = load_json(repository_root / "configs/stage1_sctsr_v4/r2_matching_policy_v1.json")
    policy["relaxed_fields"] = ["oof_group_id", "oof_fold"]
    policy["policy_digest"] = stable_digest({key: value for key, value in policy.items() if key != "policy_digest"})
    with pytest.raises(SctsrError) as error:
        validate_r2_matching_policy_mapping(policy)
    assert error.value.code is ErrorCode.CONFIGURATION_MISMATCH
