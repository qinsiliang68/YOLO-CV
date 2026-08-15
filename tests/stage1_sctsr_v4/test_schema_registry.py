import pytest
from stage1_sctsr_v4.errors import SctsrError
from stage1_sctsr_v4.schema_registry import REQUIRED_SCHEMAS, SchemaRegistry
from stage1_sctsr_v4.serialization import load_json

def test_schema_registry_passes(repository_root):
    r=SchemaRegistry.from_mapping(load_json(repository_root/'configs/stage1_sctsr_v4/schema_registry_v1.json'));r.validate();assert len(r.digest)==64
    assert dict(r.schemas) == REQUIRED_SCHEMAS

def test_schema_registry_missing_frozen_entry_rejected(repository_root):
    raw=load_json(repository_root/'configs/stage1_sctsr_v4/schema_registry_v1.json');del raw['schemas']['contract']
    with pytest.raises(SctsrError):SchemaRegistry.from_mapping(raw).validate()


def test_schema_registry_rejects_stale_or_unregistered_public_schema(repository_root):
    raw=load_json(repository_root/'configs/stage1_sctsr_v4/schema_registry_v1.json')
    raw['schemas']['prediction_artifact'] = 'stage1.sctsr.prediction.v1'
    raw['schemas']['unregistered_extra'] = 'stage1.sctsr.unregistered.v1'
    with pytest.raises(SctsrError):
        SchemaRegistry.from_mapping(raw).validate()


def test_schema_registry_registers_formal_input_and_detailed_self_audit_schemas():
    assert REQUIRED_SCHEMAS["external_file_binding"] == "stage1.sctsr.external_file_binding.v1"
    assert REQUIRED_SCHEMAS["formal_input_snapshot"] == "stage1.sctsr.formal_input_snapshot.v1"
    assert REQUIRED_SCHEMAS["implementation_self_audit"] == "stage1.sctsr.implementation_self_audit.v1"
    assert REQUIRED_SCHEMAS["self_audit_input_plan"] == "stage1.sctsr.self_audit_input_plan.v1"
    assert REQUIRED_SCHEMAS["manual_line_review"] == "stage1.sctsr.manual_line_review.v1"
    assert REQUIRED_SCHEMAS["repository_state_audit"] == "stage1.sctsr.repository_state_audit.v1"
    assert REQUIRED_SCHEMAS["changed_file_ledger"] == "stage1.sctsr.changed_file_ledger.v1"
    assert REQUIRED_SCHEMAS["legacy_evidence_audit"] == "stage1.sctsr.legacy_evidence_audit.v1"


def test_schema_registry_registers_one_attempt_formal_execution_schemas():
    assert REQUIRED_SCHEMAS["formal_execution_token"] == "stage1.sctsr.formal_execution_token.v1"
    assert REQUIRED_SCHEMAS["formal_execution_claim"] == "stage1.sctsr.formal_execution_claim.v2"
    assert REQUIRED_SCHEMAS["execution_claim_registry"] == "stage1.sctsr.execution_claim_registry.v1"
    assert REQUIRED_SCHEMAS["execution_attempt_snapshot"] == "stage1.sctsr.execution_attempt_snapshot.v2"


def test_schema_registry_registers_simple_seeded_deployment_plan():
    assert REQUIRED_SCHEMAS["deployment_plan"] == "stage1.sctsr.deployment_plan.v1"


def test_schema_registry_registers_r2_specification_audit():
    assert REQUIRED_SCHEMAS["r2_specification_audit"] == "stage1.sctsr.r2_specification_audit.v1"
