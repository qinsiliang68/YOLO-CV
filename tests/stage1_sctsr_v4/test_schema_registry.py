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
