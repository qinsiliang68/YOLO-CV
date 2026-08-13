import pytest
from stage1_sctsr_v4.errors import ErrorCode,SctsrError
from stage1_sctsr_v4.seed_registry import SeedRegistry
from stage1_sctsr_v4.serialization import load_json

def test_blocked_registry_is_valid_development_state(repository_root):
    r=SeedRegistry.from_mapping(load_json(repository_root/'configs/stage1_sctsr_v4/seed_registry_schema_v1.json'));r.validate();assert r.state=='FORMAL_SEEDS_BLOCKED_UNTIL_RELEASE'

def test_overlapping_seed_rejected():
    r=SeedRegistry('stage1.sctsr.seed_registry.v1','SYNTHETIC_FIXTURE_ONLY',(1,),(2,),(3,),(1,))
    with pytest.raises(SctsrError) as e:r.validate()
    assert e.value.code is ErrorCode.SEED_REGISTRY_INVALID

def test_formal_requires_8_and_14():
    r=SeedRegistry('stage1.sctsr.seed_registry.v1','FROZEN_BY_RELEASE_AUTHORITY',(),(),tuple(range(8)),tuple(range(20,34)))
    r.validate(formal=True,release_authorization_verified=True)
