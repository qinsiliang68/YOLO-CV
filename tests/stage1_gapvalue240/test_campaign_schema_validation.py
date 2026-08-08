from __future__ import annotations
import json
from pathlib import Path
import tomllib
import pytest
from stage1_gapvalue240.campaign_schema_validation import validate_report_schema
from stage1_gapvalue240.errors import ValidationError

ROOT=Path(__file__).resolve().parents[2]
SCHEMAS=ROOT/'schemas/stage1_gapvalue240'

def test_all_new_schemas_are_valid_and_reject_missing_required(tmp_path: Path) -> None:
    files=sorted(SCHEMAS.glob('*campaign*.schema.json'))+sorted(SCHEMAS.glob('*validation*.schema.json'))+sorted(SCHEMAS.glob('*canary*.schema.json'))+sorted(SCHEMAS.glob('source_tree_manifest.schema.json'))
    unique=[]
    for path in files:
        if path not in unique: unique.append(path)
    assert len(unique) >= 19
    for index,schema_path in enumerate(unique):
        schema=json.loads(schema_path.read_text(encoding='utf-8'))
        assert schema['$schema'].endswith('2020-12/schema')
        assert 'schema_version' in schema['required']
        invalid=tmp_path/f'invalid_{index}.json'
        invalid.write_text(json.dumps({'schema_version':schema['properties']['schema_version']['const']}),encoding='utf-8')
        with pytest.raises(ValidationError):
            validate_report_schema(invalid,schema_path)

def test_schema_validator_accepts_minimal_documentation_report(tmp_path: Path) -> None:
    report=tmp_path/'report.json'
    report.write_text(json.dumps({'schema_version':'stage1.documentation_handoff_validation.v1','status':'PASS','issues':[],'documents':[],'utf8_valid':True}),encoding='utf-8')
    loaded=validate_report_schema(report,SCHEMAS/'documentation_handoff_validation.schema.json')
    assert loaded['status']=='PASS'


def test_jsonschema_is_a_declared_runtime_dependency() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = [str(value).lower() for value in project["project"]["dependencies"]]

    assert any(value.startswith("jsonschema") for value in dependencies)
