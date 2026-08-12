from __future__ import annotations
from pathlib import Path
import pytest
from stage1_gapvalue240.campaign_documentation_validation import validate_documentation_handoff
from stage1_gapvalue240.errors import ValidationError

ROOT=Path(__file__).resolve().parents[2]

def test_repository_campaign_documents_are_utf8_and_complete(tmp_path: Path) -> None:
    report=validate_documentation_handoff([
        ROOT/'docs/stage1_gapvalue240/DYNAMIC_REPLAY_CAMPAIGN_OPERATIONS_V2.md',
        ROOT/'docs/stage1_gapvalue240/DYNAMIC_REPLAY_CAMPAIGN_OPERATOR_BRIEFING_V2.md',
        ROOT/'docs/stage1_gapvalue240/RUN_QUEUE_V2_README.md',
        ROOT/'configs/stage1_gapvalue240/README_MACHINE_CONFIG.md',
        ROOT/'docs/stage1_gapvalue240/DYNAMIC_REPLAY_CAMPAIGN_FIELD_AUDIT.md',
    ],output_path=tmp_path/'DOCUMENTATION_HANDOFF_VALIDATION.json')
    assert report['status']=='PASS'
    assert report['utf8_valid']

def test_documentation_validator_rejects_invalid_utf8(tmp_path: Path) -> None:
    bad=tmp_path/'bad.md'; bad.write_bytes(b'\xff\xfe')
    with pytest.raises(ValidationError):
        validate_documentation_handoff([bad],output_path=tmp_path/'out.json')
