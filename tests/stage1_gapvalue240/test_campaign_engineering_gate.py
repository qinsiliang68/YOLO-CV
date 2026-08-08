from __future__ import annotations

import json
from pathlib import Path

import pytest

from stage1_gapvalue240.campaign_engineering_gate import (
    ENGINEERING_GATE_SCHEMA_V2,
    REQUIRED_EVIDENCE_SCHEMAS,
    ValidationIdentity,
    bind_validation_evidence,
    build_engineering_gate_v2,
    validate_engineering_gate_v2,
)
from stage1_gapvalue240.errors import ValidationError
from stage1_gapvalue240.util import atomic_write_json


IDENTITY = ValidationIdentity(
    source_tree_sha256="A" * 64,
    queue_registry_sha256="B" * 64,
    canonical_lock_file_sha256="C" * 64,
)


def _evidence(tmp_path: Path) -> dict[str, Path]:
    raw_root = tmp_path / "raw"
    envelope_root = tmp_path / "envelopes"
    raw_root.mkdir()
    envelope_root.mkdir()
    result = {}
    for key, schema in REQUIRED_EVIDENCE_SCHEMAS.items():
        raw = raw_root / f"{key}.json"
        atomic_write_json(raw, {"schema_version": schema, "status": "PASS", "check": key})
        envelope = envelope_root / f"{key}.json"
        bind_validation_evidence(
            raw,
            evidence_type=key,
            expected_schema=schema,
            identity=IDENTITY,
            output_path=envelope,
            allowed_root=tmp_path,
        )
        result[key] = envelope
    return result


def test_engineering_gate_v2_revalidates_every_bound_report(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    gate = build_engineering_gate_v2(
        evidence,
        expected_identity=IDENTITY,
        allowed_root=tmp_path,
        output_path=tmp_path / "ENGINEERING_GATE_V2.json",
    )
    assert gate["schema_version"] == ENGINEERING_GATE_SCHEMA_V2
    assert gate["status"] == "PASS"
    assert set(gate["evidence"]) == set(REQUIRED_EVIDENCE_SCHEMAS)
    loaded = validate_engineering_gate_v2(
        tmp_path / "ENGINEERING_GATE_V2.json",
        expected_identity=IDENTITY,
        allowed_root=tmp_path,
    )
    assert loaded["status"] == "PASS"


def test_engineering_gate_cannot_be_forged_from_top_level_status(tmp_path: Path) -> None:
    forged = tmp_path / "forged.json"
    atomic_write_json(
        forged,
        {
            "schema_version": ENGINEERING_GATE_SCHEMA_V2,
            "status": "PASS",
            "identity": IDENTITY.as_dict(),
            "evidence": {},
        },
    )
    with pytest.raises(ValidationError, match="engineering gate"):
        validate_engineering_gate_v2(forged, expected_identity=IDENTITY, allowed_root=tmp_path)


def test_engineering_gate_rejects_tampered_underlying_report(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    build_engineering_gate_v2(
        evidence,
        expected_identity=IDENTITY,
        allowed_root=tmp_path,
        output_path=tmp_path / "gate.json",
    )
    envelope = json.loads(evidence["lease_concurrency_validation"].read_text(encoding="utf-8"))
    raw = tmp_path / envelope["payload_relpath"]
    raw.write_text(json.dumps({"schema_version": envelope["payload_schema"], "status": "FAIL"}), encoding="utf-8")
    with pytest.raises(ValidationError, match="engineering gate"):
        validate_engineering_gate_v2(
            tmp_path / "gate.json",
            expected_identity=IDENTITY,
            allowed_root=tmp_path,
        )


def test_engineering_gate_rejects_identity_mismatch(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    wrong = ValidationIdentity(
        source_tree_sha256="D" * 64,
        queue_registry_sha256=IDENTITY.queue_registry_sha256,
        canonical_lock_file_sha256=IDENTITY.canonical_lock_file_sha256,
    )
    with pytest.raises(ValidationError, match="engineering gate"):
        build_engineering_gate_v2(
            evidence,
            expected_identity=wrong,
            allowed_root=tmp_path,
            output_path=tmp_path / "gate.json",
        )
