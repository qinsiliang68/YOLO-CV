"""Dry-only v2 campaign chain: preregistration -> queue -> gate -> release -> assignment."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from typing import Mapping

from .campaign_assignment import build_campaign_assignment, load_campaign_assignment
from .campaign_contract_validation import build_source_tree_manifest
from .campaign_controller import build_campaign_release_manifests, load_campaign_release
from .campaign_engineering_gate import (
    REQUIRED_EVIDENCE_SCHEMAS,
    ValidationIdentity,
    bind_validation_evidence,
    build_engineering_gate_v2,
)
from .campaign_run_queue import build_campaign_run_queue
from .errors import ValidationError
from .util import atomic_write_json, sha256_file


DRY_CHAIN_SCHEMA = "stage1.dynamic_campaign_dry_generation.v2"


def dry_generate_campaign_v2(
    preregistration_dir: str | Path,
    monitor_source: str | Path,
    *,
    output_root: str | Path,
    repo_root: str | Path,
    machine_configs_dir: str | Path,
    slot_mapping: Mapping[str, str],
    raw_evidence_reports: Mapping[str, str | Path],
    campaign_id: str,
    pilot_seed_ids: tuple[str, ...],
    assignment_id: str = "ASSIGNMENT_V2_DRY",
) -> dict:
    """Build and validate the entire immutable chain without activating an assignment."""

    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"dry-generation output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    queue_dir = output / "04_run_queue_v2"
    queue_result = build_campaign_run_queue(
        preregistration_dir,
        queue_dir,
        monitor_source=monitor_source,
    )
    queue_validation = json.loads(queue_result.validation_path.read_text(encoding="utf-8"))
    if queue_validation.get("schema_version") != "stage1.dynamic_campaign_run_queue.v2":
        raise ValidationError("dry generation produced a non-v2 queue")
    evidence_root = output / "05_engineering_gate_v2"
    raw_root = evidence_root / "raw"
    envelope_root = evidence_root / "envelopes"
    raw_root.mkdir(parents=True)
    envelope_root.mkdir(parents=True)
    source_manifest = build_source_tree_manifest(repo_root, evidence_root / "SOURCE_TREE_MANIFEST.json")
    source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    identity = ValidationIdentity(
        source_tree_sha256=source_payload["root_digest"],
        queue_registry_sha256=queue_validation["job_registry_sha256"],
        canonical_lock_file_sha256=queue_validation["canonical_lock_file_sha256"],
    )
    if set(raw_evidence_reports) != set(REQUIRED_EVIDENCE_SCHEMAS):
        raise ValidationError("dry-generation evidence registry is incomplete")
    envelopes = {}
    for evidence_type, expected_schema in REQUIRED_EVIDENCE_SCHEMAS.items():
        source = Path(raw_evidence_reports[evidence_type]).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = raw_root / f"{evidence_type}.json"
        shutil.copy2(source, destination)
        envelope = envelope_root / f"{evidence_type}.json"
        bind_validation_evidence(
            destination,
            evidence_type=evidence_type,
            expected_schema=expected_schema,
            identity=identity,
            output_path=envelope,
            allowed_root=evidence_root,
        )
        envelopes[evidence_type] = envelope
    gate_path = evidence_root / "ENGINEERING_GATE_V2.json"
    build_engineering_gate_v2(
        envelopes,
        expected_identity=identity,
        allowed_root=evidence_root,
        output_path=gate_path,
    )
    releases = build_campaign_release_manifests(
        queue_dir,
        output / "06_releases_v2",
        campaign_id=campaign_id,
        pilot_seed_ids=pilot_seed_ids,
        engineering_gate_report=gate_path,
    )
    release = load_campaign_release(queue_dir, releases.pilot_release, expected_campaign_id=campaign_id)
    assignment_files = build_campaign_assignment(
        queue_dir,
        releases.pilot_release,
        output / "07_assignment_v2_dry",
        campaign_id=campaign_id,
        assignment_id=assignment_id,
        machine_configs_dir=machine_configs_dir,
        slot_mapping=slot_mapping,
        repo_root=repo_root,
    )
    assignment = load_campaign_assignment(
        queue_dir,
        releases.pilot_release,
        assignment_files.manifest_path,
        expected_campaign_id=campaign_id,
        repo_root=repo_root,
    )
    report = {
        "schema_version": DRY_CHAIN_SCHEMA,
        "status": "PASS",
        "created_at_unix": time.time(),
        "activation_status": "NOT_ACTIVATED_DRY_RUN",
        "queue": {
            "path": str(queue_dir),
            "validation_sha256": sha256_file(queue_result.validation_path),
            "registry_sha256": queue_validation["job_registry_sha256"],
            "schema_version": queue_validation["schema_version"],
        },
        "engineering_gate": {
            "path": str(gate_path),
            "sha256": sha256_file(gate_path),
        },
        "release": {
            "release_id": release.release_id,
            "path": str(release.path),
            "sha256": release.sha256,
            "job_count": len(release.job_ids),
        },
        "assignment": {
            "assignment_id": assignment.assignment_id,
            "path": str(assignment.manifest_path),
            "sha256": assignment.sha256,
            "job_count": len(assignment.rows),
            "standalone_commands_sha256": sha256_file(assignment_files.standalone_commands_path),
        },
        "identity": identity.as_dict(),
    }
    atomic_write_json(output / "DRY_GENERATION_VALIDATION.json", report, overwrite=True)
    return report
