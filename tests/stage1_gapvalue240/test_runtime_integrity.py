from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from stage1_gapvalue240.errors import ContractError, ValidationError
from stage1_gapvalue240.machine import MachineConfig
from stage1_gapvalue240.machine_assets import (
    build_machine_asset_report,
    validate_machine_asset_report,
)
from stage1_gapvalue240.runtime_contract import (
    compute_runtime_contract_hash,
    is_aggregatable_status,
    load_runtime_contract,
    validate_runtime_links,
    validation_status_for_mode,
    verify_release_identity,
    verify_all_selections_against_index,
    verify_selection_against_index,
)
from stage1_gapvalue240.util import sha256_file


ROOT = Path(__file__).resolve().parents[2]
SCIENCE = ROOT / "configs/stage1_gapvalue240/EXPERIMENT_CONTRACT.yaml"


def _write_runtime_contract(
    repo: Path,
    *,
    matrix: Path,
    selection_index: Path,
    artifact_root: Path,
    tag: str = "stage1-gapvalue240-runtime-v1.2.0",
    role_rows: int = 1,
) -> Path:
    science = repo / "configs/stage1_gapvalue240/EXPERIMENT_CONTRACT.yaml"
    science.parent.mkdir(parents=True, exist_ok=True)
    science.write_bytes(SCIENCE.read_bytes())
    checkpoint = repo / "base.pt"
    checkpoint.write_bytes(b"checkpoint")
    binding = artifact_root / "generated/site_asset_binding.json"
    binding.parent.mkdir(parents=True, exist_ok=True)
    binding.write_text(
        json.dumps({"checkpoint_filename": checkpoint.name, "full_sha256": sha256_file(checkpoint)}),
        encoding="utf-8",
    )
    roles = {}
    role_specs = (
        ("train_defect", "train_manifest", "train", 1),
        ("train_normal", "normal_train_manifest", "normal_train", 0),
        ("val_model_defect", "val_model_defect_manifest", "val_model", 1),
        ("val_model_normal", "val_model_normal_manifest", "normal_val_model", 0),
        ("val_cal_defect", "val_cal_defect_manifest", "val_cal", 1),
        ("val_cal_normal", "val_cal_normal_manifest", "normal_val_cal", 0),
        ("val_op_defect", "val_op_defect_manifest", "val_op", 1),
        ("val_op_normal", "val_op_normal_manifest", "normal_val_op", 0),
    )
    for role, key, split, label in role_specs:
        roles[role] = {
            "machine_config_key": key,
            "expected_rows": role_rows,
            "expected_split": split,
            "expected_label": label,
        }
    data = {
        "runtime_contract_id": "runtime-test",
        "runtime_contract_version": "1.2.0",
        "runtime_contract_sha256": "",
        "release": {"git_tag": tag, "require_tag_at_head": True},
        "science_contract": {
            "path": "configs/stage1_gapvalue240/EXPERIMENT_CONTRACT.yaml",
            "file_sha256": sha256_file(science),
            "semantic_sha256": "B5BF55446E0B6E4BE0911295C8E666879C6031D0F7BD6B48E6791C74A0CBC4E9",
        },
        "queue": {
            "artifact_root": artifact_root.relative_to(repo).as_posix(),
            "frozen_matrix": {
                "path": matrix.relative_to(repo).as_posix(),
                "sha256": sha256_file(matrix),
                "rows": 1,
            },
            "selection_index": {
                "path": selection_index.relative_to(repo).as_posix(),
                "sha256": sha256_file(selection_index),
                "rows": 1,
            },
            "selection_count": 1,
        },
        "checkpoint": {
            "site_binding": {
                "path": "artifacts/gapvalue240_v1_1/generated/site_asset_binding.json",
                "sha256": sha256_file(binding),
            },
            "filename": checkpoint.name,
            "sha256": sha256_file(checkpoint),
        },
        "machine_assets": {
            "identity_column": "canonical_image_relpath",
            "split_column": "split",
            "label_column": "Defect",
            "required_columns": ["canonical_image_relpath", "split", "Defect", "Filename"],
            "expected_total_rows": role_rows * 8,
            "manifest_roles": roles,
        },
        "execution_identity": {
            "formal_status": "VALIDATED",
            "dry_run_status": "DRY_RUN_VALIDATED",
            "aggregatable_statuses": ["VALIDATED"],
        },
    }
    data["runtime_contract_sha256"] = compute_runtime_contract_hash(data)
    path = repo / "configs/stage1_gapvalue240/RUNTIME_CONTRACT_v1_2.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _make_linked_repo(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    artifact = repo / "artifacts/gapvalue240_v1_1"
    selection = artifact / "generated/selections/RUN_001/selection_manifest.csv"
    selection.parent.mkdir(parents=True)
    pd.DataFrame([{"run_slot": "RUN_001", "sample_id": "n0", "y_true": 0}]).to_csv(selection, index=False)
    matrix = artifact / "generated/frozen_experiment_matrix.csv"
    matrix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"run_slot": "RUN_001", "budget": 1}]).to_csv(matrix, index=False)
    index = artifact / "generated/selection_index.csv"
    pd.DataFrame(
        [{
            "run_slot": "RUN_001",
            "selection_manifest": "generated\\selections\\RUN_001\\selection_manifest.csv",
            "sha256": sha256_file(selection),
        }]
    ).to_csv(index, index=False)
    contract_path = _write_runtime_contract(
        repo, matrix=matrix, selection_index=index, artifact_root=artifact
    )
    return repo, artifact, selection, index, contract_path


def test_runtime_contract_binds_science_matrix_index_and_selection(tmp_path):
    repo, _, selection, index, path = _make_linked_repo(tmp_path)
    contract = load_runtime_contract(path)
    report = validate_runtime_links(contract, repo)
    assert report["status"] == "PASS"
    assert report["queue"]["selection_index"]["rows"] == 1
    assert verify_selection_against_index(contract, repo, "RUN_001", selection)["status"] == "PASS"

    selection.write_text(selection.read_text(encoding="utf-8") + "RUN_001,n1,0\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="Selection SHA-256 mismatch"):
        verify_selection_against_index(contract, repo, "RUN_001", selection)

    index.write_text(index.read_text(encoding="utf-8") + "# mutation\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="selection_index SHA-256 mismatch"):
        validate_runtime_links(contract, repo)


def test_runtime_contract_hash_mismatch_is_rejected(tmp_path):
    repo, _, _, _, path = _make_linked_repo(tmp_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["queue"]["selection_count"] = 2
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ContractError, match="Runtime contract hash mismatch"):
        load_runtime_contract(path)


def test_release_tag_must_resolve_to_head_and_override_is_explicit(tmp_path):
    repo, _, _, _, path = _make_linked_repo(tmp_path)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "one"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "tag", "stage1-gapvalue240-runtime-v1.2.0"], cwd=repo, check=True)
    contract = load_runtime_contract(path)
    assert verify_release_identity(contract, repo)["status"] == "PASS"

    (repo / "next.txt").write_text("next", encoding="utf-8")
    subprocess.run(["git", "add", "next.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "two"], cwd=repo, check=True, capture_output=True)
    with pytest.raises(ValidationError, match="does not point to HEAD"):
        verify_release_identity(contract, repo)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    with pytest.raises(ValidationError, match="test override is disabled"):
        verify_release_identity(contract, repo, test_release_ref_override=head)
    assert verify_release_identity(
        contract,
        repo,
        test_release_ref_override=head,
        allow_test_override=True,
    )["override_used"] is True


def _write_manifest(path: Path, sample_id: str, split: str, label: int, image: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(f"image-{sample_id}".encode())
    pd.DataFrame(
        [{
            "canonical_image_relpath": sample_id,
            "split": split,
            "Defect": label,
            "Filename": Path(sample_id).name,
        }]
    ).to_csv(path, index=False)


def test_machine_asset_report_covers_eight_manifests_and_is_reusable_without_rescan(tmp_path):
    repo, _, _, _, path = _make_linked_repo(tmp_path)
    contract = load_runtime_contract(path)
    dataset = tmp_path / "dataset"
    manifest_root = tmp_path / "manifests"
    role_specs = contract.data["machine_assets"]["manifest_roles"]
    machine_data = {
        "machine_id": "machine_test",
        "repo_root": str(repo),
        "dataset_root": str(dataset),
        "artifact_root": str(tmp_path),
        "output_root": str(tmp_path / "out"),
        "cache_root": str(tmp_path / "cache"),
        "gpu_id": 0,
        "num_workers": 0,
    }
    machine_data["base_checkpoint"] = str(repo / "base.pt")
    for index, (role, spec) in enumerate(role_specs.items()):
        manifest = manifest_root / f"{role}.csv"
        rel_image = Path("images") / f"{role}.jpg"
        _write_manifest(manifest, rel_image.as_posix(), spec["expected_split"], spec["expected_label"], dataset / rel_image)
        machine_data[spec["machine_config_key"]] = str(manifest)
    machine = MachineConfig(path=tmp_path / "machine.yaml", data=machine_data)
    output = tmp_path / "machine_asset_report.json"

    report = build_machine_asset_report(contract, machine, output, image_verification="sha256")
    assert report["status"] == "PASS"
    assert report["total_manifest_rows"] == 8
    assert report["images"]["checked_count"] == 8
    assert report["images"]["verification_mode"] == "sha256"
    assert len(report["manifests"]) == 8
    assert all(row["columns"] for row in report["manifests"].values())
    assert all(len(row["id_digest_sha256"]) == 64 for row in report["manifests"].values())

    # Cached validation only verifies the immutable report identity and does not touch images.
    next(iter(dataset.rglob("*.jpg"))).unlink()
    cached = validate_machine_asset_report(contract, output, expected_machine_id="machine_test")
    assert cached["status"] == "PASS"

    tampered = json.loads(output.read_text(encoding="utf-8"))
    tampered["manifests"]["train_defect"]["rows"] = 99
    output.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValidationError, match="row count mismatch"):
        validate_machine_asset_report(contract, output, expected_machine_id="machine_test")


def test_machine_asset_report_rejects_cross_split_identity_overlap(tmp_path):
    repo, _, _, _, path = _make_linked_repo(tmp_path)
    contract = load_runtime_contract(path)
    dataset = tmp_path / "dataset"
    manifest_root = tmp_path / "manifests"
    machine_data = {
        "machine_id": "machine_test",
        "repo_root": str(repo),
        "dataset_root": str(dataset),
        "artifact_root": str(tmp_path),
        "output_root": str(tmp_path / "out"),
        "cache_root": str(tmp_path / "cache"),
        "gpu_id": 0,
        "num_workers": 0,
    }
    machine_data["base_checkpoint"] = str(repo / "base.pt")
    for index, (role, spec) in enumerate(contract.data["machine_assets"]["manifest_roles"].items()):
        manifest = manifest_root / f"{role}.csv"
        rel_image = Path("images/shared.jpg" if index < 2 else f"images/{role}.jpg")
        _write_manifest(manifest, rel_image.as_posix(), spec["expected_split"], spec["expected_label"], dataset / rel_image)
        machine_data[spec["machine_config_key"]] = str(manifest)
    machine = MachineConfig(path=tmp_path / "machine.yaml", data=machine_data)
    with pytest.raises(ValidationError, match="cross-manifest identity overlap"):
        build_machine_asset_report(contract, machine, tmp_path / "failed.json", image_verification="existence")


def test_dry_run_has_distinct_nonaggregatable_identity():
    assert validation_status_for_mode(True) == "DRY_RUN_VALIDATED"
    assert validation_status_for_mode(False) == "VALIDATED"
    assert is_aggregatable_status("VALIDATED") is True
    assert is_aggregatable_status("DRY_RUN_VALIDATED") is False


def test_repository_runtime_contract_binds_all_240_frozen_selections():
    contract = load_runtime_contract(
        ROOT / "configs/stage1_gapvalue240/RUNTIME_CONTRACT_v1_2.yaml"
    )
    links = validate_runtime_links(contract, ROOT)
    selections = verify_all_selections_against_index(contract, ROOT)
    assert links["selection_count"] == 240
    assert links["queue"]["frozen_matrix"]["rows"] == 240
    assert selections["selection_count"] == 240
